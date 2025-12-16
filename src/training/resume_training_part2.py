import os
import sys
import logging
import torch
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
import gc
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from config.settings import Config
from src.environments.trading_env import create_env_from_dataframes
from src.models.analyst import load_analyst
from src.data.features import get_feature_columns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(project_root, 'models/agent/resume_training_part2.log'))
    ]
)
logger = logging.getLogger('resume_training_part2')

class AgentTrainingLogger(BaseCallback):
    """
    Custom callback for detailed agent training logging.
    """
    def __init__(self, log_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_pnls = []
        self.action_counts = {0: 0, 1: 0, 2: 0}
        self.current_ep_reward = 0
        self.current_ep_length = 0
        
    def _on_step(self) -> bool:
        # Track rewards
        if len(self.locals.get('rewards', [])) > 0:
            reward = self.locals['rewards'][0]
            self.current_ep_reward += reward
            self.current_ep_length += 1

        # Track actions
        if len(self.locals.get('actions', [])) > 0:
            action = self.locals['actions'][0]
            if isinstance(action, np.ndarray):
                if action.size > 1:
                    # MultiDiscrete: [direction, size]
                    direction = int(action[0])
                    self.action_counts[direction] = self.action_counts.get(direction, 0) + 1
                else:
                    # Single action wrapped in array
                    self.action_counts[int(action.item())] = self.action_counts.get(int(action.item()), 0) + 1
            elif isinstance(action, (int, np.integer)):
                self.action_counts[int(action)] = self.action_counts.get(int(action), 0) + 1

        # Check for episode done
        dones = self.locals.get('dones', [False])
        if any(dones):
            self.episode_rewards.append(self.current_ep_reward)
            self.episode_lengths.append(self.current_ep_length)
            
            # Get info
            infos = self.locals.get('infos', [{}])
            if len(infos) > 0:
                self.episode_pnls.append(infos[0].get('total_pnl', 0.0))

            self.current_ep_reward = 0
            self.current_ep_length = 0

        # Periodic logging
        if self.n_calls % self.log_freq == 0:
            self._log_training_progress()
            
        return True

    def _log_training_progress(self):
        total_actions = sum(self.action_counts.values())
        if total_actions == 0: return

        action_pcts = {k: v/total_actions*100 for k, v in self.action_counts.items()}
        
        logger.info("-" * 50)
        logger.info(f"Training Progress @ {self.num_timesteps} steps:")
        logger.info(f"  Episodes completed: {len(self.episode_rewards)}")
        logger.info(f"  Action Distribution: Flat={action_pcts.get(0, 0):.1f}%, "
                   f"Long={action_pcts.get(1, 0):.1f}%, Short={action_pcts.get(2, 0):.1f}%")
        
        if self.episode_rewards:
            logger.info(f"  Avg Episode Reward: {np.mean(self.episode_rewards[-100:]):.2f}")
            logger.info(f"  Max Episode Reward: {np.max(self.episode_rewards):.2f}")
            
        if self.episode_pnls:
            logger.info(f"  Avg PnL: {np.mean(self.episode_pnls[-100:]):.2f} pips")
        
        logger.info("-" * 50)

class MemoryCleanupCallback(BaseCallback):
    """
    Callback to periodically clean up memory during training.
    Essential for Apple M2 with limited 8GB RAM.
    """
    def __init__(self, cleanup_freq: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.cleanup_freq = cleanup_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.cleanup_freq == 0:
            gc.collect()
            device_type = getattr(getattr(self.model, "device", None), "type", None)
            if device_type == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif device_type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if self.verbose > 0:
                print(f"Memory cleanup at step {self.n_calls}")
        return True

def resume_training_part2():
    config = Config()
    device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # 1. Load Pre-processed Data (Parquet)
    data_processed_path = config.paths.data_processed
    logger.info(f"Loading processed data from {data_processed_path}...")
    
    try:
        df_5m = pd.read_parquet(data_processed_path / 'features_5m_normalized.parquet')
        df_15m = pd.read_parquet(data_processed_path / 'features_15m_normalized.parquet')
        df_45m = pd.read_parquet(data_processed_path / 'features_45m_normalized.parquet')
        logger.info("Data loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load parquet files: {e}")
        return

    # 2. Define Feature Columns
    feature_cols = get_feature_columns()
    feature_cols = [c for c in feature_cols if c in df_5m.columns]
    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")

    # 3. Load Analyst Model
    logger.info("Loading analyst model...")
    feature_dims = {
        '5m': len(feature_cols),
        '15m': len(feature_cols),
        '45m': len(feature_cols)
    }

    analyst_path = config.paths.models_analyst / 'best.pt'
    if not analyst_path.exists():
        analyst_path = config.paths.models_analyst / 'best_model.pth'

    analyst = load_analyst(str(analyst_path), feature_dims, device=device, freeze=True)
    analyst.eval()

    # 4. Create Environment
    logger.info("Creating environment with reduced spread (1.0 pips)...")

    def make_env():
        env = create_env_from_dataframes(
            df_5m, df_15m, df_45m,
            analyst_model=analyst,
            feature_cols=feature_cols,
            config=config,
            device=device
        )
        env.spread_pips = 1.0
        return env

    # REDUCED TO 4 ENVS TO PREVENT OOM
    num_envs = 4
    logger.info(f"Using {num_envs} vectorized environments (DummyVecEnv) - Reduced for stability...")
    env = DummyVecEnv([make_env for _ in range(num_envs)])
    
    # Load VecNormalize statistics
    stats_path = os.path.join(project_root, 'models/agent/vec_normalize_phase2.pkl')
    if not os.path.exists(stats_path):
        stats_path = os.path.join(project_root, 'models/agent/vec_normalize.pkl')
        
    if os.path.exists(stats_path):
        logger.info(f"Loading VecNormalize statistics from {stats_path}...")
        env = VecNormalize.load(stats_path, env)
        env.training = True 
        env.norm_reward = True
    else:
        logger.info("Creating new VecNormalize wrapper...")
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 5. Load Crash Checkpoint
    checkpoint_path = os.path.join(project_root, 'models/agent/checkpoints_phase2/ppo_phase2_7401472_steps.zip')
    logger.info(f"Resuming from crash checkpoint: {checkpoint_path}...")
    
    model = PPO.load(checkpoint_path, env=env, device=device)
    
    # 6. Resume Training
    # We want to reach 10,000,000 total steps (approx).
    # We are at 7,401,472.
    # Remaining = 2,600,000 steps.
    steps_to_train = 2_600_000
    
    logger.info(f"Resuming training for remaining {steps_to_train} steps...")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=os.path.join(project_root, 'models/agent/checkpoints_phase2/'),
        name_prefix='ppo_phase2_part2'
    )
    
    training_logger = AgentTrainingLogger(log_freq=1000)
    memory_cleanup = MemoryCleanupCallback(cleanup_freq=5000, verbose=1)
    callback_list = CallbackList([checkpoint_callback, training_logger, memory_cleanup])

    try:
        model.learn(
            total_timesteps=steps_to_train,
            callback=callback_list,
            log_interval=1,
            reset_num_timesteps=False,
            progress_bar=True
        )
        
        save_path = os.path.join(project_root, 'models/agent/final_model_phase2_complete')
        model.save(save_path)
        env.save(os.path.join(project_root, 'models/agent/vec_normalize_phase2_complete.pkl'))
        logger.info(f"Phase 2 training complete. Model saved to {save_path}")

        # Plot Training Summary
        try:
            rewards = training_logger.episode_rewards
            pnls = training_logger.episode_pnls
            
            if len(rewards) > 0:
                plt.figure(figsize=(12, 8))
                
                plt.subplot(2, 1, 1)
                plt.plot(rewards, label='Episode Reward', alpha=0.6)
                plt.plot(pd.Series(rewards).rolling(window=50).mean(), label='50-Ep Moving Avg', color='red')
                plt.title('Phase 2 (Part 2) Training Rewards')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                if len(pnls) > 0:
                    plt.subplot(2, 1, 2)
                    plt.plot(pnls, label='Episode PnL (pips)', color='green', alpha=0.6)
                    plt.plot(pd.Series(pnls).rolling(window=50).mean(), label='50-Ep Moving Avg', color='darkgreen')
                    plt.title('Phase 2 (Part 2) Training PnL')
                    plt.xlabel('Episode')
                    plt.ylabel('PnL (pips)')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot_path = os.path.join(project_root, 'models/agent/agent_training_summary_phase2_part2.png')
                plt.savefig(plot_path)
                logger.info(f"Training summary plot saved to {plot_path}")
                
        except Exception as e:
            logger.warning(f"Could not generate training plot: {e}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted. Saving current model...")
        model.save(os.path.join(project_root, 'models/agent/interrupted_model_phase2_part2'))
        env.save(os.path.join(project_root, 'models/agent/vec_normalize_interrupted_phase2_part2.pkl'))

if __name__ == "__main__":
    resume_training_part2()
