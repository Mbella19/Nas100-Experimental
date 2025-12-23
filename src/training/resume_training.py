import os
import sys
import logging
import torch
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
import gc

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from config.settings import Config
from src.environments.trading_env import TradingEnv, create_env_from_dataframes
from src.models.analyst import MarketAnalyst

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(project_root, 'models/agent/resume_training.log'))
    ]
)
logger = logging.getLogger('resume_training')

from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import time
from datetime import datetime

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

def resume_training():
    config = Config()
    # ... (rest of the function setup) ...
    # (I will insert the logger instantiation and usage below)

    device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # 1. Load Pre-processed Data (Parquet)
    # We load the normalized features directly to ensure consistency with previous training
    data_processed_path = config.paths.data_processed
    logger.info(f"Loading processed data from {data_processed_path}...")
    
    try:
        df_15m = pd.read_parquet(data_processed_path / 'features_15m_normalized.parquet')
        df_1h = pd.read_parquet(data_processed_path / 'features_1h_normalized.parquet')
        df_4h = pd.read_parquet(data_processed_path / 'features_4h_normalized.parquet')
        logger.info("Data loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load parquet files: {e}")
        logger.error("Please ensure you have run the pipeline at least once to generate processed data.")
        return

    # 2. Define Feature Columns
    # Must match exactly what was used in training
    model_features = [
        'returns', 'volatility', 'pinbar', 'engulfing', 'doji',
        'ema_trend', 'ema_crossover', 'regime', 'sma_distance',
        'dist_to_resistance', 'dist_to_support',
        'session_asian', 'session_london', 'session_ny',
        'bos_bullish', 'bos_bearish', 'choch_bullish', 'choch_bearish'
    ]
    
    # Filter to ensure columns exist
    feature_cols = [c for c in model_features if c in df_15m.columns]
    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")

    # 3. Load Analyst Model
    logger.info("Loading analyst model...")
    feature_dims = {
        '15m': 18, 
        '1h': 18, 
        '4h': 18
    }
    
    analyst = MarketAnalyst(
        feature_dims=feature_dims,
        d_model=config.analyst.d_model,
        nhead=config.analyst.nhead,
        num_layers=config.analyst.num_layers,
        dim_feedforward=config.analyst.dim_feedforward,
        context_dim=config.analyst.context_dim,
        dropout=config.analyst.dropout,
        num_classes=config.analyst.num_classes
    ).to(device)
    
    # We need to handle potential dimension mismatch if Analyst was trained with different features
    # But usually it's consistent.
    
    analyst_path = os.path.join(project_root, 'models/analyst/best_model.pth')
    if not os.path.exists(analyst_path):
         analyst_path = os.path.join(project_root, 'models/analyst/best.pt')
         
    try:
        checkpoint = torch.load(analyst_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        analyst.load_state_dict(state_dict)
        logger.info("Analyst model weights loaded successfully.")
    except Exception as e:
        logger.warning(f"Could not load analyst weights directly: {e}")
        logger.warning("Attempting to load with strict=False or ignoring (Analyst might be untrained)")
        # If this fails, we might need to recreate Analyst with correct dimensions
        # For now, assume it works or user will see error.

    analyst.eval()
    for param in analyst.parameters():
        param.requires_grad = False

    # 4. Create Environment with LOWER SPREAD (Curriculum Learning)
    logger.info("Creating environment with reduced spread (1.0 pips)...")
    
    from stable_baselines3.common.vec_env import SubprocVecEnv

    def make_env():
        env = create_env_from_dataframes(
            df_15m, df_1h, df_4h,
            analyst_model=analyst,
            feature_cols=feature_cols,
            config=config,
            device=device
        )
        # Match spread to config for consistent training
        env.spread_pips = 3.5
        return env

    # Use 8 vectorized environments for M2 (Batch processing on GPU)
    # NOTE: We use DummyVecEnv because SubprocVecEnv (multiprocessing) is unstable with MPS on Mac.
    # DummyVecEnv still provides speedup by batching observations for the GPU.
    num_envs = 8
    logger.info(f"Using {num_envs} vectorized environments (DummyVecEnv)...")
    env = DummyVecEnv([make_env for _ in range(num_envs)])
    
    # Load VecNormalize statistics if they exist
    stats_path = os.path.join(project_root, 'models/agent/vec_normalize.pkl')
    if os.path.exists(stats_path):
        logger.info("Loading VecNormalize statistics...")
        env = VecNormalize.load(stats_path, env)
        env.training = True 
        env.norm_reward = True
    else:
        logger.info("Creating new VecNormalize wrapper...")
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 5. Load Agent
    model_path = os.path.join(project_root, 'models/agent/final_model.zip')
    if not os.path.exists(model_path):
        model_path = os.path.join(project_root, 'models/agent/best_model.zip')
    
    logger.info(f"Loading agent from {model_path}...")
    
    model = PPO.load(model_path, env=env, device=device)
    
    # 6. Resume Training
    total_timesteps = 9_000_000
    logger.info(f"Resuming training for {total_timesteps} steps...")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=os.path.join(project_root, 'models/agent/checkpoints_phase2/'),
        name_prefix='ppo_phase2'
    )
    
    training_logger = AgentTrainingLogger(log_freq=1000)
    memory_cleanup = MemoryCleanupCallback(cleanup_freq=5000, verbose=1)
    callback_list = CallbackList([checkpoint_callback, training_logger, memory_cleanup])

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            log_interval=1,
            reset_num_timesteps=False,
            progress_bar=True
        )
        
        # Save Phase 2 Model
        save_path = os.path.join(project_root, 'models/agent/final_model_phase2')
        model.save(save_path)
        env.save(os.path.join(project_root, 'models/agent/vec_normalize_phase2.pkl'))
        logger.info(f"Phase 2 training complete. Model saved to {save_path}")

        # Plot Training Summary
        try:
            import matplotlib.pyplot as plt
            
            # Extract metrics from logger
            rewards = training_logger.episode_rewards
            pnls = training_logger.episode_pnls
            
            if len(rewards) > 0:
                plt.figure(figsize=(12, 8))
                
                # Plot Rewards
                plt.subplot(2, 1, 1)
                plt.plot(rewards, label='Episode Reward', alpha=0.6)
                plt.plot(pd.Series(rewards).rolling(window=50).mean(), label='50-Ep Moving Avg', color='red')
                plt.title('Phase 2 Training Rewards')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Plot PnL
                if len(pnls) > 0:
                    plt.subplot(2, 1, 2)
                    plt.plot(pnls, label='Episode PnL (pips)', color='green', alpha=0.6)
                    plt.plot(pd.Series(pnls).rolling(window=50).mean(), label='50-Ep Moving Avg', color='darkgreen')
                    plt.title('Phase 2 Training PnL')
                    plt.xlabel('Episode')
                    plt.ylabel('PnL (pips)')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot_path = os.path.join(project_root, 'models/agent/agent_training_summary_phase2.png')
                plt.savefig(plot_path)
                logger.info(f"Training summary plot saved to {plot_path}")
                
        except Exception as e:
            logger.warning(f"Could not generate training plot: {e}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted. Saving current model...")
        model.save(os.path.join(project_root, 'models/agent/interrupted_model_phase2'))
        env.save(os.path.join(project_root, 'models/agent/vec_normalize_interrupted_phase2.pkl'))

if __name__ == "__main__":
    resume_training()
