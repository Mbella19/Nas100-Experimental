"""
PPO Sniper Agent wrapper using Stable Baselines 3.

Features:
- CPU-only PPO by default (stable SB3 training)
- Memory-efficient callbacks
- Training and inference methods
- Model saving/loading
"""

import os
import torch
import numpy as np
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import gc

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / ".mplconfig"),
)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

import gymnasium as gym


def linear_schedule(initial_value: float, final_value: float = None) -> Callable[[float], float]:
    """
    Linear learning rate schedule for PPO.
    
    Args:
        initial_value: Starting learning rate
        final_value: Ending learning rate (default: 5% of initial)
    
    Returns:
        A function that takes progress (1.0 -> 0.0) and returns current LR
    """
    if final_value is None:
        final_value = initial_value * 0.05  # Decay to 5% by default
    
    def schedule(progress_remaining: float) -> float:
        """
        progress_remaining goes from 1.0 (start) to 0.0 (end)
        """
        return final_value + progress_remaining * (initial_value - final_value)
    
    return schedule


class EntropyScheduleCallback(BaseCallback):
    """
    Callback with 4-phase stepped entropy for 1B timestep training.

    v24: Extended to 4 phases for 1B training run.

    Phase 1 (0-200M):     ent_coef = 0.10  (extra high explore)
    Phase 2 (200M-500M):  ent_coef = 0.05  (explore)
    Phase 3 (500M-800M):  ent_coef = 0.02  (transition)
    Phase 4 (800M-1B):    ent_coef = 0.01  (exploit - never below 0.01!)
    """

    def __init__(
        self,
        phase1_steps: int = 200_000_000,  # First 200M: extra high explore
        phase2_steps: int = 300_000_000,  # Next 300M (200M-500M): explore
        phase3_steps: int = 300_000_000,  # Next 300M (500M-800M): transition
        phase1_ent: float = 0.10,         # Extra high exploration
        phase2_ent: float = 0.05,         # Explore entropy
        phase3_ent: float = 0.02,         # Transition entropy
        phase4_ent: float = 0.01,         # Exploit entropy (never below 0.01!)
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.phase1_steps = phase1_steps
        self.phase2_steps = phase2_steps
        self.phase3_steps = phase3_steps
        self.phase1_ent = phase1_ent
        self.phase2_ent = phase2_ent
        self.phase3_ent = phase3_ent
        self.phase4_ent = phase4_ent
        self.current_phase = 1

    def _on_step(self) -> bool:
        steps = self.num_timesteps

        # Determine current phase and entropy
        if steps < self.phase1_steps:
            # Phase 1: Extra High Explore (0-200M)
            current_ent_coef = self.phase1_ent
            new_phase = 1
        elif steps < self.phase1_steps + self.phase2_steps:
            # Phase 2: Explore (200M-500M)
            current_ent_coef = self.phase2_ent
            new_phase = 2
        elif steps < self.phase1_steps + self.phase2_steps + self.phase3_steps:
            # Phase 3: Transition (500M-800M)
            current_ent_coef = self.phase3_ent
            new_phase = 3
        else:
            # Phase 4: Exploit (800M-1B)
            current_ent_coef = self.phase4_ent
            new_phase = 4

        # Update the model's entropy coefficient
        self.model.ent_coef = current_ent_coef

        # Log phase transitions
        if new_phase != self.current_phase:
            phase_names = {1: "EXTRA EXPLORE", 2: "EXPLORE", 3: "TRANSITION", 4: "EXPLOIT"}
            print(f"\n[EntropySchedule] Phase {new_phase} ({phase_names[new_phase]}): ent_coef = {current_ent_coef}\n")
            self.current_phase = new_phase

        return True


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


class TrainingMetricsCallback(BaseCallback):
    """
    Callback to log training metrics.
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_trades_history = []  # Track trades per episode

    def _on_step(self) -> bool:
        # Check for episode end
        if self.locals.get('dones') is not None:
            for idx, done in enumerate(self.locals['dones']):
                if done:
                    info = self.locals['infos'][idx]
                    if 'episode' in info:
                        self.episode_rewards.append(info['episode']['r'])
                        self.episode_lengths.append(info['episode']['l'])
                        
                        # Capture n_trades if available
                        n_trades = info.get('n_trades', 0)
                        self.episode_trades_history.append(n_trades)

        # Log periodically
        if self.n_calls % self.log_freq == 0 and self.verbose > 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-100:])
                mean_length = np.mean(self.episode_lengths[-100:])
                mean_trades = np.mean(self.episode_trades_history[-100:]) if self.episode_trades_history else 0.0
                
                print(f"Step {self.n_calls}: Mean Reward={mean_reward:.2f}, "
                      f"Mean Length={mean_length:.0f}, Mean Trades={mean_trades:.1f}")

        return True


class SniperAgent:
    """
    PPO-based Sniper Agent for the trading environment.

    Wraps Stable Baselines 3 PPO with:
    - CPU device selection (default)
    - Custom network architecture
    - Memory-efficient training
    - Evaluation and inference methods
    """

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 256,        # Increased from 64 for stability
        n_epochs: int = 4,            # Match config/settings.py (was 20, caused overfitting)
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.02,       # Increased for exploration
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        net_arch: Optional[list] = None,
        device: Optional[str | torch.device] = None,
        verbose: int = 1,
        seed: Optional[int] = None,
        use_lr_schedule: bool = True  # NEW: Enable linear LR decay
    ):
        """
        Initialize the Sniper Agent.

        Args:
            env: Gymnasium environment
            learning_rate: Learning rate for PPO
            n_steps: Number of steps per update
            batch_size: Minibatch size
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_range: PPO clip range
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Maximum gradient norm
            net_arch: Network architecture [hidden_sizes]
            device: 'mps', 'cuda', 'cpu', or None for auto
            verbose: Verbosity level
            seed: Random seed
        """
        self.env = env
        self.verbose = verbose

        # Network architecture
        if net_arch is None:
            net_arch = [256, 256]

        policy_kwargs = {
            'net_arch': dict(pi=net_arch, vf=net_arch)
        }

        # Device selection with fallback
        if device is None:
            device = self._select_device()
        else:
            # Normalize to string for consistent handling.
            device_type = device if isinstance(device, str) else device.type
            if device_type != "cpu":
                if verbose > 0:
                    print(f"Overriding PPO device {device_type} -> cpu")
                device = "cpu"

        # Create PPO model
        # Apply learning rate schedule if enabled
        lr_value = linear_schedule(learning_rate) if use_lr_schedule else learning_rate
        if use_lr_schedule and verbose > 0:
            print(f"Using linear LR schedule: {learning_rate:.2e} -> {learning_rate * 0.05:.2e}")
        
        try:
            self.model = PPO(
                policy="MlpPolicy",
                env=env,
                learning_rate=lr_value,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                policy_kwargs=policy_kwargs,
                device=device,
                verbose=verbose,
                seed=seed
            )
            self.device = device
            if verbose > 0:
                print(f"SniperAgent initialized on device: {device}")

        except Exception as e:
            # Fallback to CPU if MPS fails
            if device != 'cpu':
                print(f"Failed to use {device}, falling back to CPU: {e}")
                self.model = PPO(
                    policy="MlpPolicy",
                    env=env,
                    learning_rate=lr_value,  # Use scheduled LR, not raw value
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    clip_range=clip_range,
                    ent_coef=ent_coef,
                    vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm,
                    policy_kwargs=policy_kwargs,
                    device='cpu',
                    verbose=verbose,
                    seed=seed
                )
                self.device = 'cpu'
            else:
                raise

    def _select_device(self) -> str:
        """Select the best available device."""
        # NOTE: SB3 PPO is most stable on CPU (especially on Apple Silicon).
        return "cpu"

    def train(
        self,
        total_timesteps: int = 500_000,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 10_000,
        save_path: Optional[str] = None,
        callbacks: Optional[list] = None,
        callback: Optional[BaseCallback] = None,
        reset_num_timesteps: bool = True
    ) -> Dict[str, Any]:
        """
        Train the agent.

        Args:
            total_timesteps: Total training timesteps
            eval_env: Optional evaluation environment
            eval_freq: Evaluation frequency
            save_path: Path to save best model
            callbacks: Additional callbacks (list)
            callback: Single callback (for convenience)
            reset_num_timesteps: Whether to reset the current timestep count (False for resuming)

        Returns:
            Training info dictionary
        """
        ppo_log_dir = None
        if save_path:
            from stable_baselines3.common.logger import configure
            from datetime import datetime

            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            ppo_log_dir = Path(save_path) / "ppo_logs" / run_id
            ppo_log_dir.mkdir(parents=True, exist_ok=True)
            # Persist PPO training metrics (incl. loss) for offline inspection.
            # `progress.csv` contains fields like: train/loss, train/value_loss, rollout/ep_rew_mean, etc.
            self.model.set_logger(configure(str(ppo_log_dir), ["csv", "log"]))
            if self.verbose > 0:
                print(f"SB3 PPO logs (incl. loss): {ppo_log_dir}")

        # Build callback list
        callback_list = [
            MemoryCleanupCallback(cleanup_freq=5000, verbose=self.verbose),
            TrainingMetricsCallback(log_freq=2000, verbose=self.verbose),
            # Entropy Schedule: Stepped phases (uses class defaults)
            # Phase 1 (0-50M): explore @ 0.05
            # Phase 2 (50M-100M): transition @ 0.02
            # Phase 3 (100M+): exploit @ 0.01 (never below 0.01!)
            EntropyScheduleCallback(verbose=self.verbose)
        ]

        # Add Checkpoint Callback (Save every 100k steps)
        if save_path:
            checkpoint_path = Path(save_path) / "checkpoints"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            callback_list.append(CheckpointCallback(
                save_freq=500_000,  # Save every 500K steps (was 100K)
                save_path=str(checkpoint_path),
                name_prefix="sniper_model",
                save_replay_buffer=False,
                save_vecnormalize=True
            ))

        if callbacks:
            callback_list.extend(callbacks)

        # Support single callback parameter
        if callback is not None:
            callback_list.append(callback)

        # NOTE: We deliberately DO NOT use EvalCallback for model selection.
        # Using eval performance to select the "best" model causes overfitting
        # to the eval set. Instead, we save the FINAL model after all training.
        # The eval_env is only used for monitoring, not selection.
        #
        # If you want periodic eval logging (without selection), add custom callback.

        # Train
        if self.verbose > 0:
            print(f"Starting training for {total_timesteps:,} timesteps...")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=True,
            reset_num_timesteps=reset_num_timesteps
        )

        # Save final model
        if save_path is not None:
            self.save(Path(save_path) / "final_model")

        return {
            'total_timesteps': total_timesteps,
            'device': self.device,
            'ppo_log_dir': str(ppo_log_dir) if ppo_log_dir is not None else None,
        }

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
        min_action_confidence: float = 0.0
    ) -> tuple:
        """
        Predict action for given observation with optional confidence threshold.

        Args:
            observation: Current observation
            deterministic: Use deterministic policy
            min_action_confidence: Minimum probability required to take a non-flat action.
                                 If confidence < threshold, action is forced to Flat (0).
                                 Only applies to Direction (action[0]).

        Returns:
            Tuple of (action, states)
        """
        # Standard prediction
        action, states = self.model.predict(observation, deterministic=deterministic)

        # Apply confidence thresholding if requested
        if min_action_confidence > 0.0:
            # We need to get the probabilities from the policy
            # Convert observation to tensor
            obs_tensor, _ = self.model.policy.obs_to_tensor(observation)
            
            # Get distribution
            with torch.no_grad():
                dist = self.model.policy.get_distribution(obs_tensor)
            
            # Calculate probabilities from logits
            # SB3 MultiCategoricalDistribution stores logits in a specific way.
            # For MultiDiscrete, we usually have a list of Categorical distributions
            # or concatenated logits.
            
            # Helper to get probs for the first dimension (Direction)
            # The action space is MultiDiscrete.
            # dist.distribution is usually a list of Categorical distributions
            # IF using Independent(OneHotCategorical) or similar.
            
            # Accessing logits/probs directly depends on SB3 implementation details.
            # A robust way is to inspect `dist.distribution.probs` if available, 
            # or `dist.distribution` params.
            
            # For MultiDiscrete, SB3 often flattens the logits.
            # We know Direction is the first component.
            # Let's assume standard SB3 implementation for MultiDiscrete.
            
            # Safe access to probabilities for the Direction component (index 0)
            try:
                # Check if dist.distribution is a list (SB3 MultiDiscrete behavior)
                if isinstance(dist.distribution, list):
                    # Index 0 is Direction, Index 1 is Size
                    direction_dist = dist.distribution[0]
                    direction_probs = direction_dist.probs # Shape: (batch_size, 3)
                else:
                    # Fallback for other potential structures
                    # Try to access logits directly if not a list
                    all_logits = dist.distribution.logits
                    direction_logits = all_logits[:, :3]
                    direction_probs = torch.softmax(direction_logits, dim=1)

                # Get confidence of the CHOSEN action for Direction
                # Check if vectorized (batch size > 1) or single
                if len(action.shape) == 1:
                    # Single environment, action is [dir, size, ...]
                    chosen_dir = action[0]
                    confidence = direction_probs[0, chosen_dir].item()
                    
                    if confidence < min_action_confidence and chosen_dir != 0:
                        # DEBUG: Print intervention
                        # print(f"THRESHOLD INTERVENTION: Action {chosen_dir} (Conf {confidence:.2f} < {min_action_confidence}) -> FLAT")
                        # Force Flat
                        action[0] = 0
                        
                else:
                    # Vectorized environments (n_envs, n_actions)
                    # Iterate over envs
                    for i in range(len(action)):
                        chosen_dir = action[i, 0]
                        confidence = direction_probs[i, chosen_dir].item()
                        
                        if confidence < min_action_confidence and chosen_dir != 0:
                            # Force Flat
                            action[i, 0] = 0
                            
            except Exception as e:
                # print(f"DEBUG: Confidence check failed: {e}")
                pass # Silently fail if structure mismatch to avoid crashing trade execution

        return action, states

    def evaluate(
        self,
        env: gym.Env,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate the agent on an environment.

        Args:
            env: Evaluation environment
            n_episodes: Number of episodes
            deterministic: Use deterministic policy

        Returns:
            Evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        episode_pnls = []
        episode_trades = []
        episode_win_rates = []

        for ep in range(n_episodes):
            obs, info = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            episode_length = 0

            while not done and not truncated:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            if 'total_pnl' in info:
                episode_pnls.append(info['total_pnl'])

            # CRITICAL FIX: Track trade count and win rate
            n_trades = info.get('n_trades', 0)
            episode_trades.append(n_trades)

            win_rate = 0.0
            if n_trades > 0 and 'trades' in info:
                wins = sum(1 for t in info['trades'] if t.get('pnl', 0) > 0)
                win_rate = wins / n_trades
            episode_win_rates.append(win_rate)

        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'mean_pnl': np.mean(episode_pnls) if episode_pnls else 0.0,
            'mean_trades': np.mean(episode_trades) if episode_trades else 0.0,
            'win_rate': np.mean(episode_win_rates) if episode_win_rates else 0.0,
            'n_episodes': n_episodes
        }

    def save(self, path: str | Path):
        """Save the model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        if self.verbose > 0:
            print(f"Model saved to {path}")

    @classmethod
    def load(
        cls,
        path: str | Path,
        env: gym.Env,
        device: Optional[str | torch.device] = None
    ) -> 'SniperAgent':
        """
        Load a saved model.

        Args:
            path: Path to saved model
            env: Environment for the agent
            device: Device to load onto

        Returns:
            Loaded SniperAgent
        """
        agent = cls.__new__(cls)
        agent.env = env
        agent.verbose = 1

        if device is None:
            device = agent._select_device()

        agent.model = PPO.load(str(path), env=env, device=device)
        agent.device = device

        return agent


def create_agent(
    env: gym.Env,
    config: Optional[object] = None,
    device: Optional[str | torch.device] = None
) -> SniperAgent:
    """
    Factory function to create SniperAgent with config.

    Args:
        env: Trading environment
        config: AgentConfig object

    Returns:
        SniperAgent instance
    """
    if config is None:
        return SniperAgent(env, device=device)

    return SniperAgent(
        env=env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        net_arch=config.net_arch if hasattr(config, 'net_arch') else None,
        device=device
    )


def create_agent_with_config(
    env: gym.Env,
    config: Optional[object] = None,
    device: Optional[str | torch.device] = None
) -> 'SniperAgent | RecurrentSniperAgent':
    """
    Factory function to create either SniperAgent or RecurrentSniperAgent.

    This is the primary factory function for agent creation. It selects between
    standard PPO (SniperAgent) and RecurrentPPO (RecurrentSniperAgent) based on
    the config.recurrent_agent.use_recurrent flag.

    Args:
        env: Trading environment
        config: Full Config object (with agent and recurrent_agent sub-configs)
        device: Device for training

    Returns:
        SniperAgent or RecurrentSniperAgent based on config
    """
    if config is None:
        return SniperAgent(env, device=device)

    # Check if recurrent mode is enabled
    recurrent_cfg = getattr(config, 'recurrent_agent', None)
    use_recurrent = recurrent_cfg.use_recurrent if recurrent_cfg else False

    if use_recurrent:
        # Import here to avoid circular imports and only when needed
        from .recurrent_agent import create_recurrent_agent

        print("[Agent Factory] Creating RecurrentSniperAgent (LSTM-based) - EXPERIMENTAL")
        return create_recurrent_agent(
            env=env,
            agent_config=config.agent,
            recurrent_config=config.recurrent_agent,
            device=device
        )
    else:
        print("[Agent Factory] Creating SniperAgent (standard PPO)")
        return create_agent(env, config.agent, device=device)
