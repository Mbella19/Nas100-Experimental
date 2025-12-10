"""
PPO Sniper Agent wrapper using Stable Baselines 3.

Features:
- MPS device support with CPU fallback
- Memory-efficient callbacks
- Training and inference methods
- Model saving/loading
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import gc

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

import gymnasium as gym


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
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
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
    - MPS/CPU device selection
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
        n_epochs: int = 20,           # Increased from 10
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.02,       # Increased for exploration
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        net_arch: Optional[list] = None,
        device: Optional[str] = None,
        verbose: int = 1,
        seed: Optional[int] = None
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

        # Create PPO model
        try:
            self.model = PPO(
                policy="MlpPolicy",
                env=env,
                learning_rate=learning_rate,
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
                    learning_rate=learning_rate,
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
        # Note: SB3 has limited MPS support, CPU is often more stable
        if torch.backends.mps.is_available():
            # Try MPS, but be prepared to fall back
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        return 'cpu'

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
        # Build callback list
        callback_list = [
            MemoryCleanupCallback(cleanup_freq=5000, verbose=self.verbose),
            TrainingMetricsCallback(log_freq=2000, verbose=self.verbose)
        ]

        # Add Checkpoint Callback (Save every 100k steps)
        if save_path:
            checkpoint_path = Path(save_path).parent / "checkpoints"
            callback_list.append(CheckpointCallback(
                save_freq=100_000,
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
            'device': self.device
        }

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> tuple:
        """
        Predict action for given observation.

        Args:
            observation: Current observation
            deterministic: Use deterministic policy

        Returns:
            Tuple of (action, states)
        """
        action, states = self.model.predict(observation, deterministic=deterministic)
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
        device: Optional[str] = None
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
    config: Optional[object] = None
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
        return SniperAgent(env)

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
        net_arch=config.net_arch if hasattr(config, 'net_arch') else None
    )
