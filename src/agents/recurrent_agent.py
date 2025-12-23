"""
RecurrentPPO (LSTM-based) Sniper Agent wrapper using sb3-contrib.

EXPERIMENTAL: This provides an alternative to the standard MlpPolicy PPO
that can learn temporal patterns through LSTM hidden states.

Key differences from SniperAgent:
- Uses RecurrentPPO from sb3-contrib instead of PPO from stable_baselines3
- Uses MlpLstmPolicy instead of MlpPolicy
- predict() must track and pass LSTM hidden states
- Episode boundaries require state reset

CRITICAL for inference: The predict() method signature differs:
  - Standard PPO:   action, _ = model.predict(obs)
  - RecurrentPPO:   action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts)

Why RecurrentPPO for Trading?
- Markets are POMDPs (Partially Observable MDPs)
- Standard PPO only sees current observation snapshot
- LSTM can learn patterns like:
  - "Price consolidated after my entry" (requires memory)
  - "Volatility regime just shifted" (requires context)
  - "This is the 3rd rejection at this level" (requires counting)
"""

import os
import torch
import numpy as np
from typing import Optional, Dict, Any, Tuple, Callable
from pathlib import Path
import gc

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / ".mplconfig"),
)

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

import gymnasium as gym


def linear_schedule(initial_value: float, final_value: float = None) -> Callable[[float], float]:
    """
    Linear learning rate schedule for RecurrentPPO.

    Args:
        initial_value: Starting learning rate
        final_value: Ending learning rate (default: 5% of initial)

    Returns:
        A function that takes progress (1.0 -> 0.0) and returns current LR
    """
    if final_value is None:
        final_value = initial_value * 0.05

    def schedule(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)

    return schedule


class RecurrentEntropyScheduleCallback(BaseCallback):
    """
    Stepped entropy schedule callback for RecurrentPPO.

    Same phases as standard PPO:
    Phase 1 (0-20M):    ent_coef = 0.05  (explore)
    Phase 2 (20M-40M):  ent_coef = 0.025 (transition)
    Phase 3 (40M+):     ent_coef = 0.001 (exploit)
    """

    def __init__(
        self,
        phase1_steps: int = 20_000_000,
        phase2_steps: int = 20_000_000,
        phase1_ent: float = 0.05,
        phase2_ent: float = 0.025,
        phase3_ent: float = 0.001,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.phase1_steps = phase1_steps
        self.phase2_steps = phase2_steps
        self.phase1_ent = phase1_ent
        self.phase2_ent = phase2_ent
        self.phase3_ent = phase3_ent
        self.current_phase = 1

    def _on_step(self) -> bool:
        steps = self.num_timesteps

        if steps < self.phase1_steps:
            current_ent_coef = self.phase1_ent
            new_phase = 1
        elif steps < self.phase1_steps + self.phase2_steps:
            current_ent_coef = self.phase2_ent
            new_phase = 2
        else:
            current_ent_coef = self.phase3_ent
            new_phase = 3

        self.model.ent_coef = current_ent_coef

        if new_phase != self.current_phase:
            phase_names = {1: "EXPLORE", 2: "TRANSITION", 3: "EXPLOIT"}
            print(f"\n[RecurrentEntropySchedule] Phase {new_phase} ({phase_names[new_phase]}): ent_coef = {current_ent_coef}\n")
            self.current_phase = new_phase

        return True


class RecurrentMemoryCleanupCallback(BaseCallback):
    """
    Memory cleanup callback for RecurrentPPO.
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
                print(f"[Recurrent] Memory cleanup at step {self.n_calls}")
        return True


class RecurrentTrainingMetricsCallback(BaseCallback):
    """
    Training metrics callback for RecurrentPPO.
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_trades_history = []

    def _on_step(self) -> bool:
        if self.locals.get('dones') is not None:
            for idx, done in enumerate(self.locals['dones']):
                if done:
                    info = self.locals['infos'][idx]
                    if 'episode' in info:
                        self.episode_rewards.append(info['episode']['r'])
                        self.episode_lengths.append(info['episode']['l'])
                        n_trades = info.get('n_trades', 0)
                        self.episode_trades_history.append(n_trades)

        if self.n_calls % self.log_freq == 0 and self.verbose > 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-100:])
                mean_length = np.mean(self.episode_lengths[-100:])
                mean_trades = np.mean(self.episode_trades_history[-100:]) if self.episode_trades_history else 0.0

                print(f"[Recurrent] Step {self.n_calls}: Mean Reward={mean_reward:.2f}, "
                      f"Mean Length={mean_length:.0f}, Mean Trades={mean_trades:.1f}")

        return True


class RecurrentSniperAgent:
    """
    RecurrentPPO-based Sniper Agent for the trading environment.

    Uses LSTM to maintain memory across timesteps, enabling the agent
    to learn temporal patterns like:
    - "I entered a trade 5 bars ago and volatility just spiked"
    - "Price has been consolidating for 10 bars after my entry"
    - "This is the 3rd rejection at resistance"

    IMPORTANT: This agent requires special handling during inference:
    - Must track lstm_states between predict() calls
    - Must reset lstm_states on episode boundaries
    - episode_start flag must be set correctly
    """

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 3e-4,
        n_steps: int = 512,           # Smaller than standard PPO for memory
        batch_size: int = 64,         # Smaller for memory efficiency
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.02,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        # LSTM-specific
        lstm_hidden_size: int = 128,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        # Network architecture
        net_arch_pi: Optional[list] = None,
        net_arch_vf: Optional[list] = None,
        device: Optional[str] = None,
        verbose: int = 1,
        seed: Optional[int] = None,
        use_lr_schedule: bool = True
    ):
        """
        Initialize the Recurrent Sniper Agent.

        Args:
            env: Gymnasium environment
            learning_rate: Learning rate for RecurrentPPO
            n_steps: Number of steps per update (smaller for LSTM memory)
            batch_size: Minibatch size (smaller for memory efficiency)
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_range: PPO clip range
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Maximum gradient norm
            lstm_hidden_size: LSTM hidden state dimension
            n_lstm_layers: Number of LSTM layers
            shared_lstm: Share LSTM between actor and critic
            enable_critic_lstm: Use LSTM for value function
            net_arch_pi: Policy network layers before LSTM
            net_arch_vf: Value network layers before LSTM
            device: 'mps', 'cuda', 'cpu', or None for auto
            verbose: Verbosity level
            seed: Random seed
            use_lr_schedule: Enable linear LR decay
        """
        self.env = env
        self.verbose = verbose

        # Network architecture
        if net_arch_pi is None:
            net_arch_pi = [128]
        if net_arch_vf is None:
            net_arch_vf = [128]

        policy_kwargs = {
            'net_arch': dict(pi=net_arch_pi, vf=net_arch_vf),
            'lstm_hidden_size': lstm_hidden_size,
            'n_lstm_layers': n_lstm_layers,
            'shared_lstm': shared_lstm,
            'enable_critic_lstm': enable_critic_lstm,
        }

        # Device selection - RecurrentPPO also prefers CPU for stability
        if device is None:
            device = self._select_device()
        else:
            device_type = device if isinstance(device, str) else device.type
            if device_type != "cpu":
                if verbose > 0:
                    print(f"Overriding RecurrentPPO device {device_type} -> cpu")
                device = "cpu"

        # Apply learning rate schedule if enabled
        lr_value = linear_schedule(learning_rate) if use_lr_schedule else learning_rate
        if use_lr_schedule and verbose > 0:
            print(f"Using linear LR schedule: {learning_rate:.2e} -> {learning_rate * 0.05:.2e}")

        try:
            self.model = RecurrentPPO(
                policy="MlpLstmPolicy",
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
                print(f"RecurrentSniperAgent initialized on device: {device}")
                print(f"  LSTM: hidden_size={lstm_hidden_size}, layers={n_lstm_layers}")
                print(f"  Shared LSTM: {shared_lstm}, Critic LSTM: {enable_critic_lstm}")
                print(f"  n_steps={n_steps}, batch_size={batch_size}")

        except Exception as e:
            # Fallback to CPU if other device fails
            if device != 'cpu':
                print(f"Failed to use {device}, falling back to CPU: {e}")
                self.model = RecurrentPPO(
                    policy="MlpLstmPolicy",
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

        # LSTM state tracking for inference
        self._lstm_states = None
        self._episode_starts = np.ones((1,), dtype=bool)

    def _select_device(self) -> str:
        """Select the best available device."""
        # RecurrentPPO is most stable on CPU (especially on Apple Silicon)
        return "cpu"

    def reset_lstm_states(self):
        """Reset LSTM hidden states. Call at episode boundaries."""
        self._lstm_states = None
        self._episode_starts = np.ones((1,), dtype=bool)

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
        Train the recurrent agent.

        Args:
            total_timesteps: Total training timesteps
            eval_env: Optional evaluation environment
            eval_freq: Evaluation frequency
            save_path: Path to save best model
            callbacks: Additional callbacks (list)
            callback: Single callback (for convenience)
            reset_num_timesteps: Whether to reset the current timestep count

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
            self.model.set_logger(configure(str(ppo_log_dir), ["csv", "log"]))
            if self.verbose > 0:
                print(f"RecurrentPPO logs: {ppo_log_dir}")

        # Build callback list
        callback_list = [
            RecurrentMemoryCleanupCallback(cleanup_freq=5000, verbose=self.verbose),
            RecurrentTrainingMetricsCallback(log_freq=2000, verbose=self.verbose),
            RecurrentEntropyScheduleCallback(
                phase1_steps=20_000_000,
                phase2_steps=20_000_000,
                phase1_ent=0.05,
                phase2_ent=0.025,
                phase3_ent=0.001,
                verbose=self.verbose
            )
        ]

        # Add Checkpoint Callback
        if save_path:
            checkpoint_path = Path(save_path) / "checkpoints"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            callback_list.append(CheckpointCallback(
                save_freq=500_000,
                save_path=str(checkpoint_path),
                name_prefix="recurrent_model",
                save_replay_buffer=False,
                save_vecnormalize=True
            ))

        if callbacks:
            callback_list.extend(callbacks)

        if callback is not None:
            callback_list.append(callback)

        # Train
        if self.verbose > 0:
            print(f"Starting RecurrentPPO training for {total_timesteps:,} timesteps...")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=True,
            reset_num_timesteps=reset_num_timesteps
        )

        # Save final model
        if save_path is not None:
            final_path = Path(save_path) / "final_model"
            self.save(final_path)

            # Create marker file to identify this as a recurrent model
            marker_path = Path(save_path) / ".recurrent"
            marker_path.touch()

        return {
            'total_timesteps': total_timesteps,
            'device': self.device,
            'ppo_log_dir': str(ppo_log_dir) if ppo_log_dir is not None else None,
            'is_recurrent': True,
        }

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
        episode_start: bool = False,
        min_action_confidence: float = 0.0
    ) -> Tuple[np.ndarray, Any]:
        """
        Predict action with LSTM state tracking.

        CRITICAL: Unlike standard PPO, this method tracks internal LSTM states.
        Call reset_lstm_states() at episode boundaries or pass episode_start=True.

        Args:
            observation: Current observation
            deterministic: Use deterministic policy
            episode_start: Set True at episode start to reset LSTM
            min_action_confidence: Confidence threshold (same as SniperAgent)

        Returns:
            Tuple of (action, lstm_states)
        """
        if episode_start:
            self.reset_lstm_states()

        # RecurrentPPO requires episode_starts array
        self._episode_starts = np.array([episode_start], dtype=bool)

        # Get action with LSTM state tracking
        action, self._lstm_states = self.model.predict(
            observation,
            state=self._lstm_states,
            episode_start=self._episode_starts,
            deterministic=deterministic
        )

        # After first step, episode is no longer starting
        self._episode_starts = np.zeros((1,), dtype=bool)

        # Apply confidence thresholding if requested
        if min_action_confidence > 0.0:
            action = self._apply_confidence_threshold(
                observation, action, min_action_confidence
            )

        return action, self._lstm_states

    def _apply_confidence_threshold(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """Apply confidence thresholding to action (same as SniperAgent)."""
        try:
            obs_tensor, _ = self.model.policy.obs_to_tensor(observation)

            with torch.no_grad():
                # RecurrentPPO needs LSTM states for distribution
                # Use stored states
                dist = self.model.policy.get_distribution(
                    obs_tensor,
                    lstm_states=self._lstm_states,
                    episode_starts=torch.tensor([False], dtype=torch.bool)
                )

            # Get probabilities for Direction component
            if isinstance(dist.distribution, list):
                direction_dist = dist.distribution[0]
                direction_probs = direction_dist.probs
            else:
                all_logits = dist.distribution.logits
                direction_logits = all_logits[:, :3]
                direction_probs = torch.softmax(direction_logits, dim=1)

            if len(action.shape) == 1:
                chosen_dir = action[0]
                confidence = direction_probs[0, chosen_dir].item()

                if confidence < threshold and chosen_dir != 0:
                    action[0] = 0  # Force Flat
            else:
                for i in range(len(action)):
                    chosen_dir = action[i, 0]
                    confidence = direction_probs[i, chosen_dir].item()

                    if confidence < threshold and chosen_dir != 0:
                        action[i, 0] = 0

        except Exception:
            # Silently fail if structure mismatch
            pass

        return action

    def evaluate(
        self,
        env: gym.Env,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate the recurrent agent on an environment.

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
            # Reset LSTM states at episode start
            self.reset_lstm_states()

            obs, info = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            episode_length = 0
            episode_start = True  # First step is episode start

            while not done and not truncated:
                action, _ = self.predict(
                    obs,
                    deterministic=deterministic,
                    episode_start=episode_start
                )
                episode_start = False  # Subsequent steps are not episode start

                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            if 'total_pnl' in info:
                episode_pnls.append(info['total_pnl'])

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
            'n_episodes': n_episodes,
            'is_recurrent': True
        }

    def save(self, path: str | Path):
        """Save the model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        if self.verbose > 0:
            print(f"RecurrentPPO model saved to {path}")

    @classmethod
    def load(
        cls,
        path: str | Path,
        env: gym.Env,
        device: Optional[str] = None
    ) -> 'RecurrentSniperAgent':
        """
        Load a saved recurrent model.

        Args:
            path: Path to saved model
            env: Environment for the agent
            device: Device to load onto

        Returns:
            Loaded RecurrentSniperAgent
        """
        agent = cls.__new__(cls)
        agent.env = env
        agent.verbose = 1

        if device is None:
            device = agent._select_device()

        agent.model = RecurrentPPO.load(str(path), env=env, device=device)
        agent.device = device

        # Initialize LSTM states for inference
        agent._lstm_states = None
        agent._episode_starts = np.ones((1,), dtype=bool)

        print(f"RecurrentPPO model loaded from {path}")
        return agent


def create_recurrent_agent(
    env: gym.Env,
    agent_config: Optional[object] = None,
    recurrent_config: Optional[object] = None,
    device: Optional[str] = None
) -> RecurrentSniperAgent:
    """
    Factory function to create RecurrentSniperAgent with config.

    Args:
        env: Trading environment
        agent_config: AgentConfig object (for shared hyperparameters)
        recurrent_config: RecurrentAgentConfig object (for LSTM-specific params)
        device: Device for training

    Returns:
        RecurrentSniperAgent instance
    """
    if agent_config is None and recurrent_config is None:
        return RecurrentSniperAgent(env, device=device)

    # Get hyperparameters from configs
    lr = getattr(agent_config, 'learning_rate', 3e-4) if agent_config else 3e-4
    n_epochs = getattr(agent_config, 'n_epochs', 10) if agent_config else 10
    gamma = getattr(agent_config, 'gamma', 0.99) if agent_config else 0.99
    gae_lambda = getattr(agent_config, 'gae_lambda', 0.95) if agent_config else 0.95
    clip_range = getattr(agent_config, 'clip_range', 0.2) if agent_config else 0.2
    ent_coef = getattr(agent_config, 'ent_coef', 0.02) if agent_config else 0.02
    vf_coef = getattr(agent_config, 'vf_coef', 0.5) if agent_config else 0.5
    max_grad_norm = getattr(agent_config, 'max_grad_norm', 0.5) if agent_config else 0.5

    # Get recurrent-specific params
    if recurrent_config:
        n_steps = recurrent_config.n_steps
        batch_size = recurrent_config.batch_size
        lstm_hidden_size = recurrent_config.lstm_hidden_size
        n_lstm_layers = recurrent_config.n_lstm_layers
        shared_lstm = recurrent_config.shared_lstm
        enable_critic_lstm = recurrent_config.enable_critic_lstm
        net_arch_pi = list(recurrent_config.net_arch_pi)
        net_arch_vf = list(recurrent_config.net_arch_vf)
    else:
        n_steps = 512
        batch_size = 64
        lstm_hidden_size = 128
        n_lstm_layers = 1
        shared_lstm = False
        enable_critic_lstm = True
        net_arch_pi = [128]
        net_arch_vf = [128]

    return RecurrentSniperAgent(
        env=env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        lstm_hidden_size=lstm_hidden_size,
        n_lstm_layers=n_lstm_layers,
        shared_lstm=shared_lstm,
        enable_critic_lstm=enable_critic_lstm,
        net_arch_pi=net_arch_pi,
        net_arch_vf=net_arch_vf,
        device=device
    )
