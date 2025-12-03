"""
Training script for the PPO Sniper Agent.

Trains the RL agent using a frozen Market Analyst to provide
context vectors for decision making.

Memory-optimized for Apple M2 Silicon.

Features:
- Comprehensive logging of training progress
- Detailed reward and action statistics
- Training visualizations (reward curves, action distributions)
- Episode-level tracking and analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from datetime import datetime
import gc
import json

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from ..models.analyst import load_analyst, MarketAnalyst
from ..environments.trading_env import TradingEnv
from ..agents.sniper_agent import SniperAgent, create_agent
from ..utils.logging_config import setup_logging, get_logger
from ..utils.metrics import calculate_trading_metrics, TradingMetrics
from ..data.features import (
    compute_regime_labels, 
    add_market_sessions,
    detect_fractals,
    detect_structure_breaks
)

logger = get_logger(__name__)


class AgentTrainingLogger(BaseCallback):
    """
    Custom callback for detailed agent training logging.

    Tracks:
    - Episode rewards
    - Action distributions
    - PnL statistics
    - Win rate evolution
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_freq: int = 1000,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.log_dir = Path(log_dir) if log_dir else None
        self.log_freq = log_freq

        # Tracking variables
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_pnls: List[float] = []
        self.episode_trades: List[int] = []
        self.episode_win_rates: List[float] = []
        self.action_counts = {0: 0, 1: 0, 2: 0}  # Flat, Long, Short
        self.size_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # Position sizes: 0.25, 0.5, 0.75, 1.0
        self.timestep_rewards: List[float] = []

        # Current episode tracking
        self.current_ep_reward = 0
        self.current_ep_length = 0
        self.current_ep_actions = []

        # Training start time
        self.start_time = None

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def _on_training_start(self):
        self.start_time = datetime.now()
        logger.info("=" * 70)
        logger.info("PPO AGENT TRAINING STARTED")
        logger.info("=" * 70)
        logger.info(f"Total timesteps: {self.model.num_timesteps if hasattr(self.model, 'num_timesteps') else 'N/A'}")
        logger.info(f"Log directory: {self.log_dir}")
        logger.info("=" * 70)

    def _on_step(self) -> bool:
        # Track rewards
        if len(self.locals.get('rewards', [])) > 0:
            reward = self.locals['rewards'][0]
            self.timestep_rewards.append(reward)
            self.current_ep_reward += reward
            self.current_ep_length += 1

        # Track actions (direction and size for MultiDiscrete)
        if len(self.locals.get('actions', [])) > 0:
            action = self.locals['actions'][0]
            if isinstance(action, np.ndarray):
                if len(action) >= 2:
                    # MultiDiscrete action: [direction, size]
                    direction = int(action[0])
                    size = int(action[1])
                    self.action_counts[direction] = self.action_counts.get(direction, 0) + 1
                    self.size_counts[size] = self.size_counts.get(size, 0) + 1
                    self.current_ep_actions.append(direction)
                else:
                    # Single action (fallback)
                    direction = int(action[0]) if len(action) == 1 else int(action)
                    self.action_counts[direction] = self.action_counts.get(direction, 0) + 1
                    self.current_ep_actions.append(direction)
            elif isinstance(action, (int, np.integer)):
                self.action_counts[int(action)] = self.action_counts.get(int(action), 0) + 1
                self.current_ep_actions.append(int(action))

        # Check for episode done
        dones = self.locals.get('dones', [False])
        if any(dones):
            self.episode_rewards.append(self.current_ep_reward)
            self.episode_lengths.append(self.current_ep_length)

            # Get episode info if available
            infos = self.locals.get('infos', [{}])
            if len(infos) > 0 and infos[0]:
                info = infos[0]
                self.episode_pnls.append(info.get('total_pnl', 0.0))
                # CRITICAL FIX: Use 'n_trades' (what env actually sets) instead of 'total_trades'
                self.episode_trades.append(info.get('n_trades', 0))
                # Calculate win rate from trades if available
                n_trades = info.get('n_trades', 0)
                win_rate = 0.0
                if n_trades > 0 and 'trades' in info:
                    wins = sum(1 for t in info['trades'] if t.get('pnl', 0) > 0)
                    win_rate = wins / n_trades
                self.episode_win_rates.append(win_rate)

            # Log episode summary
            n_episodes = len(self.episode_rewards)
            if n_episodes % 10 == 0:
                self._log_episode_summary(n_episodes)

            # Reset current episode tracking
            self.current_ep_reward = 0
            self.current_ep_length = 0
            self.current_ep_actions = []

        # Periodic detailed logging
        if self.n_calls % self.log_freq == 0:
            self._log_training_progress()

        return True

    def _log_episode_summary(self, n_episodes: int):
        """Log summary for recent episodes."""
        recent_rewards = self.episode_rewards[-10:]
        recent_pnls = self.episode_pnls[-10:] if self.episode_pnls else [0]

        logger.info("-" * 50)
        logger.info(f"Episode {n_episodes} Summary:")
        logger.info(f"  Recent Avg Reward: {np.mean(recent_rewards):.2f}")
        logger.info(f"  Recent Avg PnL: {np.mean(recent_pnls):.2f} pips")
        logger.info(f"  Episode Length: {self.episode_lengths[-1]}")

        if self.episode_win_rates:
            logger.info(f"  Win Rate: {self.episode_win_rates[-1]*100:.1f}%")

    def _log_training_progress(self):
        """Log overall training progress."""
        total_actions = sum(self.action_counts.values())
        if total_actions == 0:
            return

        action_pcts = {k: v/total_actions*100 for k, v in self.action_counts.items()}

        logger.info("-" * 50)
        logger.info(f"Training Progress @ {self.n_calls} steps:")
        logger.info(f"  Episodes completed: {len(self.episode_rewards)}")
        logger.info(f"  Action Distribution: Flat={action_pcts.get(0, 0):.1f}%, "
                   f"Long={action_pcts.get(1, 0):.1f}%, Short={action_pcts.get(2, 0):.1f}%")

        if self.episode_rewards:
            logger.info(f"  Avg Episode Reward: {np.mean(self.episode_rewards):.2f}")
            logger.info(f"  Max Episode Reward: {np.max(self.episode_rewards):.2f}")
            logger.info(f"  Min Episode Reward: {np.min(self.episode_rewards):.2f}")

        if self.episode_pnls:
            logger.info(f"  Avg PnL: {np.mean(self.episode_pnls):.2f} pips")

        # Memory info
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

    def _on_training_end(self):
        """Log final training summary and create visualizations."""
        total_time = (datetime.now() - self.start_time).total_seconds()

        logger.info("=" * 70)
        logger.info("PPO AGENT TRAINING COMPLETED")
        logger.info("=" * 70)
        logger.info(f"Total Training Time: {total_time/60:.1f} minutes")
        logger.info(f"Total Episodes: {len(self.episode_rewards)}")
        logger.info(f"Total Timesteps: {self.n_calls}")

        if self.episode_rewards:
            logger.info(f"Final Avg Reward (last 100): {np.mean(self.episode_rewards[-100:]):.2f}")

        if self.episode_pnls:
            logger.info(f"Final Avg PnL (last 100): {np.mean(self.episode_pnls[-100:]):.2f} pips")

        logger.info("=" * 70)

        # Save metrics
        if self.log_dir:
            self._save_metrics()
            self._create_visualizations()

    def _save_metrics(self):
        """Save training metrics to JSON."""
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_pnls': self.episode_pnls,
            'episode_trades': self.episode_trades,
            'episode_win_rates': self.episode_win_rates,
            'action_counts': self.action_counts,
            'total_timesteps': self.n_calls,
            'total_episodes': len(self.episode_rewards)
        }

        metrics_path = self.log_dir / 'agent_training_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Metrics saved to: {metrics_path}")

    def _create_visualizations(self):
        """Create training visualizations."""
        if len(self.episode_rewards) < 10:
            logger.warning("Not enough episodes for visualizations")
            return

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # Episode rewards
        ax = axes[0, 0]
        ax.plot(self.episode_rewards, alpha=0.3, color='blue')
        # Smoothed
        window = min(50, len(self.episode_rewards) // 5 + 1)
        smoothed = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(self.episode_rewards)), smoothed,
               color='red', linewidth=2, label=f'Smoothed (w={window})')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # PnL per episode
        ax = axes[0, 1]
        if self.episode_pnls:
            ax.plot(self.episode_pnls, alpha=0.3, color='green')
            if len(self.episode_pnls) >= window:
                smoothed_pnl = np.convolve(self.episode_pnls, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(self.episode_pnls)), smoothed_pnl,
                       color='darkgreen', linewidth=2)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('PnL (pips)')
        ax.set_title('Episode PnL')
        ax.grid(True, alpha=0.3)

        # Win rate evolution
        ax = axes[0, 2]
        if self.episode_win_rates:
            ax.plot(self.episode_win_rates, alpha=0.5, color='purple')
            ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='50%')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Win Rate')
        ax.set_title('Win Rate Evolution')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Action distribution
        ax = axes[1, 0]
        action_names = ['Flat', 'Long', 'Short']
        action_vals = [self.action_counts.get(i, 0) for i in range(3)]
        colors = ['gray', 'green', 'red']
        bars = ax.bar(action_names, action_vals, color=colors)
        ax.set_ylabel('Count')
        ax.set_title('Action Distribution')
        # Add percentage labels
        total = sum(action_vals)
        if total > 0:
            for bar, val in zip(bars, action_vals):
                pct = val / total * 100
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{pct:.1f}%', ha='center', va='bottom')

        # Episode length distribution
        ax = axes[1, 1]
        ax.hist(self.episode_lengths, bins=30, color='blue', alpha=0.7)
        ax.axvline(x=np.mean(self.episode_lengths), color='red',
                  linestyle='--', label=f'Mean: {np.mean(self.episode_lengths):.0f}')
        ax.set_xlabel('Episode Length')
        ax.set_ylabel('Count')
        ax.set_title('Episode Length Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Cumulative PnL
        ax = axes[1, 2]
        if self.episode_pnls:
            cumulative_pnl = np.cumsum(self.episode_pnls)
            ax.plot(cumulative_pnl, color='green', linewidth=2)
            ax.fill_between(range(len(cumulative_pnl)), cumulative_pnl, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative PnL (pips)')
        ax.set_title('Cumulative PnL')
        ax.grid(True, alpha=0.3)

        plt.suptitle('PPO Agent Training Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = self.log_dir / 'agent_training_summary.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Training visualization saved to: {save_path}")

    def get_metrics(self) -> Dict:
        """Get all tracked metrics."""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_pnls': self.episode_pnls,
            'episode_trades': self.episode_trades,
            'episode_win_rates': self.episode_win_rates,
            'action_counts': self.action_counts
        }


def prepare_env_data(
    df_15m,
    df_1h,
    df_4h,
    feature_cols: list,
    lookback_15m: int = 20,
    lookback_1h: int = 24,
    lookback_4h: int = 12
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare windowed data for the trading environment.
    
    FIXED: 1H and 4H data now correctly subsampled from the aligned 15m index.
    Since df_1h and df_4h are aligned to the 15m index via forward-fill,
    we subsample every 4th bar for 1H and every 16th bar for 4H.

    Returns:
        Tuple of (data_15m, data_1h, data_4h, close_prices, market_features)
    """
    # Subsampling ratios: how many 15m bars per higher TF bar
    subsample_1h = 4   # 4 x 15m = 1H
    subsample_4h = 16  # 16 x 15m = 4H
    
    # Calculate valid range - need enough indices for subsampled lookback
    start_idx = max(lookback_15m, lookback_1h * subsample_1h, lookback_4h * subsample_4h)
    n_samples = len(df_15m) - start_idx - 1

    logger.info(f"Preparing {n_samples} samples for environment")
    logger.info(f"  15m: {lookback_15m} bars = {lookback_15m * 15 / 60:.1f} hours")
    logger.info(f"  1H: {lookback_1h} bars = {lookback_1h} hours (using {lookback_1h * subsample_1h} aligned indices)")
    logger.info(f"  4H: {lookback_4h} bars = {lookback_4h * 4} hours (using {lookback_4h * subsample_4h} aligned indices)")

    # Prepare windowed data
    data_15m = np.zeros((n_samples, lookback_15m, len(feature_cols)), dtype=np.float32)
    data_1h = np.zeros((n_samples, lookback_1h, len(feature_cols)), dtype=np.float32)
    data_4h = np.zeros((n_samples, lookback_4h, len(feature_cols)), dtype=np.float32)

    features_15m = df_15m[feature_cols].values.astype(np.float32)
    features_1h = df_1h[feature_cols].values.astype(np.float32)
    features_4h = df_4h[feature_cols].values.astype(np.float32)

    for i in range(n_samples):
        actual_idx = start_idx + i
        # 15m: direct indexing
        data_15m[i] = features_15m[actual_idx - lookback_15m:actual_idx]
        
        # FIXED: 1H - subsample every 4th bar from aligned data
        idx_range_1h = list(range(actual_idx - lookback_1h * subsample_1h, actual_idx, subsample_1h))
        data_1h[i] = features_1h[idx_range_1h]
        
        # FIXED: 4H - subsample every 16th bar from aligned data  
        idx_range_4h = list(range(actual_idx - lookback_4h * subsample_4h, actual_idx, subsample_4h))
        data_4h[i] = features_4h[idx_range_4h]

    # Close prices
    close_prices = df_15m['close'].values[start_idx:start_idx + n_samples].astype(np.float32)

    # Market features for reward shaping (includes S/R for breakout vs chase detection)
    market_cols = ['atr', 'chop', 'adx', 'regime', 'sma_distance', 'dist_to_support', 'dist_to_resistance']
    available_cols = [c for c in market_cols if c in df_15m.columns]

    if len(available_cols) > 0:
        market_features = df_15m[available_cols].values[start_idx:start_idx + n_samples].astype(np.float32)
    else:
        # Create dummy features if not available
        market_features = np.zeros((n_samples, 5), dtype=np.float32)
        market_features[:, 0] = 0.001  # Default ATR
        market_features[:, 1] = 50.0   # Default CHOP
        market_features[:, 2] = 20.0   # Default ADX

    return data_15m, data_1h, data_4h, close_prices, market_features


def create_trading_env(
    data_15m: np.ndarray,
    data_1h: np.ndarray,
    data_4h: np.ndarray,
    close_prices: np.ndarray,
    market_features: np.ndarray,
    analyst_model: Optional[MarketAnalyst] = None,
    config: Optional[object] = None,
    device: Optional[torch.device] = None,
    market_feat_mean: Optional[np.ndarray] = None,  # Pre-computed from training
    market_feat_std: Optional[np.ndarray] = None,   # Pre-computed from training
    regime_labels: Optional[np.ndarray] = None,     # Regime labels for balanced sampling
    use_regime_sampling: bool = True                # Enable regime-balanced episode starts
) -> TradingEnv:
    """
    Create the trading environment.

    Args:
        data_*: Prepared window data
        close_prices: Close prices for PnL
        market_features: Features for reward shaping
        analyst_model: Frozen Market Analyst
        config: TradingConfig
        device: Torch device

    Returns:
        TradingEnv instance
    """
    # Default configuration
    spread_pips = 1.5
    slippage_pips = 1.0     # Realistic slippage: 0.5-2 pips per trade execution
    fomo_penalty = -0.2     # Reduced: ~2 pip equivalent (was -2.0)
    chop_penalty = -0.1     # Reduced: ~1 pip equivalent (was -1.0)
    fomo_threshold_atr = 2.0
    chop_threshold = 60.0
    max_steps = 2000
    reward_scaling = 0.1    # Scale PnL to balance with penalties
    context_dim = 64
    
    # Risk Management defaults
    sl_atr_multiplier = 1.5
    tp_atr_multiplier = 3.0
    use_stop_loss = True
    use_stop_loss = True
    use_take_profit = True
    
    # Volatility Sizing
    volatility_sizing = True
    volatility_sizing = True
    risk_pips_target = 15.0
    
    # Analyst Alignment - Force agent to trade only in analyst's predicted direction
    # This gives agent the 51-53% directional edge; agent learns WHEN and HOW MUCH to trade
    enforce_analyst_alignment = True

    if config is not None:
        # FIX: Access config.trading for trading parameters (not config directly)
        trading_cfg = getattr(config, 'trading', config)
        spread_pips = getattr(trading_cfg, 'spread_pips', spread_pips)
        slippage_pips = getattr(trading_cfg, 'slippage_pips', slippage_pips)
        fomo_penalty = getattr(trading_cfg, 'fomo_penalty', fomo_penalty)
        chop_penalty = getattr(trading_cfg, 'chop_penalty', chop_penalty)
        fomo_threshold_atr = getattr(trading_cfg, 'fomo_threshold_atr', fomo_threshold_atr)
        chop_threshold = getattr(trading_cfg, 'chop_threshold', chop_threshold)
        max_steps = getattr(trading_cfg, 'max_steps_per_episode', max_steps)
        reward_scaling = getattr(trading_cfg, 'reward_scaling', reward_scaling)
        # Risk Management
        sl_atr_multiplier = getattr(trading_cfg, 'sl_atr_multiplier', sl_atr_multiplier)
        tp_atr_multiplier = getattr(trading_cfg, 'tp_atr_multiplier', tp_atr_multiplier)
        use_stop_loss = getattr(trading_cfg, 'use_stop_loss', use_stop_loss)
        use_take_profit = getattr(trading_cfg, 'use_take_profit', use_take_profit)
        # Analyst Alignment
        enforce_analyst_alignment = getattr(trading_cfg, 'enforce_analyst_alignment', enforce_analyst_alignment)

        # Log config values to verify they're applied
        logger.info(f"Config applied: fomo_penalty={fomo_penalty}, reward_scaling={reward_scaling}, "
                    f"slippage_pips={slippage_pips}, enforce_analyst_alignment={enforce_analyst_alignment}")

    if analyst_model is not None:
        context_dim = analyst_model.context_dim
        # Get num_classes from analyst (binary=2, multi-class=3)
        num_classes = getattr(analyst_model, 'num_classes', 2)
    else:
        num_classes = 2  # Default to binary

    env = TradingEnv(
        data_15m=data_15m,
        data_1h=data_1h,
        data_4h=data_4h,
        close_prices=close_prices,
        market_features=market_features,
        analyst_model=analyst_model,
        context_dim=context_dim,
        spread_pips=spread_pips,
        slippage_pips=slippage_pips,
        fomo_penalty=fomo_penalty,
        chop_penalty=chop_penalty,
        fomo_threshold_atr=fomo_threshold_atr,
        chop_threshold=chop_threshold,
        max_steps=max_steps,
        reward_scaling=reward_scaling,
        device=device,
        market_feat_mean=market_feat_mean,
        market_feat_std=market_feat_std,
        # Risk Management
        sl_atr_multiplier=sl_atr_multiplier,
        tp_atr_multiplier=tp_atr_multiplier,
        use_stop_loss=use_stop_loss,
        use_take_profit=use_take_profit,
        # Regime-balanced sampling
        regime_labels=regime_labels,
        use_regime_sampling=use_regime_sampling,
        # Volatility Sizing
        volatility_sizing=volatility_sizing,
        risk_pips_target=risk_pips_target,
        # Classification mode
        num_classes=num_classes,
        # Analyst Alignment
        enforce_analyst_alignment=enforce_analyst_alignment
    )

    return env


def train_agent(
    df_15m,
    df_1h,
    df_4h,
    feature_cols: list,
    analyst_path: str,
    save_path: str,
    config: Optional[object] = None,
    device: Optional[torch.device] = None,
    total_timesteps: int = 500_000
) -> Tuple[SniperAgent, Dict]:
    """
    Main function to train the PPO Sniper Agent.

    Args:
        df_15m: 15-minute DataFrame with features
        df_1h: 1-hour DataFrame with features
        df_4h: 4-hour DataFrame with features
        feature_cols: Feature columns used
        analyst_path: Path to trained analyst model
        save_path: Path to save agent
        config: Configuration object
        device: Torch device
        total_timesteps: Total training timesteps

    Returns:
        Tuple of (trained agent, training info)
    """
    # Setup logging for this run
    log_dir = Path(save_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(log_dir), name=__name__)

    # Device selection
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    logger.info(f"Training agent on device: {device}")

    # Load frozen analyst
    logger.info(f"Loading analyst from {analyst_path}")
    # Use the same features as Analyst training (11 base + 7 extra = 18 features)
    model_features = [
        'returns', 'volatility', 'pinbar', 'engulfing', 'doji',
        'ema_trend', 'ema_crossover', 'regime', 'sma_distance',
        'dist_to_resistance', 'dist_to_support',
        # Extra features added during training
        'session_asian', 'session_london', 'session_ny',
        'bos_bullish', 'bos_bearish', 'choch_bullish', 'choch_bearish'
    ]
    
    feature_dims = {
        '15m': len(model_features),
        '1h': len(model_features),
        '4h': len(model_features)
    }
    analyst = load_analyst(analyst_path, feature_dims, device, freeze=True)
    logger.info("Analyst loaded and frozen")

    # Log analyst info
    logger.info(f"Analyst context_dim: {analyst.context_dim}")
    logger.info(f"Analyst parameters: {sum(p.numel() for p in analyst.parameters()):,} (frozen)")

    # Prepare data
    logger.info("Preparing environment data...")
    
    # Add Market Sessions
    logger.info("Adding market session features...")
    df_15m = add_market_sessions(df_15m)
    df_1h = add_market_sessions(df_1h)
    df_4h = add_market_sessions(df_4h)

    # Add Structure Features (BOS/CHoCH)
    logger.info("Adding structure features (BOS/CHoCH)...")
    for df in [df_15m, df_1h, df_4h]:
        f_high, f_low = detect_fractals(df)
        struct_df = detect_structure_breaks(df, f_high, f_low)
        for col in struct_df.columns:
            df[col] = struct_df[col]

    # Update feature columns if not already included
    session_cols = ['session_asian', 'session_london', 'session_ny']
    struct_cols = ['bos_bullish', 'bos_bearish', 'choch_bullish', 'choch_bearish']
    
    for col in session_cols + struct_cols:
        if col not in feature_cols:
            feature_cols.append(col)

    lookback_15m = config.analyst.lookback_15m
    lookback_1h = config.analyst.lookback_1h
    lookback_4h = config.analyst.lookback_4h

    data_15m, data_1h, data_4h, close_prices, market_features = prepare_env_data(
        df_15m, df_1h, df_4h, feature_cols,
        lookback_15m, lookback_1h, lookback_4h
    )

    logger.info(f"Data shapes: 15m={data_15m.shape}, 1h={data_1h.shape}, 4h={data_4h.shape}")
    logger.info(f"Price range: {close_prices.min():.5f} - {close_prices.max():.5f}")

    # Split into train/eval
    split_idx = int(0.85 * len(close_prices))
    logger.info(f"Train samples: {split_idx}, Eval samples: {len(close_prices) - split_idx}")

    train_data = (
        data_15m[:split_idx],
        data_1h[:split_idx],
        data_4h[:split_idx],
        close_prices[:split_idx],
        market_features[:split_idx]
    )

    eval_data = (
        data_15m[split_idx:],
        data_1h[split_idx:],
        data_4h[split_idx:],
        close_prices[split_idx:],
        market_features[split_idx:]
    )

    # Compute market feature normalization stats from TRAINING data only
    # FIXED: This prevents look-ahead bias by using only training statistics
    train_market_features = train_data[4]  # market_features from train_data tuple
    market_feat_mean = train_market_features.mean(axis=0).astype(np.float32)
    market_feat_std = train_market_features.std(axis=0).astype(np.float32)
    market_feat_std = np.where(market_feat_std > 1e-8, market_feat_std, 1.0).astype(np.float32)
    
    logger.info("Market feature normalization stats (from training data):")
    logger.info(f"  Mean: {market_feat_mean}")
    logger.info(f"  Std:  {market_feat_std}")

    # Compute regime labels for balanced sampling (training data only)
    # This ensures agent learns from BULLISH, BEARISH, and RANGING markets equally
    logger.info("Computing regime labels for balanced sampling...")
    train_regime_labels = compute_regime_labels(df_15m.iloc[:split_idx], lookback=20)
    regime_counts = {
        'Bullish': (train_regime_labels == 0).sum(),
        'Ranging': (train_regime_labels == 1).sum(),
        'Bearish': (train_regime_labels == 2).sum()
    }
    logger.info(f"Regime distribution: {regime_counts}")

    # Create environments
    logger.info("Creating training environment...")
    train_env = create_trading_env(
        *train_data,
        analyst_model=analyst,
        config=config,
        device=device,
        market_feat_mean=market_feat_mean,
        market_feat_std=market_feat_std,
        regime_labels=train_regime_labels,  # Enable regime-balanced sampling
        use_regime_sampling=True
    )

    logger.info("Creating evaluation environment...")
    eval_env = create_trading_env(
        *eval_data,
        analyst_model=analyst,
        config=config,
        device=device,
        market_feat_mean=market_feat_mean,  # Use TRAINING stats for eval too
        market_feat_std=market_feat_std,
        regime_labels=None,  # Eval uses random sampling (no regime bias)
        use_regime_sampling=False
    )

    # Log environment info
    logger.info(f"Observation space: {train_env.observation_space}")
    logger.info(f"Action space: {train_env.action_space}")

    # Wrap environments
    train_env = Monitor(train_env)
    eval_env = Monitor(eval_env)

    # Create agent
    logger.info("Creating PPO agent...")
    agent = create_agent(train_env, config.agent)

    # Create training logger callback
    training_callback = AgentTrainingLogger(
        log_dir=str(log_dir),
        log_freq=5000,
        verbose=1
    )

    # Train
    logger.info(f"Starting training for {total_timesteps:,} timesteps...")
    logger.info("-" * 70)

    training_info = agent.train(
        total_timesteps=total_timesteps,
        eval_env=eval_env,
        eval_freq=10_000,
        save_path=save_path,
        callback=training_callback
    )

    # Get metrics from callback
    training_info['callback_metrics'] = training_callback.get_metrics()

    # Final evaluation
    logger.info("=" * 70)
    logger.info("Running final evaluation...")
    eval_results = agent.evaluate(eval_env, n_episodes=20)
    training_info['final_eval'] = eval_results

    logger.info("-" * 70)
    logger.info("FINAL EVALUATION RESULTS:")
    logger.info(f"  Mean Reward: {eval_results['mean_reward']:.2f} +/- {eval_results['std_reward']:.2f}")
    logger.info(f"  Mean PnL: {eval_results['mean_pnl']:.2f} pips")
    logger.info(f"  Win Rate: {eval_results.get('win_rate', 0)*100:.1f}%")
    logger.info(f"  Mean Trades per Episode: {eval_results.get('mean_trades', 0):.1f}")
    logger.info("-" * 70)

    # Save training summary
    summary = {
        'total_timesteps': total_timesteps,
        'total_episodes': len(training_callback.episode_rewards),
        'final_mean_reward': eval_results['mean_reward'],
        'final_mean_pnl': eval_results['mean_pnl'],
        'final_win_rate': eval_results.get('win_rate', 0),
        'action_distribution': training_callback.action_counts,
        'avg_episode_length': float(np.mean(training_callback.episode_lengths)) if training_callback.episode_lengths else 0
    }

    summary_path = log_dir / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Training summary saved to: {summary_path}")

    # Cleanup
    train_env.close()
    eval_env.close()
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return agent, training_info


def load_and_evaluate(
    agent_path: str,
    df_15m,
    df_1h,
    df_4h,
    feature_cols: list,
    analyst_path: str,
    device: Optional[torch.device] = None,
    n_episodes: int = 50
) -> Dict:
    """
    Load a trained agent and evaluate it.

    Args:
        agent_path: Path to saved agent
        df_*: DataFrames
        feature_cols: Feature columns
        analyst_path: Path to analyst model
        device: Torch device
        n_episodes: Number of evaluation episodes

    Returns:
        Evaluation results
    """
    if device is None:
        device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    logger.info(f"Loading agent from {agent_path}")
    logger.info(f"Evaluating on {n_episodes} episodes")

    # Load analyst
    feature_dims = {'15m': len(feature_cols), '1h': len(feature_cols), '4h': len(feature_cols)}
    analyst = load_analyst(analyst_path, feature_dims, device, freeze=True)

    # Prepare data (use last portion as test)
    data_15m, data_1h, data_4h, close_prices, market_features = prepare_env_data(
        df_15m, df_1h, df_4h, feature_cols
    )

    # Use last 15% as test
    test_start = int(0.85 * len(close_prices))
    test_data = (
        data_15m[test_start:],
        data_1h[test_start:],
        data_4h[test_start:],
        close_prices[test_start:],
        market_features[test_start:]
    )

    # Create test environment
    test_env = create_trading_env(*test_data, analyst_model=analyst, device=device)
    test_env = Monitor(test_env)

    # Load agent
    agent = SniperAgent.load(agent_path, test_env, device='cpu')  # SB3 more stable on CPU

    # Evaluate
    logger.info("Running evaluation...")
    results = agent.evaluate(test_env, n_episodes=n_episodes)

    logger.info("=" * 70)
    logger.info("EVALUATION RESULTS:")
    logger.info(f"  Mean Reward: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
    logger.info(f"  Mean PnL: {results['mean_pnl']:.2f} pips")
    logger.info(f"  Win Rate: {results.get('win_rate', 0)*100:.1f}%")
    logger.info("=" * 70)

    test_env.close()
    return results


if __name__ == '__main__':
    print("Use this module via: python -m src.training.train_agent")
    print("Or import and call train_agent() function")
