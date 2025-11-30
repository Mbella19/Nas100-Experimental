"""
Gymnasium Trading Environment for the Sniper Agent.

Features:
- Multi-Discrete action space: [Direction (3), Size (4)]
- Frozen Market Analyst provides context vectors
- Reward shaping: PnL, transaction costs, FOMO penalty, chop avoidance
- Normalized observations (prevents scale inconsistencies)

Optimized for M2 Silicon with all float32 operations.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from typing import Dict, Tuple, Optional, Any
import gc


class TradingEnv(gym.Env):
    """
    Trading environment for the PPO Sniper Agent.

    Action Space:
        Multi-Discrete([3, 4]):
        - Direction: 0=Flat/Exit, 1=Long, 2=Short
        - Size: 0=0.25x, 1=0.5x, 2=0.75x, 3=1.0x

    Observation Space:
        Box containing:
        - Context vector from frozen Analyst (context_dim)
        - Position state: [position, entry_price_norm, unrealized_pnl_norm]
        - Market features: [atr, chop, adx, regime, sma_distance]

    Reward:
        Base PnL (pips) × position_size
        - Transaction cost when opening
        - FOMO penalty when flat during momentum
        - Chop penalty when holding in ranging market
    """

    metadata = {'render_modes': ['human']}

    # Position sizing multipliers
    POSITION_SIZES = (0.25, 0.5, 0.75, 1.0)

    def __init__(
        self,
        data_15m: np.ndarray,
        data_1h: np.ndarray,
        data_4h: np.ndarray,
        close_prices: np.ndarray,
        market_features: np.ndarray,
        analyst_model: Optional[torch.nn.Module] = None,
        context_dim: int = 64,
        lookback_15m: int = 48,
        lookback_1h: int = 24,
        lookback_4h: int = 12,
        spread_pips: float = 1.5,
        fomo_penalty: float = -2.0,   # Increased from -0.5 for better balance
        chop_penalty: float = -1.0,   # Increased from -0.3 for better balance
        fomo_threshold_atr: float = 2.0,
        chop_threshold: float = 60.0,
        max_steps: int = 2000,
        reward_scaling: float = 0.1,  # Scale PnL rewards to balance with penalties
        device: Optional[torch.device] = None
    ):
        """
        Initialize the trading environment.

        Args:
            data_15m: 15-minute feature data [num_samples, lookback_15m, features]
            data_1h: 1-hour feature data [num_samples, lookback_1h, features]
            data_4h: 4-hour feature data [num_samples, lookback_4h, features]
            close_prices: Close prices for PnL calculation [num_samples]
            market_features: Additional features [num_samples, n_features]
                            Expected: [atr, chop, adx, regime, sma_distance]
            analyst_model: Frozen Market Analyst for context generation
            context_dim: Dimension of context vector
            lookback_*: Lookback windows for each timeframe
            spread_pips: Transaction cost in pips
            fomo_penalty: Penalty for being flat during momentum
            chop_penalty: Penalty for holding in ranging market
            fomo_threshold_atr: ATR multiplier for FOMO detection
            chop_threshold: Choppiness index threshold
            max_steps: Maximum steps per episode
            reward_scaling: Scale factor for PnL rewards (0.1 = ±20 pips becomes ±2.0)
                           This balances PnL with penalties for "Sniper" behavior.
            device: Torch device for analyst inference
        """
        super().__init__()

        # Store data (ensure float32)
        self.data_15m = data_15m.astype(np.float32)
        self.data_1h = data_1h.astype(np.float32)
        self.data_4h = data_4h.astype(np.float32)
        self.close_prices = close_prices.astype(np.float32)
        self.market_features = market_features.astype(np.float32)

        # Analyst model
        self.analyst = analyst_model
        self.device = device or torch.device('cpu')
        self.context_dim = context_dim

        # Lookback windows
        self.lookback_15m = lookback_15m
        self.lookback_1h = lookback_1h
        self.lookback_4h = lookback_4h

        # Trading parameters
        self.spread_pips = spread_pips
        self.fomo_penalty = fomo_penalty
        self.chop_penalty = chop_penalty
        self.fomo_threshold_atr = fomo_threshold_atr
        self.chop_threshold = chop_threshold
        self.max_steps = max_steps
        self.reward_scaling = reward_scaling  # Scale PnL to balance with penalties

        # Calculate valid range (need enough lookback data)
        self.start_idx = max(lookback_15m, lookback_1h * 4, lookback_4h * 16)
        self.end_idx = len(close_prices) - 1
        self.n_samples = self.end_idx - self.start_idx

        # Action space: Multi-Discrete([direction, size])
        # Direction: 0=Flat, 1=Long, 2=Short
        # Size: 0=0.25, 1=0.5, 2=0.75, 3=1.0
        self.action_space = spaces.MultiDiscrete([3, 4])

        # Observation space
        # Context vector + position state (3) + market features (5)
        n_market_features = market_features.shape[1] if len(market_features.shape) > 1 else 5
        obs_dim = context_dim + 3 + n_market_features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # CRITICAL: Normalize market features to prevent scale inconsistencies!
        # Market features (ATR ~0.001, CHOP 0-100, ADX 0-100) have vastly different scales.
        # Compute mean/std for normalization (fit on ALL data for simplicity in RL)
        if len(market_features.shape) > 1 and market_features.shape[1] > 0:
            self.market_feat_mean = market_features.mean(axis=0).astype(np.float32)
            self.market_feat_std = market_features.std(axis=0).astype(np.float32)
            # Prevent division by zero for constant features
            self.market_feat_std = np.where(self.market_feat_std > 1e-8,
                                           self.market_feat_std,
                                           1.0).astype(np.float32)
        else:
            self.market_feat_mean = None
            self.market_feat_std = None

        # Episode state
        self.current_idx = self.start_idx
        self.position = 0  # -1: Short, 0: Flat, 1: Long
        self.position_size = 0.0
        self.entry_price = 0.0
        self.steps = 0
        self.total_pnl = 0.0
        self.trades = []
        self.prev_unrealized_pnl = 0.0  # Track for continuous PnL rewards

        # Precompute context vectors if analyst is provided
        self._precomputed_contexts = None
        if self.analyst is not None:
            self._precompute_contexts()

    def _precompute_contexts(self):
        """Precompute all context vectors for efficiency."""
        if self.analyst is None:
            return

        print("Precomputing context vectors...")
        self.analyst.eval()

        contexts = []
        batch_size = 64

        with torch.no_grad():
            for i in range(0, self.n_samples, batch_size):
                end_i = min(i + batch_size, self.n_samples)
                actual_indices = range(self.start_idx + i, self.start_idx + end_i)

                # Get batch data
                batch_15m = torch.tensor(
                    self.data_15m[list(actual_indices)],
                    device=self.device,
                    dtype=torch.float32
                )
                batch_1h = torch.tensor(
                    self.data_1h[list(actual_indices)],
                    device=self.device,
                    dtype=torch.float32
                )
                batch_4h = torch.tensor(
                    self.data_4h[list(actual_indices)],
                    device=self.device,
                    dtype=torch.float32
                )

                # Get context
                context = self.analyst.get_context(batch_15m, batch_1h, batch_4h)
                contexts.append(context.cpu().numpy())

                # Memory cleanup
                del batch_15m, batch_1h, batch_4h, context
                if i % (batch_size * 10) == 0:
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    gc.collect()

        self._precomputed_contexts = np.vstack(contexts).astype(np.float32)
        print(f"Precomputed {len(self._precomputed_contexts)} context vectors")

    def _get_context(self, idx: int) -> np.ndarray:
        """Get context vector for current index."""
        if self._precomputed_contexts is not None:
            # Use precomputed
            context_idx = idx - self.start_idx
            if 0 <= context_idx < len(self._precomputed_contexts):
                return self._precomputed_contexts[context_idx]

        if self.analyst is not None:
            # Compute on-the-fly
            with torch.no_grad():
                x_15m = torch.tensor(
                    self.data_15m[idx:idx+1],
                    device=self.device,
                    dtype=torch.float32
                )
                x_1h = torch.tensor(
                    self.data_1h[idx:idx+1],
                    device=self.device,
                    dtype=torch.float32
                )
                x_4h = torch.tensor(
                    self.data_4h[idx:idx+1],
                    device=self.device,
                    dtype=torch.float32
                )
                context = self.analyst.get_context(x_15m, x_1h, x_4h)
                return context.cpu().numpy().flatten()

        # No analyst - return zeros
        return np.zeros(self.context_dim, dtype=np.float32)

    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        # Context vector
        context = self._get_context(self.current_idx)

        # Position state
        current_price = self.close_prices[self.current_idx]
        atr = self.market_features[self.current_idx, 0] if len(self.market_features.shape) > 1 else 1.0

        # Normalize entry price and unrealized PnL
        if self.position != 0 and atr > 0:
            entry_price_norm = (self.entry_price - current_price) / (atr * 100)
            unrealized_pnl = self._calculate_unrealized_pnl()
            unrealized_pnl_norm = unrealized_pnl / 100  # Normalize by 100 pips
        else:
            entry_price_norm = 0.0
            unrealized_pnl_norm = 0.0

        position_state = np.array([
            float(self.position),
            entry_price_norm,
            unrealized_pnl_norm
        ], dtype=np.float32)

        # Market features (NORMALIZED to prevent scale inconsistencies)
        if len(self.market_features.shape) > 1:
            market_feat_raw = self.market_features[self.current_idx]
            # Apply Z-score normalization
            if self.market_feat_mean is not None and self.market_feat_std is not None:
                market_feat = ((market_feat_raw - self.market_feat_mean) /
                              self.market_feat_std).astype(np.float32)
            else:
                market_feat = market_feat_raw
        else:
            market_feat = np.zeros(5, dtype=np.float32)

        # Combine all (all components now on similar scales)
        obs = np.concatenate([context, position_state, market_feat])

        return obs.astype(np.float32)

    def _calculate_unrealized_pnl(self) -> float:
        """Calculate unrealized PnL in pips."""
        if self.position == 0:
            return 0.0

        current_price = self.close_prices[self.current_idx]
        pip_value = 0.0001  # EURUSD pip

        if self.position == 1:  # Long
            pnl_pips = (current_price - self.entry_price) / pip_value
        else:  # Short
            pnl_pips = (self.entry_price - current_price) / pip_value

        return pnl_pips * self.position_size

    def _execute_action(self, action: np.ndarray) -> Tuple[float, dict]:
        """
        Execute trading action and calculate reward.

        Returns:
            Tuple of (reward, info_dict)
        """
        direction = action[0]  # 0=Flat, 1=Long, 2=Short
        size_idx = action[1]   # 0-3
        new_size = self.POSITION_SIZES[size_idx]

        reward = 0.0
        info = {
            'trade_opened': False,
            'trade_closed': False,
            'pnl': 0.0
        }

        current_price = self.close_prices[self.current_idx]
        prev_price = self.close_prices[self.current_idx - 1] if self.current_idx > 0 else current_price

        # Get market conditions
        if len(self.market_features.shape) > 1:
            atr = self.market_features[self.current_idx, 0]
            chop = self.market_features[self.current_idx, 1]
        else:
            atr = 0.001
            chop = 50.0

        # Calculate price move for FOMO detection
        price_move = abs(current_price - prev_price)
        pip_value = 0.0001

        # Handle position changes
        # NOTE: PnL rewards are scaled by self.reward_scaling to balance with penalties
        # This prevents the agent from ignoring "Sniper" logic when potential PnL is high
        if direction == 0:  # Flat/Exit
            if self.position != 0:
                # Close existing position
                pnl = self._calculate_unrealized_pnl()
                reward += pnl * self.reward_scaling  # Scaled PnL reward
                info['trade_closed'] = True
                info['pnl'] = pnl  # Unscaled for tracking
                self.total_pnl += pnl
                self.trades.append({
                    'entry': self.entry_price,
                    'exit': current_price,
                    'direction': self.position,
                    'size': self.position_size,
                    'pnl': pnl
                })
                self.position = 0
                self.position_size = 0.0
                self.entry_price = 0.0

        elif direction == 1:  # Long
            if self.position == -1:  # Close short first
                pnl = self._calculate_unrealized_pnl()
                reward += pnl * self.reward_scaling  # Scaled PnL reward
                info['trade_closed'] = True
                info['pnl'] = pnl
                self.total_pnl += pnl
                self.trades.append({
                    'entry': self.entry_price,
                    'exit': current_price,
                    'direction': -1,
                    'size': self.position_size,
                    'pnl': pnl
                })

            if self.position != 1:  # Open long
                self.position = 1
                self.position_size = new_size
                self.entry_price = current_price
                reward -= self.spread_pips * new_size * self.reward_scaling  # Scaled cost
                info['trade_opened'] = True

        elif direction == 2:  # Short
            if self.position == 1:  # Close long first
                pnl = self._calculate_unrealized_pnl()
                reward += pnl * self.reward_scaling  # Scaled PnL reward
                info['trade_closed'] = True
                info['pnl'] = pnl
                self.total_pnl += pnl
                self.trades.append({
                    'entry': self.entry_price,
                    'exit': current_price,
                    'direction': 1,
                    'size': self.position_size,
                    'pnl': pnl
                })

            if self.position != -1:  # Open short
                self.position = -1
                self.position_size = new_size
                self.entry_price = current_price
                reward -= self.spread_pips * new_size * self.reward_scaling  # Scaled cost
                info['trade_opened'] = True

        # CRITICAL FIX: Add unrealized PnL change to prevent "death spiral"
        # Without this, holding a profitable trade in chop accumulates penalties
        # while getting NO reward until exit (e.g., -10.0 penalties vs +2.0 exit reward)
        # With this: agent gets continuous feedback on whether position is profitable
        current_unrealized_pnl = self._calculate_unrealized_pnl()
        pnl_delta = current_unrealized_pnl - self.prev_unrealized_pnl

        if self.position != 0:
            # Add the CHANGE in unrealized PnL to reward each step
            # This balances chop/FOMO penalties with actual trade performance
            reward += pnl_delta * self.reward_scaling
            info['unrealized_pnl'] = current_unrealized_pnl
            info['pnl_delta'] = pnl_delta

        # Update for next step
        self.prev_unrealized_pnl = current_unrealized_pnl

        # FOMO penalty: flat during high momentum move
        if self.position == 0:
            if price_move > self.fomo_threshold_atr * atr:
                reward += self.fomo_penalty
                info['fomo_triggered'] = True

        # Chop penalty: holding position in ranging market
        if self.position != 0 and chop > self.chop_threshold:
            reward += self.chop_penalty
            info['chop_triggered'] = True

        return reward, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)

        # Random starting point
        if options and 'start_idx' in options:
            self.current_idx = options['start_idx']
        else:
            self.current_idx = self.np_random.integers(
                self.start_idx,
                self.end_idx - self.max_steps
            )

        # Reset state
        self.position = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.steps = 0
        self.total_pnl = 0.0
        self.trades = []
        self.prev_unrealized_pnl = 0.0  # Reset for continuous PnL tracking

        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.

        Args:
            action: [direction, size] array

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Execute action
        reward, info = self._execute_action(action)

        # Move to next step
        self.current_idx += 1
        self.steps += 1

        # Check termination
        terminated = self.current_idx >= self.end_idx
        truncated = self.steps >= self.max_steps

        # Get new observation
        obs = self._get_observation()

        # Add episode info
        info['step'] = self.steps
        info['position'] = self.position
        info['total_pnl'] = self.total_pnl
        info['n_trades'] = len(self.trades)

        return obs, reward, terminated, truncated, info

    def render(self, mode: str = 'human'):
        """Render current state."""
        if mode == 'human':
            print(f"Step: {self.steps}, Position: {self.position}, "
                  f"Size: {self.position_size:.2f}, PnL: {self.total_pnl:.2f} pips")

    def close(self):
        """Clean up resources."""
        del self._precomputed_contexts
        gc.collect()


def create_env_from_dataframes(
    df_15m: 'pd.DataFrame',
    df_1h: 'pd.DataFrame',
    df_4h: 'pd.DataFrame',
    analyst_model: Optional[torch.nn.Module] = None,
    feature_cols: Optional[list] = None,
    config: Optional[object] = None
) -> TradingEnv:
    """
    Factory function to create TradingEnv from DataFrames.

    Args:
        df_15m: 15-minute DataFrame with features
        df_1h: 1-hour DataFrame with features
        df_4h: 4-hour DataFrame with features
        analyst_model: Trained Market Analyst
        feature_cols: Feature columns to use
        config: TradingConfig object

    Returns:
        TradingEnv instance
    """
    import pandas as pd

    if feature_cols is None:
        feature_cols = ['open', 'high', 'low', 'close', 'atr',
                       'pinbar', 'engulfing', 'doji', 'ema_trend', 'regime']

    # Get default config values
    lookback_15m = 48
    lookback_1h = 24
    lookback_4h = 12

    if config is not None:
        lookback_15m = getattr(config, 'lookback_15m', 48)
        lookback_1h = getattr(config, 'lookback_1h', 24)
        lookback_4h = getattr(config, 'lookback_4h', 12)

    # Prepare windowed data
    n_samples = len(df_15m) - max(lookback_15m, lookback_1h * 4, lookback_4h * 16)

    # Create windows for each timeframe
    data_15m = np.array([
        df_15m[feature_cols].iloc[i:i + lookback_15m].values
        for i in range(n_samples)
    ], dtype=np.float32)

    data_1h = np.array([
        df_1h[feature_cols].iloc[i:i + lookback_1h].values
        for i in range(n_samples)
    ], dtype=np.float32)

    data_4h = np.array([
        df_4h[feature_cols].iloc[i:i + lookback_4h].values
        for i in range(n_samples)
    ], dtype=np.float32)

    # Close prices for PnL
    close_prices = df_15m['close'].values[lookback_15m:lookback_15m + n_samples].astype(np.float32)

    # Market features for reward shaping
    market_cols = ['atr', 'chop', 'adx', 'regime', 'sma_distance']
    available_cols = [c for c in market_cols if c in df_15m.columns]
    market_features = df_15m[available_cols].values[lookback_15m:lookback_15m + n_samples].astype(np.float32)

    return TradingEnv(
        data_15m=data_15m,
        data_1h=data_1h,
        data_4h=data_4h,
        close_prices=close_prices,
        market_features=market_features,
        analyst_model=analyst_model,
        lookback_15m=lookback_15m,
        lookback_1h=lookback_1h,
        lookback_4h=lookback_4h
    )
