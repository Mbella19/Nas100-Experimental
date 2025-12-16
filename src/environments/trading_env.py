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
import pandas as pd
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
    # Reward: pure continuous mark-to-market PnL delta (no exit "banking").
    # The agent receives PnL changes as they occur, aligning rewards with the
    # equity curve and avoiding incentive to prematurely close to "lock in" a
    # large pending exit reward.
    # Holding bonus (anti-reward-hacking):
    # - Gated: only after trade is net profitable beyond entry costs + buffer.
    # - Progress-based: pays only when reaching new profit milestones (high-water).
    # - Capped: max bonus per trade is a small fraction of entry cost (in reward units).
    HOLDING_BONUS_AMOUNT = 0.01
    HOLDING_BONUS_CAP_FRACTION_OF_ENTRY_COST = 0.2
    HOLDING_BONUS_BUFFER_FRACTION_OF_ENTRY_COST = 1.0
    HOLDING_BONUS_MILESTONE_PIPS = 5.0

    def __init__(
        self,
        data_5m: np.ndarray,
        data_15m: np.ndarray,
        data_45m: np.ndarray,
        close_prices: np.ndarray,
        market_features: np.ndarray,
        analyst_model: Optional[torch.nn.Module] = None,
        context_dim: int = 64,
        lookback_5m: int = 48,
        lookback_15m: int = 16,
        lookback_45m: int = 6,
        pip_value: float = 1.0,       # NAS100: 1 point = 1.0 price movement
        lot_size: float = 1.0,        # NAS100 CFD lot size
        point_multiplier: float = 1.0, # Dollar per point per lot
        spread_pips: float = 10.0,     # NAS100 typical spread
        slippage_pips: float = 0.0,   # NAS100 slippage
        fomo_penalty: float = 0.0,   # v15: Meaningful penalty for missing moves (was 0.0)
        chop_penalty: float = 0.0,    # Disabled for stability
        fomo_threshold_atr: float = 4.0,  # v15: Trigger on >1.5x ATR moves (was 2.0)
        chop_threshold: float = 80.0,     # Only extreme chop triggers penalty
        max_steps: int = 500,         # ~1 week for rapid regime cycling
        reward_scaling: float = 0.01,  # NAS100: 1.0 per 100 points (was 1.0 per pip for EURUSD)
        trade_entry_bonus: float = 0.01,   # NAS100: offset spread cost
        device: Optional[torch.device] = None,
        market_feat_mean: Optional[np.ndarray] = None,  # Pre-computed from training data
        market_feat_std: Optional[np.ndarray] = None,    # Pre-computed from training data
        pre_windowed: bool = True,  # FIXED: If True, data is already windowed (start_idx=0)
        # Risk Management
        sl_atr_multiplier: float = 2.0, # v17: Stop Loss = ATR * 2 (tighter for 1:6 R:R)
        tp_atr_multiplier: float = 12.0, # Take Profit = ATR * multiplier
        use_stop_loss: bool = True,     # Enable/disable stop-loss
        use_take_profit: bool = True,   # Enable/disable take-profit
        # Regime-balanced sampling
        regime_labels: Optional[np.ndarray] = None,  # 0=Bullish, 1=Ranging, 2=Bearish
        use_regime_sampling: bool = True,  # Sample episodes balanced across regimes
        # Volatility Sizing (Dollar-based risk)
        volatility_sizing: bool = True,  # Scale position size to maintain constant dollar risk
        risk_per_trade: float = 100.0,   # Dollar risk per trade (e.g., $100 per trade)
        # Classification mode
        num_classes: int = 2,  # Binary (2) vs multi-class (3) - affects observation size
        # Analyst Alignment
        enforce_analyst_alignment: bool = False,  # If True, restrict actions to analyst direction
        # Pre-computed Analyst outputs (for sequential context)
        precomputed_analyst_cache: Optional[dict] = None,  # {'contexts': np.ndarray, 'probs': np.ndarray}
        # OHLC data for visualization (real candle data)
        ohlc_data: Optional[np.ndarray] = None,  # Shape: (n_samples, 4) with [open, high, low, close]
        timestamps: Optional[np.ndarray] = None,  # Optional timestamps for real time axis
        noise_level: float = 0.001,  # Anti-overfitting noise (default enabled)
        # v16: Sparse Rewards Mode
        use_sparse_rewards: bool = True,  # If True, only give reward on trade exit (not per-bar)
        # v17: Loss Tolerance Buffer
        loss_tolerance_atr: float = 0.5,  # Allow this much ATR drawdown before sparse mode kicks in
        # v18: Forced Minimum Hold Time
        min_hold_bars: int = 6,  # Must hold for N bars before manual exit is allowed
        # Full Eyes Features
        returns: Optional[np.ndarray] = None, # Recent 5m log-returns
        agent_lookback_window: int = 0, # How many return bars to see
    ):
        """
        Initialize the trading environment.

        Args:
            data_5m: 5-minute feature data [num_samples, lookback_5m, features] (base timeframe)
            data_15m: 15-minute feature data [num_samples, lookback_15m, features]
            data_45m: 45-minute feature data [num_samples, lookback_45m, features]
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
            noise_level: Std dev of Gaussian noise to add to observations (0.0 = disabled)
            returns: Raw log-returns series for Agent's peripheral vision
            agent_lookback_window: Number of return steps to observe
        """
        super().__init__()

        # Anti-Overfitting: Gaussian Noise
        self.noise_level = noise_level 
        if self.noise_level > 0:
            print(f"Gaussian Noise Injection ENABLED: sigma={self.noise_level}")
        self.data_5m = data_5m.astype(np.float32)
        self.data_15m = data_15m.astype(np.float32)
        self.data_45m = data_45m.astype(np.float32)
        self.close_prices = close_prices.astype(np.float32)
        self.market_features = market_features.astype(np.float32)
        
        # New "Full Eyes" data
        self.returns = returns.astype(np.float32) if returns is not None else None
        self.agent_lookback_window = agent_lookback_window
        
        # OHLC data for visualization (real candle data)
        self.ohlc_data = ohlc_data  # Shape: (n_samples, 4) = [open, high, low, close]
        self.timestamps = timestamps  # Unix timestamps for real time axis

        # Analyst model
        self.analyst = analyst_model
        self.device = device or torch.device('cpu')
        self.context_dim = context_dim

        # Lookback windows
        self.lookback_5m = lookback_5m
        self.lookback_15m = lookback_15m
        self.lookback_45m = lookback_45m

        # Trading parameters
        self.pip_value = pip_value  # NAS100: 1.0 (1 point = 1.0 price movement)
        self.lot_size = lot_size    # NAS100 CFD lot size (1.0)
        self.point_multiplier = point_multiplier  # Dollar per point per lot (1.0)
        self.spread_pips = spread_pips
        self.slippage_pips = slippage_pips  # Realistic execution slippage
        self.fomo_penalty = fomo_penalty
        self.chop_penalty = chop_penalty
        self.fomo_threshold_atr = fomo_threshold_atr
        self.chop_threshold = chop_threshold
        self.max_steps = max_steps
        self.reward_scaling = reward_scaling  # Scale PnL to balance with penalties
        self.trade_entry_bonus = trade_entry_bonus  # v15: Bonus for opening positions

        # Risk Management - Stop-Loss and Take-Profit
        self.sl_atr_multiplier = sl_atr_multiplier
        self.tp_atr_multiplier = tp_atr_multiplier
        self.use_stop_loss = use_stop_loss
        self.use_stop_loss = use_stop_loss
        self.use_take_profit = use_take_profit
        
        # Volatility Sizing (Dollar-based risk)
        self.volatility_sizing = volatility_sizing
        self.risk_per_trade = risk_per_trade  # Dollar risk per trade
        
        # Analyst Alignment
        self.enforce_analyst_alignment = enforce_analyst_alignment
        self.current_probs = None  # Store for action masking
        
        # v16: Sparse Rewards - only reward on trade exit
        self.use_sparse_rewards = use_sparse_rewards
        # v17: Loss Tolerance Buffer
        self.loss_tolerance_atr = loss_tolerance_atr
        # v18: Forced Minimum Hold Time
        self.min_hold_bars = min_hold_bars
        self.entry_idx = 0  # Track when position was opened

        # Calculate valid range FIRST (needed for regime indices)
        # FIXED: If pre_windowed=True, data is already trimmed by prepare_env_data
        # so start_idx should be 0 (no double offset)
        if pre_windowed:
            self.start_idx = 0
        else:
            # Only compute start_idx if using raw DataFrames (create_env_from_dataframes)
            # Subsample ratios: 15m = 3x base (5m), 45m = 9x base (5m)
            self.start_idx = max(lookback_5m, lookback_15m * 3, lookback_45m * 9)
        
        self.end_idx = len(close_prices) - 1
        self.n_samples = self.end_idx - self.start_idx
        
        # Regime-balanced sampling (AFTER start_idx/end_idx are set)
        self.use_regime_sampling = use_regime_sampling and regime_labels is not None
        if regime_labels is not None:
            self.regime_labels = regime_labels.astype(np.int32)
            # Pre-compute indices for each regime (0=Bullish, 1=Ranging, 2=Bearish)
            self.regime_indices = {
                0: np.where(self.regime_labels == 0)[0],  # Bullish
                1: np.where(self.regime_labels == 1)[0],  # Ranging
                2: np.where(self.regime_labels == 2)[0],  # Bearish
            }
            # Filter to valid range for episode starts
            max_start = max(self.start_idx + 1, self.end_idx - max_steps)
            for regime in self.regime_indices:
                valid = self.regime_indices[regime]
                valid = valid[(valid >= self.start_idx) & (valid < max_start)]
                self.regime_indices[regime] = valid
            # Log regime distribution
            print(f"Regime sampling enabled: Bullish={len(self.regime_indices[0])}, "
                  f"Ranging={len(self.regime_indices[1])}, Bearish={len(self.regime_indices[2])}")
        else:
            self.regime_labels = None
            self.regime_indices = None

        # Action space: Multi-Discrete([direction, size])
        # Direction: 0=Flat, 1=Long, 2=Short
        # Size: 0=0.25, 1=0.5, 2=0.75, 3=1.0
        self.action_space = spaces.MultiDiscrete([3, 4])

        # Store num_classes for observation construction
        self.num_classes = num_classes

        # Observation space
        # Context vector + position state (3) + market features (5) + analyst_metrics
        # Binary (2 classes): [p_down, p_up, edge, confidence, uncertainty] = 5
        # Multi-class (3 classes): [p_down, p_neutral, p_up, edge, confidence, uncertainty] = 6
        n_market_features = market_features.shape[1] if len(market_features.shape) > 1 else 5
        analyst_metrics_dim = 5 if num_classes == 2 else 6

        # Obs Dim = Context + Position(3) + Market + Analyst + SL/TP(2) + Returns
        obs_dim = context_dim + 3 + n_market_features + analyst_metrics_dim + 2 + self.agent_lookback_window

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # CRITICAL: Normalize market features to prevent scale inconsistencies!
        # Market features (ATR ~0.001, CHOP 0-100, ADX 0-100) have vastly different scales.
        # FIXED: Use pre-computed stats from training data to prevent look-ahead bias.
        if market_feat_mean is not None and market_feat_std is not None:
            # Use pre-computed statistics (no look-ahead bias)
            self.market_feat_mean = market_feat_mean.astype(np.float32)
            self.market_feat_std = market_feat_std.astype(np.float32)
        elif len(market_features.shape) > 1 and market_features.shape[1] > 0:
            # Fallback: compute from provided data (should only be used with training data)
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
        # Holding-bonus state (reset on every new trade)
        self._holding_bonus_paid = 0.0
        self._holding_bonus_level = 0

        # Precompute context vectors if analyst is provided
        self._precomputed_contexts = None
        self._precomputed_probs = None
        
        # Use pre-computed cache if provided (for sequential context)
        self._precomputed_activations = {}
        if precomputed_analyst_cache is not None:
            print("Using pre-computed Analyst cache (sequential context)")
            self._precomputed_contexts = precomputed_analyst_cache['contexts'].astype(np.float32)
            self._precomputed_probs = precomputed_analyst_cache['probs'].astype(np.float32)
            
            # Load activations if available
            if 'activations_15m' in precomputed_analyst_cache and precomputed_analyst_cache['activations_15m'] is not None:
                self._precomputed_activations['15m'] = precomputed_analyst_cache['activations_15m'].astype(np.float32)
            if 'activations_1h' in precomputed_analyst_cache and precomputed_analyst_cache['activations_1h'] is not None:
                self._precomputed_activations['1h'] = precomputed_analyst_cache['activations_1h'].astype(np.float32)
            if 'activations_4h' in precomputed_analyst_cache and precomputed_analyst_cache['activations_4h'] is not None:
                self._precomputed_activations['4h'] = precomputed_analyst_cache['activations_4h'].astype(np.float32)
                
            print(f"Loaded {len(self._precomputed_contexts)} cached context vectors")
        elif self.analyst is not None:
            self._precompute_contexts()

    def _precompute_contexts(self):
        """Precompute all context vectors for efficiency."""
        if self.analyst is None:
            return

        print("Precomputing context vectors...")
        self.analyst.eval()

        contexts = []
        probs_list = []
        batch_size = 64

        with torch.no_grad():
            for i in range(0, self.n_samples, batch_size):
                end_i = min(i + batch_size, self.n_samples)
                actual_indices = range(self.start_idx + i, self.start_idx + end_i)

                # Get batch data (base/mid/high)
                batch_5m = torch.tensor(
                    self.data_5m[list(actual_indices)],
                    device=self.device,
                    dtype=torch.float32
                )
                batch_15m = torch.tensor(
                    self.data_15m[list(actual_indices)],
                    device=self.device,
                    dtype=torch.float32
                )
                batch_45m = torch.tensor(
                    self.data_45m[list(actual_indices)],
                    device=self.device,
                    dtype=torch.float32
                )

                # Get context AND probabilities
                if hasattr(self.analyst, 'get_probabilities'):
                    res = self.analyst.get_probabilities(batch_5m, batch_15m, batch_45m)
                    
                    if len(res) == 3:
                         context, probs, weights = res
                    else:
                         context, probs = res
                         weights = None
                         
                    contexts.append(context.cpu().numpy())
                    probs_list.append(probs.cpu().numpy())
                else:
                    # Fallback for old models
                    context = self.analyst.get_context(batch_5m, batch_15m, batch_45m)
                    contexts.append(context.cpu().numpy())
                    # Default probs (neutral)
                    dummy_probs = np.zeros((len(context), 3), dtype=np.float32)
                    dummy_probs[:, 1] = 1.0 # All neutral
                    probs_list.append(dummy_probs)

                # Memory cleanup
                del batch_5m, batch_15m, batch_45m, context
                if i % (batch_size * 10) == 0:
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    gc.collect()

        self._precomputed_contexts = np.vstack(contexts).astype(np.float32)
        self._precomputed_probs = np.vstack(probs_list).astype(np.float32)
            
        print(f"Precomputed {len(self._precomputed_contexts)} context vectors and probabilities")

    def _get_analyst_data(self, idx: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[Dict[str, np.ndarray]]]:
        """Get context vector, probabilities, attention weights, and activations for current index."""
        if self._precomputed_contexts is not None and self._precomputed_probs is not None:
            # Use precomputed
            context_idx = idx - self.start_idx
            if 0 <= context_idx < len(self._precomputed_contexts):
                # Get activations if available
                activations = None
                if self._precomputed_activations:
                    activations = {
                        k: v[context_idx] for k, v in self._precomputed_activations.items()
                    }

                return (
                    self._precomputed_contexts[context_idx],
                    self._precomputed_probs[context_idx],
                    None,
                    activations
                )

        if self.analyst is not None:
            # Compute on-the-fly
            with torch.no_grad():
                x_5m = torch.tensor(
                    self.data_5m[idx:idx+1],
                    device=self.device,
                    dtype=torch.float32
                )
                x_15m = torch.tensor(
                    self.data_15m[idx:idx+1],
                    device=self.device,
                    dtype=torch.float32
                )
                x_45m = torch.tensor(
                    self.data_45m[idx:idx+1],
                    device=self.device,
                    dtype=torch.float32
                )
                
                if hasattr(self.analyst, 'get_activations'):
                    context, activations = self.analyst.get_activations(x_5m, x_15m, x_45m)
                    
                    # Convert activations to numpy
                    activations_np = {
                        k: v.cpu().numpy().flatten() for k, v in activations.items()
                    }
                    
                    # Get probs
                    if hasattr(self.analyst, 'get_probabilities'):
                        res = self.analyst.get_probabilities(x_5m, x_15m, x_45m)

                        if isinstance(res, (tuple, list)) and len(res) == 3:
                            _, probs, _ = res
                        else:
                            _, probs = res
                        probs = probs.cpu().numpy().flatten()
                    else:
                        probs = np.array([0.5, 0.5], dtype=np.float32)
                        
                    return context.cpu().numpy().flatten(), probs, None, activations_np
                
                elif hasattr(self.analyst, 'get_probabilities'):
                    # Check if get_probabilities returns 3 values (new version)
                    result = self.analyst.get_probabilities(x_5m, x_15m, x_45m)
                    if len(result) == 3:
                        context, probs, weights = result
                        weights = weights.cpu().numpy().flatten() if weights is not None else None
                    else:
                        context, probs = result
                        weights = None
                    return context.cpu().numpy().flatten(), probs.cpu().numpy().flatten(), weights, None
                else:
                    context = self.analyst.get_context(x_5m, x_15m, x_45m)
                    # Dummy probs - match num_classes
                    if self.num_classes == 2:
                        probs = np.array([0.5, 0.5], dtype=np.float32)  # Binary: neutral
                    else:
                        probs = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Multi: neutral
                    return context.cpu().numpy().flatten(), probs, None, None

        # No analyst - return zeros with correct probs size
        if self.num_classes == 2:
            dummy_probs = np.array([0.5, 0.5], dtype=np.float32)  # Binary: neutral
        else:
            dummy_probs = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Multi: neutral
        return (
            np.zeros(self.context_dim, dtype=np.float32),
            dummy_probs,
            None,
            None
        )

    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        # Context vector and probabilities
        # Get Analyst context
        context, probs, weights, activations = self._get_analyst_data(self.current_idx)
        self.current_probs = probs  # Store for action enforcement
        self.current_activations = activations # Store for info

        # Calculate Analyst metrics for observation
        # [p_down, p_up, confidence, edge]
        if len(probs) == 2 or self.num_classes == 2:
            p_down = probs[0]
            p_up = probs[1] if len(probs) > 1 else 1 - p_down
            confidence = max(p_down, p_up)
            edge = p_up - p_down
            uncertainty = 1.0 - confidence
            analyst_metrics = np.array([p_down, p_up, edge, confidence, uncertainty], dtype=np.float32)
        else:
            # Multi-class: [p_down, p_neutral, p_up]
            p_down = probs[0]
            p_neutral = probs[1]
            p_up = probs[2]
            confidence = np.max(probs) # Use np.max for multi-class confidence
            edge = p_up - p_down
            uncertainty = 1.0 - confidence
            analyst_metrics = np.array([p_down, p_neutral, p_up, edge, confidence, uncertainty], dtype=np.float32)

        # Position state
        current_price = self.close_prices[self.current_idx]
        atr = self.market_features[self.current_idx, 0] if len(self.market_features.shape) > 1 else 1.0

        # Normalize entry price and unrealized PnL
        # CRITICAL FIX: Use floor for ATR to prevent division by near-zero
        atr_safe = max(atr, 1e-6)
        if self.position != 0:
            # FIX: entry_price_norm should be POSITIVE when position is profitable
            # Previously was inverted for Long positions (winning Long = negative value)
            if self.position == 1:  # Long: positive when price goes UP (winning)
                entry_price_norm = (current_price - self.entry_price) / (atr_safe * 100)
            else:  # Short: positive when price goes DOWN (winning)
                entry_price_norm = (self.entry_price - current_price) / (atr_safe * 100)
            # Clip to prevent extreme values
            entry_price_norm = np.clip(entry_price_norm, -10.0, 10.0)
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

        # Combine all features
        # [context (32), position (3), market (5), analyst (5), sl_tp (2)]
        
        # SL/TP Blind Spot Fix: Calculate normalized distance to expected SL/TP levels
        dist_sl_norm = 0.0
        dist_tp_norm = 0.0
        
        if self.position != 0 and len(self.market_features.shape) > 1:
            atr = self.market_features[self.current_idx, 0]
            if atr > 1e-8:
                # Replicate logic from _check_stop_loss_take_profit
                sl_pips = max((atr * self.sl_atr_multiplier) / self.pip_value, 5.0)
                tp_pips = max((atr * self.tp_atr_multiplier) / self.pip_value, 5.0)
                pip_value = self.pip_value
                
                if self.position == 1: # Long
                    sl_price = self.entry_price - sl_pips * pip_value
                    tp_price = self.entry_price + tp_pips * pip_value
                    # Normalize by ATR (e.g. 1.0 = current price is 1 ATR away from level)
                    # For SL: Positive = Safe, Negative = Hit/Danger
                    dist_sl_norm = (current_price - sl_price) / atr
                    # For TP: Positive = Far, Negative = Hit/Close
                    dist_tp_norm = (tp_price - current_price) / atr
                else: # Short
                    sl_price = self.entry_price + sl_pips * pip_value
                    tp_price = self.entry_price - tp_pips * pip_value
                    # For Short: SL is ABOVE, Price is BELOW
                    dist_sl_norm = (sl_price - current_price) / atr
                    dist_tp_norm = (current_price - tp_price) / atr

        obs = np.concatenate([
            context,
            position_state,
            market_feat,
            analyst_metrics,
            np.array([dist_sl_norm, dist_tp_norm], dtype=np.float32)
        ])
        
        # Append "Full Eyes" features
        if self.agent_lookback_window > 0 and self.returns is not None:
            # Slice recent returns
            # Use current_idx + 1 because the 'returns' array aligns with close_prices
            # We want [t - lookback + 1 ... t]
            idx_start = self.current_idx - self.agent_lookback_window + 1
            idx_end = self.current_idx + 1
            if idx_start < 0:
                # Pad with zeros if we are at the very beginning (unlikely due to start_idx)
                returns_slice = np.zeros(self.agent_lookback_window, dtype=np.float32)
                valid_len = idx_end
                returns_slice[-valid_len:] = self.returns[0:idx_end]
            else:
                returns_slice = self.returns[idx_start:idx_end]
            
            # Multiply by 100 for normalization (returns are small floats)
            obs = np.concatenate([obs, returns_slice * 100])
                
        # Anti-Overfitting: Inject Gaussian Noise
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, size=obs.shape).astype(np.float32)
            obs += noise

        return obs.astype(np.float32)

    def _calculate_unrealized_pnl(self) -> float:
        """Calculate unrealized PnL in pips."""
        if self.position == 0:
            return 0.0

        current_price = self.close_prices[self.current_idx]
        pip_value = self.pip_value  # NAS100: 1.0 per point

        if self.position == 1:  # Long
            pnl_pips = (current_price - self.entry_price) / pip_value
        else:  # Short
            pnl_pips = (self.entry_price - current_price) / pip_value

        return pnl_pips * self.position_size

    def _check_stop_loss_take_profit(self) -> Tuple[float, dict]:
        """
        Check and execute stop-loss or take-profit if triggered.

        FIXED: Now uses High/Low to detect intra-bar SL/TP hits, not just Close.
        This creates more realistic training by penalizing the agent for positions
        that would have been stopped out by intra-bar wicks in real trading.

        This method is called BEFORE the agent's action to enforce risk management.
        Stop-loss cuts losing positions early to prevent catastrophic losses.
        Take-profit locks in gains to improve risk/reward ratio.

        Returns:
            Tuple of (reward, info_dict) if triggered, (0.0, {}) otherwise
        """
        # No position = nothing to check
        if self.position == 0:
            return 0.0, {}

        # Get current bar OHLC
        close_price = self.close_prices[self.current_idx]
        pip_value = self.pip_value  # NAS100: 1.0 per point

        # FIXED: Get High/Low for accurate intra-bar SL/TP detection
        if self.ohlc_data is not None:
            # ohlc_data shape: (n_samples, 4) = [open, high, low, close]
            high_price = float(self.ohlc_data[self.current_idx, 1])
            low_price = float(self.ohlc_data[self.current_idx, 2])
        else:
            # Fallback: use close price for all (legacy behavior)
            high_price = close_price
            low_price = close_price

        # Get ATR for dynamic thresholds
        if len(self.market_features.shape) > 1:
            atr = self.market_features[self.current_idx, 0]
        else:
            atr = 0.001  # Default fallback

        # Calculate dynamic thresholds in pips/points
        sl_pips_threshold = (atr * self.sl_atr_multiplier) / self.pip_value
        tp_pips_threshold = (atr * self.tp_atr_multiplier) / self.pip_value

        # Enforce minimums (e.g. 5 pips) to prevent noise stop-outs
        sl_pips_threshold = max(sl_pips_threshold, 5.0)
        tp_pips_threshold = max(tp_pips_threshold, 5.0)

        # Calculate SL/TP price levels
        if self.position == 1:  # Long
            sl_price = self.entry_price - sl_pips_threshold * pip_value
            tp_price = self.entry_price + tp_pips_threshold * pip_value

            # For Long: SL triggered if Low <= SL price, TP triggered if High >= TP price
            # IMPORTANT: Check SL first (worst case) to be conservative
            if self.use_stop_loss and low_price <= sl_price:
                # Exit at SL level (not at the low - we don't know exact fill)
                exit_price = sl_price
                pnl_pips = -sl_pips_threshold * self.position_size
                # FIX: Keep PnL in PIPS (same unit as _calculate_unrealized_pnl)
                # Previously multiplied by 10 (dollars) which broke reward calculation
                pnl = pnl_pips

                self.total_pnl += pnl
                self.trades.append({
                    'entry': self.entry_price,
                    'exit': exit_price,
                    'direction': self.position,
                    'size': self.position_size,
                    'pnl': pnl,
                    'close_reason': 'stop_loss'
                })

                # Calculate final delta before reset (both in PIPS now)
                final_delta = pnl - self.prev_unrealized_pnl

                # Reset
                self.position = 0
                self.position_size = 0.0
                self.entry_price = 0.0
                self.prev_unrealized_pnl = 0.0
                self._holding_bonus_paid = 0.0
                self._holding_bonus_level = 0

                sl_reward = final_delta * self.reward_scaling

                return sl_reward, {
                    'stop_loss_triggered': True,
                    'trade_closed': True,
                    'close_reason': 'stop_loss',
                    'pnl': pnl
                }

            if self.use_take_profit and high_price >= tp_price:
                # Exit at TP level
                exit_price = tp_price
                pnl_pips = tp_pips_threshold * self.position_size
                # FIX: Keep PnL in PIPS (same unit as _calculate_unrealized_pnl)
                pnl = pnl_pips

                self.total_pnl += pnl
                self.trades.append({
                    'entry': self.entry_price,
                    'exit': exit_price,
                    'direction': self.position,
                    'size': self.position_size,
                    'pnl': pnl,
                    'close_reason': 'take_profit'
                })

                # Calculate final delta before reset (both in PIPS now)
                final_delta = pnl - self.prev_unrealized_pnl

                self.position = 0
                self.position_size = 0.0
                self.entry_price = 0.0
                self.prev_unrealized_pnl = 0.0
                self._holding_bonus_paid = 0.0
                self._holding_bonus_level = 0

                tp_reward = final_delta * self.reward_scaling

                return tp_reward, {
                    'take_profit_triggered': True,
                    'trade_closed': True,
                    'close_reason': 'take_profit',
                    'pnl': pnl
                }

        else:  # Short (position == -1)
            sl_price = self.entry_price + sl_pips_threshold * pip_value
            tp_price = self.entry_price - tp_pips_threshold * pip_value

            # For Short: SL triggered if High >= SL price, TP triggered if Low <= TP price
            if self.use_stop_loss and high_price >= sl_price:
                # Exit at SL level
                exit_price = sl_price
                pnl_pips = -sl_pips_threshold * self.position_size
                # FIX: Keep PnL in PIPS (same unit as _calculate_unrealized_pnl)
                pnl = pnl_pips

                self.total_pnl += pnl
                self.trades.append({
                    'entry': self.entry_price,
                    'exit': exit_price,
                    'direction': self.position,
                    'size': self.position_size,
                    'pnl': pnl,
                    'close_reason': 'stop_loss'
                })

                # Calculate final delta before reset (both in PIPS now)
                final_delta = pnl - self.prev_unrealized_pnl

                self.position = 0
                self.position_size = 0.0
                self.entry_price = 0.0
                self.prev_unrealized_pnl = 0.0
                self._holding_bonus_paid = 0.0
                self._holding_bonus_level = 0

                sl_reward = final_delta * self.reward_scaling

                return sl_reward, {
                    'stop_loss_triggered': True,
                    'trade_closed': True,
                    'close_reason': 'stop_loss',
                    'pnl': pnl
                }

            if self.use_take_profit and low_price <= tp_price:
                # Exit at TP level
                exit_price = tp_price
                pnl_pips = tp_pips_threshold * self.position_size
                # FIX: Keep PnL in PIPS (same unit as _calculate_unrealized_pnl)
                pnl = pnl_pips

                self.total_pnl += pnl
                self.trades.append({
                    'entry': self.entry_price,
                    'exit': exit_price,
                    'direction': self.position,
                    'size': self.position_size,
                    'pnl': pnl,
                    'close_reason': 'take_profit'
                })

                # Calculate final delta before reset (both in PIPS now)
                final_delta = pnl - self.prev_unrealized_pnl

                self.position = 0
                self.position_size = 0.0
                self.entry_price = 0.0
                self.prev_unrealized_pnl = 0.0
                self._holding_bonus_paid = 0.0
                self._holding_bonus_level = 0

                tp_reward = final_delta * self.reward_scaling

                return tp_reward, {
                    'take_profit_triggered': True,
                    'trade_closed': True,
                    'close_reason': 'take_profit',
                    'pnl': pnl
                }

        return 0.0, {}

    def _execute_action(self, action: np.ndarray) -> Tuple[float, dict]:
        """
        Execute trading action and calculate reward.

        Returns:
            Tuple of (reward, info_dict)
        """
        direction = action[0]  # 0=Flat, 1=Long, 2=Short
        size_idx = action[1]   # 0-3
        
        # Enforce Analyst Alignment (Action Masking)
        if self.enforce_analyst_alignment and self.current_probs is not None:
            # Determine Analyst Direction
            # Binary: [p_down, p_up] -> 0=Down, 1=Up
            # Multi: [p_down, p_neutral, p_up] -> 0=Down, 1=Neutral, 2=Up
            
            analyst_dir = 0 # Default Flat
            
            if len(self.current_probs) == 2:
                # Binary: 0=Short, 1=Long (mapped to env: 2=Short, 1=Long)
                p_down, p_up = self.current_probs
                if p_up > 0.5:
                    analyst_dir = 1 # Long
                elif p_down > 0.5:
                    analyst_dir = 2 # Short
                # Else neutral/uncertain
                
            elif len(self.current_probs) == 3:
                # Multi: 0=Down, 1=Neutral, 2=Up
                p_down, p_neutral, p_up = self.current_probs
                max_idx = np.argmax(self.current_probs)
                if max_idx == 2: # Up
                    analyst_dir = 1 # Long
                elif max_idx == 0: # Down
                    analyst_dir = 2 # Short
                else:
                    analyst_dir = 0 # Flat
            
            # Check for violation
            # If Analyst is Long (1), Agent cannot be Short (2)
            # If Analyst is Short (2), Agent cannot be Long (1)
            # If Analyst is Flat (0), Agent must be Flat (0)
            
            violation = False
            if analyst_dir == 1 and direction == 2: # Analyst Long, Agent Short
                violation = True
            elif analyst_dir == 2 and direction == 1: # Analyst Short, Agent Long
                violation = True
            elif analyst_dir == 0 and direction != 0: # Analyst Flat, Agent Active
                violation = True
                
            if violation:
                # Force Flat Action
                direction = 0
                # Optional: Add small penalty? No, just prevent the action.
                # The agent will learn that this action does nothing.
        
        base_size = self.POSITION_SIZES[size_idx]
        
        # Dollar-Based Volatility Sizing: Maintain constant dollar risk per trade
        # Size = Risk($) / ($/pip × SL_pips)
        # This ensures each trade risks the same dollar amount regardless of volatility
        new_size = base_size
        
        if self.volatility_sizing and len(self.market_features.shape) > 1:
            atr = self.market_features[self.current_idx, 0]
            # Calculate SL distance in pips/points
            sl_pips = (atr * self.sl_atr_multiplier) / self.pip_value
            sl_pips = max(sl_pips, 5.0)  # Minimum 5 points
            
            # NAS100 dollar risk sizing:
            # Risk($) = size(lots) × sl_pips(points) × $/point
            # $/point per 1 lot = pip_value × lot_size × point_multiplier
            dollars_per_pip = self.pip_value * self.lot_size * self.point_multiplier
            risk_amount = self.risk_per_trade * base_size  # Scale risk by agent's size choice
            new_size = risk_amount / (dollars_per_pip * sl_pips)
            
            # Clip to reasonable limits to prevent extreme leverage
            new_size = np.clip(new_size, 0.1, 50.0)  # Max 50 lots

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
        pip_value = self.pip_value  # NAS100: 1.0 per point

        # Handle position changes
        # Reward structure: Pure continuous PnL delta rewards every step.
        # On close/flip: capture the final delta BEFORE resetting position so the last
        # price leg is not missed, without paying any extra "banked" exit reward.
        
        # v18: MINIMUM HOLD TIME CHECK
        # Block manual exits AND position flips before min_hold_bars have passed
        # This prevents scalping by forcing agent to hold trades longer
        # SL/TP are NOT affected - they are checked BEFORE this action handling
        if self.position != 0 and self.min_hold_bars > 0:
            bars_held = self.current_idx - self.entry_idx
            if bars_held < self.min_hold_bars:
                # Check if action would close or flip the position
                would_close_or_flip = (
                    direction == 0 or  # Flat/Exit
                    (self.position == 1 and direction == 2) or  # Long→Short flip
                    (self.position == -1 and direction == 1)    # Short→Long flip
                )
                if would_close_or_flip:
                    # BLOCK: Force agent to keep current position
                    direction = 1 if self.position == 1 else 2  # Keep Long/Short
                    info['exit_blocked'] = True
                    info['bars_held'] = bars_held
                    info['min_hold_bars'] = self.min_hold_bars
        
        if direction == 0:  # Flat/Exit
            if self.position != 0:
                # CRITICAL: Calculate final delta BEFORE resetting position
                # This captures the last price leg that would otherwise be missed
                final_unrealized = self._calculate_unrealized_pnl()
                final_delta = final_unrealized - self.prev_unrealized_pnl
                reward += final_delta * self.reward_scaling

                # REMOVED: Direction bonus was causing reward-PnL divergence
                # The bonus (+2.5 for ANY profitable trade) was 50x larger than
                # the PnL reward for tiny winners, teaching agent to make many
                # small trades to collect bonuses regardless of actual profitability.
                # PnL delta (above) is now the ONLY source of reward for exits.

                # Record trade statistics
                info['trade_closed'] = True
                info['pnl'] = final_unrealized  # Unscaled for tracking
                info['pnl_delta'] = final_delta
                self.total_pnl += final_unrealized
                self.trades.append({
                    'entry': self.entry_price,
                    'exit': current_price,
                    'direction': self.position,
                    'size': self.position_size,
                    'pnl': final_unrealized
                })

                # NOW reset position state
                self.position = 0
                self.position_size = 0.0
                self.entry_price = 0.0
                self.prev_unrealized_pnl = 0.0
                self._holding_bonus_paid = 0.0
                self._holding_bonus_level = 0

        elif direction == 1:  # Long
            if self.position == -1:  # Close short first
                # CRITICAL: Calculate final delta BEFORE resetting position
                final_unrealized = self._calculate_unrealized_pnl()
                final_delta = final_unrealized - self.prev_unrealized_pnl
                reward += final_delta * self.reward_scaling

                # REMOVED: Direction bonus (see comment in Flat/Exit case above)

                info['trade_closed'] = True
                info['pnl'] = final_unrealized
                info['pnl_delta'] = final_delta
                self.total_pnl += final_unrealized
                self.trades.append({
                    'entry': self.entry_price,
                    'exit': current_price,
                    'direction': -1,
                    'size': self.position_size,
                    'pnl': final_unrealized
                })

                # NOW reset position state before opening new one
                self.position = 0
                self.position_size = 0.0
                self.entry_price = 0.0
                self.prev_unrealized_pnl = 0.0
                self._holding_bonus_paid = 0.0
                self._holding_bonus_level = 0

            if self.position != 1:  # Open long
                self.position = 1
                self.position_size = new_size
                self.entry_price = current_price
                self.entry_idx = self.current_idx  # v18: Track entry bar for min hold time
                # Reset holding-bonus state for the new trade
                self._holding_bonus_paid = 0.0
                self._holding_bonus_level = 0
                # Total execution cost = spread + slippage (realistic modeling)
                exec_cost = (self.spread_pips + self.slippage_pips) * new_size
                reward -= exec_cost * self.reward_scaling
                # Include execution cost in total_pnl to match backtest accounting
                self.total_pnl -= exec_cost
                info['trade_opened'] = True

                # v15 FIX: Trade entry bonus to encourage exploration
                # Without this, every trade starts with negative reward (entry cost)
                # which teaches the agent that trading is bad before it can learn profitability
                reward += self.trade_entry_bonus
                info['trade_entry_bonus'] = self.trade_entry_bonus

        elif direction == 2:  # Short
            if self.position == 1:  # Close long first
                # CRITICAL: Calculate final delta BEFORE resetting position
                final_unrealized = self._calculate_unrealized_pnl()
                final_delta = final_unrealized - self.prev_unrealized_pnl
                reward += final_delta * self.reward_scaling

                # REMOVED: Direction bonus (see comment in Flat/Exit case above)

                info['trade_closed'] = True
                info['pnl'] = final_unrealized
                info['pnl_delta'] = final_delta
                self.total_pnl += final_unrealized
                self.trades.append({
                    'entry': self.entry_price,
                    'exit': current_price,
                    'direction': 1,
                    'size': self.position_size,
                    'pnl': final_unrealized
                })

                # NOW reset position state before opening new one
                self.position = 0
                self.position_size = 0.0
                self.entry_price = 0.0
                self.prev_unrealized_pnl = 0.0
                self._holding_bonus_paid = 0.0
                self._holding_bonus_level = 0

            if self.position != -1:  # Open short
                self.position = -1
                self.position_size = new_size
                self.entry_price = current_price
                self.entry_idx = self.current_idx  # v18: Track entry bar for min hold time
                # Reset holding-bonus state for the new trade
                self._holding_bonus_paid = 0.0
                self._holding_bonus_level = 0
                # Total execution cost = spread + slippage (realistic modeling)
                exec_cost = (self.spread_pips + self.slippage_pips) * new_size
                reward -= exec_cost * self.reward_scaling
                # Include execution cost in total_pnl to match backtest accounting
                self.total_pnl -= exec_cost
                info['trade_opened'] = True

                # v15 FIX: Trade entry bonus to encourage exploration
                # Without this, every trade starts with negative reward (entry cost)
                # which teaches the agent that trading is bad before it can learn profitability
                reward += self.trade_entry_bonus
                info['trade_entry_bonus'] = self.trade_entry_bonus

        # Continuous PnL feedback: reward based on CHANGE in unrealized PnL each step.
        # v16: DISABLED by default (use_sparse_rewards=True) because it causes scalping!
        # v17: LOSS TOLERANCE MODE - Allow normal retracements, only cut deep losers
        #      - Profitable trades: get per-bar reward (let winners run)
        #      - Small drawdown (< loss_tolerance_atr): still get per-bar reward (hold through noise)
        #      - Deep loss (> loss_tolerance_atr): NO per-bar feedback (encourage exit)
        #
        # NOTE: This block only runs for OPEN positions. On exit, the final_delta is
        # captured above BEFORE resetting position, and prev_unrealized_pnl is reset to 0.
        # Then this block sees position=0 and skips, preventing double-counting.
        if self.position != 0:
            current_unrealized_pnl = self._calculate_unrealized_pnl()
            pnl_delta = current_unrealized_pnl - self.prev_unrealized_pnl
            
            # Get current ATR for loss tolerance calculation
            if len(self.market_features.shape) > 1:
                current_atr = self.market_features[self.current_idx, 0]
            else:
                current_atr = 10.0  # Default ATR ~10 points for NAS100
            
            # Calculate loss tolerance in points (negative threshold)
            loss_tolerance_points = -(current_atr * self.loss_tolerance_atr * self.position_size)
            
            # v17: Give continuous rewards when trade is:
            # - Profitable (current_unrealized_pnl > 0), OR
            # - Within acceptable drawdown (current_unrealized_pnl > loss_tolerance_points)
            if not self.use_sparse_rewards:
                # Original mode: reward all PnL changes
                reward += pnl_delta * self.reward_scaling
            elif current_unrealized_pnl > loss_tolerance_points:
                # Sparse mode with loss tolerance: reward when profitable OR in acceptable drawdown
                reward += pnl_delta * self.reward_scaling
            # Else: deep loss beyond tolerance, no per-bar reward (encourage exit)

            # Holding bonus (gated + progress-based + capped):
            # Pay only when the trade achieves a NEW profit milestone beyond break-even.
            # This rewards "let winners run" without incentivizing time-based farming.
            # NOTE: Holding bonus is ALWAYS active (even in sparse mode) to encourage holds
            if current_unrealized_pnl > 0:
                entry_cost_points = (self.spread_pips + self.slippage_pips) * self.position_size
                buffer_points = entry_cost_points * float(self.HOLDING_BONUS_BUFFER_FRACTION_OF_ENTRY_COST)
                net_unrealized_pnl = current_unrealized_pnl - entry_cost_points

                # Gate on being net profitable beyond costs + buffer.
                if net_unrealized_pnl > buffer_points:
                    milestone_points = float(self.HOLDING_BONUS_MILESTONE_PIPS) * self.position_size
                    if milestone_points > 1e-8:
                        # Level 0: not eligible
                        # Level 1: just crossed buffer
                        # Level k: crossed buffer + (k-1)*milestone_points
                        level = int(np.floor((net_unrealized_pnl - buffer_points) / milestone_points)) + 1
                        if level > self._holding_bonus_level:
                            levels_gained = level - self._holding_bonus_level
                            desired_bonus = levels_gained * float(self.HOLDING_BONUS_AMOUNT)

                            entry_cost_reward = entry_cost_points * self.reward_scaling
                            cap_reward = entry_cost_reward * float(self.HOLDING_BONUS_CAP_FRACTION_OF_ENTRY_COST)
                            remaining = cap_reward - self._holding_bonus_paid

                            if remaining > 0:
                                paid_bonus = min(desired_bonus, remaining)
                                reward += paid_bonus
                                self._holding_bonus_paid += paid_bonus

                            # Always update level once the milestone is reached to avoid re-paying
                            # if the trade retraces and later re-crosses the same milestone.
                            self._holding_bonus_level = level
            info['unrealized_pnl'] = current_unrealized_pnl
            info['pnl_delta'] = pnl_delta
            self.prev_unrealized_pnl = current_unrealized_pnl
        else:
            # Position is flat - ensure tracking is reset
            self.prev_unrealized_pnl = 0.0
            self._holding_bonus_paid = 0.0
            self._holding_bonus_level = 0

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
        elif self.use_regime_sampling and self.regime_indices is not None:
            # REGIME-BALANCED SAMPLING: Equal probability for each regime
            # This prevents directional bias from unbalanced training data
            available_regimes = [r for r in [0, 1, 2] if len(self.regime_indices[r]) > 0]
            if len(available_regimes) > 0:
                # Randomly pick a regime
                chosen_regime = self.np_random.choice(available_regimes)
                # Randomly pick a starting index from that regime
                regime_idx = self.np_random.integers(0, len(self.regime_indices[chosen_regime]))
                self.current_idx = self.regime_indices[chosen_regime][regime_idx]
            else:
                # Fallback to random if no regime indices available
                max_start = max(self.start_idx + 1, self.end_idx - self.max_steps)
                self.current_idx = self.np_random.integers(self.start_idx, max_start)
        else:
            # FIXED: Ensure valid range for random start
            max_start = max(self.start_idx + 1, self.end_idx - self.max_steps)
            self.current_idx = self.np_random.integers(
                self.start_idx,
                max_start
            )

        # Reset state
        self.position = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.steps = 0
        self.total_pnl = 0.0
        self.trades = []
        self.prev_unrealized_pnl = 0.0  # Reset for continuous PnL tracking
        self._holding_bonus_paid = 0.0
        self._holding_bonus_level = 0

        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.

        Args:
            action: [direction, size] array

        Returns:
            observation, reward, terminated, truncated, info
        """
        # FIRST: Check stop-loss/take-profit BEFORE agent action
        # This enforces risk management regardless of what the agent wants to do
        sl_tp_reward, sl_tp_info = self._check_stop_loss_take_profit()

        # THEN: Execute agent's action (which may open new positions or do nothing)
        action_reward, action_info = self._execute_action(action)

        # Combine rewards and info
        reward = sl_tp_reward + action_reward
        info = {**action_info, **sl_tp_info}  # SL/TP info takes precedence

        # Move to next step
        self.current_idx += 1
        self.steps += 1

        # Check termination
        terminated = self.current_idx >= self.end_idx
        truncated = self.steps >= self.max_steps

        # EPISODE BOUNDARY FIX:
        # If the episode ends with an open position, force-close it so the final
        # mark-to-market PnL delta is realized and trade statistics are complete.
        if (terminated or truncated) and self.position != 0:
            exit_idx = min(self.current_idx, len(self.close_prices) - 1)
            exit_price = float(self.close_prices[exit_idx])
            pip_value = self.pip_value

            if self.position == 1:  # Long
                pnl_pips = (exit_price - self.entry_price) / pip_value
            else:  # Short
                pnl_pips = (self.entry_price - exit_price) / pip_value

            final_unrealized = pnl_pips * self.position_size
            final_delta = final_unrealized - self.prev_unrealized_pnl

            forced_close_reward = final_delta * self.reward_scaling
            reward += forced_close_reward

            self.total_pnl += final_unrealized
            self.trades.append({
                'entry': self.entry_price,
                'exit': exit_price,
                'direction': self.position,
                'size': self.position_size,
                'pnl': final_unrealized,
                'close_reason': 'episode_end'
            })

            # Reset position state
            self.position = 0
            self.position_size = 0.0
            self.entry_price = 0.0
            self.prev_unrealized_pnl = 0.0
            self._holding_bonus_paid = 0.0
            self._holding_bonus_level = 0

            info['episode_end_forced_close'] = True
            info['episode_end_forced_close_reward'] = forced_close_reward
            info['episode_end_forced_close_pnl'] = final_unrealized

        # Get new observation (guard against out-of-bounds access at episode end)
        if terminated or truncated:
            # Return dummy observation - episode is over, this won't be used for training
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._get_observation()

        # Add episode info
        info['step'] = self.steps
        info['position'] = self.position
        info['position_size'] = self.position_size
        info['entry_price'] = self.entry_price if self.position != 0 else None
        info['current_price'] = self.close_prices[min(self.current_idx, len(self.close_prices) - 1)]
        info['unrealized_pnl'] = self._calculate_unrealized_pnl()
        info['total_pnl'] = self.total_pnl
        info['n_trades'] = len(self.trades)

        # Market features for visualization
        if len(self.market_features.shape) > 1 and self.current_idx < len(self.market_features):
            mf = self.market_features[min(self.current_idx, len(self.market_features) - 1)]
            info['atr'] = float(mf[0]) if len(mf) > 0 else 0.0
            info['chop'] = float(mf[1]) if len(mf) > 1 else 50.0
            info['adx'] = float(mf[2]) if len(mf) > 2 else 25.0
            info['regime'] = int(mf[3]) if len(mf) > 3 else 1
            info['sma_distance'] = float(mf[4]) if len(mf) > 4 else 0.0

        # Analyst predictions
        if hasattr(self, 'current_probs') and self.current_probs is not None:
            info['p_down'] = float(self.current_probs[0])
            info['p_up'] = float(self.current_probs[-1])
            
        # Analyst activations (for visualization)
        if hasattr(self, 'current_activations') and self.current_activations is not None:
            info['analyst_activations'] = self.current_activations

        # Real OHLC data for visualization
        if self.ohlc_data is not None and self.current_idx < len(self.ohlc_data):
            ohlc = self.ohlc_data[self.current_idx]
            info['ohlc'] = {
                'open': float(ohlc[0]),
                'high': float(ohlc[1]),
                'low': float(ohlc[2]),
                'close': float(ohlc[3]),
            }
            if self.timestamps is not None and self.current_idx < len(self.timestamps):
                info['ohlc']['timestamp'] = int(self.timestamps[self.current_idx])

        # Pass trades list for win rate calculation (only on episode end to save memory)
        if terminated or truncated:
            info['trades'] = self.trades.copy()

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
    config: Optional[object] = None,
    device: Optional[torch.device] = None,
    noise_level: float = 0.001
) -> TradingEnv:
    """
    Factory function to create TradingEnv from DataFrames.
    
    FIXED: 1H and 4H data now correctly subsampled from the aligned 15m index.

    Args:
        df_15m: 15-minute DataFrame with features
        df_1h: 1-hour DataFrame with features (aligned to 15m index)
        df_4h: 4-hour DataFrame with features (aligned to 15m index)
        analyst_model: Trained Market Analyst
        feature_cols: Feature columns to use
        config: TradingConfig object
        device: Torch device for analyst inference

    Returns:
        TradingEnv instance
    """
    import pandas as pd

    if feature_cols is None:
        feature_cols = ['open', 'high', 'low', 'close', 'atr',
                       'pinbar', 'engulfing', 'doji', 'ema_trend', 'regime']

    # Get default config values (5m/15m/45m system)
    # We keep legacy names in the signature for compatibility, but these now mean:
    #   df_15m -> 5m base
    #   df_1h  -> 15m mid
    #   df_4h  -> 45m high
    lookback_5m = 48
    lookback_15m = 16
    lookback_45m = 6

    if config is not None:
        # Support both new and legacy config field names
        lookback_5m = getattr(config, 'lookback_5m', getattr(config, 'lookback_15m', lookback_5m))
        lookback_15m = getattr(config, 'lookback_15m', getattr(config, 'lookback_1h', lookback_15m))
        lookback_45m = getattr(config, 'lookback_45m', getattr(config, 'lookback_4h', lookback_45m))

    # Subsampling ratios: how many 5m bars per higher TF bar
    subsample_15m = 3   # 3 x 5m = 15m
    subsample_45m = 9   # 9 x 5m = 45m

    # Calculate valid range - need enough indices for subsampled lookback
    start_idx = max(lookback_5m, lookback_15m * subsample_15m, lookback_45m * subsample_45m)
    n_samples = len(df_15m) - start_idx

    # Get feature arrays
    features_15m = df_15m[feature_cols].values.astype(np.float32)
    features_1h = df_1h[feature_cols].values.astype(np.float32)
    features_4h = df_4h[feature_cols].values.astype(np.float32)

    # Create windows for each timeframe
    data_15m = np.zeros((n_samples, lookback_5m, len(feature_cols)), dtype=np.float32)
    data_1h = np.zeros((n_samples, lookback_15m, len(feature_cols)), dtype=np.float32)
    data_4h = np.zeros((n_samples, lookback_45m, len(feature_cols)), dtype=np.float32)

    for i in range(n_samples):
        actual_idx = start_idx + i
        # 15m: direct indexing (includes current candle)
        data_15m[i] = features_15m[actual_idx - lookback_5m + 1:actual_idx + 1]

        # 15m mid timeframe: subsample every 3rd bar from aligned data
        idx_range_1h = list(range(
            actual_idx - (lookback_15m - 1) * subsample_15m,
            actual_idx + 1,
            subsample_15m
        ))
        data_1h[i] = features_1h[idx_range_1h]

        # 45m high timeframe: subsample every 9th bar from aligned data
        idx_range_4h = list(range(
            actual_idx - (lookback_45m - 1) * subsample_45m,
            actual_idx + 1,
            subsample_45m
        ))
        data_4h[i] = features_4h[idx_range_4h]

    # Close prices for PnL
    close_prices = df_15m['close'].values[start_idx:start_idx + n_samples].astype(np.float32)

    # Real OHLC data for visualization
    ohlc_data = None
    timestamps = None
    if all(col in df_15m.columns for col in ['open', 'high', 'low', 'close']):
        ohlc_data = df_15m[['open', 'high', 'low', 'close']].values[start_idx:start_idx + n_samples].astype(np.float32)
    if df_15m.index.dtype == 'datetime64[ns]' or hasattr(df_15m.index, 'to_pydatetime'):
        try:
            timestamps = (df_15m.index[start_idx:start_idx + n_samples].astype('int64') // 10**9).values
        except:
            pass  # Keep timestamps as None if conversion fails

    # Market features for reward shaping (includes S/R for breakout vs chase detection)
    market_cols = ['atr', 'chop', 'adx', 'regime', 'sma_distance', 'dist_to_support', 'dist_to_resistance']
    available_cols = [c for c in market_cols if c in df_15m.columns]
    market_features = df_15m[available_cols].values[start_idx:start_idx + n_samples].astype(np.float32)

    # Extract config values (defaults for NAS100)
    pip_value = 1.0  # NAS100: 1 point = 1.0 price movement
    spread_pips = 2.5  # NAS100 typical spread (was 0.2 for EURUSD)
    fomo_penalty = -0.05  # Reduced from -0.5 (was dominating PnL rewards)
    chop_penalty = -0.01  # Reduced from -0.1
    fomo_threshold_atr = 2.0  # Only trigger on significant moves
    chop_threshold = 80.0  # Only extreme chop triggers penalty
    reward_scaling = 0.01  # NAS100: 1 reward per 100 points
    sl_atr_multiplier = 1.0
    tp_atr_multiplier = 3.0
    use_stop_loss = True
    use_take_profit = True
    volatility_sizing = True
    risk_per_trade = 100.0  # Dollar risk per trade
    enforce_analyst_alignment = False  # DISABLED: Soft masking breaks PPO gradients
    num_classes = 2

    if config is not None:
        pip_value = getattr(config, 'pip_value', pip_value)
        spread_pips = getattr(config, 'spread_pips', spread_pips)
        fomo_penalty = getattr(config, 'fomo_penalty', fomo_penalty)
        chop_penalty = getattr(config, 'chop_penalty', chop_penalty)
        fomo_threshold_atr = getattr(config, 'fomo_threshold_atr', fomo_threshold_atr)
        chop_threshold = getattr(config, 'chop_threshold', chop_threshold)
        reward_scaling = getattr(config, 'reward_scaling', reward_scaling)
        sl_atr_multiplier = getattr(config, 'sl_atr_multiplier', sl_atr_multiplier)
        tp_atr_multiplier = getattr(config, 'tp_atr_multiplier', tp_atr_multiplier)
        use_stop_loss = getattr(config, 'use_stop_loss', use_stop_loss)
        use_take_profit = getattr(config, 'use_take_profit', use_take_profit)
        volatility_sizing = getattr(config, 'volatility_sizing', volatility_sizing)
        risk_per_trade = getattr(config, 'risk_per_trade', risk_per_trade)
        enforce_analyst_alignment = getattr(config, 'enforce_analyst_alignment', enforce_analyst_alignment)
        noise_level = getattr(config, 'noise_level', noise_level)

    if analyst_model is not None:
        num_classes = getattr(analyst_model, 'num_classes', 2)

    return TradingEnv(
        # Map legacy arg names to TradingEnv signature
        data_5m=data_15m,
        data_15m=data_1h,
        data_45m=data_4h,
        close_prices=close_prices,
        market_features=market_features,
        analyst_model=analyst_model,
        lookback_5m=lookback_5m,
        lookback_15m=lookback_15m,
        lookback_45m=lookback_45m,
        device=device,
        # Config Params (NAS100)
        pip_value=pip_value,
        spread_pips=spread_pips,
        fomo_penalty=fomo_penalty,
        chop_penalty=chop_penalty,
        fomo_threshold_atr=fomo_threshold_atr,
        chop_threshold=chop_threshold,
        reward_scaling=reward_scaling,
        sl_atr_multiplier=sl_atr_multiplier,
        tp_atr_multiplier=tp_atr_multiplier,
        use_stop_loss=use_stop_loss,
        use_take_profit=use_take_profit,
        volatility_sizing=volatility_sizing,
        risk_per_trade=risk_per_trade,
        enforce_analyst_alignment=enforce_analyst_alignment,
        num_classes=num_classes,
        # Visualization data
        ohlc_data=ohlc_data,
        timestamps=timestamps,
        noise_level=noise_level
    )
