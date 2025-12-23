"""
Configuration settings for the Hybrid NAS100 Trading System.

This module contains all hyperparameters, paths, and constants.
Optimized for Apple M2 Silicon with 8GB RAM constraints.
"""

import torch
import gc
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from pathlib import Path


def get_device() -> torch.device:
    """Get the optimal device for computation."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clear_memory():
    """Clear GPU/MPS memory cache and run garbage collection."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


@dataclass
class PathConfig:
    """Path configurations for data and model storage."""

    # Base directory (relative to project root)
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    # Training data directory (external location)
    training_data_dir: Path = field(
        default_factory=lambda: Path("/Users/gervaciusjr/Desktop/Oanda data")
    )

    @property
    def data_raw(self) -> Path:
        return self.training_data_dir

    @property
    def data_processed(self) -> Path:
        return self.base_dir / "data" / "processed"

    @property
    def models_analyst(self) -> Path:
        return self.base_dir / "models" / "analyst"

    @property
    def models_agent(self) -> Path:
        return self.base_dir / "models" / "agent"

    @property
    def models_agent_recurrent(self) -> Path:
        """RecurrentPPO model directory for parallel experiments."""
        return self.base_dir / "models" / "agent_recurrent"

    def ensure_dirs(self):
        """Create all necessary directories."""
        for path in [self.data_raw, self.data_processed,
                     self.models_analyst, self.models_agent,
                     self.models_agent_recurrent]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Data processing configuration."""

    # Data file names
    raw_file: str = "NAS100_USD_1min_data.csv"
    processed_file: str = "nas100_processed.parquet"
    datetime_format: str = "ISO8601"  # Auto-detect or ISO format

    # Timeframes (use lowercase 'h' for pandas 2.0+ compatibility)
    # UPDATED: Changed from 15m/1h/4h to 5m/15m/45m for faster trading analysis
    timeframes: Dict[str, str] = field(default_factory=lambda: {
        '5m': '5min',
        '15m': '15min',
        '45m': '45min'
    })

    # Lookback windows (number of candles)
    # v9 FIX: INCREASED lookbacks to provide proper context WITHOUT overlapping prediction window
    # Rule: lookback > prediction horizon to avoid temporal confusion
    # Subsample ratios: 15m = 3x base (5m), 45m = 9x base (5m)
    lookback_windows: Dict[str, int] = field(default_factory=lambda: {
        '5m': 48,    # 4 Hours - 2x prediction horizon (proper context)
        '15m': 16,   # 4 Hours - captures trading session
        '45m': 6     # 4.5 Hours - captures trend
    })

    # Train/validation/test splits
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Memory-efficient chunk size for processing
    chunk_size: int = 100_000


@dataclass
class AnalystConfig:
    """Market Analyst configuration (supports both Transformer and TCN)."""
    # v13: Architecture selection - TCN is more stable for binary classification
    architecture: str = "tcn"   # "transformer" or "tcn"

    # Shared architecture settings
    d_model: int = 32           # Hidden dimension (reduced for smaller Analyst)
    nhead: int = 4              # Transformer only: attention heads (64/4 = 16 dim/head)
    num_layers: int = 2         # Transformer only: encoder layers
    dim_feedforward: int = 128  # Transformer only: FFN hidden dim (4x d_model)
    dropout: float = 0.3        # v14: Standard regularization for small model
    context_dim: int = 32       # Output context vector dimension (Matched to d_model)

    # Input noise regularization (training only)
    # Adds small Gaussian noise to input features during Analyst training to reduce overfitting.
    # 0.0 disables. Typical useful range: 0.002–0.01 on normalized features.
    input_noise_std: float = 0.0

    # TCN-specific settings (v13)
    tcn_num_blocks: int = 4     # Number of residual blocks (dilations: 1, 2, 4, 8) to cover full 48-bar lookback
    tcn_kernel_size: int = 3    # Convolution kernel size

    batch_size: int = 128       # Keep at 128
    learning_rate: float = 1e-4 # REDUCED from 3e-4 - more stable convergence
    weight_decay: float = 1e-4  # FIXED: Was 1e-2 (100x too high!) - standard value
    max_epochs: int = 100
    patience: int = 20  # Increased from 15 - more time to find recall balance

    cache_clear_interval: int = 50

    # v9 FIX: TARGET DEFINITION - reduced horizon and smoothing for more predictable signal
    future_window: int = 24      # 2 Hours (24 * 5m) - shorter = more predictable
    smooth_window: int = 12      # 1 Hour (12 * 5m) - less smoothing = preserves signal

    # Binary classification mode
    num_classes: int = 2        # Binary: 0=Down, 1=Up
    use_binary_target: bool = True  # Use binary direction target
    min_move_atr_threshold: float = 7.0  # v9 FIX: Was 0.3 - lower = 4x more training data

    # Auxiliary losses (multi-task learning)
    # v14: RE-ENABLED - easier tasks (regime ~70%, volatility ~65%) provide
    # stronger gradients to shared encoder, reducing overfitting and improving direction
    use_auxiliary_losses: bool = True   # v14: Enabled for multi-task learning
    aux_volatility_weight: float = 0.2  # v14: Volatility prediction (MSE)
    aux_regime_weight: float = 0.4      # v14: INCREASED - Regime is easier, stronger gradients

    # Gradient accumulation for smoother updates (effective batch = batch_size * steps)
    gradient_accumulation_steps: int = 2  # Effective batch size = 128 * 2 = 256

    # Multi-horizon prediction (addresses Target Mismatch)
    # DISABLED: Gradient conflicts between horizons caused recall oscillation
    use_multi_horizon: bool = False  # Was True - disabled to focus on single target
    multi_horizon_weights: Dict[str, float] = field(default_factory=lambda: {
        '1h': 0.0,   # 1-hour horizon weight - disabled
        '2h': 0.0,   # 2-hour horizon weight - disabled
        '4h': 1.0    # 4-hour horizon weight (primary target)
    })

    # Legacy 3-class config (kept for compatibility)
    class_std_thresholds: Tuple[float, float] = (-0.15, 0.15)

    # Input Lookback Windows (Must match DataConfig)
    # v9 FIX: INCREASED lookbacks to provide proper context WITHOUT overlapping prediction window
    # UPDATED: Changed from 15m/1h/4h to 5m/15m/45m
    lookback_5m: int = 48       # 4 Hours - 6x prediction horizon (proper context)
    lookback_15m: int = 16      # 4 Hours - captures trading session
    lookback_45m: int = 6       # 4.5 Hours - captures trend


@dataclass
class InstrumentConfig:
    """Instrument-specific parameters for NAS100 (Oanda CFD)."""
    name: str = "NAS100"
    pip_value: float = 1.0           # 1 point = 1.0 price movement (NOT 0.1 tick size)
    lot_size: float = 1.0            # CFD lot ($1 per point per lot)
    point_multiplier: float = 1.0    # PnL: points × pip_value × lot_size × multiplier = $1/point
    min_body_points: float = 2.0     # Pattern detection: min body size (2 points)
    min_range_points: float = 5.0    # Pattern detection: min range size (5 points)


@dataclass
class TradingConfig:
    """Trading environment configuration for NAS100."""
    # Toggle Market Analyst usage
    # If False, agent trains with only raw market features (no analyst context/metrics)
    use_analyst: bool = True  # v23: Re-enabled for directional context

    spread_pips: float = 3.5    # NAS100 spread with buffer for realistic execution
    slippage_pips: float = 0.0  # NAS100 slippage

    # Confidence filtering: Only take trades when agent probability >= threshold
    min_action_confidence: float = 0.0  # Filter low-confidence trades (0.0 = disabled)

    # NEW: Enforce Analyst Alignment (Action Masking)
    # If True, Agent can ONLY trade in direction of Analyst (or Flat)
    # When enabled, if agent tries to trade against analyst prediction, action is forced to Flat
    # This constrains the agent to follow the Analyst's directional conviction
    enforce_analyst_alignment: bool = True  # ENABLED - Agent must align with Analyst direction
    
    # NEW: Risk-Based Sizing (Not Fixed Lots)
    risk_multipliers: Tuple[float, ...] = (1.5, 2.0, 2.5, 3.0)
    
    # NEW: ATR-Based Stops (Not Fixed Pips)
    # v23.5 FIX: Changed from 2.0/12.0 (1:6 inverted R/R) to 3.0/6.0 (1:2 standard R/R)
    # Training logs showed 54% win rate but negative PnL because losses were 6x larger than wins.
    # With 1:2 R/R, only need 33% win rate to break even. Current 54% should be profitable.
    sl_atr_multiplier: float = 3.0   # v23.5: Give trades room through normal volatility
    tp_atr_multiplier: float = 6.0   # v23.5: Achievable within NAS100 daily range (2x SL distance)
    
    # Risk Limits
    max_position_size: float = 5.0
    
    # Dollar-Based Position Sizing
    risk_per_trade: float = 100.0  # Dollar risk per trade (e.g., $100 per trade)
    
    # Reward Params (calibrated for NAS100)
    # NAS100 has ~100-200 point daily range vs EURUSD ~50-100 pip range
    # reward_scaling = 0.01 means 100 points = 1.0 reward (similar magnitude to EURUSD)
    fomo_penalty: float = 0.0    # Moderate penalty for missing high-momentum moves
    chop_penalty: float = 0.0     # Disabled (can cause over-penalization in legitimate ranging trades)
    fomo_threshold_atr: float = 4.0  # Trigger on >4x ATR moves
    chop_threshold: float = 80.0     # Only extreme chop triggers penalty
    reward_scaling: float = 0.01    # 1.0 reward per 100 points (NAS100 calibration)

    # Trade entry bonus: Offsets entry cost to encourage exploration
    # NAS100 spread ~2.5 points × 0.01 = 0.025 reward cost, so bonus = 0.01
    trade_entry_bonus: float = 0.01  # v23.5: Reduced to offset ~40% of spread cost
    
    # v21: Progressive Rewards Mode (replaces v16 sparse mode)
    # Now using asymmetric rewards: reward profit increases, tolerate pullbacks
    # - Profit increase → positive reward
    # - Minor pullback (within loss_tolerance_atr) → ZERO reward (no penalty!)
    # - Deep drawdown (beyond loss_tolerance_atr) → negative reward
    use_sparse_rewards: bool = False  # Disabled - using progressive reward system
    
    # v17: Loss Tolerance Buffer (used with sparse_rewards=True)
    # Allow some drawdown before stopping per-bar rewards
    # This lets agent hold through normal retracements without panic exits
    # Set to 0.5x ATR = if trade is down < 0.5x ATR, still get per-bar feedback
    loss_tolerance_atr: float = 1.5  # Allow 1.5x ATR drawdown before sparse mode kicks in
    
    # v18: Forced Minimum Hold Time
    # WARNING: min_hold_bars > 0 BREAKS PPO! When action is blocked, the sampled action
    # differs from executed action, causing wrong gradient updates.
    # v23.2 FIX: MUST be 0. Use reward shaping (not action forcing) to discourage scalping.
    min_hold_bars: int = 0  # v23.2: Disabled - action forcing breaks PPO gradients
    early_exit_profit_atr: float = 3.0  # Allow early exit if profit > 3x ATR (overrides min_hold_bars)
    break_even_atr: float = 2.0  # Move SL to break-even when profit reaches 2x ATR
    
    # These are mostly unused now but keep for compatibility if needed
    use_stop_loss: bool = True
    use_take_profit: bool = True
    
    # Environment settings
    max_steps_per_episode: int = 1500   # Increased for min_hold=12 (~125 trades/episode)
    initial_balance: float = 10000.0
    
    # Validation
    noise_level: float = 0.01  # Reduced to 2% to encourage more activity (was 5%)

    # NEW: "Full Eyes" Agent Features
    agent_lookback_window: int = 12   # Increased to 12 as requested (60 mins of 5m bars)
    include_structure_features: bool = True  # Agent sees BOS/CHoCH


@dataclass
class RecurrentAgentConfig:
    """RecurrentPPO (LSTM-based) Agent configuration - EXPERIMENTAL.

    Uses sb3-contrib RecurrentPPO with MlpLstmPolicy for temporal pattern learning.
    This allows the agent to build internal memory state across timesteps.
    """
    # Enable/disable recurrent mode (default: False = use standard PPO)
    use_recurrent: bool = False  # Disabled - using standard PPO (no LSTM, no analyst)

    # LSTM-specific hyperparameters
    lstm_hidden_size: int = 128       # Reduced from default 256 for M2 8GB
    n_lstm_layers: int = 1            # Single layer for efficiency
    shared_lstm: bool = False         # Separate actor/critic LSTMs (more expressive)
    enable_critic_lstm: bool = True   # LSTM for value function too

    # RecurrentPPO typically needs smaller batch sizes due to sequence handling
    # These override AgentConfig values when use_recurrent=True
    n_steps: int = 512                # Smaller rollout (sequence memory overhead)
    batch_size: int = 64              # Smaller batch for memory efficiency

    # Network architecture (feeds into LSTM)
    net_arch_pi: List[int] = field(default_factory=lambda: [128])  # Policy pre-LSTM
    net_arch_vf: List[int] = field(default_factory=lambda: [128])  # Value pre-LSTM


@dataclass
class AgentConfig:
    """PPO Sniper Agent configuration."""

    # PPO hyperparameters (v23: Fixed for continuous PnL rewards)
    # Previous config had broken minibatch ratio (8192/1024=8 minibatches) which
    # caused gradient noise with sparse rewards. Now using standard PPO setup.
    learning_rate: float = 1e-4  # v23: Reduced from 2e-4 for stability
    n_steps: int = 2048         # v23: Faster feedback (was 8192)
    batch_size: int = 256       # v23: 2048/256 = 8 minibatches (was 1024)
    n_epochs: int = 4           # v23: Reduced from 10 to prevent overfitting
    # v23: Standard discount factor for trading (trade-level rewards, not episode-end)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.05        # Initial value (decays to 0.001 via EntropyScheduleCallback)
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Training
    total_timesteps: int = 1_000_000_000  # 1B steps (Increased back to 1B)

    # Policy network
    # FIX v15: [64, 64] may bottleneck for 49-dim input with 12-action output
    policy_type: str = "MlpPolicy"
    net_arch: List[int] = field(default_factory=lambda: [256, 256])


@dataclass
class FeatureConfig:
    """Feature engineering configuration for NAS100."""

    # Price action patterns (thresholds in price units, not pips)
    # NAS100: 1 point = 0.1 price units, so 2.0 = 20 points, 5.0 = 50 points
    pinbar_wick_ratio: float = 2.0    # Wick must be > 2x body
    doji_body_ratio: float = 0.1      # Body < 10% of range
    min_body_points: float = 2.0      # Min body size in points (NAS100)
    min_range_points: float = 5.0     # Min candle range in points (NAS100)

    # Market structure
    fractal_window: int = 5           # Williams fractal window
    sr_lookback: int = 100            # S/R level lookback

    # Trend indicators
    sma_period: int = 50
    ema_fast: int = 12
    ema_slow: int = 26

    # Regime indicators
    chop_period: int = 14
    adx_period: int = 14
    atr_period: int = 14

    # Volatility sizing reference (NAS100 calibration)
    risk_pips_target: float = 50.0    # Reference risk ~50 points (was 15 for EURUSD)


@dataclass
class Config:
    """Master configuration combining all sub-configs."""

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    analyst: AnalystConfig = field(default_factory=AnalystConfig)
    instrument: InstrumentConfig = field(default_factory=InstrumentConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    recurrent_agent: RecurrentAgentConfig = field(default_factory=RecurrentAgentConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)

    # Global settings
    seed: int = 42
    dtype: torch.dtype = torch.float32  # NEVER use float64 on M2
    device: torch.device = field(default_factory=get_device)

    def __post_init__(self):
        """Ensure directories exist and set random seeds."""
        self.paths.ensure_dirs()
        torch.manual_seed(self.seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(self.seed)


# Global configuration instance
config = Config()
