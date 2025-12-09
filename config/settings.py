"""
Configuration settings for the Hybrid EURUSD Trading System.

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
        default_factory=lambda: Path("/Users/gervaciusjr/Desktop/Market data/Data")
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

    def ensure_dirs(self):
        """Create all necessary directories."""
        for path in [self.data_raw, self.data_processed,
                     self.models_analyst, self.models_agent]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Data processing configuration."""

    # Data file names
    raw_file: str = "eurusd_m1_20220101_20251130.csv"
    processed_file: str = "eurusd_processed.parquet"
    datetime_format: str = "ISO8601"  # Auto-detect or ISO format

    # Timeframes (use lowercase 'h' for pandas 2.0+ compatibility)
    timeframes: Dict[str, str] = field(default_factory=lambda: {
        '15m': '15min',
        '1h': '1h',
        '4h': '4h'
    })

    # Lookback windows (number of candles)
    # v9 FIX: INCREASED lookbacks to provide proper context WITHOUT overlapping prediction window
    # Rule: lookback > prediction horizon to avoid temporal confusion
    lookback_windows: Dict[str, int] = field(default_factory=lambda: {
        '15m': 48,   # 12 hours - 6x prediction horizon (proper context)
        '1h': 16,    # 16 hours - captures full trading session
        '4h': 6      # 24 hours - captures daily trend
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
    architecture: str = "tcn"   # "transformer" or "tcn" (TCN recommended)

    # Shared architecture settings
    d_model: int = 32           # Hidden dimension (Reduced to 32)
    nhead: int = 4              # Transformer only: attention heads
    num_layers: int = 2         # Transformer only: encoder layers
    dim_feedforward: int = 128  # Transformer only: FFN hidden dim (4x d_model)
    dropout: float = 0.3        # v14: Standard regularization for small model
    context_dim: int = 32       # Output context vector dimension (Matched to d_model)

    # TCN-specific settings (v13)
    tcn_num_blocks: int = 3     # Number of residual blocks (dilations: 1, 2, 4)
    tcn_kernel_size: int = 3    # Convolution kernel size

    batch_size: int = 128       # Keep at 128
    learning_rate: float = 1e-4 # REDUCED from 3e-4 - more stable convergence
    weight_decay: float = 1e-4  # FIXED: Was 1e-2 (100x too high!) - standard value
    max_epochs: int = 100
    patience: int = 50  # Increased from 15 - more time to find recall balance

    cache_clear_interval: int = 50

    # v9 FIX: TARGET DEFINITION - reduced horizon and smoothing for more predictable signal
    future_window: int = 16      # 2 Hours (was 4H) - shorter = more predictable
    smooth_window: int = 8      # 1 Hour (was 3H) - less smoothing = preserves signal

    # Binary classification mode
    num_classes: int = 2        # Binary: 0=Down, 1=Up
    use_binary_target: bool = True  # Use binary direction target
    min_move_atr_threshold: float = 7  # v9 FIX: Was 0.3 - lower = 4x more training data

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
    lookback_15m: int = 48      # 12 Hours - 6x prediction horizon (proper context)
    lookback_1h: int = 16       # 16 Hours - captures full trading session
    lookback_4h: int = 6        # 24 Hours - captures daily trend


@dataclass
@dataclass
class TradingConfig:
    """Trading environment configuration."""
    spread_pips: float = 0.2    # Razor/Raw spread
    slippage_pips: float = 0.5  # Includes commission + slippage
    
    # NEW: Enforce Analyst Alignment (Action Masking)
    # If True, Agent can ONLY trade in direction of Analyst (or Flat)
    # DISABLED: Soft masking breaks PPO gradients - agent samples action X, gets
    # masked to Flat, but PPO updates as if X led to the reward. This causes
    # frozen action distributions and no learning.
    enforce_analyst_alignment: bool = False
    
    # NEW: Risk-Based Sizing (Not Fixed Lots)
    risk_multipliers: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    
    # NEW: ATR-Based Stops (Not Fixed Pips)
    sl_atr_multiplier: float = 1.0
    tp_atr_multiplier: float = 3.0
    
    # Risk Limits
    max_position_size: float = 5.0
    
    # Reward Params
    # FIX v15: Previous reward_scaling (0.1) was too weak - trades didn't produce
    # meaningful learning signal. FOMO penalty (-0.05) was too small to matter.
    # New values encourage exploration while still being proportional to PnL.
    fomo_penalty: float = -1.0    # Meaningful penalty for missing moves (was -0.05)
    chop_penalty: float = 0.0     # Disabled
    fomo_threshold_atr: float = 1.5  # Trigger on >1.5x ATR moves (was 2.0)
    chop_threshold: float = 80.0     # Only extreme chop triggers penalty
    reward_scaling: float = 1.0     # 1.0 per 1 pip (was 0.1 = 1.0 per 10 pips)

    # Trade entry bonus: Offsets entry cost to encourage exploration
    # Without this, every trade starts negative (spread + slippage) discouraging action
    trade_entry_bonus: float = 0.5  # Bonus for opening a position
    
    # These are mostly unused now but keep for compatibility if needed
    use_stop_loss: bool = True
    use_take_profit: bool = True
    
    # Environment settings
    max_steps_per_episode: int = 500    # Reduced to ~1 week (was 2000) for rapid regime cycling
    initial_balance: float = 10000.0
    
    # Validation
    noise_level: float = 0.05  # Reduced to 2% to encourage more activity (was 5%)


@dataclass
class AgentConfig:
    """PPO Sniper Agent configuration."""

    # PPO hyperparameters (from CLAUDE.md spec)
    # FIX v15: Previous ent_coef (0.01) caused rapid policy collapse to flat.
    # For a 12-action discrete space, higher entropy is needed to maintain exploration.
    learning_rate: float = 3e-4  # Higher initial LR for faster learning (was 1e-4)
    n_steps: int = 2048         # Timesteps per update
    batch_size: int = 256       # Minibatch size
    n_epochs: int = 10          # Reduced to prevent overfitting (was 20)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.1        # HIGH entropy to force exploration (was 0.01)
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Training
    total_timesteps: int = 20_000_000

    # Policy network
    # FIX v15: [64, 64] may bottleneck for 49-dim input with 12-action output
    policy_type: str = "MlpPolicy"
    net_arch: List[int] = field(default_factory=lambda: [256, 256])


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""

    # Price action patterns
    pinbar_wick_ratio: float = 2.0    # Wick must be > 2x body
    doji_body_ratio: float = 0.1      # Body < 10% of range

    # Market structure
    fractal_window: int = 5           # Williams fractal window
    sr_lookback: int = 100            # S/R level lookback

    # Trend indicators
    sma_period: int = 200
    ema_fast: int = 12
    ema_slow: int = 26

    # Regime indicators
    chop_period: int = 14
    adx_period: int = 14
    atr_period: int = 14


@dataclass
class VisualizationConfig:
    """Real-time training visualization configuration."""

    # Enable/disable visualization
    enabled: bool = True

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # Update rates
    update_hz: int = 10  # WebSocket updates per second

    # Buffer sizes
    max_snapshots: int = 10000
    max_price_bars: int = 500
    max_trades: int = 100

    # Frontend URL (for CORS)
    frontend_url: str = "http://localhost:3000"


@dataclass
class Config:
    """Master configuration combining all sub-configs."""

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    analyst: AnalystConfig = field(default_factory=AnalystConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

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
