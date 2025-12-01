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
        default_factory=lambda: Path("/Users/gervaciusjr/Desktop/AI Trading Bot/Training data")
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

    # Input data
    raw_file: str = "eurusd_m1_5y_part2_no_gaps.csv"
    datetime_format: str = "%Y-%m-%d %H:%M:%S"

    # Timeframes (use lowercase 'h' for pandas 2.0+ compatibility)
    timeframes: Dict[str, str] = field(default_factory=lambda: {
        '15m': '15min',
        '1h': '1h',
        '4h': '4h'
    })

    # Lookback windows (number of candles)
    lookback_windows: Dict[str, int] = field(default_factory=lambda: {
        '15m': 48,   # 12 hours of 15m candles
        '1h': 24,    # 24 hours of 1H candles
        '4h': 12     # 48 hours of 4H candles
    })

    # Train/validation/test splits
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Memory-efficient chunk size for processing
    chunk_size: int = 100_000


@dataclass
class AnalystConfig:
    """Market Analyst (Transformer) configuration."""

    # Architecture (memory-optimized for M2)
    d_model: int = 64           # Model dimension
    nhead: int = 4              # Attention heads
    num_layers: int = 2         # Transformer layers
    dim_feedforward: int = 128  # FFN dimension
    dropout: float = 0.1
    context_dim: int = 64       # Output context vector dimension

    # Training
    batch_size: int = 64        # Increased from 32 to reduce gradient noise
    learning_rate: float = 1e-3  # Increased from 1e-4 to escape mode collapse
    weight_decay: float = 1e-5
    max_epochs: int = 100
    patience: int = 10          # Early stopping patience

    # Memory management
    cache_clear_interval: int = 50  # Clear cache every N batches

    # Target computation
    future_window: int = 12     # Candles ahead for smoothed return
    smooth_window: int = 12     # Rolling window for smoothing
    num_classes: int = 3        # Simplified: Down/Neutral/Up (was 5)
    # Wider neutral zone to filter noise - anything between -0.25σ and +0.25σ is Neutral
    class_std_thresholds: Tuple[float, float] = (-0.25, 0.25)


@dataclass
class TradingConfig:
    """Trading environment configuration."""

    # Transaction costs (in pips)
    spread_pips: float = 1.5

    # Position sizing (multipliers)
    position_sizes: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)

    # Reward shaping
    # NOTE: Penalties are now scaled higher to compete with PnL rewards.
    # With reward_scaling=0.1, a ±20 pip trade becomes ±2.0 reward,
    # making penalties (-2.0, -1.0) meaningful for "Sniper" behavior.
    fomo_penalty: float = -2.0      # Penalty for being flat during momentum
    chop_penalty: float = -1.0      # Penalty for holding in ranging market
    fomo_threshold_atr: float = 2.0 # ATR multiplier for FOMO detection
    chop_threshold: float = 60.0    # Choppiness index threshold
    reward_scaling: float = 0.1     # Scale PnL to balance with penalties

    # Environment settings
    max_steps_per_episode: int = 2000
    initial_balance: float = 10000.0  # Not used for now, but can be extended


@dataclass
class AgentConfig:
    """PPO Sniper Agent configuration."""

    # PPO hyperparameters (from CLAUDE.md spec)
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01       # Entropy coefficient for exploration
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Training
    total_timesteps: int = 500_000

    # Policy network
    policy_type: str = "MlpPolicy"
    net_arch: List[int] = field(default_factory=lambda: [128, 128])


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
class Config:
    """Master configuration combining all sub-configs."""

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    analyst: AnalystConfig = field(default_factory=AnalystConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
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
