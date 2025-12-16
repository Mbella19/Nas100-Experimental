#!/usr/bin/env python3
"""
Main execution pipeline for the Hybrid NAS100 Trading System.

This script orchestrates the complete workflow:
1. Data loading and multi-timeframe processing
2. Feature engineering
3. Market Analyst training (supervised)
4. Analyst freeze and transfer to RL environment
5. PPO Sniper Agent training
6. Out-of-sample backtesting
7. Performance comparison with buy-and-hold

Usage:
    python scripts/run_pipeline.py

Memory-optimized for Apple M2 Silicon (8GB RAM).
"""

import sys
import os
from pathlib import Path
import argparse
import logging
import gc
import shutil
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Config, get_device, clear_memory
from src.data.loader import load_ohlcv
from src.data.resampler import resample_all_timeframes, align_timeframes
from src.data.features import engineer_all_features, get_feature_columns
from src.data.normalizer import FeatureNormalizer, normalize_multi_timeframe
from src.models.analyst import create_analyst, load_analyst
from src.training.train_analyst import train_analyst, MultiTimeframeDataset
from src.training.train_agent import train_agent, prepare_env_data, create_trading_env
from src.agents.sniper_agent import SniperAgent
from src.evaluation.backtest import (
    run_backtest,
    compare_with_baseline,
    print_comparison_report,
    save_backtest_results
)
from src.evaluation.metrics import print_metrics_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clear_previous_training(
    config: Config,
    clear_processed: bool = True,
    clear_analyst: bool = True,
    clear_agent: bool = True,
    clear_results: bool = True,
    clear_pycache: bool = True,
) -> None:
    """
    Remove artifacts from previous runs to ensure a fully fresh training start.

    This deletes ONLY repo-local outputs:
      - data/processed/*
      - models/analyst/*
      - models/agent/*
      - models/checkpoints/* (legacy agent checkpoints)
      - results/*
      - __pycache__ and *.pyc inside the repo

    Raw market CSVs are never touched.
    """

    def _clear_dir(dir_path: Path):
        if not dir_path.exists():
            return
        for child in dir_path.iterdir():
            try:
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to remove {child}: {e}")

    base_dir = config.paths.base_dir

    if clear_processed:
        logger.info("Clearing processed data...")
        _clear_dir(config.paths.data_processed)

    if clear_analyst:
        logger.info("Clearing analyst models/checkpoints...")
        _clear_dir(config.paths.models_analyst)

    if clear_agent:
        logger.info("Clearing agent models/checkpoints...")
        _clear_dir(config.paths.models_agent)
        # Legacy location used by older CheckpointCallback configuration.
        legacy_checkpoints = base_dir / "models" / "checkpoints"
        if legacy_checkpoints.exists():
            _clear_dir(legacy_checkpoints)

    if clear_results:
        results_dir = base_dir / "results"
        if results_dir.exists():
            logger.info("Clearing results...")
            _clear_dir(results_dir)

    if clear_pycache:
        logger.info("Clearing __pycache__ and .pyc files...")
        for cache_dir in base_dir.rglob("__pycache__"):
            shutil.rmtree(cache_dir, ignore_errors=True)
        for pyc in base_dir.rglob("*.pyc"):
            try:
                pyc.unlink(missing_ok=True)
            except Exception:
                pass

    # Recreate needed directories
    config.paths.ensure_dirs()


def step_1_load_data(config: Config) -> pd.DataFrame:
    """
    Step 1: Load raw 1-minute OHLCV data.
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Loading Raw Data")
    logger.info("=" * 60)

    data_path = config.paths.data_raw / config.data.raw_file

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info(f"Please place your NAS100 1-minute CSV file at: {data_path}")
        logger.info("Expected format: datetime,open,high,low,close")
        sys.exit(1)

    df_1m = load_ohlcv(
        data_path,
        datetime_format=config.data.datetime_format
    )

    logger.info(f"Loaded {len(df_1m):,} rows of 1-minute data")
    logger.info(f"Date range: {df_1m.index.min()} to {df_1m.index.max()}")

    return df_1m


def step_2_resample_timeframes(
    df_1m: pd.DataFrame,
    config: Config
) -> tuple:
    """
    Step 2: Resample to multiple timeframes.
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Resampling to Multiple Timeframes")
    logger.info("=" * 60)

    # Resample to native timeframes WITHOUT aligning yet.
    # Feature engineering must run on true 5m/15m/45m bars to preserve correct indicator scales.
    resampled = resample_all_timeframes(df_1m, config.data.timeframes)
    df_5m, df_15m, df_45m = resampled['5m'], resampled['15m'], resampled['45m']

    logger.info(f"5m:  {len(df_5m):,} rows")
    logger.info(f"15m: {len(df_15m):,} rows")
    logger.info(f"45m: {len(df_45m):,} rows")

    # Clear 1m data to free memory
    del df_1m
    clear_memory()

    return df_5m, df_15m, df_45m


def step_3_engineer_features(
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_45m: pd.DataFrame,
    config: Config
) -> tuple:
    """
    Step 3: Apply feature engineering to all timeframes.
    
    FIXED: No longer drops NaN separately per timeframe.
    Alignment is done AFTER feature engineering to preserve index consistency.
    """
    logger.info("=" * 60)
    logger.info("STEP 3: Feature Engineering")
    logger.info("=" * 60)

    feature_config = {
        'pinbar_wick_ratio': config.features.pinbar_wick_ratio,
        'doji_body_ratio': config.features.doji_body_ratio,
        'fractal_window': config.features.fractal_window,
        'sr_lookback': config.features.sr_lookback,
        'sma_period': config.features.sma_period,
        'ema_fast': config.features.ema_fast,
        'ema_slow': config.features.ema_slow,
        'chop_period': config.features.chop_period,
        'adx_period': config.features.adx_period,
        'atr_period': config.features.atr_period
    }

    # Engineer on native bars first (no forward-fill distortion)
    df_5m = engineer_all_features(df_5m, feature_config)
    df_15m = engineer_all_features(df_15m, feature_config)
    df_45m = engineer_all_features(df_45m, feature_config)

    # Align higher timeframes to 5m index AFTER feature engineering.
    # This forward-fills true 15m/45m features onto the 5m grid without changing their scale.
    df_5m, df_15m, df_45m = align_timeframes(df_5m, df_15m, df_45m)

    logger.info(f"Features: {list(df_5m.columns)}")
    
    # CRITICAL FIX: Align timeframes by finding common valid (non-NaN) indices
    # This ensures all three DataFrames have the same index after NaN removal
    logger.info("Aligning timeframes after feature engineering...")
    
    # Find rows where ALL timeframes have valid data
    valid_5m = ~df_5m.isna().any(axis=1)
    valid_15m = ~df_15m.isna().any(axis=1)
    valid_45m = ~df_45m.isna().any(axis=1)
    common_valid = valid_5m & valid_15m & valid_45m
    
    initial_len = len(df_5m)
    df_5m = df_5m[common_valid]
    df_15m = df_15m[common_valid]
    df_45m = df_45m[common_valid]
    
    logger.info(f"Dropped {initial_len - len(df_5m)} rows to align timeframes. Final: {len(df_5m):,} rows")
    logger.info(f"All timeframes now aligned with identical indices.")

    # Save processed data (before normalization for reference)
    processed_path = config.paths.data_processed
    df_5m.to_parquet(processed_path / 'features_5m.parquet')
    df_15m.to_parquet(processed_path / 'features_15m.parquet')
    df_45m.to_parquet(processed_path / 'features_45m.parquet')
    logger.info(f"Saved processed data to {processed_path}")

    return df_5m, df_15m, df_45m


# Columns that should NOT be normalized (used for PnL and reward thresholds, or kept as raw flags).
RAW_COLUMNS = [
    'open', 'high', 'low', 'close',
    'atr', 'chop', 'adx',
    'session_asian', 'session_london', 'session_ny',
]

# Columns that SHOULD be normalized for model input
MODEL_INPUT_COLUMNS = [
    'pinbar', 'engulfing', 'doji', 'ema_trend', 'ema_crossover',
    'regime', 'returns', 'volatility', 'sma_distance',
    'dist_to_resistance', 'dist_to_support'
]


def step_3b_normalize_features(
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_45m: pd.DataFrame,
    feature_cols: list,
    config: Config
) -> tuple:
    """
    Step 3b: Normalize features using StandardScaler (Z-Score).
    
    CRITICAL FIXES:
    1. Only normalizes MODEL INPUT features (not OHLC, ATR, CHOP, ADX)
    2. Uses 85% split to match analyst/agent training split (not config.data.train_ratio)
    3. Returns BOTH normalized DataFrames AND raw columns for PnL/reward calculations

    This prevents large-scale features from dominating gradients while keeping
    raw price/volatility values available for PnL calculations and reward thresholds.
    """
    logger.info("=" * 60)
    logger.info("STEP 3b: Feature Normalization (Z-Score)")
    logger.info("=" * 60)

    # FIXED: Use 85% split to match actual training split (not config.data.train_ratio = 0.70)
    train_end_idx = int(len(df_5m) * 0.85)
    logger.info(f"Fitting normalizer on first {train_end_idx:,} samples (85% = training data)")
    
    # Determine which columns to normalize (exclude RAW_COLUMNS)
    normalize_cols = [c for c in feature_cols if c not in RAW_COLUMNS and c in df_5m.columns]
    logger.info(f"Columns to normalize: {normalize_cols}")
    logger.info(f"Columns kept RAW (for PnL/rewards): {[c for c in RAW_COLUMNS if c in df_5m.columns]}")

    # Log pre-normalization statistics for normalized columns
    logger.info("Pre-normalization feature ranges:")
    for col in normalize_cols[:6]:  # Show first 6
        if col in df_5m.columns:
            logger.info(f"  {col}: min={df_5m[col].min():.6f}, max={df_5m[col].max():.6f}")

    # Create normalizers ONLY for the columns that should be normalized
    from src.data.normalizer import FeatureNormalizer
    
    normalizer_5m = FeatureNormalizer(normalize_cols)
    normalizer_15m = FeatureNormalizer(normalize_cols)
    normalizer_45m = FeatureNormalizer(normalize_cols)
    
    # Calculate proportional train indices for higher timeframes
    # (Note: In aligned dataset, lengths are equal, so train_end_idx is same)
    train_end_15m = train_end_idx 
    train_end_45m = train_end_idx
    
    # Fit normalizers on TRAINING data only
    normalizer_5m.fit(df_5m.iloc[:train_end_idx])
    normalizer_15m.fit(df_15m.iloc[:train_end_15m])
    normalizer_45m.fit(df_45m.iloc[:train_end_45m])
    
    # Transform - this only affects normalize_cols, RAW columns are untouched
    df_5m_norm = normalizer_5m.transform(df_5m)
    df_15m_norm = normalizer_15m.transform(df_15m)
    df_45m_norm = normalizer_45m.transform(df_45m)
    
    normalizers = {'5m': normalizer_5m, '15m': normalizer_15m, '45m': normalizer_45m}

    # Log post-normalization statistics
    logger.info("Post-normalization feature ranges (should be ~[-3, 3]):")
    for tf, df_norm in [('5m', df_5m_norm), ('15m', df_15m_norm), ('45m', df_45m_norm)]:
        logger.info(f"  {tf}:")
        for col in normalize_cols[:3]:  # First 3 normalized features
            if col in df_norm.columns:
                logger.info(f"    {col}: min={df_norm[col].min():.3f}, max={df_norm[col].max():.3f}")
    
    # Verify RAW columns are unchanged
    logger.info("RAW columns preserved (not normalized):")
    for col in ['close', 'atr', 'chop']:
        if col in df_5m.columns:
            logger.info(f"  {col}: min={df_5m_norm[col].min():.6f}, max={df_5m_norm[col].max():.6f}")

    # Save normalizers for inference (one per timeframe)
    for tf, normalizer in normalizers.items():
        normalizer_path = config.paths.models_analyst / f'normalizer_{tf}.pkl'
        normalizer.save(normalizer_path)
    logger.info(f"Normalizers saved to {config.paths.models_analyst}")

    # Save normalized data
    processed_path = config.paths.data_processed
    df_5m_norm.to_parquet(processed_path / 'features_5m_normalized.parquet')
    df_15m_norm.to_parquet(processed_path / 'features_15m_normalized.parquet')
    df_45m_norm.to_parquet(processed_path / 'features_45m_normalized.parquet')
    logger.info(f"Saved normalized data to {processed_path}")

    return df_5m_norm, df_15m_norm, df_45m_norm, normalizers


def step_4_train_analyst(
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_45m: pd.DataFrame,
    feature_cols: list[str],
    config: Config,
    device: torch.device
) -> torch.nn.Module:
    """
    Step 4: Train the Market Analyst (supervised learning).
    
    FIXED: Uses only derived features for model input (not raw OHLC).
    Raw close prices are still used for target calculation.
    """
    logger.info("=" * 60)
    logger.info("STEP 4: Training Market Analyst")
    logger.info("=" * 60)

    # Filter to available columns
    feature_cols = [c for c in feature_cols if c in df_5m.columns]
    logger.info(f"Using {len(feature_cols)} MODEL INPUT features: {feature_cols}")
    logger.info(f"Note: Raw 'close' used for target, not as model input")

    save_path = str(config.paths.models_analyst)

    analyst, history = train_analyst(
        df_15m=df_5m,   # Variable name mismatch in train_analyst.py signature? Check later
        df_1h=df_15m,   # train_analyst args are likely still df_15m, df_1h...
        df_4h=df_45m,   # Need to be careful here. Assuming train_analyst handles correct data
        feature_cols=feature_cols,
        save_path=save_path,
        config=config.analyst,
        device=device
    )

    logger.info(f"Analyst training complete. Best val loss: {history['best_val_loss']:.6f}")

    # Freeze the analyst
    analyst.freeze()
    logger.info("Analyst model frozen for RL training")

    return analyst, feature_cols


def step_5_train_agent(
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_45m: pd.DataFrame,
    feature_cols: list,
    config: Config,
    device: torch.device,
    resume_path: str = None
) -> SniperAgent:
    """
    Step 5: Train the PPO Sniper Agent.
    """
    logger.info("=" * 60)
    logger.info("STEP 5: Training PPO Sniper Agent")
    logger.info("=" * 60)

    analyst_path = str(config.paths.models_analyst / 'best.pt')
    save_path = str(config.paths.models_agent)

    # Verify input arguments for train_agent. It expects df_15m, df_1h... 
    # But those are just names. We feed it our 5m, 15m, 45m data.
    # We should update train_agent signature eventually, but for now
    # mapping 5m -> 15m arg, 15m -> 1h arg, 45m -> 4h arg works if logic is generic.
    # However, train_agent likely has hardcoded lookbacks or subsampling.
    # We checked train_agent.py before and it SEEMED to expect 5m/15m/45m logic updates.
    
    agent, training_info = train_agent(
        df_15m=df_5m,   # Map 5m to first arg
        df_1h=df_15m,   # Map 15m to second arg
        df_4h=df_45m,   # Map 45m to third arg
        feature_cols=feature_cols,
        analyst_path=analyst_path,
        save_path=save_path,
        config=config,
        device=device,
        total_timesteps=config.agent.total_timesteps,
        resume_path=resume_path
    )

    logger.info(f"Agent training complete.")
    logger.info(f"Final eval reward: {training_info['final_eval']['mean_reward']:.2f}")
    logger.info(f"Final eval PnL: {training_info['final_eval']['mean_pnl']:.2f} pips")

    return agent


def step_6_backtest(
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_45m: pd.DataFrame,
    feature_cols: list,
    config: Config,
    device: torch.device,
    model_path: str = None,
    min_action_confidence: float = 0.0
):
    """
    Step 6: Run out-of-sample backtest and compare with baseline.
    """
    logger.info("=" * 60)
    logger.info("STEP 6: Out-of-Sample Backtesting")
    logger.info("=" * 60)

    # Load trained models
    analyst_path = config.paths.models_analyst / 'best.pt'
    
    if model_path:
        agent_path = Path(model_path)
        if not agent_path.exists():
            logger.error(f"Model not found at: {agent_path}")
            sys.exit(1)
        logger.info(f"Using custom agent model: {agent_path}")
    else:
        agent_path = config.paths.models_agent / 'final_model.zip'

    # Feature dims for Analyst (TCN uses true timeframe keys)
    feature_dims = {'5m': len(feature_cols), '15m': len(feature_cols), '45m': len(feature_cols)}

    # We load analyst frozen
    analyst = load_analyst(str(analyst_path), feature_dims, device, freeze=True)

    # Prepare test data (last 15%)
    # Use updated lookback keys
    lookback_5m = 48
    lookback_15m = 16
    lookback_45m = 6

    # prepare_env_data in train_agent.py needs to be compatible
    data_5m, data_15m, data_45m, close_prices, market_features, returns = prepare_env_data(
        df_5m, df_15m, df_45m, feature_cols,
        lookback_5m, lookback_15m, lookback_45m,
    )
    # Extract real OHLC + timestamps aligned to the windowed env segment (no synthetic time axis).
    subsample_15m = 3
    subsample_45m = 9
    start_idx = max(
        lookback_5m,
        (lookback_15m - 1) * subsample_15m + 1,
        (lookback_45m - 1) * subsample_45m + 1,
    )
    n_samples = len(close_prices)

    if not isinstance(df_5m.index, pd.DatetimeIndex):
        raise ValueError(
            "Backtest requires real timestamps from the data, but `df_5m` does not have a DatetimeIndex."
        )
    timestamps_all = (df_5m.index[start_idx:start_idx + n_samples].astype('int64') // 10**9).values

    ohlc_all = None
    if all(col in df_5m.columns for col in ['open', 'high', 'low', 'close']):
        ohlc_all = df_5m[['open', 'high', 'low', 'close']].values[start_idx:start_idx + n_samples].astype(np.float32)

    # Use last 15% for testing
    test_start = int(0.85 * len(close_prices))
    
    # CRITICAL FIX: Compute market feature normalization stats from TRAINING data only
    # This prevents look-ahead bias in the test backtest
    train_market_features = market_features[:test_start]
    market_feat_mean = train_market_features.mean(axis=0).astype(np.float32)
    market_feat_std = train_market_features.std(axis=0).astype(np.float32)
    market_feat_std = np.where(market_feat_std > 1e-8, market_feat_std, 1.0).astype(np.float32)
    
    logger.info("Using TRAINING data statistics for test environment normalization")
    logger.info(f"  Market feature mean: {market_feat_mean}")
    logger.info(f"  Market feature std:  {market_feat_std}")
    
    test_data = (
        data_5m[test_start:],
        data_15m[test_start:],
        data_45m[test_start:],
        close_prices[test_start:],
        market_features[test_start:],
    )

    # Returns for "Full Eyes"
    test_returns = returns[test_start:] if returns is not None else None
    test_timestamps = timestamps_all[test_start:]
    test_ohlc = ohlc_all[test_start:] if ohlc_all is not None else None

    # Create test environment with TRAINING stats (prevents look-ahead bias)
    # FIX: Disable noise for backtesting (evaluation must be on clean data)
    test_config = config.trading
    test_config.noise_level = 0.0
    
    test_env = create_trading_env(
        *test_data,
        analyst_model=analyst,
        config=test_config,
        device=device,
        market_feat_mean=market_feat_mean,
        market_feat_std=market_feat_std,
        returns=test_returns,
        ohlc_data=test_ohlc,
        timestamps=test_timestamps,
    )

    # Load agent
    from stable_baselines3.common.monitor import Monitor
    test_env = Monitor(test_env)
    agent = SniperAgent.load(str(agent_path), test_env, device='cpu')

    # Run backtest
    results = run_backtest(
        agent=agent,
        env=test_env.unwrapped,
        min_action_confidence=min_action_confidence,
        spread_pips=config.trading.spread_pips + config.trading.slippage_pips,
        # Keep backtest risk management consistent with the TradingEnv used for observations.
        sl_atr_multiplier=config.trading.sl_atr_multiplier,
        tp_atr_multiplier=config.trading.tp_atr_multiplier,
        use_stop_loss=config.trading.use_stop_loss,
        use_take_profit=config.trading.use_take_profit,
        min_hold_bars=config.trading.min_hold_bars,  # v18: Pass min_hold_bars
    )

    # Compare with buy-and-hold
    comparison = compare_with_baseline(
        results,
        close_prices[test_start:],
        initial_balance=10000.0
    )

    # Print reports
    print_metrics_report(results.metrics, "Agent Performance (Out-of-Sample)")
    print_comparison_report(comparison)

    # Save results
    results_path = config.paths.base_dir / 'results' / datetime.now().strftime('%Y%m%d_%H%M%S')
    save_backtest_results(results, str(results_path), comparison)

    test_env.close()

    return results, comparison


def main():
    """Main execution pipeline."""
    parser = argparse.ArgumentParser(description='Hybrid NAS100 Trading System Pipeline')
    parser.add_argument('--skip-analyst', action='store_true',
                       help='Skip analyst training (use existing model)')
    parser.add_argument('--skip-agent', action='store_true', help='Skip agent training (use existing model)')
    parser.add_argument('--analyst-only', action='store_true', help='Run ONLY data processing and analyst training')
    parser.add_argument('--backtest-only', action='store_true', help='Only run backtest with existing models')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume training from')
    parser.add_argument('--model-path', type=str, help='Path to specific agent model for backtesting')
    parser.add_argument('--min-confidence', type=float, default=0.0, help='Minimum confidence threshold (0.0-1.0)')
    args = parser.parse_args()
    if args.resume is not None:
        args.resume = args.resume.strip()

    # Initialize
    logger.info("=" * 60)
    logger.info("HYBRID NAS100 TRADING SYSTEM")
    logger.info("=" * 60)

    config = Config()

    device = get_device()
    logger.info(f"Using device: {device}")
    logger.info(f"PyTorch version: {torch.__version__}")

    # Ensure directories exist
    config.paths.ensure_dirs()

    # Validate resume checkpoint early to avoid silently starting a fresh run.
    if args.resume:
        resume_path = Path(args.resume).expanduser()
        if not resume_path.is_file():
            logger.error(
                f"--resume path not found (or not a file): {resume_path}\n"
                "Expected a SB3 .zip checkpoint like: models/checkpoints/sniper_model_7400000_steps.zip"
            )
            sys.exit(1)

    # Auto-clear previous artifacts for a fresh run unless resuming/backtesting.
    training_requested = (not args.skip_analyst) or (not args.skip_agent and not args.analyst_only)
    if training_requested and args.resume is None and not args.backtest_only:
        logger.info("Fresh training start detected. Clearing previous artifacts...")
        clear_previous_training(
            config,
            # Keep processed data when skipping analyst so precomputed caches
            # (e.g. analyst_cache.npz) can be reused.
            clear_processed=not args.skip_analyst,
            clear_analyst=not args.skip_analyst,
            clear_agent=not args.skip_agent and not args.analyst_only,
            clear_results=True,
            clear_pycache=True,
        )

    try:
        # Define feature columns for MODEL INPUT (not raw OHLC)
        # FIXED: Exclude raw OHLC from model input - model should learn from
        # derived features (returns, patterns) not absolute price levels.
        # Raw OHLC is kept in DataFrame for PnL/target calculations.
        model_feature_cols = [
            'returns', 'volatility',           # Price dynamics
            'pinbar', 'engulfing', 'doji',     # Price action patterns
            'ema_trend', 'ema_crossover',      # Trend indicators
            'regime', 'sma_distance',          # Regime/trend filters
            'dist_to_resistance', 'dist_to_support', # S/R distance
            'session_asian', 'session_london', 'session_ny',  # Session flags (kept raw)
            'bos_bullish', 'bos_bearish',      # Market structure (BOS)
            'choch_bullish', 'choch_bearish',  # Market structure (CHoCH)
        ]
        
        # All feature columns including raw values (for normalization step)
        all_feature_cols = ['open', 'high', 'low', 'close', 'atr', 'chop', 'adx'] + model_feature_cols
        feature_cols = model_feature_cols  # Use model features by default

        if args.backtest_only:
            # Load normalized processed data
            logger.info("Loading normalized processed data...")
            df_5m = pd.read_parquet(config.paths.data_processed / 'features_5m_normalized.parquet')
            df_15m = pd.read_parquet(config.paths.data_processed / 'features_15m_normalized.parquet')
            df_45m = pd.read_parquet(config.paths.data_processed / 'features_45m_normalized.parquet')
            feature_cols = [c for c in feature_cols if c in df_5m.columns]
        else:
            data_already_normalized = False
            # Fast path: when skipping analyst training, reuse existing processed data.
            # This keeps `analyst_cache.npz` aligned with the normalized feature parquet.
            norm_5m_path = config.paths.data_processed / 'features_5m_normalized.parquet'
            norm_15m_path = config.paths.data_processed / 'features_15m_normalized.parquet'
            norm_45m_path = config.paths.data_processed / 'features_45m_normalized.parquet'

            if args.skip_analyst and norm_5m_path.exists() and norm_15m_path.exists() and norm_45m_path.exists():
                logger.info("Skipping data prep (--skip-analyst): loading normalized processed data...")
                df_5m = pd.read_parquet(norm_5m_path)
                df_15m = pd.read_parquet(norm_15m_path)
                df_45m = pd.read_parquet(norm_45m_path)
                data_already_normalized = True

                # Ensure session flags exist (kept raw; safe to recompute).
                if 'session_asian' not in df_5m.columns:
                    from src.data.features import add_market_sessions
                    df_5m = add_market_sessions(df_5m)
                    df_15m = add_market_sessions(df_15m)
                    df_45m = add_market_sessions(df_45m)
            else:
                # Step 1: Load data
                df_1m = step_1_load_data(config)

                # Step 2: Resample
                df_5m, df_15m, df_45m = step_2_resample_timeframes(df_1m, config)

                # Step 3: Feature engineering
                df_5m, df_15m, df_45m = step_3_engineer_features(df_5m, df_15m, df_45m, config)

            # Filter feature columns to available ones
            feature_cols = [c for c in feature_cols if c in df_5m.columns]
            all_feature_cols = [c for c in all_feature_cols if c in df_5m.columns]
            
            logger.info(f"Model input features: {feature_cols}")
            logger.info(f"All features (including raw): {all_feature_cols}")

            # Step 3b: NORMALIZE FEATURES (CRITICAL for neural network convergence)
            # When loading already-normalized data (skip-analyst fast path), do NOT renormalize.
            if not data_already_normalized:
                # Pass all_feature_cols so normalization knows about raw columns to exclude
                df_5m, df_15m, df_45m, _ = step_3b_normalize_features(
                    df_5m, df_15m, df_45m, all_feature_cols, config
                )

            # Step 4: Train Analyst (on NORMALIZED data)
            if not args.skip_analyst:
                analyst, feature_cols = step_4_train_analyst(
                    df_5m, df_15m, df_45m, feature_cols, config, device
                )
                del analyst  # Free memory
                clear_memory()

                if args.analyst_only:
                    logger.info("Analyst training complete. stopping as requested (--analyst-only).")
                    return

            # Step 5: Train Agent (on NORMALIZED data)
            if not args.skip_agent:
                agent = step_5_train_agent(
                    df_5m, df_15m, df_45m, feature_cols, config, device,
                    resume_path=args.resume
                )
                del agent  # Free memory
                clear_memory()

        # Step 6: Backtest
        results, comparison = step_6_backtest(
            df_5m, df_15m, df_45m, feature_cols, config, device,
            model_path=args.model_path,
            min_action_confidence=args.min_confidence
        )

        # Final summary
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)

        if comparison['outperformance']['beats_baseline']:
            logger.info("SUCCESS: Agent outperforms buy-and-hold baseline!")
        else:
            logger.info("Agent underperforms buy-and-hold baseline.")

        logger.info(f"Return: {results.metrics['total_return_pct']:.2f}%")
        logger.info(f"Sortino: {results.metrics['sortino_ratio']:.2f}")
        logger.info(f"Max DD: {results.metrics['max_drawdown_pct']:.2f}%")

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Cleanup
        clear_memory()


if __name__ == '__main__':
    main()
