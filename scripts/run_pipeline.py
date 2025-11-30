#!/usr/bin/env python3
"""
Main execution pipeline for the Hybrid EURUSD Trading System.

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
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Config, get_device, clear_memory
from src.data.loader import load_ohlcv
from src.data.resampler import create_multi_timeframe_dataset
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
        logger.info(f"Please place your EURUSD 1-minute CSV file at: {data_path}")
        logger.info("Expected format: datetime,open,high,low,close,volume")
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

    df_15m, df_1h, df_4h = create_multi_timeframe_dataset(df_1m, config.data.timeframes)

    logger.info(f"15m: {len(df_15m):,} rows")
    logger.info(f"1H:  {len(df_1h):,} rows")
    logger.info(f"4H:  {len(df_4h):,} rows")

    # Clear 1m data to free memory
    del df_1m
    clear_memory()

    return df_15m, df_1h, df_4h


def step_3_engineer_features(
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
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

    df_15m = engineer_all_features(df_15m, feature_config)
    df_1h = engineer_all_features(df_1h, feature_config)
    df_4h = engineer_all_features(df_4h, feature_config)

    logger.info(f"Features: {list(df_15m.columns)}")
    
    # CRITICAL FIX: Align timeframes by finding common valid (non-NaN) indices
    # This ensures all three DataFrames have the same index after NaN removal
    logger.info("Aligning timeframes after feature engineering...")
    
    # Find rows where ALL timeframes have valid data
    valid_15m = ~df_15m.isna().any(axis=1)
    valid_1h = ~df_1h.isna().any(axis=1)
    valid_4h = ~df_4h.isna().any(axis=1)
    common_valid = valid_15m & valid_1h & valid_4h
    
    initial_len = len(df_15m)
    df_15m = df_15m[common_valid]
    df_1h = df_1h[common_valid]
    df_4h = df_4h[common_valid]
    
    logger.info(f"Dropped {initial_len - len(df_15m)} rows to align timeframes. Final: {len(df_15m):,} rows")
    logger.info(f"All timeframes now aligned with identical indices.")

    # Save processed data (before normalization for reference)
    processed_path = config.paths.data_processed
    df_15m.to_parquet(processed_path / 'features_15m.parquet')
    df_1h.to_parquet(processed_path / 'features_1h.parquet')
    df_4h.to_parquet(processed_path / 'features_4h.parquet')
    logger.info(f"Saved processed data to {processed_path}")

    return df_15m, df_1h, df_4h


# Columns that should NOT be normalized (used for PnL and reward thresholds)
RAW_COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'atr', 'chop', 'adx']

# Columns that SHOULD be normalized for model input
MODEL_INPUT_COLUMNS = [
    'pinbar', 'engulfing', 'doji', 'ema_trend', 'ema_crossover',
    'regime', 'returns', 'volatility', 'sma_distance',
    'dist_to_resistance', 'dist_to_support'
]


def step_3b_normalize_features(
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
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
    train_end_idx = int(len(df_15m) * 0.85)
    logger.info(f"Fitting normalizer on first {train_end_idx:,} samples (85% = training data)")
    
    # Determine which columns to normalize (exclude RAW_COLUMNS)
    normalize_cols = [c for c in feature_cols if c not in RAW_COLUMNS and c in df_15m.columns]
    logger.info(f"Columns to normalize: {normalize_cols}")
    logger.info(f"Columns kept RAW (for PnL/rewards): {[c for c in RAW_COLUMNS if c in df_15m.columns]}")

    # Log pre-normalization statistics for normalized columns
    logger.info("Pre-normalization feature ranges:")
    for col in normalize_cols[:6]:  # Show first 6
        if col in df_15m.columns:
            logger.info(f"  {col}: min={df_15m[col].min():.6f}, max={df_15m[col].max():.6f}")

    # Create normalizers ONLY for the columns that should be normalized
    from src.data.normalizer import FeatureNormalizer
    
    normalizer_15m = FeatureNormalizer(normalize_cols)
    normalizer_1h = FeatureNormalizer(normalize_cols)
    normalizer_4h = FeatureNormalizer(normalize_cols)
    
    # Calculate proportional train indices for higher timeframes
    train_ratio = train_end_idx / len(df_15m)
    train_end_1h = int(len(df_1h) * train_ratio)
    train_end_4h = int(len(df_4h) * train_ratio)
    
    # Fit normalizers on TRAINING data only
    normalizer_15m.fit(df_15m.iloc[:train_end_idx])
    normalizer_1h.fit(df_1h.iloc[:train_end_1h])
    normalizer_4h.fit(df_4h.iloc[:train_end_4h])
    
    # Transform - this only affects normalize_cols, RAW columns are untouched
    df_15m_norm = normalizer_15m.transform(df_15m)
    df_1h_norm = normalizer_1h.transform(df_1h)
    df_4h_norm = normalizer_4h.transform(df_4h)
    
    normalizers = {'15m': normalizer_15m, '1h': normalizer_1h, '4h': normalizer_4h}

    # Log post-normalization statistics
    logger.info("Post-normalization feature ranges (should be ~[-3, 3]):")
    for tf, df_norm in [('15m', df_15m_norm), ('1h', df_1h_norm), ('4h', df_4h_norm)]:
        logger.info(f"  {tf}:")
        for col in normalize_cols[:3]:  # First 3 normalized features
            if col in df_norm.columns:
                logger.info(f"    {col}: min={df_norm[col].min():.3f}, max={df_norm[col].max():.3f}")
    
    # Verify RAW columns are unchanged
    logger.info("RAW columns preserved (not normalized):")
    for col in ['close', 'atr', 'chop']:
        if col in df_15m.columns:
            logger.info(f"  {col}: min={df_15m_norm[col].min():.6f}, max={df_15m_norm[col].max():.6f}")

    # Save normalizers for inference (one per timeframe)
    for tf, normalizer in normalizers.items():
        normalizer_path = config.paths.models_analyst / f'normalizer_{tf}.pkl'
        normalizer.save(normalizer_path)
    logger.info(f"Normalizers saved to {config.paths.models_analyst}")

    # Save normalized data
    processed_path = config.paths.data_processed
    df_15m_norm.to_parquet(processed_path / 'features_15m_normalized.parquet')
    df_1h_norm.to_parquet(processed_path / 'features_1h_normalized.parquet')
    df_4h_norm.to_parquet(processed_path / 'features_4h_normalized.parquet')
    logger.info(f"Saved normalized data to {processed_path}")

    return df_15m_norm, df_1h_norm, df_4h_norm, normalizers


def step_4_train_analyst(
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
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

    # Feature columns for model INPUT (exclude raw OHLC)
    # FIXED: Model should learn from derived features, not absolute price levels
    feature_cols = [
        'returns', 'volatility',           # Price dynamics
        'pinbar', 'engulfing', 'doji',     # Price action patterns
        'ema_trend', 'ema_crossover',      # Trend indicators
        'regime', 'sma_distance',          # Regime/trend filters
        'dist_to_resistance', 'dist_to_support'  # S/R distance
    ]

    # Filter to available columns
    feature_cols = [c for c in feature_cols if c in df_15m.columns]
    logger.info(f"Using {len(feature_cols)} MODEL INPUT features: {feature_cols}")
    logger.info(f"Note: Raw 'close' used for target, not as model input")

    save_path = str(config.paths.models_analyst)

    analyst, history = train_analyst(
        df_15m=df_15m,
        df_1h=df_1h,
        df_4h=df_4h,
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
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    feature_cols: list,
    config: Config,
    device: torch.device
) -> SniperAgent:
    """
    Step 5: Train the PPO Sniper Agent.
    """
    logger.info("=" * 60)
    logger.info("STEP 5: Training PPO Sniper Agent")
    logger.info("=" * 60)

    analyst_path = str(config.paths.models_analyst / 'best.pt')
    save_path = str(config.paths.models_agent)

    agent, training_info = train_agent(
        df_15m=df_15m,
        df_1h=df_1h,
        df_4h=df_4h,
        feature_cols=feature_cols,
        analyst_path=analyst_path,
        save_path=save_path,
        config=config.agent,
        device=device,
        total_timesteps=config.agent.total_timesteps
    )

    logger.info(f"Agent training complete.")
    logger.info(f"Final eval reward: {training_info['final_eval']['mean_reward']:.2f}")
    logger.info(f"Final eval PnL: {training_info['final_eval']['mean_pnl']:.2f} pips")

    return agent


def step_6_backtest(
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    feature_cols: list,
    config: Config,
    device: torch.device
):
    """
    Step 6: Run out-of-sample backtest and compare with baseline.
    """
    logger.info("=" * 60)
    logger.info("STEP 6: Out-of-Sample Backtesting")
    logger.info("=" * 60)

    # Load trained models
    analyst_path = config.paths.models_analyst / 'best.pt'
    agent_path = config.paths.models_agent / 'final_model.zip'

    feature_dims = {'15m': len(feature_cols), '1h': len(feature_cols), '4h': len(feature_cols)}
    analyst = load_analyst(str(analyst_path), feature_dims, device, freeze=True)

    # Prepare test data (last 15%)
    lookback_15m = 48
    lookback_1h = 24
    lookback_4h = 12

    data_15m, data_1h, data_4h, close_prices, market_features = prepare_env_data(
        df_15m, df_1h, df_4h, feature_cols,
        lookback_15m, lookback_1h, lookback_4h
    )

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
        data_15m[test_start:],
        data_1h[test_start:],
        data_4h[test_start:],
        close_prices[test_start:],
        market_features[test_start:]
    )

    # Create test environment with TRAINING stats (prevents look-ahead bias)
    test_env = create_trading_env(
        *test_data,
        analyst_model=analyst,
        config=config.trading,
        device=device,
        market_feat_mean=market_feat_mean,
        market_feat_std=market_feat_std
    )

    # Load agent
    from stable_baselines3.common.monitor import Monitor
    test_env = Monitor(test_env)
    agent = SniperAgent.load(str(agent_path), test_env, device='cpu')

    # Run backtest
    results = run_backtest(agent, test_env.unwrapped)

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
    parser = argparse.ArgumentParser(description='Hybrid EURUSD Trading System Pipeline')
    parser.add_argument('--skip-analyst', action='store_true',
                       help='Skip analyst training (use existing model)')
    parser.add_argument('--skip-agent', action='store_true',
                       help='Skip agent training (use existing model)')
    parser.add_argument('--backtest-only', action='store_true',
                       help='Only run backtest with existing models')
    args = parser.parse_args()

    # Initialize
    logger.info("=" * 60)
    logger.info("HYBRID EURUSD TRADING SYSTEM")
    logger.info("=" * 60)

    config = Config()
    device = get_device()
    logger.info(f"Using device: {device}")
    logger.info(f"PyTorch version: {torch.__version__}")

    # Ensure directories exist
    config.paths.ensure_dirs()

    try:
        # Define feature columns for MODEL INPUT (not raw OHLC)
        # FIXED: Exclude raw OHLC from model input - model should learn from
        # derived features (returns, patterns) not absolute price levels.
        # Raw OHLC is kept in DataFrame for PnL/target calculations.
        model_feature_cols = [
            'returns', 'volatility',           # Price dynamics (normalized)
            'pinbar', 'engulfing', 'doji',     # Price action patterns
            'ema_trend', 'ema_crossover',      # Trend indicators
            'regime', 'sma_distance',          # Regime/trend filters
            'dist_to_resistance', 'dist_to_support'  # S/R distance (new)
        ]
        
        # All feature columns including raw values (for normalization step)
        all_feature_cols = ['open', 'high', 'low', 'close', 'atr', 'chop', 'adx'] + model_feature_cols
        feature_cols = model_feature_cols  # Use model features by default

        if args.backtest_only:
            # Load normalized processed data
            logger.info("Loading normalized processed data...")
            df_15m = pd.read_parquet(config.paths.data_processed / 'features_15m_normalized.parquet')
            df_1h = pd.read_parquet(config.paths.data_processed / 'features_1h_normalized.parquet')
            df_4h = pd.read_parquet(config.paths.data_processed / 'features_4h_normalized.parquet')
            feature_cols = [c for c in feature_cols if c in df_15m.columns]
        else:
            # Step 1: Load data
            df_1m = step_1_load_data(config)

            # Step 2: Resample
            df_15m, df_1h, df_4h = step_2_resample_timeframes(df_1m, config)

            # Step 3: Feature engineering
            df_15m, df_1h, df_4h = step_3_engineer_features(df_15m, df_1h, df_4h, config)

            # Filter feature columns to available ones
            feature_cols = [c for c in feature_cols if c in df_15m.columns]
            all_feature_cols = [c for c in all_feature_cols if c in df_15m.columns]
            
            logger.info(f"Model input features: {feature_cols}")
            logger.info(f"All features (including raw): {all_feature_cols}")

            # Step 3b: NORMALIZE FEATURES (CRITICAL for neural network convergence)
            # Pass all_feature_cols so normalization knows about raw columns to exclude
            df_15m, df_1h, df_4h, normalizer = step_3b_normalize_features(
                df_15m, df_1h, df_4h, all_feature_cols, config
            )

            # Step 4: Train Analyst (on NORMALIZED data)
            if not args.skip_analyst:
                analyst, feature_cols = step_4_train_analyst(
                    df_15m, df_1h, df_4h, config, device
                )
                del analyst  # Free memory
                clear_memory()

            # Step 5: Train Agent (on NORMALIZED data)
            if not args.skip_agent:
                agent = step_5_train_agent(
                    df_15m, df_1h, df_4h, feature_cols, config, device
                )
                del agent  # Free memory
                clear_memory()

        # Step 6: Backtest
        results, comparison = step_6_backtest(
            df_15m, df_1h, df_4h, feature_cols, config, device
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
