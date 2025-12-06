"""
Multi-timeframe resampling module.

Resamples 1-minute OHLCV data to 15m, 1H, and 4H timeframes
with proper gap handling via forward-fill on complete datetime index.

CRITICAL: Uses label='right' + closed='left' to prevent look-ahead bias!
- closed='left': Bins are [10:00, 11:00) - includes 10:00, excludes 11:00
- label='right': Row labeled 11:00 contains 10:00-10:59 data

At 10:15, we can only know the COMPLETED 9:00-10:00 hourly candle (labeled 10:00 with right),
not the in-progress 10:00-11:00 candle. This prevents future data leakage.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def create_complete_index(
    start: datetime,
    end: datetime,
    freq: str = '1min'
) -> pd.DatetimeIndex:
    """
    Create a complete datetime index without gaps.

    Args:
        start: Start datetime
        end: End datetime
        freq: Frequency string (e.g., '1min', '15min', '1H', '4H')

    Returns:
        Complete DatetimeIndex
    """
    return pd.date_range(start=start, end=end, freq=freq)


def resample_ohlcv(
    df: pd.DataFrame,
    freq: str,
    fill_gaps: bool = True
) -> pd.DataFrame:
    """
    Resample OHLCV data to a higher timeframe.

    CRITICAL: Uses label='right' to prevent look-ahead bias!
    - The timestamp represents when the candle COMPLETED, not when it started.
    - This ensures that at time T, we only have access to candles that
      completed at or before time T (no future data leakage).

    Example with 1H resample:
    - Data from 10:00-10:59 is labeled at 11:00 (when it completed)
    - At 10:30, forward-fill gives us the 10:00 candle (9:00-9:59 data) - CORRECT
    - Without label='right', at 10:30 we'd get 10:00 candle with 10:00-10:59 data - WRONG

    Args:
        df: DataFrame with datetime index and OHLCV columns
        freq: Target frequency (e.g., '15min', '1H', '4H')
        fill_gaps: Whether to forward-fill gaps

    Returns:
        Resampled DataFrame
    """
    # OHLCV resampling rules
    ohlcv_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }

    # Select only columns that exist
    rules = {col: rule for col, rule in ohlcv_rules.items() if col in df.columns}

    # Resample with label='right' + closed='left' to prevent look-ahead bias
    # closed='left': interval is [10:00, 11:00) - includes 10:00, excludes 11:00
    # label='right': timestamp is END of period (when candle completes)
    resampled = df.resample(freq, label='right', closed='left').agg(rules)

    if fill_gaps:
        # Create complete index and reindex
        complete_idx = create_complete_index(
            resampled.index.min(),
            resampled.index.max(),
            freq
        )
        resampled = resampled.reindex(complete_idx)

        # Forward-fill missing values
        resampled = resampled.ffill()
    else:
        # If NOT filling gaps, drop the empty bins created by resample()
        # These are the weekend gaps we want to remove
        resampled = resampled.dropna()

    # Ensure float32
    for col in resampled.columns:
        resampled[col] = resampled[col].astype(np.float32)

    logger.info(f"Resampled to {freq}: {len(resampled):,} rows")

    return resampled


def resample_all_timeframes(
    df_1m: pd.DataFrame,
    timeframes: Optional[Dict[str, str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Resample 1-minute data to all required timeframes.

    Args:
        df_1m: 1-minute OHLCV DataFrame
        timeframes: Dict mapping names to frequencies
                   Default: {'15m': '15min', '1h': '1H', '4h': '4H'}

    Returns:
        Dictionary of resampled DataFrames
    """
    if timeframes is None:
        timeframes = {
            '15m': '15min',
            '1h': '1H',
            '4h': '4H'
        }

    result = {}
    for name, freq in timeframes.items():
        logger.info(f"Resampling to {name} ({freq})...")
        # Disable gap filling to remove weekend flat lines
        result[name] = resample_ohlcv(df_1m, freq, fill_gaps=False)

    return result


def align_timeframes(
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Align multiple timeframe DataFrames to have matching indices.

    The alignment is done by forward-filling higher timeframe data
    onto the 15m index (the fastest timeframe).

    CRITICAL: Prevents look-ahead bias via label='right' resampling!
    - Higher timeframe candles are labeled at their COMPLETION time
    - Forward-fill ensures at time T, we only see candles that completed at/before T

    Example at 10:15 (15m candle):
    - 1H candles labeled: 09:00 (8:00-8:59), 10:00 (9:00-9:59), 11:00 (10:00-10:59)
    - Forward-fill at 10:15 gives us the 10:00 candle (9:00-9:59 data) ✓ CORRECT
    - Without label='right', we'd get 10:00 candle with 10:00-10:59 data ✗ WRONG

    Args:
        df_15m: 15-minute DataFrame
        df_1h: 1-hour DataFrame
        df_4h: 4-hour DataFrame

    Returns:
        Tuple of aligned DataFrames (15m, 1h, 4h)
    """
    # Use 15m index as the base (most granular)
    base_index = df_15m.index

    # Find common date range
    start = max(df_15m.index.min(), df_1h.index.min(), df_4h.index.min())
    end = min(df_15m.index.max(), df_1h.index.max(), df_4h.index.max())

    # Filter to common range
    df_15m = df_15m.loc[start:end].copy()
    df_1h = df_1h.loc[start:end].copy()
    df_4h = df_4h.loc[start:end].copy()

    # Align 1h and 4h to 15m index by reindexing and forward-filling
    # This ensures each 15m candle has corresponding higher TF context
    df_1h_aligned = df_1h.reindex(df_15m.index, method='ffill')
    df_4h_aligned = df_4h.reindex(df_15m.index, method='ffill')

    # Add suffix to avoid column name conflicts when merging
    df_1h_aligned = df_1h_aligned.add_suffix('_1h')
    df_4h_aligned = df_4h_aligned.add_suffix('_4h')

    # Drop any rows with NaN (at the start before first higher TF candle)
    valid_mask = ~(df_1h_aligned.isna().any(axis=1) | df_4h_aligned.isna().any(axis=1))
    df_15m = df_15m[valid_mask]
    df_1h_aligned = df_1h_aligned[valid_mask]
    df_4h_aligned = df_4h_aligned[valid_mask]

    # Remove suffix for return (keep original column names)
    df_1h_aligned.columns = [col.replace('_1h', '') for col in df_1h_aligned.columns]
    df_4h_aligned.columns = [col.replace('_4h', '') for col in df_4h_aligned.columns]

    logger.info(f"Aligned timeframes: {len(df_15m):,} rows from {start} to {end}")

    return df_15m, df_1h_aligned, df_4h_aligned


def create_multi_timeframe_dataset(
    df_1m: pd.DataFrame,
    timeframes: Optional[Dict[str, str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Complete pipeline: resample and align all timeframes.

    Args:
        df_1m: 1-minute OHLCV DataFrame
        timeframes: Optional custom timeframe mapping

    Returns:
        Tuple of aligned (15m, 1h, 4h) DataFrames
    """
    # Resample to all timeframes
    resampled = resample_all_timeframes(df_1m, timeframes)

    # Align timeframes
    df_15m, df_1h, df_4h = align_timeframes(
        resampled['15m'],
        resampled['1h'],
        resampled['4h']
    )

    return df_15m, df_1h, df_4h


def get_lookback_data(
    df: pd.DataFrame,
    idx: int,
    lookback: int
) -> pd.DataFrame:
    """
    Get lookback window of data ending at idx.

    Args:
        df: DataFrame
        idx: Current index position
        lookback: Number of rows to look back

    Returns:
        DataFrame slice with lookback rows
    """
    start_idx = max(0, idx - lookback + 1)
    return df.iloc[start_idx:idx + 1]
