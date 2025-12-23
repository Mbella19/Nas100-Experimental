"""
Feature engineering module for the hybrid trading system.

Implements:
- Price Action Patterns: Pinbar, Engulfing, Doji
- Market Structure: Fractal S/R, Distance to S/R
- Trend Filters: SMA distance, EMA crossovers
- Regime Detection: Choppiness Index, ADX
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Price Action Patterns
# =============================================================================

def detect_pinbar(
    df: pd.DataFrame,
    wick_ratio: float = 2.0,
    min_body_pips: float = 2.0,    # NAS100: 2 points (was 0.0002 for EURUSD)
    min_range_pips: float = 5.0   # NAS100: 5 points (was 0.0005 for EURUSD)
) -> pd.Series:
    """
    Detect pinbar candles (rejection candles with long wicks).

    A bullish pinbar has a lower wick > 2x body.
    A bearish pinbar has an upper wick > 2x body.

    Args:
        df: OHLCV DataFrame
        wick_ratio: Minimum wick-to-body ratio
        min_body_pips: Minimum body size in price units (2.0 = 2 points for NAS100)
        min_range_pips: Minimum candle range in price units (5.0 = 5 points for NAS100)

    Returns:
        Series with values: 1 (bullish), -1 (bearish), 0 (none)
    """
    body = abs(df['close'] - df['open'])
    total_range = df['high'] - df['low']
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']

    # Filter out noise candles that are too small to be meaningful
    valid_candle = (body > min_body_pips) & (total_range > min_range_pips)

    # Avoid division by zero
    body_safe = body.replace(0, 1e-10)

    bullish_pinbar = (lower_wick / body_safe > wick_ratio) & (lower_wick > upper_wick) & valid_candle
    bearish_pinbar = (upper_wick / body_safe > wick_ratio) & (upper_wick > lower_wick) & valid_candle

    result = pd.Series(0, index=df.index, dtype=np.float32)
    result[bullish_pinbar] = 1
    result[bearish_pinbar] = -1

    return result


def detect_engulfing(df: pd.DataFrame) -> pd.Series:
    """
    Detect engulfing patterns.

    Bullish engulfing: Current green candle fully engulfs previous red candle.
    Bearish engulfing: Current red candle fully engulfs previous green candle.

    Returns:
        Series with values: 1 (bullish), -1 (bearish), 0 (none)
    """
    prev_open = df['open'].shift(1)
    prev_close = df['close'].shift(1)
    prev_body_high = df[['open', 'close']].shift(1).max(axis=1)
    prev_body_low = df[['open', 'close']].shift(1).min(axis=1)

    curr_body_high = df[['open', 'close']].max(axis=1)
    curr_body_low = df[['open', 'close']].min(axis=1)

    # Previous candle direction
    prev_bearish = prev_close < prev_open
    prev_bullish = prev_close > prev_open

    # Current candle direction
    curr_bullish = df['close'] > df['open']
    curr_bearish = df['close'] < df['open']

    # Engulfing conditions
    bullish_engulfing = (
        prev_bearish &
        curr_bullish &
        (curr_body_low < prev_body_low) &
        (curr_body_high > prev_body_high)
    )

    bearish_engulfing = (
        prev_bullish &
        curr_bearish &
        (curr_body_high > prev_body_high) &
        (curr_body_low < prev_body_low)
    )

    result = pd.Series(0, index=df.index, dtype=np.float32)
    result[bullish_engulfing] = 1
    result[bearish_engulfing] = -1

    return result


def detect_doji(
    df: pd.DataFrame,
    body_ratio: float = 0.1
) -> pd.Series:
    """
    Detect doji candles (indecision candles with tiny bodies).

    Args:
        df: OHLCV DataFrame
        body_ratio: Maximum body-to-range ratio for doji

    Returns:
        Series with 1 (doji) or 0 (not doji)
    """
    body = abs(df['close'] - df['open'])
    total_range = df['high'] - df['low']

    # Avoid division by zero
    total_range = total_range.replace(0, 1e-10)

    is_doji = (body / total_range) < body_ratio

    return is_doji.astype(np.float32)


# =============================================================================
# Market Structure
# =============================================================================

def detect_fractals(
    df: pd.DataFrame,
    n: int = 5
) -> Tuple[pd.Series, pd.Series]:
    """
    Detect Williams Fractals for Support/Resistance levels.

    IMPORTANT: Uses DELAYED detection to prevent look-ahead bias!
    A fractal is only marked AFTER it's confirmed (when we have n bars after it).

    At time T, we can only know about fractals at time T-n or earlier,
    because we need n bars AFTER the fractal point to confirm it.

    A fractal high at bar i requires:
    - high[i] > high[i-1], high[i-2] (past - OK)
    - high[i] > high[i+1], high[i+2] (future - must wait for these)

    So at bar i+2, we can finally confirm the fractal at bar i.
    We mark the fractal at i+2 (current bar) with the VALUE from bar i.

    Args:
        df: OHLCV DataFrame
        n: Total window size (must be odd, e.g., 5 means 2 bars each side)

    Returns:
        Tuple of (fractal_highs, fractal_lows) as boolean Series
        NOTE: These are DELAYED - the fractal occurred n//2 bars AGO
    """
    half_n = n // 2

    fractal_highs = pd.Series(False, index=df.index)
    fractal_lows = pd.Series(False, index=df.index)

    # Start from position where we have enough PAST data to confirm a fractal
    # At position i, we're checking if position (i - half_n) was a fractal
    # This means we're only using data from [i - n + 1] to [i] (all past/current)
    for i in range(n - 1, len(df)):
        # The candidate fractal point is half_n bars AGO
        fractal_idx = i - half_n

        # Window is [fractal_idx - half_n, fractal_idx + half_n] = [i - n + 1, i]
        # All of this is past data relative to current position i
        window_start = fractal_idx - half_n
        window_end = fractal_idx + half_n + 1  # +1 for slice

        window_high = df['high'].iloc[window_start:window_end]
        window_low = df['low'].iloc[window_start:window_end]

        # Check if the candidate point (half_n bars ago) is a fractal
        candidate_high = df['high'].iloc[fractal_idx]
        candidate_low = df['low'].iloc[fractal_idx]

        # Mark at CURRENT position (i), indicating we NOW KNOW about this fractal
        # The actual S/R level is at df['high'].iloc[fractal_idx]
        # FIXED: Use tolerance-based comparison to handle float precision issues
        if (abs(candidate_high - window_high.max()) < 1e-10 and 
            candidate_high > window_high.iloc[0] and 
            candidate_high > window_high.iloc[-1]):
            fractal_highs.iloc[i] = True

        if (abs(candidate_low - window_low.min()) < 1e-10 and 
            candidate_low < window_low.iloc[0] and 
            candidate_low < window_low.iloc[-1]):
            fractal_lows.iloc[i] = True

    return fractal_highs, fractal_lows


def get_fractal_levels(
    df: pd.DataFrame,
    fractal_window: int = 5
) -> Tuple[pd.Series, pd.Series]:
    """
    Get the actual price levels of fractals (with delayed detection).

    Since fractals are detected with a delay, at position i where
    fractal_highs[i] = True, the actual fractal price is from
    position i - (fractal_window // 2).

    Returns:
        Tuple of (resistance_prices, support_prices) as Series
        NaN where no fractal was detected
    """
    half_n = fractal_window // 2
    fractal_highs, fractal_lows = detect_fractals(df, fractal_window)

    # Get the actual prices where fractals occurred
    resistance_prices = pd.Series(np.nan, index=df.index, dtype=np.float32)
    support_prices = pd.Series(np.nan, index=df.index, dtype=np.float32)

    for i in range(len(df)):
        if fractal_highs.iloc[i]:
            # The actual fractal was half_n bars ago
            actual_fractal_idx = i - half_n
            if actual_fractal_idx >= 0:
                resistance_prices.iloc[i] = df['high'].iloc[actual_fractal_idx]

        if fractal_lows.iloc[i]:
            actual_fractal_idx = i - half_n
            if actual_fractal_idx >= 0:
                support_prices.iloc[i] = df['low'].iloc[actual_fractal_idx]

    return resistance_prices, support_prices


def get_sr_levels(
    df: pd.DataFrame,
    fractal_window: int = 5,
    lookback: int = 100
) -> Tuple[List[float], List[float]]:
    """
    Get current Support and Resistance levels from recent fractals.

    Uses delayed fractal detection (no look-ahead bias).

    Returns:
        Tuple of (resistance_levels, support_levels)
    """
    resistance_prices, support_prices = get_fractal_levels(df.tail(lookback + fractal_window), fractal_window)

    resistance = resistance_prices.dropna().tolist()
    support = support_prices.dropna().tolist()

    return resistance, support


def distance_to_nearest_sr(
    price: pd.Series,
    df: pd.DataFrame,
    atr: pd.Series,
    fractal_window: int = 5,
    lookback: int = 100
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate ATR-normalized distance to nearest S/R levels.

    Uses delayed fractal detection (no look-ahead bias).

    Returns:
        Tuple of (distance_to_resistance, distance_to_support)
    """
    dist_to_r = pd.Series(np.nan, index=price.index, dtype=np.float32)
    dist_to_s = pd.Series(np.nan, index=price.index, dtype=np.float32)

    # Get fractal prices (delayed detection - no future leak)
    resistance_prices, support_prices = get_fractal_levels(df, fractal_window)

    for i in range(lookback, len(price)):
        # Get recent fractal levels (all detected BEFORE current bar)
        recent_resistances = resistance_prices.iloc[max(0, i-lookback):i].dropna()
        recent_supports = support_prices.iloc[max(0, i-lookback):i].dropna()

        current_price = price.iloc[i]
        current_atr = atr.iloc[i]

        if current_atr > 0:
            # Distance to nearest resistance (above current price)
            above_resistances = recent_resistances[recent_resistances > current_price]
            if len(above_resistances) > 0:
                dist_to_r.iloc[i] = (above_resistances.min() - current_price) / current_atr

            # Distance to nearest support (below current price)
            below_supports = recent_supports[recent_supports < current_price]
            if len(below_supports) > 0:
                dist_to_s.iloc[i] = (current_price - below_supports.max()) / current_atr

    return dist_to_r.fillna(0), dist_to_s.fillna(0)


# =============================================================================
# Trend Filters
# =============================================================================

def sma(df: pd.DataFrame, period: int = 50) -> pd.Series:
    """Simple Moving Average."""
    return df['close'].rolling(window=period).mean().astype(np.float32)


def ema(df: pd.DataFrame, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return df['close'].ewm(span=period, adjust=False).mean().astype(np.float32)


def sma_distance(
    df: pd.DataFrame,
    atr: pd.Series,
    period: int = 50
) -> pd.Series:
    """
    Calculate ATR-normalized distance from SMA(period).

    Positive = price above SMA (bullish)
    Negative = price below SMA (bearish)
    
    FIXED: Added clipping to prevent extreme values when ATR is near zero.
    """
    sma_val = sma(df, period)
    distance = (df['close'] - sma_val) / atr.replace(0, 1e-10)
    # Clip to prevent extreme values
    distance = distance.clip(-100, 100)
    return distance.astype(np.float32)


def ema_crossover(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26
) -> pd.Series:
    """
    EMA crossover signal.

    Returns: 1 (bullish cross), -1 (bearish cross), 0 (no cross)
    """
    ema_fast = ema(df, fast)
    ema_slow = ema(df, slow)

    # Current and previous relationship
    above_now = ema_fast > ema_slow
    above_prev = ema_fast.shift(1) > ema_slow.shift(1)

    result = pd.Series(0, index=df.index, dtype=np.float32)
    result[above_now & ~above_prev] = 1   # Bullish crossover
    result[~above_now & above_prev] = -1  # Bearish crossover

    return result


def ema_trend(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26
) -> pd.Series:
    """
    EMA trend direction (not just crossover).

    Returns: 1 (fast > slow), -1 (fast < slow)
    """
    ema_fast = ema(df, fast)
    ema_slow = ema(df, slow)

    result = pd.Series(0, index=df.index, dtype=np.float32)
    result[ema_fast > ema_slow] = 1
    result[ema_fast < ema_slow] = -1

    return result


# =============================================================================
# Regime Detection
# =============================================================================

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range.
    """
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)

    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_val = tr.rolling(window=period).mean()

    return atr_val.astype(np.float32)


def choppiness_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Choppiness Index (CHOP).

    Values > 61.8 indicate ranging/choppy market.
    Values < 38.2 indicate trending market.
    
    FIXED: Now uses raw True Range instead of averaged ATR.
    """
    # Calculate raw True Range (not averaged)
    high = df['high']
    low = df['low']
    close_prev = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close_prev)
    tr3 = abs(low - close_prev)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of True Range over period
    tr_sum = true_range.rolling(window=period).sum()

    high_max = df['high'].rolling(window=period).max()
    low_min = df['low'].rolling(window=period).min()

    # Prevent division by zero and log of zero
    range_diff = (high_max - low_min).replace(0, 1e-10)
    ratio = tr_sum / range_diff
    # CRITICAL FIX: clip to 1.0 (not 1e-10) to prevent negative CHOP values
    # When ratio < 1, log10(ratio) < 0, producing invalid negative CHOP
    # CHOP should always be 0-100, so ratio must be >= 1.0
    ratio = ratio.clip(lower=1.0)

    chop = 100 * np.log10(ratio) / np.log10(period)

    return chop.astype(np.float32)


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average Directional Index (ADX).

    Values > 25 indicate trending market.
    Values < 20 indicate ranging market.
    
    FIXED: Corrected pandas assignment bug for +DM/-DM calculation.
    """
    high = df['high']
    low = df['low']
    close = df['close']

    # +DM and -DM
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    # FIXED: Use np.where for proper conditional assignment
    plus_dm_values = np.where(
        (up_move > down_move) & (up_move > 0),
        up_move,
        0.0
    )
    minus_dm_values = np.where(
        (down_move > up_move) & (down_move > 0),
        down_move,
        0.0
    )
    
    plus_dm = pd.Series(plus_dm_values, index=df.index, dtype=np.float32)
    minus_dm = pd.Series(minus_dm_values, index=df.index, dtype=np.float32)

    # True Range
    atr_val = atr(df, period)

    # Smoothed +DI and -DI using Wilder's smoothing (alpha = 1/period)
    # Standard ADX uses Wilder's EMA, NOT span-based EMA
    # Wilder's: alpha = 1/N ≈ 0.071 for N=14
    # Span-based: alpha = 2/(N+1) ≈ 0.133 for N=14 (too reactive)
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_val.replace(0, 1e-10))
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_val.replace(0, 1e-10))

    # DX and ADX (also using Wilder's smoothing)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_val = dx.ewm(alpha=1/period, adjust=False).mean()

    return adx_val.astype(np.float32)


def market_regime(
    df: pd.DataFrame,
    chop_period: int = 14,
    adx_period: int = 14
) -> pd.Series:
    """
    Classify market regime based on CHOP and ADX.

    Returns:
        1: Trending (tradeable)
        0: Neutral
        -1: Ranging/Choppy (avoid)
    """
    chop = choppiness_index(df, chop_period)
    adx_val = adx(df, adx_period)

    result = pd.Series(0, index=df.index, dtype=np.float32)

    # Trending: low chop + high ADX (both conditions)
    trending = (chop < 38.2) & (adx_val > 25)
    # Ranging: high chop + low ADX
    ranging = (chop > 61.8) & (adx_val < 20)

    result[trending] = 1
    result[ranging] = -1

    return result


def detect_market_regime_direction(
    df: pd.DataFrame,
    lookback: int = 20,
    trend_threshold: float = 0.10,
    chop_period: int = 14,
    adx_period: int = 14
) -> pd.Series:
    """
    Classify market regime into BULLISH, BEARISH, or RANGING.

    This is crucial for regime-balanced training to prevent directional bias.
    If training data is predominantly bearish, the agent learns to short.
    By balancing across regimes, we ensure the agent learns both directions.

    Args:
        df: OHLCV DataFrame
        lookback: Period for trend calculation
        trend_threshold: Threshold for trend classification (in ATR units)
        chop_period: Period for choppiness calculation
        adx_period: Period for ADX calculation

    Returns:
        Series with values:
            1: BULLISH (uptrend)
            0: RANGING (sideways/choppy)
           -1: BEARISH (downtrend)
    """
    # Calculate trend direction using price change over lookback
    price_change = df['close'].diff(lookback)

    # Calculate ATR for normalization
    atr_val = atr(df, lookback)
    atr_val = atr_val.replace(0, 1e-10)  # Avoid division by zero

    # Normalized price change (in ATR units)
    normalized_change = price_change / (atr_val * lookback)

    # Get regime indicators
    chop = choppiness_index(df, chop_period)
    adx_val = adx(df, adx_period)

    # Initialize as ranging
    result = pd.Series(0, index=df.index, dtype=np.float32)

    # Bullish: price went up significantly AND not too choppy
    bullish = (normalized_change > trend_threshold) & (chop < 55)

    # Bearish: price went down significantly AND not too choppy
    bearish = (normalized_change < -trend_threshold) & (chop < 55)

    # If very choppy (CHOP > 60), force to ranging regardless of direction
    very_choppy = chop > 60

    result[bullish] = 1
    result[bearish] = -1
    result[very_choppy] = 0  # Override to ranging if very choppy

    return result


def compute_regime_labels(
    df: pd.DataFrame,
    lookback: int = 20
) -> np.ndarray:
    """
    Compute regime labels for regime-balanced sampling.

    Args:
        df: DataFrame with OHLCV data
        lookback: Period for regime detection

    Returns:
        Array of regime labels: 0=BULLISH, 1=RANGING, 2=BEARISH
        (Converted to 0,1,2 for use as class indices in stratified sampling)
    """
    regime = detect_market_regime_direction(df, lookback)

    # Convert to 0, 1, 2 for stratified sampling
    # -1 (bearish) -> 2
    #  0 (ranging) -> 1
    #  1 (bullish) -> 0
    labels = np.where(regime == 1, 0, np.where(regime == 0, 1, 2))

    return labels.astype(np.int32)


# =============================================================================
# Complete Feature Engineering
# =============================================================================

def engineer_all_features(
    df: pd.DataFrame,
    config: Optional[dict] = None,
    include_sr_features: bool = True
) -> pd.DataFrame:
    """
    Apply all feature engineering to OHLCV DataFrame.
    
    IMPORTANT: Does NOT drop NaN rows - alignment is handled by the pipeline.
    This preserves index alignment between timeframes.

    Args:
        df: OHLCV DataFrame
        config: Optional configuration dict
        include_sr_features: Whether to include S/R distance features (slower)

    Returns:
        DataFrame with all features added (may contain NaN at edges)
    """
    if config is None:
        config = {
            'pinbar_wick_ratio': 2.0,
            'doji_body_ratio': 0.1,
            'fractal_window': 5,
            'sr_lookback': 100,
            'sma_period': 50,
            'ema_fast': 12,
            'ema_slow': 26,
            'chop_period': 14,
            'adx_period': 14,
            'atr_period': 14
        }

    result = df.copy()
    logger.info("Engineering features...")

    # ATR (needed for other features)
    result['atr'] = atr(df, config['atr_period'])

    # Price Action Patterns
    result['pinbar'] = detect_pinbar(df, config['pinbar_wick_ratio'])
    result['engulfing'] = detect_engulfing(df)
    result['doji'] = detect_doji(df, config['doji_body_ratio'])

    # Trend Filters
    result['sma_distance'] = sma_distance(df, result['atr'], config['sma_period'])
    result['ema_crossover'] = ema_crossover(df, config['ema_fast'], config['ema_slow'])
    result['ema_trend'] = ema_trend(df, config['ema_fast'], config['ema_slow'])

    # Regime Detection
    result['chop'] = choppiness_index(df, config['chop_period'])
    result['adx'] = adx(df, config['adx_period'])
    result['regime'] = market_regime(df, config['chop_period'], config['adx_period'])

    # S/R Distance Features (market structure signals)
    if include_sr_features:
        logger.info("  Computing S/R distance features...")
        dist_to_r, dist_to_s = distance_to_nearest_sr(
            df['close'], df, result['atr'],
            fractal_window=config['fractal_window'],
            lookback=config['sr_lookback']
        )
        result['dist_to_resistance'] = dist_to_r.clip(-50, 50).astype(np.float32)
        result['dist_to_support'] = dist_to_s.clip(-50, 50).astype(np.float32)

    # Market Sessions
    result = add_market_sessions(result)

    # Structure Features (BOS/CHoCH)
    f_high, f_low = detect_fractals(df, n=config['fractal_window'])
    struct_df = detect_structure_breaks(df, f_high, f_low, n=config['fractal_window'])
    for col in struct_df.columns:
        result[col] = struct_df[col]

    # Returns (for normalization and targets)
    result['returns'] = df['close'].pct_change().astype(np.float32)

    # Volatility
    result['volatility'] = result['returns'].rolling(20).std().astype(np.float32)

    # DO NOT drop NaN rows - alignment is handled by the pipeline
    # This preserves index alignment between timeframes
    nan_count = result.isna().any(axis=1).sum()
    logger.info(f"Features complete. {len(result):,} rows, {nan_count:,} rows with NaN (will be aligned in pipeline)")

    # Ensure all float32
    for col in result.columns:
        if result[col].dtype == np.float64:
            result[col] = result[col].astype(np.float32)

    return result


def create_smoothed_target(
    df: pd.DataFrame,
    future_window: int = 24,
    smooth_window: int = 24,
    scale_factor: float = 100.0
) -> pd.Series:
    """
    Create the smoothed future return target for Analyst training.

    Target = ((smoothed future close / current close) - 1) * scale_factor

    This teaches the model sustained momentum, not noise.
    
    FIXED: 
    - Added min_periods=1 to reduce NaN data loss at edges.
    - Scale by 100 to get PERCENTAGE returns (prevents mode collapse to near-zero)
    
    Without scaling, targets are ~0.0001 and the model learns trivial solution
    of always predicting 0. With scale_factor=100, targets are ~0.01 (1% moves)
    which forces the model to learn real patterns.

    Args:
        df: DataFrame with 'close' column
        future_window: How many candles ahead
        smooth_window: Rolling window for smoothing
        scale_factor: Multiply returns by this (100 = percentage returns)

    Returns:
        Series of target values (in percentage if scale_factor=100)
    """
    # Use min_periods=1 to reduce NaN values at the edges
    future_smoothed = df['close'].shift(-future_window).rolling(smooth_window, min_periods=1).mean()
    target = (future_smoothed / df['close']) - 1
    
    # Scale to percentage returns to prevent mode collapse
    # Without this, targets are ~0.0001 and model predicts ~0 for everything
    target = target * scale_factor

    return target.astype(np.float32)


def create_return_classes(
    target: pd.Series,
    class_std_thresholds: Tuple = (-0.5, 0.5),
    train_end_idx: Optional[int] = None
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Convert a continuous smoothed-return target into discrete classes.

    Supports both 3-class and 5-class schemes based on threshold count:

    3-class scheme (2 thresholds: down_thresh, up_thresh):
        0: Down      (< down_thresh * std)
        1: Neutral   [down_thresh * std, up_thresh * std]
        2: Up        (> up_thresh * std)

    5-class scheme (4 thresholds: strong_down, weak_down, weak_up, strong_up):
        0: Strong Down   (< strong_down * std)
        1: Weak Down     [strong_down * std, weak_down * std)
        2: Neutral       [weak_down * std, weak_up * std]
        3: Weak Up       (weak_up * std, strong_up * std]
        4: Strong Up     (> strong_up * std)

    Args:
        target: Smoothed return series (already scaled)
        class_std_thresholds: Multipliers of target std that define boundaries
                             2 values for 3-class, 4 values for 5-class

    Returns:
        Tuple of (class labels Series with NaNs preserved, metadata dict)
    """
    # IMPORTANT: To prevent look-ahead bias, compute std on TRAINING portion only
    if train_end_idx is not None:
        std_source = target.iloc[:train_end_idx].dropna()
    else:
        std_source = target.dropna()

    target_std = float(std_source.std())
    num_thresholds = len(class_std_thresholds)

    if num_thresholds == 2:
        # 3-class scheme: Down / Neutral / Up
        down_thresh = class_std_thresholds[0] * target_std
        up_thresh = class_std_thresholds[1] * target_std

        def _assign_class(value: float) -> float:
            if pd.isna(value):
                return np.nan
            if value < down_thresh:
                return 0  # Down
            if value <= up_thresh:
                return 1  # Neutral
            return 2  # Up

        meta = {
            'target_std': target_std,
            'num_classes': 3,
            'down_threshold': down_thresh,
            'up_threshold': up_thresh,
            # Legacy keys for compatibility
            'strong_down_threshold': down_thresh,
            'weak_down_threshold': down_thresh,
            'weak_up_threshold': up_thresh,
            'strong_up_threshold': up_thresh
        }

    elif num_thresholds == 4:
        # 5-class scheme: Strong Down / Weak Down / Neutral / Weak Up / Strong Up
        boundaries = {
            'strong_down': class_std_thresholds[0] * target_std,
            'weak_down': class_std_thresholds[1] * target_std,
            'weak_up': class_std_thresholds[2] * target_std,
            'strong_up': class_std_thresholds[3] * target_std
        }

        def _assign_class(value: float) -> float:
            if pd.isna(value):
                return np.nan
            if value < boundaries['strong_down']:
                return 0
            if value < boundaries['weak_down']:
                return 1
            if value <= boundaries['weak_up']:
                return 2
            if value <= boundaries['strong_up']:
                return 3
            return 4

        meta = {
            'target_std': target_std,
            'num_classes': 5,
            'strong_down_threshold': boundaries['strong_down'],
            'weak_down_threshold': boundaries['weak_down'],
            'weak_up_threshold': boundaries['weak_up'],
            'strong_up_threshold': boundaries['strong_up']
        }
    else:
        raise ValueError(f"class_std_thresholds must have 2 or 4 values, got {num_thresholds}")

    labels = target.apply(_assign_class).astype(np.float32)

    return labels, meta


def create_binary_direction_target(
    df: pd.DataFrame,
    future_window: int = 16,
    smooth_window: int = 12,
    min_move_atr: float = 0.3,
    atr_period: int = 14
) -> Tuple[pd.Series, pd.Series, Dict[str, float]]:
    """
    Create binary Up/Down labels, excluding weak/neutral moves.

    This addresses the directional confusion problem where the model achieves
    71% recall on Neutral but only ~50% on Up/Down. By excluding weak moves,
    we train only on clear directional signals.

    Args:
        df: DataFrame with 'close' column (and optionally 'atr')
        future_window: How many candles ahead (16 = 4 hours on 15m)
        smooth_window: Rolling window for smoothing
        min_move_atr: Minimum move in ATR units to count as directional
        atr_period: ATR calculation period if 'atr' not in df

    Returns:
        Tuple of:
            - labels: 0=Down, 1=Up (NaN for excluded neutral moves)
            - valid_mask: Boolean mask for training samples
            - meta: Metadata dict with thresholds and stats
    """
    # Calculate smoothed future return
    future_smoothed = df['close'].shift(-future_window).rolling(
        smooth_window, min_periods=1
    ).mean()
    future_return = (future_smoothed / df['close']) - 1

    # Get or calculate ATR
    if 'atr' in df.columns:
        atr = df['atr']
    else:
        # Calculate ATR from price
        high_low = df['high'] - df['low'] if 'high' in df.columns else df['close'].diff().abs()
        atr = high_low.rolling(atr_period, min_periods=1).mean()

    # Normalize ATR to percentage terms (like future_return)
    atr_pct = atr / df['close']

    # Threshold: at least min_move_atr * ATR move
    threshold = min_move_atr * atr_pct

    # Create labels: 0=Down, 1=Up, NaN=Excluded (neutral/weak)
    labels = pd.Series(index=df.index, dtype=np.float32)
    labels[:] = np.nan  # Start all as NaN (excluded)

    down_mask = future_return < -threshold
    up_mask = future_return > threshold

    labels[down_mask] = 0  # Down
    labels[up_mask] = 1    # Up
    # Neutral moves (between -threshold and +threshold) remain NaN

    # Valid mask for filtering dataset
    valid_mask = ~labels.isna()

    # Metadata
    n_down = down_mask.sum()
    n_up = up_mask.sum()
    n_neutral = len(df) - n_down - n_up
    n_valid = valid_mask.sum()

    meta = {
        'num_classes': 2,
        'min_move_atr': min_move_atr,
        'future_window': future_window,
        'smooth_window': smooth_window,
        'n_down': int(n_down),
        'n_up': int(n_up),
        'n_neutral_excluded': int(n_neutral),
        'n_valid': int(n_valid),
        'pct_excluded': float(n_neutral / len(df) * 100),
        'class_balance': float(n_up / (n_down + n_up)) if (n_down + n_up) > 0 else 0.5
    }

    logger.info(
        f"Binary direction target: {n_down} Down, {n_up} Up, "
        f"{n_neutral} excluded ({meta['pct_excluded']:.1f}%)"
    )

    return labels, valid_mask, meta


def create_auxiliary_targets(
    df: pd.DataFrame,
    future_window: int = 16,
    atr_period: int = 14,
    adx_threshold: float = 25.0
) -> Tuple[pd.Series, pd.Series]:
    """
    Create auxiliary targets for multi-task learning:
    1. Volatility target: Future ATR / Current ATR (regression)
    2. Regime target: Trending (1) vs Ranging (0) based on ADX

    These auxiliary losses provide regularization and help the model
    learn better representations.

    Args:
        df: DataFrame with OHLC data
        future_window: How many candles ahead
        atr_period: ATR calculation period
        adx_threshold: ADX value above which market is "trending"

    Returns:
        Tuple of (volatility_target, regime_target)
    """
    # Calculate ATR
    if 'atr' in df.columns:
        atr = df['atr']
    else:
        high_low = df['high'] - df['low'] if 'high' in df.columns else df['close'].diff().abs()
        atr = high_low.rolling(atr_period, min_periods=1).mean()

    # Volatility target: Future ATR / Current ATR
    future_atr = atr.shift(-future_window)
    volatility_target = (future_atr / atr).fillna(1.0).astype(np.float32)
    # Clip extreme values
    volatility_target = volatility_target.clip(0.5, 2.0)

    # Regime target: 1 if ADX > threshold (trending), else 0
    if 'adx' in df.columns:
        regime_target = (df['adx'] > adx_threshold).astype(np.float32)
    else:
        # Simple proxy: use ATR percentile
        atr_rolling_pct = atr.rolling(100, min_periods=20).apply(
            lambda x: (x[-1] > np.percentile(x, 60)).astype(float),
            raw=True
        )
        regime_target = atr_rolling_pct.fillna(0.0).astype(np.float32)

    return volatility_target, regime_target


def create_multi_horizon_targets(
    df: pd.DataFrame,
    horizons: Dict[str, int] = None,
    smooth_window: int = 4,
    min_move_atr: float = 0.3,
    atr_period: int = 14
) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series], Dict[str, Dict]]:
    """
    Create binary direction targets at MULTIPLE time horizons.

    This addresses the Target Mismatch problem by training on multiple horizons:
    - Short horizons (1H) are easier to predict and provide "stepping stones"
    - The model learns features useful across multiple time scales
    - Acts as implicit regularization against overfitting to noisy 4H target

    Args:
        df: DataFrame with 'close' column (and optionally 'atr')
        horizons: Dict mapping horizon name to future_window (e.g., {'1h': 4, '2h': 8, '4h': 16})
                  All windows are in 15-minute candles
        smooth_window: Rolling window for smoothing (shared across horizons)
        min_move_atr: Minimum move in ATR units to count as directional
        atr_period: ATR calculation period if 'atr' not in df

    Returns:
        Tuple of:
            - labels_dict: Dict[horizon_name, labels_series] (0=Down, 1=Up, NaN=excluded)
            - valid_masks_dict: Dict[horizon_name, valid_mask_series]
            - meta_dict: Dict[horizon_name, metadata_dict]
    """
    if horizons is None:
        # Default: 1H, 2H, 4H horizons (in 15-minute candles)
        horizons = {
            '1h': 4,    # 4 × 15min = 1 hour
            '2h': 8,    # 8 × 15min = 2 hours
            '4h': 16    # 16 × 15min = 4 hours (primary target)
        }

    # Get or calculate ATR once
    if 'atr' in df.columns:
        atr = df['atr']
    else:
        high_low = df['high'] - df['low'] if 'high' in df.columns else df['close'].diff().abs()
        atr = high_low.rolling(atr_period, min_periods=1).mean()

    # Normalize ATR to percentage terms
    atr_pct = atr / df['close']

    labels_dict = {}
    valid_masks_dict = {}
    meta_dict = {}

    for horizon_name, future_window in horizons.items():
        # Adjust smooth window based on horizon (shorter horizon = less smoothing)
        # This prevents over-smoothing short horizons
        adjusted_smooth = min(smooth_window, max(2, future_window // 2))

        # Calculate smoothed future return for this horizon
        future_smoothed = df['close'].shift(-future_window).rolling(
            adjusted_smooth, min_periods=1
        ).mean()
        future_return = (future_smoothed / df['close']) - 1

        # Threshold: at least min_move_atr * ATR move
        threshold = min_move_atr * atr_pct

        # Create labels: 0=Down, 1=Up, NaN=Excluded
        labels = pd.Series(index=df.index, dtype=np.float32)
        labels[:] = np.nan

        down_mask = future_return < -threshold
        up_mask = future_return > threshold

        labels[down_mask] = 0  # Down
        labels[up_mask] = 1    # Up

        # Valid mask
        valid_mask = ~labels.isna()

        # Metadata
        n_down = down_mask.sum()
        n_up = up_mask.sum()
        n_neutral = len(df) - n_down - n_up

        meta = {
            'horizon_name': horizon_name,
            'future_window': future_window,
            'adjusted_smooth_window': adjusted_smooth,
            'n_down': int(n_down),
            'n_up': int(n_up),
            'n_neutral_excluded': int(n_neutral),
            'n_valid': int(valid_mask.sum()),
            'pct_excluded': float(n_neutral / len(df) * 100),
            'class_balance': float(n_up / (n_down + n_up)) if (n_down + n_up) > 0 else 0.5
        }

        labels_dict[horizon_name] = labels
        valid_masks_dict[horizon_name] = valid_mask
        meta_dict[horizon_name] = meta

        logger.info(
            f"Multi-horizon target [{horizon_name}]: {n_down} Down, {n_up} Up, "
            f"{n_neutral} excluded ({meta['pct_excluded']:.1f}%)"
        )

    return labels_dict, valid_masks_dict, meta_dict


# =============================================================================
# Market Sessions
# =============================================================================

def add_market_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add market session flags (London, NY, Asian).
    
    Assumes index is DatetimeIndex in UTC.
    
    Sessions (approx UTC):
    - Asian: 00:00 - 09:00
    - London: 08:00 - 17:00
    - NY: 13:00 - 22:00
    
    Args:
        df: DataFrame with DatetimeIndex
        
    Returns:
        DataFrame with added session columns
    """
    df = df.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("DataFrame index is not DatetimeIndex. Skipping session features.")
        return df
        
    hours = df.index.hour
    
    # Asian Session (Tokyo/Sydney): ~00:00 to 09:00 UTC
    df['session_asian'] = ((hours >= 0) & (hours < 9)).astype(int)
    
    # London Session: ~08:00 to 17:00 UTC
    df['session_london'] = ((hours >= 8) & (hours < 17)).astype(int)
    
    # New York Session: ~13:00 to 22:00 UTC
    df['session_ny'] = ((hours >= 13) & (hours < 22)).astype(int)
    
    return df


def detect_structure_breaks(
    df: pd.DataFrame,
    fractal_highs: pd.Series,
    fractal_lows: pd.Series,
    n: int = 5
) -> pd.DataFrame:
    """
    Detect Break of Structure (BOS) and Change of Character (CHoCH).
    
    Logic:
    - Track most recent confirmed Swing High/Low (from fractals).
    - Track current Structure Trend (Bullish/Bearish).
    - BOS: Continuation break (e.g. Bullish Trend + Break High).
    - CHoCH: Reversal break (e.g. Bearish Trend + Break High).
    
    Args:
        df: OHLCV DataFrame
        fractal_highs: Boolean series of confirmed fractal highs
        fractal_lows: Boolean series of confirmed fractal lows
        n: Window size used for fractal detection (to find price level)
        
    Returns:
        DataFrame with columns: bos_bullish, bos_bearish, choch_bullish, choch_bearish
    """
    # Initialize result columns
    results = pd.DataFrame(0, index=df.index, columns=[
        'bos_bullish', 'bos_bearish', 'choch_bullish', 'choch_bearish'
    ], dtype=np.int32)
    
    # State
    last_high = np.nan
    last_low = np.nan
    last_high_broken = False
    last_low_broken = False
    
    # Trend: 1=Bullish, -1=Bearish, 0=Unknown
    trend = 0
    
    # Half window to find the actual fractal candle
    half_n = n // 2
    
    # Iterate
    # We can't vectorise easily because trend state depends on previous breaks
    for i in range(len(df)):
        # 1. Update Structure Points if a NEW fractal is confirmed NOW
        if fractal_highs.iloc[i]:
            # The fractal was at i - half_n
            # But wait, fractal_highs[i] is True means we JUST confirmed it.
            # So the level is valid from now on.
            idx = i - half_n
            if idx >= 0:
                last_high = df['high'].iloc[idx]
                last_high_broken = False # Reset break status for new level
        
        if fractal_lows.iloc[i]:
            idx = i - half_n
            if idx >= 0:
                last_low = df['low'].iloc[idx]
                last_low_broken = False # Reset break status for new level
                
        # 2. Check for Breaks
        close = df['close'].iloc[i]
        
        # Break High
        if not np.isnan(last_high) and not last_high_broken and close > last_high:
            last_high_broken = True
            
            if trend == 1:
                results.iloc[i, results.columns.get_loc('bos_bullish')] = 1
            elif trend == -1:
                results.iloc[i, results.columns.get_loc('choch_bullish')] = 1
                trend = 1 # Reversal to Bullish
            else:
                trend = 1 # Initialize
                
        # Break Low
        if not np.isnan(last_low) and not last_low_broken and close < last_low:
            last_low_broken = True
            
            if trend == -1:
                results.iloc[i, results.columns.get_loc('bos_bearish')] = 1
            elif trend == 1:
                results.iloc[i, results.columns.get_loc('choch_bearish')] = 1
                trend = -1 # Reversal to Bearish
            else:
                trend = -1 # Initialize
                
    return results


def get_feature_columns(include_ohlcv: bool = False) -> List[str]:
    """Get list of feature column names."""
    features = [
        'atr', 'pinbar', 'engulfing', 'doji',
        'sma_distance', 'ema_crossover', 'ema_trend',
        'chop', 'adx', 'regime', 'returns', 'volatility',
        'session_asian', 'session_london', 'session_ny',
        'bos_bullish', 'bos_bearish', 'choch_bullish', 'choch_bearish'
    ]
    if include_ohlcv:
        features = ['open', 'high', 'low', 'close'] + features
    return features
