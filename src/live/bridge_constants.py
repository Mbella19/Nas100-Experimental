"""
Constants shared by the MT5 live bridge.

Keep this module lightweight: no heavy imports (SB3/matplotlib/etc).
"""

from __future__ import annotations

# Ordered Analyst/Agent model input features (per timeframe).
# IMPORTANT: Ordering matters and must match the ordering used during training.
MODEL_FEATURE_COLS: list[str] = [
    "returns",
    "volatility",
    "pinbar",
    "engulfing",
    "doji",
    "ema_trend",
    "ema_crossover",
    "regime",
    "sma_distance",
    "dist_to_resistance",
    "dist_to_support",
    "session_asian",
    "session_london",
    "session_ny",
    "bos_bullish",
    "bos_bearish",
    "choch_bullish",
    "choch_bearish",
]

# Market feature columns used in the RL observation.
MARKET_FEATURE_COLS: list[str] = [
    "atr",
    "chop",
    "adx",
    "regime",
    "sma_distance",
    "dist_to_support",
    "dist_to_resistance",
    "session_asian",
    "session_london",
    "session_ny",
    "bos_bullish",
    "bos_bearish",
    "choch_bullish",
    "choch_bearish",
]

# TradingEnv size mapping (must match TradingEnv.POSITION_SIZES)
POSITION_SIZES: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
