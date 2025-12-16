#!/usr/bin/env python3
"""
Prepare deployment artifacts needed for MT5 live inference.

Creates:
- `models/agent/market_feat_stats.npz`

This file stores the market-feature mean/std computed on the same TRAINING portion
used during agent training (first 85% of samples after windowing). The live bridge
uses these stats to normalize `market_features` exactly like TradingEnv does.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def _load_parquet(path: Path):
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"Expected DatetimeIndex in {path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MT5 bridge deployment artifacts")
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args()

    import numpy as np

    from config.settings import Config
    from src.live.bridge_constants import MARKET_FEATURE_COLS
    from src.utils.logging_config import setup_logging, get_logger

    logger = get_logger(__name__)

    if args.log_dir:
        setup_logging(args.log_dir, name=__name__)

    cfg = Config()

    df_5m_path = cfg.paths.data_processed / "features_5m_normalized.parquet"
    df_5m = _load_parquet(df_5m_path)

    lookback_5m = int(cfg.analyst.lookback_5m)
    lookback_15m = int(cfg.analyst.lookback_15m)
    lookback_45m = int(cfg.analyst.lookback_45m)
    subsample_15m = 3
    subsample_45m = 9

    start_idx = max(
        lookback_5m,
        (lookback_15m - 1) * subsample_15m + 1,
        (lookback_45m - 1) * subsample_45m + 1,
    )
    n_samples = len(df_5m) - start_idx
    if n_samples <= 0:
        raise ValueError("Not enough rows in normalized 5m data to build stats.")

    for col in MARKET_FEATURE_COLS:
        if col not in df_5m.columns:
            logger.warning("Missing market feature '%s' in %s (filling 0.0)", col, df_5m_path)
            df_5m[col] = 0.0

    market_features = df_5m[MARKET_FEATURE_COLS].values[start_idx:start_idx + n_samples].astype(np.float32)
    split_idx = int(0.85 * len(market_features))
    train_market = market_features[:split_idx]

    mean = train_market.mean(axis=0).astype(np.float32)
    std = train_market.std(axis=0).astype(np.float32)
    std = np.where(std > 1e-8, std, 1.0).astype(np.float32)

    out_path = cfg.paths.models_agent / "market_feat_stats.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, cols=np.array(MARKET_FEATURE_COLS), mean=mean, std=std)

    logger.info("Saved %s", out_path)
    logger.info("cols=%s", MARKET_FEATURE_COLS)
    logger.info("mean=%s", mean.tolist())
    logger.info("std=%s", std.tolist())


if __name__ == "__main__":
    main()
