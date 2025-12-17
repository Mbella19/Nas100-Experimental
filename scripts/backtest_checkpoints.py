#!/usr/bin/env python3
"""
Run out-of-sample (OOS) backtests across all saved PPO checkpoints and
compile a Markdown report.

Default behavior mirrors `scripts/run_pipeline.py` backtest logic:
- Uses normalized processed data in `data/processed/*_normalized.parquet`
- Uses the last 15% of samples for OOS evaluation (after windowing)
- Uses the same TradingEnv + Backtester loop (`src/evaluation/backtest.run_backtest`)

Example:
  python scripts/backtest_checkpoints.py
  python scripts/backtest_checkpoints.py --min-confidence 0.6
  python scripts/backtest_checkpoints.py --limit 50
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

# Ensure project root is on the import path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Config, get_device  # noqa: E402
from src.agents.sniper_agent import SniperAgent  # noqa: E402
from src.evaluation.backtest import calculate_buy_and_hold, run_backtest  # noqa: E402
from src.live.bridge_constants import MODEL_FEATURE_COLS  # noqa: E402
from src.models.analyst import load_analyst  # noqa: E402
from src.training.precompute_analyst import load_cached_analyst_outputs  # noqa: E402
from src.training.train_agent import create_trading_env, prepare_env_data  # noqa: E402
from src.utils.logging_config import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)


@dataclass(frozen=True)
class BacktestDataset:
    env: Any  # Monitor-wrapped TradingEnv
    env_unwrapped: Any  # TradingEnv
    close_prices_test: np.ndarray
    timestamps_test: np.ndarray
    ohlc_test: Optional[np.ndarray]
    bh_return_pct: float
    bh_final_balance: float
    test_start_ts: pd.Timestamp
    test_end_ts: pd.Timestamp


def _parse_checkpoint_steps(path: Path) -> Optional[int]:
    match = re.search(r"_(\d+)_steps\.zip$", path.name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _discover_checkpoints(search_roots: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    checkpoints: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.zip"):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            checkpoints.append(resolved)

    def _sort_key(p: Path) -> tuple[int, str]:
        steps = _parse_checkpoint_steps(p)
        # Sort unknown-step zips last.
        return (steps if steps is not None else 10**18, p.name)

    checkpoints.sort(key=_sort_key)
    return checkpoints


def _load_backtest_dataset(
    cfg: Config,
    *,
    min_action_confidence: float,
) -> BacktestDataset:
    processed = cfg.paths.data_processed
    df_5m = pd.read_parquet(processed / "features_5m_normalized.parquet")
    df_15m = pd.read_parquet(processed / "features_15m_normalized.parquet")
    df_45m = pd.read_parquet(processed / "features_45m_normalized.parquet")

    required_feature_cols = list(MODEL_FEATURE_COLS)
    for col in required_feature_cols:
        if col not in df_5m.columns:
            df_5m[col] = 0.0
        if col not in df_15m.columns:
            df_15m[col] = 0.0
        if col not in df_45m.columns:
            df_45m[col] = 0.0

    lookback_5m = int(cfg.analyst.lookback_5m)
    lookback_15m = int(cfg.analyst.lookback_15m)
    lookback_45m = int(cfg.analyst.lookback_45m)

    data_5m, data_15m, data_45m, close_prices, market_features, returns = prepare_env_data(
        df_5m,
        df_15m,
        df_45m,
        required_feature_cols,
        lookback_5m,
        lookback_15m,
        lookback_45m,
    )

    subsample_15m = 3
    subsample_45m = 9
    start_idx = max(
        lookback_5m,
        (lookback_15m - 1) * subsample_15m + 1,
        (lookback_45m - 1) * subsample_45m + 1,
    )
    n_samples = len(close_prices)

    if not isinstance(df_5m.index, pd.DatetimeIndex):
        raise ValueError("Expected DatetimeIndex in processed 5m data for timestamps.")

    timestamps_all = (df_5m.index[start_idx:start_idx + n_samples].astype("int64") // 10**9).values

    ohlc_all: Optional[np.ndarray] = None
    if all(c in df_5m.columns for c in ("open", "high", "low", "close")):
        ohlc_all = (
            df_5m[["open", "high", "low", "close"]]
            .values[start_idx:start_idx + n_samples]
            .astype(np.float32)
        )

    test_start = int(0.85 * len(close_prices))

    train_market = market_features[:test_start]
    market_feat_mean = train_market.mean(axis=0).astype(np.float32)
    market_feat_std = train_market.std(axis=0).astype(np.float32)
    market_feat_std = np.where(market_feat_std > 1e-8, market_feat_std, 1.0).astype(np.float32)

    test_data = (
        data_5m[test_start:],
        data_15m[test_start:],
        data_45m[test_start:],
        close_prices[test_start:],
        market_features[test_start:],
    )
    test_returns = returns[test_start:] if returns is not None else None
    timestamps_test = timestamps_all[test_start:]
    ohlc_test = ohlc_all[test_start:] if ohlc_all is not None else None

    # Load Analyst (context_dim/num_classes must match agent training).
    feature_dims = {"5m": len(required_feature_cols), "15m": len(required_feature_cols), "45m": len(required_feature_cols)}
    analyst_path = cfg.paths.models_analyst / "best.pt"
    analyst = load_analyst(str(analyst_path), feature_dims, device=cfg.device, freeze=True)

    # Prefer cached Analyst outputs to avoid recomputation overhead.
    precomputed_cache = None
    cache_path = processed / "analyst_cache.npz"
    if cache_path.exists():
        try:
            full_cache = load_cached_analyst_outputs(str(cache_path))
            if len(full_cache["contexts"]) == len(close_prices) and len(full_cache["probs"]) == len(close_prices):
                precomputed_cache = {
                    "contexts": full_cache["contexts"][test_start:],
                    "probs": full_cache["probs"][test_start:],
                }
            else:
                logger.warning(
                    "Analyst cache shape mismatch: contexts=%s probs=%s expected=%s; ignoring cache.",
                    getattr(full_cache.get("contexts"), "shape", None),
                    getattr(full_cache.get("probs"), "shape", None),
                    (len(close_prices),),
                )
        except Exception as e:
            logger.warning("Failed to load analyst_cache.npz (%s); falling back to on-the-fly Analyst.", e)

    # Disable noise for backtesting.
    cfg.trading.noise_level = 0.0

    env = create_trading_env(
        *test_data,
        analyst_model=analyst,
        config=cfg.trading,
        device=cfg.device,
        market_feat_mean=market_feat_mean,
        market_feat_std=market_feat_std,
        returns=test_returns,
        ohlc_data=ohlc_test,
        timestamps=timestamps_test,
        precomputed_analyst_cache=precomputed_cache,
    )

    from stable_baselines3.common.monitor import Monitor

    monitor_env = Monitor(env)

    # Buy-and-hold baseline (same window).
    bh_equity, bh_metrics = calculate_buy_and_hold(close_prices[test_start:], initial_balance=10000.0)
    bh_return_pct = float(bh_metrics.get("total_return_pct", 0.0))
    bh_final_balance = float(bh_metrics.get("final_balance", bh_equity[-1] if len(bh_equity) else 10000.0))

    test_start_ts = pd.to_datetime(int(timestamps_test[0]), unit="s")
    test_end_ts = pd.to_datetime(int(timestamps_test[-1]), unit="s")

    logger.info(
        "OOS dataset ready | bars=%d | %s → %s | min_conf=%.2f",
        len(close_prices[test_start:]),
        test_start_ts,
        test_end_ts,
        min_action_confidence,
    )

    return BacktestDataset(
        env=monitor_env,
        env_unwrapped=env,
        close_prices_test=close_prices[test_start:],
        timestamps_test=timestamps_test,
        ohlc_test=ohlc_test,
        bh_return_pct=bh_return_pct,
        bh_final_balance=bh_final_balance,
        test_start_ts=test_start_ts,
        test_end_ts=test_end_ts,
    )


def _format_float(value: Any, digits: int = 2) -> str:
    try:
        f = float(value)
    except Exception:
        return "n/a"
    return f"{f:.{digits}f}"


def _load_state(state_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if not state_path.exists():
        return rows, failures

    for line in state_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue

        rec_type = rec.get("type")
        data = rec.get("data", rec)
        if rec_type == "success":
            rows.append(data)
        elif rec_type == "failure":
            failures.append(data)

    return rows, failures


def _append_state(state_path: Path, rec_type: str, data: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"type": rec_type, "data": data}, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest all PPO checkpoints (OOS) and write a Markdown report.")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Agent confidence threshold for backtest (0.0 disables).")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of checkpoints (0 = no cap).")
    parser.add_argument("--output", type=str, default=None, help="Output .md path (default: results/checkpoint_oos_report_*.md)")
    parser.add_argument("--state", type=str, default=None, help="Optional JSONL state path for resume/progress (default: output with .jsonl suffix).")
    parser.add_argument("--resume", action="store_true", help="Resume from existing --state file (skip completed checkpoints).")
    parser.add_argument("--log-dir", type=str, default=None, help="Optional log directory")
    args = parser.parse_args()

    if args.log_dir:
        setup_logging(args.log_dir, name=__name__)

    # Reduce per-step noise (and speed up large checkpoint sweeps).
    logging.getLogger("src.evaluation.backtest").setLevel(logging.WARNING)

    cfg = Config()
    cfg.device = get_device()

    search_roots = [
        cfg.paths.models_agent / "checkpoints",
        cfg.paths.base_dir / "models" / "checkpoints",
        cfg.paths.models_agent,
    ]
    checkpoints = _discover_checkpoints(search_roots)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under: {', '.join(str(p) for p in search_roots)}")

    if args.limit and args.limit > 0:
        checkpoints = checkpoints[: int(args.limit)]

    dataset = _load_backtest_dataset(cfg, min_action_confidence=float(args.min_confidence))

    out_path = Path(args.output) if args.output else (cfg.paths.base_dir / "results" / f"checkpoint_oos_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    state_path = Path(args.state) if args.state else out_path.with_suffix(".jsonl")
    if state_path.exists() and not args.resume:
        raise FileExistsError(
            f"State file exists: {state_path} (pass --resume or choose a different --state/--output)"
        )

    spread_total = float(cfg.trading.spread_pips) + float(cfg.trading.slippage_pips)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if args.resume and state_path.exists():
        rows, failures = _load_state(state_path)

    completed: set[str] = set()
    for r in rows:
        ckpt_key = r.get("checkpoint_abs") or r.get("checkpoint")
        if ckpt_key:
            completed.add(str(ckpt_key))
    for f in failures:
        ckpt_key = f.get("checkpoint_abs") or f.get("checkpoint")
        if ckpt_key:
            completed.add(str(ckpt_key))

    t0 = time.time()
    remaining = [c for c in checkpoints if str(c) not in completed]
    logger.info(
        "Evaluating %d checkpoint(s)... (resume=%s | done=%d | remaining=%d | state=%s)",
        len(checkpoints),
        bool(args.resume),
        len(completed),
        len(remaining),
        state_path,
    )

    for i, ckpt in enumerate(checkpoints, start=1):
        ckpt_abs = str(ckpt)
        if ckpt_abs in completed:
            continue
        checkpoint_display = (
            str(ckpt.relative_to(cfg.paths.base_dir))
            if ckpt.is_relative_to(cfg.paths.base_dir)
            else ckpt_abs
        )
        steps = _parse_checkpoint_steps(ckpt)
        label = f"{steps:,}" if steps is not None else ckpt.name
        logger.info("[%d/%d] Backtesting %s", i, len(checkpoints), label)

        start = time.time()
        try:
            agent = SniperAgent.load(str(ckpt), dataset.env, device="cpu")
            result = run_backtest(
                agent=agent,
                env=dataset.env_unwrapped,
                initial_balance=10000.0,
                deterministic=True,
                min_action_confidence=float(args.min_confidence),
                spread_pips=spread_total,
                sl_atr_multiplier=float(cfg.trading.sl_atr_multiplier),
                tp_atr_multiplier=float(cfg.trading.tp_atr_multiplier),
                use_stop_loss=bool(cfg.trading.use_stop_loss),
                use_take_profit=bool(cfg.trading.use_take_profit),
                min_hold_bars=int(cfg.trading.min_hold_bars),
            )
            elapsed = time.time() - start
            metrics = dict(result.metrics)
            total_return = float(metrics.get("total_return_pct", 0.0))
            beats_bh = total_return > float(dataset.bh_return_pct)

            row = {
                "checkpoint": checkpoint_display,
                "checkpoint_abs": ckpt_abs,
                "steps": steps,
                "elapsed_sec": elapsed,
                "beats_bh": beats_bh,
                **metrics,
            }
            rows.append(row)
            _append_state(state_path, "success", row)
            completed.add(ckpt_abs)
        except Exception as e:
            elapsed = time.time() - start
            failure = {
                "checkpoint": checkpoint_display,
                "checkpoint_abs": ckpt_abs,
                "steps": steps,
                "elapsed_sec": elapsed,
                "error": str(e),
            }
            failures.append(failure)
            _append_state(state_path, "failure", failure)
            completed.add(ckpt_abs)
        finally:
            # Aggressive cleanup between checkpoints (598+ loads).
            try:
                del agent  # type: ignore[name-defined]
            except Exception:
                pass
            try:
                del result  # type: ignore[name-defined]
            except Exception:
                pass
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    total_elapsed = time.time() - t0

    # Sort rows by steps ascending (unknown steps last).
    rows.sort(key=lambda r: (r.get("steps") if r.get("steps") is not None else 10**18))

    # Summaries
    best_by_net = max(rows, key=lambda r: float(r.get("net_pnl_cash", float("-inf"))), default=None)
    best_by_sortino = max(rows, key=lambda r: float(r.get("sortino_ratio", float("-inf"))), default=None)

    header = [
        f"# OOS Checkpoint Backtest Report ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})",
        "",
        "## Setup",
        f"- OOS window: {dataset.test_start_ts} → {dataset.test_end_ts} ({len(dataset.close_prices_test):,} x 5m bars)",
        f"- Min hold bars: {int(cfg.trading.min_hold_bars)}",
        f"- SL/TP: {float(cfg.trading.sl_atr_multiplier)}x / {float(cfg.trading.tp_atr_multiplier)}x ATR",
        f"- Spread+slippage (backtest): {spread_total:.2f} points",
        f"- Confidence threshold (backtest): {float(args.min_confidence):.2f}",
        f"- Buy & Hold: return={dataset.bh_return_pct:.2f}% final=${dataset.bh_final_balance:,.2f}",
        f"- State: `{state_path}`",
        f"- Checkpoints evaluated: {len(checkpoints)} | successes: {len(rows)} | failures: {len(failures)} | wall time: {total_elapsed/60:.1f} min",
        "",
        "## Best",
    ]

    if best_by_net:
        header.append(
            f"- Best net PnL: `{best_by_net['checkpoint']}` | return={_format_float(best_by_net.get('total_return_pct'))}% | net=${_format_float(best_by_net.get('net_pnl_cash'))} | maxDD={_format_float(best_by_net.get('max_drawdown_pct'))}%"
        )
    if best_by_sortino:
        header.append(
            f"- Best Sortino: `{best_by_sortino['checkpoint']}` | sortino={_format_float(best_by_sortino.get('sortino_ratio'))} | return={_format_float(best_by_sortino.get('total_return_pct'))}% | maxDD={_format_float(best_by_sortino.get('max_drawdown_pct'))}%"
        )

    header.append("")
    header.append("## Results")
    header.append(
        "| steps | checkpoint | return% | net$ | maxDD% | sharpe | sortino | win% | PF | trades | avgDur(bars) | beats B&H | sec |"
    )
    header.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|---:|")

    lines = list(header)

    for r in rows:
        steps = r.get("steps")
        steps_str = f"{int(steps):,}" if isinstance(steps, int) else "n/a"
        lines.append(
            "| "
            + " | ".join(
                [
                    steps_str,
                    f"`{r.get('checkpoint')}`",
                    _format_float(r.get("total_return_pct")),
                    _format_float(r.get("net_pnl_cash")),
                    _format_float(r.get("max_drawdown_pct")),
                    _format_float(r.get("sharpe_ratio")),
                    _format_float(r.get("sortino_ratio")),
                    _format_float(r.get("win_rate_pct")),
                    _format_float(r.get("profit_factor")),
                    str(int(r.get("total_trades", 0))),
                    _format_float(r.get("avg_trade_duration_bars")),
                    "✓" if r.get("beats_bh") else "✗",
                    _format_float(r.get("elapsed_sec"), digits=1),
                ]
            )
            + " |"
        )

    if failures:
        lines.append("")
        lines.append("## Failures")
        for f in failures:
            steps = f.get("steps")
            steps_str = f"{int(steps):,}" if isinstance(steps, int) else "n/a"
            lines.append(f"- {steps_str} | `{f.get('checkpoint')}` | {f.get('error')}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote report: %s", out_path)


if __name__ == "__main__":
    # Make matplotlib/fontconfig happy if any dependency imports it.
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()
