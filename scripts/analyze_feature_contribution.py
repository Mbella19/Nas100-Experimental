#!/usr/bin/env python3
"""
Feature contribution / noise diagnostics for the Hybrid NAS100 Trading System.

This script is designed to answer:
1) Which engineered features appear to contribute very little to the Analyst
   (and therefore likely contribute little to the Agent via the Analyst context)?
2) Which features look sparse / redundant / heavily clipped (potential noise)?

It produces:
- CSVs with raw + normalized feature statistics
- Correlation (redundancy) report on the Analyst training subset
- Permutation importance for the trained Analyst checkpoint (per timeframe + all-timeframes)

Notes:
- Uses the *existing* processed parquet files in `data/processed/`.
- Uses the *existing* trained Analyst checkpoint in `models/analyst/best.pt`.
- Does not retrain anything.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Ensure project imports work when running as a script.
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Config
from src.data.features import create_binary_direction_target
from src.models.analyst import load_analyst


os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))


ANALYST_FEATURE_COLS_DEFAULT: List[str] = [
    # Price dynamics
    "returns",
    "volatility",
    # Price action patterns
    "pinbar",
    "engulfing",
    "doji",
    # Trend indicators
    "ema_trend",
    "ema_crossover",
    # Regime / trend filters
    "regime",
    "sma_distance",
    # Market structure
    "dist_to_resistance",
    "dist_to_support",
    # Sessions
    "session_asian",
    "session_london",
    "session_ny",
    # BOS / CHoCH
    "bos_bullish",
    "bos_bearish",
    "choch_bullish",
    "choch_bearish",
]


ENV_MARKET_COLS_DEFAULT: List[str] = [
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


@dataclass(frozen=True)
class AnalystLabelSpec:
    future_window: int
    smooth_window: int
    min_move_atr: float


@dataclass(frozen=True)
class Lookbacks:
    lookback_5m: int
    lookback_15m: int
    lookback_45m: int
    subsample_15m: int = 3
    subsample_45m: int = 9

    @property
    def start_idx(self) -> int:
        # Match Analyst training dataset safety margin.
        return int(
            max(
                self.lookback_5m,
                self.lookback_15m * self.subsample_15m,
                self.lookback_45m * self.subsample_45m,
            )
        )


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _balanced_accuracy_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    return float(0.5 * (tpr + tnr))


def _bce_loss(y_true: np.ndarray, p_up: np.ndarray, eps: float = 1e-7) -> float:
    y_true = y_true.astype(np.float32)
    p = np.clip(p_up.astype(np.float32), eps, 1.0 - eps)
    return float(-(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)).mean())


def _predict_p_up(
    model: torch.nn.Module,
    x_5m: np.ndarray,
    x_15m: np.ndarray,
    x_45m: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    n = x_5m.shape[0]
    p_up = np.empty((n,), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            t5 = torch.tensor(x_5m[start:end], device=device, dtype=torch.float32)
            t15 = torch.tensor(x_15m[start:end], device=device, dtype=torch.float32)
            t45 = torch.tensor(x_45m[start:end], device=device, dtype=torch.float32)

            # TCNAnalyst.get_probabilities returns (context, probs, _)
            res = model.get_probabilities(t5, t15, t45)
            probs = res[1] if isinstance(res, (tuple, list)) and len(res) >= 2 else res
            probs_np = probs.detach().cpu().numpy().astype(np.float32)

            if probs_np.shape[1] < 2:
                raise ValueError(f"Expected probs with 2 columns, got shape {probs_np.shape}")
            p_up[start:end] = probs_np[:, 1]
    return p_up


def _build_windows(
    features_5m: np.ndarray,
    features_15m: np.ndarray,
    features_45m: np.ndarray,
    sample_indices: np.ndarray,
    lookbacks: Lookbacks,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(sample_indices.shape[0])
    n_features = int(features_5m.shape[1])

    x_5m = np.empty((n, lookbacks.lookback_5m, n_features), dtype=np.float32)
    x_15m = np.empty((n, lookbacks.lookback_15m, n_features), dtype=np.float32)
    x_45m = np.empty((n, lookbacks.lookback_45m, n_features), dtype=np.float32)

    for i, idx in enumerate(sample_indices.tolist()):
        idx_i = int(idx)

        x_5m[i] = features_5m[idx_i - lookbacks.lookback_5m + 1 : idx_i + 1]

        idx_range_15m = list(
            range(
                idx_i - (lookbacks.lookback_15m - 1) * lookbacks.subsample_15m,
                idx_i + 1,
                lookbacks.subsample_15m,
            )
        )
        x_15m[i] = features_15m[idx_range_15m]

        idx_range_45m = list(
            range(
                idx_i - (lookbacks.lookback_45m - 1) * lookbacks.subsample_45m,
                idx_i + 1,
                lookbacks.subsample_45m,
            )
        )
        x_45m[i] = features_45m[idx_range_45m]

    return x_5m, x_15m, x_45m


def _feature_stats(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    cols = [c for c in columns if c in df.columns]
    out_rows: List[Dict[str, object]] = []

    for col in cols:
        s = df[col]
        values = s.to_numpy()
        non_na = values[~pd.isna(values)]
        n = len(values)
        n_non_na = len(non_na)
        zero_frac = float((non_na == 0).mean()) if n_non_na else 0.0
        nan_frac = float((pd.isna(values)).mean()) if n else 0.0

        unique_count = int(pd.Series(non_na).nunique(dropna=False)) if n_non_na else 0

        # Clipping fraction (for normalized features clipped to [-5, 5])
        clip_frac = float((np.abs(non_na) >= 4.999).mean()) if n_non_na else 0.0

        out_rows.append(
            {
                "feature": col,
                "dtype": str(s.dtype),
                "count": int(n),
                "count_non_na": int(n_non_na),
                "nan_frac": nan_frac,
                "zero_frac": zero_frac,
                "unique": unique_count,
                "mean": float(np.mean(non_na)) if n_non_na else np.nan,
                "std": float(np.std(non_na)) if n_non_na else np.nan,
                "min": float(np.min(non_na)) if n_non_na else np.nan,
                "max": float(np.max(non_na)) if n_non_na else np.nan,
                "clip_frac_abs_ge_5": clip_frac,
            }
        )

    return pd.DataFrame(out_rows).sort_values(["zero_frac", "nan_frac"], ascending=[False, False])


def _high_corr_pairs(
    df: pd.DataFrame,
    columns: Sequence[str],
    threshold: float,
) -> pd.DataFrame:
    cols = [c for c in columns if c in df.columns]
    if len(cols) < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "corr", "abs_corr"])

    corr = df[cols].corr(method="pearson")
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs: List[Dict[str, object]] = []

    abs_vals = np.abs(upper.values)
    hits = np.argwhere(abs_vals >= threshold)
    for i, j in hits.tolist():
        fa = str(upper.index[i])
        fb = str(upper.columns[j])
        c = float(upper.iat[i, j])
        pairs.append({"feature_a": fa, "feature_b": fb, "corr": c, "abs_corr": abs(c)})

    if not pairs:
        return pd.DataFrame(columns=["feature_a", "feature_b", "corr", "abs_corr"])

    return pd.DataFrame(pairs).sort_values("abs_corr", ascending=False)


def _permutation_importance_analyst(
    model: torch.nn.Module,
    x_5m: np.ndarray,
    x_15m: np.ndarray,
    x_45m: np.ndarray,
    y_true: np.ndarray,
    feature_names: Sequence[str],
    batch_size: int,
    device: torch.device,
    n_repeats: int,
    seed: int,
    include_per_timeframe: bool,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_samples = int(x_5m.shape[0])
    n_features = int(x_5m.shape[2])

    if n_features != len(feature_names):
        raise ValueError(
            f"feature_names length ({len(feature_names)}) != n_features ({n_features})"
        )

    base_p_up = _predict_p_up(model, x_5m, x_15m, x_45m, batch_size, device)
    base_pred = (base_p_up >= 0.5).astype(np.int32)
    base_bal_acc = _balanced_accuracy_binary(y_true, base_pred)
    base_bce = _bce_loss(y_true, base_p_up)

    records: List[Dict[str, object]] = []

    def _eval_with_arrays(xx5: np.ndarray, xx15: np.ndarray, xx45: np.ndarray) -> Tuple[float, float]:
        p_up = _predict_p_up(model, xx5, xx15, xx45, batch_size, device)
        pred = (p_up >= 0.5).astype(np.int32)
        return _balanced_accuracy_binary(y_true, pred), _bce_loss(y_true, p_up)

    for feat_idx, feat_name in enumerate(feature_names):
        # Permute this feature across ALL timeframes together.
        deltas_bal: List[float] = []
        deltas_bce: List[float] = []
        for _ in range(n_repeats):
            perm = rng.permutation(n_samples)

            x5p = x_5m.copy()
            x15p = x_15m.copy()
            x45p = x_45m.copy()

            x5p[:, :, feat_idx] = x_5m[perm, :, feat_idx]
            x15p[:, :, feat_idx] = x_15m[perm, :, feat_idx]
            x45p[:, :, feat_idx] = x_45m[perm, :, feat_idx]

            bal_acc, bce = _eval_with_arrays(x5p, x15p, x45p)
            deltas_bal.append(base_bal_acc - bal_acc)
            deltas_bce.append(bce - base_bce)

        records.append(
            {
                "scope": "all_timeframes",
                "timeframe": "all",
                "feature": feat_name,
                "delta_bal_acc_mean": float(np.mean(deltas_bal)),
                "delta_bal_acc_std": float(np.std(deltas_bal)),
                "delta_bce_mean": float(np.mean(deltas_bce)),
                "delta_bce_std": float(np.std(deltas_bce)),
            }
        )

    if include_per_timeframe:
        # Per-timeframe permutation importance (more expensive).
        for timeframe in ("5m", "15m", "45m"):
            for feat_idx, feat_name in enumerate(feature_names):
                deltas_bal = []
                deltas_bce = []
                for _ in range(n_repeats):
                    perm = rng.permutation(n_samples)

                    if timeframe == "5m":
                        x5p = x_5m.copy()
                        x5p[:, :, feat_idx] = x_5m[perm, :, feat_idx]
                        bal_acc, bce = _eval_with_arrays(x5p, x_15m, x_45m)
                    elif timeframe == "15m":
                        x15p = x_15m.copy()
                        x15p[:, :, feat_idx] = x_15m[perm, :, feat_idx]
                        bal_acc, bce = _eval_with_arrays(x_5m, x15p, x_45m)
                    else:
                        x45p = x_45m.copy()
                        x45p[:, :, feat_idx] = x_45m[perm, :, feat_idx]
                        bal_acc, bce = _eval_with_arrays(x_5m, x_15m, x45p)

                    deltas_bal.append(base_bal_acc - bal_acc)
                    deltas_bce.append(bce - base_bce)

                records.append(
                    {
                        "scope": "single_timeframe",
                        "timeframe": timeframe,
                        "feature": feat_name,
                        "delta_bal_acc_mean": float(np.mean(deltas_bal)),
                        "delta_bal_acc_std": float(np.std(deltas_bal)),
                        "delta_bce_mean": float(np.mean(deltas_bce)),
                        "delta_bce_std": float(np.std(deltas_bce)),
                    }
                )

    df = pd.DataFrame(records)
    df = df.sort_values(["scope", "delta_bce_mean"], ascending=[True, False])
    df.attrs["baseline_bal_acc"] = base_bal_acc
    df.attrs["baseline_bce"] = base_bce
    return df


def _format_float(x: float, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "nan"
    return f"{x:.{digits}f}"


def _write_summary_md(
    out_path: Path,
    baseline_bal_acc: float,
    baseline_bce: float,
    importance_df: pd.DataFrame,
    stats_raw: pd.DataFrame,
    stats_norm: pd.DataFrame,
    corr_pairs: pd.DataFrame,
    agent_group_sensitivity: Optional[pd.DataFrame] = None,
    agent_feature_sensitivity: Optional[pd.DataFrame] = None,
    agent_returns_stats: Optional[Dict[str, float]] = None,
) -> None:
    lines: List[str] = []
    lines.append("# Feature Contribution Report")
    lines.append("")
    lines.append("## Analyst Baseline (validation subset used for importance)")
    lines.append(f"- Balanced accuracy: `{_format_float(baseline_bal_acc, 4)}`")
    lines.append(f"- BCE loss: `{_format_float(baseline_bce, 4)}`")
    lines.append("")

    # Global importance ranking (all timeframes)
    global_df = importance_df[importance_df["scope"] == "all_timeframes"].copy()
    global_df = global_df.sort_values("delta_bce_mean", ascending=False)

    lines.append("## Analyst Permutation Importance (all timeframes together)")
    lines.append("Higher Δ means the feature mattered more; near-zero Δ suggests low contribution.")
    lines.append("")
    lines.append("| Rank | Feature | ΔBCE (mean±std) | ΔBalAcc (mean±std) |")
    lines.append("|---:|---|---:|---:|")
    for i, row in enumerate(global_df.head(18).itertuples(index=False), start=1):
        lines.append(
            f"| {i} | `{row.feature}` | "
            f"{_format_float(row.delta_bce_mean, 4)}±{_format_float(row.delta_bce_std, 4)} | "
            f"{_format_float(row.delta_bal_acc_mean, 4)}±{_format_float(row.delta_bal_acc_std, 4)} |"
        )
    lines.append("")

    low_df = global_df.sort_values("delta_bce_mean", ascending=True).head(8)
    lines.append("## Lowest-Contribution Candidates (all-timeframe permutation)")
    lines.append("| Feature | ΔBCE | ΔBalAcc |")
    lines.append("|---|---:|---:|")
    for row in low_df.itertuples(index=False):
        lines.append(
            f"| `{row.feature}` | {_format_float(row.delta_bce_mean, 4)} | {_format_float(row.delta_bal_acc_mean, 4)} |"
        )
    lines.append("")

    # Sparsity view from raw stats
    if not stats_raw.empty:
        raw_sel = stats_raw.set_index("feature")
        lines.append("## Raw Feature Sparsity / Clipping Signals")
        lines.append("`zero_frac` is on raw engineered values (before z-score).")
        lines.append("")
        lines.append("| Feature | zero_frac | mean | std | min | max |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for feat in low_df["feature"].tolist():
            if feat in raw_sel.index:
                r = raw_sel.loc[feat]
                lines.append(
                    f"| `{feat}` | {_format_float(float(r['zero_frac']), 4)} | "
                    f"{_format_float(float(r['mean']), 4)} | {_format_float(float(r['std']), 4)} | "
                    f"{_format_float(float(r['min']), 4)} | {_format_float(float(r['max']), 4)} |"
                )
        lines.append("")

    if not corr_pairs.empty:
        lines.append("## High-Correlation (Redundancy) Pairs")
        lines.append("These pairs have `|corr| >= threshold` on the Analyst training subset.")
        lines.append("")
        lines.append("| feature_a | feature_b | corr |")
        lines.append("|---|---|---:|")
        for row in corr_pairs.head(20).itertuples(index=False):
            lines.append(f"| `{row.feature_a}` | `{row.feature_b}` | {_format_float(row.corr, 4)} |")
        lines.append("")

    if agent_group_sensitivity is not None and not agent_group_sensitivity.empty:
        lines.append("## Agent Observation Sensitivity (policy action-change permutation)")
        lines.append("Fraction of actions that change when shuffling each observation block across a sampled batch.")
        lines.append("")
        lines.append("| Block | Any change | Direction change | Size change |")
        lines.append("|---|---:|---:|---:|")
        for row in agent_group_sensitivity.itertuples(index=False):
            lines.append(
                f"| `{row.name}` | {_format_float(row.any_changed, 3)} | "
                f"{_format_float(row.direction_changed, 3)} | {_format_float(row.size_changed, 3)} |"
            )
        lines.append("")

    if agent_feature_sensitivity is not None and not agent_feature_sensitivity.empty:
        # Market features (top/bottom)
        mkt = agent_feature_sensitivity[agent_feature_sensitivity["scope"] == "market_feature"].copy()
        if not mkt.empty:
            mkt = mkt.sort_values("any_changed", ascending=False)
            lines.append("## Agent Market-Feature Sensitivity (per-feature)")
            lines.append("")
            lines.append("| Rank | Feature | Any change | Direction change | Size change |")
            lines.append("|---:|---|---:|---:|---:|")
            for i, row in enumerate(mkt.head(10).itertuples(index=False), start=1):
                lines.append(
                    f"| {i} | `{row.name}` | {_format_float(row.any_changed, 3)} | "
                    f"{_format_float(row.direction_changed, 3)} | {_format_float(row.size_changed, 3)} |"
                )
            lines.append("")
            lines.append("### Lowest-Sensitivity Market Features")
            lines.append("| Feature | Any change |")
            lines.append("|---|---:|")
            for row in mkt.tail(8).sort_values("any_changed", ascending=True).itertuples(index=False):
                lines.append(f"| `{row.name}` | {_format_float(row.any_changed, 3)} |")
            lines.append("")

    if agent_returns_stats is not None:
        lines.append("## Agent Returns-Slice Scale (as seen by policy)")
        lines.append(
            f"- mean `{_format_float(agent_returns_stats.get('mean', np.nan), 3)}`, "
            f"std `{_format_float(agent_returns_stats.get('std', np.nan), 3)}`, "
            f"min `{_format_float(agent_returns_stats.get('min', np.nan), 3)}`, "
            f"max `{_format_float(agent_returns_stats.get('max', np.nan), 3)}`"
        )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze feature contribution/noise.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Directory containing processed parquet files.",
    )
    parser.add_argument(
        "--analyst-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "models" / "analyst" / "best.pt",
        help="Path to trained Analyst checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "feature_audit",
        help="Directory to write report artifacts.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--corr-threshold", type=float, default=0.95)
    parser.add_argument(
        "--include-per-timeframe",
        action="store_true",
        help="Also compute per-timeframe permutation importance (3x slower).",
    )
    parser.add_argument(
        "--agent-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "models" / "agent" / "checkpoints" / "sniper_model_400000_steps.zip",
        help="Optional PPO checkpoint for agent-side sensitivity analysis.",
    )
    parser.add_argument(
        "--analyst-cache",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "analyst_cache.npz",
        help="Precomputed Analyst cache used to build realistic agent observations.",
    )
    parser.add_argument(
        "--agent-samples",
        type=int,
        default=4000,
        help="Number of observations to sample for agent-side sensitivity.",
    )
    parser.add_argument(
        "--skip-agent",
        action="store_true",
        help="Skip agent-side sensitivity analysis even if checkpoints exist.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu"],
        help="Inference device for the Analyst importance run (CPU recommended).",
    )
    return parser.parse_args()


def _agent_observation_sensitivity(
    df5_norm: pd.DataFrame,
    market_cols: Sequence[str],
    agent_checkpoint: Path,
    analyst_cache_path: Path,
    market_feat_mean: np.ndarray,
    market_feat_std: np.ndarray,
    agent_lookback_window: int,
    seed: int,
    n_samples: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Compute agent-side sensitivity by permuting observation columns and measuring
    fraction of action changes under the current policy.

    This is a policy-usage diagnostic (not a training-time gradient attribution),
    but it reliably flags observation components the policy is effectively ignoring.
    """
    from src.environments.trading_env import TradingEnv
    from src.agents.sniper_agent import SniperAgent

    cache = np.load(str(analyst_cache_path), allow_pickle=True)
    cache_start_idx = int(cache["start_idx"])
    contexts = cache["contexts"].astype(np.float32)
    probs = cache["probs"].astype(np.float32)

    # Trim arrays to align with analyst_cache (cache already starts after warmup).
    close_prices = df5_norm["close"].to_numpy(dtype=np.float32)[cache_start_idx:]
    ohlc = df5_norm[["open", "high", "low", "close"]].to_numpy(dtype=np.float32)[cache_start_idx:]
    returns = df5_norm["returns"].to_numpy(dtype=np.float32)[cache_start_idx:]
    market_features = df5_norm[list(market_cols)].to_numpy(dtype=np.float32)[cache_start_idx:]

    if contexts.shape[0] != close_prices.shape[0]:
        raise RuntimeError(
            "Analyst cache length mismatch vs processed parquet. "
            f"contexts={contexts.shape[0]} close_prices={close_prices.shape[0]} "
            f"(cache_start_idx={cache_start_idx})"
        )

    # Build env using precomputed analyst cache (matches training setup).
    dummy = np.zeros((len(close_prices), 1, 1), dtype=np.float32)
    env = TradingEnv(
        data_5m=dummy,
        data_15m=dummy,
        data_45m=dummy,
        close_prices=close_prices,
        market_features=market_features,
        analyst_model=None,
        context_dim=int(contexts.shape[1]),
        market_feat_mean=market_feat_mean.astype(np.float32),
        market_feat_std=market_feat_std.astype(np.float32),
        pre_windowed=True,
        use_regime_sampling=False,
        regime_labels=None,
        noise_level=0.0,
        use_sparse_rewards=True,
        ohlc_data=ohlc,
        returns=returns,
        agent_lookback_window=int(agent_lookback_window),
        precomputed_analyst_cache={"contexts": contexts, "probs": probs},
    )

    agent = SniperAgent.load(str(agent_checkpoint), env)

    rng = np.random.default_rng(seed)

    # Sample observations by rolling the environment with the current policy.
    obs_list: List[np.ndarray] = []
    obs, _ = env.reset(seed=seed)
    for _ in range(int(n_samples)):
        obs_list.append(np.asarray(obs, dtype=np.float32).copy())
        action, _ = agent.model.predict(obs, deterministic=True)
        obs, _, done, truncated, _ = env.step(action)
        if done or truncated:
            obs, _ = env.reset()

    obs_batch = np.stack(obs_list).astype(np.float32)
    base_actions, _ = agent.model.predict(obs_batch, deterministic=True)
    base_actions = np.asarray(base_actions)

    def _action_change_stats(new_actions: np.ndarray) -> Tuple[float, float, float]:
        new_actions = np.asarray(new_actions)
        any_changed = float((new_actions != base_actions).any(axis=1).mean())
        direction_changed = float((new_actions[:, 0] != base_actions[:, 0]).mean())
        size_changed = float((new_actions[:, 1] != base_actions[:, 1]).mean())
        return any_changed, direction_changed, size_changed

    def _permute_cols(col_indices: Sequence[int]) -> Tuple[float, float, float]:
        perm = rng.permutation(obs_batch.shape[0])
        x = obs_batch.copy()
        cols = np.asarray(col_indices, dtype=np.int64)
        x[:, cols] = x[perm][:, cols]
        acts, _ = agent.model.predict(x, deterministic=True)
        return _action_change_stats(np.asarray(acts))

    # Compute block boundaries (must match TradingEnv._get_observation layout).
    context_dim = int(contexts.shape[1])
    pos_dim = 3
    n_market = int(market_features.shape[1])
    analyst_metrics_dim = 5  # binary
    sl_tp_dim = 2
    returns_dim = int(agent_lookback_window)
    obs_dim = int(obs_batch.shape[1])

    expected_obs_dim = context_dim + pos_dim + n_market + analyst_metrics_dim + sl_tp_dim + returns_dim
    if obs_dim != expected_obs_dim:
        raise RuntimeError(f"Obs dim mismatch: obs={obs_dim}, expected={expected_obs_dim}")

    market_start = context_dim + pos_dim
    market_end = market_start + n_market
    analyst_start = market_end
    analyst_end = analyst_start + analyst_metrics_dim
    sltp_start = analyst_end
    sltp_end = sltp_start + sl_tp_dim
    returns_start = sltp_end
    returns_end = returns_start + returns_dim

    group_rows = []
    for name, cols in [
        ("context", range(0, context_dim)),
        ("position_state", range(context_dim, context_dim + pos_dim)),
        ("market_features", range(market_start, market_end)),
        ("analyst_metrics", range(analyst_start, analyst_end)),
        ("sl_tp_dist", range(sltp_start, sltp_end)),
        ("returns_slice", range(returns_start, returns_end)),
    ]:
        any_c, dir_c, size_c = _permute_cols(list(cols))
        group_rows.append(
            {"scope": "group", "name": name, "any_changed": any_c, "direction_changed": dir_c, "size_changed": size_c}
        )

    group_df = pd.DataFrame(group_rows).sort_values("any_changed", ascending=False)

    feature_rows = []
    # Per-market feature
    for j, name in enumerate(list(market_cols)):
        idx = market_start + j
        any_c, dir_c, size_c = _permute_cols([idx])
        feature_rows.append(
            {
                "scope": "market_feature",
                "name": name,
                "any_changed": any_c,
                "direction_changed": dir_c,
                "size_changed": size_c,
            }
        )
    # Per-returns feature (label oldest..newest)
    for k in range(returns_dim):
        idx = returns_start + k
        lag = (returns_dim - 1) - k
        any_c, dir_c, size_c = _permute_cols([idx])
        feature_rows.append(
            {
                "scope": "returns_dim",
                "name": f"returns_t-{lag}",
                "any_changed": any_c,
                "direction_changed": dir_c,
                "size_changed": size_c,
            }
        )

    feature_df = pd.DataFrame(feature_rows).sort_values(["scope", "any_changed"], ascending=[True, False])

    rs = obs_batch[:, returns_start:returns_end]
    returns_stats = {
        "mean": float(rs.mean()),
        "std": float(rs.std()),
        "min": float(rs.min()),
        "max": float(rs.max()),
    }

    return group_df, feature_df, returns_stats


def main() -> int:
    args = _parse_args()
    out_dir: Path = args.output_dir
    _ensure_dir(out_dir)

    config = Config()
    label_spec = AnalystLabelSpec(
        future_window=config.analyst.future_window,
        smooth_window=config.analyst.smooth_window,
        min_move_atr=config.analyst.min_move_atr_threshold,
    )
    lookbacks = Lookbacks(
        lookback_5m=config.analyst.lookback_5m,
        lookback_15m=config.analyst.lookback_15m,
        lookback_45m=config.analyst.lookback_45m,
    )

    feature_cols = list(ANALYST_FEATURE_COLS_DEFAULT)

    # Load minimal columns needed for labels + Analyst features.
    cols_5m_norm = sorted(
        set(
            feature_cols
            + ["open", "high", "low", "close", "returns"]
            + ENV_MARKET_COLS_DEFAULT
        )
    )
    df5_norm = pd.read_parquet(
        args.processed_dir / "features_5m_normalized.parquet",
        columns=cols_5m_norm,
    )
    df15_norm = pd.read_parquet(
        args.processed_dir / "features_15m_normalized.parquet",
        columns=feature_cols,
    )
    df45_norm = pd.read_parquet(
        args.processed_dir / "features_45m_normalized.parquet",
        columns=feature_cols,
    )

    # Raw (pre-normalization) for interpretable sparsity stats.
    df5_raw = pd.read_parquet(args.processed_dir / "features_5m.parquet", columns=sorted(set(feature_cols)))

    # Labels (binary): Up/Down, excluding neutral.
    labels, valid_mask, meta = create_binary_direction_target(
        df5_norm,
        future_window=label_spec.future_window,
        smooth_window=label_spec.smooth_window,
        min_move_atr=label_spec.min_move_atr,
    )
    valid_indices = np.where(valid_mask.to_numpy(dtype=bool))[0]
    valid_indices = valid_indices[valid_indices >= lookbacks.start_idx]

    if valid_indices.size < 200:
        raise RuntimeError(
            f"Too few valid samples after filtering ({valid_indices.size}). "
            "Lower AnalystConfig.min_move_atr_threshold or use more data."
        )

    # Chronological split matching Analyst training: 85% / 15% over valid samples.
    train_count = int(0.85 * valid_indices.size)
    val_indices = valid_indices[train_count:]

    # Build window tensors for validation subset (used for importance).
    f5 = df5_norm[feature_cols].to_numpy(dtype=np.float32)
    f15 = df15_norm[feature_cols].to_numpy(dtype=np.float32)
    f45 = df45_norm[feature_cols].to_numpy(dtype=np.float32)

    x5_val, x15_val, x45_val = _build_windows(f5, f15, f45, val_indices, lookbacks)
    y_val = labels.iloc[val_indices].to_numpy(dtype=np.float32).astype(np.int32)

    # Load Analyst checkpoint.
    device = torch.device(args.device)
    feature_dims = {"5m": len(feature_cols), "15m": len(feature_cols), "45m": len(feature_cols)}
    analyst = load_analyst(str(args.analyst_checkpoint), feature_dims, device=device, freeze=True)

    # Analyst permutation importance.
    imp_df = _permutation_importance_analyst(
        model=analyst,
        x_5m=x5_val,
        x_15m=x15_val,
        x_45m=x45_val,
        y_true=y_val,
        feature_names=feature_cols,
        batch_size=args.batch_size,
        device=device,
        n_repeats=max(1, int(args.n_repeats)),
        seed=int(args.seed),
        include_per_timeframe=bool(args.include_per_timeframe),
    )

    baseline_bal_acc = float(imp_df.attrs.get("baseline_bal_acc", np.nan))
    baseline_bce = float(imp_df.attrs.get("baseline_bce", np.nan))

    # Feature stats (raw + normalized) on full dataset for scanability.
    stats_cols = sorted(set(feature_cols + ENV_MARKET_COLS_DEFAULT))
    stats_raw = _feature_stats(df5_raw, stats_cols)
    stats_norm = _feature_stats(df5_norm, stats_cols)

    # Correlation pairs on the Analyst TRAIN subset (valid_indices[:train_count]).
    train_rows = valid_indices[:train_count]
    corr_source = df5_norm.iloc[train_rows][feature_cols]
    corr_pairs = _high_corr_pairs(corr_source, feature_cols, threshold=float(args.corr_threshold))

    # Agent-side sensitivity (optional).
    agent_group_df: Optional[pd.DataFrame] = None
    agent_feature_df: Optional[pd.DataFrame] = None
    agent_returns_stats: Optional[Dict[str, float]] = None

    # Build the env market feature mean/std like train_agent.py does (fit on 85% train slice).
    market_cols = list(ENV_MARKET_COLS_DEFAULT)
    if not args.skip_agent and args.agent_checkpoint.exists() and args.analyst_cache.exists():
        # Ensure required cols exist
        for col in ["open", "high", "low", "close", "returns"] + market_cols:
            if col not in df5_norm.columns:
                raise RuntimeError(f"Required column missing for agent analysis: {col}")

        cache = np.load(str(args.analyst_cache), allow_pickle=True)
        cache_start_idx = int(cache["start_idx"])
        market_feats_full = df5_norm[market_cols].to_numpy(dtype=np.float32)[cache_start_idx:]
        split = int(0.85 * len(market_feats_full))
        mean = market_feats_full[:split].mean(axis=0).astype(np.float32)
        std = market_feats_full[:split].std(axis=0).astype(np.float32)
        std = np.where(std > 1e-8, std, 1.0).astype(np.float32)

        agent_group_df, agent_feature_df, agent_returns_stats = _agent_observation_sensitivity(
            df5_norm=df5_norm,
            market_cols=market_cols,
            agent_checkpoint=args.agent_checkpoint,
            analyst_cache_path=args.analyst_cache,
            market_feat_mean=mean,
            market_feat_std=std,
            agent_lookback_window=int(getattr(config.trading, "agent_lookback_window", 0)),
            seed=int(args.seed),
            n_samples=int(args.agent_samples),
        )

        agent_group_df.to_csv(out_dir / "agent_group_sensitivity.csv", index=False)
        agent_feature_df.to_csv(out_dir / "agent_feature_sensitivity.csv", index=False)
        pd.Series(agent_returns_stats).to_csv(out_dir / "agent_returns_slice_stats.csv")

    # Persist artifacts.
    imp_path = out_dir / "analyst_permutation_importance.csv"
    raw_path = out_dir / "feature_stats_5m_raw.csv"
    norm_path = out_dir / "feature_stats_5m_normalized.csv"
    corr_path = out_dir / "high_corr_pairs.csv"
    summary_path = out_dir / "SUMMARY.md"

    imp_df.to_csv(imp_path, index=False)
    stats_raw.to_csv(raw_path, index=False)
    stats_norm.to_csv(norm_path, index=False)
    corr_pairs.to_csv(corr_path, index=False)
    _write_summary_md(
        summary_path,
        baseline_bal_acc=baseline_bal_acc,
        baseline_bce=baseline_bce,
        importance_df=imp_df,
        stats_raw=stats_raw,
        stats_norm=stats_norm,
        corr_pairs=corr_pairs,
        agent_group_sensitivity=agent_group_df,
        agent_feature_sensitivity=agent_feature_df,
        agent_returns_stats=agent_returns_stats,
    )

    # Console short summary.
    global_imp = imp_df[imp_df["scope"] == "all_timeframes"].sort_values("delta_bce_mean", ascending=False)
    low_imp = global_imp.tail(6).sort_values("delta_bce_mean", ascending=True)

    print(f"Saved report to: {out_dir}")
    print(f"Analyst baseline (val): bal_acc={baseline_bal_acc:.4f}  bce={baseline_bce:.4f}")
    print("Lowest-contribution (all-timeframe permutation):")
    for row in low_imp.itertuples(index=False):
        print(
            f"  {row.feature:>18s}  ΔBCE={row.delta_bce_mean:+.4f}  "
            f"ΔBalAcc={row.delta_bal_acc_mean:+.4f}"
        )

    # Flag potential observation scaling pitfall: normalized returns are ~N(0,1) and later *100 in env.
    # We don't mutate code here; this is surfaced for human review.
    if "returns" in df5_norm.columns:
        ret_std = float(df5_norm["returns"].std())
        print(
            f"NOTE: `features_5m_normalized.parquet:returns` std≈{ret_std:.3f}. "
            "TradingEnv appends `returns_slice * 100` to observations; "
            "if `returns` is already z-scored, this can dominate obs scale."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
