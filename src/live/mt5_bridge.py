"""
MT5 ↔ Python bridge for live inference.

Design goals:
- Match the offline training/backtest conditions as closely as possible:
  - Ingest 1-minute OHLC from MT5 (server time) + UTC offset
  - Convert timestamps to UTC (timezone-naive like the training pipeline)
  - Rebuild 5m/15m/45m bars using the same pandas resampling semantics
    (label='right', closed='left') to avoid look-ahead bias
  - Apply the same feature engineering and saved normalizers
  - Build observations with the same ordering/scaling as TradingEnv
  - Run frozen Analyst + PPO Agent inference and return trade instructions

The MT5-side EA is expected to connect via TCP, send a length-prefixed JSON
payload, then read a length-prefixed JSON response.
"""

from __future__ import annotations

import json
import socketserver
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from config.settings import Config, get_device
from src.agents.sniper_agent import SniperAgent
from src.agents.recurrent_agent import RecurrentSniperAgent
from src.data.features import engineer_all_features
from src.data.normalizer import FeatureNormalizer
from src.data.resampler import resample_all_timeframes, align_timeframes
from src.models.analyst import load_analyst
from src.utils.logging_config import setup_logging, get_logger

from .bridge_constants import MODEL_FEATURE_COLS, MARKET_FEATURE_COLS, POSITION_SIZES

logger = get_logger(__name__)


# =============================================================================
# Feature/Observation conventions (must match training)
# =============================================================================


@dataclass(frozen=True)
class MarketFeatureStats:
    """Z-score stats applied to `market_features` inside the observation (FALLBACK only)."""

    cols: Tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "MarketFeatureStats":
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        cols = tuple(data["cols"].tolist())
        mean = data["mean"].astype(np.float32)
        std = data["std"].astype(np.float32)
        std = np.where(std > 1e-8, std, 1.0).astype(np.float32)
        return cls(cols=cols, mean=mean, std=std)


class RollingMarketNormalizer:
    """
    O(1) rolling window normalizer for market features.
    
    Uses circular buffer + running sums to compute rolling mean/std,
    exactly matching TradingEnv's implementation for training parity.
    """
    
    ROLLING_WINDOW_SIZE = 5760  # 20 days of 5m bars (matches TradingEnv)
    ROLLING_MIN_SAMPLES = 100   # Minimum samples before using rolling stats
    
    def __init__(self, n_features: int, fallback_stats: MarketFeatureStats):
        self.n_features = n_features
        self.fallback_stats = fallback_stats
        
        # Circular buffer for O(1) updates
        self.buffer = np.zeros((self.ROLLING_WINDOW_SIZE, n_features), dtype=np.float32)
        self.idx = 0  # Current write position
        self.count = 0  # Samples added (up to ROLLING_WINDOW_SIZE)
        
        # Running sums for O(1) mean/std calculation
        self.rolling_sum = np.zeros(n_features, dtype=np.float64)
        self.rolling_sum_sq = np.zeros(n_features, dtype=np.float64)
    
    def update_and_normalize(self, market_feat_row: np.ndarray) -> np.ndarray:
        """
        Update rolling buffer with new row and return normalized features.
        Uses rolling stats if enough samples, otherwise falls back to global stats.
        """
        market_feat_row = market_feat_row.astype(np.float32)
        
        # Always update the rolling buffer
        if self.count >= self.ROLLING_WINDOW_SIZE:
            # Evict oldest value from running sums
            old_val = self.buffer[self.idx]
            self.rolling_sum -= old_val
            self.rolling_sum_sq -= old_val ** 2
        
        # Add new value
        self.buffer[self.idx] = market_feat_row
        self.rolling_sum += market_feat_row
        self.rolling_sum_sq += market_feat_row ** 2
        
        # Update circular index
        self.idx = (self.idx + 1) % self.ROLLING_WINDOW_SIZE
        if self.count < self.ROLLING_WINDOW_SIZE:
            self.count += 1
        
        # Calculate normalized features
        if self.count >= self.ROLLING_MIN_SAMPLES:
            # O(1) rolling mean/std calculation
            n = self.count
            rolling_means = (self.rolling_sum / n).astype(np.float32)
            variance = (self.rolling_sum_sq / n) - (rolling_means ** 2)
            rolling_stds = np.maximum(np.sqrt(np.maximum(variance, 0)), 1e-6).astype(np.float32)
            
            normalized = ((market_feat_row - rolling_means) / rolling_stds).astype(np.float32)
        else:
            # Fallback to global stats until we have enough samples
            normalized = ((market_feat_row - self.fallback_stats.mean) / 
                         self.fallback_stats.std).astype(np.float32)
        
        # Safety clip to ±5.0 (matches TradingEnv)
        return np.clip(normalized, -5.0, 5.0)
    
    def warmup(self, historical_rows: np.ndarray) -> None:
        """
        Pre-fill the rolling buffer with historical data.
        Call this at startup with the last N market feature rows.
        """
        if historical_rows is None or len(historical_rows) == 0:
            return
        
        # Take last ROLLING_WINDOW_SIZE rows
        warmup_data = historical_rows[-self.ROLLING_WINDOW_SIZE:].astype(np.float32)
        n_warmup = len(warmup_data)
        
        # Fill buffer
        self.buffer[:n_warmup] = warmup_data
        self.idx = n_warmup % self.ROLLING_WINDOW_SIZE
        self.count = n_warmup
        
        # Compute running sums
        self.rolling_sum = warmup_data.sum(axis=0).astype(np.float64)
        self.rolling_sum_sq = (warmup_data ** 2).sum(axis=0).astype(np.float64)
        
        logger.info(f"Rolling normalizer warmed up with {n_warmup} samples")


@dataclass
class BridgeConfig:
    """Runtime configuration for the MT5 bridge server."""

    host: str = "127.0.0.1"
    port: int = 5555
    main_symbol: str = "NAS100"
    decision_tf_minutes: int = 5

    # Persist incoming M1 bars so restarts don't require MT5 bootstrap.
    history_dir: Path = field(default_factory=lambda: Path("data") / "live")
    max_m1_rows: int = 60 * 24 * 30  # ~30 days

    # Minimum M1 rows required before trading is enabled.
    # Conservative: sized to cover long-window features (e.g., 45m SMA(50)).
    min_m1_rows: int = 10_000

    # Execution mapping (EA expects lots).
    lot_scale: float = 1.0

    # Feature pipeline window sizes (recompute on a tail window for speed).
    tail_5m_bars: int = 600
    tail_15m_bars: int = 400
    tail_45m_bars: int = 260

    # Safety / testing
    dry_run: bool = False


def _read_exact(sock, n: int) -> bytes:
    """Read exactly n bytes from a socket."""
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Socket closed while reading")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _decode_length_prefixed_json(sock) -> Dict[str, Any]:
    header = _read_exact(sock, 4)
    (length,) = struct.unpack(">I", header)
    if length <= 0 or length > 50_000_000:
        raise ValueError(f"Invalid payload length: {length}")
    payload = _read_exact(sock, length)
    return json.loads(payload.decode("utf-8"))


def _encode_length_prefixed_json(obj: Dict[str, Any]) -> bytes:
    payload = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return struct.pack(">I", len(payload)) + payload


def _rates_array_to_df(
    rates: list,
    utc_offset_sec: int,
) -> pd.DataFrame:
    """
    Convert an MT5 rates array to a DataFrame indexed by UTC (timezone-naive).

    Expected per-row formats:
      [time, open, high, low, close] or [time, open, high, low, close, ...]
    Where `time` is in broker/server time; `utc_offset_sec` converts it to UTC.
    """
    if not rates:
        return pd.DataFrame(columns=["open", "high", "low", "close"])

    rows = []
    for row in rates:
        if not isinstance(row, (list, tuple)) or len(row) < 5:
            continue
        t_server = int(row[0])
        t_utc = t_server - int(utc_offset_sec)
        rows.append((t_utc, float(row[1]), float(row[2]), float(row[3]), float(row[4])))

    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close"])

    arr = np.array(rows, dtype=np.float64)
    idx = pd.to_datetime(arr[:, 0].astype(np.int64), unit="s", utc=True).tz_localize(None)
    df = pd.DataFrame(
        {
            "open": arr[:, 1].astype(np.float32),
            "high": arr[:, 2].astype(np.float32),
            "low": arr[:, 3].astype(np.float32),
            "close": arr[:, 4].astype(np.float32),
        },
        index=idx,
    )
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _merge_append_ohlc(existing: pd.DataFrame, new_df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if existing is None or existing.empty:
        merged = new_df.copy()
    else:
        merged = pd.concat([existing, new_df], axis=0)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    if len(merged) > max_rows:
        merged = merged.iloc[-max_rows:].copy()
    return merged


def _save_history(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Small, fast persistence format
    df.to_parquet(path)


def _load_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"History file has invalid index: {path}")
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = df[col].astype(np.float32)
    return df.sort_index()


def _build_observation(
    *,
    analyst: torch.nn.Module,
    agent_env_cfg: Config,
    rolling_normalizer: RollingMarketNormalizer,  # v27: Use rolling normalizer for training parity
    x_5m: np.ndarray,
    x_15m: np.ndarray,
    x_45m: np.ndarray,
    market_feat_row: np.ndarray,
    returns_row_window: np.ndarray,
    position: int,
    entry_price: float,
    current_price: float,
    position_size: float,
    time_in_trade_norm: float = 0.0,  # v25: How long held (0-1)
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Construct an observation vector with the same ordering as TradingEnv._get_observation().
    """
    device = next(analyst.parameters()).device if hasattr(analyst, "parameters") else torch.device("cpu")

    x_5m_t = torch.tensor(x_5m[None, ...], device=device, dtype=torch.float32)
    x_15m_t = torch.tensor(x_15m[None, ...], device=device, dtype=torch.float32)
    x_45m_t = torch.tensor(x_45m[None, ...], device=device, dtype=torch.float32)

    with torch.no_grad():
        res = analyst.get_probabilities(x_5m_t, x_15m_t, x_45m_t)
        if isinstance(res, (tuple, list)) and len(res) == 3:
            context, probs, _ = res
        else:
            context, probs = res

    context_np = context.cpu().numpy().flatten().astype(np.float32)
    probs_np = probs.cpu().numpy().flatten().astype(np.float32)

    # Analyst metrics (binary vs multi-class)
    if len(probs_np) == 2:
        p_down = float(probs_np[0])
        p_up = float(probs_np[1])
        confidence = max(p_down, p_up)
        edge = p_up - p_down
        uncertainty = 1.0 - confidence
        analyst_metrics = np.array([p_down, p_up, edge, confidence, uncertainty], dtype=np.float32)
    else:
        p_down = float(probs_np[0])
        p_neutral = float(probs_np[1])
        p_up = float(probs_np[2])
        confidence = float(np.max(probs_np))
        edge = p_up - p_down
        uncertainty = 1.0 - confidence
        analyst_metrics = np.array(
            [p_down, p_neutral, p_up, edge, confidence, uncertainty], dtype=np.float32
        )

    # Position state (mirrors TradingEnv normalization)
    atr = float(market_feat_row[0]) if len(market_feat_row) > 0 else 1.0
    atr_safe = max(atr, 1e-6)

    if position != 0:
        if position == 1:
            entry_price_norm = (current_price - entry_price) / (atr_safe * 100.0)
        else:
            entry_price_norm = (entry_price - current_price) / (atr_safe * 100.0)
        entry_price_norm = float(np.clip(entry_price_norm, -10.0, 10.0))

        pip_value = agent_env_cfg.instrument.pip_value
        if position == 1:
            unrealized_pnl = (current_price - entry_price) / pip_value
        else:
            unrealized_pnl = (entry_price - current_price) / pip_value
        unrealized_pnl *= float(position_size)
        unrealized_pnl_norm = float(unrealized_pnl / 100.0)
    else:
        entry_price_norm = 0.0
        unrealized_pnl_norm = 0.0

    # v25: Position state now has 4 elements (matching TradingEnv)
    position_state = np.array([float(position), entry_price_norm, unrealized_pnl_norm, time_in_trade_norm], dtype=np.float32)

    # v27: Use rolling window normalization for training parity
    # No need to check column ordering - rolling_normalizer handles this
    market_feat_norm = rolling_normalizer.update_and_normalize(market_feat_row)

    # SL/TP distance features (mirrors TradingEnv)
    dist_sl_norm = 0.0
    dist_tp_norm = 0.0
    if position != 0 and atr > 1e-8:
        pip_value = agent_env_cfg.instrument.pip_value
        sl_pips = max((atr * agent_env_cfg.trading.sl_atr_multiplier) / pip_value, 5.0)
        tp_pips = max((atr * agent_env_cfg.trading.tp_atr_multiplier) / pip_value, 5.0)

        if position == 1:
            sl_price = entry_price - sl_pips * pip_value
            tp_price = entry_price + tp_pips * pip_value
            dist_sl_norm = (current_price - sl_price) / atr
            dist_tp_norm = (tp_price - current_price) / atr
        else:
            sl_price = entry_price + sl_pips * pip_value
            tp_price = entry_price - tp_pips * pip_value
            dist_sl_norm = (sl_price - current_price) / atr
            dist_tp_norm = (current_price - tp_price) / atr

    obs = np.concatenate(
        [
            context_np,
            position_state,
            market_feat_norm,
            analyst_metrics,
            np.array([dist_sl_norm, dist_tp_norm], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)

    # Full-eyes returns window (already normalized in the pipeline; env multiplies by 100)
    lookback = int(agent_env_cfg.trading.agent_lookback_window)
    if lookback > 0:
        if returns_row_window.shape[0] != lookback:
            raise ValueError(f"returns window mismatch: {returns_row_window.shape[0]} != {lookback}")
        obs = np.concatenate([obs, (returns_row_window.astype(np.float32) * 100.0)], axis=0)

    info = {
        "p_down": float(probs_np[0]) if probs_np.size >= 1 else 0.5,
        "p_up": float(probs_np[-1]) if probs_np.size >= 2 else 0.5,
    }
    return obs.astype(np.float32), info


class MT5BridgeState:
    def __init__(self, cfg: BridgeConfig, system_cfg: Config):
        self.cfg = cfg
        self.system_cfg = system_cfg

        self.history_dir = cfg.history_dir
        self.history_dir.mkdir(parents=True, exist_ok=True)

        self.m1_path = self.history_dir / f"{cfg.main_symbol}_M1.parquet"
        self.m1: pd.DataFrame = _load_history(self.m1_path)

        # Load artifacts
        self.normalizers: Dict[str, FeatureNormalizer] = {
            "5m": FeatureNormalizer.load(system_cfg.paths.models_analyst / "normalizer_5m.pkl"),
            "15m": FeatureNormalizer.load(system_cfg.paths.models_analyst / "normalizer_15m.pkl"),
            "45m": FeatureNormalizer.load(system_cfg.paths.models_analyst / "normalizer_45m.pkl"),
        }

        market_stats_path = system_cfg.paths.models_agent / "market_feat_stats.npz"
        self.market_feat_stats = MarketFeatureStats.load(market_stats_path)
        
        # v27: Rolling window normalizer for market features (matches TradingEnv)
        self.rolling_normalizer = RollingMarketNormalizer(
            n_features=len(MARKET_FEATURE_COLS),
            fallback_stats=self.market_feat_stats
        )

        feature_dims = {k: len(MODEL_FEATURE_COLS) for k in ("5m", "15m", "45m")}
        analyst_path = system_cfg.paths.models_analyst / "best.pt"
        self.analyst = load_analyst(str(analyst_path), feature_dims, device=system_cfg.device, freeze=True)
        self.analyst.eval()

        obs_dim = self._expected_obs_dim()
        dummy_env = _make_dummy_env(obs_dim)

        # Detect if this is a recurrent model by checking for marker file
        # Check both standard and recurrent directories
        recurrent_marker = system_cfg.paths.models_agent_recurrent / ".recurrent"
        recurrent_model_path = system_cfg.paths.models_agent_recurrent / "final_model.zip"
        standard_model_path = system_cfg.paths.models_agent / "final_model.zip"

        self.is_recurrent = False
        if recurrent_marker.exists() and recurrent_model_path.exists():
            # Load recurrent model
            self.is_recurrent = True
            self.agent = RecurrentSniperAgent.load(str(recurrent_model_path), dummy_env, device="cpu")
            self.agent.reset_lstm_states()
            logger.info("Loaded RecurrentPPO agent from %s", recurrent_model_path)
        else:
            # Load standard PPO model
            self.agent = SniperAgent.load(str(standard_model_path), dummy_env, device="cpu")
            logger.info("Loaded standard PPO agent from %s", standard_model_path)

        # Track position changes for LSTM episode boundary detection
        self._last_position: int = 0

        self.last_decision_label_utc: Optional[pd.Timestamp] = None
        # v18 parity: track entry bar for min-hold enforcement (fallback when MT5 omits open_time).
        self._fallback_entry_pos: Optional[int] = None
        self._fallback_entry_label_utc: Optional[pd.Timestamp] = None
        self._fallback_position: int = 0

        logger.info(
            "MT5 bridge ready | symbol=%s | obs_dim=%d | feature_dim=%d",
            cfg.main_symbol,
            obs_dim,
            len(MODEL_FEATURE_COLS),
        )

    def _expected_obs_dim(self) -> int:
        context_dim = int(getattr(self.analyst, "context_dim", self.system_cfg.analyst.context_dim))
        analyst_metrics_dim = 5 if int(getattr(self.analyst, "num_classes", 2)) == 2 else 6
        n_market = len(MARKET_FEATURE_COLS)
        returns_dim = int(self.system_cfg.trading.agent_lookback_window)
        # v25: Position now has 4 elements: [position, entry_price_norm, unrealized_pnl_norm, time_in_trade_norm]
        return context_dim + 4 + n_market + analyst_metrics_dim + 2 + returns_dim

    def update_from_payload(self, payload: Dict[str, Any]) -> None:
        utc_offset_sec = int(payload.get("time", {}).get("utc_offset_sec", 0))

        rates = payload.get("rates", {})
        m1_rates = rates.get("m1") or rates.get("1m") or rates.get("M1") or []
        m1_df = _rates_array_to_df(m1_rates, utc_offset_sec=utc_offset_sec)
        if not m1_df.empty:
            self.m1 = _merge_append_ohlc(self.m1, m1_df, self.cfg.max_m1_rows)
            _save_history(self.m1, self.m1_path)

    def _should_decide_now(self, payload: Dict[str, Any]) -> Tuple[bool, Optional[pd.Timestamp]]:
        utc_offset_sec = int(payload.get("time", {}).get("utc_offset_sec", 0))
        rates = payload.get("rates", {})
        m1_rates = rates.get("m1") or rates.get("1m") or []
        if not m1_rates:
            return False, None

        last_row = m1_rates[-1]
        if not isinstance(last_row, (list, tuple)) or len(last_row) < 1:
            return False, None

        t_server = int(last_row[0])
        t_utc_open = t_server - utc_offset_sec
        t_utc_close = t_utc_open + 60

        if t_utc_close % (self.cfg.decision_tf_minutes * 60) != 0:
            return False, None

        label = pd.to_datetime(t_utc_close, unit="s", utc=True).tz_localize(None)
        if self.last_decision_label_utc is not None and label <= self.last_decision_label_utc:
            return False, label
        return True, label

    def decide(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.update_from_payload(payload)

        should_decide, label = self._should_decide_now(payload)
        if not should_decide:
            return {"action": 999, "reason": "no_new_5m_bar"}

        if len(self.m1) < self.cfg.min_m1_rows:
            return {
                "action": 999,
                "reason": f"warming_up_m1_history ({len(self.m1)}/{self.cfg.min_m1_rows})",
            }

        # Build features on a tail window for speed
        df_1m = self.m1.copy()

        # Resample to native bars (no gap filling, no weekend flats)
        resampled = resample_all_timeframes(df_1m, self.system_cfg.data.timeframes)
        df_5m = resampled["5m"]
        df_15m = resampled["15m"]
        df_45m = resampled["45m"]

        feature_cfg = {
            "pinbar_wick_ratio": self.system_cfg.features.pinbar_wick_ratio,
            "doji_body_ratio": self.system_cfg.features.doji_body_ratio,
            "fractal_window": self.system_cfg.features.fractal_window,
            "sr_lookback": self.system_cfg.features.sr_lookback,
            "sma_period": self.system_cfg.features.sma_period,
            "ema_fast": self.system_cfg.features.ema_fast,
            "ema_slow": self.system_cfg.features.ema_slow,
            "chop_period": self.system_cfg.features.chop_period,
            "adx_period": self.system_cfg.features.adx_period,
            "atr_period": self.system_cfg.features.atr_period,
        }

        df_5m = engineer_all_features(df_5m, feature_cfg)
        df_15m = engineer_all_features(df_15m, feature_cfg)
        df_45m = engineer_all_features(df_45m, feature_cfg)

        # Align higher TFs to 5m grid AFTER feature engineering
        df_5m, df_15m, df_45m = align_timeframes(df_5m, df_15m, df_45m)

        # Drop rows with any NaN across timeframes (mirrors pipeline)
        common_valid = (~df_5m.isna().any(axis=1)) & (~df_15m.isna().any(axis=1)) & (~df_45m.isna().any(axis=1))
        df_5m = df_5m[common_valid]
        df_15m = df_15m[common_valid]
        df_45m = df_45m[common_valid]

        # Apply saved normalizers (per timeframe). RAW columns remain unscaled.
        df_5m_n = self.normalizers["5m"].transform(df_5m)
        df_15m_n = self.normalizers["15m"].transform(df_15m)
        df_45m_n = self.normalizers["45m"].transform(df_45m)

        if label not in df_5m_n.index:
            # Not enough data to form this label in our UTC-resampled timeline.
            return {"action": 999, "reason": "label_not_ready"}

        # Find positional index
        pos = int(df_5m_n.index.get_loc(label))
        utc_offset_sec = int(payload.get("time", {}).get("utc_offset_sec", 0))

        lookback_5m = int(self.system_cfg.analyst.lookback_5m)
        lookback_15m = int(self.system_cfg.analyst.lookback_15m)
        lookback_45m = int(self.system_cfg.analyst.lookback_45m)
        subsample_15m = 3
        subsample_45m = 9
        start_idx = max(
            lookback_5m,
            (lookback_15m - 1) * subsample_15m + 1,
            (lookback_45m - 1) * subsample_45m + 1,
        )

        if pos < start_idx:
            return {"action": 999, "reason": "insufficient_lookback"}

        features_5m = df_5m_n[MODEL_FEATURE_COLS].values.astype(np.float32)
        features_15m = df_15m_n[MODEL_FEATURE_COLS].values.astype(np.float32)
        features_45m = df_45m_n[MODEL_FEATURE_COLS].values.astype(np.float32)

        x_5m = features_5m[pos - lookback_5m + 1:pos + 1]

        idx_range_15m = list(
            range(pos - (lookback_15m - 1) * subsample_15m, pos + 1, subsample_15m)
        )
        x_15m = features_15m[idx_range_15m]

        idx_range_45m = list(
            range(pos - (lookback_45m - 1) * subsample_45m, pos + 1, subsample_45m)
        )
        x_45m = features_45m[idx_range_45m]

        # Market features row (order must match MARKET_FEATURE_COLS)
        for col in MARKET_FEATURE_COLS:
            if col not in df_5m_n.columns:
                df_5m_n[col] = 0.0
        market_feat_row = df_5m_n[MARKET_FEATURE_COLS].iloc[pos].values.astype(np.float32)

        # Returns window (already normalized by FeatureNormalizer; env multiplies by 100)
        lookback_ret = int(self.system_cfg.trading.agent_lookback_window)
        returns_series = df_5m_n["returns"].values.astype(np.float32)
        returns_window = returns_series[pos - lookback_ret + 1:pos + 1] if lookback_ret > 0 else np.array([], dtype=np.float32)
        if lookback_ret > 0 and returns_window.shape[0] != lookback_ret:
            return {"action": 999, "reason": "returns_window_not_ready"}

        # Position state from MT5
        pos_payload = payload.get("position", {}) or {}
        pos_type = int(pos_payload.get("type", -1))
        mt5_volume = float(pos_payload.get("volume", 0.0))
        mt5_entry = float(pos_payload.get("price", 0.0))
        # Optional: MT5 position open time (server epoch seconds). If present, enables exact
        # bar-based min_hold_bars enforcement even across bridge restarts.
        mt5_open_time_server: Optional[int] = None
        if "open_time" in pos_payload:
            try:
                mt5_open_time_server = int(pos_payload.get("open_time") or 0)
            except (TypeError, ValueError):
                mt5_open_time_server = None

        if pos_type == 0:
            position = 1
        elif pos_type == 1:
            position = -1
        else:
            position = 0

        current_price = float(df_5m_n["close"].iloc[pos])
        entry_price = float(mt5_entry) if position != 0 else current_price
        # Keep observation scaling consistent with training env: invert lot_scale so the agent
        # sees the same "position_size" units it was trained on.
        lot_scale = float(self.cfg.lot_scale) if float(self.cfg.lot_scale) > 0 else 1.0
        position_size = float(mt5_volume / lot_scale) if position != 0 else 0.0

        # v18 parity: compute bars held since entry (5m bars), using MT5 open_time when available.
        entry_pos: Optional[int] = None
        entry_label_utc: Optional[pd.Timestamp] = None
        if position == 0:
            self._fallback_entry_pos = None
            self._fallback_entry_label_utc = None
            self._fallback_position = 0
        else:
            if mt5_open_time_server is not None and mt5_open_time_server > 0:
                try:
                    open_time_utc = pd.to_datetime(
                        int(mt5_open_time_server) - int(utc_offset_sec), unit="s", utc=True
                    ).tz_localize(None)
                    # Map to the most recent 5m label at-or-before the open time.
                    entry_label = df_5m_n.index.asof(open_time_utc)
                    if entry_label is not None and not pd.isna(entry_label):
                        entry_label_utc = pd.to_datetime(entry_label)
                        entry_pos = int(df_5m_n.index.get_loc(entry_label_utc))
                except Exception:
                    entry_pos = None
                    entry_label_utc = None

            # Fallback: derive entry from first observation of an open position.
            if entry_pos is None:
                if self._fallback_position != position or self._fallback_entry_pos is None:
                    self._fallback_entry_pos = pos
                    self._fallback_entry_label_utc = label
                self._fallback_position = position
                entry_pos = self._fallback_entry_pos
                entry_label_utc = self._fallback_entry_label_utc

        # v25: Calculate time in trade (normalized to [0, 1])
        if position != 0 and entry_pos is not None:
            bars_held_for_obs = max(0, int(pos - entry_pos))
            time_in_trade_norm = min(bars_held_for_obs / 100.0, 1.0)
        else:
            time_in_trade_norm = 0.0

        obs, obs_info = _build_observation(
            analyst=self.analyst,
            agent_env_cfg=self.system_cfg,
            rolling_normalizer=self.rolling_normalizer,  # v27: Use rolling normalizer
            x_5m=x_5m,
            x_15m=x_15m,
            x_45m=x_45m,
            market_feat_row=market_feat_row,
            returns_row_window=returns_window,
            position=position,
            entry_price=entry_price,
            current_price=current_price,
            position_size=position_size,
            time_in_trade_norm=time_in_trade_norm,  # v25: Pass time in trade
        )

        if self.cfg.dry_run:
            self.last_decision_label_utc = label
            return {"action": 999, "reason": "dry_run", **obs_info}

        # Detect episode boundary for LSTM state management
        # Episode boundary = position changed from open to flat
        episode_start = (self._last_position != 0 and position == 0)
        self._last_position = position

        if self.is_recurrent:
            # RecurrentPPO needs episode_start flag for LSTM state management
            action, _ = self.agent.predict(
                obs,
                deterministic=True,
                episode_start=episode_start,
                min_action_confidence=float(self.system_cfg.trading.min_action_confidence),
            )
        else:
            # Standard PPO
            action, _ = self.agent.predict(
                obs,
                deterministic=True,
                min_action_confidence=float(self.system_cfg.trading.min_action_confidence),
            )
        action = np.array(action).astype(np.int32).flatten()
        if action.size < 2:
            return {"action": 999, "reason": "invalid_agent_action"}

        direction = int(action[0])
        size_idx = int(action[1])
        size_idx = int(np.clip(size_idx, 0, 3))

        # v18 parity: forced minimum hold time (parity with TradingEnv/backtest: blocks EXIT and FLIPS).
        # v19 parity: profit-based early exit override
        exit_blocked = False
        bars_held = 0
        min_hold_bars = int(getattr(self.system_cfg.trading, "min_hold_bars", 0))
        early_exit_profit_atr = float(getattr(self.system_cfg.trading, "early_exit_profit_atr", 3.0))
        
        if position != 0 and min_hold_bars > 0 and entry_pos is not None:
            bars_held = max(0, int(pos - entry_pos))
            if bars_held < min_hold_bars:
                would_close_or_flip = (
                    direction == 0 or  # Flat/Exit
                    (position == 1 and direction == 2) or  # Long→Short flip
                    (position == -1 and direction == 1)    # Short→Long flip
                )
                if would_close_or_flip:
                    # v19: Check for profit-based early exit override
                    allow_early_exit = False
                    if early_exit_profit_atr > 0 and entry_price > 0:
                        atr_check = float(market_feat_row[0])
                        pip_value_check = float(self.system_cfg.instrument.pip_value)
                        current_price = current_close
                        
                        if position == 1:  # Long
                            unrealized_pnl = (current_price - entry_price) / pip_value_check
                        else:  # Short
                            unrealized_pnl = (entry_price - current_price) / pip_value_check
                        
                        profit_threshold = early_exit_profit_atr * atr_check
                        if unrealized_pnl > profit_threshold:
                            allow_early_exit = True
                            obs_info["early_exit_profit"] = True
                            obs_info["profit_pips"] = float(unrealized_pnl)
                    
                    if not allow_early_exit:
                        direction = 1 if position == 1 else 2
                        exit_blocked = True
                        obs_info["exit_blocked"] = True
                        obs_info["bars_held"] = int(bars_held)
                        obs_info["min_hold_bars"] = int(min_hold_bars)
                        if entry_label_utc is not None:
                            obs_info["entry_label_utc"] = int(entry_label_utc.timestamp())

        atr = float(market_feat_row[0])
        pip_value = float(self.system_cfg.instrument.pip_value)

        base_size = float(POSITION_SIZES[size_idx])
        lots = base_size
        if bool(getattr(self.system_cfg.trading, "volatility_sizing", True)):
            sl_pips = (atr * float(self.system_cfg.trading.sl_atr_multiplier)) / pip_value
            sl_pips = max(sl_pips, 5.0)
            # Dollar-based risk sizing (matches training/backtest):
            # Size = Risk($) / ($/pip × SL_pips)
            risk_per_trade = float(getattr(self.system_cfg.trading, "risk_per_trade", 100.0))
            dollars_per_pip = pip_value * float(self.system_cfg.instrument.lot_size) * float(self.system_cfg.instrument.point_multiplier)
            risk_amount = risk_per_trade * base_size
            lots = risk_amount / (dollars_per_pip * sl_pips)
            lots = float(np.clip(lots, 0.1, 50.0))

        # Apply lot_scale for live trading
        lots = float(np.clip(lots * float(self.cfg.lot_scale), 0.0, 1000.0))

        sl_price = 0.0
        tp_price = 0.0
        if direction in (1, 2):
            sl_pips = max((atr * float(self.system_cfg.trading.sl_atr_multiplier)) / pip_value, 5.0)
            tp_pips = max((atr * float(self.system_cfg.trading.tp_atr_multiplier)) / pip_value, 5.0)
            if direction == 1:
                sl_price = current_price - sl_pips * pip_value
                tp_price = current_price + tp_pips * pip_value
            else:
                sl_price = current_price + sl_pips * pip_value
                tp_price = current_price - tp_pips * pip_value

        self.last_decision_label_utc = label

        logger.info(
            "Decision @ %s | action=%d size=%.2f sl=%.2f tp=%.2f | p_up=%.3f p_down=%.3f%s",
            label,
            direction,
            round(lots, 2),
            float(sl_price),
            float(tp_price),
            float(obs_info.get("p_up", 0.5)),
            float(obs_info.get("p_down", 0.5)),
            " | exit_blocked" if exit_blocked else "",
        )

        return {
            "action": direction,
            "size": round(lots, 2),
            "sl": float(sl_price),
            "tp": float(tp_price),
            "ts_utc": int(label.timestamp()),
            **obs_info,
        }


def _make_dummy_env(obs_dim: int):
    import gymnasium as gym
    from gymnasium import spaces

    class _DummyEnv(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
            self.action_space = spaces.MultiDiscrete([3, 4])

        def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
            super().reset(seed=seed)
            return np.zeros((obs_dim,), dtype=np.float32), {}

        def step(self, action):
            return np.zeros((obs_dim,), dtype=np.float32), 0.0, True, False, {}

    return _DummyEnv()


class _BridgeHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        try:
            payload = _decode_length_prefixed_json(self.request)
            response = self.server.bridge_state.decide(payload)
        except Exception as e:
            logger.exception("Bridge error: %s", e)
            response = {"action": 999, "reason": "server_error"}

        self.request.sendall(_encode_length_prefixed_json(response))


class MT5BridgeServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True

    def __init__(self, server_address, handler_class, bridge_state: MT5BridgeState):
        super().__init__(server_address, handler_class)
        self.bridge_state = bridge_state


def run_mt5_bridge(
    bridge_cfg: BridgeConfig,
    system_cfg: Optional[Config] = None,
    log_dir: Optional[str | Path] = None,
) -> None:
    if system_cfg is None:
        system_cfg = Config()

    if system_cfg.device is None:
        system_cfg.device = get_device()

    setup_logging(str(log_dir) if log_dir is not None else None, name=__name__)

    state = MT5BridgeState(bridge_cfg, system_cfg)
    server = MT5BridgeServer((bridge_cfg.host, bridge_cfg.port), _BridgeHandler, state)

    logger.info("Listening on %s:%d", bridge_cfg.host, bridge_cfg.port)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
