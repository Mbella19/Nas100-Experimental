# MT5 ↔ Python Live Bridge (1:1-ish with training)

This project trains on 1-minute OHLC, then rebuilds **5m / 15m / 45m** bars using **label='right', closed='left'** to avoid look-ahead. The live bridge mirrors that by sending **closed M1 OHLC** from MT5 → Python, converting to **UTC**, then running the same resample → features → normalization → analyst+agent inference.

## 1) Prereqs (once)

- Train or ensure you already have trained models + normalizers:
  - `models/analyst/best.pt`
  - `models/analyst/normalizer_5m.pkl`, `normalizer_15m.pkl`, `normalizer_45m.pkl`
  - `models/agent/final_model.zip`

- Build the market-feature normalization stats used in the RL observation:
  - `python scripts/prepare_mt5_bridge_artifacts.py`
  - Output: `models/agent/market_feat_stats.npz`

## 2) Start the Python bridge server

`python scripts/run_mt5_bridge.py --host 127.0.0.1 --port 5555 --symbol NAS100`

Useful options:
- Dry run (no trades, logs decisions): `--dry-run`
- Lot scaling (model size → MT5 lots): `--lot-scale 0.1`

## 3) Attach the MT5 EA

- Compile `ClaudeConnector.mq5` in MetaEditor.
- Attach it to the chart of your main symbol (e.g., NAS100).
- Set:
  - `ServerHost` / `ServerPort` to match the Python bridge
  - `BootstrapOnStart=true` and `BootstrapBars=12000` (or higher if you want more warmup)

## Notes on parity

- The EA sends server↔UTC offset so Python can compute session features in UTC.
- Trading decisions are only emitted on **completed 5-minute bars**; all other minutes return a **no-op** response to MT5.
- Different brokers will still differ in *prices/spreads/fills*, but bar timing + feature logic stays consistent because the bridge rebuilds bars in Python from M1.
