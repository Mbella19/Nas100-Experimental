#!/usr/bin/env python3
"""
Run the MT5 ↔ Python live bridge server.

This starts a TCP server that accepts a length-prefixed JSON payload from an MT5 EA
and responds with length-prefixed JSON containing {action,size,sl,tp}.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path (so `config/` and `src/` are importable)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MT5 ↔ Python bridge server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--symbol", default="NAS100", help="Main symbol name (must match EA)")
    parser.add_argument("--lot-scale", type=float, default=1.0, help="Scale model size to MT5 lots")
    parser.add_argument("--min-m1-rows", type=int, default=10_000, help="Warmup minutes before trading")
    parser.add_argument("--history-dir", default="data/live", help="Where to persist live bars")
    parser.add_argument("--dry-run", action="store_true", help="Run inference but never trade (noop responses)")
    parser.add_argument("--log-dir", default=None, help="Optional log directory")
    args = parser.parse_args()

    # Delay heavy imports until AFTER argparse (keeps `--help` fast).
    from config.settings import Config
    from src.live.mt5_bridge import BridgeConfig, run_mt5_bridge

    system_cfg = Config()

    bridge_cfg = BridgeConfig(
        host=args.host,
        port=args.port,
        main_symbol=args.symbol,
        lot_scale=args.lot_scale,
        min_m1_rows=args.min_m1_rows,
        history_dir=Path(args.history_dir),
        dry_run=args.dry_run,
    )

    run_mt5_bridge(bridge_cfg, system_cfg=system_cfg, log_dir=args.log_dir)


if __name__ == "__main__":
    main()
