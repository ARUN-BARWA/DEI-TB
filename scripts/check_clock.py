"""Check that the trade timestamps are UTC by aligning 1m returns with a reference venue.

Your Delta files carry naive timestamps. Download any free reference 1-minute kline file for the
same coin and month, e.g. Binance USD-M futures from data.binance.vision:
    https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2026-09.zip
then run (after ingest_trades.py):

    python scripts/check_clock.py --symbol BTCUSD --ref BTCUSDT-1m-2026-09.zip

A best shift of 0 means the timestamps are UTC. About +/-330 would mean IST.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading_bot.config import load_settings  # noqa: E402
from trading_bot.data.io import read_ohlcv  # noqa: E402
from trading_bot.data.storage import Store  # noqa: E402
from trading_bot.data.validate import estimate_clock_offset  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", required=True)
    p.add_argument("--ref", required=True, help="reference 1m OHLCV file (CSV/zip/Parquet) with UTC timestamps")
    p.add_argument("--root", default=str(load_settings().data_root))
    p.add_argument("--max-shift", type=int, default=720, help="minutes searched in each direction")
    args = p.parse_args()

    ours = Store(args.root).load_bars(args.symbol, "1m")
    ours = ours.loc[~ours["is_filled"], "close"]
    ref = read_ohlcv(args.ref)["close"]
    best, corr = estimate_clock_offset(ours, ref, args.max_shift)
    print(f"best shift: {best:+d} min  (corr {corr[best]:.3f})")
    for s in (0, -330, 330, -60, 60):
        if s in corr.index:
            print(f"  shift {s:+5d} min: corr {corr[s]:.3f}")
    if best == 0:
        print("timestamps are consistent with UTC.")
    else:
        print("WARNING: timestamps are NOT aligned with the reference; fix the timezone before any research.")


if __name__ == "__main__":
    main()
