"""Download free Binance USD-M futures 1-minute klines (data.binance.vision), SHA-256 verified.

    python scripts/fetch_binance.py --dest D:/data/binance             # symbols/months from settings.yaml
    python scripts/fetch_binance.py --dest ... --symbols BTCUSDT --start 2024-07 --end 2025-12

Already-downloaded files that still match their published checksum are skipped. Layout:
<dest>/<SYMBOL>/<SYMBOL>-1m-<YYYY-MM>.zip   (read by scripts/ingest_binance.py)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading_bot.config import load_settings  # noqa: E402
from trading_bot.data.binance import BASE_URL, download_month, months_between  # noqa: E402


def main() -> None:
    settings = load_settings()
    comb = settings.combined
    last_month = (pd.Timestamp(comb.switch) - pd.offsets.MonthBegin(1)).strftime("%Y-%m")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", required=True)
    p.add_argument("--symbols", nargs="+", default=[v["binance"] for v in comb.symbols.values()])
    p.add_argument("--start", default=comb.binance_start)
    p.add_argument("--end", default=last_month, help="last month (inclusive); default: month before the Delta switch")
    p.add_argument("--base-url", default=BASE_URL)
    args = p.parse_args()

    failed = []
    for sym in args.symbols:
        for month in months_between(args.start, args.end):
            try:
                path = download_month(sym, month, args.dest, base_url=args.base_url)
                print(f"ok   {path}")
            except Exception as e:  # keep going; report at the end
                failed.append((sym, month, repr(e)))
                print(f"FAIL {sym} {month}: {e}")
    if failed:
        print(f"\n{len(failed)} downloads failed (re-run to retry):")
        for f in failed:
            print("  ", *f)
        sys.exit(1)


if __name__ == "__main__":
    main()
