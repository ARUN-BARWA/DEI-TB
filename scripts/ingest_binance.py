"""Ingest Binance klines and build the combined research series (Binance before the switch, Delta after).

    python scripts/ingest_binance.py --src D:/data/binance          # after fetch_binance.py and ingest_trades.py

For every coin in settings.yaml `combined.symbols` (e.g. BTC = Binance BTCUSDT then Delta BTCUSD):
  * Binance minutes are stored as BINANCE_<SYMBOL> (data_store/minutes/BINANCE_<SYMBOL>/<YYYY-MM>.parquet);
  * the combined series is completed with a forced segment break at the switch and resampled to every
    timeframe as data_store/bars/<tf>/<COIN>.parquet;
  * data_store/reports/combined_<COIN>.json holds per-source quality figures.
Then: python scripts/build_dataset.py --symbols BTC ETH SOL DOGE
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading_bot.config import load_settings  # noqa: E402
from trading_bot.data.bars import complete_minute_bars, resample_bars  # noqa: E402
from trading_bot.data.binance import combine_sources, kline_quality, read_kline_zip  # noqa: E402
from trading_bot.data.storage import Store  # noqa: E402
from trading_bot.data.validate import minute_bar_quality  # noqa: E402


def main() -> None:
    settings = load_settings()
    comb = settings.combined
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="folder written by fetch_binance.py")
    p.add_argument("--root", default=str(settings.data_root))
    p.add_argument("--coins", nargs="+", default=list(comb.symbols))
    p.add_argument("--timeframes", nargs="+", default=["1m", "5m", "15m", "1h", "4h"])
    p.add_argument("--max-gap", default="6h")
    args = p.parse_args()

    store = Store(args.root)
    store.reports_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for coin in args.coins:
        bsym, dsym = comb.symbols[coin]["binance"], comb.symbols[coin]["delta"]
        files = sorted((Path(args.src) / bsym).glob(f"{bsym}-1m-*.zip"))
        if not files:
            sys.exit(f"no kline files for {bsym} under {args.src}")
        monthly = {}
        for f in files:
            month = f.stem.split("-1m-")[1]
            m = read_kline_zip(f)
            store.write_frame(m, store.minutes_path(f"BINANCE_{bsym}", month))
            monthly[month] = kline_quality(m)
        binance = store.load_trade_minutes(f"BINANCE_{bsym}")
        delta = store.load_trade_minutes(dsym)
        combined = combine_sources(binance, delta, comb.switch)
        minutes = complete_minute_bars(combined.drop(columns="venue"), max_gap=args.max_gap, breaks=[comb.switch])
        for tf in args.timeframes:
            store.write_frame(resample_bars(minutes, tf), store.bars_path(coin, tf))
        q = minute_bar_quality(minutes, max_gap=args.max_gap)
        (store.reports_dir / f"combined_{coin}.json").write_text(json.dumps(
            {"coin": coin, "binance": bsym, "delta": dsym, "switch": comb.switch, "bars": q, "binance_months": monthly}, indent=1))
        n_bin = int((combined["venue"] == "binance").sum())
        rows.append({"coin": coin, "binance_months": len(files), "first_bar": q["first_bar"][:16], "last_bar": q["last_bar"][:16],
                     "binance_minutes": n_bin, "delta_minutes": len(combined) - n_bin, "segments": q["segments"],
                     "pct_min_no_trade": q["pct_minutes_without_trades"],
                     "ohlc_inconsistent": sum(v["ohlc_inconsistent"] for v in monthly.values()),
                     "ann_vol": round(q["ann_vol_from_1m"], 3)})
        print(f"{coin}: {len(files)} Binance months + Delta {dsym} -> bars {args.timeframes}")
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print("\n" + pd.DataFrame(rows).to_string(index=False))
    print(f"\nnext: python scripts/build_dataset.py --symbols {' '.join(args.coins)}")


if __name__ == "__main__":
    main()
