"""Ingest Delta monthly tick-trade archives -> validated Parquet trades, bars and quality reports.

    python scripts/ingest_trades.py --src D:/delta/trades                 # all symbols found
    python scripts/ingest_trades.py --src D:/delta/trades --symbols BTCUSD ETHUSD SOLUSD
    python scripts/ingest_trades.py --src ... --force                     # re-ingest everything

Incremental: a source file whose size and modification time match the manifest is skipped.
Raw files are only read, never modified. Output goes to the store root (default ./data_store):
trades, 1m/5m/15m/1h/4h bars, data_store/reports/<SYMBOL>.json and data_store/reports/summary.csv.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading_bot.config import load_settings  # noqa: E402
from trading_bot.data.bars import complete_minute_bars, minute_bars_to_frame, resample_bars  # noqa: E402
from trading_bot.data.storage import Store, TradeParquetWriter  # noqa: E402
from trading_bot.data.trades import discover_trade_files, ingest_trade_file  # noqa: E402
from trading_bot.data.validate import minute_bar_quality  # noqa: E402

log = logging.getLogger("ingest")


def fingerprint(path: Path) -> dict:
    st = path.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def ingest_files(store: Store, files, force: bool, chunksize: int, spike_threshold: float) -> set[str]:
    manifest = store.load_manifest()
    touched: set[str] = set()
    for f in files:
        key = f"{f.symbol}/{f.month}"
        fp = fingerprint(f.path)
        entry = manifest.get(key)
        outputs_exist = store.minutes_path(f.symbol, f.month).exists() and store.trades_path(f.symbol, f.month).exists()
        if not force and entry and entry.get("fingerprint") == fp and outputs_exist:
            log.info("skip %s (unchanged)", key)
            continue
        t0 = time.time()
        writer = TradeParquetWriter(store.trades_path(f.symbol, f.month))
        try:
            minutes, report = ingest_trade_file(f, chunksize=chunksize, spike_threshold=spike_threshold, trade_sink=writer)
        except Exception:
            writer.abort()
            raise
        writer.close()
        if minutes.empty:
            log.warning("%s: no valid trades", key)
            continue
        store.write_frame(minute_bars_to_frame(minutes), store.minutes_path(f.symbol, f.month))
        manifest[key] = {"source": str(f.path), "fingerprint": fp, "validation": report,
                         "ingested_at": pd.Timestamp.now(tz="UTC").isoformat()}
        store.save_manifest(manifest)
        touched.add(f.symbol)
        log.info("%s: %s trades kept (%s dropped) in %.1fs", key, f"{report['kept']:,}", f"{report['dropped']:,}", time.time() - t0)
        for problem in ("bad_timestamp", "bad_price", "bad_size", "symbol_mismatch", "outside_month", "out_of_order", "unknown_side"):
            if report[problem]:
                log.warning("%s: %s = %s", key, problem, f"{report[problem]:,}")
        if report["aggressor_side_consistent"] is False:
            log.warning("%s: buyer_role mapping looks inverted (buy-aggressor prints are not above sell-aggressor prints)", key)
    return touched


def build_bars_and_reports(store: Store, symbols: list[str], timeframes: list[str], max_gap: str, round_trip_fee_bp: float) -> pd.DataFrame:
    manifest = store.load_manifest()
    rows = []
    store.reports_dir.mkdir(parents=True, exist_ok=True)
    for sym in symbols:
        minutes = complete_minute_bars(store.load_trade_minutes(sym), max_gap=max_gap)
        for tf in timeframes:
            store.write_frame(resample_bars(minutes, tf), store.bars_path(sym, tf))
        quality = minute_bar_quality(minutes, max_gap=max_gap)
        files = {k: v["validation"] for k, v in sorted(manifest.items()) if k.startswith(f"{sym}/")}
        report = {"symbol": sym, "bars": quality, "files": files}
        (store.reports_dir / f"{sym}.json").write_text(json.dumps(report, indent=1))

        spreads = [v["spread_proxy_bp_median"] for v in files.values() if v["spread_proxy_bp_median"] is not None]
        spread = float(np.median(spreads)) if spreads else float("nan")
        consistent = [v["aggressor_side_consistent"] for v in files.values() if v["aggressor_side_consistent"] is not None]
        rows.append({
            "symbol": sym,
            "months": len(files),
            "first_bar": quality["first_bar"][:16],
            "last_bar": quality["last_bar"][:16],
            "trades": sum(v["kept"] for v in files.values()),
            "dropped": sum(v["dropped"] for v in files.values()),
            "trades_per_day_med": round(quality["median_trades_per_day"]),
            "pct_min_no_trade": quality["pct_minutes_without_trades"],
            "segments": quality["segments"],
            "out_of_order": sum(v["out_of_order"] for v in files.values()),
            "dup_rows": sum(v["consecutive_duplicates"] for v in files.values()),
            "spikes": sum(v["isolated_spikes"] for v in files.values()),
            "outliers_1m": quality["return_outliers_10sigma"],
            "aggressor_ok": all(consistent) if consistent else None,
            "spread_proxy_bp": round(spread, 2),
            "taker_round_trip_bp": round(round_trip_fee_bp + spread, 2),
            "med_abs_1m_ret_bp": round(quality["median_abs_1m_return_bp"], 2),
            "ann_vol": round(quality["ann_vol_from_1m"], 3),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(store.reports_dir / "summary.csv", index=False)
    return summary


def main() -> None:
    settings = load_settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="folder containing the monthly trade archives (searched recursively)")
    p.add_argument("--root", default=str(settings.data_root), help="research store root")
    p.add_argument("--symbols", nargs="*", default=None, help="default: every symbol found in --src")
    p.add_argument("--timeframes", nargs="+", default=["1m", "5m", "15m", "1h", "4h"])
    p.add_argument("--max-gap", default="6h", help="trade gap that splits segments (must be >= largest timeframe)")
    p.add_argument("--spike-threshold", type=float, default=0.01, help="isolated single-trade excursion (log return) to count")
    p.add_argument("--chunksize", type=int, default=2_000_000)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    store = Store(args.root)
    files = discover_trade_files(args.src, args.symbols)
    if not files:
        sys.exit(f"no trade files found under {args.src}")
    log.info("found %d files for %d symbols", len(files), len({f.symbol for f in files}))
    ingest_files(store, files, args.force, args.chunksize, args.spike_threshold)

    symbols = sorted({f.symbol for f in files})
    fee_bp = 2 * settings.costs.taker_fee_effective * 1e4
    summary = build_bars_and_reports(store, symbols, args.timeframes, args.max_gap, fee_bp)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\n" + summary.to_string(index=False))
    print(f"\nreports: {store.reports_dir}  (taker_round_trip_bp = 2 x {fee_bp / 2:.2f} bp fee incl. GST + spread proxy)")


if __name__ == "__main__":
    main()
