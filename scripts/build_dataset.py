"""Build research datasets (features + labels) from the ingested bars, and the cost-hurdle report.

    python scripts/build_dataset.py                                   # settings.yaml symbols x timeframes
    python scripts/build_dataset.py --symbols BTCUSD --timeframes 15m 1h

Writes data_store/datasets/<tf>/<SYMBOL>.parquet, data_store/reports/cost_hurdle.csv and
data_store/reports/folds.csv. Every statistic printed here uses the development period only
(labels known before the holdout starts).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading_bot.config import load_settings  # noqa: E402
from trading_bot.data.bars import tf_to_timedelta  # noqa: E402
from trading_bot.data.storage import Store  # noqa: E402
from trading_bot.research.dataset import build_dataset, cost_hurdle_table, feature_columns, round_trip_cost  # noqa: E402
from trading_bot.research.splits import fold_masks, monthly_folds  # noqa: E402


def main() -> None:
    settings = load_settings()
    rs = settings.research
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(settings.data_root))
    p.add_argument("--symbols", nargs="+", default=list(settings.symbols))
    p.add_argument("--timeframes", nargs="+", default=list(settings.timeframes))
    args = p.parse_args()

    store = Store(args.root)
    folds = monthly_folds(rs.data_start, rs.first_test_month, rs.holdout_start, rs.data_end, rs.val_months)
    hurdles, fold_rows = [], []
    for sym in args.symbols:
        cost = round_trip_cost(store, settings.delta_symbol(sym), settings.costs)
        path = store.load_bars(sym, "1m")
        for tf in args.timeframes:
            t0 = time.time()
            bars = store.load_bars(sym, tf)
            ds = build_dataset(bars, tf, rs.horizons, cost=cost, path=path if tf != "1m" else None,
                               tb_k=rs.triple_barrier_k, vol_window=rs.vol_window)
            store.write_frame(ds, store.dataset_path(sym, tf))
            print(f"{sym} {tf}: {len(ds):,} rows, {len(feature_columns(ds))} features, cost {cost * 1e4:.2f} bp, {time.time() - t0:.1f}s")

            h_tbl = cost_hurdle_table(ds, rs.horizons, cost, rs.holdout_start)
            step = tf_to_timedelta(tf)
            h_tbl.insert(0, "horizon", [str(step * h).replace("0 days ", "") for h in h_tbl["h_bars"]])
            h_tbl.insert(0, "tf", tf)
            h_tbl.insert(0, "symbol", sym)
            h_tbl.insert(3, "cost_bp", round(cost * 1e4, 2))
            hurdles.append(h_tbl)

            h_max = max(rs.horizons)
            for fold in folds:
                if fold.is_holdout:
                    continue  # sample counts for the holdout are not needed until the final evaluation
                m = fold_masks(ds["decision_time"], ds[f"label_end_{h_max}"], fold, rs.embargo)
                fold_rows.append({"symbol": sym, "tf": tf, "fold": fold.name, **{k: int(v.sum()) for k, v in m.items()}})

    hurdle = pd.concat(hurdles, ignore_index=True)
    store.reports_dir.mkdir(parents=True, exist_ok=True)
    hurdle.to_csv(store.reports_dir / "cost_hurdle.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(store.reports_dir / "folds.csv", index=False)

    print("\nFolds (expanding train, purged + embargo; holdout locked):")
    for f in folds:
        print(f"  {f.name:12s} train {f.train_start:%Y-%m-%d}..{f.val_start:%Y-%m-%d}  val ..{f.test_start:%Y-%m-%d}  "
              f"test {f.test_start:%Y-%m-%d}..{f.test_end:%Y-%m-%d}{'  [LOCKED]' if f.is_holdout else ''}")
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 500):
        print("\nCost hurdle (development period): how often |move over the horizon| exceeds a taker round trip")
        print(hurdle.to_string(index=False))
    print(f"\nwrote {store.reports_dir / 'cost_hurdle.csv'} and {store.reports_dir / 'folds.csv'}")


if __name__ == "__main__":
    main()
