"""Phase 4: walk-forward evaluation of the baseline models, net of costs.

    python scripts/run_baselines.py                          # 5m, 15m, 1h, 4h x settings symbols x horizons
    python scripts/run_baselines.py --timeframes 1h 4h --models ridge lgbm_return
    python scripts/run_baselines.py --delay 1                # robustness: enter one bar late
    python scripts/run_baselines.py --lag-minutes 2          # latency / bid-ask bounce: price trades 2 min late
    python scripts/run_baselines.py --symbols BTC ETH SOL DOGE   # combined Binance + Delta series

Development folds only (the holdout stays locked). Writes data_store/reports/baselines/<run_id>/
(summary.csv, grid.csv, folds.csv, trades.parquet) and appends to data_store/experiments/experiments.csv.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading_bot.config import load_settings  # noqa: E402
from trading_bot.data.storage import Store  # noqa: E402
from trading_bot.models.baselines import all_baselines  # noqa: E402
from trading_bot.research.dataset import round_trip_cost  # noqa: E402
from trading_bot.research.experiments import log_experiments  # noqa: E402
from trading_bot.research.evaluate import QUANTILE_GRID, benjamini_hochberg, minute_price_lookup  # noqa: E402
from trading_bot.research.splits import monthly_folds  # noqa: E402
from trading_bot.research.walkforward import run_walk_forward, summarize  # noqa: E402


def experiment_rows(summary_grid: pd.DataFrame, folds: pd.DataFrame, run_id: str, symbols, costs, fee_rt, funding, n_configs, delay):
    rows = []
    pooled = summary_grid[summary_grid["symbol"] == "ALL"]
    for _, r in pooled.iterrows():
        f = folds[(folds["model"] == r["model"]) & (folds["tf"] == r["tf"]) & (folds["h"] == r["h"])]
        rows.append({
            "run_id": run_id, "model": r["model"], "features": "registry (all)" if r["model"] in (
                "ridge", "logistic_tb", "lgbm_return", "lgbm_tb") else "rule",
            "target": "y_ret/(rv_48 sqrt h)" if r["model"] in ("ridge", "lgbm_return") else (
                "triple barrier" if r["model"] in ("logistic_tb", "lgbm_tb") else "none"),
            "lookback": "<=192 bars", "horizon": int(r["h"]), "timeframe": r["tf"], "symbols": " ".join(symbols),
            "parameters": {"delay": delay, "theta_by_fold": dict(zip(f["fold"], f["theta"].round(6)))},
            "cv_metrics": {"ic_val": r["ic_val"], "val_t_pooled": r["val_t_pooled"]},
            "walk_forward_metrics": {"ic_test": r["ic_test"], "rank_ic_test": r["rank_ic_test"], "folds_positive": r["folds_positive"]},
            "backtest_metrics": {k: r[k] for k in ("trades", "trades_per_day", "hit_rate", "mean_gross_bp", "mean_net_bp", "t_net",
                                                    "total_net_pct", "p_signflip", "long_share") if k in r},
            "fees_bp": round(fee_rt * 1e4, 2), "slippage_bp": {s: round((c - fee_rt) * 1e4, 2) for s, c in costs.items()},
            "funding_bp_per_8h": funding * 1e4, "max_drawdown_pct": r["max_dd_pct"], "sharpe": r["sharpe_ann"],
            "n_configs_tested": n_configs, "notes": "phase 4 baseline; research evaluator (taker, fixed horizon, non-overlapping)",
        })
    return rows


def main() -> None:
    settings = load_settings()
    rs = settings.research
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(settings.data_root))
    p.add_argument("--symbols", nargs="+", default=list(settings.symbols))
    p.add_argument("--timeframes", nargs="+", default=["5m", "15m", "1h", "4h"])
    p.add_argument("--horizons", nargs="+", type=int, default=list(rs.horizons))
    p.add_argument("--models", nargs="*", default=None, help="subset of model names")
    p.add_argument("--delay", type=int, default=0, help="bars between decision and entry")
    p.add_argument("--lag-minutes", type=int, default=0, help="price entries/exits N minutes late from 1m bars")
    p.add_argument("--funding-per-8h", type=float, default=settings.costs.funding_assumed_per_8h)
    args = p.parse_args()

    store = Store(args.root)
    folds = [f for f in monthly_folds(rs.data_start, rs.first_test_month, rs.holdout_start, rs.data_end, rs.val_months) if not f.is_holdout]
    costs = {s: round_trip_cost(store, settings.delta_symbol(s), settings.costs) for s in args.symbols}

    def factory(features):
        models = all_baselines(features)
        return [m for m in models if args.models is None or m.name in args.models]

    n_models = len(factory(["ret_1"]))
    n_configs = n_models * len(args.timeframes) * len(args.horizons) * len(QUANTILE_GRID)
    run_id = (pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S") + (f"_delay{args.delay}" if args.delay else "")
              + (f"_lag{args.lag_minutes}m" if args.lag_minutes else ""))
    lookups = {s: minute_price_lookup(store.load_bars(s, "1m")) for s in args.symbols} if args.lag_minutes else None
    out = store.reports_dir / "baselines" / run_id
    out.mkdir(parents=True, exist_ok=True)

    grids, summaries, fold_tables, trade_tables = [], [], [], []
    for tf in args.timeframes:
        t0 = time.time()
        datasets = {s: pd.read_parquet(store.dataset_path(s, tf)) for s in args.symbols}
        print(f"{tf}: {sum(len(d) for d in datasets.values()):,} rows")
        res = run_walk_forward(datasets, tf, args.horizons, folds, costs, factory, rs.embargo,
                               args.funding_per_8h, args.delay, log=print, price_lookups=lookups, lag_minutes=args.lag_minutes)
        grid, summary = summarize(res, args.symbols)
        grids.append(grid), summaries.append(summary), fold_tables.append(res.folds), trade_tables.append(res.trades)
        print(f"{tf}: finished in {time.time() - t0:.0f}s")

    grid = pd.concat(grids, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)
    # False-discovery control across every strategy row shown (reference models are not tests of skill).
    strat = ~summary["model"].isin(["always_long", "always_short"])
    summary["q_bh"] = np.nan
    summary.loc[strat, "q_bh"] = np.round(benjamini_hochberg(summary.loc[strat, "p_signflip"]), 4)
    folds_df = pd.concat(fold_tables, ignore_index=True)
    trades = pd.concat(trade_tables, ignore_index=True)
    grid.to_csv(out / "grid.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    folds_df.to_csv(out / "folds.csv", index=False)
    trades.to_parquet(out / "trades.parquet")
    fee_rt = 2 * settings.costs.taker_fee_effective
    log_experiments(store.experiments_path, experiment_rows(grid, folds_df, run_id, args.symbols, costs, fee_rt,
                                                             args.funding_per_8h, n_configs, args.delay))

    cols = ["tf", "model", "h", "val_t_pooled", "ic_test", "ic_test_t", "rank_ic_test", "trades", "trades_per_day", "hit_rate",
            "mean_gross_bp", "mean_net_bp", "t_net", "p_signflip", "q_bh", "sharpe_ann", "max_dd_pct", "long_share",
            "drift_long_net_bp", "drift_short_net_bp", "folds_positive"]
    with pd.option_context("display.width", 300, "display.max_columns", 40, "display.max_rows", 500):
        print(f"\nTest results, all symbols pooled, horizon chosen per model by VALIDATION t-stat ({n_configs} configs tried):")
        print(summary[[c for c in cols if c in summary.columns]].to_string(index=False))
    print(f"\nwrote {out}\nlogged {len(grid[grid['symbol'] == 'ALL'])} rows to {store.experiments_path}")


if __name__ == "__main__":
    main()
