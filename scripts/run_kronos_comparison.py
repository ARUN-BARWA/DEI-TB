"""Phase 5: is Kronos adding trading-relevant information, or is it just good at forecasting candles?

    python scripts/run_kronos_comparison.py --kronos-tag Kronos-small_emb
    python scripts/run_kronos_comparison.py --kronos-tag Kronos-mini_paths --timeframes 4h --extend

Same walk-forward folds, costs, thresholds-on-validation and evaluator as the baselines, on exactly the
rows where Kronos features exist (every arm sees the same rows). Arms: B (registry features), K (Kronos
only), BK (both), kronos_direct (Kronos's own forecast). The key output is the incremental table: does
BK beat B, fold by fold, in IC and in net trading result? Writes data_store/reports/kronos_comparison/<run_id>/.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading_bot.config import load_settings  # noqa: E402
from trading_bot.data.storage import Store  # noqa: E402
from trading_bot.models.kronos_heads import comparison_models  # noqa: E402
from trading_bot.research.dataset import round_trip_cost  # noqa: E402
from trading_bot.research.evaluate import QUANTILE_GRID, benjamini_hochberg, minute_price_lookup, t_stat  # noqa: E402
from trading_bot.research.experiments import log_experiments, rows_from_grid  # noqa: E402
from trading_bot.research.splits import monthly_folds  # noqa: E402
from trading_bot.research.walkforward import run_walk_forward, summarize  # noqa: E402


def load_joined(store: Store, sym: str, tf: str, tag: str, horizons) -> pd.DataFrame:
    ds = pd.read_parquet(store.dataset_path(sym, tf))
    kr = pd.read_parquet(store.kronos_path(tag, tf, sym))
    ds = ds.join(kr.drop(columns=["kr_close"], errors="ignore"), how="left")
    unavailable = ds["kr_entropy"].isna()
    for h in horizons:
        ds.loc[unavailable, f"label_end_{h}"] = pd.NaT  # rows without Kronos features are excluded for every arm
    return ds


def incremental_table(folds: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Paired comparison per (tf, target, h): K and BK against B, fold by fold."""
    rows = []
    for target in ("lgbm_return", "lgbm_tb"):
        for (tf, h), g in folds[folds["model"].str.endswith(target)].groupby(["tf", "h"]):
            ic = g.pivot_table(index="fold", columns="model", values="ic_test")
            base = f"B_{target}"
            if base not in ic:
                continue
            for arm in (f"K_{target}", f"BK_{target}"):
                if arm not in ic:
                    continue
                d = (ic[arm] - ic[base]).dropna()
                pooled = grid[(grid["symbol"] == "ALL") & (grid["tf"] == tf) & (grid["h"] == h)].set_index("model")
                rows.append({
                    "tf": tf, "target": target, "h": h, "arm": arm.split("_")[0],
                    "ic_B": round(ic[base].mean(), 4), "ic_arm": round(ic[arm].mean(), 4),
                    "ic_diff": round(d.mean(), 4), "ic_diff_t": round(t_stat(d.to_numpy()), 2),
                    "p_two_sided": round(float(2 * student_t.sf(abs(t_stat(d.to_numpy())), max(len(d) - 1, 1))), 4) if len(d) > 1 else np.nan,
                    "folds_arm_better": f"{int((d > 0).sum())}/{len(d)}",
                    "net_bp_B": pooled.loc[base, "mean_net_bp"] if base in pooled.index else np.nan,
                    "net_bp_arm": pooled.loc[arm, "mean_net_bp"] if arm in pooled.index else np.nan,
                    "trades_B": pooled.loc[base, "trades"] if base in pooled.index else np.nan,
                    "trades_arm": pooled.loc[arm, "trades"] if arm in pooled.index else np.nan,
                })
    table = pd.DataFrame(rows)
    if not table.empty:
        table.insert(table.columns.get_loc("p_two_sided") + 1, "q_bh", np.round(benjamini_hochberg(table["p_two_sided"]), 4))
    return table


def describe(model: str) -> tuple[str, str]:
    arm = model.split("_")[0]
    feats = {"B": "registry", "K": "kronos (scalars + train-fitted PCA of embedding)", "BK": "registry + kronos",
             "kronos": "kronos path mean"}.get(arm, arm)
    target = "triple barrier" if model.endswith("_tb") else ("none" if model == "kronos_direct" else "y_ret/(rv_48 sqrt h)")
    return feats, target


def main() -> None:
    settings = load_settings()
    rs = settings.research
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(settings.data_root))
    p.add_argument("--kronos-tag", required=True)
    p.add_argument("--symbols", nargs="+", default=list(settings.combined.symbols) or list(settings.symbols))
    p.add_argument("--timeframes", nargs="+", default=["1h", "4h"])
    p.add_argument("--horizons", nargs="+", type=int, default=list(rs.horizons))
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--lag-minutes", type=int, default=0)
    p.add_argument("--extend", action="store_true")
    p.add_argument("--funding-per-8h", type=float, default=settings.costs.funding_assumed_per_8h)
    args = p.parse_args()

    store = Store(args.root)
    folds = [f for f in monthly_folds(rs.data_start, rs.first_test_month, rs.holdout_start, rs.data_end, rs.val_months) if not f.is_holdout]
    costs = {s: round_trip_cost(store, settings.delta_symbol(s), settings.costs) for s in args.symbols}
    lookups = {s: minute_price_lookup(store.load_bars(s, "1m")) for s in args.symbols} if args.lag_minutes else None

    def factory(features):
        return [m for m in comparison_models(features) if args.models is None or m.name in args.models]

    run_id = (pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S") + f"_{args.kronos_tag}"
              + (f"_lag{args.lag_minutes}m" if args.lag_minutes else "") + ("_extend" if args.extend else ""))
    out = store.reports_dir / "kronos_comparison" / run_id
    out.mkdir(parents=True, exist_ok=True)

    grids, summaries, fold_tables, trade_tables = [], [], [], []
    n_models = 0
    for tf in args.timeframes:
        t0 = time.time()
        datasets = {s: load_joined(store, s, tf, args.kronos_tag, args.horizons) for s in args.symbols}
        n_rows = sum(int(d["kr_entropy"].notna().sum()) for d in datasets.values())
        print(f"{tf}: {n_rows:,} rows with Kronos features")
        n_models = max(n_models, len(factory(list(next(iter(datasets.values())).columns))))
        res = run_walk_forward(datasets, tf, args.horizons, folds, costs, factory, rs.embargo, args.funding_per_8h,
                               log=lambda m: print(m, end="\r", flush=True), price_lookups=lookups,
                               lag_minutes=args.lag_minutes, extend=args.extend)
        grid, summary = summarize(res, args.symbols)
        grids.append(grid), summaries.append(summary), fold_tables.append(res.folds), trade_tables.append(res.trades)
        print(f"\n{tf}: finished in {time.time() - t0:.0f}s")

    grid = pd.concat(grids, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)
    folds_df = pd.concat(fold_tables, ignore_index=True)
    summary["q_bh"] = np.round(benjamini_hochberg(summary["p_signflip"]), 4)
    inc = incremental_table(folds_df, grid)
    grid.to_csv(out / "grid.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    folds_df.to_csv(out / "folds.csv", index=False)
    inc.to_csv(out / "incremental.csv", index=False)
    pd.concat(trade_tables, ignore_index=True).to_parquet(out / "trades.parquet")

    fee_rt = 2 * settings.costs.taker_fee_effective
    n_configs = n_models * len(args.timeframes) * len(args.horizons) * len(QUANTILE_GRID)
    log_experiments(store.experiments_path, rows_from_grid(
        grid, folds_df, run_id=run_id, symbols=args.symbols, costs=costs, fee_rt=fee_rt, funding=args.funding_per_8h,
        n_configs=n_configs, describe=describe, lookback=f"kronos tag {args.kronos_tag}",
        extra_params={"kronos_tag": args.kronos_tag, "lag_minutes": args.lag_minutes, "extend": args.extend},
        notes="phase 5 Kronos comparison (B/K/BK on identical rows)"))

    cols = ["tf", "model", "h", "val_t_pooled", "ic_test", "ic_test_t", "trades", "hit_rate", "mean_gross_bp", "mean_net_bp",
            "t_net", "p_signflip", "q_bh", "sharpe_ann", "long_share", "drift_long_net_bp", "drift_short_net_bp", "folds_positive"]
    with pd.option_context("display.width", 300, "display.max_columns", 40, "display.max_rows", 500):
        print(f"\nTest results (horizon chosen by validation; {n_configs} configs):")
        print(summary[[c for c in cols if c in summary.columns]].to_string(index=False))
        print("\nIncremental value vs B (same rows, same folds; fold-paired t-test on IC, BH q across rows):")
        print(inc.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
