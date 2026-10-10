"""Walk-forward model comparison on the research datasets (shared by baselines, Kronos and ensembles).

For every horizon h and development fold:
  1. purge/embargo masks per symbol from ``label_end_h`` (see ``splits.fold_masks``);
  2. each model is fitted on the pooled training rows of all symbols (fresh instance per fold);
  3. the trading threshold is chosen on the validation rows (pooled over symbols);
  4. the test month is traded with that threshold, per symbol, and IC / rank IC are recorded.
The holdout fold is never touched here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from trading_bot.data.bars import tf_to_timedelta
from trading_bot.models.baselines import Model
from trading_bot.research.dataset import feature_columns
from trading_bot.research.evaluate import (
    TradeRules,
    choose_threshold,
    ic_t_stat,
    information_coefficient,
    simulate_trades,
    t_stat,
    trade_stats,
)
from trading_bot.research.splits import Fold, fold_masks


@dataclass
class WalkForwardResult:
    folds: pd.DataFrame  # one row per (model, h, fold): theta, val t, IC/rank IC val and test
    trades: pd.DataFrame  # every test trade (and validation trade, flagged) with model/h/fold keys
    test_days: dict  # fold name -> DatetimeIndex of calendar days with test data


def _pool(datasets: dict[str, pd.DataFrame], masks: dict[str, dict[str, np.ndarray]], part: str) -> pd.DataFrame:
    return pd.concat([d.loc[masks[s][part]].assign(symbol=s) for s, d in datasets.items()])


def run_walk_forward(
    datasets: dict[str, pd.DataFrame],
    tf: str,
    horizons,
    folds: list[Fold],
    costs: dict[str, float],
    model_factory: Callable[[list[str]], list[Model]],
    embargo: str = "12h",
    funding_per_8h: float = 0.0,
    delay: int = 0,
    log: Callable[[str], None] = print,
    price_lookups: dict | None = None,
    lag_minutes: int = 0,
    extend: bool = False,
) -> WalkForwardResult:
    """``price_lookups`` (symbol -> ``evaluate.minute_price_lookup``) with ``lag_minutes`` prices trades
    from 1-minute bars N minutes after the decision (latency / bid-ask-bounce test)."""
    if any(f.is_holdout for f in folds):
        raise ValueError("walk-forward research runs on development folds only; the holdout is evaluated separately")
    features = sorted(set.intersection(*(set(feature_columns(d)) for d in datasets.values())))
    bars_per_8h = pd.Timedelta("8h") / tf_to_timedelta(tf)
    emb = pd.Timedelta(embargo)
    fold_rows, trade_frames, test_days = [], [], {}

    for h in horizons:
        rules = {s: TradeRules(h, costs[s], funding_per_8h / bars_per_8h, delay, lag_minutes, extend) for s in datasets}
        lookups = price_lookups or {}
        for fold in folds:
            masks = {s: fold_masks(d["decision_time"], d[f"label_end_{h}"], fold, embargo) for s, d in datasets.items()}
            train, val = _pool(datasets, masks, "train"), _pool(datasets, masks, "val")
            if train.empty or val.empty:
                log(f"  {tf} h={h} {fold.name}: no train/val rows, skipped")
                continue
            days = pd.DatetimeIndex(sorted(set().union(*[
                set(pd.DatetimeIndex(d.loc[masks[s]["test"], "decision_time"]).floor("D")) for s, d in datasets.items()])))
            test_days[fold.name] = days
            for model in model_factory(features):
                model.fit(train, val, h)
                scores = {s: model.predict(d) for s, d in datasets.items()}
                val_masks = {s: masks[s]["val"] for s in datasets}
                theta, val_t = choose_threshold(datasets, scores, val_masks, rules, fold.test_start - emb,
                                                price_lookups=lookups)
                if model.reference:
                    theta = 0.0  # drift benchmark: trade every opportunity, report it whatever validation said
                row = {"model": model.name, "tf": tf, "h": h, "fold": fold.name, "theta": theta, "val_t": round(val_t, 2),
                       "abstain": bool(np.isinf(theta))}
                for part in ("val", "test"):
                    s_all = np.concatenate([scores[s][masks[s][part]] for s in datasets])
                    y_all = np.concatenate([d.loc[masks[s][part], f"y_ret_{h}"].to_numpy() for s, d in datasets.items()])
                    row[f"ic_{part}"], row[f"rank_ic_{part}"] = (round(v, 4) for v in information_coefficient(s_all, y_all))
                    row[f"n_{part}"] = int((np.isfinite(s_all) & np.isfinite(y_all)).sum())
                fold_rows.append(row)
                for part, end in (("val", fold.test_start - emb), ("test", fold.test_end)):
                    for s, d in datasets.items():
                        t = simulate_trades(d, scores[s], theta, rules[s], masks[s][part], end, symbol=s, price_at=lookups.get(s))
                        if not t.empty:
                            trade_frames.append(t.assign(model=model.name, tf=tf, h=h, fold=fold.name, part=part))
            log(f"  {tf} h={h} {fold.name}: done")
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame(
        columns=["symbol", "entry_time", "exit_time", "side", "gross", "net", "hold_bars", "model", "tf", "h", "fold", "part"])
    return WalkForwardResult(pd.DataFrame(fold_rows), trades, test_days)


def summarize(result: WalkForwardResult, symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Grid: test metrics per (model, tf, h, symbol and ALL), trades pooled over folds.
    Summary: per (model, tf), the horizon with the best pooled *validation* t-stat, with its test metrics."""
    folds, trades = result.folds, result.trades
    all_days = pd.DatetimeIndex(sorted(set().union(*[set(d) for d in result.test_days.values()]))) if result.test_days else pd.DatetimeIndex([])
    test = trades[trades["part"] == "test"] if not trades.empty else trades
    val = trades[trades["part"] == "val"] if not trades.empty else trades
    grid = []
    for (model, tf, h), g in folds.groupby(["model", "tf", "h"], sort=False):
        tt = test[(test["model"] == model) & (test["tf"] == tf) & (test["h"] == h)]
        vv = val[(val["model"] == model) & (val["tf"] == tf) & (val["h"] == h)]
        ic_pooled = float((g["ic_test"] * g["n_test"]).sum() / max(g["n_test"].sum(), 1)) if g["ic_test"].notna().any() else float("nan")
        base = {"model": model, "tf": tf, "h": h,
                "ic_test": round(g["ic_test"].mean(), 4), "rank_ic_test": round(g["rank_ic_test"].mean(), 4),
                "ic_test_t": round(ic_t_stat(ic_pooled, int(g["n_test"].sum()), int(h)), 2),
                "ic_val": round(g["ic_val"].mean(), 4), "folds_abstained": int(g["abstain"].sum()),
                "val_t_pooled": round(t_stat(vv["net"].to_numpy()), 2) if len(vv) else 0.0}
        fold_means = tt.groupby("fold")["net"].mean()
        base["folds_positive"] = f"{int((fold_means > 0).sum())}/{len(g)}"
        for sym in ["ALL"] + list(symbols):
            sub = tt if sym == "ALL" else tt[tt["symbol"] == sym]
            grid.append({**base, "symbol": sym, **trade_stats(sub, all_days)})
    grid = pd.DataFrame(grid)
    if grid.empty:
        return grid, grid
    pooled = grid[grid["symbol"] == "ALL"]
    best = pooled.loc[pooled.groupby(["model", "tf"])["val_t_pooled"].idxmax()]
    return grid, best.reset_index(drop=True)
