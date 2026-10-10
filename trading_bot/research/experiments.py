"""Append-only experiment log (experiments.csv), one row per evaluated configuration."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable

import pandas as pd

COLUMNS = [
    "experiment_id", "date", "git_commit", "run_id", "model", "features", "target", "lookback", "horizon",
    "timeframe", "symbols", "parameters", "cv_metrics", "walk_forward_metrics", "backtest_metrics",
    "fees_bp", "slippage_bp", "funding_bp_per_8h", "max_drawdown_pct", "sharpe", "n_configs_tested", "notes",
]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parents[2], timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def log_experiments(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    start = 0
    if path.exists():
        start = len(pd.read_csv(path, usecols=["experiment_id"]))
    out = []
    for i, r in enumerate(rows):
        rec = {c: r.get(c, "") for c in COLUMNS}
        rec["experiment_id"] = f"E{start + i + 1:05d}"
        rec["date"] = rec["date"] or pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
        rec["git_commit"] = rec["git_commit"] or git_commit()
        for k in ("parameters", "cv_metrics", "walk_forward_metrics", "backtest_metrics"):
            if isinstance(rec[k], (dict, list)):
                rec[k] = json.dumps(rec[k], default=str, sort_keys=True)
        out.append(rec)
    pd.DataFrame(out, columns=COLUMNS).to_csv(path, mode="a", header=not path.exists(), index=False)


def rows_from_grid(grid: pd.DataFrame, folds: pd.DataFrame, *, run_id: str, symbols, costs: dict, fee_rt: float,
                   funding: float, n_configs: int, notes: str, describe: Callable[[str], tuple[str, str]],
                   extra_params: dict | None = None, lookback: str = "") -> list[dict]:
    """One experiments.csv row per (model, tf, h) from the pooled ('ALL') rows of a walk-forward grid.

    ``describe(model) -> (features, target)``.
    """
    rows = []
    for _, r in grid[grid["symbol"] == "ALL"].iterrows():
        f = folds[(folds["model"] == r["model"]) & (folds["tf"] == r["tf"]) & (folds["h"] == r["h"])]
        feats, target = describe(r["model"])
        rows.append({
            "run_id": run_id, "model": r["model"], "features": feats, "target": target, "lookback": lookback,
            "horizon": int(r["h"]), "timeframe": r["tf"], "symbols": " ".join(symbols),
            "parameters": {**(extra_params or {}), "theta_by_fold": dict(zip(f["fold"], f["theta"].round(6)))},
            "cv_metrics": {"ic_val": r["ic_val"], "val_t_pooled": r["val_t_pooled"]},
            "walk_forward_metrics": {k: r[k] for k in ("ic_test", "ic_test_t", "rank_ic_test", "folds_positive") if k in r},
            "backtest_metrics": {k: r[k] for k in ("trades", "trades_per_day", "hit_rate", "mean_gross_bp", "mean_net_bp", "t_net",
                                                    "total_net_pct", "p_signflip", "long_share") if k in r},
            "fees_bp": round(fee_rt * 1e4, 2), "slippage_bp": {s: round((c - fee_rt) * 1e4, 2) for s, c in costs.items()},
            "funding_bp_per_8h": funding * 1e4, "max_drawdown_pct": r["max_dd_pct"], "sharpe": r["sharpe_ann"],
            "n_configs_tested": n_configs, "notes": notes,
        })
    return rows
