"""Append-only experiment log (experiments.csv), one row per evaluated configuration."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

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
