import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tests.test_features import synth_minute_bars
from trading_bot.data.bars import resample_bars
from trading_bot.data.storage import Store
from trading_bot.research.dataset import build_dataset, cost_hurdle_table, feature_columns, label_columns

REPO = Path(__file__).resolve().parents[1]
HOLDOUT = "2026-07-01"


def _bars(minutes, tf):
    return resample_bars(minutes, tf)


def test_dataset_layout():
    m = synth_minute_bars(n_minutes=3000, start="2026-03-01")
    b = _bars(m, "15m")
    ds = build_dataset(b, "15m", horizons=(1, 4), cost=0.0012, path=m)
    assert (ds["decision_time"] == ds.index + pd.Timedelta("15min")).all()
    feats, labels = feature_columns(ds), label_columns(ds)
    assert len(feats) >= 40 and not set(feats) & set(labels)
    assert not any(c.startswith(("y_", "label_end_")) for c in feats)
    assert {"y_ret_4", "y_tb_4", "label_end_4", "y_mfe_1"} <= set(labels)


def test_hurdle_table_never_sees_the_holdout():
    m = synth_minute_bars(n_minutes=6000, start="2026-06-28", gap_at=10_000)
    b = _bars(m, "5m")
    base = cost_hurdle_table(build_dataset(b, "5m", (1, 6), 0.0012, path=m), (1, 6), 0.0012, HOLDOUT)

    m2, b2 = m.copy(), b.copy()
    after_m, after_b = m2.index >= pd.Timestamp(HOLDOUT, tz="UTC"), b2.index >= pd.Timestamp(HOLDOUT, tz="UTC")
    assert after_b.any() and (~after_b).any()
    for frame, mask in ((m2, after_m), (b2, after_b)):
        frame.loc[mask, ["open", "high", "low", "close"]] *= np.exp(np.linspace(0.05, 0.5, mask.sum()))[:, None]
    changed = cost_hurdle_table(build_dataset(b2, "5m", (1, 6), 0.0012, path=m2), (1, 6), 0.0012, HOLDOUT)
    pd.testing.assert_frame_equal(base, changed)
    assert base["n"].iloc[0] < (~after_b).sum()


def test_build_dataset_script(tmp_path):
    store = Store(tmp_path)
    m = synth_minute_bars(n_minutes=4000, start="2026-02-20")
    for tf in ("1m", "5m", "15m"):
        store.write_frame(resample_bars(m, tf), store.bars_path("BTCUSD", tf))
    cmd = [sys.executable, str(REPO / "scripts" / "build_dataset.py"), "--root", str(tmp_path),
           "--symbols", "BTCUSD", "--timeframes", "5m", "15m"]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert res.returncode == 0, res.stderr
    assert store.dataset_path("BTCUSD", "15m").exists()
    hurdle = pd.read_csv(store.reports_dir / "cost_hurdle.csv")
    assert set(hurdle["tf"]) == {"5m", "15m"} and (hurdle["cost_bp"] > 11).all()
    folds = pd.read_csv(store.reports_dir / "folds.csv")
    assert "holdout" not in set(folds["fold"])
    assert "LOCKED" in res.stdout
