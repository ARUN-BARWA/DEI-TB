import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_bot.data.storage import Store
from trading_bot.models.baselines import RidgeReturn, all_baselines
from trading_bot.research.dataset import build_dataset
from trading_bot.research.evaluate import (
    TradeRules,
    choose_threshold,
    information_coefficient,
    sign_flip_pvalue,
    simulate_trades,
    trade_stats,
)
from trading_bot.research.experiments import log_experiments
from trading_bot.research.splits import monthly_folds
from trading_bot.research.walkforward import run_walk_forward, summarize

REPO = Path(__file__).resolve().parents[1]
COST = 0.0012


def frame(closes, segments=None, start="2026-03-01"):
    n = len(closes)
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"close": np.asarray(closes, float),
                         "segment_id": np.zeros(n, int) if segments is None else segments,
                         "decision_time": idx + pd.Timedelta("1h")}, index=idx)


def test_simulate_trades_rules():
    f = frame([100, 101, 102, 103, 104, 105, 106, 107])
    s = np.array([1, 1, 1, -1, 0, 1, 1, 1], float)
    t = simulate_trades(f, s, theta=0.5, rules=TradeRules(h=2, round_trip_cost=0.001, funding_per_bar=0.0001))
    assert list(t["entry_time"]) == list(f["decision_time"].iloc[[0, 2, 5]])        # non-overlapping, re-entry at exit bar
    assert list(t["side"]) == [1, 1, 1]
    assert t["gross"].iloc[0] == pytest.approx(np.log(102 / 100))
    assert t["net"].iloc[0] == pytest.approx(np.log(102 / 100) - 0.001 - 0.0002)
    late = simulate_trades(f, s, 0.5, TradeRules(h=2, round_trip_cost=0.0, delay=1))
    assert late["gross"].iloc[0] == pytest.approx(np.log(103 / 101))                  # entry one bar later
    mask = np.zeros(8, bool)
    mask[3] = True
    short = simulate_trades(f, s, 0.5, TradeRules(h=2, round_trip_cost=0.0), entry_mask=mask)
    assert len(short) == 1 and short["side"].iloc[0] == -1 and short["gross"].iloc[0] == pytest.approx(-np.log(105 / 103))


def test_trades_never_span_a_data_hole_or_the_exit_limit():
    f = frame([100] * 10, segments=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    s = np.ones(10)
    t = simulate_trades(f, s, 0.5, TradeRules(h=2, round_trip_cost=0.0))
    assert set(t["entry_time"]) == set(f["decision_time"].iloc[[0, 3, 5, 7]])
    capped = simulate_trades(f, s, 0.5, TradeRules(h=2, round_trip_cost=0.0), max_exit_time=f["decision_time"].iloc[6])
    assert capped["exit_time"].max() <= f["decision_time"].iloc[6]


def test_sign_flip_pvalue_detects_direction_skill():
    rng = np.random.default_rng(0)
    raw = rng.normal(0, 0.01, 400)
    skilled = pd.DataFrame({"side": np.sign(raw + rng.normal(0, 0.01, 400)), "gross": 0.0, "net": 0.0})
    skilled["gross"] = skilled["side"] * raw
    skilled["net"] = skilled["gross"] - COST
    assert sign_flip_pvalue(skilled) < 0.01
    random = skilled.copy()
    random["side"] = rng.choice([-1.0, 1.0], 400)
    random["gross"] = random["side"] * raw
    random["net"] = random["gross"] - COST
    assert sign_flip_pvalue(random) > 0.05


def test_trade_stats_and_drift_benchmarks():
    days = pd.date_range("2026-03-01", periods=3, freq="D", tz="UTC")
    t = pd.DataFrame({"exit_time": [days[0], days[0], days[2]], "side": [1.0, -1.0, 1.0],
                      "gross": [0.02, 0.01, -0.01]})
    t["net"] = t["gross"] - 0.001
    st = trade_stats(t, days)
    assert st["trades"] == 3 and st["hit_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert st["mean_net_bp"] == pytest.approx((0.019 + 0.009 - 0.011) / 3 * 1e4, abs=0.01)
    # underlying long returns 0.02, -0.01, -0.01
    assert st["drift_long_net_bp"] == pytest.approx(((0.02 - 0.01 - 0.01) / 3 - 0.001) * 1e4, abs=0.01)
    assert st["max_dd_pct"] == pytest.approx(-1.1, abs=1e-6)


def test_information_coefficient():
    x = np.arange(100.0)
    assert information_coefficient(x, 2 * x + 1) == pytest.approx((1.0, 1.0))
    assert all(np.isnan(v) for v in information_coefficient(np.ones(100), x))


# ---------------------------------------------------------------- end-to-end with a planted signal


def planted_dataset(seed: int, strength: float, n_hours: int = 24 * 181) -> pd.DataFrame:
    """Hourly bars Jan-Jun 2026 whose next-hour return depends on a known feature `planted` (if strength > 0)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n_hours)
    r = np.r_[0.0, 0.004 * (strength * x[:-1] + rng.normal(0, 1, n_hours - 1))]
    close = 30_000 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]]
    idx = pd.date_range("2026-01-01", periods=n_hours, freq="1h", tz="UTC", name="timestamp")
    vol = rng.lognormal(5, 0.5, n_hours)
    bars = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.001, "low": np.minimum(open_, close) * 0.999,
                         "close": close, "volume": vol, "amount": vol * close, "n_trades": rng.integers(50, 500, n_hours),
                         "buy_volume": vol * 0.5, "sell_volume": vol * 0.5, "vwap": close, "is_filled": False,
                         "segment_id": 0}, index=idx)
    ds = build_dataset(bars, "1h", horizons=(1, 3), cost=COST)
    ds["planted"] = x
    return ds


def _run(strength: float):
    datasets = {"AAA": planted_dataset(1, strength), "BBB": planted_dataset(2, strength)}
    folds = [f for f in monthly_folds("2026-01-01", "2026-03", "2026-07-01", "2026-09-02") if not f.is_holdout]
    res = run_walk_forward(datasets, "1h", (1,), folds, {"AAA": COST, "BBB": COST},
                           lambda feats: [RidgeReturn(feats)] + [m for m in all_baselines(feats) if m.name in ("lgbm_return", "always_long")],
                           log=lambda *_: None)
    return summarize(res, ["AAA", "BBB"])


def test_walk_forward_finds_a_planted_signal():
    grid, summary = _run(strength=0.35)
    for name in ("ridge", "lgbm_return"):
        r = summary[summary["model"] == name].iloc[0]
        assert r["ic_test"] > 0.15, name
        assert r["mean_net_bp"] > 0 and r["p_signflip"] < 0.01, name
        assert r["folds_positive"] == "4/4", name


def test_reference_models_always_trade():
    grid, summary = _run(strength=0.0)
    r = summary[summary["model"] == "always_long"].iloc[0]
    assert r["trades"] > 1000 and r["long_share"] == 1.0
    assert r["mean_net_bp"] == pytest.approx(r["mean_gross_bp"] - COST * 1e4, abs=0.01)


def test_walk_forward_does_not_invent_an_edge_in_noise():
    grid, summary = _run(strength=0.0)
    for name in ("ridge", "lgbm_return"):
        r = summary[summary["model"] == name].iloc[0]
        assert abs(r["ic_test"]) < 0.05, name
        assert r["trades"] == 0 or r["p_signflip"] > 0.01, name


def test_experiment_log_appends(tmp_path):
    path = tmp_path / "experiments.csv"
    log_experiments(path, [{"model": "a", "parameters": {"x": 1}}])
    log_experiments(path, [{"model": "b"}, {"model": "c"}])
    df = pd.read_csv(path)
    assert list(df["experiment_id"]) == ["E00001", "E00002", "E00003"]
    assert df["parameters"].iloc[0] == '{"x": 1}'


def test_run_baselines_script(tmp_path):
    store = Store(tmp_path)
    for i, sym in enumerate(("AAA", "BBB")):
        store.write_frame(planted_dataset(i + 1, 0.35), store.dataset_path(sym, "1h"))
    cmd = [sys.executable, str(REPO / "scripts" / "run_baselines.py"), "--root", str(tmp_path), "--symbols", "AAA", "BBB",
           "--timeframes", "1h", "--horizons", "1", "--models", "always_long", "momentum_12", "ridge"]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert res.returncode == 0, res.stderr
    runs = list((tmp_path / "reports" / "baselines").iterdir())
    assert len(runs) == 1 and (runs[0] / "summary.csv").exists() and (runs[0] / "trades.parquet").exists()
    exp = pd.read_csv(store.experiments_path)
    assert set(exp["model"]) == {"always_long", "momentum_12", "ridge"}
    assert "holdout" not in pd.read_csv(runs[0] / "folds.csv")["fold"].unique()
