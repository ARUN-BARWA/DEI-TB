import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_tiny_kronos
from tests.test_evaluate_baselines import planted_dataset
from trading_bot.data.storage import Store
from trading_bot.models import kronos_features as kf
from trading_bot.models.baselines import Rule
from trading_bot.models.kronos_heads import FeatureSetModel, KronosDirect, comparison_models
from trading_bot.models.baselines import LGBMReturn, RidgeReturn
from trading_bot.models.kronos_model import KronosWrapper, normalize_windows, prepare_windows, summarize_paths
from trading_bot.research.dataset import build_dataset
from trading_bot.research.evaluate import TradeRules, simulate_trades

REPO = Path(__file__).resolve().parents[1]


def hourly_bars(n=600, seed=0, start="2024-07-01", gap_at=None):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC", name="timestamp")
    seg = np.zeros(n, int)
    if gap_at is not None:
        idx = idx.where(np.arange(n) < gap_at, idx + pd.Timedelta("2D"))
        seg[gap_at:] = 1
    close = 60_000 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    open_ = np.r_[close[0], close[:-1]]
    vol = rng.lognormal(5, 0.5, n)
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.002, "low": np.minimum(open_, close) * 0.998,
                         "close": close, "volume": vol, "amount": vol * close, "n_trades": rng.integers(50, 500, n),
                         "buy_volume": vol * 0.55, "sell_volume": vol * 0.45, "vwap": close, "is_filled": False,
                         "segment_id": seg}, index=idx)


@pytest.fixture(scope="module")
def wrapper():
    return KronosWrapper(*make_tiny_kronos(), device="cpu", max_context=2048, max_batch_rows=64, vram_headroom_mb=None)


def test_normalize_windows_matches_prepare_windows():
    b = hourly_bars(200)
    wins = [b.iloc[i - 48:i][["open", "high", "low", "close", "volume", "amount"]] for i in (60, 120, 199)]
    ref = prepare_windows([w.reset_index(drop=True) for w in wins], [pd.Series(w.index) for w in wins])
    x, mean, std = normalize_windows(np.stack([w.to_numpy(np.float32) for w in wins]))
    np.testing.assert_allclose(x, ref.x, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(mean, ref.mean, rtol=1e-6)


def test_path_stats_batch_equals_summarize_paths():
    rng = np.random.default_rng(0)
    paths = 100 * np.exp(rng.normal(0, 0.01, (3, 8, 6, 6)))
    last = np.array([100.0, 101.0, 99.0])
    batch = kf.path_stats_batch(paths, last, [1, 3, 6])
    for i in range(3):
        one = summarize_paths(paths[i], last[i], [1, 3, 6])
        for k, v in one.items():
            assert batch[k][i] == pytest.approx(v, rel=1e-9, abs=1e-12), k


def test_kronos_features_warmup_segments_and_paths(wrapper):
    b = hourly_bars(300, gap_at=150)
    f = kf.kronos_features(wrapper, b, "1h", lookback=32, with_paths=True, pred_len=3, n_samples=4, chunk=50, log=lambda m: None)
    avail = f["kr_entropy"].notna().to_numpy()
    assert not avail[:31].any() and avail[31:150].all()           # warm-up in segment 0
    assert not avail[150:181].any() and avail[181:].all()         # warm-up restarts after the hole
    assert {"kr_emb_0", "kr_recon", "kr_r1_mean", "kr_r3_p_up", "kr_r3_q95"} <= set(f.columns)
    assert "kr_r6_mean" not in f.columns                          # horizons beyond pred_len are dropped
    np.testing.assert_allclose(f["kr_close"].dropna(), b["close"][avail])


def test_kronos_embedding_features_are_causal(wrapper):
    b = hourly_bars(240)
    t = 150
    future = b.copy()
    # A ramp, not a constant: Kronos z-scores each window, so a uniform rescale would be invisible.
    future.iloc[t + 1:, :4] *= np.linspace(1.0, 1.5, len(b) - t - 1)[:, None]
    a = kf.kronos_features(wrapper, b, "1h", lookback=48, chunk=40, log=lambda m: None)
    c = kf.kronos_features(wrapper, future, "1h", lookback=48, chunk=40, log=lambda m: None)
    cols = [col for col in a.columns if not col.startswith("kr_r")]
    pd.testing.assert_frame_equal(a.loc[a.index[: t + 1], cols], c.loc[c.index[: t + 1], cols])
    assert not np.allclose(a["kr_entropy"].iloc[t + 60], c["kr_entropy"].iloc[t + 60])


def test_kronos_feature_cache_resume_and_staleness(wrapper, tmp_path, monkeypatch):
    b = hourly_bars(200)
    first = kf.kronos_features(wrapper, b, "1h", 32, chunk=64, cache_dir=tmp_path, log=lambda m: None)
    assert len(list(tmp_path.glob("*.parquet"))) == 3
    calls = []
    real = kf._chunk_features
    monkeypatch.setattr(kf, "_chunk_features", lambda *a, **k: calls.append(1) or real(*a, **k))
    again = kf.kronos_features(wrapper, b, "1h", 32, chunk=64, cache_dir=tmp_path, log=lambda m: None)
    pd.testing.assert_frame_equal(first, again)
    assert calls == []                                            # everything came from the cache
    changed = b.copy()
    changed.iloc[100:, 3] *= 1.01                                 # bars changed after caching
    kf.kronos_features(wrapper, changed, "1h", 32, chunk=64, cache_dir=tmp_path, log=lambda m: None)
    assert len(calls) == 2                                        # the two chunks touching changed closes recomputed


def _kronos_frame(n=800, seed=0):
    rng = np.random.default_rng(seed)
    df = planted_dataset(seed, 0.0, n_hours=n)
    for i in range(20):
        df[f"kr_emb_{i}"] = rng.normal(0, 1, n)
    df["kr_entropy"] = rng.uniform(0, 1, n)
    df["kr_r3_mean"] = rng.normal(0, 0.01, n)
    df["kr_r12_mean"] = rng.normal(0, 0.01, n)
    return df


def test_feature_sets_use_only_their_inputs_and_pca_is_fit_on_train():
    df = _kronos_frame()
    feats = [c for c in df.columns if not c.startswith(("y_", "label_end_")) and c not in ("decision_time", "segment_id", "is_filled", "close")]
    models = {m.name: m for m in comparison_models(feats)}
    assert not any(c.startswith("kr_") for c in models["B_lgbm_return"].base) and not models["B_lgbm_return"].emb
    assert all(c.startswith("kr_") for c in models["K_lgbm_return"].base) and models["K_lgbm_return"].emb
    assert any(c.startswith("kr_") for c in models["BK_lgbm_tb"].base) and any(not c.startswith("kr_") for c in models["BK_lgbm_tb"].base)
    train, val = df.iloc[:500], df.iloc[500:650]
    m = FeatureSetModel("K_ridge", RidgeReturn, feats, use_registry=False, use_kronos=True, n_pca=4).fit(train, val, 1)
    comps = m.pca_.components_.copy()
    other = df.copy()
    other.loc[other.index[500:], [f"kr_emb_{i}" for i in range(20)]] *= 50       # val/test embeddings change
    m2 = FeatureSetModel("K_ridge", RidgeReturn, feats, use_registry=False, use_kronos=True, n_pca=4).fit(other.iloc[:500], other.iloc[500:650], 1)
    np.testing.assert_allclose(m2.pca_.components_, comps)
    assert np.isfinite(m.predict(df)).sum() > 700
    lg = FeatureSetModel("BK_lgbm_return", LGBMReturn, feats, use_registry=True, use_kronos=True, n_pca=4).fit(train, val, 1)
    assert "kr_pc_0" in lg.learner_.features


def test_kronos_direct_uses_nearest_horizon():
    df = _kronos_frame(200)
    d = KronosDirect(list(df.columns)).fit(df, df, 24)
    assert d.h == 12 and np.allclose(d.predict(df), df["kr_r12_mean"])
    assert KronosDirect(list(df.columns)).fit(df, df, 3).h == 3


def test_hold_extension_saves_round_trips():
    n = 12
    idx = pd.date_range("2026-03-01", periods=n, freq="1h", tz="UTC")
    f = pd.DataFrame({"close": 100 + np.arange(n, dtype=float), "segment_id": 0, "decision_time": idx + pd.Timedelta("1h")}, index=idx)
    s = np.array([1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 0, 0], float)
    plain = simulate_trades(f, s, 0.5, TradeRules(h=2, round_trip_cost=0.001))
    ext = simulate_trades(f, s, 0.5, TradeRules(h=2, round_trip_cost=0.001, extend=True))
    assert len(plain) == 5 and len(ext) == 2
    assert ext["hold_bars"].tolist() == [6, 4]                    # long 0->6, then short 6->10
    assert ext["gross"].iloc[0] == pytest.approx(np.log(106 / 100))
    assert ext["net"].sum() > plain["net"].sum()                  # fewer round trips
    assert ext["gross"].sum() == pytest.approx(plain["gross"].sum())


def test_signed_rule_learns_direction_on_train_only():
    df = planted_dataset(3, 0.0, n_hours=1500)
    df["y_ret_1"] = -0.001 * df["ret_12"].fillna(0)                # contrarian relationship
    r = Rule("trend_12_sf", lambda d: d["ret_12"], "ret_12", learn_sign=True).fit(df, df, 1)
    assert r.sign == -1.0
    assert np.allclose(r.predict(df), -df["ret_12"], equal_nan=True)


def test_phase5_scripts_end_to_end(tmp_path):
    store = Store(tmp_path)
    for i, sym in enumerate(("BTC", "ETH")):
        bars = hourly_bars(24 * 150, seed=i)                      # 2024-07-01 .. 2024-11-28
        store.write_frame(bars, store.bars_path(sym, "1h"))
        store.write_frame(build_dataset(bars, "1h", horizons=(1, 3), cost=0.0012), store.dataset_path(sym, "1h"))
    py = [sys.executable]
    r1 = subprocess.run(py + [str(REPO / "scripts" / "kronos_features.py"), "--root", str(tmp_path), "--symbols", "BTC", "ETH",
                              "--timeframes", "1h", "--tiny", "--paths", "--n-samples", "4", "--pred-len", "3", "--lookback", "48"],
                        capture_output=True, text=True, cwd=REPO)
    assert r1.returncode == 0, r1.stderr
    assert store.kronos_path("tiny_paths", "1h", "BTC").exists()
    r2 = subprocess.run(py + [str(REPO / "scripts" / "run_kronos_comparison.py"), "--root", str(tmp_path), "--kronos-tag", "tiny_paths",
                              "--symbols", "BTC", "ETH", "--timeframes", "1h", "--horizons", "1", "3",
                              "--models", "B_lgbm_return", "BK_lgbm_return", "K_ridge", "kronos_direct"],
                        capture_output=True, text=True, cwd=REPO)
    assert r2.returncode == 0, r2.stderr
    run = next((tmp_path / "reports" / "kronos_comparison").iterdir())
    inc = pd.read_csv(run / "incremental.csv")
    assert set(inc["arm"]) == {"BK"} and set(inc["h"]) == {1, 3}
    summary = pd.read_csv(run / "summary.csv")
    assert {"B_lgbm_return", "BK_lgbm_return", "K_ridge", "kronos_direct"} == set(summary["model"])
    r3 = subprocess.run(py + [str(REPO / "scripts" / "kronos_calibration.py"), "--root", str(tmp_path), "--kronos-tag", "tiny_paths",
                              "--symbols", "BTC", "ETH", "--timeframes", "1h"], capture_output=True, text=True, cwd=REPO)
    assert r3.returncode == 0, r3.stderr
    cal = pd.read_csv(tmp_path / "reports" / "kronos_calibration_tiny_paths.csv")
    assert set(cal["h"]) == {1, 3} and "ALL" in set(cal["symbol"])
