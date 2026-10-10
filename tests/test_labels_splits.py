import numpy as np
import pandas as pd
import pytest

from trading_bot.labels.targets import forward_labels
from trading_bot.research.splits import Fold, HoldoutLocked, development_mask, fold_masks, monthly_folds


def bars_from(closes, highs=None, lows=None, freq="5min", segment=None):
    n = len(closes)
    idx = pd.date_range("2026-03-01", periods=n, freq=freq, tz="UTC", name="timestamp")
    closes = np.asarray(closes, float)
    return pd.DataFrame(
        {"open": closes, "high": closes if highs is None else highs, "low": closes if lows is None else lows,
         "close": closes, "segment_id": np.zeros(n, int) if segment is None else segment},
        index=idx,
    )


def test_forward_returns_excursions_and_dead_zone():
    c = [100, 101, 99, 102, 100]
    b = bars_from(c, highs=[100, 103, 99.5, 102, 100], lows=[100, 100.5, 98, 101, 99])
    y = forward_labels(b, "5m", horizons=[1, 2], cost=0.015, vol_window=2, tb_k=0.0, min_width=0.5)
    assert y["y_ret_2"].iloc[0] == pytest.approx(np.log(99 / 100))
    assert y["y_mfe_2"].iloc[0] == pytest.approx(np.log(103 / 100))   # bars 1..2 only, never bar 0
    assert y["y_mae_2"].iloc[0] == pytest.approx(np.log(98 / 100))
    r = np.diff(np.log(c))
    assert y["y_rv_2"].iloc[0] == pytest.approx(np.sqrt(r[0] ** 2 + r[1] ** 2))
    assert y["y_dir_1"].iloc[2] == 1      # +3.0% > 1.5% cost
    assert y["y_dir_1"].iloc[0] == 0      # +1.0% inside the dead zone
    assert y["y_dir_2"].iloc[1] == 0      # log(102/101) ~ 1% < cost
    assert y["y_ret_2"].iloc[3:].isna().all()
    assert y["label_end_2"].iloc[0] == b.index[0] + pd.Timedelta("15min")


def test_labels_never_cross_segment_end():
    b = bars_from([100, 101, 102, 103, 104, 105], segment=[0, 0, 0, 1, 1, 1])
    y = forward_labels(b, "5m", horizons=[2])
    assert np.isfinite(y["y_ret_2"].iloc[0]) and y["y_ret_2"].iloc[1:3].isna().all()
    assert np.isfinite(y["y_ret_2"].iloc[3])


def test_triple_barrier_on_bars():
    n = 60
    base = np.full(n, 100.0)
    rng = np.random.default_rng(0)
    base[:50] = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, 50)))
    base[50:] = base[49]             # flat after the decision bar unless a barrier is set below
    c = base.copy()
    hi, lo = c.copy(), c.copy()
    t = 49
    # sigma over the last 48 returns ~ 0.1%; width with k=1, h=3: ~0.17% (min_width 0)
    hi[t + 2] = c[t] * 1.01          # upper touched at step 2
    lo[t + 3] = c[t] * 0.99          # lower only later
    y = forward_labels(bars_from(c, hi, lo), "5m", horizons=[3], tb_k=1.0, min_width=0.0)
    assert y["y_tb_3"].iloc[t] == 1 and y["y_tb_bars_3"].iloc[t] == 2
    lo[t + 2] = c[t] * 0.99          # both in the same bar -> ambiguous, labelled 0
    y = forward_labels(bars_from(c, hi, lo), "5m", horizons=[3], tb_k=1.0, min_width=0.0)
    assert y["y_tb_3"].iloc[t] == 0 and bool(y["y_tb_amb_3"].iloc[t])
    assert y["y_tb_3"].iloc[:47].isna().all()   # no sigma yet -> no barrier


def test_min_width_defaults_to_cost():
    c = 100 * np.exp(np.cumsum(np.r_[0, np.full(59, 0.0001)]))  # tiny moves, far below cost
    y = forward_labels(bars_from(c, c * 1.0001, c * 0.9999), "5m", horizons=[3], cost=0.01, tb_k=1.0)
    assert (y["y_tb_3"].dropna() == 0).all()


def test_triple_barrier_resolved_inside_bars_with_minute_path():
    # 5m bar where the minute path shows the low first, then the high: the 5m bar alone is ambiguous.
    minutes = pd.date_range("2026-03-01", periods=60 * 5, freq="1min", tz="UTC", name="timestamp")
    rng = np.random.default_rng(1)
    mc = 100 * np.exp(np.cumsum(rng.normal(0, 2e-4, len(minutes))))
    path = pd.DataFrame({"open": mc, "high": mc, "low": mc, "close": mc, "segment_id": 0}, index=minutes)
    bars = path.resample("5min").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    bars["segment_id"] = 0
    t = 50                                    # decision at close of 5m bar t -> next bar starts at minute 5(t+1)
    c = bars["close"].iloc[t]
    first = 5 * (t + 1)
    path.iloc[first + 1, path.columns.get_loc("low")] = c * 0.98    # minute 2 of the next bar: down first
    path.iloc[first + 3, path.columns.get_loc("high")] = c * 1.02   # minute 4: then up
    bars = path.resample("5min").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    bars["segment_id"] = 0
    on_bars = forward_labels(bars, "5m", horizons=[1], tb_k=1.0, min_width=0.0)
    on_path = forward_labels(bars, "5m", horizons=[1], tb_k=1.0, min_width=0.0, path=path)
    assert bool(on_bars["y_tb_amb_1"].iloc[t]) and on_bars["y_tb_1"].iloc[t] == 0
    assert on_path["y_tb_1"].iloc[t] == -1 and not on_path["y_tb_amb_1"].iloc[t]
    assert on_path["y_tb_bars_1"].iloc[t] == pytest.approx(2 / 5)
    np.testing.assert_array_equal(on_bars["y_ret_1"].to_numpy(), on_path["y_ret_1"].to_numpy())


def test_monthly_folds_layout():
    folds = monthly_folds("2026-01-01", "2026-03", "2026-07-01", "2026-09-02")
    assert [f.name for f in folds] == ["dev_2026-03", "dev_2026-04", "dev_2026-05", "dev_2026-06", "holdout"]
    f = folds[0]
    assert (f.train_start, f.val_start, f.test_start, f.test_end) == tuple(
        pd.Timestamp(x, tz="UTC") for x in ("2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"))
    h = folds[-1]
    assert h.is_holdout and h.val_start == pd.Timestamp("2026-06-01", tz="UTC") and h.test_end == pd.Timestamp("2026-09-02", tz="UTC")
    with pytest.raises(ValueError):
        monthly_folds("2026-01-01", "2026-02", "2026-07-01", "2026-09-02")  # no train before Jan validation


def test_fold_masks_purge_embargo_and_order():
    idx = pd.date_range("2026-01-01", "2026-04-01", freq="1h", tz="UTC", inclusive="left")
    decision = pd.Series(idx + pd.Timedelta("1h"))
    label_end = decision + pd.Timedelta("1D")
    fold = monthly_folds("2026-01-01", "2026-03", "2026-04-01", "2026-04-02")[0]
    m = fold_masks(decision, label_end, fold, embargo="12h")
    assert not (m["train"] & m["val"]).any() and not (m["val"] & m["test"]).any()
    assert label_end[m["train"]].max() <= fold.val_start - pd.Timedelta("12h")
    assert label_end[m["val"]].max() <= fold.test_start - pd.Timedelta("12h")
    assert decision[m["train"]].max() < decision[m["val"]].min() < decision[m["test"]].min()
    assert label_end[m["test"]].max() <= fold.test_end
    nat = label_end.copy()
    nat.iloc[:] = pd.NaT
    assert not any(v.any() for v in fold_masks(decision, nat, fold).values())


def test_holdout_is_locked_and_audited(tmp_path):
    h = monthly_folds("2026-01-01", "2026-03", "2026-07-01", "2026-09-02")[-1]
    t = pd.Series(pd.date_range("2026-01-01", periods=10, freq="30D", tz="UTC"))
    with pytest.raises(HoldoutLocked):
        fold_masks(t, t, h)
    log = tmp_path / "holdout_access.log"
    fold_masks(t, t, h, allow_holdout=True, audit_log=log, reason="final evaluation")
    assert "holdout\tfinal evaluation" in log.read_text()


def test_development_mask():
    le = pd.Series(pd.to_datetime(["2026-06-30 23:00", "2026-07-01 00:00", "2026-07-01 00:05", None], utc=True))
    assert development_mask(le, "2026-07-01").tolist() == [True, True, False, False]


def test_fold_requires_ordered_periods():
    t = pd.Timestamp("2026-03-01", tz="UTC")
    with pytest.raises(ValueError):
        Fold("bad", t, t, t, t)
