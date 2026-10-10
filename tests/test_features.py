from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_bot.data.bars import complete_minute_bars, resample_bars
from trading_bot.features import registry
from trading_bot.features.registry import all_features, compute_features, registry_markdown

REPO = Path(__file__).resolve().parents[1]


def synth_minute_bars(n_minutes=4000, seed=0, empty_frac=0.15, gap_at=2500, gap="8h", start="2026-02-01") -> pd.DataFrame:
    """Completed 1m bars with empty minutes and one long gap (two segments)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_minutes, freq="1min", tz="UTC", name="timestamp")
    idx = idx.where(np.arange(n_minutes) < gap_at, idx + pd.Timedelta(gap))
    keep = rng.random(n_minutes) > empty_frac
    keep[0] = True
    if gap_at < n_minutes:
        keep[gap_at] = True
    close = 50_000 * np.exp(np.cumsum(rng.standard_t(4, n_minutes) * 4e-4))
    open_ = np.concatenate([[close[0]], close[:-1]]) * np.exp(rng.normal(0, 1e-4, n_minutes))
    hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 3e-4, n_minutes)))
    lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 3e-4, n_minutes)))
    vol = rng.lognormal(4, 1, n_minutes)
    buy = vol * rng.uniform(0.2, 0.8, n_minutes)
    m = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol, "amount": vol * close,
         "n_trades": rng.integers(1, 50, n_minutes), "buy_volume": buy, "sell_volume": vol - buy},
        index=idx,
    )[keep]
    return complete_minute_bars(m, max_gap="6h")


@pytest.fixture(scope="module")
def bars5():
    return resample_bars(synth_minute_bars(), "5m")


def _assert_same(a: pd.DataFrame, b: pd.DataFrame, rows):
    for col in a.columns:
        np.testing.assert_array_equal(a[col].iloc[rows].to_numpy(), b[col].iloc[rows].to_numpy(), err_msg=col)


@pytest.mark.parametrize("cut", [60, 250, 499, 500, 517, 700])
def test_every_feature_uses_only_past_bars_truncation(bars5, cut):
    """Recompute every registered feature on history truncated at `cut`; overlapping rows must be identical."""
    full = compute_features(bars5)
    short = compute_features(bars5.iloc[:cut])
    _assert_same(full, short, slice(0, cut))


def test_every_feature_ignores_rewritten_future(bars5):
    t = 400
    future = bars5.copy()
    cols = ["open", "high", "low", "close", "vwap"]
    future.iloc[t + 1:, [future.columns.get_loc(c) for c in cols]] *= 1.37
    future.iloc[t + 1:, future.columns.get_loc("volume")] *= 9.0
    future.iloc[t + 1:, future.columns.get_loc("buy_volume")] = 0.0
    _assert_same(compute_features(bars5), compute_features(future), slice(0, t + 1))


def test_features_do_not_cross_segment_gaps(bars5):
    f = compute_features(bars5)
    first_of_seg2 = np.flatnonzero(bars5["segment_id"].to_numpy() == 1)[0]
    assert np.isnan(f["ret_1"].iloc[first_of_seg2])
    assert np.isnan(f["rv_48"].iloc[first_of_seg2 + 10])
    assert np.isfinite(f["ret_1"].iloc[first_of_seg2 + 1])


def test_every_feature_produces_values_after_warmup(bars5):
    f = compute_features(bars5)
    for spec in all_features():
        assert f[spec.name].notna().sum() > 0, spec.name
        assert f[spec.name].iloc[: spec.lookback - 1].isna().all() or spec.lookback <= 1 or spec.name in (
            "trade_size_z_48", "vol_of_vol_48"), f"{spec.name} has values inside its declared warm-up"


def test_unavailable_features_are_refused(bars5):
    registry.register("_peek", "test", 1, "uses the next bar", lambda b: b["close"].shift(-1), available_at_time_t=False)
    try:
        with pytest.raises(ValueError, match="not available"):
            compute_features(bars5, names=["ret_1", "_peek"])
    finally:
        registry.REGISTRY.pop("_peek")


def test_feature_docs_are_up_to_date():
    assert (REPO / "docs" / "features.md").read_text() == registry_markdown(), "run scripts/gen_feature_docs.py"
