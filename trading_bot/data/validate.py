"""Data-quality checks on completed bars, and a clock-offset check against a reference venue."""

from __future__ import annotations

import numpy as np
import pandas as pd


def robust_return_outliers(close: pd.Series, k: float = 10.0, window: int = 1440) -> pd.Series:
    """Bars whose log return exceeds ``k`` x a trailing robust sigma (MAD of past returns only)."""
    r = np.log(close).diff()
    mad = r.abs().shift(1).rolling(window, min_periods=max(30, window // 10)).median()
    sigma = 1.4826 * mad
    return (r.abs() > k * sigma) & sigma.gt(0)


def minute_bar_quality(bars: pd.DataFrame, max_gap: str = "6h") -> dict:
    """Summary statistics of completed 1-minute bars for one symbol."""
    seg_sizes = bars.groupby("segment_id").size()
    seg_bounds = bars.reset_index().groupby("segment_id")["timestamp"].agg(["min", "max"])
    breaks = [
        {"from": seg_bounds["max"].iloc[i].isoformat(), "to": seg_bounds["min"].iloc[i + 1].isoformat(),
         "hours": round((seg_bounds["min"].iloc[i + 1] - seg_bounds["max"].iloc[i]) / pd.Timedelta("1h"), 2)}
        for i in range(len(seg_bounds) - 1)
    ]
    filled = bars["is_filled"].to_numpy()
    runs = _longest_true_run(filled)
    outliers = robust_return_outliers(bars["close"])
    daily = bars.resample("1D").agg({"n_trades": "sum", "volume": "sum", "amount": "sum"})
    daily = daily[daily["n_trades"] > 0]
    hour_share = bars.groupby(bars.index.hour)["volume"].sum()
    hour_share = (hour_share / hour_share.sum()).round(4)
    r = np.log(bars["close"]).diff()
    return {
        "first_bar": bars.index[0].isoformat(),
        "last_bar": bars.index[-1].isoformat(),
        "minutes": int(len(bars)),
        "pct_minutes_without_trades": round(100 * float(filled.mean()), 2),
        "longest_no_trade_run_min": int(runs),
        "segments": int(len(seg_sizes)),
        "segment_breaks_over_max_gap": breaks[:50],
        "max_gap": str(max_gap),
        "return_outliers_10sigma": int(outliers.sum()),
        "return_outlier_times": [t.isoformat() for t in bars.index[outliers.to_numpy()][:20]],
        "median_trades_per_day": float(daily["n_trades"].median()) if len(daily) else 0.0,
        "median_contracts_per_day": float(daily["volume"].median()) if len(daily) else 0.0,
        "median_amount_per_day": float(daily["amount"].median()) if len(daily) else 0.0,
        "ann_vol_from_1m": float(r.std() * np.sqrt(525_600)),
        "median_abs_1m_return_bp": float(r.abs().median() * 1e4),
        "taker_buy_share": float(bars["buy_volume"].sum() / max(bars["volume"].sum(), 1e-12)),
        "volume_share_by_utc_hour": {int(h): float(v) for h, v in hour_share.items()},
    }


def _longest_true_run(x: np.ndarray) -> int:
    best = cur = 0
    for v in x:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def estimate_clock_offset(ours: pd.Series, ref: pd.Series, max_shift_min: int = 720) -> tuple[int, pd.Series]:
    """Find the shift (minutes) that best aligns our 1-minute closes with a reference venue's.

    Returns ``(best_shift, corr_by_shift)``. ``best_shift = s`` means our timestamp t corresponds
    to reference time t + s minutes, i.e. our clock is ``-s`` minutes off UTC if the reference is
    UTC. Same-venue noise is irrelevant: cross-venue 1-minute returns of the same coin correlate
    strongly only when aligned. IST-labelled data would show up near -330 or +330.
    """
    a = np.log(ours.astype(float)).diff()
    b = np.log(ref.astype(float)).diff()
    a = a[~a.index.duplicated()]
    b = b[~b.index.duplicated()]
    grid = pd.date_range(min(a.index.min(), b.index.min()).floor("1min"), max(a.index.max(), b.index.max()), freq="1min")
    av = a.reindex(grid).to_numpy(dtype=float)
    bv = b.reindex(grid).to_numpy(dtype=float)
    n = len(grid)
    corrs = {}
    for s in range(-max_shift_min, max_shift_min + 1):
        x, y = (av[: n - s], bv[s:]) if s >= 0 else (av[-s:], bv[: n + s])  # our t vs reference t + s
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() >= 100:
            corrs[s] = float(np.corrcoef(x[ok], y[ok])[0, 1])
    corr = pd.Series(corrs, name="corr", dtype=float).sort_index()
    if corr.empty:
        raise ValueError("not enough overlapping minutes between the two series")
    return int(corr.idxmax()), corr
