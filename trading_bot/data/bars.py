"""Turn sparse trade-minute bars into a complete bar grid and higher timeframes.

Conventions (used everywhere downstream):

* Index = bar **open** time, UTC. A bar labelled t covers trades in [t, t + tf).
* A bar is known only at its close: ``available_at = t + tf``. Features/decisions keyed by a bar
  must use ``available_at`` (see :func:`available_at`).
* Minutes with no trades inside a segment are filled flat at the previous close with zero
  volume and ``is_filled=True`` (the price did not trade, it did not "stay" there for sure).
* A **segment** is a stretch without a trade gap longer than ``max_gap``. Gaps longer than that
  (missing files, outages) are not filled; every window used for modelling must lie inside one
  segment (``segment_id``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BAR_COLUMNS = ["open", "high", "low", "close", "volume", "amount", "n_trades", "buy_volume", "sell_volume", "vwap"]
TIMEFRAMES = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}


def tf_to_timedelta(tf: str) -> pd.Timedelta:
    return pd.Timedelta(TIMEFRAMES.get(tf, tf))


def available_at(index: pd.DatetimeIndex, tf: str) -> pd.DatetimeIndex:
    """Earliest time a bar's values are knowable (its close)."""
    return index + tf_to_timedelta(tf)


def minute_bars_to_frame(minute_bars: pd.DataFrame) -> pd.DataFrame:
    """int-us-indexed bars from ``trades.merge_partial_bars`` -> UTC DatetimeIndex, sorted."""
    out = minute_bars.drop(columns=[c for c in ("first_ts", "last_ts") if c in minute_bars.columns]).copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index.to_numpy(dtype=np.int64), unit="us", utc=True), name="timestamp")
    return out.sort_index()


def complete_minute_bars(trade_minutes: pd.DataFrame, max_gap: str | pd.Timedelta = "6h",
                         breaks: list | None = None) -> pd.DataFrame:
    """Fill empty minutes inside segments; split segments at trade gaps longer than ``max_gap``.

    ``breaks``: extra times at which a new segment must start even without a gap (e.g. a switch of
    data source), so no feature or label window mixes the two sides.
    """
    if trade_minutes.empty:
        raise ValueError("no bars")
    if not trade_minutes.index.is_unique:
        raise ValueError("duplicate minute bars; merge partial bars before completing")
    df = trade_minutes.sort_index()
    idx = df.index.to_series()
    new_seg = idx.diff() > pd.Timedelta(max_gap)
    for b in breaks or []:
        b = pd.Timestamp(b)
        b = b.tz_localize("UTC") if b.tzinfo is None else b
        new_seg |= (idx >= b) & (idx.shift(1) < b)
    seg = new_seg.cumsum().astype(np.int64)

    pieces = []
    for seg_id, part in df.groupby(seg.to_numpy(), sort=True):
        grid = pd.date_range(part.index[0], part.index[-1], freq="1min", name="timestamp")
        full = part.reindex(grid)
        filled = full["close"].isna().to_numpy()
        close = full["close"].ffill()
        for c in ("open", "high", "low"):
            full[c] = full[c].fillna(close)
        full["close"] = close
        for c in ("volume", "amount", "buy_volume", "sell_volume"):
            full[c] = full[c].fillna(0.0)
        full["n_trades"] = full["n_trades"].fillna(0).astype(np.int64)
        full["is_filled"] = filled
        full["segment_id"] = np.int64(seg_id)
        pieces.append(full)
    out = pd.concat(pieces)
    out["vwap"] = _vwap(out)
    return out[BAR_COLUMNS + ["is_filled", "segment_id"]]


def _vwap(df: pd.DataFrame) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        v = df["amount"] / df["volume"]
    return v.where(df["volume"] > 0, df["close"])


def resample_bars(minute_bars: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Aggregate completed 1-minute bars to ``tf`` (aligned to UTC epoch), per segment.

    Adds ``n_minutes`` and ``is_partial`` (fewer minutes than the timeframe, i.e. a bar at a
    segment edge) and ``is_filled`` (no trades at all in the bar).
    """
    step = tf_to_timedelta(tf)
    if step < pd.Timedelta("1min") or step % pd.Timedelta("1min"):
        raise ValueError(f"timeframe {tf} must be a whole number of minutes")
    if tf == "1m":
        out = minute_bars.copy()
        out["n_minutes"] = 1
        out["is_partial"] = False
        return out

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "amount": "sum",
        "n_trades": "sum",
        "buy_volume": "sum",
        "sell_volume": "sum",
    }
    pieces = []
    for seg_id, part in minute_bars.groupby("segment_id", sort=True):
        r = part.resample(step, label="left", closed="left", origin="epoch")
        bars = r.agg(agg)
        bars["n_minutes"] = r["close"].count()
        bars = bars[bars["n_minutes"] > 0]
        bars["segment_id"] = np.int64(seg_id)
        pieces.append(bars)
    out = pd.concat(pieces)
    if not out.index.is_unique:
        raise ValueError(f"segments share a {tf} bar; use max_gap >= {tf}")
    out["n_trades"] = out["n_trades"].astype(np.int64)
    out["n_minutes"] = out["n_minutes"].astype(np.int64)
    out["is_partial"] = out["n_minutes"] < int(step / pd.Timedelta("1min"))
    out["is_filled"] = out["n_trades"] == 0
    out["vwap"] = _vwap(out)
    out.index.name = "timestamp"
    return out[BAR_COLUMNS + ["is_filled", "segment_id", "n_minutes", "is_partial"]]
