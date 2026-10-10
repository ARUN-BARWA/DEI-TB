"""Forward-looking targets for decision time t + tf (the close of bar t). Never used as features.

For horizon h (bars) the outcome window is bars t+1 .. t+h, i.e. the interval
[t + tf, t + (h + 1) tf). Columns per horizon:

    y_ret_{h}      log(close_{t+h} / close_t)
    y_dir_{h}      +1 / -1 if y_ret exceeds +cost / -cost, else 0 (cost-aware dead zone)
    y_mfe_{h}      log(max high over the window / close_t)    (max favourable excursion for a long)
    y_mae_{h}      log(min low over the window / close_t)     (max adverse excursion for a long)
    y_rv_{h}       sqrt(sum of squared 1-bar log returns over the window)
    y_tb_{h}       triple barrier: +1 upper first, -1 lower first, 0 neither (time barrier) or ambiguous
    y_tb_amb_{h}   both barriers inside the same finest bar (order unknown) -> labelled 0
    y_tb_bars_{h}  time to the first touch in bars of tf (h if the time barrier ended it)
    label_end_{h}  t + (h + 1) tf: when the label is fully known (used for purging)

Barrier half-width = max(k * sigma_t * sqrt(h), min_width), where sigma_t is the std of the last
``vol_window`` one-bar returns up to and including bar t (known at decision time), and
min_width defaults to the round-trip cost: a barrier closer than the cost cannot be a profitable
trade. With a 1-minute ``path`` the barriers are resolved minute by minute inside higher-timeframe
bars; otherwise on the bars themselves.

Labels are NaN when the outcome window leaves the bar's data segment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from trading_bot.data.bars import tf_to_timedelta


def _first_hits(hi: np.ndarray, lo: np.ndarray, up: np.ndarray, dn: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """hi/lo: (N, m) windows. Returns label, ambiguous flag, steps to first touch (1-based, m if none)."""
    m = hi.shape[1]
    up_hit = hi >= up[:, None]
    dn_hit = lo <= dn[:, None]
    first_up = np.where(up_hit.any(axis=1), up_hit.argmax(axis=1), m)
    first_dn = np.where(dn_hit.any(axis=1), dn_hit.argmax(axis=1), m)
    label = np.where(first_up < first_dn, 1, np.where(first_dn < first_up, -1, 0)).astype(float)
    amb = (first_up == first_dn) & (first_up < m)
    steps = np.minimum(np.minimum(first_up, first_dn) + 1, m).astype(float)
    return label, amb, steps


def _segment_labels(seg: pd.DataFrame, horizons, cost: float, tb_k: float, min_width: float, vol_window: int) -> dict[str, np.ndarray]:
    close = seg["close"].to_numpy(dtype=float)
    high = seg["high"].to_numpy(dtype=float)
    low = seg["low"].to_numpy(dtype=float)
    n = len(seg)
    lc = np.log(close)
    r1 = np.diff(lc, prepend=np.nan)
    sigma = pd.Series(r1).rolling(vol_window).std().to_numpy()
    out: dict[str, np.ndarray] = {}
    for h in horizons:
        ret = np.full(n, np.nan)
        mfe = np.full(n, np.nan)
        mae = np.full(n, np.nan)
        rv = np.full(n, np.nan)
        tb = np.full(n, np.nan)
        amb = np.zeros(n, dtype=bool)
        tb_bars = np.full(n, np.nan)
        m = n - h  # rows with a complete outcome window inside this segment
        if m > 0:
            ret[:m] = lc[h:] - lc[:m]
            hw = sliding_window_view(high[1:], h)[:m]
            lw = sliding_window_view(low[1:], h)[:m]
            mfe[:m] = np.log(hw.max(axis=1) / close[:m])
            mae[:m] = np.log(lw.min(axis=1) / close[:m])
            rv[:m] = np.sqrt((sliding_window_view(r1[1:], h)[:m] ** 2).sum(axis=1))
            width = np.maximum(tb_k * sigma[:m] * np.sqrt(h), min_width)
            ok = np.isfinite(width)
            lab, a, steps = _first_hits(hw, lw, close[:m] * np.exp(width), close[:m] * np.exp(-width))
            tb[:m] = np.where(ok, lab, np.nan)
            amb[:m] = a & ok
            tb_bars[:m] = np.where(ok, steps, np.nan)
        direction = np.where(ret > cost, 1.0, np.where(ret < -cost, -1.0, 0.0))
        out[f"y_ret_{h}"] = ret
        out[f"y_dir_{h}"] = np.where(np.isnan(ret), np.nan, direction)
        out[f"y_mfe_{h}"] = mfe
        out[f"y_mae_{h}"] = mae
        out[f"y_rv_{h}"] = rv
        out[f"y_tb_{h}"] = tb
        out[f"y_tb_amb_{h}"] = amb
        out[f"y_tb_bars_{h}"] = tb_bars
        w_full = np.full(n, np.nan)
        if m > 0:
            w_full[:m] = np.maximum(tb_k * sigma[:m] * np.sqrt(h), min_width)
        out[f"_width_{h}"] = w_full  # used by path resolution, dropped from the output
    return out


def _resolve_on_path(labels: pd.DataFrame, bars: pd.DataFrame, path: pd.DataFrame, tf: str, horizons, max_cells: int = 5_000_000) -> None:
    """Re-resolve triple barriers on finer (1-minute) bars, in place."""
    step = tf_to_timedelta(tf)
    path_step = path.index.to_series().diff().min()
    per_bar = int(step / path_step)
    p_time = path.index.as_unit("ns").asi8
    p_hi = path["high"].to_numpy(dtype=float)
    p_lo = path["low"].to_numpy(dtype=float)
    p_seg = path["segment_id"].to_numpy()
    start_time = (bars.index + step).as_unit("ns").asi8  # decision time = first path bar of the window
    starts = np.searchsorted(p_time, start_time)
    close = bars["close"].to_numpy(dtype=float)
    seg = bars["segment_id"].to_numpy()
    for h in horizons:
        m = h * per_bar
        width = labels[f"_width_{h}"].to_numpy()
        valid = np.isfinite(width) & (starts + m <= len(p_time))
        idx = np.flatnonzero(valid)
        s = starts[idx]
        contiguous = (p_time[np.minimum(s + m - 1, len(p_time) - 1)] - p_time[s]) == (m - 1) * int(path_step / pd.Timedelta(1, "ns"))
        aligned = p_time[s] == start_time[idx]
        same_seg = (p_seg[s] == seg[idx]) & (p_seg[np.minimum(s + m - 1, len(p_seg) - 1)] == seg[idx])
        keep = contiguous & aligned & same_seg
        idx, s = idx[keep], s[keep]
        tb = np.full(len(bars), np.nan)
        amb = np.zeros(len(bars), dtype=bool)
        tb_bars = np.full(len(bars), np.nan)
        hw_all = sliding_window_view(p_hi, m)
        lw_all = sliding_window_view(p_lo, m)
        chunk = max(64, max_cells // m)  # bounds memory: chunk x m window cells per array
        for c in range(0, len(idx), chunk):
            ii, ss = idx[c:c + chunk], s[c:c + chunk]
            lab, a, steps = _first_hits(hw_all[ss], lw_all[ss], close[ii] * np.exp(width[ii]), close[ii] * np.exp(-width[ii]))
            tb[ii], amb[ii], tb_bars[ii] = lab, a, steps / per_bar
        has_label = labels[f"y_ret_{h}"].notna().to_numpy()
        labels[f"y_tb_{h}"] = np.where(has_label, tb, np.nan)
        labels[f"y_tb_amb_{h}"] = amb & has_label
        labels[f"y_tb_bars_{h}"] = np.where(has_label, tb_bars, np.nan)


def forward_labels(
    bars: pd.DataFrame,
    tf: str,
    horizons=(1, 3, 6, 12, 24),
    cost: float = 0.0,
    path: pd.DataFrame | None = None,
    tb_k: float = 1.0,
    min_width: float | None = None,
    vol_window: int = 48,
) -> pd.DataFrame:
    """Targets for every bar (see module docstring). ``cost`` is the round-trip cost as a log return."""
    min_width = cost if min_width is None else min_width
    segments = bars.groupby("segment_id", sort=True) if "segment_id" in bars.columns else [(0, bars)]
    parts = [pd.DataFrame(_segment_labels(s, horizons, cost, tb_k, min_width, vol_window), index=s.index) for _, s in segments]
    labels = pd.concat(parts).reindex(bars.index)
    step = tf_to_timedelta(tf)
    if path is not None and step > path.index.to_series().diff().min():
        b = bars if "segment_id" in bars.columns else bars.assign(segment_id=0)
        p = path if "segment_id" in path.columns else path.assign(segment_id=0)
        _resolve_on_path(labels, b, p, tf, horizons)
    for h in horizons:
        labels[f"label_end_{h}"] = bars.index + step * (h + 1)
        labels[f"y_tb_amb_{h}"] = labels[f"y_tb_amb_{h}"].astype(bool)
    return labels.drop(columns=[c for c in labels.columns if c.startswith("_width_")])
