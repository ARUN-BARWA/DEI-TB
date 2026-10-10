"""Kronos features for every decision bar: causal, per data segment, cached in resumable chunks.

For the bar t of a segment (decision at its close) the window is the last ``lookback`` bars of that
segment ending at t: [open, high, low, close, volume, amount] plus Kronos time stamps, normalised
exactly as upstream (per window). Rows with fewer than ``lookback`` bars of history in their segment
get NaN. Columns (all prefixed ``kr_``):

    kr_emb_<i>        last-position hidden state (d_model values)
    kr_entropy        entropy of the next coarse-token distribution / log(vocab)
    kr_recon          tokenizer reconstruction error over the last 8 bars
    kr_r<h>_<stat>    with paths: statistics of sampled forecast paths (see ``path_stats_batch``)
    kr_close          close at the decision bar (cache consistency check)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from trading_bot.data.bars import tf_to_timedelta
from trading_bot.models.kronos_model import FEATURE_COLS, KronosWrapper, normalize_windows, time_stamps

PATH_STATS = ("mean", "std", "q05", "q25", "q50", "q75", "q95", "p_up", "mfe", "mae", "path_vol")


def path_stats_batch(paths: np.ndarray, last_close: np.ndarray, horizons: Sequence[int]) -> dict[str, np.ndarray]:
    """Vectorised ``kronos_model.summarize_paths`` over windows: paths (B, N, P, 6), last_close (B,)."""
    prices = np.maximum(paths[..., :4], 1e-12)
    hi, lo, close = prices.max(axis=-1), prices.min(axis=-1), prices[..., 3]  # (B, N, P)
    last = np.asarray(last_close, dtype=float)[:, None]
    out: dict[str, np.ndarray] = {}
    for h in horizons:
        if not 1 <= h <= close.shape[2]:
            raise ValueError(f"horizon {h} outside 1..{close.shape[2]}")
        r = np.log(close[:, :, h - 1] / last)
        q = np.quantile(r, [0.05, 0.25, 0.5, 0.75, 0.95], axis=1)
        step = np.diff(np.log(np.concatenate([np.repeat(last[:, :, None], close.shape[1], axis=1), close[:, :, :h]], axis=2)), axis=2)
        out.update({
            f"r{h}_mean": r.mean(axis=1), f"r{h}_std": r.std(axis=1),
            f"r{h}_q05": q[0], f"r{h}_q25": q[1], f"r{h}_q50": q[2], f"r{h}_q75": q[3], f"r{h}_q95": q[4],
            f"r{h}_p_up": (r > 0).mean(axis=1),
            f"r{h}_mfe": np.log(hi[:, :, :h].max(axis=2) / last).mean(axis=1),
            f"r{h}_mae": np.log(lo[:, :, :h].min(axis=2) / last).mean(axis=1),
            f"r{h}_path_vol": step.std(axis=2).mean(axis=1),
        })
    return out


def _chunk_features(wrapper: KronosWrapper, raw: np.ndarray, stamps: np.ndarray, index: pd.DatetimeIndex, close: np.ndarray,
                    ends: np.ndarray, lookback: int, step: pd.Timedelta, with_paths: bool, pred_len: int, n_samples: int,
                    horizons, temperature: float, top_p: float, seed: int) -> pd.DataFrame:
    win = sliding_window_view(raw, (lookback, raw.shape[1]))[:, 0]          # (n - L + 1, L, 6)
    swin = sliding_window_view(stamps, (lookback, stamps.shape[1]))[:, 0]   # (n - L + 1, L, 5)
    starts = ends - lookback + 1
    x, mean, std = normalize_windows(win[starts], wrapper.clip)
    xs = np.ascontiguousarray(swin[starts])
    emb = wrapper.embed_prepared(x, xs)
    cols: dict[str, np.ndarray] = {f"kr_emb_{i}": emb.hidden_last[:, i].astype(np.float32) for i in range(emb.hidden_last.shape[1])}
    cols["kr_entropy"] = emb.s1_entropy
    cols["kr_recon"] = emb.recon_error
    cols["kr_close"] = close[ends]
    if with_paths:
        k = np.tile(np.arange(1, pred_len + 1), len(ends)) * int(step / pd.Timedelta(1, "ns"))
        future = index[ends].repeat(pred_len) + pd.to_timedelta(k, unit="ns")
        ys = time_stamps(pd.DatetimeIndex(future)).reshape(len(ends), pred_len, -1)
        paths = wrapper.sample_paths_prepared(x, xs, ys, mean, std, pred_len, n_samples, temperature, 0, top_p, seed)
        for k, v in path_stats_batch(paths, close[ends], horizons).items():
            cols[f"kr_{k}"] = v
    return pd.DataFrame(cols, index=index[ends])


def kronos_features(
    wrapper: KronosWrapper,
    bars: pd.DataFrame,
    tf: str,
    lookback: int,
    with_paths: bool = False,
    pred_len: int = 12,
    n_samples: int = 16,
    horizons: Sequence[int] = (1, 3, 6, 12),
    temperature: float = 1.0,
    top_p: float = 0.9,
    seed: int = 0,
    chunk: int = 1024,
    cache_dir: str | Path | None = None,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Kronos features aligned to ``bars.index`` (NaN where a full window is not available)."""
    if lookback > wrapper.max_context:
        raise ValueError(f"lookback {lookback} > max_context {wrapper.max_context}")
    horizons = [h for h in horizons if h <= pred_len]
    step = tf_to_timedelta(tf)
    cache = Path(cache_dir) if cache_dir is not None else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    segments = bars.groupby("segment_id", sort=True) if "segment_id" in bars.columns else [(0, bars)]
    jobs = []
    for sid, seg in segments:
        n = len(seg)
        for start in range(lookback - 1, n, chunk):
            jobs.append((sid, seg, np.arange(start, min(start + chunk, n))))
    total = sum(len(j[2]) for j in jobs)
    done, t0, parts = 0, time.time(), []
    for sid, seg, ends in jobs:
        path = cache / f"seg{sid}_{ends[0]:07d}.parquet" if cache is not None else None
        close = seg["close"].to_numpy(dtype=float)
        part = None
        if path is not None and path.exists():
            part = pd.read_parquet(path)
            if len(part) != len(ends) or not np.allclose(part["kr_close"].to_numpy(), close[ends]):
                part = None  # bars changed since this chunk was cached
        if part is None:
            raw = seg[FEATURE_COLS].to_numpy(dtype=np.float32)
            part = _chunk_features(wrapper, raw, time_stamps(seg.index), seg.index, close, ends, lookback, step,
                                   with_paths, pred_len, n_samples, horizons, temperature, top_p, seed + int(ends[0]) + 7919 * int(sid))
            if path is not None:
                tmp = path.with_suffix(".tmp")
                part.to_parquet(tmp)
                tmp.replace(path)
        parts.append(part)
        done += len(ends)
        rate = done / max(time.time() - t0, 1e-9)
        log(f"    {done:,}/{total:,} decisions ({rate:.1f}/s, ~{(total - done) / max(rate, 1e-9) / 60:.0f} min left)")
    if not parts:
        return pd.DataFrame(index=bars.index)
    return pd.concat(parts).reindex(bars.index)
