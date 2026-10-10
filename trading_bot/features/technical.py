"""Price, candle, mean-reversion, volume and calendar features (all scale-free)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_bot.features.registry import log_ret, register, rolling_z, safe_div

# ---------------------------------------------------------------- price / momentum

for _k in (1, 3, 6, 12, 24, 48):
    register(f"ret_{_k}", "price", _k + 1, f"log return over the last {_k} bars",
             lambda b, k=_k: np.log(b["close"]).diff(k))

for _k in (3, 12):
    register(f"accel_{_k}", "price", 2 * _k + 1, f"ret_{_k} now minus ret_{_k} one window earlier",
             lambda b, k=_k: np.log(b["close"]).diff(k) - np.log(b["close"]).diff(k).shift(k))

register("jump_48", "price", 50, "|1-bar return| / std of the previous 48 one-bar returns (jump size in sigmas)",
         lambda b: log_ret(b).abs() / log_ret(b).rolling(48).std().shift(1))

# ---------------------------------------------------------------- candle shape

register("range_rel", "candle", 1, "(high - low) / close",
         lambda b: (b["high"] - b["low"]) / b["close"])
register("body_ratio", "candle", 1, "(close - open) / (high - low); 0 when the bar has no range",
         lambda b: safe_div(b["close"] - b["open"], b["high"] - b["low"]).fillna(0.0))
register("upper_wick", "candle", 1, "(high - max(open, close)) / (high - low); 0 when no range",
         lambda b: safe_div(b["high"] - b[["open", "close"]].max(axis=1), b["high"] - b["low"]).fillna(0.0))
register("lower_wick", "candle", 1, "(min(open, close) - low) / (high - low); 0 when no range",
         lambda b: safe_div(b[["open", "close"]].min(axis=1) - b["low"], b["high"] - b["low"]).fillna(0.0))

# ---------------------------------------------------------------- mean reversion

for _w in (24, 96):
    register(f"zclose_{_w}", "mean_reversion", _w, f"z-score of close vs its last {_w} bars",
             lambda b, w=_w: rolling_z(b["close"], w))

register("vwap_dist_24", "mean_reversion", 24, "log(close / VWAP of the last 24 bars)",
         lambda b: np.log(b["close"] / safe_div(b["amount"].rolling(24).sum(), b["volume"].rolling(24).sum())))

# ---------------------------------------------------------------- volume

for _w in (48, 192):
    register(f"logvol_z_{_w}", "volume", _w, f"z-score of log(1 + volume) vs the last {_w} bars",
             lambda b, w=_w: rolling_z(np.log1p(b["volume"]), w))

register("vol_mom_12", "volume", 24, "log((volume of last 12 bars + 1) / (volume of the 12 bars before + 1))",
         lambda b: np.log((b["volume"].rolling(12).sum() + 1) / (b["volume"].rolling(12).sum().shift(12) + 1)))
register("ntrades_z_48", "volume", 48, "z-score of log(1 + trade count) vs the last 48 bars",
         lambda b: rolling_z(np.log1p(b["n_trades"]), 48))
register("trade_size_z_48", "volume", 48, "z-score of log(average trade size); bars without trades skipped (>= 24 needed)",
         lambda b: rolling_z(np.log(safe_div(b["volume"], b["n_trades"])), 48, min_periods=24))
register("pv_corr_48", "volume", 49, "rolling correlation of |1-bar return| and log(1 + volume) over 48 bars",
         lambda b: log_ret(b).abs().rolling(48).corr(np.log1p(b["volume"])))

# ---------------------------------------------------------------- calendar (UTC)


def _hour(b: pd.DataFrame) -> pd.Series:
    idx = b.index
    return pd.Series(idx.hour + idx.minute / 60.0, index=idx)


def _dow(b: pd.DataFrame) -> pd.Series:
    idx = b.index
    return pd.Series(idx.dayofweek + (idx.hour + idx.minute / 60.0) / 24.0, index=idx)


register("hour_sin", "calendar", 1, "sin of UTC time of day of the bar", lambda b: np.sin(2 * np.pi * _hour(b) / 24))
register("hour_cos", "calendar", 1, "cos of UTC time of day of the bar", lambda b: np.cos(2 * np.pi * _hour(b) / 24))
register("dow_sin", "calendar", 1, "sin of UTC day of week", lambda b: np.sin(2 * np.pi * _dow(b) / 7))
register("dow_cos", "calendar", 1, "cos of UTC day of week", lambda b: np.cos(2 * np.pi * _dow(b) / 7))
