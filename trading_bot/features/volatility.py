"""Volatility and return-distribution features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_bot.features.registry import log_ret, register, safe_div

for _w in (12, 48, 192):
    register(f"rv_{_w}", "volatility", _w + 1, f"std of 1-bar log returns over the last {_w} bars",
             lambda b, w=_w: log_ret(b).rolling(w).std())

for _s in (12, 48):
    register(f"ewm_vol_{_s}", "volatility", _s + 1, f"sqrt of EWMA (span {_s}) of squared 1-bar log returns",
             lambda b, s=_s: np.sqrt((log_ret(b) ** 2).ewm(span=s, min_periods=s).mean()))

for _w in (12, 48):
    register(f"parkinson_{_w}", "volatility", _w, f"Parkinson high-low volatility over {_w} bars",
             lambda b, w=_w: np.sqrt((np.log(b["high"] / b["low"]) ** 2).rolling(w).mean() / (4 * np.log(2))))

register("downside_vol_48", "volatility", 49, "sqrt(mean of squared negative 1-bar returns) over 48 bars",
         lambda b: np.sqrt((log_ret(b).clip(upper=0) ** 2).rolling(48).mean()))
register("upside_vol_48", "volatility", 49, "sqrt(mean of squared positive 1-bar returns) over 48 bars",
         lambda b: np.sqrt((log_ret(b).clip(lower=0) ** 2).rolling(48).mean()))
register("vol_ratio_12_48", "volatility", 49, "rv_12 / rv_48 (volatility regime change)",
         lambda b: safe_div(log_ret(b).rolling(12).std(), log_ret(b).rolling(48).std()))
register("vol_of_vol_48", "volatility", 61, "std over 48 bars of log(rv_12)",
         lambda b: np.log(log_ret(b).rolling(12).std().where(lambda v: v > 0)).rolling(48, min_periods=24).std())


def _atr_rel(b: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = b["close"].shift(1)
    # True range needs the previous close, so the first bar of a segment has none.
    tr = pd.concat([b["high"] - b["low"], (b["high"] - prev).abs(), (b["low"] - prev).abs()], axis=1).max(axis=1, skipna=False)
    return tr.ewm(alpha=1 / n, min_periods=n).mean() / b["close"]


register("atr_14_rel", "volatility", 15, "Wilder ATR(14) / close", _atr_rel)
register("skew_48", "distribution", 49, "skewness of 1-bar log returns over 48 bars", lambda b: log_ret(b).rolling(48).skew())
register("kurt_48", "distribution", 49, "excess kurtosis of 1-bar log returns over 48 bars", lambda b: log_ret(b).rolling(48).kurt())
register("filled_frac_48", "data_quality", 48, "share of the last 48 bars without any trade",
         lambda b: b["is_filled"].astype(float).rolling(48).mean())
