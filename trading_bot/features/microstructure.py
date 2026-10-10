"""Order-flow features from aggressor-signed trade volume (no order book in the data)."""

from __future__ import annotations

from trading_bot.features.registry import log_ret, register, rolling_z, safe_div


def _signed(b):
    return b["buy_volume"] - b["sell_volume"]


def _total(b):
    return b["buy_volume"] + b["sell_volume"]


register("flow_imb_1", "order_flow", 1, "(taker buy - taker sell) / (buy + sell) volume in the bar; 0 without trades",
         lambda b: safe_div(_signed(b), _total(b)).fillna(0.0))
for _k in (6, 24):
    register(f"flow_imb_{_k}", "order_flow", _k, f"taker volume imbalance over the last {_k} bars",
             lambda b, k=_k: safe_div(_signed(b).rolling(k).sum(), _total(b).rolling(k).sum()))
register("signed_vol_z_48", "order_flow", 48, "z-score of signed taker volume vs the last 48 bars",
         lambda b: rolling_z(_signed(b), 48))
register("flow_ret_corr_48", "order_flow", 49, "correlation of 1-bar return and signed volume over 48 bars (price impact)",
         lambda b: log_ret(b).rolling(48).corr(_signed(b)))
register("trade_intensity_12_48", "order_flow", 48, "mean trades per bar, last 12 / last 48",
         lambda b: safe_div(b["n_trades"].rolling(12).mean(), b["n_trades"].rolling(48).mean()))
