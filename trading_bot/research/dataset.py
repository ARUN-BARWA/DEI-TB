"""Assemble the research dataset: one row per bar = one decision point at the bar's close."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from trading_bot.config import CostSettings
from trading_bot.data.bars import available_at
from trading_bot.data.storage import Store
from trading_bot.features.registry import compute_features
from trading_bot.labels.targets import forward_labels
from trading_bot.research.splits import development_mask

META_COLUMNS = ["decision_time", "segment_id", "is_filled", "close"]


def round_trip_cost(store: Store, symbol: str, costs: CostSettings) -> float:
    """Taker fee in and out (incl. GST) + median effective-spread proxy from the ingest report."""
    spread_bp = 0.0
    report = store.reports_dir / f"{symbol}.json"
    if report.exists():
        files = json.loads(report.read_text())["files"].values()
        spreads = [f["spread_proxy_bp_median"] for f in files if f.get("spread_proxy_bp_median") is not None]
        spread_bp = float(np.median(spreads)) if spreads else 0.0
    return 2 * costs.taker_fee_effective + spread_bp / 1e4


def build_dataset(
    bars: pd.DataFrame,
    tf: str,
    horizons=(1, 3, 6, 12, 24),
    cost: float = 0.0,
    path: pd.DataFrame | None = None,
    tb_k: float = 1.0,
    vol_window: int = 48,
) -> pd.DataFrame:
    feats = compute_features(bars)
    labels = forward_labels(bars, tf, horizons, cost=cost, path=path, tb_k=tb_k, vol_window=vol_window)
    meta = pd.DataFrame(
        {
            "decision_time": available_at(bars.index, tf),
            "segment_id": bars["segment_id"].to_numpy() if "segment_id" in bars else 0,
            "is_filled": bars["is_filled"].to_numpy() if "is_filled" in bars else False,
            "close": bars["close"].to_numpy(),
        },
        index=bars.index,
    )
    ds = pd.concat([meta, feats, labels], axis=1)
    ds.attrs.update({"tf": tf, "horizons": list(horizons), "cost": cost})
    return ds


def feature_columns(ds: pd.DataFrame) -> list[str]:
    return [c for c in ds.columns if c not in META_COLUMNS and not c.startswith(("y_", "label_end_"))]


def label_columns(ds: pd.DataFrame) -> list[str]:
    return [c for c in ds.columns if c.startswith(("y_", "label_end_"))]


def cost_hurdle_table(ds: pd.DataFrame, horizons, cost: float, holdout_start) -> pd.DataFrame:
    """How often does the price move enough to pay for a round trip? Development period only.

    No model involved: if |move| rarely exceeds the cost at a timeframe/horizon, no forecaster
    can make taker trading there profitable.
    """
    rows = []
    for h in horizons:
        dev = development_mask(ds[f"label_end_{h}"], holdout_start) & ds[f"y_ret_{h}"].notna().to_numpy()
        r = ds.loc[dev, f"y_ret_{h}"]
        tb = ds.loc[dev, f"y_tb_{h}"].dropna()
        rows.append({
            "h_bars": h,
            "n": int(dev.sum()),
            "med_abs_ret_bp": round(float(r.abs().median() * 1e4), 2),
            "p_abs_ret_gt_cost": round(float((r.abs() > cost).mean()), 3),
            "med_abs_ret_over_cost": round(float(r.abs().median() / cost), 2) if cost > 0 else np.nan,
            "tb_long": round(float((tb == 1).mean()), 3),
            "tb_short": round(float((tb == -1).mean()), 3),
            "tb_none": round(float((tb == 0).mean()), 3),
            "tb_ambiguous": round(float(ds.loc[dev, f"y_tb_amb_{h}"].mean()), 4),
        })
    return pd.DataFrame(rows)
