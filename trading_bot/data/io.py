"""Read local OHLCV files (CSV or Parquet) into one canonical frame.

Canonical output: UTC ``DatetimeIndex`` named ``timestamp`` = bar **open** time, sorted, unique,
with float columns ``open, high, low, close`` and, when present, ``volume`` and ``amount``.
Column names are matched case-insensitively against common aliases. Numeric timestamps are
interpreted as epoch seconds / ms / us / ns by magnitude.

Phase 2 extends this with validation reports and Parquet partitioning.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": ("timestamp", "time", "datetime", "date", "open_time", "ts", "start", "time_open"),
    "open": ("open", "o"),
    "high": ("high", "h"),
    "low": ("low", "l"),
    "close": ("close", "c"),
    "volume": ("volume", "vol", "v", "base_volume", "volume_contracts"),
    "amount": ("amount", "turnover", "quote_volume", "quote_asset_volume", "value", "notional"),
}


def _epoch_unit(values: np.ndarray) -> str:
    m = float(np.nanmedian(np.abs(values)))
    if m > 1e17:
        return "ns"
    if m > 1e14:
        return "us"
    if m > 1e11:
        return "ms"
    return "s"


def parse_timestamps(col: pd.Series) -> pd.DatetimeIndex:
    if pd.api.types.is_numeric_dtype(col):
        return pd.DatetimeIndex(pd.to_datetime(col, unit=_epoch_unit(col.to_numpy(dtype=float)), utc=True))
    return pd.DatetimeIndex(pd.to_datetime(col, utc=True))


def normalize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    lower = {c.lower().strip(): c for c in raw.columns}
    picked: dict[str, str] = {}
    for canon, names in ALIASES.items():
        for name in names:
            if name in lower:
                picked[canon] = lower[name]
                break
    missing = [c for c in ("timestamp", "open", "high", "low", "close") if c not in picked]
    if missing:
        raise ValueError(f"could not find columns {missing} among {list(raw.columns)}")

    out = pd.DataFrame(index=parse_timestamps(raw[picked["timestamp"]]).rename("timestamp"))
    for canon in ("open", "high", "low", "close", "volume", "amount"):
        if canon in picked:
            out[canon] = pd.to_numeric(raw[picked[canon]], errors="coerce").to_numpy(dtype=float)
    out = out.sort_index()
    return out[~out.index.duplicated(keep="last")]


def read_ohlcv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        raw = pd.read_parquet(path)
    else:
        raw = pd.read_csv(path)
    if raw.index.name and raw.index.name.lower() in ALIASES["timestamp"]:
        raw = raw.reset_index()
    return normalize_ohlcv(raw)
