"""Free Binance USD-M futures 1-minute klines (data.binance.vision) as auxiliary research history.

Why: the Delta export covers 2026-01 -> 2026-09 only. Binance USD-M perpetual klines from 2024-07
(after the Kronos pretraining cutoff) add ~18 months on which both baselines and Kronos are
out-of-sample. Klines carry taker-buy volume and trade counts, so every bar column (and therefore
every feature) has the same meaning as for Delta trades. Costs stay Delta's: we would trade on Delta.

Kline CSV columns (header present in recent files, absent in old ones):
    open_time, open, high, low, close, volume, close_time, quote_volume, count,
    taker_buy_volume, taker_buy_quote_volume, ignore
"""

from __future__ import annotations

import hashlib
import io
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from trading_bot.data.bars import BAR_COLUMNS
from trading_bot.data.io import parse_timestamps

BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"
KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count",
                 "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def kline_filename(symbol: str, month: str, interval: str = "1m") -> str:
    return f"{symbol}-{interval}-{month}.zip"


def kline_url(symbol: str, month: str, interval: str = "1m", base_url: str = BASE_URL) -> str:
    return f"{base_url}/{symbol}/{interval}/{kline_filename(symbol, month, interval)}"


def months_between(start: str, end: str) -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fetch(url: str, retries: int = 4, timeout: float = 60.0) -> bytes:
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("unreachable")


def download_month(symbol: str, month: str, dest: str | Path, base_url: str = BASE_URL, interval: str = "1m") -> Path:
    """Download one monthly kline zip and verify it against the published SHA-256. Skips verified files."""
    dest = Path(dest) / symbol
    dest.mkdir(parents=True, exist_ok=True)
    name = kline_filename(symbol, month, interval)
    target = dest / name
    url = kline_url(symbol, month, interval, base_url)
    expected = _fetch(url + ".CHECKSUM").decode().split()[0].strip().lower()
    if target.exists() and sha256_of(target) == expected:
        return target
    data = _fetch(url)
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"checksum mismatch for {name}")
    tmp = target.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(target)
    return target


def read_kline_zip(path: str | Path) -> pd.DataFrame:
    """One monthly kline zip -> trade minutes (minutes with >= 1 trade) in the research bar format."""
    with zipfile.ZipFile(path) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"{path} should contain exactly one CSV, found {members}")
        raw_bytes = zf.read(members[0])
    first = raw_bytes.split(b"\n", 1)[0].split(b",")[0].strip()
    has_header = not first.replace(b".", b"").isdigit()
    raw = pd.read_csv(io.BytesIO(raw_bytes), header=0 if has_header else None)
    if not has_header:
        raw.columns = KLINE_COLUMNS[: raw.shape[1]]
    return klines_to_minutes(raw)


def klines_to_minutes(raw: pd.DataFrame) -> pd.DataFrame:
    idx = parse_timestamps(raw["open_time"]).rename("timestamp")
    vol = pd.to_numeric(raw["volume"], errors="coerce").to_numpy(dtype=float)
    buy = pd.to_numeric(raw["taker_buy_volume"], errors="coerce").to_numpy(dtype=float)
    out = pd.DataFrame(
        {
            "open": pd.to_numeric(raw["open"], errors="coerce").to_numpy(dtype=float),
            "high": pd.to_numeric(raw["high"], errors="coerce").to_numpy(dtype=float),
            "low": pd.to_numeric(raw["low"], errors="coerce").to_numpy(dtype=float),
            "close": pd.to_numeric(raw["close"], errors="coerce").to_numpy(dtype=float),
            "volume": vol,
            "amount": pd.to_numeric(raw["quote_volume"], errors="coerce").to_numpy(dtype=float),
            "n_trades": pd.to_numeric(raw["count"], errors="coerce").fillna(0).to_numpy(dtype=np.int64),
            "buy_volume": buy,
            "sell_volume": vol - buy,
        },
        index=idx,
    ).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out[(out["n_trades"] > 0) & (out["volume"] > 0)]
    return out.dropna(subset=["open", "high", "low", "close"])


def kline_quality(minutes: pd.DataFrame) -> dict:
    ohlc_bad = ~((minutes["low"] <= minutes[["open", "close"]].min(axis=1)) & (minutes[["open", "close"]].max(axis=1) <= minutes["high"]))
    return {
        "minutes_with_trades": int(len(minutes)),
        "first": minutes.index[0].isoformat() if len(minutes) else None,
        "last": minutes.index[-1].isoformat() if len(minutes) else None,
        "ohlc_inconsistent": int(ohlc_bad.sum()),
        "not_minute_aligned": int((minutes.index.second != 0).sum() + (minutes.index.microsecond != 0).sum()),
    }


def combine_sources(early: pd.DataFrame, late: pd.DataFrame, switch: str | pd.Timestamp) -> pd.DataFrame:
    """Research series: ``early`` (e.g. Binance) before ``switch``, ``late`` (Delta) from ``switch`` on."""
    t = pd.Timestamp(switch)
    t = t.tz_localize("UTC") if t.tzinfo is None else t
    cols = [c for c in BAR_COLUMNS if c != "vwap"]
    a = early.loc[early.index < t, cols].assign(venue="binance")
    b = late.loc[late.index >= t, cols].assign(venue="delta")
    return pd.concat([a, b]).sort_index()
