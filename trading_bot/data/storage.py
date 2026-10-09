"""On-disk layout of the normalised research store (all Parquet, all UTC).

    {root}/trades/{SYMBOL}/{YYYY-MM}.parquet        clean ticks: ts(us), price, size, side
    {root}/minutes/{SYMBOL}/{YYYY-MM}.parquet       trade minutes only (pre-filling), per source month
    {root}/bars/{tf}/{SYMBOL}.parquet               completed bars, index = bar open time
    {root}/manifest.json                            source file fingerprints + per-file validation
    {root}/reports/...                              data-quality reports
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TRADE_SCHEMA = pa.schema([("ts", pa.int64()), ("price", pa.float64()), ("size", pa.float64()), ("side", pa.int8())])


class Store:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def trades_path(self, symbol: str, month: str) -> Path:
        return self.root / "trades" / symbol / f"{month}.parquet"

    def minutes_path(self, symbol: str, month: str) -> Path:
        return self.root / "minutes" / symbol / f"{month}.parquet"

    def bars_path(self, symbol: str, tf: str) -> Path:
        return self.root / "bars" / tf / f"{symbol}.parquet"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    # ---------------------------------------------------------------- manifest

    def load_manifest(self) -> dict:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text())
        return {}

    def save_manifest(self, manifest: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True))
        tmp.replace(self.manifest_path)

    # ---------------------------------------------------------------- readers

    def symbols_with_minutes(self) -> list[str]:
        d = self.root / "minutes"
        return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []

    def load_trade_minutes(self, symbol: str) -> pd.DataFrame:
        files = sorted((self.root / "minutes" / symbol).glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no minute files for {symbol}")
        return pd.concat([pd.read_parquet(f) for f in files]).sort_index()

    def load_bars(self, symbol: str, tf: str) -> pd.DataFrame:
        return pd.read_parquet(self.bars_path(symbol, tf))

    def load_trades(self, symbol: str, months: list[str] | None = None) -> pd.DataFrame:
        files = sorted((self.root / "trades" / symbol).glob("*.parquet"))
        if months is not None:
            files = [f for f in files if f.stem in set(months)]
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        return df.sort_values("ts", kind="stable", ignore_index=True)

    # ---------------------------------------------------------------- writers

    def write_frame(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        df.to_parquet(tmp)
        tmp.replace(path)


class TradeParquetWriter:
    """Streams chunks of clean trades into one Parquet file (written atomically on close)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.tmp = path.with_suffix(".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(self.tmp, TRADE_SCHEMA, compression="zstd")

    def __call__(self, trades: pd.DataFrame) -> None:
        t = trades.sort_values("ts", kind="stable")
        self._writer.write_table(pa.Table.from_pandas(t[["ts", "price", "size", "side"]], schema=TRADE_SCHEMA, preserve_index=False))

    def close(self) -> None:
        self._writer.close()
        self.tmp.replace(self.path)

    def abort(self) -> None:
        self._writer.close()
        self.tmp.unlink(missing_ok=True)
