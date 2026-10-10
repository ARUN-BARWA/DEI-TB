"""Ingest Delta tick-trade files (monthly CSV / zipped CSV) in bounded memory.

Raw schema (Delta "futures-trades-monthly" export):
    product_symbol, price, size, timestamp, buyer_role

Normalised trade frame:
    ts    int64  microseconds since epoch, UTC (raw timestamps are naive; assumed UTC,
                 checked separately with ``validate.estimate_clock_offset``)
    price float64
    size  float64  contracts
    side  int8   +1 buyer was taker (buy aggressor), -1 buyer was maker (sell aggressor), 0 unknown

Files are read in chunks. Each chunk is validated (statistics carried across chunk boundaries)
and reduced to partial 1-minute bars; partials are merged exactly using first/last trade times,
so results do not depend on the chunk size or on chunk boundaries.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

RAW_COLUMNS = ["product_symbol", "price", "size", "timestamp", "buyer_role"]
US_PER_MIN = 60_000_000
FILENAME_RE = re.compile(r"(?P<symbol>[A-Z0-9]+)-(?P<year>\d{4})-(?P<month>\d{2})")


@dataclass(frozen=True)
class TradeFile:
    path: Path
    symbol: str
    month: str  # "YYYY-MM"

    @property
    def month_bounds_us(self) -> tuple[int, int]:
        start = pd.Timestamp(f"{self.month}-01", tz="UTC")
        end = start + pd.offsets.MonthBegin(1)
        return int(start.value // 1000), int(end.value // 1000)


def parse_trade_filename(path: str | Path) -> TradeFile:
    """``futures-trades-monthly-BTCUSD-2026-09.zip`` -> (BTCUSD, 2026-09)."""
    path = Path(path)
    m = FILENAME_RE.search(path.name)
    if not m:
        raise ValueError(f"cannot infer symbol/month from file name {path.name!r}")
    return TradeFile(path, m["symbol"], f"{m['year']}-{m['month']}")


def discover_trade_files(src: str | Path, symbols: list[str] | None = None, pattern: str = "*trades*") -> list[TradeFile]:
    src = Path(src)
    paths = sorted(p for p in src.rglob(pattern) if p.is_file() and p.suffix.lower() in (".zip", ".csv", ".gz"))
    files = [parse_trade_filename(p) for p in paths]
    if symbols:
        wanted = set(symbols)
        files = [f for f in files if f.symbol in wanted]
    seen: dict[tuple[str, str], Path] = {}
    for f in files:
        key = (f.symbol, f.month)
        if key in seen:
            raise ValueError(f"two files for {key}: {seen[key]} and {f.path}; remove one to avoid double counting")
        seen[key] = f.path
    return files


def _open_csv(path: Path):
    if path.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(path)
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"{path} should contain exactly one CSV, found {members}")
        return io.TextIOWrapper(zf.open(members[0]), encoding="utf-8")
    return path


def read_raw_chunks(path: str | Path, chunksize: int = 2_000_000) -> Iterator[pd.DataFrame]:
    src = _open_csv(Path(path))
    reader = pd.read_csv(src, usecols=RAW_COLUMNS, chunksize=chunksize, dtype=str, keep_default_na=True)
    try:
        yield from reader
    finally:
        reader.close()


def normalise_chunk(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw string columns -> typed columns. Unparseable values become NaN / NaT (counted by the validator)."""
    ts = pd.to_datetime(raw["timestamp"], format="ISO8601", utc=True, errors="coerce")
    # Explicit unit: pandas 2 parses to ns, pandas 3 may infer us. NaT becomes NAT_US.
    ts_us = ts.dt.tz_convert(None).dt.as_unit("us").to_numpy().view(np.int64)
    role = raw["buyer_role"].astype("string").str.strip().str.lower()
    side = np.where((role == "taker").fillna(False), 1, np.where((role == "maker").fillna(False), -1, 0)).astype(np.int8)
    return pd.DataFrame(
        {
            "symbol": raw["product_symbol"].astype("string").str.strip().to_numpy(),
            "ts": ts_us,
            "price": pd.to_numeric(raw["price"], errors="coerce").to_numpy(dtype=float),
            "size": pd.to_numeric(raw["size"], errors="coerce").to_numpy(dtype=float),
            "side": side,
        }
    )


NAT_US = np.iinfo(np.int64).min


@dataclass
class TradeValidator:
    """Accumulates data-quality statistics for one file across chunks. Never modifies data."""

    expected_symbol: str
    month_bounds_us: tuple[int, int] | None = None
    spike_threshold: float = 0.01  # isolated single-trade excursion that reverts, as a log return
    gap_alert_s: float = 300.0
    spread_pair_max_s: float = 1.0
    max_spread_samples: int = 500_000

    rows: int = 0
    kept: int = 0
    bad_timestamp: int = 0
    bad_price: int = 0
    bad_size: int = 0
    unknown_side: int = 0
    symbol_mismatch: int = 0
    outside_month: int = 0
    out_of_order: int = 0
    consecutive_duplicates: int = 0
    isolated_spikes: int = 0
    spike_times_us: list = field(default_factory=list)  # first 50, for inspection
    gaps_over_alert: int = 0
    max_gap_s: float = 0.0
    max_gap_at_us: int | None = None
    min_ts: int | None = None
    max_ts: int | None = None
    min_price_step: float = float("inf")
    min_size: float = float("inf")
    non_integer_sizes: int = 0
    # Aggressor-side sanity: a buy-aggressor trade right after a sell-aggressor trade should
    # print at a higher price (ask vs bid). Also a direct effective-spread proxy.
    sell_to_buy_n: int = 0
    sell_to_buy_dp_sum: float = 0.0
    buy_to_sell_n: int = 0
    buy_to_sell_dp_sum: float = 0.0
    spread_samples_bp: list = field(default_factory=list)
    _tail: pd.DataFrame | None = None

    def update(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Validate a normalised chunk; return the rows that are safe to aggregate."""
        n = len(chunk)
        self.rows += n
        bad_ts = chunk["ts"].to_numpy() == NAT_US
        bad_px = ~(chunk["price"].to_numpy() > 0)
        bad_sz = ~(chunk["size"].to_numpy() > 0)
        mismatch = (chunk["symbol"] != self.expected_symbol).fillna(True).to_numpy(dtype=bool)
        self.bad_timestamp += int(bad_ts.sum())
        self.bad_price += int(bad_px.sum())
        self.bad_size += int(bad_sz.sum())
        self.symbol_mismatch += int(mismatch.sum())
        self.unknown_side += int((chunk["side"].to_numpy() == 0).sum())
        keep = ~(bad_ts | bad_px | bad_sz | mismatch)
        if self.month_bounds_us is not None:
            lo, hi = self.month_bounds_us
            ts = chunk["ts"].to_numpy()
            outside = keep & ((ts < lo) | (ts >= hi))
            self.outside_month += int(outside.sum())
            keep &= ~outside
        good = chunk.loc[keep, ["ts", "price", "size", "side"]].reset_index(drop=True)
        self.kept += len(good)
        if good.empty:
            return good

        ts = good["ts"].to_numpy()
        self.min_ts = int(ts.min()) if self.min_ts is None else min(self.min_ts, int(ts.min()))
        self.max_ts = int(ts.max()) if self.max_ts is None else max(self.max_ts, int(ts.max()))
        sz = good["size"].to_numpy()
        self.min_size = min(self.min_size, float(sz.min()))
        self.non_integer_sizes += int((sz != np.round(sz)).sum())

        # Sequential statistics in file order, including the last two rows of the previous chunk.
        n_tail = 0 if self._tail is None else len(self._tail)
        comb = good if self._tail is None else pd.concat([self._tail, good], ignore_index=True)
        cts = comb["ts"].to_numpy()
        cpx = comb["price"].to_numpy()
        csd = comb["side"].to_numpy()
        csz = comb["size"].to_numpy()
        new = slice(max(n_tail, 1), len(comb))  # rows that have a predecessor and are new

        dts = np.diff(cts)[new.start - 1:]
        self.out_of_order += int((dts < 0).sum())
        gaps = dts / 1e6
        if gaps.size:
            i = int(np.argmax(gaps))
            if gaps[i] > self.max_gap_s:
                self.max_gap_s = float(gaps[i])
                self.max_gap_at_us = int(cts[new.start + i - 1])
            self.gaps_over_alert += int((gaps > self.gap_alert_s).sum())

        dp = np.diff(cpx)[new.start - 1:]
        nz = np.abs(dp[dp != 0])
        if nz.size:
            self.min_price_step = min(self.min_price_step, float(nz.min()))

        same = (np.diff(cts) == 0) & (np.diff(cpx) == 0) & (np.diff(csz) == 0) & (np.diff(csd) == 0)
        self.consecutive_duplicates += int(same[new.start - 1:].sum())

        prev_side, cur_side = csd[:-1][new.start - 1:], csd[1:][new.start - 1:]
        close_in_time = (dts >= 0) & (dts <= self.spread_pair_max_s * 1e6)
        s2b = (prev_side == -1) & (cur_side == 1) & close_in_time
        b2s = (prev_side == 1) & (cur_side == -1) & close_in_time
        self.sell_to_buy_n += int(s2b.sum())
        self.sell_to_buy_dp_sum += float(dp[s2b].sum())
        self.buy_to_sell_n += int(b2s.sum())
        self.buy_to_sell_dp_sum += float(dp[b2s].sum())
        mid_px = cpx[1:][new.start - 1:]
        bp = (dp[s2b] / mid_px[s2b]) * 1e4
        room = self.max_spread_samples - len(self.spread_samples_bp)
        if room > 0 and bp.size:
            self.spread_samples_bp.extend(bp[:room].tolist())

        if len(comb) >= 3:
            # Middles comb[1:-1]. The carried tail's first row was already a middle in the previous
            # chunk (it sits at index 0 here); its second row was not (it had no successor), so
            # every middle of comb is new.
            r = np.diff(np.log(cpx))
            r_in, r_out = r[:-1], r[1:]
            spike = (np.abs(r_in) > self.spike_threshold) & (np.abs(r_out) > self.spike_threshold) & (np.sign(r_in) != np.sign(r_out))
            self.isolated_spikes += int(spike.sum())
            room = 50 - len(self.spike_times_us)
            if room > 0 and spike.any():
                self.spike_times_us.extend(cts[1:-1][spike][:room].tolist())

        self._tail = comb.iloc[-2:].reset_index(drop=True)
        return good

    def report(self) -> dict:
        def iso(us):
            return None if us is None else pd.Timestamp(us, unit="us", tz="UTC").isoformat()

        sp = np.asarray(self.spread_samples_bp)
        return {
            "rows": self.rows,
            "kept": self.kept,
            "dropped": self.rows - self.kept,
            "bad_timestamp": self.bad_timestamp,
            "bad_price": self.bad_price,
            "bad_size": self.bad_size,
            "symbol_mismatch": self.symbol_mismatch,
            "outside_month": self.outside_month,
            "unknown_side": self.unknown_side,
            "out_of_order": self.out_of_order,
            "consecutive_duplicates": self.consecutive_duplicates,
            "isolated_spikes": self.isolated_spikes,
            "spike_times": [iso(t) for t in self.spike_times_us],
            "gaps_over_alert": self.gaps_over_alert,
            "max_gap_s": round(self.max_gap_s, 3),
            "max_gap_at": iso(self.max_gap_at_us),
            "first_trade": iso(self.min_ts),
            "last_trade": iso(self.max_ts),
            "min_price_step": None if np.isinf(self.min_price_step) else self.min_price_step,
            "min_size": None if np.isinf(self.min_size) else self.min_size,
            "non_integer_sizes": self.non_integer_sizes,
            "sell_to_buy_mean_dp": self.sell_to_buy_dp_sum / self.sell_to_buy_n if self.sell_to_buy_n else None,
            "buy_to_sell_mean_dp": self.buy_to_sell_dp_sum / self.buy_to_sell_n if self.buy_to_sell_n else None,
            "aggressor_side_consistent": (
                None
                if not (self.sell_to_buy_n and self.buy_to_sell_n)
                else bool(self.sell_to_buy_dp_sum / self.sell_to_buy_n > 0 > self.buy_to_sell_dp_sum / self.buy_to_sell_n)
            ),
            "spread_proxy_bp_median": float(np.median(sp)) if sp.size else None,
            "spread_proxy_bp_mean": float(sp.mean()) if sp.size else None,
            "spread_proxy_samples": int(sp.size),
        }


PARTIAL_COLUMNS = ["open", "high", "low", "close", "first_ts", "last_ts", "volume", "amount", "n_trades", "buy_volume", "sell_volume"]


def partial_minute_bars(trades: pd.DataFrame) -> pd.DataFrame:
    """Reduce a chunk of normalised trades to partial 1-minute bars (index = minute start, us)."""
    if trades.empty:
        return pd.DataFrame(columns=PARTIAL_COLUMNS, index=pd.Index([], dtype=np.int64, name="bar"))
    t = trades.sort_values("ts", kind="stable")
    bar = (t["ts"].to_numpy() // US_PER_MIN) * US_PER_MIN
    size = t["size"].to_numpy()
    side = t["side"].to_numpy()
    df = pd.DataFrame(
        {
            "bar": bar,
            "price": t["price"].to_numpy(),
            "ts": t["ts"].to_numpy(),
            "size": size,
            "amount": t["price"].to_numpy() * size,
            "buy": np.where(side > 0, size, 0.0),
            "sell": np.where(side < 0, size, 0.0),
        }
    )
    g = df.groupby("bar", sort=True)
    out = pd.DataFrame(
        {
            "open": g["price"].first(),
            "high": g["price"].max(),
            "low": g["price"].min(),
            "close": g["price"].last(),
            "first_ts": g["ts"].min(),
            "last_ts": g["ts"].max(),
            "volume": g["size"].sum(),
            "amount": g["amount"].sum(),
            "n_trades": g["size"].size(),
            "buy_volume": g["buy"].sum(),
            "sell_volume": g["sell"].sum(),
        }
    )
    return out


def merge_partial_bars(partials: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine partial bars from several chunks into exact 1-minute bars.

    Open comes from the partial with the earliest first trade, close from the one with the latest
    last trade (ties resolved by chunk order, i.e. file order).
    """
    parts = [p.assign(_seq=i) for i, p in enumerate(partials) if not p.empty]
    if not parts:
        return pd.DataFrame(columns=PARTIAL_COLUMNS, index=pd.Index([], dtype=np.int64, name="bar"))
    allp = pd.concat(parts).reset_index()
    by_first = allp.sort_values(["bar", "first_ts", "_seq"], kind="stable").groupby("bar")
    by_last = allp.sort_values(["bar", "last_ts", "_seq"], kind="stable").groupby("bar")
    g = allp.groupby("bar")
    out = pd.DataFrame(
        {
            "open": by_first["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": by_last["close"].last(),
            "first_ts": g["first_ts"].min(),
            "last_ts": g["last_ts"].max(),
            "volume": g["volume"].sum(),
            "amount": g["amount"].sum(),
            "n_trades": g["n_trades"].sum(),
            "buy_volume": g["buy_volume"].sum(),
            "sell_volume": g["sell_volume"].sum(),
        }
    )
    out["n_trades"] = out["n_trades"].astype(np.int64)
    return out


def ingest_trade_file(
    tf: TradeFile,
    chunksize: int = 2_000_000,
    spike_threshold: float = 0.01,
    trade_sink=None,
) -> tuple[pd.DataFrame, dict]:
    """Validate one file and build its 1-minute bars (trade minutes only, no empty-bar filling).

    ``trade_sink(df)`` is called with each chunk of clean, normalised trades (e.g. a Parquet writer).
    """
    validator = TradeValidator(tf.symbol, tf.month_bounds_us, spike_threshold=spike_threshold)
    partials = []
    for raw in read_raw_chunks(tf.path, chunksize=chunksize):
        good = validator.update(normalise_chunk(raw))
        if good.empty:
            continue
        if trade_sink is not None:
            trade_sink(good)
        partials.append(partial_minute_bars(good))
    return merge_partial_bars(partials), validator.report()
