import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_bot.data.bars import complete_minute_bars, resample_bars
from trading_bot.data.binance import combine_sources, download_month, kline_url, read_kline_zip
from trading_bot.data.storage import Store
from trading_bot.research.evaluate import (
    TradeRules,
    benjamini_hochberg,
    ic_t_stat,
    minute_price_lookup,
    simulate_trades,
)

REPO = Path(__file__).resolve().parents[1]
HEADER = "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore"


def kline_csv(start: str, n: int, seed: int = 0, unit: str = "ms", header: bool = True, zero_rows=()) -> str:
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    ot = (t.asi8 // (1_000_000 if unit == "ms" else 1_000))
    close = 60_000 * np.exp(np.cumsum(rng.normal(0, 5e-4, n)))
    open_ = np.r_[close[0], close[:-1]]
    vol = rng.uniform(1, 10, n)
    buy = vol * rng.uniform(0.2, 0.8, n)
    count = rng.integers(10, 100, n)
    for z in zero_rows:
        vol[z] = buy[z] = 0.0
        count[z] = 0
    lines = [HEADER] if header else []
    for i in range(n):
        hi, lo = max(open_[i], close[i]) * 1.0002, min(open_[i], close[i]) * 0.9998
        lines.append(f"{ot[i]},{open_[i]:.2f},{hi:.2f},{lo:.2f},{close[i]:.2f},{vol[i]:.4f},{ot[i] + 59999},"
                     f"{vol[i] * close[i]:.2f},{count[i]},{buy[i]:.4f},{buy[i] * close[i]:.2f},0")
    return "\n".join(lines) + "\n"


def write_kline_zip(path: Path, csv: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(path.with_suffix(".csv").name, csv)
    return path


def test_complete_minute_bars_forced_break_without_gap():
    idx = pd.date_range("2025-12-31 23:55", periods=10, freq="1min", tz="UTC", name="timestamp")
    m = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0, "amount": 1.0,
                      "n_trades": 1, "buy_volume": 0.5, "sell_volume": 0.5}, index=idx)
    out = complete_minute_bars(m, breaks=["2026-01-01"])
    assert out["segment_id"].nunique() == 2
    assert out.loc[out.index < pd.Timestamp("2026-01-01", tz="UTC"), "segment_id"].eq(0).all()
    assert out.loc[out.index >= pd.Timestamp("2026-01-01", tz="UTC"), "segment_id"].eq(1).all()


@pytest.mark.parametrize("unit,header", [("ms", True), ("ms", False), ("us", True)])
def test_read_kline_zip(tmp_path, unit, header):
    path = write_kline_zip(tmp_path / "BTCUSDT-1m-2025-01.zip", kline_csv("2025-01-01", 50, unit=unit, header=header, zero_rows=(7,)))
    m = read_kline_zip(path)
    assert len(m) == 49 and m.index[0] == pd.Timestamp("2025-01-01", tz="UTC")
    assert pd.Timestamp("2025-01-01 00:07", tz="UTC") not in m.index        # minute without trades dropped
    np.testing.assert_allclose(m["buy_volume"] + m["sell_volume"], m["volume"])
    assert list(m.columns) == ["open", "high", "low", "close", "volume", "amount", "n_trades", "buy_volume", "sell_volume"]


def _publish(root: Path, symbol: str, month: str, csv: str, corrupt_checksum=False) -> None:
    z = write_kline_zip(root / symbol / "1m" / f"{symbol}-1m-{month}.zip", csv)
    digest = hashlib.sha256(z.read_bytes()).hexdigest()
    if corrupt_checksum:
        digest = "0" * 64
    (z.parent / (z.name + ".CHECKSUM")).write_text(f"{digest}  {z.name}\n")


def test_download_month_verifies_checksum(tmp_path):
    server = tmp_path / "server"
    _publish(server, "BTCUSDT", "2025-01", kline_csv("2025-01-01", 20))
    _publish(server, "ETHUSDT", "2025-01", kline_csv("2025-01-01", 20), corrupt_checksum=True)
    base = server.as_uri()
    assert kline_url("BTCUSDT", "2025-01", base_url=base).endswith("/BTCUSDT/1m/BTCUSDT-1m-2025-01.zip")
    got = download_month("BTCUSDT", "2025-01", tmp_path / "dest", base_url=base)
    assert got.read_bytes() == (server / "BTCUSDT" / "1m" / "BTCUSDT-1m-2025-01.zip").read_bytes()
    mtime = got.stat().st_mtime_ns
    download_month("BTCUSDT", "2025-01", tmp_path / "dest", base_url=base)
    assert got.stat().st_mtime_ns == mtime                                   # verified file not re-downloaded
    with pytest.raises(ValueError, match="checksum"):
        download_month("ETHUSDT", "2025-01", tmp_path / "dest", base_url=base)
    assert not (tmp_path / "dest" / "ETHUSDT" / "ETHUSDT-1m-2025-01.zip").exists()


def test_combine_sources_switches_venue():
    idx = pd.date_range("2025-12-31 23:50", periods=20, freq="1min", tz="UTC", name="timestamp")
    cols = {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0, "amount": 1.0, "n_trades": 1,
            "buy_volume": 0.5, "sell_volume": 0.5}
    early, late = pd.DataFrame(cols, index=idx), pd.DataFrame({**cols, "close": 2.0}, index=idx)
    c = combine_sources(early, late, "2026-01-01")
    assert (c.loc[c.index < "2026-01-01", "venue"] == "binance").all() and (c.loc[c.index >= "2026-01-01", "close"] == 2.0).all()
    assert c.index.is_unique and len(c) == 20


def test_ingest_binance_script(tmp_path):
    src, root = tmp_path / "binance", tmp_path / "store"
    for month, start in (("2025-11", "2025-11-30 22:00"), ("2025-12", "2025-12-31 20:00")):
        write_kline_zip(src / "BTCUSDT" / f"BTCUSDT-1m-{month}.zip", kline_csv(start, 120, seed=len(month) + int(month[-1])))
    store = Store(root)
    d_idx = pd.date_range("2026-01-01", periods=600, freq="1min", tz="UTC", name="timestamp")
    delta = pd.DataFrame({"open": 90_000.0, "high": 90_010.0, "low": 89_990.0, "close": 90_000.0, "volume": 5.0,
                          "amount": 450_000.0, "n_trades": 3, "buy_volume": 3.0, "sell_volume": 2.0}, index=d_idx)
    store.write_frame(delta, store.minutes_path("BTCUSD", "2026-01"))
    res = subprocess.run([sys.executable, str(REPO / "scripts" / "ingest_binance.py"), "--src", str(src), "--root", str(root),
                          "--coins", "BTC", "--timeframes", "1m", "1h"], capture_output=True, text=True, cwd=REPO)
    assert res.returncode == 0, res.stderr
    m1 = store.load_bars("BTC", "1m")
    switch = pd.Timestamp("2026-01-01", tz="UTC")
    assert m1.loc[m1.index < switch, "close"].lt(80_000).all() and m1.loc[m1.index >= switch, "close"].eq(90_000).all()
    assert m1.loc[m1.index >= switch, "segment_id"].nunique() == 1
    assert set(m1.loc[m1.index < switch, "segment_id"]).isdisjoint(set(m1.loc[m1.index >= switch, "segment_id"]))
    assert store.bars_path("BTC", "1h").exists() and (root / "reports" / "combined_BTC.json").exists()


def test_minute_lag_pricing():
    minutes = pd.date_range("2026-03-01", periods=180, freq="1min", tz="UTC", name="timestamp")
    m1 = pd.DataFrame({"open": 0.0, "high": 0.0, "low": 0.0, "close": 100.0 + np.arange(180.0), "segment_id": 0}, index=minutes)
    bars = resample_bars(m1.assign(volume=1.0, amount=1.0, n_trades=1, buy_volume=0.5, sell_volume=0.5, is_filled=False), "1h")
    frame = bars.assign(decision_time=bars.index + pd.Timedelta("1h"))
    rules0 = TradeRules(h=1, round_trip_cost=0.0)
    lookup = minute_price_lookup(m1)
    plain = simulate_trades(frame, np.ones(len(frame)), 0.5, rules0)
    via_minutes = simulate_trades(frame, np.ones(len(frame)), 0.5, rules0, price_at=lookup)
    np.testing.assert_allclose(plain["gross"], via_minutes["gross"])               # lag 0 == bar closes
    lagged = simulate_trades(frame, np.ones(len(frame)), 0.5, TradeRules(h=1, round_trip_cost=0.0, lag_minutes=2), price_at=lookup)
    # decision 01:00 -> entry at the last trade before 01:02 (close of minute 61 = 161), exit before 02:02 (221)
    assert lagged["gross"].iloc[0] == pytest.approx(np.log(221 / 161))
    assert lagged["entry_time"].iloc[0] == pd.Timestamp("2026-03-01 01:02", tz="UTC")
    with pytest.raises(ValueError, match="price lookup"):
        simulate_trades(frame, np.ones(len(frame)), 0.5, TradeRules(h=1, round_trip_cost=0.0, lag_minutes=2))


def test_benjamini_hochberg_and_ic_t():
    q = benjamini_hochberg([0.01, 0.04, 0.03, 0.2, np.nan])
    # sorted p 0.01, 0.03, 0.04, 0.2 -> p*m/rank 0.04, 0.06, 0.0533, 0.2 -> running min from the top
    np.testing.assert_allclose(q[:4], [0.04, 0.16 / 3, 0.16 / 3, 0.2])
    assert np.isnan(q[4])
    assert ic_t_stat(0.1, 2400, 24) == pytest.approx(0.1 * np.sqrt(98 / 0.99))
    assert np.isnan(ic_t_stat(0.1, 30, 24))
