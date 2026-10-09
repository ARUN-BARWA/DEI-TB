import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_bot.data.bars import available_at, complete_minute_bars, minute_bars_to_frame, resample_bars
from trading_bot.data.storage import Store
from trading_bot.data.trades import (
    TradeFile,
    TradeValidator,
    discover_trade_files,
    ingest_trade_file,
    normalise_chunk,
    parse_trade_filename,
)
from trading_bot.data.validate import estimate_clock_offset

REPO = Path(__file__).resolve().parents[1]


def synth_trades(symbol="BTCUSD", start="2026-09-01", n=5000, seed=0, tick=0.5, span_s=6 * 3600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start)
    offs = np.sort(rng.integers(0, span_s * 1_000_000, n))
    price = 78_000 + tick * np.cumsum(rng.integers(-2, 3, n))
    return pd.DataFrame(
        {
            "product_symbol": symbol,
            "price": price,
            "size": rng.integers(1, 200, n).astype(float),
            "timestamp": (t0 + pd.to_timedelta(offs, unit="us")).strftime("%Y-%m-%d %H:%M:%S.%f"),
            "buyer_role": rng.choice(["maker", "taker"], n),
        }
    )


def write_zip(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(path.with_suffix(".csv").name, df.to_csv(index=False))
    return path


def reference_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """Straightforward full-data 1m aggregation to compare against the chunked pipeline."""
    t = df.assign(ts=pd.to_datetime(df["timestamp"], utc=True)).sort_values("ts", kind="stable")
    t["side"] = np.where(t["buyer_role"] == "taker", 1, -1)
    g = t.groupby(t["ts"].dt.floor("1min"))
    return pd.DataFrame(
        {
            "open": g["price"].first(),
            "high": g["price"].max(),
            "low": g["price"].min(),
            "close": g["price"].last(),
            "volume": g["size"].sum(),
            "amount": (t["price"] * t["size"]).groupby(t["ts"].dt.floor("1min")).sum(),
            "n_trades": g["size"].size(),
            "buy_volume": t["size"].where(t["side"] > 0, 0.0).groupby(t["ts"].dt.floor("1min")).sum(),
        }
    )


def test_parse_filename():
    f = parse_trade_filename("x/futures-trades-monthly-BTCUSD-2026-09.zip")
    assert (f.symbol, f.month) == ("BTCUSD", "2026-09")
    lo, hi = f.month_bounds_us
    assert pd.Timestamp(lo, unit="us", tz="UTC") == pd.Timestamp("2026-09-01", tz="UTC")
    assert pd.Timestamp(hi, unit="us", tz="UTC") == pd.Timestamp("2026-10-01", tz="UTC")
    with pytest.raises(ValueError):
        parse_trade_filename("trades.csv")


def test_duplicate_source_months_rejected(tmp_path):
    df = synth_trades(n=10)
    write_zip(df, tmp_path / "a" / "futures-trades-monthly-BTCUSD-2026-09.zip")
    write_zip(df, tmp_path / "b" / "futures-trades-monthly-BTCUSD-2026-09.zip")
    with pytest.raises(ValueError, match="double counting"):
        discover_trade_files(tmp_path)


@pytest.mark.parametrize("chunksize", [7, 333, 10_000_000])
def test_minute_bars_match_reference_for_any_chunk_size(tmp_path, chunksize):
    df = synth_trades(n=3000)
    path = write_zip(df, tmp_path / "futures-trades-monthly-BTCUSD-2026-09.zip")
    minutes, report = ingest_trade_file(parse_trade_filename(path), chunksize=chunksize)
    got = minute_bars_to_frame(minutes)
    ref = reference_minutes(df)
    pd.testing.assert_index_equal(got.index, ref.index.rename("timestamp"))
    for c in ref.columns:
        np.testing.assert_allclose(got[c].to_numpy(dtype=float), ref[c].to_numpy(dtype=float), err_msg=c)
    assert report["kept"] == 3000 and report["dropped"] == 0


def test_out_of_order_across_chunks_keeps_true_open_close(tmp_path):
    df = synth_trades(n=400, span_s=600)
    shuffled = df.sample(frac=1.0, random_state=1).reset_index(drop=True)  # badly ordered file
    path = write_zip(shuffled, tmp_path / "futures-trades-monthly-BTCUSD-2026-09.zip")
    minutes, report = ingest_trade_file(parse_trade_filename(path), chunksize=37)
    got = minute_bars_to_frame(minutes)
    ref = reference_minutes(df)
    np.testing.assert_allclose(got["open"], ref["open"])
    np.testing.assert_allclose(got["close"], ref["close"])
    assert report["out_of_order"] > 0


def _chunk(rows):
    return normalise_chunk(pd.DataFrame(rows, columns=["product_symbol", "price", "size", "timestamp", "buyer_role"]).astype(str))


def test_validator_counts_and_drops_bad_rows():
    v = TradeValidator("BTCUSD", TradeFile(Path("x"), "BTCUSD", "2026-09").month_bounds_us)
    good = v.update(_chunk([
        ["BTCUSD", "100", "1", "2026-09-01 00:00:00.1", "maker"],
        ["BTCUSD", "-1", "1", "2026-09-01 00:00:00.2", "maker"],     # bad price
        ["BTCUSD", "100", "0", "2026-09-01 00:00:00.3", "maker"],    # bad size
        ["ETHUSD", "100", "1", "2026-09-01 00:00:00.4", "maker"],    # wrong symbol
        ["BTCUSD", "100", "1", "not a time", "maker"],               # bad timestamp
        ["BTCUSD", "100", "1", "2026-08-31 23:59:59", "maker"],      # previous month
        ["BTCUSD", "100.5", "1", "2026-09-01 00:00:00.5", "unknown"],  # kept, side 0
    ]))
    r = v.report()
    assert len(good) == 2
    assert (r["bad_price"], r["bad_size"], r["symbol_mismatch"], r["bad_timestamp"], r["outside_month"], r["unknown_side"]) == (1, 1, 1, 1, 1, 1)
    assert r["rows"] == 7 and r["kept"] == 2


def test_validator_sequence_stats_carry_across_chunks():
    v = TradeValidator("BTCUSD", None, spike_threshold=0.01, gap_alert_s=60)
    v.update(_chunk([
        ["BTCUSD", "100", "1", "2026-09-01 00:00:00.0", "maker"],
        ["BTCUSD", "100.5", "1", "2026-09-01 00:00:00.5", "taker"],   # sell->buy: +0.5
        ["BTCUSD", "100.5", "1", "2026-09-01 00:00:00.5", "taker"],   # exact duplicate of previous
    ]))
    v.update(_chunk([
        ["BTCUSD", "110", "1", "2026-09-01 00:00:00.6", "taker"],     # spike up ...
        ["BTCUSD", "100", "1", "2026-09-01 00:00:00.7", "maker"],     # ... and back; buy->sell: -10
        ["BTCUSD", "100", "1", "2026-09-01 00:00:00.65", "maker"],    # out of order
        ["BTCUSD", "100", "1", "2026-09-01 00:05:00", "maker"],       # 5 min gap
    ]))
    r = v.report()
    assert r["consecutive_duplicates"] == 1
    assert r["isolated_spikes"] == 1
    assert r["out_of_order"] == 1
    assert r["gaps_over_alert"] == 1 and r["max_gap_s"] == pytest.approx(299.35)
    assert r["sell_to_buy_mean_dp"] == pytest.approx(0.5)
    assert r["buy_to_sell_mean_dp"] == pytest.approx(-10.0)
    assert r["aggressor_side_consistent"] is True
    assert r["min_price_step"] == pytest.approx(0.5)


def test_complete_minute_bars_fills_and_segments():
    idx = pd.DatetimeIndex(["2026-09-01 00:00", "2026-09-01 00:03", "2026-09-01 10:00"], tz="UTC", name="timestamp")
    m = pd.DataFrame(
        {"open": [1.0, 2.0, 3.0], "high": [1.5, 2.5, 3.5], "low": [0.5, 1.5, 2.5], "close": [1.2, 2.2, 3.2],
         "volume": [10.0, 20.0, 30.0], "amount": [12.0, 44.0, 96.0], "n_trades": [1, 2, 3],
         "buy_volume": [5.0, 0.0, 30.0], "sell_volume": [5.0, 20.0, 0.0]},
        index=idx,
    )
    out = complete_minute_bars(m, max_gap="6h")
    assert list(out["segment_id"].unique()) == [0, 1]
    assert len(out) == 5  # 00:00..00:03 (4 minutes) + 10:00
    gap = out.loc["2026-09-01 00:01"]
    assert gap["is_filled"] and gap["volume"] == 0 and gap["n_trades"] == 0
    assert gap["open"] == gap["high"] == gap["low"] == gap["close"] == 1.2 == gap["vwap"]
    assert out.loc["2026-09-01 00:03", "vwap"] == pytest.approx(44.0 / 20.0)
    assert not out[["open", "high", "low", "close", "vwap"]].isna().any().any()


def test_resample_matches_direct_aggregation(tmp_path):
    df = synth_trades(n=4000, span_s=3 * 3600)
    path = write_zip(df, tmp_path / "futures-trades-monthly-BTCUSD-2026-09.zip")
    minutes, _ = ingest_trade_file(parse_trade_filename(path), chunksize=500)
    m1 = complete_minute_bars(minute_bars_to_frame(minutes))
    b15 = resample_bars(m1, "15m")
    t = df.assign(ts=pd.to_datetime(df["timestamp"], utc=True))
    g = t.groupby(t["ts"].dt.floor("15min"))
    np.testing.assert_allclose(b15["open"], g["price"].first())
    np.testing.assert_allclose(b15["high"], g["price"].max())
    np.testing.assert_allclose(b15["low"], g["price"].min())
    np.testing.assert_allclose(b15["close"], g["price"].last())
    np.testing.assert_allclose(b15["volume"], g["size"].sum())
    assert (b15.index.minute % 15 == 0).all()
    assert b15["n_minutes"].max() == 15
    assert available_at(b15.index[:1], "15m")[0] == b15.index[0] + pd.Timedelta("15min")


def test_resample_rejects_segments_sharing_a_bar():
    idx = pd.DatetimeIndex(["2026-09-01 00:00", "2026-09-01 02:30"], tz="UTC", name="timestamp")
    m = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0, "amount": 1.0,
                      "n_trades": 1, "buy_volume": 1.0, "sell_volume": 0.0}, index=idx)
    m1 = complete_minute_bars(m, max_gap="1h")
    assert m1["segment_id"].nunique() == 2
    with pytest.raises(ValueError, match="max_gap"):
        resample_bars(m1, "4h")
    edge = resample_bars(m1, "15m")
    assert edge["is_partial"].all()


def test_clock_offset_recovers_ist_shift():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-09-01", periods=5000, freq="1min", tz="UTC")
    true_px = pd.Series(80_000 * np.exp(np.cumsum(rng.normal(0, 5e-4, len(idx)))), index=idx)
    ref = true_px * np.exp(rng.normal(0, 5e-5, len(idx)))
    ours = true_px.copy()
    ours.index = ours.index + pd.Timedelta(minutes=330)  # labelled in IST instead of UTC
    best, corr = estimate_clock_offset(ours, ref, max_shift_min=400)
    assert best == -330
    assert corr[-330] > 0.9 > corr[0]
    best0, _ = estimate_clock_offset(true_px, ref, max_shift_min=30)
    assert best0 == 0


def test_ingest_script_end_to_end_and_incremental(tmp_path):
    src, root = tmp_path / "src", tmp_path / "store"
    write_zip(synth_trades("BTCUSD", "2026-08-01", n=2000, span_s=4 * 3600, seed=1), src / "futures-trades-monthly-BTCUSD-2026-08.zip")
    write_zip(synth_trades("BTCUSD", "2026-09-01", n=2000, span_s=4 * 3600, seed=2), src / "futures-trades-monthly-BTCUSD-2026-09.zip")
    write_zip(synth_trades("SOLUSD", "2026-09-01", n=500, span_s=3600, seed=3, tick=0.01), src / "futures-trades-monthly-SOLUSD-2026-09.zip")
    cmd = [sys.executable, str(REPO / "scripts" / "ingest_trades.py"), "--src", str(src), "--root", str(root)]

    first = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert first.returncode == 0, first.stderr
    store = Store(root)
    for tf in ("1m", "5m", "15m", "1h", "4h"):
        bars = store.load_bars("BTCUSD", tf)
        assert bars.index.is_monotonic_increasing and bars.index.is_unique
        assert not bars[["open", "high", "low", "close"]].isna().any().any()
    b1 = store.load_bars("BTCUSD", "1m")
    assert b1["segment_id"].nunique() == 2  # August and September data are separated by a long gap
    trades = store.load_trades("BTCUSD")
    assert len(trades) == 4000 and trades["ts"].is_monotonic_increasing
    assert set(trades["side"].unique()) <= {-1, 1}
    summary = pd.read_csv(root / "reports" / "summary.csv")
    assert set(summary["symbol"]) == {"BTCUSD", "SOLUSD"}
    report = json.loads((root / "reports" / "BTCUSD.json").read_text())
    assert set(report["files"]) == {"BTCUSD/2026-08", "BTCUSD/2026-09"}

    second = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert second.returncode == 0, second.stderr
    assert second.stderr.count("(unchanged)") == 3
