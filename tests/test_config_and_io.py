import pandas as pd
import pytest

from trading_bot.config import Mode, load_settings
from trading_bot.data.io import normalize_ohlcv, read_ohlcv


def _write(tmp_path, text):
    p = tmp_path / "settings.yaml"
    p.write_text(text)
    return p


def test_default_settings_never_allow_orders():
    s = load_settings(env={})
    assert s.mode is Mode.SIGNAL_ONLY
    assert s.trading_enabled is False
    assert not s.orders_allowed(env={"TRADING_ENABLED": "true"})


def test_orders_need_mode_config_and_env(tmp_path):
    p = _write(tmp_path, "mode: PAPER\ntrading_enabled: true\n")
    s = load_settings(p, env={})
    assert not s.orders_allowed(env={})
    assert not s.orders_allowed(env={"TRADING_ENABLED": "false"})
    assert not s.orders_allowed(env={"TRADING_ENABLED": "True"})  # exact match only
    assert s.orders_allowed(env={"TRADING_ENABLED": "true"})


def test_env_kill_switch_overrides_live_config(tmp_path):
    p = _write(tmp_path, "mode: LIVE\ntrading_enabled: true\n")
    assert not load_settings(p, env={}).orders_allowed(env={"TRADING_ENABLED": "false"})


def test_config_false_overrides_env_true(tmp_path):
    p = _write(tmp_path, "mode: LIVE\ntrading_enabled: false\n")
    assert not load_settings(p, env={}).orders_allowed(env={"TRADING_ENABLED": "true"})


def test_string_boolean_rejected(tmp_path):
    p = _write(tmp_path, 'trading_enabled: "false"\n')
    with pytest.raises(ValueError):
        load_settings(p, env={})


def test_mode_env_override_and_validation(tmp_path):
    p = _write(tmp_path, "mode: SIGNAL_ONLY\n")
    assert load_settings(p, env={"MODE": "paper"}).mode is Mode.PAPER
    with pytest.raises(ValueError):
        load_settings(p, env={"MODE": "YOLO"})


def test_lookback_cannot_exceed_context(tmp_path):
    p = _write(tmp_path, "kronos:\n  max_context: 512\n  lookback: 600\n")
    with pytest.raises(ValueError):
        load_settings(p, env={})


def test_effective_fees_include_gst():
    c = load_settings(env={}).costs
    assert c.taker_fee_effective == pytest.approx(0.0005 * 1.18)


@pytest.mark.parametrize(
    "ts, expected",
    [
        ([1735689600, 1735689900], "2025-01-01 00:00"),          # seconds
        ([1735689600000, 1735689900000], "2025-01-01 00:00"),    # milliseconds
        (["2025-01-01T00:00:00Z", "2025-01-01T00:05:00Z"], "2025-01-01 00:00"),
    ],
)
def test_normalize_timestamps(ts, expected):
    raw = pd.DataFrame({"Time": ts, "Open": [1, 2], "High": [2, 3], "Low": [0.5, 1], "Close": [1.5, 2.5], "Volume": [10, 20]})
    out = normalize_ohlcv(raw)
    assert out.index[0] == pd.Timestamp(expected, tz="UTC")
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]


def test_read_csv_sorts_and_dedupes(tmp_path):
    p = tmp_path / "btc.csv"
    p.write_text(
        "timestamp,open,high,low,close,volume,turnover\n"
        "2025-01-01 00:05:00,2,3,1,2,5,10\n"
        "2025-01-01 00:00:00,1,2,0.5,1.5,4,6\n"
        "2025-01-01 00:05:00,2,3,1,2.2,6,13\n"
    )
    df = read_ohlcv(p)
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert df["close"].iloc[-1] == 2.2  # last duplicate wins
    assert "amount" in df.columns


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="could not find"):
        normalize_ohlcv(pd.DataFrame({"timestamp": [1], "close": [1.0]}))
