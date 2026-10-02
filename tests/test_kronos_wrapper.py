import numpy as np
import pandas as pd
import pytest
import torch

from tests.conftest import make_ohlcv
from trading_bot.models.kronos_model import (
    KronosWrapper,
    force_causal_dependency_layer,
    future_timestamps,
    prepare_windows,
    summarize_paths,
)
from model.kronos import KronosPredictor  # upstream reference; importable once the wrapper set sys.path  # noqa: E402

LOOKBACK, PRED_LEN = 48, 6


def _window(df: pd.DataFrame, end: int, lookback: int = LOOKBACK):
    w = df.iloc[end - lookback:end]
    x_ts = pd.Series(w.index)
    y_ts = future_timestamps(w.index[-1], "5min", PRED_LEN)
    return w.reset_index(drop=True), x_ts, y_ts


def test_mean_of_paths_matches_upstream_predict(tiny_kronos, ohlcv):
    """Our per-path sampler must be the upstream sampler, not a re-implementation."""
    model, tokenizer = tiny_kronos
    w, x_ts, y_ts = _window(ohlcv, 120)
    n = 8

    torch.manual_seed(123)
    upstream = KronosPredictor(model, tokenizer, device="cpu", max_context=512).predict(
        df=w, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=PRED_LEN, T=1.0, top_p=0.9, sample_count=n, verbose=False
    )

    wrapper = KronosWrapper(model, tokenizer, device="cpu", max_context=512)
    paths = wrapper.sample_paths([w], [x_ts], [y_ts], pred_len=PRED_LEN, n_samples=n, top_p=0.9, seed=123)

    assert paths.shape == (1, n, PRED_LEN, 6)
    np.testing.assert_allclose(paths[0].mean(axis=0), upstream.to_numpy(), rtol=1e-5, atol=0)
    # Paths genuinely differ from each other (the distribution is preserved, not collapsed).
    assert paths[0, :, :, 3].std(axis=0).max() > 0


def test_batched_windows_keep_their_own_scale(tiny_kronos):
    model, tokenizer = tiny_kronos
    btc = make_ohlcv(100, seed=1)
    eth = make_ohlcv(100, seed=2)
    eth[["open", "high", "low", "close"]] /= 20.0
    wrapper = KronosWrapper(model, tokenizer, device="cpu")
    windows = [_window(btc, 100), _window(eth, 100)]
    paths = wrapper.sample_paths(
        [w for w, _, _ in windows], [x for _, x, _ in windows], [y for _, _, y in windows],
        pred_len=PRED_LEN, n_samples=4, seed=0,
    )
    btc_med = np.median(paths[0, :, :, 3])
    eth_med = np.median(paths[1, :, :, 3])
    assert 10 < btc_med / eth_med < 40


def test_window_longer_than_context_is_rejected(tiny_kronos, ohlcv):
    model, tokenizer = tiny_kronos
    wrapper = KronosWrapper(model, tokenizer, device="cpu", max_context=32)
    w, x_ts, y_ts = _window(ohlcv, 120)
    with pytest.raises(ValueError, match="max_context"):
        wrapper.sample_paths([w], [x_ts], [y_ts], pred_len=PRED_LEN, n_samples=2)


def test_embedding_depends_only_on_past(tiny_kronos):
    """Changing bars after the decision time must not change features at the decision time."""
    model, tokenizer = tiny_kronos
    wrapper = KronosWrapper(model, tokenizer, device="cpu")
    df = make_ohlcv(200, seed=3)
    altered = df.copy()
    altered.iloc[120:, :] *= 1.5  # rewrite the "future"

    a = wrapper.embed(*map(list, zip(_window(df, 120)[:2])))
    b = wrapper.embed(*map(list, zip(_window(altered, 120)[:2])))
    np.testing.assert_array_equal(a.hidden_last, b.hidden_last)
    np.testing.assert_array_equal(a.s1_entropy, b.s1_entropy)
    assert a.hidden_last.shape == (1, 32)
    assert 0.0 <= float(a.s1_entropy[0]) <= 1.0
    assert np.isfinite(a.recon_error).all()


def test_hidden_state_is_causal_within_window(tiny_kronos):
    """Hidden state at position t is the same whether or not later tokens are in the sequence."""
    model, tokenizer = tiny_kronos
    prep = prepare_windows(*map(list, zip(_window(make_ohlcv(100), 100)[:2])))
    x = torch.from_numpy(prep.x)
    stamp = torch.from_numpy(prep.x_stamp)
    with torch.no_grad():
        s1, s2 = tokenizer.encode(x, half=True)
        _, h_full = model.decode_s1(s1, s2, stamp)
        _, h_cut = model.decode_s1(s1[:, :30], s2[:, :30], stamp[:, :30])
    torch.testing.assert_close(h_full[:, :30], h_cut, rtol=1e-5, atol=1e-5)


def test_upstream_eval_cross_attention_leaks_and_fix_restores_causality(tiny_kronos):
    """Documents an upstream issue: in eval mode the s2 head at position t sees positions > t."""
    model, tokenizer = tiny_kronos
    prep = prepare_windows(*map(list, zip(_window(make_ohlcv(100), 100)[:2])))
    x = torch.from_numpy(prep.x)
    stamp = torch.from_numpy(prep.x_stamp)
    t = 30
    with torch.no_grad():
        s1, s2 = tokenizer.encode(x, half=True)
        _, h_full = model.decode_s1(s1, s2, stamp)
        _, h_cut = model.decode_s1(s1[:, :t], s2[:, :t], stamp[:, :t])

        leaky_full = model.decode_s2(h_full, s1)[:, :t]
        leaky_cut = model.decode_s2(h_cut, s1[:, :t])
        assert not torch.allclose(leaky_full, leaky_cut, atol=1e-5), "expected upstream eval-mode leak"

        with force_causal_dependency_layer(model):
            fixed_full = model.decode_s2(h_full, s1)[:, :t]
            fixed_cut = model.decode_s2(h_cut, s1[:, :t])
        torch.testing.assert_close(fixed_full, fixed_cut, rtol=1e-5, atol=1e-5)

        # Last-position inference (what forecasting uses) is unaffected by the fix.
        with force_causal_dependency_layer(model):
            last_fixed = model.decode_s2(h_full, s1)[:, -1]
        torch.testing.assert_close(model.decode_s2(h_full, s1)[:, -1], last_fixed, rtol=1e-5, atol=1e-5)

    attn = model.dep_layer.cross_attn
    assert attn.training is False and attn.resid_dropout.training is False


def test_summarize_paths_statistics():
    last = 100.0
    # 4 paths, 2 steps; closes at h=2: 110, 105, 95, 90 -> 2 up, 2 down.
    closes = np.array([[105, 110], [102, 105], [98, 95], [96, 90]], dtype=float)
    paths = np.zeros((4, 2, 6))
    for ch in range(4):
        paths[:, :, ch] = closes
    paths[:, :, 1] = closes * 1.01  # high
    paths[:, :, 2] = closes * 0.99  # low
    s = summarize_paths(paths, last, horizons=[1, 2])
    r2 = np.log(closes[:, 1] / last)
    assert s["r2_p_up"] == 0.5
    assert s["r2_mean"] == pytest.approx(r2.mean())
    assert s["r2_q50"] == pytest.approx(np.median(r2))
    assert s["r2_mfe"] == pytest.approx(np.mean(np.log(closes.max(axis=1) * 1.01 / last)))
    assert s["r2_mae"] == pytest.approx(np.mean(np.log(closes.min(axis=1) * 0.99 / last)))
    with pytest.raises(ValueError):
        summarize_paths(paths, last, horizons=[3])


def test_future_timestamps():
    ts = future_timestamps(pd.Timestamp("2025-01-01 00:55", tz="UTC"), "5min", 3)
    assert list(ts) == list(pd.date_range("2025-01-01 01:00", periods=3, freq="5min", tz="UTC"))
