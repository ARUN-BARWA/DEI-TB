import numpy as np
import pandas as pd
import pytest
import torch

from trading_bot.models.kronos_model import Kronos, KronosTokenizer


def make_tiny_kronos(seed: int = 0) -> tuple[Kronos, KronosTokenizer]:
    """Randomly initialised Kronos built from the real upstream classes (no weight download)."""
    torch.manual_seed(seed)
    tokenizer = KronosTokenizer(
        d_in=6, d_model=32, n_heads=2, ff_dim=64, n_enc_layers=2, n_dec_layers=2,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        s1_bits=4, s2_bits=4, beta=0.05, gamma0=1.0, gamma=1.0, zeta=1.0, group_size=4,
    )
    model = Kronos(
        s1_bits=4, s2_bits=4, n_layers=2, d_model=32, n_heads=2, ff_dim=64,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0, token_dropout_p=0.0, learn_te=True,
    )
    return model.eval(), tokenizer.eval()


def make_ohlcv(n: int, seed: int = 0, freq: str = "5min", start: str = "2025-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 50_000 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0, 0.001, n)) * close
    df = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": rng.lognormal(3, 0.5, n),
        },
        index=pd.date_range(start, periods=n, freq=freq, tz="UTC", name="timestamp"),
    )
    df["amount"] = df["volume"] * df["close"]
    return df


@pytest.fixture
def tiny_kronos():
    return make_tiny_kronos()


@pytest.fixture
def ohlcv():
    return make_ohlcv(200)
