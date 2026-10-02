"""Thin, leak-safe wrapper around the upstream Kronos model.

Upstream Kronos is used unmodified (``third_party/Kronos`` pinned submodule). This wrapper adds
what a trading system needs and the upstream API does not expose:

* ``sample_paths``: every sampled forecast path, instead of upstream's mean over samples, so we
  can estimate P(up), quantiles, MAE/MFE and forecast dispersion.
* ``embed``: the causal last-position hidden state, the entropy of the next coarse-token
  distribution, and tokenizer reconstruction error. These are features for a trading head.
* ``force_causal_dependency_layer``: upstream's s2 cross-attention is non-causal in eval mode
  (``is_causal = self.training``), which leaks future positions whenever per-position s2 logits
  are computed over a full sequence (e.g. upstream's fine-tune validation loss). Inference that
  only reads the last position is unaffected. Use this context manager for any eval-mode
  sequence loss.

Normalisation follows upstream exactly: per-window, per-channel z-score using only the context
window, then clipping to +/- ``clip``. Each window therefore depends only on its own rows, so
features at decision time t are computed from the window ending at t and never from later data.
"""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_KRONOS_PATH = Path(os.environ.get("KRONOS_PATH", Path(__file__).resolve().parents[2] / "third_party" / "Kronos"))
if not (_KRONOS_PATH / "model" / "kronos.py").exists():
    raise ImportError(
        f"Kronos source not found at {_KRONOS_PATH}. Run `git submodule update --init` "
        "or set KRONOS_PATH to a Kronos checkout."
    )
if str(_KRONOS_PATH) not in sys.path:
    sys.path.insert(0, str(_KRONOS_PATH))

from model.kronos import Kronos, KronosTokenizer, auto_regressive_inference, calc_time_stamps  # noqa: E402

PRICE_COLS = ["open", "high", "low", "close"]
FEATURE_COLS = PRICE_COLS + ["volume", "amount"]
CLOSE_IDX = FEATURE_COLS.index("close")

# Context limits of the released checkpoints.
MAX_CONTEXT = {"Kronos-mini": 2048, "Kronos-small": 512, "Kronos-base": 512}


def resolve_device(device: str | None = "auto") -> str:
    if device not in (None, "auto"):
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def future_timestamps(last_ts: pd.Timestamp, freq: str | pd.Timedelta, n: int) -> pd.Series:
    """Timestamps of the next ``n`` bars after ``last_ts`` (crypto trades 24/7, no gaps)."""
    step = pd.Timedelta(freq)
    return pd.Series([last_ts + step * (i + 1) for i in range(n)])


@dataclass(frozen=True)
class PreparedWindows:
    """Normalised model inputs for B windows of equal length T."""

    x: np.ndarray  # (B, T, 6) normalised + clipped
    x_stamp: np.ndarray  # (B, T, 5)
    mean: np.ndarray  # (B, 6)
    std: np.ndarray  # (B, 6)
    last_close: np.ndarray  # (B,) raw close at the decision time


@dataclass(frozen=True)
class KronosEmbedding:
    hidden_last: np.ndarray  # (B, d_model) hidden state at the decision bar
    hidden_mean: np.ndarray  # (B, d_model) mean hidden state over the last `pool` bars
    s1_entropy: np.ndarray  # (B,) entropy of next coarse-token distribution, divided by log(vocab)
    recon_error: np.ndarray  # (B,) tokenizer reconstruction MSE (normalised units) over the last `pool` bars


def prepare_windows(dfs: Sequence[pd.DataFrame], timestamps: Sequence[pd.Series], clip: float = 5.0) -> PreparedWindows:
    """Replicates ``KronosPredictor.predict`` preprocessing for each window independently."""
    xs, stamps, means, stds, last_close = [], [], [], [], []
    for df, ts in zip(dfs, timestamps, strict=True):
        missing = [c for c in PRICE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"window is missing columns {missing}")
        df = df.copy()
        if "volume" not in df.columns:
            df["volume"] = 0.0
            df["amount"] = 0.0
        if "amount" not in df.columns:
            df["amount"] = df["volume"] * df[PRICE_COLS].mean(axis=1)
        x = df[FEATURE_COLS].to_numpy(dtype=np.float32)
        if np.isnan(x).any():
            raise ValueError("window contains NaN in price/volume columns")
        if len(ts) != len(x):
            raise ValueError(f"timestamps length {len(ts)} != window length {len(x)}")
        mean, std = x.mean(axis=0), x.std(axis=0)
        xs.append(np.clip((x - mean) / (std + 1e-5), -clip, clip))
        stamps.append(calc_time_stamps(pd.Series(pd.to_datetime(ts)).reset_index(drop=True)).to_numpy(dtype=np.float32))
        means.append(mean)
        stds.append(std)
        last_close.append(float(df["close"].iloc[-1]))
    lengths = {len(x) for x in xs}
    if len(lengths) != 1:
        raise ValueError(f"all windows must have the same length, got {sorted(lengths)}")
    return PreparedWindows(
        x=np.stack(xs).astype(np.float32),
        x_stamp=np.stack(stamps).astype(np.float32),
        mean=np.stack(means),
        std=np.stack(stds),
        last_close=np.asarray(last_close, dtype=np.float64),
    )


@contextlib.contextmanager
def force_causal_dependency_layer(model: Kronos) -> Iterator[None]:
    """Make the s2 cross-attention causal in eval mode without enabling dropout.

    Upstream sets ``is_causal = self.training`` inside ``MultiHeadCrossAttentionWithRoPE``,
    so in eval mode position t attends to hidden states of positions > t. We flip only that
    module's ``training`` flag, keep its dropout off, and restore everything on exit.
    """
    attn = model.dep_layer.cross_attn
    prev_training, prev_dropout_p = attn.training, attn.attn_dropout_p
    prev_resid = attn.resid_dropout.training
    attn.training = True
    attn.attn_dropout_p = 0.0
    attn.resid_dropout.training = False
    try:
        yield
    finally:
        attn.training = prev_training
        attn.attn_dropout_p = prev_dropout_p
        attn.resid_dropout.training = prev_resid


class KronosWrapper:
    def __init__(
        self,
        model: Kronos,
        tokenizer: KronosTokenizer,
        device: str | None = "auto",
        max_context: int = 512,
        clip: float = 5.0,
    ) -> None:
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer.to(self.device).eval()
        self.max_context = max_context
        self.clip = clip

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "NeoQuasar/Kronos-small",
        tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-base",
        device: str | None = "auto",
        max_context: int | None = None,
        clip: float = 5.0,
    ) -> "KronosWrapper":
        model = Kronos.from_pretrained(model_id)
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)
        if max_context is None:
            name = model_id.rstrip("/").split("/")[-1]
            max_context = MAX_CONTEXT.get(name, 512)
        return cls(model, tokenizer, device=device, max_context=max_context, clip=clip)

    # ------------------------------------------------------------------ forecasting

    @torch.no_grad()
    def sample_paths(
        self,
        dfs: Sequence[pd.DataFrame],
        x_timestamps: Sequence[pd.Series],
        y_timestamps: Sequence[pd.Series],
        pred_len: int,
        n_samples: int = 16,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        seed: int | None = None,
        max_batch: int | None = None,
    ) -> np.ndarray:
        """Return sampled paths in price space, shape (B, n_samples, pred_len, 6).

        Rows are laid out exactly as upstream lays out ``sample_count`` copies (sample index
        fastest), so with the same seed the mean over our paths equals upstream ``predict``
        with ``sample_count=n_samples``.
        """
        if any(len(df) > self.max_context for df in dfs):
            raise ValueError(f"window longer than max_context={self.max_context}; truncate it to the lookback first")
        prep = prepare_windows(dfs, x_timestamps, clip=self.clip)
        y_stamp = np.stack(
            [calc_time_stamps(pd.Series(pd.to_datetime(ts)).reset_index(drop=True)).to_numpy(dtype=np.float32) for ts in y_timestamps]
        )
        if y_stamp.shape[1] != pred_len:
            raise ValueError(f"y_timestamps length {y_stamp.shape[1]} != pred_len {pred_len}")

        b = prep.x.shape[0]
        x = np.repeat(prep.x, n_samples, axis=0)
        xs = np.repeat(prep.x_stamp, n_samples, axis=0)
        ys = np.repeat(y_stamp, n_samples, axis=0)

        if seed is not None:
            torch.manual_seed(seed)
        rows = x.shape[0]
        step = rows if max_batch is None else max(1, max_batch)
        out = []
        for start in range(0, rows, step):
            sl = slice(start, start + step)
            preds = auto_regressive_inference(
                self.tokenizer,
                self.model,
                torch.from_numpy(x[sl]).to(self.device),
                torch.from_numpy(xs[sl]).to(self.device),
                torch.from_numpy(ys[sl]).to(self.device),
                self.max_context,
                pred_len,
                self.clip,
                temperature,
                top_k,
                top_p,
                sample_count=1,  # one path per row; upstream's mean over a single sample is the identity
            )
            out.append(preds[:, -pred_len:, :])
        paths = np.concatenate(out, axis=0).reshape(b, n_samples, pred_len, len(FEATURE_COLS))
        return paths * (prep.std[:, None, None, :] + 1e-5) + prep.mean[:, None, None, :]

    # ------------------------------------------------------------------ representations

    @torch.no_grad()
    def embed(self, dfs: Sequence[pd.DataFrame], x_timestamps: Sequence[pd.Series], pool: int = 8) -> KronosEmbedding:
        """Causal features at the last bar of each window. One forward pass, no sampling."""
        if any(len(df) > self.max_context for df in dfs):
            raise ValueError(f"window longer than max_context={self.max_context}")
        prep = prepare_windows(dfs, x_timestamps, clip=self.clip)
        x = torch.from_numpy(prep.x).to(self.device)
        stamp = torch.from_numpy(prep.x_stamp).to(self.device)

        s1_ids, s2_ids = self.tokenizer.encode(x, half=True)
        s1_logits, hidden = self.model.decode_s1(s1_ids, s2_ids, stamp)

        logp = F.log_softmax(s1_logits[:, -1, :].float(), dim=-1)
        entropy = -(logp.exp() * logp).sum(-1) / np.log(logp.shape[-1])

        recon = self.tokenizer.decode([s1_ids, s2_ids], half=True)
        k = min(pool, x.shape[1])
        recon_err = ((recon[:, -k:, :] - x[:, -k:, :]) ** 2).mean(dim=(1, 2))

        return KronosEmbedding(
            hidden_last=hidden[:, -1, :].float().cpu().numpy(),
            hidden_mean=hidden[:, -k:, :].mean(dim=1).float().cpu().numpy(),
            s1_entropy=entropy.cpu().numpy(),
            recon_error=recon_err.float().cpu().numpy(),
        )


# ---------------------------------------------------------------------- path statistics


def summarize_paths(paths: np.ndarray, last_close: float, horizons: Sequence[int]) -> dict[str, float]:
    """Distributional statistics of sampled paths for one window.

    ``paths``: (n_samples, pred_len, 6) in price space. Returns, per horizon h (in bars):
    mean/std/quantiles of log(close_{t+h}/close_t), P(r_h > 0), and the mean over paths of the
    max favourable / adverse log excursion up to h. A decoded bar's high/low can be internally
    inconsistent, so the excursions use max/min over all four price channels.
    """
    if paths.ndim != 3:
        raise ValueError("paths must have shape (n_samples, pred_len, 6)")
    pred_len = paths.shape[1]
    prices = np.maximum(paths[:, :, :4], 1e-12)
    hi = prices.max(axis=2)
    lo = prices.min(axis=2)
    close = prices[:, :, CLOSE_IDX]
    out: dict[str, float] = {}
    for h in horizons:
        if not 1 <= h <= pred_len:
            raise ValueError(f"horizon {h} outside 1..{pred_len}")
        r = np.log(close[:, h - 1] / last_close)
        q05, q25, q50, q75, q95 = np.quantile(r, [0.05, 0.25, 0.5, 0.75, 0.95])
        step_r = np.diff(np.log(np.concatenate([np.full((close.shape[0], 1), last_close), close[:, :h]], axis=1)), axis=1)
        out.update(
            {
                f"r{h}_mean": float(r.mean()),
                f"r{h}_std": float(r.std()),
                f"r{h}_q05": float(q05),
                f"r{h}_q25": float(q25),
                f"r{h}_q50": float(q50),
                f"r{h}_q75": float(q75),
                f"r{h}_q95": float(q95),
                f"r{h}_p_up": float((r > 0).mean()),
                f"r{h}_mfe": float(np.log(hi[:, :h].max(axis=1) / last_close).mean()),
                f"r{h}_mae": float(np.log(lo[:, :h].min(axis=1) / last_close).mean()),
                f"r{h}_path_vol": float(step_r.std(axis=1).mean()),
            }
        )
    return out
