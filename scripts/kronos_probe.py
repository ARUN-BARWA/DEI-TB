"""One end-to-end Kronos forecast on your own data file. A smoke test, not an evaluation.

    python scripts/kronos_probe.py --file D:/data/BTCUSD_1h.csv
    python scripts/kronos_probe.py --file BTCUSD_5m.parquet --at "2025-03-01 12:00" --pred-len 12 --n-samples 32

Prints the data's detected bar size and coverage, the forecast distribution at a few horizons,
the causal embedding features, and (if data exists after --at) the realised returns for reference.
One forecast says nothing about skill; evaluation over thousands of decisions is Phase 5.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading_bot.config import load_settings  # noqa: E402
from trading_bot.data.io import read_ohlcv  # noqa: E402
from trading_bot.models.kronos_model import KronosWrapper, future_timestamps, summarize_paths  # noqa: E402


def main() -> None:
    cfg = load_settings().kronos
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", required=True)
    p.add_argument("--at", default=None, help="decision bar open time (UTC); default = last bar")
    p.add_argument("--model", default=cfg.model_id)
    p.add_argument("--tokenizer", default=cfg.tokenizer_id)
    p.add_argument("--lookback", type=int, default=cfg.lookback)
    p.add_argument("--pred-len", type=int, default=12)
    p.add_argument("--n-samples", type=int, default=cfg.n_samples)
    p.add_argument("--device", default=cfg.device)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    df = read_ohlcv(args.file)
    bar = df.index.to_series().diff().median()
    gaps = int((df.index.to_series().diff() > bar).sum())
    print(f"rows={len(df):,}  span={df.index[0]} -> {df.index[-1]}  bar={bar}  gaps(>1 bar)={gaps}")

    end = len(df) if args.at is None else int(df.index.searchsorted(pd.Timestamp(args.at, tz="UTC"), side="right"))
    if end < args.lookback:
        sys.exit(f"need {args.lookback} bars before the decision time, have {end}")
    window = df.iloc[end - args.lookback:end]
    t = window.index[-1]
    if t <= pd.Timestamp(cfg.pretrain_cutoff, tz="UTC"):
        print(f"NOTE: decision time {t} is before Kronos's pretraining cutoff ({cfg.pretrain_cutoff}); "
              "this period may be in Kronos's training data.")

    wrapper = KronosWrapper.from_pretrained(args.model, args.tokenizer, device=args.device)
    x_ts = pd.Series(window.index)
    y_ts = future_timestamps(t, bar, args.pred_len)
    w = window.reset_index(drop=True)
    paths = wrapper.sample_paths([w], [x_ts], [y_ts], pred_len=args.pred_len, n_samples=args.n_samples, seed=args.seed)
    last_close = float(window["close"].iloc[-1])
    horizons = sorted({1, min(3, args.pred_len), min(6, args.pred_len), args.pred_len})
    summary = summarize_paths(paths[0], last_close, horizons)

    print(f"\ndecision bar {t}  close={last_close:,.2f}  model={args.model} device={wrapper.device}")
    print(f"{'h':>3} {'E[r] bp':>9} {'sd bp':>8} {'P(up)':>6} {'q05 bp':>8} {'q95 bp':>8} {'MFE bp':>8} {'MAE bp':>8}  realised bp")
    for h in horizons:
        realised = ""
        if end - 1 + h < len(df):
            realised = f"{np.log(df['close'].iloc[end - 1 + h] / last_close) * 1e4:9.1f}"
        print(f"{h:>3} {summary[f'r{h}_mean']*1e4:9.1f} {summary[f'r{h}_std']*1e4:8.1f} {summary[f'r{h}_p_up']:6.2f} "
              f"{summary[f'r{h}_q05']*1e4:8.1f} {summary[f'r{h}_q95']*1e4:8.1f} "
              f"{summary[f'r{h}_mfe']*1e4:8.1f} {summary[f'r{h}_mae']*1e4:8.1f}  {realised}")

    emb = wrapper.embed([w], [x_ts])
    print(f"\nembedding dim={emb.hidden_last.shape[1]}  s1_entropy={emb.s1_entropy[0]:.3f}  "
          f"recon_error={emb.recon_error[0]:.4f}")


if __name__ == "__main__":
    main()
