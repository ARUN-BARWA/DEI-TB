"""Measure Kronos inference cost on this machine (latency, peak VRAM) and project research cost.

Examples (on the research machine):
    python scripts/kronos_benchmark.py                                   # Kronos-mini + small, CUDA if available
    python scripts/kronos_benchmark.py --models NeoQuasar/Kronos-base --contexts 512
    python scripts/kronos_benchmark.py --tiny --device cpu               # offline smoke test, random weights

Writes reports/kronos_benchmark.csv. Synthetic prices are used: cost does not depend on the data.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading_bot.models.kronos_model import KronosWrapper, future_timestamps, resolve_device  # noqa: E402

TOKENIZER_FOR = {
    "NeoQuasar/Kronos-mini": "NeoQuasar/Kronos-Tokenizer-2k",
    "NeoQuasar/Kronos-small": "NeoQuasar/Kronos-Tokenizer-base",
    "NeoQuasar/Kronos-base": "NeoQuasar/Kronos-Tokenizer-base",
}
BARS_PER_YEAR = {"1m": 525_600, "5m": 105_120, "15m": 35_040, "1h": 8_760, "4h": 2_190}


def _load(model_id: str, device: str, tiny: bool) -> KronosWrapper:
    if tiny:
        from tests.conftest import make_tiny_kronos

        model, tokenizer = make_tiny_kronos()
        return KronosWrapper(model, tokenizer, device=device, max_context=2048)
    return KronosWrapper.from_pretrained(model_id, TOKENIZER_FOR.get(model_id, "NeoQuasar/Kronos-Tokenizer-base"), device=device)


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def run(args: argparse.Namespace) -> pd.DataFrame:
    from tests.conftest import make_ohlcv

    device = resolve_device(args.device)
    print(f"device={device}" + (f" ({torch.cuda.get_device_name(0)})" if device.startswith("cuda") else ""))
    rows = []
    models = ["tiny-random"] if args.tiny else args.models
    for model_id in models:
        wrapper = _load(model_id, device, args.tiny)
        for ctx in args.contexts:
            if ctx > wrapper.max_context:
                continue
            df = make_ohlcv(ctx + max(args.pred_lens), seed=0, freq="5min")
            windows = [df.iloc[i:i + ctx] for i in range(args.batch)]
            x_ts = [pd.Series(w.index) for w in windows]
            dfs = [w.reset_index(drop=True) for w in windows]
            for pred_len in args.pred_lens:
                y_ts = [future_timestamps(w.index[-1], "5min", pred_len) for w in windows]
                for n in args.samples:
                    rec = {"model": model_id, "device": device, "context": ctx, "pred_len": pred_len,
                           "n_samples": n, "batch_windows": args.batch, "mode": "sample"}
                    try:
                        if device.startswith("cuda"):
                            torch.cuda.reset_peak_memory_stats()
                        wrapper.sample_paths(dfs, x_ts, y_ts, pred_len=pred_len, n_samples=n, seed=0)  # warm-up
                        _sync(device)
                        t0 = time.perf_counter()
                        for _ in range(args.repeats):
                            wrapper.sample_paths(dfs, x_ts, y_ts, pred_len=pred_len, n_samples=n, seed=0)
                        _sync(device)
                        sec_per_decision = (time.perf_counter() - t0) / args.repeats / args.batch
                        rec["ms_per_decision"] = round(sec_per_decision * 1000, 2)
                        rec["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20) if device.startswith("cuda") else None
                        for tf, bars in BARS_PER_YEAR.items():
                            rec[f"hours_per_year_{tf}"] = round(sec_per_decision * bars / 3600, 1)
                    except torch.cuda.OutOfMemoryError:
                        rec["ms_per_decision"] = "OOM"
                        torch.cuda.empty_cache()
                    rows.append(rec)
                    print(rec)
            # Single-pass embedding cost (no sampling) for the feature-extractor route.
            wrapper.embed(dfs, x_ts)  # warm-up
            _sync(device)
            t0 = time.perf_counter()
            for _ in range(args.repeats):
                wrapper.embed(dfs, x_ts)
            _sync(device)
            sec = (time.perf_counter() - t0) / args.repeats / args.batch
            rows.append({"model": model_id, "device": device, "context": ctx, "pred_len": 0, "n_samples": 0,
                         "batch_windows": args.batch, "ms_per_decision": round(sec * 1000, 2), "mode": "embed",
                         **{f"hours_per_year_{tf}": round(sec * b / 3600, 2) for tf, b in BARS_PER_YEAR.items()}})
            print(rows[-1])
        del wrapper
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-small"])
    p.add_argument("--device", default="auto")
    p.add_argument("--contexts", nargs="+", type=int, default=[128, 256, 512])
    p.add_argument("--pred-lens", nargs="+", type=int, default=[1, 12])
    p.add_argument("--samples", nargs="+", type=int, default=[1, 16, 32])
    p.add_argument("--batch", type=int, default=4, help="windows per call (decision timestamps batched together)")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--tiny", action="store_true", help="random tiny model, no download (smoke test)")
    p.add_argument("--out", default="reports/kronos_benchmark.csv")
    args = p.parse_args()

    df = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
