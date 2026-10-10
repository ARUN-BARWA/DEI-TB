"""Phase 5: compute and cache Kronos features for every decision bar (resumable; GPU recommended).

    # cheap: embeddings + entropy + reconstruction error (one forward pass per bar)
    python scripts/kronos_features.py --model NeoQuasar/Kronos-small --timeframes 1h 4h
    # + sampled forecast paths (16 paths x 12 steps; hours on a GTX 1650 at 1h, run overnight)
    python scripts/kronos_features.py --model NeoQuasar/Kronos-mini --timeframes 4h 1h --paths

Interrupt any time: finished chunks are cached and skipped on the next run. Output:
data_store/kronos/<tag>/<tf>/<SYMBOL>.parquet (+ meta.json), tag = <model>_<emb|paths>.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading_bot.config import load_settings  # noqa: E402
from trading_bot.data.storage import Store  # noqa: E402
from trading_bot.models.kronos_features import kronos_features  # noqa: E402
from trading_bot.models.kronos_model import KronosWrapper  # noqa: E402

TOKENIZER_FOR = {"NeoQuasar/Kronos-mini": "NeoQuasar/Kronos-Tokenizer-2k"}
DEFAULT_LOOKBACK = {"15m": 512, "1h": 256, "4h": 128}  # ~5, ~11 and ~21 days of context


def main() -> None:
    settings = load_settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(settings.data_root))
    p.add_argument("--symbols", nargs="+", default=list(settings.combined.symbols) or list(settings.symbols))
    p.add_argument("--timeframes", nargs="+", default=["4h", "1h"])
    p.add_argument("--model", default=settings.kronos.model_id)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--lookback", type=int, default=None, help="default: 512 at 15m, 256 at 1h, 128 at 4h")
    p.add_argument("--paths", action="store_true", help="also sample forecast paths (expensive)")
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--pred-len", type=int, default=12)
    p.add_argument("--temperature", type=float, default=settings.kronos.temperature)
    p.add_argument("--top-p", type=float, default=settings.kronos.top_p)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=settings.kronos.device)
    p.add_argument("--max-batch-rows", type=int, default=128)
    p.add_argument("--tag", default=None)
    p.add_argument("--tiny", action="store_true", help="random tiny model (smoke test, no download)")
    args = p.parse_args()

    store = Store(args.root)
    if args.tiny:
        from tests.conftest import make_tiny_kronos

        wrapper = KronosWrapper(*make_tiny_kronos(), device="cpu", max_context=2048, max_batch_rows=args.max_batch_rows)
        model_short = "tiny"
    else:
        tok = args.tokenizer or TOKENIZER_FOR.get(args.model, "NeoQuasar/Kronos-Tokenizer-base")
        wrapper = KronosWrapper.from_pretrained(args.model, tok, device=args.device, max_batch_rows=args.max_batch_rows)
        model_short = args.model.rstrip("/").split("/")[-1]
    tag = args.tag or f"{model_short}_{'paths' if args.paths else 'emb'}"
    print(f"model {args.model if not args.tiny else 'tiny'} on {wrapper.device}, tag {tag}")

    for tf in args.timeframes:
        lookback = min(args.lookback or DEFAULT_LOOKBACK.get(tf, 256), wrapper.max_context)
        for sym in args.symbols:
            t0 = time.time()
            bars = store.load_bars(sym, tf)
            print(f"{sym} {tf}: {len(bars):,} bars, lookback {lookback}")
            feats = kronos_features(wrapper, bars, tf, lookback, with_paths=args.paths, pred_len=args.pred_len,
                                    n_samples=args.n_samples, temperature=args.temperature, top_p=args.top_p, seed=args.seed,
                                    cache_dir=store.kronos_chunks_dir(tag, tf, sym, lookback),
                                    log=lambda m: print(m, end="\r", flush=True))
            store.write_frame(feats, store.kronos_path(tag, tf, sym))
            print(f"\n{sym} {tf}: {feats['kr_entropy'].notna().sum():,} rows with features in {time.time() - t0:.0f}s")
        meta = {"model": args.model if not args.tiny else "tiny", "tf": tf, "lookback": lookback, "paths": args.paths,
                "n_samples": args.n_samples, "pred_len": args.pred_len, "temperature": args.temperature, "top_p": args.top_p,
                "seed": args.seed}
        store.kronos_path(tag, tf, "meta").with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"\nnext: python scripts/run_kronos_comparison.py --kronos-tag {tag}")


if __name__ == "__main__":
    main()
