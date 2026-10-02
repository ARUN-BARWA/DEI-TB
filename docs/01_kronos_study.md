# Phase 1 — Kronos study: findings and deliverables

Kronos is vendored **unmodified** as a git submodule at `third_party/Kronos`, pinned to commit
`67b630e` (2026-04-13). Upgrading it is a deliberate, tested change: bump the submodule and re-run the tests.

## Decisions taken from your answers

| Topic | Decision |
|---|---|
| Hardware | GTX 1650 (4 GB, Turing, no tensor cores). Default **Kronos-small** for forecasting, **Kronos-mini** where long context helps. Kronos-base is inference-only on this card. Keep fp32 (fp16 gains are small without tensor cores); confirm with the benchmark. |
| Symbols | BTCUSD, ETHUSD, plus **SOLUSD** as the more volatile leg (liquid enough that cost estimates stay realistic; DOGEUSD/XRPUSD are alternatives). Very small caps are avoided because their spreads would dominate any backtest. |
| Data | Your local files are the primary source. The API adapter comes later (Phase 12–13). |
| Extra data | Free/open sources only, e.g. Binance's public bulk archive (`data.binance.vision`: klines, aggTrades, funding, OI metrics for USDⓈ-M perps). Used only as auxiliary research data; final validation stays on Delta data. |

## Confirmed findings (from code, verified by tests where marked)

1. **Pretraining cutoff: June 2024** (paper, via search). Any decision time on or before 2024-06-30 may be in
   Kronos's training data. **All Kronos test windows start 2024-07-01 or later.** That gives roughly 27 months of clean
   out-of-sample data up to now. The probe script warns when it is asked for an earlier date.
2. **Upstream `predict()` averages sample paths.** Our `KronosWrapper.sample_paths` returns every path. With the same
   seed, its mean equals upstream `predict(sample_count=N)` to float32 precision, so it is the upstream sampler, not a
   re-implementation. *(test `test_mean_of_paths_matches_upstream_predict`)*
3. **The causal hidden state is a valid time-t feature.** The hidden state at position t is identical whether or not
   later tokens are present, and changing bars after t leaves the features at t unchanged.
   *(tests `test_hidden_state_is_causal_within_window`, `test_embedding_depends_only_on_past`)*
4. **Upstream leak in eval mode.** `MultiHeadCrossAttentionWithRoPE` uses `is_causal = self.training`. In eval mode the
   s2 head at position t attends to positions after t. Effects:
   - forecasting is **not** affected, because it only reads the last position (verified);
   - upstream `train_predictor.py` computes its **validation loss in eval mode over full sequences**, so the s2 part of
     val loss sees the future and checkpoint selection is biased.

   `force_causal_dependency_layer()` fixes it without touching upstream. Phase 6 fine-tuning will use it.
   *(test `test_upstream_eval_cross_attention_leaks_and_fix_restores_causality`)*
5. **Upstream demo splits overlap** (train/val/test date ranges intersect in `finetune/config.py`). Not reused.
6. **Normalisation is per-window z-score + ±5σ clip.** It is causal, but it removes level and scale, so the trading head
   must get scale features (realised vol, ATR in bp) separately. Clipping hides crash bars. A future option is to
   measure how often |z| > 5 occurs in our data.
7. **No KV cache.** Cost grows with `pred_len × samples × context²`. The single-pass `embed()` route (no sampling) is far
   cheaper, which favours option C for the low timeframes. `scripts/kronos_benchmark.py` quantifies this on your GPU.
8. Decoded bars can be internally inconsistent (high < close). The path statistics use max/min over all four price
   channels for MFE/MAE.

## Deliverables

| File | Purpose |
|---|---|
| `trading_bot/models/kronos_model.py` | `KronosWrapper` (`sample_paths`, `embed`), `summarize_paths`, `force_causal_dependency_layer`, `future_timestamps` |
| `trading_bot/config/` | settings with safety defaults (`SIGNAL_ONLY`; orders only if config **and** env `TRADING_ENABLED=true`) |
| `trading_bot/data/io.py` | reads your local CSV/Parquet with common column aliases and epoch-unit detection |
| `scripts/kronos_benchmark.py` | latency / VRAM / research-hours-per-year table on your machine |
| `scripts/kronos_probe.py` | one forecast + embedding on your own file (smoke test) |
| `tests/` | 21 tests, all passing on CPU with tiny random-weight Kronos models built from the real upstream classes |

### Path-statistics features (per horizon h, from N sampled paths)

`r{h}_mean, r{h}_std, r{h}_q05..q95, r{h}_p_up, r{h}_mfe, r{h}_mae, r{h}_path_vol`

### Embedding features (single forward pass)

`hidden_last` (d_model), `hidden_mean` (last 8 bars), `s1_entropy` (next-token uncertainty, 0–1),
`recon_error` (tokenizer novelty)

## Phase 1 exit gate

| Item | Status |
|---|---|
| Wrapper reproduces upstream `predict()` mean | ✅ tested |
| Leak audit of Kronos | ✅ one upstream issue found, workaround tested |
| Pretraining cutoff documented | ✅ June 2024 |
| Latency / VRAM table on the GTX 1650 | ⏳ **needs you to run** `python scripts/kronos_benchmark.py` |
| Real-weight smoke test on your data | ⏳ **needs you to run** `python scripts/kronos_probe.py --file <your BTCUSD file>` |
