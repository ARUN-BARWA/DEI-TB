# Phase 3 — Research dataset: features, labels, walk-forward splits

## What Phase 2 told us (your run, 2026-10-10)

| | Finding | Consequence |
|---|---|---|
| Clock | Best alignment with Binance at shift 0 (corr 0.973; ±330 min ≈ 0) | Timestamps are UTC. No correction needed |
| Aggressor side | `aggressor_ok = True` for all 8 symbols | `buyer_role` mapping confirmed; order-flow features are valid |
| Coverage | Every symbol ends **2026-09-01 23:59**: the September file holds one day | Usable data is 2026-01-01 → 2026-09-01. Holdout = **Jul 1 → Sep 1** |
| Gaps | BTC/ETH/SOL/DOGE/PEPE each have 5 segments: whole UTC days missing on Mar 20, May 11, Jun 24 and Jul 27–31 (all weekdays) | Days missing from the export, not quiet markets (BTC trades ~290k times a day). No window crosses them |
| Liquidity | BTC 290k, ETH 204k, SOL 42k, DOGE 3.8k trades/day. PEPE/MON/PNUT/ZRO 186–426/day with 80–90% empty minutes | Research symbols: **BTCUSD, ETHUSD, SOLUSD, DOGEUSD**. The four small caps are too thin without an order book |
| Costs | Spread proxy: BTC 0.14, ETH 0.47, SOL 1.24, DOGE 2.43 bp. Taker round trip 11.9–14.2 bp | Fees, not spread, dominate for BTC/ETH. The median 1-minute BTC move is 2.5 bp, about 1/5 of a round trip |
| Spikes | ETH 359 single-trade ≥1% excursions that revert at once (BTC 7, SOL 51) | They are real prints but would trigger last-price stops. The validator now records their timestamps (re-run ingest with `--force` to see them). The backtester will need a stop-trigger price that models Delta's mark-price option |

## Kronos benchmark (your GTX 1650)

Clean measurements, per decision, 8-window batches would be faster:

| model | context | mode | ms / decision | hours per year of 15m bars | hours per year of 1h bars |
|---|---|---|---|---|---|
| Kronos-mini | 512 | embed (1 pass) | 11.7 | 0.1 | 0.03 |
| Kronos-mini | 512 | 16 paths × 12 steps | 657 | 6.4 | 1.6 |
| Kronos-mini | 256 | 16 paths × 12 steps | 321 | 3.1 | 0.8 |
| Kronos-small | 128 | 16 paths × 12 steps | 885 | 8.6 | 2.2 |
| Kronos-small | 256 | 16 paths × 12 steps | 1,763 | 17.2 | 4.3 |

**Kronos-small at 256/512 context is not trustworthy.** Times jump 20–80× while PyTorch reports only 0.2–2.2 GB in
use, and they are non-monotonic: 32 samples ran faster than 16, and the single-pass embedding went from 17 ms to
504 ms. That signature fits Windows' *CUDA Sysmem Fallback*: once VRAM (4 GB minus desktop and CUDA context) is
exhausted, the driver silently pages GPU memory into system RAM, and PyTorch keeps reusing those slow blocks.

The fix is in this commit. `KronosWrapper` now caps PyTorch just below the VRAM that is actually free
(`limit_vram`) and bounds sequences per forward pass (`max_batch_rows=128`), so an overflow is a clean OOM, not a
silent slowdown. The benchmark also clears the cache between configs and records reserved memory. Optionally, set
NVIDIA Control Panel → Manage 3D settings → *CUDA – Sysmem Fallback Policy* → *Prefer No Sysmem Fallback*.

Budget for our ~0.67 years × 4 symbols with Kronos-mini: embeddings at every timeframe ≈ 5 h total (1m included).
Sampled paths ≈ 17 h at 15m, 4 h at 1h, ~50 h at 5m (or ~17 h with a decision stride of 3). Phase 5 therefore
starts with **mini embeddings everywhere + mini sampled paths at 15m/1h/4h**. Kronos-small comes in only after a
clean re-benchmark.

## Dataset

`python scripts/build_dataset.py` writes one Parquet file per symbol × timeframe to
`data_store/datasets/<tf>/<SYMBOL>.parquet`. Each row is a decision point at the **close** of a bar
(`decision_time = bar open + tf`) and contains:

- **Meta:** `decision_time`, `segment_id`, `is_filled`, `close`.
- **47 features** in 9 families (price, candle, mean reversion, volume, calendar, volatility, distribution, order
  flow, data quality). Every one is listed with its lookback and `available_at_time_t` in
  [`features.md`](features.md), which is generated from the registry.
- **Labels** for horizons h ∈ {1, 3, 6, 12, 24} bars: `y_ret`, cost-dead-zone `y_dir`, `y_mfe`, `y_mae`, `y_rv`,
  triple barrier `y_tb` (+ `y_tb_amb`, `y_tb_bars`), and `label_end` for purging. See
  `trading_bot/labels/targets.py`.

Label design choices:

- **The triple barrier is resolved on 1-minute bars** inside 5m/15m/1h/4h bars, so "which barrier came first" is
  observed rather than guessed. Only when both are hit inside the same minute is the label ambiguous (labelled 0 and
  flagged).
- **The barrier half-width is never below the round-trip cost:** `max(k·σ_t·√h, cost)`, with σ_t from past returns
  only. A LONG/SHORT label therefore always means a move that would have paid for the trade.
- `y_dir` has a dead zone of ± cost, so "up by less than the fees" is not a win.

## Walk-forward splits (`trading_bot/research/splits.py`)

| fold | train | validation | test |
|---|---|---|---|
| dev_2026-03 | Jan | Feb | Mar |
| dev_2026-04 | Jan–Feb | Mar | Apr |
| dev_2026-05 | Jan–Mar | Apr | May |
| dev_2026-06 | Jan–Apr | May | Jun |
| **holdout** | Jan–May | Jun | **Jul 1 → Sep 1** (locked) |

- **Purging:** a sample is used only if its label is fully known before the next period starts.
- **Embargo:** an extra 12 h gap at each boundary.
- **Holdout lock:** the holdout fold raises `HoldoutLocked` unless explicitly unlocked, and every unlock is appended
  to `data_store/holdout_access.log`.
- Descriptive statistics (such as the cost-hurdle table) use development rows only.

## Leakage protection (tests)

| Test | What it catches |
|---|---|
| truncation test over **every** registered feature (6 cut points) | any use of future bars |
| rewritten-future test | same, by changing prices and volumes after t |
| segment test | rolling windows crossing data gaps |
| `available_at_time_t=False` refusal | marked-late features used as inputs |
| holdout-invariance test | statistics that would change if holdout prices changed |
| fold tests | train/val/test overlap, missing purge or embargo |

Mutation checks were run by deliberately breaking the code. A next-bar feature, a centred rolling mean and a
full-sample z-score were each caught by the truncation test, and loosening the purge by 10 minutes failed the
holdout-invariance test.

## The cost-hurdle report

`data_store/reports/cost_hurdle.csv` answers, with no model at all, *how often does the price move enough to pay for
a taker round trip?* It is computed per symbol × timeframe × horizon on the development period. Its columns:

- the median |move| in bp, and P(|move| > cost);
- the triple-barrier label mix (share LONG / SHORT / NO_TRADE).

If the median move is a small fraction of the cost (as it is for 1-minute bars), even a perfect direction forecast
is mostly un-tradeable there. This table is the first objective input to choosing timeframes.
