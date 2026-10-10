# Phase 4 results — what the baselines say, and why we extend the history

## Findings from your two runs (Delta, Mar–Jun 2026 test months)

1. **5m and 15m: no baseline survives costs.** Every strategy row is negative net; most models abstain because
   nothing was profitable on validation. This matches the cost-hurdle prediction (break-even IC ≥ 0.1 at those
   horizons). Low-timeframe taker trading is set aside.

2. **The strongest candidate is short-horizon mean reversion at 1h/4h.**

   | | net bp/trade | t | p (sign-flip) | folds + | IC val / test |
   |---|---|---|---|---|---|
   | `mean_reversion_24`, 1h, h=24 | +28.1 | 1.41 | 0.021 | 2/4 | 0.084 / 0.072 |
   | `mean_reversion_24`, 4h, h=3 | +22.6 | 1.28 | 0.023 | 1/4 | 0.103 / 0.066 |

   It beats both always-long and always-short traded at the same moments, and its IC has the same sign on validation
   and test. **But it does not clear the bar:**
   - about 40 strategy rows were tested, so ~2 rows with p < 0.05 are expected by chance;
   - it is positive in at most half the folds;
   - the 1h version falls from +28 bp to −0.3 bp with a one-bar (1-hour) entry delay.

3. **Much of the remaining "profit" is drift.** Jan–Jun 2026 trended down (triple-barrier labels short-skewed).
   Several positive rows trade mostly short and earn **less** than always-short at the same entry times:
   - `momentum_48` at 4h: 39 vs 36 bp;
   - `order_flow_24` at 4h: 48 vs 86 bp;
   - `ridge` at 4h: 46 vs 113 bp;
   - `lgbm_return` at 1h with delay: 16.6 vs 16.6 bp. It *is* always-short.

   Drift-harvesting fails the moment the regime turns, and the holdout (Jul–Sep) may well be a different regime.

4. **The sample is too small to decide.** At +28 bp per trade with ~250 bp per-trade volatility over 24h, t ≈ 1.4
   from 171 trades. t ≈ 3 would need ~4× more trades. 4h horizons are worse still: an h=24 (4-day) label gives only
   ~7 independent outcomes per symbol per test month. Large 4h ICs (e.g. 0.28 for `lgbm_return` at h=24) are noise
   for that reason; the new `ic_test_t` column makes this visible.

## Changes made in response

### Latency test priced from 1-minute bars (`--lag-minutes N`)

A one-bar delay at 1h means a 60-minute delay, which is harsher than reality (seconds). The collapse of the
mean-reversion edge under it has two very different explanations:

- **(a) fast but real reversion:** the edge survives a few minutes of delay and is executable;
- **(b) bid-ask-bounce / spike bias:** the rule buys after a down-close, but that closing print is often a sell
  hitting the bid, or one of the single-trade spikes (ETH has 359). Nobody can buy at that price, so the edge
  vanishes as soon as entry is priced a minute later.

`--lag-minutes 1/2/5` prices entry and exit at the last trade N minutes after the decision, from your 1-minute
bars. If the edge survives 1–2 minutes, (a) is plausible; if it is gone at 1 minute, it was (b).

### Multiple-testing control

- `q_bh` is the Benjamini–Hochberg false-discovery q-value across all strategy rows shown. **Treat a row as a
  candidate only if q_bh < 0.10** (and folds positive ≥ 3/4).
- `ic_test_t` is an IC t-stat using `rows / h` effective observations (labels overlap).

### Longer history: free Binance USDⓈ-M klines, Jul 2024 → Dec 2025

- **Why it is admissible:** it is after the Kronos pretraining cutoff (June 2024), so both baselines and Kronos are
  out-of-sample on it.
- **What changes:** with the Delta data, 21 monthly test folds (Oct 2024 → Jun 2026) instead of 4.
- **Same features:** Binance klines include taker-buy volume and trade counts, so every bar column and every feature
  means the same thing as for Delta trades.
- **Same costs:** costs stay Delta's (fees + Delta's measured spread), since we would trade on Delta.
- **Combined series:** BTC/ETH/SOL/DOGE are Binance before 2026-01-01 and Delta from then. The switch forces a new
  data segment, so no feature or label window mixes venues.
- **Holdout unchanged:** Delta Jul 1 → Sep 1 2026, still locked.

```bash
python scripts/fetch_binance.py --dest <folder>            # ~18 months x 4 symbols, SHA-256 verified
python scripts/ingest_binance.py --src <folder>            # builds BTC/ETH/SOL/DOGE combined bars
python scripts/build_dataset.py --symbols BTC ETH SOL DOGE --timeframes 1h 4h 15m
python scripts/run_baselines.py --symbols BTC ETH SOL DOGE --timeframes 1h 4h
python scripts/run_baselines.py --symbols BTC ETH SOL DOGE --timeframes 1h 4h --lag-minutes 2
```

`research.data_start` is now 2024-07-01 and `first_test_month` 2024-10. Delta-only datasets simply have no rows in
the early folds (they are skipped), so the Delta-only results reproduce unchanged.

## What this means for Kronos (Phase 5)

The comparison "baselines vs Kronos vs both" will be run on the combined 1h/4h series with the same evaluator,
latency test and FDR control. On the 4-month Delta sample alone, any Kronos result would be as inconclusive as the
baselines are.
