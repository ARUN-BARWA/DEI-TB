# Phase 5 — Kronos vs baselines on the combined history

## What the combined-history baselines showed (21 test months, Oct 2024 → Jun 2026)

1. **The Delta-only mean-reversion "edge" did not generalise.** Over 21 folds, `mean_reversion_24` at 1h has
   IC 0.0006 and −27.6 bp per trade. The +28 bp seen in Mar–Jun 2026 was sample-specific, which is why the history
   was extended.
2. **There is real predictive information, but it is smaller than taker costs.**

   | model | IC (t) | gross bp | net bp | p sign-flip | q_bh | folds + |
   |---|---|---|---|---|---|---|
   | `lgbm_tb` 1h, h=3 | 0.028 (4.1) | 18.4 | +5.2 | 0.0005 | 0.008 | 9/21 |
   | same, priced 2 min late | | 21.7 | +8.5 | 0.0005 | 0.008 | 9/21 |
   | `lgbm_return` 4h, h=3 | 0.054 (4.0) | 12.4 | −2.0 | 0.04 | 0.09 | 6/21 |
   | `lgbm_tb` 4h, h=1 | 0.036 (4.7) | 7.2 | −6.2 | 0.05 | 0.09 | 3/21 |

   Direction skill is significant: the sign-flip q-value is 0.008 and the IC t-stats are about 4. But the gross
   edge per trade is the same order as the ~13 bp round trip. This matches the Phase 4 break-even analysis: IC ≈ 0.03
   at 3–6h needs ≈ 0.08–0.11 under taker costs.
3. **The 2-minute latency test changes nothing material.** The 1h/4h results are not bid-ask-bounce artifacts.
4. **4h shows a reversal effect at 2–8 day lookbacks.** `momentum_48` has IC −0.107 (t −3.7), −0.114 on validation.
   The fixed-direction momentum rule loses because it bets on trend. Flipping its sign after seeing test results
   would be snooping. Instead, the new `*_sf` rules learn their direction on each fold's **training** rows.
5. **No strategy meets the bar** (q < 0.10 and a clear majority of positive folds). The best has 11/21 positive folds.

## What changed in the evaluator

- **`--extend` (hold through a persisting signal).** At a scheduled exit, if the signal (known at that moment) still
  points the same way above threshold, the position is held another h bars instead of being closed and re-opened.
  That saves a full round trip. It applies identically to every model and uses no future information. When the edge
  per trade is about the size of the cost, this is a first-order effect.
- **Sign-learned rules** `trend_12_sf`, `trend_48_sf`, `zclose_24_sf`, `flow_24_sf`, added to `run_baselines.py`.

## Phase 5 design

### Kronos features (`scripts/kronos_features.py`, cached and resumable)

For every decision bar, the trailing window inside the bar's own data segment (default 256 bars at 1h ≈ 11 days,
128 at 4h ≈ 21 days) gives:

- **embedding mode:** the causal last-position hidden state (`kr_emb_*`), next-token entropy and tokenizer
  reconstruction error;
- **paths mode (`--paths`):** additionally 16 sampled 12-step forecast paths, summarised per horizon: mean, std,
  quantiles, P(up), MFE/MAE, path vol.

Causality is tested: rewriting bars after t leaves the features at t bit-identical. No window crosses a data hole or
the Binance→Delta switch.

### Comparison (`scripts/run_kronos_comparison.py`)

All arms run on **exactly the same rows** (those with Kronos features), with the same folds, costs, LightGBM settings,
targets and validation-chosen thresholds:

| arm | inputs |
|---|---|
| `B_*` | the 47 registry features: what the baselines already know |
| `K_*` | Kronos only: scalars + 16-component PCA of the embedding, **PCA fitted on each fold's training rows** |
| `BK_*` | both |
| `K_ridge` | Kronos features → linear head |
| `kronos_direct` | Kronos's own path-mean forecast, no head (needs `--paths`) |

The **incremental table** compares K and BK against B fold by fold (paired t-test on test IC, BH q-values across all
comparisons) and also shows net bp per trade.

### Forecast quality (`scripts/kronos_calibration.py`)

This asks whether Kronos is a good forecaster, independent of trading:

- coverage of its 50% and 90% forecast bands;
- dispersion ratio (realised / predicted spread; > 1 means over-confident, as the Phase 1 probe hinted);
- IC of the path mean and of P(up);
- direction hit rate;
- Brier score against the base rate.

## Decision rules (fixed before seeing any Kronos result)

- **Kronos adds information** if, for the arm and horizon chosen by validation, BK − B test IC is positive with fold-
  paired q_bh < 0.10, and BK is better in ≥ 60% of folds.
- **Kronos adds trading value** if, in addition, a BK (or K) strategy row has all of:
  - q_bh < 0.10, t_net > 2 and ≥ 13/21 positive folds;
  - higher net bp per trade than the matching B row;
  - the same conclusion with `--lag-minutes 2`.
- If neither holds, **Kronos is not used in the strategy**, per the project rule: forecasting skill alone is not a
  reason to include it. It may still be revisited as a regime/volatility input.

## Run plan on the GTX 1650 (times from your benchmark)

```bash
# 1) embeddings with Kronos-small: ~20 min at 1h + ~5 min at 4h
python scripts/kronos_features.py --model NeoQuasar/Kronos-small --timeframes 4h 1h
python scripts/run_kronos_comparison.py --kronos-tag Kronos-small_emb
python scripts/run_kronos_comparison.py --kronos-tag Kronos-small_emb --extend
python scripts/run_baselines.py --symbols BTC ETH SOL DOGE --timeframes 1h 4h --extend   # baselines, same rule

# 2) sampled paths with Kronos-mini: ~1 h at 4h, ~7 h at 1h (run overnight; interrupt/resume safely)
python scripts/kronos_features.py --model NeoQuasar/Kronos-mini --timeframes 4h 1h --paths
python scripts/kronos_calibration.py --kronos-tag Kronos-mini_paths
python scripts/run_kronos_comparison.py --kronos-tag Kronos-mini_paths
```

Kronos-small paths at 1h would take ~36 h on this GPU, so they are left out unless the mini results justify it.
