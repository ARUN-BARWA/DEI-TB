# Phase 4 — Cost-aware baselines

## What Phase 3 told us (your run)

### The four data holes are missing export days, not market closures

| Missing (UTC) | Weekday |
|---|---|
| 2026-03-20 | Friday |
| 2026-05-11 | Monday |
| 2026-06-24 | Wednesday |
| 2026-07-27 → 07-31 | Monday → Friday (inside the holdout) |

Delta perpetuals trade 24/7, and each hole is exactly one or more whole UTC days. These are days missing from the
export. If you can re-download those days, the pipeline will fill them in incrementally. Until then, no feature,
label or evaluated trade spans a hole.

### Kronos-small, clean re-run

No OOM and no cliffs: time now scales linearly with samples, steps and context. That confirms the earlier
slowdowns were memory spill, which the VRAM cap fixed.

| Kronos-small, context 512 | ms / decision | hours per year at 15m | at 1h | at 4h |
|---|---|---|---|---|
| embedding (1 pass) | 24 | 0.24 | 0.06 | 0.01 |
| 16 paths × 12 steps | 2,886 | 28 | 7 | 1.8 |

### The cost hurdle sets a minimum skill per horizon

From the median |move| / cost ratios in `cost_hurdle.csv`, the break-even information coefficient for taker
trading. This is an approximation: a linear-Gaussian signal, trading the strongest ~13% of signals (mean |z| ≈ 1.9),
so E[r | trade] ≈ 1.9 · IC · σ_h, giving IC_be ≈ 0.355 / (median|move| / cost):

| horizon (bars) | 1 | 3 | 6 | 12 | 24 |
|---|---|---|---|---|---|
| BTC 5m | 0.71 | 0.40 | 0.28 | 0.20 | 0.14 |
| BTC 15m | 0.40 | 0.23 | 0.16 | 0.11 | 0.08 |
| BTC 1h | 0.20 | 0.11 | 0.08 | 0.05 | **0.03** |
| BTC 4h | 0.09 | 0.05 | **0.03** | **0.02** | **0.015** |
| ETH 1h | 0.17 | 0.10 | 0.06 | 0.04 | **0.03** |
| SOL 1h | 0.15 | 0.09 | 0.05 | **0.035** | **0.02** |
| DOGE 1h | 0.16 | 0.09 | 0.06 | 0.04 | **0.03** |

Realistic intraday ICs in crypto are roughly 0.02–0.06. So with taker fills, **holding periods of ~6 hours to 2 days
(1h bars with h = 6–24, 4h bars) are where an edge can plausibly survive costs.** 5m bars would need implausible
skill, and 1m bars are worse (median move = 0.23 × cost). The approximation ignores funding, which matters more at
long holds, and nonlinear edges concentrated in tails. Maker execution would cut the hurdle ~2.5×, but needs fill
modelling that has to wait for the Phase 8 backtester.

The triple-barrier mix is short-skewed in January–June (ETH 4h h=24: 50% short vs 18% long). A model can look
skilful just by learning "short". Every result below is therefore shown next to always-long and always-short traded
at the same moments.

## Protocol (`trading_bot/research/evaluate.py`, `walkforward.py`)

The same evaluator will judge Kronos and the ensembles, so comparisons are like-for-like.

1. **Models.** Each model scores every bar: sign = direction, magnitude = conviction. Learned models train on pooled
   rows of all four symbols (scale-free features), with a fresh model per fold.
2. **Threshold on validation only.** The model trades when |score| ≥ θ. θ is picked from a 7-point quantile grid to
   maximise the validation t-stat of net trade returns. If nothing is positive on validation, the model **abstains**
   on test.
3. **Trading the test month.**
   - Enter at the decision bar's close and exit after h bars. `--delay 1` re-runs with a one-bar entry delay.
   - Trades never overlap and never span a data hole.
   - Every trade pays the full taker round trip (11.8 bp fees incl. GST + the symbol's spread proxy) plus an assumed
     1 bp / 8h funding, whatever its side. That is conservative until funding history is available.
4. **Drift benchmarks.** `always_long` and `always_short` trade every opportunity as reference points.
5. **Reported metrics.**
   - IC and rank IC on all test rows.
   - Trades, hit rate, mean gross/net bp, t-stat, annualised Sharpe of daily P&L, max drawdown, long share.
   - **Sign-flip p-value:** random directions at the same entry times, testing direction skill.
   - What always-long and always-short would have earned at the same moments.
   - Folds positive out of 4.
6. **Horizon choice.** The headline row per model × timeframe uses the horizon with the best **validation** t-stat,
   never the best test result. The full grid is in `grid.csv`.

Every configuration is appended to `data_store/experiments/experiments.csv`, including the number of configurations
tried. That count feeds the deflated-Sharpe correction in Phase 8.

## Models

| name | type | score |
|---|---|---|
| `always_long`, `always_short` | reference | +1 / −1 |
| `momentum_12`, `momentum_48` | rule | ret_k / (rv_48 √k) |
| `mean_reversion_24` | rule | −zclose_24 |
| `order_flow_24` | rule | taker volume imbalance over 24 bars |
| `ridge` | linear | vol-normalised forward return, all 47 features |
| `logistic_tb` | linear | P(long) − P(short), triple-barrier classes |
| `lgbm_return` | gradient boosting | vol-normalised return, Huber loss, early stopping on validation |
| `lgbm_tb` | gradient boosting | P(long) − P(short), triple-barrier classes |

Hyper-parameters are fixed and conservative (`LGBM_PARAMS`). Nothing is tuned on test.

## Pipeline checks (tests)

- **Planted signal:** synthetic data where a feature drives next-bar returns. Ridge and LightGBM find it with
  IC ≈ 0.30 (theoretical 0.33), +6–7 bp net per trade after 12 bp costs, positive in 4/4 folds, p = 0.0005.
- **Pure noise:** strategy models abstain or show no significance (LightGBM lost ≈ the cost on 71 trades,
  p = 0.48), and the drift benchmarks lose exactly the cost.
- Unit tests cover non-overlap, the entry delay, holes, the exit cap, costs and funding, the sign-flip test, drift
  benchmarks and the experiment log.

Runtime at your scale: about 4 minutes for 5m (the heaviest), less for the other timeframes.
