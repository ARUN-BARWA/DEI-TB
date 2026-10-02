# DEI-TB — Research Design (Phase 0)

Status: **approved; Phase 1 done** (see `docs/01_kronos_study.md`). Decisions from your answers are under
"Decisions" at the end.

Sources actually inspected for this document:

- Kronos source, `github.com/shiyu-coder/Kronos` @ HEAD (cloned, read `model/kronos.py`, `model/module.py`, `finetune/*`).
- Delta's official Python client, `github.com/delta-exchange/python-rest-client` (endpoints, auth signing, base URLs).
- Web search snippets about Delta India fees and WebSocket channels.

Not reachable from the build environment (network policy blocked them): `docs.delta.exchange`, `api.india.delta.exchange`,
`arxiv.org`, `huggingface.co`, `docs.tardis.dev`. Every Delta fact below marked **[verify]** must be confirmed
against the live docs/API in Phase 2 before code depends on it.

---

## 1. Kronos architecture assessment

### 1.1 What it is (from the code, not the marketing)

Two separately trained networks:

**(a) `KronosTokenizer`** — a vector-quantising autoencoder over candles.

- Input per bar: 6 channels `[open, high, low, close, volume, amount]`.
- `Linear(6→d_model)` → causal Transformer encoder (`is_causal=True` in `MultiHeadAttentionWithRoPE`) →
  `Linear(d_model→s1_bits+s2_bits)` → **Binary Spherical Quantizer (BSQ)**.
- Each bar becomes two discrete tokens: a coarse `s1` token (`s1_bits`) and a fine `s2` token (`s2_bits`),
  i.e. a hierarchical "price language" vocabulary.
- Decoder reconstructs the bar from `s1` only and from `s1+s2`. Loss = MSE(recon_pre) + MSE(recon_full) + BSQ
  entropy/commitment loss.

**(b) `Kronos`** — a decoder-only autoregressive Transformer over those tokens.

- `HierarchicalEmbedding(s1,s2)` + `TemporalEmbedding(minute, hour, weekday, day, month)` → N causal RoPE
  Transformer blocks → RMSNorm.
- `DualHead`: predicts `s1_{t+1}` from the hidden state; a `DependencyAwareLayer` (cross-attention conditioned on the
  `s1` embedding) then predicts `s2_{t+1}`.
- Training objective: **next-token cross-entropy** `(CE_s1 + CE_s2)/2`. Nothing in the objective refers to
  returns, direction, volatility, costs or P&L.

| Checkpoint | Tokenizer | Context | Params | Released |
|---|---|---|---|---|
| Kronos-mini | Tokenizer-2k | 2048 | 4.1M | yes |
| Kronos-small | Tokenizer-base | 512 | 24.7M | yes |
| Kronos-base | Tokenizer-base | 512 | 102.3M | yes |
| Kronos-large | Tokenizer-base | 512 | 499M | **no** |

Licence: MIT.

### 1.2 Input representation / preprocessing (`KronosPredictor.predict`)

- Requires `open, high, low, close`; `volume` and `amount` are optional (filled with 0, or
  `amount = volume × mean(OHLC)`).
- **Per-window z-score**: mean/std computed over the *context window only*, per channel, then clipped to ±5σ.
  At inference this is causal (no look-ahead). In the fine-tune `QlibDataset` the stats come from the lookback part
  only, which is also correct.
- Consequences:
  1. The model never sees absolute price level or absolute volatility. A 1% move in a calm window and a 1% move in a
     wild window look different, and the same z-pattern can mean very different dollar moves. **Scale features
     (realised vol, ATR in bps) have to be added back in the trading head.**
  2. The ±5σ clip truncates exactly the crash/squeeze bars that matter most for risk.
  3. Raw OHLC levels are z-scored, not returns, so a strongly trending window has a large std and the pattern is
     compressed.

### 1.3 Prediction mechanism (`auto_regressive_inference`)

- Encode the context into tokens, then for each of `pred_len` steps: run a full forward pass over the window
  (**no KV cache**), sample `s1` with temperature/top-p, sample `s2`, append, and roll the window once it exceeds
  `max_context`.
- `sample_count` paths are generated in parallel, then **`np.mean(preds, axis=1)` averages them before returning.**
  The public API therefore throws away the forecast distribution. For P(up), quantiles, expected MAE/MFE or
  uncertainty we need our own wrapper that returns every path (a small change in our code, no fork of Kronos).
- Cost ≈ `pred_len × samples × O(context²)` per decision timestamp. A walk-forward over one year of 5m bars
  (~105k timestamps) × 32 samples × 12 steps is ~40M forward passes. That is **feasible on a GPU only if we stride
  decisions** (e.g. one every k bars) or cache. On CPU it is not feasible at 1m/5m. This is a hard design constraint.

### 1.4 Representations available without modification

- `Kronos.decode_s1(...)` returns `(s1_logits, context)`, where `context` is `[B, T, d_model]`. Because every
  attention layer is causal, **the hidden state at position t is a legitimate time-t feature**. Option C (embedding
  extractor) is therefore clean.
- The `s1_logits` at the last position give a one-step-ahead categorical distribution over coarse tokens. Its
  **entropy** is a cheap, single-forward-pass uncertainty feature (no sampling needed).
- Tokenizer reconstruction error on the latest bars works as a novelty/regime-shift detector, since out-of-distribution
  candles reconstruct poorly.

### 1.5 Fine-tuning implementation

- Two stages: tokenizer (`train_tokenizer.py`) then predictor (`train_predictor.py`). Both are DDP/`torchrun`, use
  Comet logging, and are hard-wired to Qlib pickles. Each needs a thin adapter for our Parquet dataset.
- Windows are sampled uniformly at random from all symbols (fine for training, since splits are by time).
- **Leakage in the demo config:** train ends 2022-12-31 but validation starts 2022-09-01; validation ends 2024-06-30
  but test starts 2024-04-01. The splits overlap. We will not reuse their split logic.
- The demo backtest signal is `mean(pred_close) − last_close` ranked cross-sectionally (top-K A-shares). A
  single-instrument perpetual strategy has no cross-section, so this does not transfer directly.

### 1.6 Limitations relevant to us

1. **Pretraining contamination.** Kronos was pretrained on 12B+ K-line records from 45+ exchanges including crypto,
   with a corpus that **ends June 2024** (confirmed in Phase 1). Any period up to 2024-06-30 is potentially
   in-sample for Kronos, so **Kronos test windows start 2024-07-01 or later.**
2. Univariate OHLCVA only: no funding, OI, basis, order book or cross-asset input. Those must come from separate
   models.
3. The objective is token likelihood, not trading utility. A good CE/MSE does not imply sign accuracy where it matters
   (large moves), and says nothing about whether the edge survives costs.
4. Point forecasts are means of sampled paths, which tend to be smooth and mean-reverting. The "forecast close minus
   last close" signal is mostly a sampling-noise-plus-drift estimate unless validated.
5. Context of 512 bars covers 8.5h at 1m, 42h at 5m, 5.3d at 15m, 21d at 1h and 85d at 4h. Multi-timeframe input
   means running separate Kronos instances per timeframe.
6. Inference is expensive (no KV cache), which limits how often the 1m/5m backtests can call it.
7. Some repo comments are AI-generated (the authors say so). We trust the code, not the comments.

### 1.7 Verdict on options A–E

All five get tested (Phase 5–7, section 7), but my prior going in is:

| Option | Prior | Reason |
|---|---|---|
| A. Kronos as direct forecaster | weak | Unaligned objective, averaged paths, crypto 5m returns are near-unpredictable in mean |
| B. Fine-tune on Delta crypto | moderate | Adapts the token distribution; still not trading-aware; needs GPU and careful splits |
| C. Kronos as feature extractor | **most promising** | Causal embeddings + entropy + path-distribution stats go into a cost-aware head |
| D. Trading head on the frozen/fine-tuned backbone | promising, risky | Easy to overfit (a 25–100M-param backbone vs a few thousand independent trade outcomes) |
| E. Kronos as one alpha among several | **target architecture** | Diversification; Kronos earns a non-zero weight only if it adds incremental OOS value |

---

## 2. Delta Exchange India API capabilities

### 2.1 Endpoints and auth (confirmed from the official client)

| | Value |
|---|---|
| REST prod (India) | `https://api.india.delta.exchange` |
| REST testnet (India) | `https://cdn-ind.testnet.deltaex.org` (UI: testnet.delta.exchange) |
| WebSocket prod (India) | `wss://socket.india.delta.exchange` (from search results) **[verify]** |
| Auth headers | `api-key`, `timestamp` (unix seconds, string), `signature` |
| Signature | `hex(HMAC_SHA256(secret, METHOD + timestamp + path + query_string + body))` where `query_string = "?" + "&".join(k=quote_plus(v))` and `body = json.dumps(payload, separators=(',',':'))` |
| Timestamp tolerance | a few seconds **[verify]**, so the clock must be NTP-synced |

REST endpoints present in the official client:

| Purpose | Endpoint |
|---|---|
| Contract specs | `GET /v2/products`, `GET /v2/products/{id}` (tick_size, contract_value, margins, leverage limits **[verify fields]**) |
| Assets / indices | `GET /v2/assets`, `GET /v2/indices` |
| Tickers (incl. mark, OI, funding) | `GET /v2/tickers`, `GET /v2/tickers/{symbol}` |
| Historical candles | `GET /v2/history/candles?resolution=&symbol=&start=&end=` (unix seconds) |
| L2 snapshot | `GET /v2/l2orderbook/{symbol}` |
| Recent public trades | `GET /v2/trades/{symbol}` |
| Orders | `POST/PUT/DELETE /v2/orders`, `POST/PUT/DELETE /v2/orders/batch`, `DELETE /v2/orders/all`, `GET /v2/orders`, `/v2/orders/{id}`, `/v2/orders/client_order_id/{coid}`, `/v2/orders/history` |
| Bracket orders (exchange-side SL/TP) | `POST/PUT /v2/orders/bracket` |
| Order params | `limit_price, size (contracts), side, order_type (limit/market), time_in_force, post_only, reduce_only, client_order_id`, stop orders incl. trailing |
| Leverage | `GET/POST /v2/products/{id}/orders/leverage` |
| Positions | `GET /v2/positions`, `GET /v2/positions/margined`, `POST /v2/positions/change_margin`, `PUT /v2/positions/auto_topup`, `POST /v2/positions/close_all` |
| Margin mode | `PUT /v2/users/margin_mode` |
| Balances / ledger | `GET /v2/wallet/balances`, `GET /v2/wallet/transactions` (funding payments appear here) |
| Fills | `GET /v2/fills` |

To confirm in Phase 2 **[verify]**:

- Candle resolutions (expected `1m,3m,5m,15m,30m,1h,2h,4h,6h,1d,1w`) and the max candles per request (pagination).
- Special candle symbols for **mark price, funding and OI history** (expected `MARK:<sym>`, `FUNDING:<sym>`,
  `OI:<sym>`-style prefixes). If they don't exist, funding/OI history must come from our own recorder.
- Earliest available history per symbol. Delta India is a younger venue than Delta Global, so **"train 2021–2023"
  may not be possible on India data.**
- A dead-man switch / cancel-after endpoint. If none exists, our process must cancel on disconnect itself.
- Rate limits (weight-based limits are expected).

### 2.2 WebSocket channels (from search results, **[verify]** names)

Public: `v2/ticker`, `l2_orderbook` (snapshots), `l2_updates` (incremental), `all_trades`, `mark_price`,
`candlestick_{res}`, `funding_rate`. Private (auth): `orders`, `positions`, `user_trades`, `margins`.

### 2.3 Fees (from public sources, **[verify]** against our account tier)

- Perpetuals: taker 0.05%, maker 0.02% of notional, **plus 18% GST on the fee**, giving effective taker ≈ 5.9 bp
  and maker ≈ 2.36 bp.
- **A taker-in, taker-out round trip costs ≈ 11.8 bp plus spread/slippage.** For scale, BTC's typical absolute 5m
  return is on the order of 10–20 bp. At 1m/5m with taker execution, costs eat most plausible edges. This is the
  single most important economic fact for the project, and why timeframe must be chosen by net-of-cost results.
- Funding is exchanged periodically (interval **[verify]**) and must be modelled explicitly in the backtest and
  the edge calculation.

### 2.4 Data-availability implications

- OHLCV history: available via REST (depth TBD).
- **Historical L2 and trade-tick data is not available from the REST API** (only snapshots and recent trades). Any
  microstructure or Hawkes-on-trades alpha can only be researched on data **we record ourselves from the day the
  recorder starts.** Recommendation: start a WebSocket recorder in Phase 2, in parallel with the research.
- Third-party vendors (e.g. Tardis) list Delta, but it is unclear whether that covers India. **[verify]**

---

## 3. Proposed system architecture

```
                         ┌──────────────────────────── config/settings.yaml (+ env secrets) ─┐
                         │                                                                    │
 Delta REST ──► data/downloader ─┐                                                            │
 Delta WS  ──► data/recorder ────┼─► data/storage (Parquet, partitioned, immutable raw)       │
                                 │         │                                                  │
                                 │   data/validate (gaps, dupes, OHLC sanity, clock)          │
                                 │         │                                                  │
                                 │   features/*  (each feature: available_at, lookback, OOS)  │
                                 │         │                                                  │
            ┌───────────────┬────┴─────┬───┴──────────┬─────────────┐                         │
         Kronos          Momentum   MeanRev/Vol    Microstructure  Hawkes/Funding/Regime       │
     (frozen/finetuned)    alpha      alpha          alpha (later)   alphas                    │
            │ embeddings, path stats, entropy                                                  │
            └───────────────┴──────────┴──────────────┴─────────────┘                         │
                                     │  out-of-fold predictions only                          │
                              models/prediction_head  →  E[r_h], P(up), σ̂_h, E[MAE], E[MFE]   │
                                     │                                                        │
                              signals/generator: edge = E[r] − fees − slip − funding          │
                                     │            trade iff edge>k·σ̂ & conf>τ (τ from val)    │
                              risk/engine  (independent; can veto/resize; owns leverage)      │
                                     │                                                        │
                    ┌────────────────┼──────────────────┐                                     │
              SIGNAL_ONLY        PAPER (simulator)   LIVE (delta_india adapter)               │
                    └────────────────┴──────────────────┘                                     │
                                     │                                                        │
                       monitoring/  (decision log, alerts, kill switch TRADING_ENABLED) ◄─────┘
```

Design rules:

1. **One code path for backtest, paper and live.** The strategy produces `OrderIntent`s and only the
   `ExchangeInterface` implementation changes (`SimulatedExchange` in backtest/paper, `DeltaIndiaExchange` in live).
   This removes "it worked in backtest" divergence.
2. **Event-driven core** (bar-close events, plus trade/book events later). The same event loop drives historical
   replay and live.
3. **The risk engine is a separate module with veto power.** Models never emit leverage.
4. **Every decision is logged** as one append-only JSONL/Parquet record: inputs hash, model version, prediction,
   edge breakdown, risk checks passed/failed, order intent, fills.
5. **Mode gating:** `MODE` defaults to `SIGNAL_ONLY`. LIVE needs `MODE=LIVE`, `TRADING_ENABLED=true`, API keys
   present, and a CLI confirmation flag. Missing any one means no orders.

Project layout follows your section 22, with additions: `data/recorder.py`, `data/validate.py`,
`features/registry.py` (feature metadata + availability), `labels/` (targets), `research/` (experiment runner +
`experiments.csv`), and `common/` (types, clock, decimal rounding).

---

## 4. Recommended data schema

Storage: Parquet, partitioned `/{dataset}/{symbol}/{resolution}/year=YYYY/month=MM/`. Raw data is immutable; derived
data is versioned. All timestamps are **UTC, int64 microseconds**.

**Every row stores both the event time and the time we could have known it** (`ts_event`, `ts_available`).
Features join on `ts_available ≤ decision_time`, never on `ts_event`.

| Table | Key | Columns |
|---|---|---|
| `instruments` (SCD-2) | symbol, valid_from | product_id, contract_type, contract_value, tick_size, min_size, quoting/settling asset, initial/maint. margin rules, max leverage, fee rates, funding interval, valid_to |
| `candles` | symbol, resolution, ts_open | open, high, low, close, volume (contracts), turnover (quote), n_trades?, ts_close, **ts_available = ts_close + ingest_lag**, source |
| `mark_candles` / `index_candles` | same | OHLC of mark / index |
| `funding` | symbol, ts | funding_rate (realised), predicted_rate, interval, ts_available |
| `open_interest` | symbol, ts | oi_contracts, oi_notional |
| `trades` (recorded) | symbol, trade_id | ts_exchange, ts_recv, price, size, aggressor_side |
| `l2_updates` (recorded) | symbol, seq | ts_exchange, ts_recv, side, price, size, is_snapshot (+ periodic checksum/snapshot) |
| `l2_snapshots_1s` (derived) | symbol, ts | top-N levels, mid, spread, depth@bps bands |
| `features` | symbol, resolution, ts_decision | feature columns; companion `feature_registry.yaml` with lookback, `available_at_time_t`, warmup bars, source tables |
| `labels` | symbol, resolution, ts_decision, horizon | r_h, dir, triple-barrier label, t_touch, MAE, MFE, **ts_label_end** (used for purging) |
| `predictions` | model_id, fold, ts_decision | OOF outputs only |
| `decisions` / `orders` / `fills` / `positions` / `pnl` | — | full audit trail |

Data validation (Phase 2): duplicate/missing bars, `low ≤ open,close ≤ high`, zero-volume runs, outliers vs index
price, timezone/DST errors, candle-vs-trades reconciliation where both exist, and per-symbol coverage reports.

---

## 5. Recommended targets

For decision time t (bar close) and horizon h ∈ {1, 3, 6, 12, 24} bars on each timeframe:

| Target | Definition | Use |
|---|---|---|
| `r_h` | log(P_{t+h}/P_t), entry at t+latency, using **mid/mark** and the **trade price** (both) | regression, IC |
| `r_h_net` | r_h − round-trip cost(fee+spread+slip) − funding accrued over [t,t+h] | **primary economic target** |
| `dir_h` | 1[r_h > c] / 1[r_h < −c] / 0, where c = round-trip cost | classification with a cost-aware dead zone |
| `vol_h` | realised vol over (t, t+h] | sizing, uncertainty |
| `MAE_h`, `MFE_h` | max adverse / favourable excursion over (t, t+h] (using highs/lows) | stops, targets |
| Triple barrier | upper = +k_u·σ̂_t, lower = −k_d·σ̂_t, vertical = h; σ̂_t from **past** data only | LONG/SHORT/NO_TRADE labels |
| `hit_stop_first` | 1[lower barrier touched before upper] | P(stop before target) |
| Meta-label | 1[primary signal's trade was net-profitable] | trains a "take this trade?" filter |

Leakage rules: a label occupies [t, t+h]. During CV, **purge** training rows whose label window overlaps the test
fold and **embargo** ≥ h bars after it. Barrier widths use σ̂ estimated strictly before t. Within one bar,
intrabar ordering of high/low is unknown, so the triple-barrier label marks a conservative "both touched →
stop first" case and flags it.

---

## 6. Baseline models

Kronos has to beat these net of costs to earn a place. In rough order of complexity:

1. **Flat / buy-and-hold / always-short** — sanity anchors (perpetual funding makes these non-trivial).
2. **Random-entry with the same turnover** — gives the null distribution of Sharpe for the same cost drag.
3. **Time-series momentum** — sign of past k-bar return, vol-scaled (Moskowitz/Ooi/Pedersen 2012; Liu & Tsyvinski
   2021 for crypto).
4. **Short-horizon mean reversion** — z-score of the last return vs its EWMA vol.
5. **Funding / basis carry** — fade extreme funding.
6. **Linear models** (ridge / logistic) on the hand-built feature set.
7. **LightGBM / CatBoost** on the same features. This is the strongest "no-Kronos" baseline and the one that matters.
8. **Volatility models** (EWMA, HAR-RV, GARCH(1,1)) as baselines for σ̂, since Kronos's volatility forecasts must
   beat these too.

---

## 7. Kronos integration strategy

All variants produce **out-of-fold predictions** for the same timestamps, so they can be compared head-to-head.

| ID | Variant | Kronos output used |
|---|---|---|
| K0 | Zero-shot direct (option A) | E[r_h] = mean over N sampled paths of close_{t+h}/close_t − 1 |
| K1 | Zero-shot path distribution | from N paths: P(r_h>0), quantiles, path-σ, expected MAE/MFE, s1-logit entropy |
| K2 | Frozen embeddings (option C) | last hidden state (d_model) → PCA (fit on train fold only) → features |
| K3 | K1+K2 → linear / LightGBM / CatBoost / MLP head (options C/D) | |
| K4 | Fine-tuned on Delta train folds (option B), then K1–K3 | tokenizer + predictor fine-tune per walk-forward fold |
| K5 | Small trainable head on the frozen backbone, cost-aware loss (option D) | e.g. a quantile/Gaussian NLL head on r_h |
| K6 | Kronos alpha inside the multi-alpha ensemble (option E) | |

The central research question ("does Kronos add information?") is answered by the **nested comparison**, with the
same folds, costs and risk engine:

```
B  = baseline features → LightGBM
K  = Kronos features only → same head
BK = baseline + Kronos features → same head
E  = alpha ensemble without Kronos   vs   E+K = ensemble with Kronos
```

Kronos is kept **only if BK beats B (and E+K beats E)**:

- on net-of-cost Sharpe / expectancy across most walk-forward folds,
- with a paired test on per-period returns (Diebold–Mariano on losses; block-bootstrap CI of the Sharpe difference),
- **on data after the Kronos pretraining cutoff**,
- and robust to the perturbations in section 8.

Otherwise it is dropped, whatever its MSE says.

Compute plan: Kronos-small/mini first; 1h and 15m first (cheap); a decision stride for 5m/1m; all Kronos outputs cached
to Parquet keyed by (model_hash, symbol, resolution, ts) so experiments never recompute them. A GPU is effectively
required for K0/K1/K4 at scale; **what hardware do you have?**

---

## 8. Validation & backtesting methodology

**Splits.** Anchored or rolling walk-forward. For example (dates set after Phase 2 confirms history depth):

`train [T0, T1) → purge/embargo → validation [T1, T2) → test [T2, T3)`, rolled forward by one test period.

- Hyper-parameters, thresholds τ and min-edge k are chosen **on validation only**.
- The final hold-out (most recent ~3–6 months, *after the Kronos cutoff*) is touched **once**, at the end.
- No random shuffling anywhere. Purged k-fold with embargo (López de Prado, *AFML* ch. 7) only *within* the training
  span for inner model selection.

**Backtester (event-driven):**

- Decisions at bar close t. Order arrives at t + latency (configurable, randomised).
- Fills:
  - Market orders walk the recorded book where available; otherwise use a spread + slippage model ∝ σ and size/ADV.
  - Limit orders fill only if price trades *through* the limit (queue-position-conservative), with partial fills.
- Fees with GST, maker/taker by fill type. **Funding** is charged on positions held at each funding timestamp
  using the *realised* rate.
- Stops/targets are evaluated on intrabar high/low with a pessimistic ordering. **Liquidation** uses the
  exchange's maintenance-margin rule on the mark price.
- Contract rounding: size in integer contracts × contract_value, prices to tick_size.

**Metrics:** everything in your section 18. On top of that:

- the **Deflated Sharpe Ratio** (Bailey & López de Prado 2014) using the number of trials logged in `experiments.csv`,
- the **Probability of Backtest Overfitting** via CSCV (Bailey et al. 2017),
- an IC decay curve over horizons,
- performance split by regime/vol tercile and by fold.

**Robustness gates** (a strategy must pass all of these to move to paper):

| Gate | Pass condition |
|---|---|
| Positive folds | net Sharpe > 0 in ≥ 70% of WF folds |
| Fees ×1.5, slippage ×2 | Sharpe stays > 0 |
| Latency +1 bar | Sharpe stays > 0 |
| Parameter perturbation | ±20% on every parameter does not flip the sign |
| Transfer | works on ≥ 2 symbols (e.g. BTC, ETH) without re-tuning |
| Selection bias | DSR > 0.95 and PBO < 0.3 |

**Leakage tests** (automated unit tests):

- recompute every feature on truncated data and assert identical values at t (catches any future dependence);
- shift labels by +1 and confirm performance collapses;
- a shuffled-label null distribution.

---

## 9. Risk architecture

```
signal ──► PreTradeRisk ──► OrderIntent ──► ExecutionEngine ──► Exchange
              ▲   (can resize to 0)               │
              └──── PortfolioState / RiskState ◄──┘ (positions, PnL, margin from exchange — source of truth)
                               ▲
                        CircuitBreaker (independent loop, can cancel-all / flatten)
```

**Sizing** (risk, not model): `notional = equity × target_vol / σ̂_forecast`, capped by:

- `risk_per_trade`: `notional × stop_distance ≤ r% of equity`,
- `max_notional`, `max_position_contracts`, `max_portfolio_exposure`.

Confidence may only **shrink** size (e.g. a fractional-Kelly cap ≤ ¼). It can never raise leverage beyond what the
vol/stop rule gives. Vol-targeting has empirical support (Moreira & Muir 2017) and keeps risk roughly stable across
regimes.

**Leverage** is derived, not chosen: `leverage = notional / allocated_margin`, and must satisfy:

- `leverage ≤ max_leverage_config ≤ exchange_max`,
- `liquidation_distance ≥ m × stop_distance` (e.g. m=3, so the stop always fires well before liquidation),
- `liquidation_distance ≥ q × σ̂_h·√h` (e.g. q=6).

Isolated-margin liquidation price is computed from the exchange's own margin formula (Phase 12 verifies it
against `/v2/positions/margined`).

**Pre-trade checklist** (every order is rejected on any failure, and the reason is logged):

- required_margin ≤ available_margin × utilisation_cap
- estimated loss at stop (incl. fees + slippage) ≤ risk budget
- liquidation distance OK
- funding exposure over the expected holding period
- total/directional exposure
- orders-per-minute
- price within X bp of mark (fat-finger guard)
- instrument tradable
- data freshness (stale data → no trade)

**Circuit breakers** (independent of the strategy loop):

`TRADING_ENABLED=false` · `max_daily_loss` · `max_drawdown` · `max_consecutive_losses` · `max_orders_per_minute` ·
API error-rate · WS disconnect/staleness · position mismatch between local and exchange state.

On trip: cancel all orders → block new entries → apply the configured `emergency_policy` (`hold`, `reduce`, or
`flatten` via reduce-only market orders) → alert → require manual reset. Every live position also carries an
**exchange-side bracket stop** (`/v2/orders/bracket`) so protection survives our process crashing.

Unit tests cover sizing, leverage, liquidation distance, PnL (inverse vs linear contracts), fees+GST, funding,
contract/price rounding, every limit, and the kill switch.

---

## 10. Phased implementation plan (with exit gates)

| Phase | Deliverable | Exit gate |
|---|---|---|
| **1. Kronos study** | `kronos_model.py` wrapper (pinned Kronos commit as a submodule/vendored, unmodified) returning all sample paths, embeddings, entropy; CPU smoke test; cost benchmark (ms per call by model/context/samples); pretraining cutoff documented | Wrapper reproduces the upstream `predict()` mean exactly; latency table |
| **2. Delta data** | Settings/config, `ExchangeInterface` (read-only part), REST downloader with pagination/retries/rate limit, Parquet storage, validation report; **WS recorder for trades/L2/funding/mark started ASAP** | Coverage + quality report per symbol/timeframe; all **[verify]** items resolved |
| **3. Research dataset** | Feature registry with availability metadata, labels (section 5), purged WF splitter | Truncation-leakage test passes on every feature |
| **4. Baselines** | Section 6 models + a *simple* cost-aware evaluator | Baseline net-of-cost table per timeframe |
| **5. Kronos inference** | K0/K1/K2 cached outputs over the full history | IC / rank-IC / calibration vs baselines |
| **6. Kronos fine-tune** | K4 per WF fold, single-GPU adapter | Fine-tuned vs zero-shot on validation |
| **7. Prediction head** | K3/K5 heads, multi-output (E[r], P(up), σ̂, MAE/MFE) | OOF metrics logged in `experiments.csv` |
| **8. Event-driven backtester** | Section 8 engine + metrics + robustness suite | Engine unit tests; reproduces hand-computed trades |
| **9. Quant alphas** | Vol, Hawkes (univariate + cross-excitation, fitted by MLE on train folds only), entropy/Hurst/OU, regime, funding/OI | Each alpha: standalone IC + **incremental** value over B and BK |
| **10. Ensemble** | Stacked / constrained-weight ensemble fitted on OOF predictions | E vs E+K comparison: the core research answer |
| **11. Risk engine** | Section 9 | 100% of risk unit tests pass |
| **12. Delta execution adapter** | Auth, orders, brackets, leverage, positions; **testnet only** | Testnet round-trip incl. kill switch drill |
| **13. Signal-only live** | Live loop producing the section-13 signal records | ≥ 2–4 weeks; live signals match an offline replay of the same data |
| **14. Paper trading** | Simulator on live data | ≥ 4–8 weeks; paper P&L within backtest CI; no unexplained divergences |
| **15. Controlled live** | Explicit opt-in, minimum size, hard caps | Your sign-off, per stage |

Phases 1 and 2 are independent and can run together. The recorder in Phase 2 should start as early as possible,
because microstructure history only accrues from that day.

---

## Decisions (answers to the Phase 0 questions)

1. **Compute:** local GTX 1650 (4 GB). Use Kronos-small/mini, stride decisions at 1m/5m, and favour single-pass
   embeddings there.
2. **Symbols:** BTCUSD, ETHUSD, SOLUSD (the more volatile leg).
3. **Where it runs:** your machine, on **your local data files**. Phase 2 becomes *local data ingest + validation*
   instead of an API downloader. The Delta API adapter and live recorder move to Phases 12–13, so microstructure
   research depends on what your files contain.
4. **Other sources:** free/open only (e.g. Binance public archive), auxiliary research only.

## References (cited from knowledge; could not re-fetch from this environment)

- Shi et al., *Kronos: A Foundation Model for the Language of Financial Markets*, arXiv:2508.02739 (AAAI 2026).
- López de Prado, *Advances in Financial Machine Learning* (2018): triple-barrier, meta-labelling, purged/embargoed CV.
- Bailey & López de Prado, "The Deflated Sharpe Ratio" (2014); Bailey, Borwein, López de Prado, Zhu, "The Probability
  of Backtest Overfitting" (2017).
- Harvey, Liu & Zhu, "…and the Cross-Section of Expected Returns" (2016): multiple-testing thresholds.
- Moskowitz, Ooi & Pedersen, "Time Series Momentum" (2012); Liu & Tsyvinski, "Risks and Returns of Cryptocurrency" (2021).
- Moreira & Muir, "Volatility-Managed Portfolios" (2017).
- Bacry, Mastromatteo & Muzy, "Hawkes Processes in Finance" (2015).
- Cont, Kukanov & Stoikov, "The Price Impact of Order Book Events" (2014): order-flow imbalance.
- Corsi, "A Simple Approximate Long-Memory Model of Realized Volatility" (HAR-RV, 2009).
