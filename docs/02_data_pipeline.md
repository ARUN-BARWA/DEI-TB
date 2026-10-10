# Phase 2 — Delta tick data: ingest, validation, bars

## What your data is, and what it changes

| | |
|---|---|
| Source | Delta "futures-trades-monthly" exports, one zip per symbol-month |
| Schema | `product_symbol, price, size, timestamp, buyer_role` (tick trades) |
| Symbols | BTCUSD, ETHUSD, SOLUSD, DOGEUSD, PEPEUSD, MONUSD, PNUTUSD, ZROUSD |
| Period | 2026-01-01 → 2026-09-01 (the September file holds one day; ~8 months) |
| Not included | order book, funding, open interest, mark/index price, liquidations |

Consequences for the research plan:

1. **No Kronos contamination at all.** Every bar is after the June 2024 pretraining cutoff, so every Kronos result
   on this data is genuinely out-of-sample for the pretrained model.
2. **Trades with aggressor side unlock microstructure research**, even without an order book:
   - signed volume, order-flow imbalance, trade intensity, trade-size distribution;
   - Hawkes processes on buy and sell trade arrivals (your section 7);
   - an **effective-spread estimate from Delta's own prints**, which calibrates the cost model to the venue we will
     trade.
3. **Only 9 months of history.** This is the main statistical limitation:
   - Walk-forward folds test March–June for development; **2026-07-01 → 2026-09-01 is the holdout, touched once**
     (see `docs/03_research_dataset.md`). Data you collect after September adds forward test months.
   - At 1h that is only ~6,500 bars per symbol, so lower timeframes (5m/15m) carry most of the statistical power,
     but they also carry more cost.
   - **Recommendation:** add free Binance USDⓈ-M data (data.binance.vision: klines + aggTrades with aggressor flag,
     2020 onward) for training and robustness checks across regimes. Delta remains the venue for cost calibration and
     final validation. Kronos features on Binance data are only usable from 2024-07 onward (contamination).
4. **Funding is not in the data** but is a real cost of holding perpetuals. Until we fetch Delta's funding history
   (public API, later), the backtester will charge a conservative assumed funding rate against every position.
5. **Contract values are not in the data.** `size` is in contracts. Bars store `amount = Σ price × size` (notional ÷
   contract value). That is fine for modelling (Kronos z-scores every window), but position sizing will need each
   product's `contract_value` from `/v2/products` before Phase 8.

## Pipeline

```
monthly zip ──► chunked CSV reader ──► normalise ──► TradeValidator ──┬─► trades/{SYM}/{YYYY-MM}.parquet
                                         (ts us UTC, price,            └─► partial 1m bars ─► exact merge
                                          size, side ±1)                         │
                                                                     minutes/{SYM}/{YYYY-MM}.parquet
                                                                                 │
                                    complete grid + segments ◄───────────────────┘
                                                 │
                         bars/{1m,5m,15m,1h,4h}/{SYM}.parquet  +  reports/{SYM}.json, reports/summary.csv
```

- `side = +1` when `buyer_role == taker` (buyer aggressed), `-1` when `buyer_role == maker` (seller aggressed).
  The validator **tests this mapping on the data**: a buy-aggressor trade right after a sell-aggressor trade should
  print higher (ask vs bid). `aggressor_ok` in the summary reports the result.
- Timestamps are naive in the files and assumed UTC. `scripts/check_clock.py` verifies this against a free reference
  venue; IST-labelled data would show up as a ±330-minute shift.
- Bar index = bar **open** time. A bar is usable only from its close (`bars.available_at`).
- Minutes without trades are filled flat at the previous close with zero volume, and flagged `is_filled`.
- Trade gaps longer than `--max-gap` (default 6h) start a new `segment_id` and are never filled. Model windows must
  stay inside one segment. Higher-timeframe bars at segment edges are flagged `is_partial`.
- Nothing is silently cleaned. Malformed rows are dropped **and counted**; suspicious-but-valid trades (isolated price
  spikes, exact duplicates, out-of-order rows) are kept and counted, so we decide on evidence.
- Results do not depend on chunk size; tested with chunks of 7 rows up to the whole file, including shuffled files.

Measured here: 5M trades ingest in ~15s with ~1 GB peak RAM at the default chunk size
(`--chunksize 500000` lowers it); about 28 MB of Parquet per 5M trades.

## Data-quality report columns (`reports/summary.csv`)

| Column | Meaning |
|---|---|
| `trades`, `dropped` | valid trades kept; malformed rows dropped |
| `trades_per_day_med` | liquidity; small caps with few trades per day are not tradeable at low timeframes |
| `pct_min_no_trade` | share of 1m bars with no trade |
| `segments` | 1 + number of trade gaps longer than `max_gap` (missing months, outages) |
| `out_of_order`, `dup_rows` | file-order problems |
| `spikes` | single trades ≥1% away from both neighbours that revert immediately |
| `outliers_1m` | 1m returns > 10× trailing robust σ |
| `aggressor_ok` | `buyer_role` → side mapping confirmed by prices |
| `spread_proxy_bp` | median price jump from a sell-aggressor to the next buy-aggressor trade within 1s, i.e. effective spread |
| `taker_round_trip_bp` | 2 × taker fee incl. GST + spread proxy: the minimum move a taker trade must beat |
| `med_abs_1m_ret_bp`, `ann_vol` | volatility scale |

## Note on your probe output (1m BTCUSD)

- `device=cpu`: PyTorch on that machine is not using the GTX 1650. Check with
  `python -c "import torch; print(torch.cuda.is_available())"`. If it prints `False`, reinstall with
  `pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121`.
- At h=1 the whole 5–95% forecast band (−5.1 … +9.1 bp) sits inside the ~11.8 bp taker round-trip fee, before spread.
  This is a concrete illustration of why 1m taker trading is unlikely to be viable.
- At h=12 the forecast σ (5.5 bp) is barely above h=1 (4.7 bp). A random walk would scale by √12 to ~16 bp. **If this
  holds across many forecasts, Kronos's sampled distribution is badly under-dispersed**, and its P(up) and quantiles
  must be calibrated before use. Phase 5 measures this (coverage of the q05–q95 band, PIT histograms) over thousands
  of decision points. One forecast proves nothing either way.
