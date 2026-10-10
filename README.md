# DEI-TB

Research-grade crypto trading system for Delta Exchange India, built around the
[Kronos](https://github.com/shiyu-coder/Kronos) financial foundation model.

The pipeline is **research → backtest → paper trade → controlled live**. The default mode is `SIGNAL_ONLY`, and nothing
in this repository can place an order yet.

- Design: [`docs/00_research_design.md`](docs/00_research_design.md)
- Phase 1 (Kronos study): [`docs/01_kronos_study.md`](docs/01_kronos_study.md)
- Phase 2 (tick data pipeline): [`docs/02_data_pipeline.md`](docs/02_data_pipeline.md)
- Phase 3 (features, labels, splits): [`docs/03_research_dataset.md`](docs/03_research_dataset.md), [`docs/features.md`](docs/features.md)
- Phase 4 (cost-aware baselines): [`docs/04_baselines.md`](docs/04_baselines.md), results and next steps: [`docs/04b_baseline_findings.md`](docs/04b_baseline_findings.md)
- Phase 5 (Kronos vs baselines): [`docs/05_kronos_comparison.md`](docs/05_kronos_comparison.md)

## Setup

```bash
git clone --recurse-submodules https://github.com/ARUN-BARWA/DEI-TB.git
cd DEI-TB
# (if you cloned without --recurse-submodules)
git submodule update --init

python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
# PyTorch with CUDA for a GTX 1650 (pick the CUDA build matching your driver):
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pytest -q
```

Kronos weights download from Hugging Face on first use (`NeoQuasar/Kronos-small`, `NeoQuasar/Kronos-mini`, ...).

## Phase 1 scripts

```bash
# Inference cost on your GPU -> reports/kronos_benchmark.csv
python scripts/kronos_benchmark.py

# One forecast on your own data (CSV or Parquet with timestamp/open/high/low/close[/volume])
python scripts/kronos_probe.py --file path/to/BTCUSD_1h.csv
python scripts/kronos_probe.py --file path/to/BTCUSD_1h.csv --at "2025-03-01 12:00" --n-samples 32
```

## Phase 2: ingest your Delta trade archives

```bash
# Folder with futures-trades-monthly-<SYMBOL>-<YYYY-MM>.zip files (searched recursively; never modified)
python scripts/ingest_trades.py --src path/to/delta_trades
# -> data_store/trades, data_store/bars/{1m,5m,15m,1h,4h}, data_store/reports/summary.csv + <SYMBOL>.json

# Verify the timestamps are UTC against a free reference file, e.g. Binance
# https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2026-09.zip
python scripts/check_clock.py --symbol BTCUSD --ref BTCUSDT-1m-2026-09.zip
```

Re-running `ingest_trades.py` only processes new or changed files (`--force` re-does everything).

## Phase 3: research datasets

```bash
python scripts/build_dataset.py            # symbols x timeframes from settings.yaml
# -> data_store/datasets/<tf>/<SYMBOL>.parquet, data_store/reports/cost_hurdle.csv, folds.csv
python scripts/gen_feature_docs.py         # after adding/changing a feature (a test checks docs/features.md)
```

## Phase 4: baselines

```bash
pip install -r requirements.txt            # adds scikit-learn, lightgbm
python scripts/run_baselines.py            # 5m, 15m, 1h, 4h; development folds only
python scripts/run_baselines.py --delay 1  # robustness: enter one bar late
# -> data_store/reports/baselines/<run_id>/{summary,grid,folds}.csv + trades.parquet
#    and data_store/experiments/experiments.csv
```

## Longer history (free Binance USD-M klines, Jul 2024 -> Dec 2025)

```bash
python scripts/fetch_binance.py --dest path/to/binance     # SHA-256 verified downloads
python scripts/ingest_binance.py --src path/to/binance     # combined BTC/ETH/SOL/DOGE series (Binance, then Delta)
python scripts/build_dataset.py --symbols BTC ETH SOL DOGE --timeframes 1h 4h 15m
python scripts/run_baselines.py --symbols BTC ETH SOL DOGE --timeframes 1h 4h [--lag-minutes 2]
```

## Phase 5: Kronos vs baselines

```bash
python scripts/kronos_features.py --model NeoQuasar/Kronos-small --timeframes 4h 1h           # embeddings (~25 min)
python scripts/kronos_features.py --model NeoQuasar/Kronos-mini --timeframes 4h 1h --paths    # + sampled paths (hours)
python scripts/kronos_calibration.py --kronos-tag Kronos-mini_paths                          # forecast quality
python scripts/run_kronos_comparison.py --kronos-tag Kronos-small_emb [--extend] [--lag-minutes 2]
```

## Safety

- `trading_bot/config/settings.yaml` ships with `mode: SIGNAL_ONLY` and `trading_enabled: false`.
- Orders are only ever allowed when the mode is `PAPER`/`LIVE`, **and** `trading_enabled: true`, **and** the
  environment variable `TRADING_ENABLED` is exactly `true`. Setting `TRADING_ENABLED=false` stops everything.
- API credentials are read only from the `DELTA_API_KEY` / `DELTA_API_SECRET` environment variables, never from files
  in this repo.
