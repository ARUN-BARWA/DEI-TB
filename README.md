# DEI-TB

Research-grade crypto trading system for Delta Exchange India, built around the
[Kronos](https://github.com/shiyu-coder/Kronos) financial foundation model.

The pipeline is **research → backtest → paper trade → controlled live**. The default mode is `SIGNAL_ONLY`, and nothing
in this repository can place an order yet.

- Design: [`docs/00_research_design.md`](docs/00_research_design.md)
- Phase 1 (Kronos study): [`docs/01_kronos_study.md`](docs/01_kronos_study.md)

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

## Safety

- `trading_bot/config/settings.yaml` ships with `mode: SIGNAL_ONLY` and `trading_enabled: false`.
- Orders are only ever allowed when the mode is `PAPER`/`LIVE`, **and** `trading_enabled: true`, **and** the
  environment variable `TRADING_ENABLED` is exactly `true`. Setting `TRADING_ENABLED=false` stops everything.
- API credentials are read only from the `DELTA_API_KEY` / `DELTA_API_SECRET` environment variables, never from files
  in this repo.
