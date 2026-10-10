"""Settings loading with conservative safety defaults.

The trading mode and kill switch are resolved here so every entry point shares one rule:
orders are allowed only if the config says ``trading_enabled: true`` AND the environment
variable ``TRADING_ENABLED`` is exactly ``"true"``. Any other value, or a missing variable,
means no orders.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_SETTINGS_PATH = Path(__file__).with_name("settings.yaml")


class Mode(str, Enum):
    SIGNAL_ONLY = "SIGNAL_ONLY"
    PAPER = "PAPER"
    LIVE = "LIVE"


@dataclass(frozen=True)
class KronosSettings:
    model_id: str = "NeoQuasar/Kronos-small"
    tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-base"
    max_context: int = 512
    lookback: int = 400
    device: str = "auto"
    n_samples: int = 16
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 0
    clip: float = 5.0
    pretrain_cutoff: str = "2024-06-30"

    def __post_init__(self) -> None:
        if self.lookback > self.max_context:
            raise ValueError(f"kronos.lookback ({self.lookback}) exceeds max_context ({self.max_context})")
        if self.n_samples < 1:
            raise ValueError("kronos.n_samples must be >= 1")


@dataclass(frozen=True)
class CostSettings:
    taker_fee: float = 0.0005
    maker_fee: float = 0.0002
    gst_rate: float = 0.18
    # Funding history is not in the data yet: charge this to every position, long or short (conservative).
    funding_assumed_per_8h: float = 0.0001

    @property
    def taker_fee_effective(self) -> float:
        return self.taker_fee * (1.0 + self.gst_rate)

    @property
    def maker_fee_effective(self) -> float:
        return self.maker_fee * (1.0 + self.gst_rate)


@dataclass(frozen=True)
class ResearchSettings:
    data_start: str = "2026-01-01"
    first_test_month: str = "2026-03"
    holdout_start: str = "2026-07-01"  # final holdout: evaluated once, at the very end
    data_end: str = "2026-09-02"
    val_months: int = 1
    embargo: str = "12h"
    horizons: tuple[int, ...] = (1, 3, 6, 12, 24)
    triple_barrier_k: float = 1.0
    vol_window: int = 48

    def __post_init__(self) -> None:
        object.__setattr__(self, "horizons", tuple(int(h) for h in self.horizons))


@dataclass(frozen=True)
class CombinedSettings:
    """Research series per coin: Binance USD-M history before ``switch``, Delta trades from it."""

    switch: str = "2026-01-01"
    binance_start: str = "2024-07"
    symbols: dict = field(default_factory=dict)  # name -> {"binance": ..., "delta": ...}


@dataclass(frozen=True)
class Settings:
    mode: Mode = Mode.SIGNAL_ONLY
    trading_enabled: bool = False
    symbols: tuple[str, ...] = ("BTCUSD", "ETHUSD")
    timeframes: tuple[str, ...] = ("1h",)
    data_root: Path = Path("./data_store")
    kronos: KronosSettings = field(default_factory=KronosSettings)
    costs: CostSettings = field(default_factory=CostSettings)
    research: ResearchSettings = field(default_factory=ResearchSettings)
    combined: CombinedSettings = field(default_factory=CombinedSettings)

    def delta_symbol(self, symbol: str) -> str:
        """The Delta product a research symbol trades as (its costs come from that product's report)."""
        return self.combined.symbols.get(symbol, {}).get("delta", symbol)

    def orders_allowed(self, env: Mapping[str, str] | None = None) -> bool:
        """True only when the mode can place orders and both kill-switch layers are on."""
        env = os.environ if env is None else env
        if self.mode is Mode.SIGNAL_ONLY:
            return False
        return self.trading_enabled and env.get("TRADING_ENABLED", "") == "true"


def _parse_mode(value: Any) -> Mode:
    try:
        return Mode(str(value).upper())
    except ValueError:
        raise ValueError(f"Unknown mode {value!r}; expected one of {[m.value for m in Mode]}") from None


def load_settings(path: str | Path | None = None, env: Mapping[str, str] | None = None) -> Settings:
    """Load settings from YAML. ``MODE`` in the environment overrides the file's mode.

    ``trading_enabled`` in YAML must be a real boolean; strings such as "false" are rejected
    because ``bool("false")`` is True.
    """
    env = os.environ if env is None else env
    raw = yaml.safe_load(Path(path or DEFAULT_SETTINGS_PATH).read_text()) or {}

    trading_enabled = raw.get("trading_enabled", False)
    if not isinstance(trading_enabled, bool):
        raise ValueError("trading_enabled must be a YAML boolean (true/false)")

    mode = _parse_mode(env.get("MODE", raw.get("mode", Mode.SIGNAL_ONLY.value)))

    return Settings(
        mode=mode,
        trading_enabled=trading_enabled,
        symbols=tuple(raw.get("symbols", Settings.symbols)),
        timeframes=tuple(raw.get("timeframes", Settings.timeframes)),
        data_root=Path(raw.get("data", {}).get("root", "./data_store")),
        kronos=KronosSettings(**raw.get("kronos", {})),
        costs=CostSettings(**raw.get("costs", {})),
        research=ResearchSettings(**raw.get("research", {})),
        combined=CombinedSettings(**raw.get("combined", {})),
    )
