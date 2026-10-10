"""Walk-forward splits with purging, embargo and a locked final holdout.

Every fold is forward-only: train < validation < test in time, train always starting at
``data_start`` (expanding window). For a fold:

    train : decision in [train_start, val_start - embargo)  and label_end <= val_start - embargo
    val   : decision in [val_start,  test_start - embargo)  and label_end <= test_start - embargo
    test  : decision in [test_start, test_end)               and label_end <= test_end

``label_end`` comes from the labels (t + (h + 1) tf), so purging removes every sample whose
outcome overlaps the next period. The holdout fold refuses to produce masks unless explicitly
unlocked, and each unlock is appended to an audit log.
"""

from __future__ import annotations

import getpass
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


class HoldoutLocked(RuntimeError):
    pass


@dataclass(frozen=True)
class Fold:
    name: str
    train_start: pd.Timestamp
    val_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    is_holdout: bool = False

    def __post_init__(self) -> None:
        if not (self.train_start < self.val_start < self.test_start < self.test_end):
            raise ValueError(f"fold {self.name}: periods must be strictly ordered")


def _ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def monthly_folds(data_start, first_test_month, holdout_start, data_end, val_months: int = 1) -> list[Fold]:
    """One development fold per test month from ``first_test_month`` up to the holdout, then the holdout.

    Each fold validates on the ``val_months`` before its test month and trains on everything
    earlier. The holdout tests on [holdout_start, data_end).
    """
    data_start, holdout_start, data_end = _ts(data_start), _ts(holdout_start), _ts(data_end)
    folds = []
    test_start = _ts(f"{first_test_month}-01")
    while test_start < holdout_start:
        test_end = min(test_start + pd.offsets.MonthBegin(1), holdout_start)
        val_start = test_start - pd.offsets.MonthBegin(val_months)
        if val_start <= data_start:
            raise ValueError(f"no training data before validation for test month {test_start:%Y-%m}")
        folds.append(Fold(f"dev_{test_start:%Y-%m}", data_start, val_start, test_start, test_end))
        test_start = test_end
    folds.append(Fold("holdout", data_start, holdout_start - pd.offsets.MonthBegin(val_months), holdout_start, data_end, is_holdout=True))
    return folds


def fold_masks(
    decision_time: pd.Series | pd.DatetimeIndex,
    label_end: pd.Series | pd.DatetimeIndex,
    fold: Fold,
    embargo: str | pd.Timedelta = "12h",
    allow_holdout: bool = False,
    audit_log: str | Path | None = None,
    reason: str = "",
) -> dict[str, np.ndarray]:
    if fold.is_holdout:
        if not allow_holdout:
            raise HoldoutLocked("the holdout is locked; pass allow_holdout=True only for the final, one-time evaluation")
        if audit_log is not None:
            path = Path(audit_log)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as f:
                f.write(f"{pd.Timestamp.now(tz='UTC').isoformat()}\t{getpass.getuser()}\t{fold.name}\t{reason}\n")
    dt = pd.DatetimeIndex(decision_time)
    le = pd.DatetimeIndex(label_end)
    emb = pd.Timedelta(embargo)
    has_label = ~le.isna()
    train = (dt >= fold.train_start) & (dt < fold.val_start - emb) & (le <= fold.val_start - emb)
    val = (dt >= fold.val_start) & (dt < fold.test_start - emb) & (le <= fold.test_start - emb)
    test = (dt >= fold.test_start) & (dt < fold.test_end) & (le <= fold.test_end)
    return {k: np.asarray(v & has_label) for k, v in (("train", train), ("val", val), ("test", test))}


def development_mask(label_end: pd.Series | pd.DatetimeIndex, holdout_start) -> np.ndarray:
    """Rows whose outcome is fully known before the holdout starts (safe for any descriptive statistic)."""
    le = pd.DatetimeIndex(label_end)
    return np.asarray(~le.isna() & (le <= _ts(holdout_start)))
