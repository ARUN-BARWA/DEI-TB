"""Baseline models that Kronos must beat. Common interface:

    model.fit(train, val, h)   # train rows only; val only for early stopping
    model.predict(frame) -> np.ndarray of scores (sign = direction, magnitude = conviction)

Rule baselines need no fitting. Learned baselines use every registered feature, a volatility-
normalised target y_ret_h / (rv_48 * sqrt(h)) clipped to +/-5 (heavy tails, heteroskedasticity),
and statistics (imputation, scaling) estimated on the training rows only.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd

SEED = 0


def vol_normalised_target(df: pd.DataFrame, h: int, clip: float = 5.0) -> pd.Series:
    y = df[f"y_ret_{h}"] / (df["rv_48"] * np.sqrt(h))
    return y.clip(-clip, clip)


class Model:
    name = "model"
    learned = False
    reference = False  # reference models (drift benchmarks) always trade; no validation threshold

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, h: int) -> "Model":
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def params(self) -> dict:
        return {}


class Constant(Model):
    reference = True

    def __init__(self, name: str, value: float) -> None:
        self.name, self.value = name, value

    def predict(self, frame):
        return np.full(len(frame), self.value)

    def params(self):
        return {"value": self.value}


class Rule(Model):
    def __init__(self, name: str, fn: Callable[[pd.DataFrame], pd.Series], description: str) -> None:
        self.name, self.fn, self.description = name, fn, description

    def predict(self, frame):
        return np.asarray(self.fn(frame), dtype=float)

    def params(self):
        return {"rule": self.description}


def rule_baselines() -> list[Model]:
    return [
        Constant("always_long", 1.0),
        Constant("always_short", -1.0),
        Rule("momentum_12", lambda d: d["ret_12"] / (d["rv_48"] * np.sqrt(12)), "ret_12 / (rv_48 sqrt 12)"),
        Rule("momentum_48", lambda d: d["ret_48"] / (d["rv_48"] * np.sqrt(48)), "ret_48 / (rv_48 sqrt 48)"),
        Rule("mean_reversion_24", lambda d: -d["zclose_24"], "-zclose_24"),
        Rule("order_flow_24", lambda d: d["flow_imb_24"], "flow_imb_24"),
    ]


class _Tabular(Model):
    learned = True

    def __init__(self, features: list[str]) -> None:
        self.features = list(features)

    def _xy(self, df: pd.DataFrame, h: int, target: str):
        if target == "reg":
            y = vol_normalised_target(df, h)
        else:  # triple barrier -> classes 0 (short), 1 (none), 2 (long)
            y = df[f"y_tb_{h}"] + 1
        ok = y.notna() & df["rv_48"].notna()
        return df.loc[ok, self.features], y[ok]


class _Standardised(_Tabular):
    def _fit_scaler(self, x: pd.DataFrame) -> None:
        self.median_ = x.median()
        z = x.fillna(self.median_)
        self.mean_, self.std_ = z.mean(), z.std().replace(0, 1.0)

    def _transform(self, x: pd.DataFrame) -> np.ndarray:
        z = (x[self.features].fillna(self.median_) - self.mean_) / self.std_
        return z.clip(-5, 5).fillna(0.0).to_numpy()


class RidgeReturn(_Standardised):
    """Ridge regression on the vol-normalised forward return."""

    def __init__(self, features, alpha: float = 100.0) -> None:
        super().__init__(features)
        self.name, self.alpha = "ridge", alpha

    def fit(self, train, val, h):
        from sklearn.linear_model import Ridge

        x, y = self._xy(train, h, "reg")
        self._fit_scaler(x)
        self.model_ = Ridge(alpha=self.alpha).fit(self._transform(x), y.to_numpy())
        return self

    def predict(self, frame):
        return self.model_.predict(self._transform(frame))

    def params(self):
        return {"alpha": self.alpha}


class LogisticBarrier(_Standardised):
    """Multinomial logistic regression on triple-barrier labels; score = P(long) - P(short)."""

    def __init__(self, features, c: float = 0.05) -> None:
        super().__init__(features)
        self.name, self.c = "logistic_tb", c

    def fit(self, train, val, h):
        from sklearn.linear_model import LogisticRegression

        x, y = self._xy(train, h, "tb")
        self._fit_scaler(x)
        self.model_ = LogisticRegression(C=self.c, max_iter=500).fit(self._transform(x), y.astype(int).to_numpy())
        return self

    def predict(self, frame):
        p = self.model_.predict_proba(self._transform(frame))
        cls = list(self.model_.classes_)
        long_ = p[:, cls.index(2)] if 2 in cls else 0.0
        short = p[:, cls.index(0)] if 0 in cls else 0.0
        return long_ - short

    def params(self):
        return {"C": self.c}


LGBM_PARAMS = dict(
    learning_rate=0.03, num_leaves=15, min_child_samples=200, subsample=0.7, subsample_freq=1,
    colsample_bytree=0.7, reg_lambda=1.0, n_estimators=1000, random_state=SEED, deterministic=True,
    force_row_wise=True, verbose=-1,
)


class LGBMReturn(_Tabular):
    """LightGBM (Huber loss) on the vol-normalised forward return; early stopping on validation."""

    def __init__(self, features, **params) -> None:
        super().__init__(features)
        self.name = "lgbm_return"
        self.kw = {**LGBM_PARAMS, **params}

    def fit(self, train, val, h):
        import lightgbm as lgb

        x, y = self._xy(train, h, "reg")
        xv, yv = self._xy(val, h, "reg")
        self.model_ = lgb.LGBMRegressor(objective="huber", alpha=1.0, **self.kw)
        with warnings.catch_warnings():  # eval_set is deprecated in lightgbm>=4.7 but its replacement is missing in older versions
            warnings.filterwarnings("ignore", message=".*eval_set.*")
            self.model_.fit(x, y, eval_set=[(xv, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
        return self

    def predict(self, frame):
        return self.model_.predict(frame[self.features])

    def params(self):
        return {k: v for k, v in self.kw.items() if k not in ("verbose", "force_row_wise")}


class LGBMBarrier(_Tabular):
    """LightGBM multiclass on triple-barrier labels; score = P(long) - P(short)."""

    def __init__(self, features, **params) -> None:
        super().__init__(features)
        self.name = "lgbm_tb"
        self.kw = {**LGBM_PARAMS, **params}

    def fit(self, train, val, h):
        import lightgbm as lgb

        x, y = self._xy(train, h, "tb")
        xv, yv = self._xy(val, h, "tb")
        self.model_ = lgb.LGBMClassifier(objective="multiclass", **self.kw)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*eval_set.*")
            self.model_.fit(x, y.astype(int), eval_set=[(xv, yv.astype(int))], callbacks=[lgb.early_stopping(50, verbose=False)])
        return self

    def predict(self, frame):
        p = self.model_.predict_proba(frame[self.features])
        cls = list(self.model_.classes_)
        long_ = p[:, cls.index(2)] if 2 in cls else 0.0
        short = p[:, cls.index(0)] if 0 in cls else 0.0
        return long_ - short

    def params(self):
        return {k: v for k, v in self.kw.items() if k not in ("verbose", "force_row_wise")}


def all_baselines(features: list[str]) -> list[Model]:
    return rule_baselines() + [RidgeReturn(features), LogisticBarrier(features), LGBMReturn(features), LGBMBarrier(features)]
