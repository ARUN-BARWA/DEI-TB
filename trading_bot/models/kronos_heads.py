"""Models for the central question: does Kronos add trading-relevant information beyond the baselines?

Arms (identical LightGBM settings and targets; only the inputs differ):

    B   registry features                       -> what the baselines already know
    K   Kronos features only                    -> Kronos as a representation / forecaster
    BK  registry + Kronos features              -> incremental value of Kronos
    kronos_direct                               -> Kronos's own forecast, no head (option A)

The Kronos embedding (d_model values) is reduced with PCA fitted on each fold's **training rows only**.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_bot.models.baselines import LGBMBarrier, LGBMReturn, Model, RidgeReturn

KR = "kr_"


def embedding_columns(columns) -> list[str]:
    return [c for c in columns if c.startswith("kr_emb_")]


def kronos_scalar_columns(columns) -> list[str]:
    return [c for c in columns if c.startswith(KR) and not c.startswith("kr_emb_") and c != "kr_close"]


def registry_columns(columns) -> list[str]:
    return [c for c in columns if not c.startswith(KR)]


class _PCA:
    def __init__(self, n: int) -> None:
        self.n = n

    def fit(self, x: np.ndarray) -> "_PCA":
        x = x[np.isfinite(x).all(axis=1)]
        self.mean_ = x.mean(axis=0)
        _, _, vt = np.linalg.svd(x - self.mean_, full_matrices=False)
        self.components_ = vt[: min(self.n, vt.shape[0])]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        out = (x - self.mean_) @ self.components_.T  # NaN rows stay NaN
        return out


class FeatureSetModel(Model):
    """Wraps a baseline learner, giving it a chosen feature set plus train-fitted PCA of the embedding."""

    learned = True

    def __init__(self, name: str, learner_cls, features: list[str], use_registry: bool, use_kronos: bool, n_pca: int = 16) -> None:
        self.name, self.learner_cls, self.n_pca = name, learner_cls, n_pca
        self.use_kronos = use_kronos
        self.base = (registry_columns(features) if use_registry else []) + (kronos_scalar_columns(features) if use_kronos else [])
        self.emb = embedding_columns(features) if use_kronos else []

    def _augment(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.emb:
            return frame
        comps = self.pca_.transform(frame[self.emb].to_numpy(dtype=float))
        return frame.assign(**{f"kr_pc_{i}": comps[:, i] for i in range(comps.shape[1])})

    def fit(self, train, val, h):
        if self.emb:
            self.pca_ = _PCA(self.n_pca).fit(train[self.emb].to_numpy(dtype=float))
        pcs = [f"kr_pc_{i}" for i in range(self.pca_.components_.shape[0])] if self.emb else []
        self.learner_ = self.learner_cls(self.base + pcs)
        self.learner_.fit(self._augment(train), self._augment(val), h)
        return self

    def predict(self, frame):
        return self.learner_.predict(self._augment(frame))

    def params(self):
        return {"arm": self.name.split("_")[0], "n_features": len(self.base), "n_pca": self.n_pca if self.emb else 0,
                **self.learner_cls([]).params()}


class KronosDirect(Model):
    """Option A: Kronos's sampled-path mean forecast at the horizon (nearest available) as the score."""

    def __init__(self, features: list[str]) -> None:
        self.name = "kronos_direct"
        self.available = sorted(int(c[len("kr_r"):-len("_mean")]) for c in features if c.startswith("kr_r") and c.endswith("_mean"))
        self.h = None

    def fit(self, train, val, h):
        if not self.available:
            raise ValueError("no Kronos path statistics; compute features with --paths")
        self.h = min(self.available, key=lambda a: (abs(a - h), -a))
        return self

    def predict(self, frame):
        return frame[f"kr_r{self.h}_mean"].to_numpy(dtype=float)

    def params(self):
        return {"path_horizon": self.h}


def comparison_models(features: list[str]) -> list[Model]:
    has_paths = any(c.startswith("kr_r") and c.endswith("_mean") for c in features)
    models: list[Model] = [
        FeatureSetModel("B_lgbm_return", LGBMReturn, features, use_registry=True, use_kronos=False),
        FeatureSetModel("K_lgbm_return", LGBMReturn, features, use_registry=False, use_kronos=True),
        FeatureSetModel("BK_lgbm_return", LGBMReturn, features, use_registry=True, use_kronos=True),
        FeatureSetModel("B_lgbm_tb", LGBMBarrier, features, use_registry=True, use_kronos=False),
        FeatureSetModel("K_lgbm_tb", LGBMBarrier, features, use_registry=False, use_kronos=True),
        FeatureSetModel("BK_lgbm_tb", LGBMBarrier, features, use_registry=True, use_kronos=True),
        FeatureSetModel("K_ridge", RidgeReturn, features, use_registry=False, use_kronos=True),
    ]
    if has_paths:
        models.append(KronosDirect(features))
    return models
