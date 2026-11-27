"""Base interface, regressor factory, and S-Learner baseline for heterogeneous treatment effects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from causal.identification import CausalAssumptionError

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def build_base_regressor(
    learner_type: str = "gradient_boosting",
    seed: int = 42,
    n_estimators: int = 120,
    max_depth: int = 4,
    learning_rate: float = 0.08,
) -> RegressorMixin:
    """Instantiate a regression estimator for meta-learner outcome/CATE models."""
    kind = learner_type.lower()
    if kind == "random_forest":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth + 2,
            min_samples_leaf=10,
            random_state=seed,
            n_jobs=-1,
        )
    if kind == "gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=n_estimators,
            max_depth=max_depth + 1,
            learning_rate=learning_rate,
            min_samples_leaf=20,
            random_state=seed,
        )
    if kind == "sklearn_gbr":
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.85,
            random_state=seed,
        )
    if kind == "xgboost":
        if not HAS_XGBOOST:
            raise ImportError("xgboost requested but not installed.")
        return XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=seed))
    raise ValueError(f"Unsupported learner_type '{learner_type}'.")


def _validate_treatment_and_features(
    X: pd.DataFrame | np.ndarray,
    treatment: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_arr = np.asarray(X, dtype=float)
    t_arr = np.asarray(treatment, dtype=int)
    y_arr = np.asarray(y, dtype=float)

    if len(x_arr) != len(t_arr) or len(x_arr) != len(y_arr):
        raise ValueError("X, treatment, and y must have identical lengths.")
    if len(x_arr) < 10:
        raise CausalAssumptionError("Insufficient observations (< 10) to fit uplift estimator.")
    n_treated = int((t_arr == 1).sum())
    n_control = int((t_arr == 0).sum())
    if n_treated == 0 or n_control == 0:
        raise CausalAssumptionError(
            f"Uplift estimation requires both treated and control units (treated={n_treated}, control={n_control})."
        )
    if not np.all(np.isin(t_arr, [0, 1])):
        raise ValueError("Treatment indicator must be binary (0 or 1).")
    return x_arr, t_arr, y_arr


class BaseUpliftEstimator(ABC):
    """Common interface for heterogeneous treatment effect (CATE) estimators."""

    name: str = "BaseUpliftEstimator"

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        treatment: pd.Series | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "BaseUpliftEstimator":
        """Fit outcome/treatment models on observational or experimental data."""

    @abstractmethod
    def predict_cate(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Estimate Conditional Average Treatment Effect tau(x) = E[Y(1) - Y(0) | X=x]."""

    def predict_ate(self, X: pd.DataFrame | np.ndarray) -> float:
        """Estimate sample Average Treatment Effect mean(tau(x))."""
        return float(np.mean(self.predict_cate(X)))


class SLearner(BaseUpliftEstimator):
    """Single-model S-Learner baseline including treatment indicator as a feature."""

    name = "S-Learner"

    def __init__(
        self,
        learner_type: str = "gradient_boosting",
        seed: int = 42,
        **model_kwargs: Any,
    ) -> None:
        self.learner_type = learner_type
        self.seed = seed
        self.model_kwargs = model_kwargs
        self.model = build_base_regressor(learner_type=learner_type, seed=seed, **model_kwargs)
        self._is_fitted = False

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        treatment: pd.Series | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "SLearner":
        x_arr, t_arr, y_arr = _validate_treatment_and_features(X, treatment, y)
        xt = np.column_stack([x_arr, t_arr.astype(float)])
        self.model.fit(xt, y_arr)
        self._is_fitted = True
        return self

    def predict_potential_outcomes(self, X: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self._is_fitted:
            raise RuntimeError("SLearner must be fitted before prediction.")
        x_arr = np.asarray(X, dtype=float)
        xt0 = np.column_stack([x_arr, np.zeros(len(x_arr), dtype=float)])
        xt1 = np.column_stack([x_arr, np.ones(len(x_arr), dtype=float)])
        mu0 = self.model.predict(xt0)
        mu1 = self.model.predict(xt1)
        return mu0, mu1

    def predict_cate(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        mu0, mu1 = self.predict_potential_outcomes(X)
        return mu1 - mu0
