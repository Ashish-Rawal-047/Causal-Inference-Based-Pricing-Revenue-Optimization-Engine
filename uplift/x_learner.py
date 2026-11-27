"""X-Learner implementation with imputed treatment effects and propensity weighting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from uplift.base import BaseUpliftEstimator, _validate_treatment_and_features, build_base_regressor


class XLearner(BaseUpliftEstimator):
    """X-Learner (Kunzel et al., 2019) for imbalanced/heterogeneous treatment effect estimation.

    Stage 1: Fit outcome models mu_0(x) on control and mu_1(x) on treated.
    Impute individual treatment effects:
        D_1 = Y_1 - mu_0(X_1)   (for treated units)
        D_0 = mu_1(X_0) - Y_0   (for control units)
    Stage 2: Fit CATE models tau_1(x) on (X_1, D_1) and tau_0(x) on (X_0, D_0).
    Propensity weighting:
        tau(x) = g(x) * tau_0(x) + (1 - g(x)) * tau_1(x)
    where g(x) = clip(P(T=1 | X=x), clip_low, 1 - clip_low).
    """

    name = "X-Learner"

    def __init__(
        self,
        learner_type: str = "gradient_boosting",
        propensity_clip: float = 0.05,
        seed: int = 42,
        **model_kwargs: Any,
    ) -> None:
        self.learner_type = learner_type
        self.propensity_clip = propensity_clip
        self.seed = seed
        self.model_kwargs = model_kwargs

        self.mu0 = build_base_regressor(learner_type=learner_type, seed=seed, **model_kwargs)
        self.mu1 = build_base_regressor(learner_type=learner_type, seed=seed + 1, **model_kwargs)
        self.tau0 = build_base_regressor(learner_type=learner_type, seed=seed + 2, **model_kwargs)
        self.tau1 = build_base_regressor(learner_type=learner_type, seed=seed + 3, **model_kwargs)
        self.propensity_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500, C=1.0, random_state=seed),
        )
        self._is_fitted = False

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        treatment: pd.Series | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "XLearner":
        x_arr, t_arr, y_arr = _validate_treatment_and_features(X, treatment, y)
        mask_0 = t_arr == 0
        mask_1 = t_arr == 1

        # Stage 1: outcome regressions
        self.mu0.fit(x_arr[mask_0], y_arr[mask_0])
        self.mu1.fit(x_arr[mask_1], y_arr[mask_1])

        # Imputed treatment effects
        d_1 = y_arr[mask_1] - self.mu0.predict(x_arr[mask_1])
        d_0 = self.mu1.predict(x_arr[mask_0]) - y_arr[mask_0]

        # Stage 2: CATE regressions on imputed treatment effects
        self.tau1.fit(x_arr[mask_1], d_1)
        self.tau0.fit(x_arr[mask_0], d_0)

        # Propensity model g(x) = P(T=1 | X=x)
        self.propensity_model.fit(x_arr, t_arr)
        self._is_fitted = True
        return self

    def predict_propensity(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("XLearner must be fitted before prediction.")
        x_arr = np.asarray(X, dtype=float)
        g = self.propensity_model.predict_proba(x_arr)[:, 1]
        return np.clip(g, self.propensity_clip, 1.0 - self.propensity_clip)

    def predict_cate(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("XLearner must be fitted before calling predict_cate.")
        x_arr = np.asarray(X, dtype=float)
        g = self.predict_propensity(x_arr)
        t0 = self.tau0.predict(x_arr)
        t1 = self.tau1.predict(x_arr)
        return g * t0 + (1.0 - g) * t1
