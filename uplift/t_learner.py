"""T-Learner implementation for heterogeneous treatment effect estimation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if __name__ == "__main__" and str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config.loader import get_logger
from preprocessing.cleaning import clean_transaction_data
from preprocessing.feature_engineering import add_engineered_features, build_feature_matrix
from simulation.data_generator import generate_synthetic_marketplace_data
from uplift.base import BaseUpliftEstimator, _validate_treatment_and_features, build_base_regressor

logger = get_logger(__name__)


class TLearner(BaseUpliftEstimator):
    """Two-Model T-Learner: fits separate outcome models mu_0(x) and mu_1(x).

    tau_hat(x) = mu_1(x) - mu_0(x)
    """

    name = "T-Learner"

    def __init__(
        self,
        learner_type: str = "gradient_boosting",
        seed: int = 42,
        **model_kwargs: Any,
    ) -> None:
        self.learner_type = learner_type
        self.seed = seed
        self.model_kwargs = model_kwargs
        self.model_control = build_base_regressor(learner_type=learner_type, seed=seed, **model_kwargs)
        self.model_treated = build_base_regressor(learner_type=learner_type, seed=seed + 1, **model_kwargs)
        self._is_fitted = False

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        treatment: pd.Series | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "TLearner":
        x_arr, t_arr, y_arr = _validate_treatment_and_features(X, treatment, y)
        mask_0 = t_arr == 0
        mask_1 = t_arr == 1
        self.model_control.fit(x_arr[mask_0], y_arr[mask_0])
        self.model_treated.fit(x_arr[mask_1], y_arr[mask_1])
        self._is_fitted = True
        return self

    def predict_potential_outcomes(self, X: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self._is_fitted:
            raise RuntimeError("TLearner must be fitted before calling predict.")
        x_arr = np.asarray(X, dtype=float)
        mu0 = self.model_control.predict(x_arr)
        mu1 = self.model_treated.predict(x_arr)
        return mu0, mu1

    def predict_cate(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        mu0, mu1 = self.predict_potential_outcomes(X)
        return mu1 - mu0


if __name__ == "__main__":
    raw_df = generate_synthetic_marketplace_data(n_rows=20_000, seed=42)
    cleaned_df, _ = clean_transaction_data(raw_df)
    feat_df = add_engineered_features(cleaned_df)
    X_mat, _ = build_feature_matrix(feat_df, include_price=False)
    t_vec = feat_df["treatment_flag"].to_numpy()
    y_vec = feat_df["margin"].to_numpy()

    t_learner = TLearner(learner_type="gradient_boosting", seed=42)
    t_learner.fit(X_mat, t_vec, y_vec)
    tau_hat = t_learner.predict_cate(X_mat)
    print(f"T-Learner Estimated ATE (Margin): {tau_hat.mean():.3f}")
    if "true_cate_margin" in feat_df.columns:
        tau_true = feat_df["true_cate_margin"].to_numpy()
        pehe = float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))
        print(f"Ground-Truth ATE (Margin):        {tau_true.mean():.3f}")
        print(f"PEHE (RMSE vs true CATE):         {pehe:.3f}")
