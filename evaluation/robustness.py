"""Robustness, bootstrap uncertainty, confounding stress tests, and distribution shift."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from analysis.elasticity import estimate_controlled_loglog_elasticity, estimate_naive_loglog_elasticity
from causal.did import estimate_difference_in_differences
from causal.iv import estimate_2sls_iv
from config.loader import get_logger
from preprocessing.cleaning import clean_transaction_data
from preprocessing.feature_engineering import add_engineered_features
from simulation.data_generator import generate_synthetic_marketplace_data

logger = get_logger(__name__)


def bootstrap_causal_elasticity(
    df: pd.DataFrame,
    n_bootstrap: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute bootstrap confidence intervals for Naive OLS, Controlled OLS, and 2SLS IV."""
    rng = np.random.default_rng(seed)
    n = len(df)
    naive_boots: list[float] = []
    ctrl_boots: list[float] = []
    iv_boots: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = df.iloc[idx].reset_index(drop=True)
        naive_boots.append(estimate_naive_loglog_elasticity(sample).elasticity)
        ctrl_boots.append(estimate_controlled_loglog_elasticity(sample, compute_vif=False).elasticity)
        iv_boots.append(estimate_2sls_iv(sample).elasticity)

    def _summarize(vals: list[float], name: str) -> dict[str, float | str]:
        arr = np.asarray(vals, dtype=float)
        return {
            "estimator": name,
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)),
            "ci_2_5": float(np.percentile(arr, 2.5)),
            "ci_97_5": float(np.percentile(arr, 97.5)),
        }

    return {
        "n_bootstrap": n_bootstrap,
        "summary_df": pd.DataFrame(
            [
                _summarize(naive_boots, "Naive Log-Log OLS"),
                _summarize(ctrl_boots, "Controlled Log-Log OLS"),
                _summarize(iv_boots, "Causal 2SLS IV"),
            ]
        ),
    }


def run_robustness_suite(
    seed: int = 42,
    n_rows: int = 12_000,
    confounding_levels: Sequence[float] = (0.0, 0.5, 1.0, 1.8),
    assignment_modes: Sequence[str] = ("randomized", "observational", "skewed"),
    elasticity_shifts: Sequence[float] = (-0.25, 0.0, 0.25),
    sample_sizes: Sequence[int] = (5_000, 10_000, 25_000),
) -> dict[str, pd.DataFrame]:
    """Evaluate causal estimator stability across DGP assumptions and distribution shifts."""
    conf_rows: list[dict[str, Any]] = []
    for c_str in confounding_levels:
        raw = generate_synthetic_marketplace_data(n_rows=n_rows, seed=seed, confounding_strength=c_str)
        cln, _ = clean_transaction_data(raw)
        feat = add_engineered_features(cln)
        naive = estimate_naive_loglog_elasticity(feat)
        ctrl = estimate_controlled_loglog_elasticity(feat)
        iv = estimate_2sls_iv(feat)
        did = estimate_difference_in_differences(feat)
        true_el = float(feat["true_elasticity"].mean())
        conf_rows.append(
            {
                "confounding_strength": float(c_str),
                "true_elasticity": true_el,
                "naive_ols": naive.elasticity,
                "controlled_ols": ctrl.elasticity,
                "iv_2sls": iv.elasticity,
                "did_implied": did.implied_did_elasticity,
                "naive_abs_error": abs(naive.elasticity - true_el),
                "controlled_abs_error": abs(ctrl.elasticity - true_el),
                "iv_abs_error": abs(iv.elasticity - true_el),
            }
        )

    assign_rows: list[dict[str, Any]] = []
    for mode in assignment_modes:
        raw = generate_synthetic_marketplace_data(n_rows=n_rows, seed=seed, treatment_assignment=mode)
        cln, _ = clean_transaction_data(raw)
        feat = add_engineered_features(cln)
        iv = estimate_2sls_iv(feat)
        true_el = float(feat["true_elasticity"].mean())
        assign_rows.append(
            {
                "treatment_assignment": mode,
                "treated_rate": float(feat["treatment_flag"].mean()),
                "true_elasticity": true_el,
                "iv_2sls": iv.elasticity,
                "iv_abs_error": abs(iv.elasticity - true_el),
                "first_stage_f": iv.first_stage_f_stat,
            }
        )

    shift_rows: list[dict[str, Any]] = []
    for e_shift in elasticity_shifts:
        raw = generate_synthetic_marketplace_data(
            n_rows=n_rows, seed=seed, elasticity_shift=e_shift, demand_shock_scale=0.45
        )
        cln, _ = clean_transaction_data(raw)
        feat = add_engineered_features(cln)
        ctrl = estimate_controlled_loglog_elasticity(feat)
        iv = estimate_2sls_iv(feat)
        true_el = float(feat["true_elasticity"].mean())
        shift_rows.append(
            {
                "elasticity_shift": float(e_shift),
                "true_elasticity": true_el,
                "controlled_ols": ctrl.elasticity,
                "iv_2sls": iv.elasticity,
                "iv_abs_error": abs(iv.elasticity - true_el),
            }
        )

    size_rows: list[dict[str, Any]] = []
    for sz in sample_sizes:
        raw = generate_synthetic_marketplace_data(n_rows=sz, seed=seed)
        cln, _ = clean_transaction_data(raw)
        feat = add_engineered_features(cln)
        iv = estimate_2sls_iv(feat)
        true_el = float(feat["true_elasticity"].mean())
        size_rows.append(
            {
                "sample_size": int(sz),
                "true_elasticity": true_el,
                "iv_2sls": iv.elasticity,
                "iv_se": iv.std_error,
                "iv_abs_error": abs(iv.elasticity - true_el),
                "first_stage_f": iv.first_stage_f_stat,
            }
        )

    return {
        "confounding_sensitivity": pd.DataFrame(conf_rows),
        "assignment_mechanisms": pd.DataFrame(assign_rows),
        "distribution_shift": pd.DataFrame(shift_rows),
        "sample_size_scaling": pd.DataFrame(size_rows),
    }
