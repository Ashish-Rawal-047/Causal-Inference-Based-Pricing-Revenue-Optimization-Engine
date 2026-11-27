"""Instrumental Variables (2SLS) estimator implemented from first principles."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

if __name__ == "__main__" and str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from causal.identification import CausalAssumptionError
from config.loader import get_logger
from preprocessing.cleaning import clean_transaction_data
from preprocessing.feature_engineering import add_engineered_features
from simulation.data_generator import generate_synthetic_marketplace_data

logger = get_logger(__name__)


@dataclass
class IVResult:
    """Comprehensive output of a Two-Stage Least Squares (2SLS) elasticity estimation."""

    elasticity: float
    std_error: float
    ci_low: float
    ci_high: float
    p_value: float
    first_stage_coef: float
    first_stage_se: float
    first_stage_f_stat: float
    first_stage_partial_r2: float
    weak_instrument_flag: bool
    wu_hausman_stat: float
    wu_hausman_p_value: float
    n_obs: int
    instrument_balance_max_smd: float
    ols_comparison_elasticity: float
    assumptions_note: str


def _prepare_controls(
    df: pd.DataFrame,
    control_numeric: Sequence[str],
    control_categorical: Sequence[str],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    num = [c for c in control_numeric if c in df.columns]
    if num:
        parts.append(df[num].astype(float))
    cat = [c for c in control_categorical if c in df.columns and df[c].nunique() > 1]
    if cat:
        parts.append(pd.get_dummies(df[cat].astype(str), drop_first=True, dtype=float))
    if not parts:
        return pd.DataFrame(index=df.index)
    X_ctrl = pd.concat(parts, axis=1)
    keep = [c for c in X_ctrl.columns if X_ctrl[c].std() > 1e-12]
    return X_ctrl[keep]


def estimate_2sls_iv(
    df: pd.DataFrame,
    instrument_col: str = "supplier_cost_shock",
    treatment_col: str = "log_price",
    outcome_col: str = "log_quantity",
    control_numeric: Sequence[str] | None = None,
    control_categorical: Sequence[str] | None = ("region", "category", "customer_cohort"),
    weak_f_threshold: float = 10.0,
) -> IVResult:
    """Estimate causal price elasticity using Two-Stage Least Squares (2SLS).

    Stage 1: treatment = pi_0 + pi_1 * Z + Gamma * X + v
    Stage 2: outcome   = beta_0 + beta_1 * treatment_hat + Phi * X + u

    Asymptotic HC1 heteroskedasticity-robust standard errors are computed using the
    structural residuals u_i = Y_i - W_i' beta_2sls (with actual treatment, not predicted).
    """
    for req in (instrument_col, treatment_col, outcome_col):
        if req not in df.columns:
            raise CausalAssumptionError(f"Required column '{req}' not found for 2SLS IV estimation.")

    if len(df) < 30:
        raise CausalAssumptionError(f"Insufficient observations ({len(df)}) for 2SLS IV estimation.")

    z = pd.to_numeric(df[instrument_col], errors="coerce").to_numpy(dtype=float)
    d = pd.to_numeric(df[treatment_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[outcome_col], errors="coerce").to_numpy(dtype=float)

    if np.any(~np.isfinite(z)) or np.any(~np.isfinite(d)) or np.any(~np.isfinite(y)):
        raise CausalAssumptionError("Non-finite values detected in instrument, treatment, or outcome.")
    if np.std(z) < 1e-10:
        raise CausalAssumptionError(f"Instrument '{instrument_col}' has zero variation.")
    if np.std(d) < 1e-10:
        raise CausalAssumptionError(f"Treatment '{treatment_col}' has zero variation.")

    if control_numeric is None:
        control_numeric = [
            c
            for c in (
                "log_competitor_price",
                "promotion_flag",
                "is_weekend",
                "seasonality_sin",
                "seasonality_cos",
                "log_historical_orders",
                "seller_availability",
                "festival_flag",
            )
            if c in df.columns
        ]

    X_ctrl = _prepare_controls(df, control_numeric, control_categorical or ())
    n = len(df)

    # Design matrices:
    # Exogenous matrix X_exog = [1, X_ctrl]
    X_exog_df = sm.add_constant(X_ctrl, has_constant="add")
    X_exog = X_exog_df.to_numpy(dtype=float)

    # First stage: [1, Z, X_ctrl]
    Z_mat = np.column_stack([X_exog[:, 0], z, X_exog[:, 1:]]) if X_exog.shape[1] > 1 else np.column_stack([X_exog[:, 0], z])
    fs_restricted = sm.OLS(d, X_exog).fit()
    fs_unrestricted = sm.OLS(d, Z_mat).fit(cov_type="HC1")

    d_hat = fs_unrestricted.fittedvalues
    v_hat = d - d_hat

    # Partial F-test on excluded instrument Z
    rss_r = float(np.sum(fs_restricted.resid ** 2))
    rss_u = float(np.sum(fs_unrestricted.resid ** 2))
    df_resid_u = max(n - Z_mat.shape[1], 1)
    first_stage_f = ((rss_r - rss_u) / 1.0) / max(rss_u / df_resid_u, 1e-12)
    partial_r2 = (rss_r - rss_u) / max(rss_r, 1e-12)
    weak_flag = bool(first_stage_f < weak_f_threshold)
    if weak_flag:
        logger.warning(
            "Weak instrument detected for '%s': first-stage F = %.2f (< %.1f). 2SLS estimates may be severely biased.",
            instrument_col,
            first_stage_f,
            weak_f_threshold,
        )

    # Second stage matrix: W = [1, d, X_ctrl], W_hat = [1, d_hat, X_ctrl]
    W = np.column_stack([X_exog[:, 0], d, X_exog[:, 1:]]) if X_exog.shape[1] > 1 else np.column_stack([X_exog[:, 0], d])
    W_hat = (
        np.column_stack([X_exog[:, 0], d_hat, X_exog[:, 1:]])
        if X_exog.shape[1] > 1
        else np.column_stack([X_exog[:, 0], d_hat])
    )

    k = W.shape[1]
    wt_w_inv = np.linalg.pinv(W_hat.T @ W_hat)
    beta_2sls = wt_w_inv @ (W_hat.T @ y)

    # Structural residuals using actual treatment W (crucial for valid 2SLS SEs!)
    u_struct = y - W @ beta_2sls
    # HC1 robust sandwich covariance: (W_hat' W_hat)^-1 (sum_i u_i^2 w_hat_i w_hat_i') (W_hat' W_hat)^-1 * n/(n-k)
    scaled_w_hat = W_hat * u_struct[:, None]
    meat = scaled_w_hat.T @ scaled_w_hat
    vcov = (n / max(n - k, 1)) * (wt_w_inv @ meat @ wt_w_inv)

    elasticity = float(beta_2sls[1])
    se = float(np.sqrt(max(vcov[1, 1], 1e-16)))
    t_stat = elasticity / se
    p_value = float(2.0 * stats.t.sf(abs(t_stat), df=max(n - k, 1)))
    t_crit = float(stats.t.ppf(0.975, df=max(n - k, 1)))
    ci_low = elasticity - t_crit * se
    ci_high = elasticity + t_crit * se

    # Wu-Hausman endogeneity test: augment OLS with first-stage residual v_hat
    W_aug = np.column_stack([W, v_hat])
    aug_fit = sm.OLS(y, W_aug).fit(cov_type="HC1")
    wu_t = float(aug_fit.tvalues[-1])
    wu_stat = wu_t ** 2
    wu_p = float(aug_fit.pvalues[-1])

    # OLS comparison on the exact same controls
    ols_fit = sm.OLS(y, W).fit(cov_type="HC1")
    ols_el = float(ols_fit.params[1])

    # Independence diagnostic: check observed covariate balance across median split of instrument Z
    z_high = z >= np.median(z)
    smds: list[float] = []
    for col in X_ctrl.columns:
        vals = X_ctrl[col].to_numpy(dtype=float)
        m1, m0 = vals[z_high].mean(), vals[~z_high].mean()
        s_pool = np.sqrt(0.5 * (vals[z_high].var() + vals[~z_high].var()))
        if s_pool > 1e-12:
            smds.append(abs(m1 - m0) / s_pool)
    max_smd = float(max(smds)) if smds else 0.0

    note = (
        f"2SLS using '{instrument_col}'. Relevance F={first_stage_f:.2f} "
        f"({'WEAK' if weak_flag else 'STRONG'}). Max observed covariate SMD across Z median={max_smd:.3f}. "
        "Note: Exclusion restriction and independence from unobserved confounders cannot be proven from observational data alone."
    )

    return IVResult(
        elasticity=elasticity,
        std_error=se,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        first_stage_coef=float(fs_unrestricted.params[1]),
        first_stage_se=float(fs_unrestricted.bse[1]),
        first_stage_f_stat=float(first_stage_f),
        first_stage_partial_r2=float(partial_r2),
        weak_instrument_flag=weak_flag,
        wu_hausman_stat=wu_stat,
        wu_hausman_p_value=wu_p,
        n_obs=n,
        instrument_balance_max_smd=max_smd,
        ols_comparison_elasticity=ols_el,
        assumptions_note=note,
    )


if __name__ == "__main__":
    raw_df = generate_synthetic_marketplace_data(n_rows=30_000, seed=42)
    cleaned_df, _ = clean_transaction_data(raw_df)
    feat_df = add_engineered_features(cleaned_df)
    res = estimate_2sls_iv(feat_df)
    print("=== 2SLS Instrumental Variable Elasticity ===")
    print(f"2SLS Elasticity:   {res.elasticity:.4f} (SE={res.std_error:.4f}, 95% CI=[{res.ci_low:.4f}, {res.ci_high:.4f}])")
    print(f"Controlled OLS:    {res.ols_comparison_elasticity:.4f}")
    if "true_elasticity" in feat_df.columns:
        print(f"True Mean Elast.:  {feat_df['true_elasticity'].mean():.4f}")
    print(f"First-stage F:     {res.first_stage_f_stat:.2f} (Weak={res.weak_instrument_flag})")
    print(f"Wu-Hausman p-val:  {res.wu_hausman_p_value:.4e}")
    print(f"Notes: {res.assumptions_note}")
