"""Comprehensive causal validity diagnostics: balance, overlap, SUTVA, and sensitivity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from causal.identification import CausalAssumptionError, get_identification_assumptions_registry
from preprocessing.feature_engineering import build_feature_matrix


@dataclass
class OverlapBalanceDiagnostics:
    """Diagnostics for positivity/overlap and covariate balance (SMD)."""

    propensity_min: float
    propensity_max: float
    treated_propensity_mean: float
    control_propensity_mean: float
    common_support_fraction: float
    positivity_satisfied: bool
    balance_table: pd.DataFrame
    max_raw_smd: float
    max_ipw_smd: float


@dataclass
class SensitivityResult:
    """Omitted-variable sensitivity analysis (Cinelli & Hazlett partial-R2 robustness)."""

    baseline_estimate: float
    baseline_se: float
    t_statistic: float
    robustness_value_rv_q1: float
    partial_r2_yd_given_x: float
    sensitivity_grid: pd.DataFrame
    interpretation: str


def check_treatment_overlap_and_balance(
    df: pd.DataFrame,
    treatment_col: str = "treatment_flag",
    common_support_bounds: tuple[float, float] = (0.05, 0.95),
) -> OverlapBalanceDiagnostics:
    """Evaluate positivity/overlap and covariate Standardized Mean Differences (SMD) before/after IPW."""
    if treatment_col not in df.columns:
        raise CausalAssumptionError(f"Treatment column '{treatment_col}' not found.")

    t = pd.to_numeric(df[treatment_col], errors="coerce").to_numpy()
    uniq = set(np.unique(t[~np.isnan(t)]))
    if uniq != {0, 1}:
        raise CausalAssumptionError(
            f"Binary treatment variation required in '{treatment_col}'; found unique values {sorted(uniq)}."
        )

    X_mat, feat_names = build_feature_matrix(df, include_price=False)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_mat)

    prop_model = LogisticRegression(max_iter=500, C=1.0, random_state=42)
    prop_model.fit(X_scaled, t.astype(int))
    e_hat = np.clip(prop_model.predict_proba(X_scaled)[:, 1], 1e-4, 1.0 - 1e-4)

    lo, hi = common_support_bounds
    in_support = (e_hat >= lo) & (e_hat <= hi)
    support_frac = float(in_support.mean())

    # Stabilized IPW weights
    p_treat = float(t.mean())
    w_ipw = np.where(t == 1, p_treat / e_hat, (1.0 - p_treat) / (1.0 - e_hat))
    w_ipw = np.clip(w_ipw, 0.05, 20.0)

    balance_rows: list[dict[str, Any]] = []
    for col in feat_names[:20]:  # report top 20 core & dummy features cleanly
        vals = X_mat[col].to_numpy(dtype=float)
        v1 = vals[t == 1]
        v0 = vals[t == 0]
        pooled_sd = np.sqrt(0.5 * (v1.var() + v0.var()))
        raw_smd = abs(v1.mean() - v0.mean()) / pooled_sd if pooled_sd > 1e-12 else 0.0

        w1 = w_ipw[t == 1]
        w0 = w_ipw[t == 0]
        wm1 = np.average(v1, weights=w1)
        wm0 = np.average(v0, weights=w0)
        ipw_smd = abs(wm1 - wm0) / pooled_sd if pooled_sd > 1e-12 else 0.0

        balance_rows.append(
            {
                "feature": col,
                "treated_mean": float(v1.mean()),
                "control_mean": float(v0.mean()),
                "raw_smd": float(raw_smd),
                "ipw_smd": float(ipw_smd),
            }
        )

    bal_df = pd.DataFrame(balance_rows).sort_values("raw_smd", ascending=False).reset_index(drop=True)
    positivity_ok = bool(
        support_frac >= 0.80
        and e_hat[t == 1].min() < 0.85
        and e_hat[t == 0].max() > 0.15
    )

    return OverlapBalanceDiagnostics(
        propensity_min=float(e_hat.min()),
        propensity_max=float(e_hat.max()),
        treated_propensity_mean=float(e_hat[t == 1].mean()),
        control_propensity_mean=float(e_hat[t == 0].mean()),
        common_support_fraction=support_frac,
        positivity_satisfied=positivity_ok,
        balance_table=bal_df,
        max_raw_smd=float(bal_df["raw_smd"].max()) if not bal_df.empty else 0.0,
        max_ipw_smd=float(bal_df["ipw_smd"].max()) if not bal_df.empty else 0.0,
    )


def run_omitted_variable_sensitivity(
    df: pd.DataFrame,
    treatment_col: str = "log_price",
    outcome_col: str = "log_quantity",
) -> SensitivityResult:
    """Compute Cinelli & Hazlett (2020) omitted-variable sensitivity diagnostics.

    Measures the minimum partial R^2 strength (Robustness Value RV_{q=1}) an unmeasured
    confounder would need with BOTH treatment and outcome (conditional on X) to bring
    the point estimate to zero.
    """
    X_mat, _ = build_feature_matrix(df, include_price=False)
    d = pd.to_numeric(df[treatment_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[outcome_col], errors="coerce").to_numpy(dtype=float)

    X_design = sm.add_constant(pd.concat([pd.Series(d, name=treatment_col, index=df.index), X_mat], axis=1), has_constant="add")
    fit = sm.OLS(y, X_design).fit(cov_type="HC3")
    est = float(fit.params[treatment_col])
    se = float(fit.bse[treatment_col])
    t_val = float(fit.tvalues[treatment_col])
    df_resid = max(int(fit.df_resid), 1)

    # Partial R^2 of treatment with outcome given covariates: t^2 / (t^2 + df)
    r2_yd = (t_val ** 2) / ((t_val ** 2) + df_resid)
    # Cinelli & Hazlett Robustness Value RV_{q=1}: 0.5 * (sqrt(f^4 + 4*f^2) - f^2) where f^2 = t^2 / df
    f_sq = (t_val ** 2) / df_resid
    f_val = np.sqrt(f_sq)
    rv = float(0.5 * (np.sqrt(f_val ** 4 + 4.0 * f_val ** 2) - f_val ** 2))

    # Sensitivity bias grid over hypothetical confounder partial R^2 values (R2_dz, R2_yz)
    r_grid = [0.01, 0.05, 0.10, 0.20, 0.35]
    # Standard error scaling factor for omitted variable bias magnitude: |bias| = se * sqrt(df) * sqrt(r2_yz * r2_dz / (1 - r2_dz))
    grid_rows = []
    for r_dz in r_grid:
        for r_yz in r_grid:
            max_bias = se * np.sqrt(df_resid) * np.sqrt((r_yz * r_dz) / max(1.0 - r_dz, 1e-9))
            worst_case_est = est + max_bias if est < 0 else est - max_bias
            grid_rows.append(
                {
                    "confounder_r2_with_treatment": r_dz,
                    "confounder_r2_with_outcome": r_yz,
                    "max_abs_bias": float(max_bias),
                    "worst_case_estimate": float(worst_case_est),
                    "sign_preserved": bool(np.sign(worst_case_est) == np.sign(est)),
                }
            )

    return SensitivityResult(
        baseline_estimate=est,
        baseline_se=se,
        t_statistic=t_val,
        robustness_value_rv_q1=rv,
        partial_r2_yd_given_x=float(r2_yd),
        sensitivity_grid=pd.DataFrame(grid_rows),
        interpretation=(
            f"An unmeasured confounder would need to explain at least {rv * 100:.2f}% of the residual variance "
            f"of BOTH {treatment_col} and {outcome_col} simultaneously to reduce the controlled OLS estimate "
            f"({est:.4f}) to zero. In our synthetic marketplace, u_demand actually induces substantial bias, "
            "which is why 2SLS IV and DiD are required."
        ),
    )


def check_sutva_and_interference_risk(df: pd.DataFrame) -> dict[str, Any]:
    """Assess structural indicators of potential SUTVA / interference violations."""
    work = df.copy()
    work["date"] = pd.to_datetime(work["timestamp"]).dt.date
    prod_day = work.groupby(["product_id", "date"], as_index=False).agg(
        daily_orders=("order_id", "count"),
        avg_avail=("seller_availability", "mean") if "seller_availability" in work.columns else ("quantity", "count"),
        promo_share=("promotion_flag", "mean"),
    )
    high_contention_share = float((prod_day["daily_orders"] >= 3).mean())
    return {
        "product_days_with_multiple_orders_share": high_contention_share,
        "sutva_assessment": (
            "Synthetic DGP enforces unit-level potential outcomes (SUTVA holds by construction), "
            f"though {high_contention_share:.1%} of product-days have >=3 concurrent orders, illustrating "
            "where real marketplaces could experience inventory depletion or cross-customer spillovers."
        ),
    }


def run_causal_diagnostics_report(
    df: pd.DataFrame,
    iv_result: Any | None = None,
    did_result: Any | None = None,
) -> dict[str, Any]:
    """Run the full causal diagnostics suite and return structured assumptions & tests."""
    overlap = check_treatment_overlap_and_balance(df)
    sensitivity = run_omitted_variable_sensitivity(df)
    sutva = check_sutva_and_interference_risk(df)
    registry = [item.__dict__ for item in get_identification_assumptions_registry()]

    summary: dict[str, Any] = {
        "overlap_balance": overlap,
        "sensitivity": sensitivity,
        "sutva": sutva,
        "assumptions_registry": pd.DataFrame(registry),
    }
    if iv_result is not None:
        summary["iv_diagnostics"] = {
            "first_stage_f": iv_result.first_stage_f_stat,
            "weak_instrument": iv_result.weak_instrument_flag,
            "wu_hausman_p": iv_result.wu_hausman_p_value,
            "max_cov_smd_across_z": iv_result.instrument_balance_max_smd,
        }
    if did_result is not None:
        summary["did_diagnostics"] = {
            "pre_trend_slope": did_result.pre_trend_diff_slope,
            "pre_trend_p": did_result.pre_trend_p_value,
            "parallel_trends_plausible": did_result.parallel_trends_plausible,
        }
    return summary
