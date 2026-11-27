"""Difference-in-Differences (DiD) estimator with parallel trends and event-study diagnostics."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

if __name__ == "__main__" and str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import statsmodels.api as sm

from causal.identification import CausalAssumptionError
from config.loader import get_logger
from preprocessing.cleaning import clean_transaction_data
from preprocessing.feature_engineering import add_engineered_features
from simulation.data_generator import generate_synthetic_marketplace_data

logger = get_logger(__name__)


@dataclass
class DiDResult:
    """Structured Difference-in-Differences estimation and diagnostic results."""

    ate_log_quantity: float
    std_error_log_q: float
    ci_low_log_q: float
    ci_high_log_q: float
    p_value_log_q: float
    ate_quantity: float
    std_error_qty: float
    ci_low_qty: float
    ci_high_qty: float
    did_price_change_log: float
    implied_did_elasticity: float | None
    cell_means_log_q: dict[str, float]
    cell_means_qty: dict[str, float]
    pre_trend_diff_slope: float
    pre_trend_p_value: float
    parallel_trends_plausible: bool
    event_study_df: pd.DataFrame
    n_obs: int
    assumptions_note: str


def estimate_difference_in_differences(
    df: pd.DataFrame,
    group_col: str = "did_group",
    post_col: str = "post_period",
    timestamp_col: str = "timestamp",
    control_numeric: Sequence[str] | None = None,
    control_categorical: Sequence[str] | None = ("category", "customer_cohort"),
    n_event_bins: int = 12,
) -> DiDResult:
    """Estimate Difference-in-Differences treatment effect and pre-trend diagnostics.

    Model:
        Y = b0 + b1*Treated + b2*Post + b3*(Treated * Post) + Gamma*X + e
    where b3 is the DiD ATE.
    """
    for req in (group_col, post_col, "log_quantity", "quantity", "log_price"):
        if req not in df.columns:
            raise CausalAssumptionError(f"Required column '{req}' missing for DiD estimation.")

    g = pd.to_numeric(df[group_col], errors="coerce").to_numpy()
    p = pd.to_numeric(df[post_col], errors="coerce").to_numpy()

    if set(np.unique(g[~np.isnan(g)])) != {0, 1}:
        raise CausalAssumptionError(
            f"DiD requires both control (0) and treated (1) groups in '{group_col}'."
        )
    if set(np.unique(p[~np.isnan(p)])) != {0, 1}:
        raise CausalAssumptionError(
            f"DiD requires both pre (0) and post (1) periods in '{post_col}'."
        )

    # Check all 2x2 cells are populated
    cell_counts = {
        "control_pre": int(((g == 0) & (p == 0)).sum()),
        "control_post": int(((g == 0) & (p == 1)).sum()),
        "treated_pre": int(((g == 1) & (p == 0)).sum()),
        "treated_post": int(((g == 1) & (p == 1)).sum()),
    }
    if min(cell_counts.values()) < 5:
        raise CausalAssumptionError(f"Insufficient observations in 2x2 DiD cells: {cell_counts}")

    work = df.copy()
    work["_treated"] = g.astype(float)
    work["_post"] = p.astype(float)
    work["_did_interaction"] = work["_treated"] * work["_post"]

    if control_numeric is None:
        control_numeric = [
            c
            for c in (
                "log_competitor_price",
                "is_weekend",
                "seasonality_sin",
                "seasonality_cos",
                "log_historical_orders",
                "seller_availability",
                "festival_flag",
            )
            if c in work.columns
        ]

    parts: list[pd.DataFrame] = [work[["_treated", "_post", "_did_interaction"]]]
    avail_num = [c for c in control_numeric if c in work.columns]
    if avail_num:
        parts.append(work[avail_num].astype(float))
    avail_cat = [c for c in (control_categorical or ()) if c in work.columns and work[c].nunique() > 1]
    if avail_cat:
        parts.append(pd.get_dummies(work[avail_cat].astype(str), drop_first=True, dtype=float))

    X_df = pd.concat(parts, axis=1)
    keep = [c for c in X_df.columns if X_df[c].std() > 1e-12]
    X = sm.add_constant(X_df[keep], has_constant="add")

    # Fit log_quantity DiD, quantity DiD, and log_price first-stage DiD
    fit_log_q = sm.OLS(work["log_quantity"].to_numpy(dtype=float), X).fit(cov_type="HC3")
    fit_qty = sm.OLS(work["quantity"].to_numpy(dtype=float), X).fit(cov_type="HC3")
    fit_log_p = sm.OLS(work["log_price"].to_numpy(dtype=float), X).fit(cov_type="HC3")

    ate_lq = float(fit_log_q.params["_did_interaction"])
    se_lq = float(fit_log_q.bse["_did_interaction"])
    ci_lq = fit_log_q.conf_int().loc["_did_interaction"]
    p_lq = float(fit_log_q.pvalues["_did_interaction"])

    ate_q = float(fit_qty.params["_did_interaction"])
    se_q = float(fit_qty.bse["_did_interaction"])
    ci_q = fit_qty.conf_int().loc["_did_interaction"]

    did_lp = float(fit_log_p.params["_did_interaction"])
    implied_elast = (ate_lq / did_lp) if abs(did_lp) > 1e-4 else None

    cell_means_lq = {
        "treated_pre": float(work.loc[(g == 1) & (p == 0), "log_quantity"].mean()),
        "treated_post": float(work.loc[(g == 1) & (p == 1), "log_quantity"].mean()),
        "control_pre": float(work.loc[(g == 0) & (p == 0), "log_quantity"].mean()),
        "control_post": float(work.loc[(g == 0) & (p == 1), "log_quantity"].mean()),
    }
    cell_means_q = {
        "treated_pre": float(work.loc[(g == 1) & (p == 0), "quantity"].mean()),
        "treated_post": float(work.loc[(g == 1) & (p == 1), "quantity"].mean()),
        "control_pre": float(work.loc[(g == 0) & (p == 0), "quantity"].mean()),
        "control_post": float(work.loc[(g == 0) & (p == 1), "quantity"].mean()),
    }

    # Pre-treatment parallel trends test (linear differential trend on post == 0)
    pre_df = work.loc[work["_post"] == 0].copy()
    ts_all = pd.to_datetime(work[timestamp_col])
    post_start = ts_all.loc[work["_post"] == 1].min()
    pre_days = (pd.to_datetime(pre_df[timestamp_col]) - post_start).dt.total_seconds() / (86400.0 * 30.0)
    pre_df["_pre_time"] = pre_days
    pre_df["_treat_x_time"] = pre_df["_treated"] * pre_df["_pre_time"]

    X_pre = sm.add_constant(pre_df[["_treated", "_pre_time", "_treat_x_time"]], has_constant="add")
    pre_fit = sm.OLS(pre_df["log_quantity"].to_numpy(dtype=float), X_pre).fit(cov_type="HC3")
    pre_slope = float(pre_fit.params["_treat_x_time"])
    pre_p = float(pre_fit.pvalues["_treat_x_time"])
    parallel_plausible = bool(pre_p > 0.05)

    # Event-study binned around post_start (e.g., ~30-day bins)
    rel_months = np.floor(((ts_all - post_start).dt.total_seconds() / (86400.0 * 30.0))).astype(int)
    half_bins = max(2, n_event_bins // 2)
    rel_months = np.clip(rel_months, -half_bins, half_bins - 1)
    work["_event_bin"] = rel_months

    unique_bins = sorted(b for b in work["_event_bin"].unique() if b != -1)
    es_cols: list[str] = []
    es_data: dict[str, np.ndarray] = {"_treated": work["_treated"].to_numpy()}
    for b in unique_bins:
        bin_ind = (work["_event_bin"] == b).to_numpy(dtype=float)
        es_data[f"_bin_{b}"] = bin_ind
        col_name = f"_treat_x_bin_{b}"
        es_data[col_name] = work["_treated"].to_numpy() * bin_ind
        es_cols.append((b, col_name))

    X_es = pd.concat([pd.DataFrame(es_data, index=work.index), X_df[[c for c in keep if c not in ("_treated", "_post", "_did_interaction")]]], axis=1)
    X_es = sm.add_constant(X_es, has_constant="add")
    es_fit = sm.OLS(work["log_quantity"].to_numpy(dtype=float), X_es).fit(cov_type="HC3")

    es_rows = [{"relative_period": -1, "estimate": 0.0, "std_error": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0}]
    for b, col_name in es_cols:
        est = float(es_fit.params[col_name])
        se_b = float(es_fit.bse[col_name])
        ci_b = es_fit.conf_int().loc[col_name]
        es_rows.append(
            {
                "relative_period": int(b),
                "estimate": est,
                "std_error": se_b,
                "ci_low": float(ci_b.iloc[0]),
                "ci_high": float(ci_b.iloc[1]),
                "p_value": float(es_fit.pvalues[col_name]),
            }
        )
    event_study_df = pd.DataFrame(es_rows).sort_values("relative_period").reset_index(drop=True)

    note = (
        f"DiD log(Q) ATE={ate_lq:.4f} (p={p_lq:.3e}); implied price elasticity={implied_elast:.4f}. "
        f"Pre-trend differential slope={pre_slope:.4f} (p={pre_p:.3f}). "
        "Note: Non-significant pre-trend divergence is consistent with parallel trends, NOT formal proof."
        if implied_elast is not None
        else f"DiD log(Q) ATE={ate_lq:.4f} (p={p_lq:.3e}). Pre-trend slope p={pre_p:.3f}."
    )

    return DiDResult(
        ate_log_quantity=ate_lq,
        std_error_log_q=se_lq,
        ci_low_log_q=float(ci_lq.iloc[0]),
        ci_high_log_q=float(ci_lq.iloc[1]),
        p_value_log_q=p_lq,
        ate_quantity=ate_q,
        std_error_qty=se_q,
        ci_low_qty=float(ci_q.iloc[0]),
        ci_high_qty=float(ci_q.iloc[1]),
        did_price_change_log=did_lp,
        implied_did_elasticity=implied_elast,
        cell_means_log_q=cell_means_lq,
        cell_means_qty=cell_means_q,
        pre_trend_diff_slope=pre_slope,
        pre_trend_p_value=pre_p,
        parallel_trends_plausible=parallel_plausible,
        event_study_df=event_study_df,
        n_obs=len(work),
        assumptions_note=note,
    )


if __name__ == "__main__":
    raw_df = generate_synthetic_marketplace_data(n_rows=30_000, seed=42)
    cleaned_df, _ = clean_transaction_data(raw_df)
    feat_df = add_engineered_features(cleaned_df)
    res = estimate_difference_in_differences(feat_df)
    print("=== Difference-in-Differences Estimation ===")
    print(f"DiD ATE (log Q):    {res.ate_log_quantity:.4f} (95% CI=[{res.ci_low_log_q:.4f}, {res.ci_high_log_q:.4f}])")
    print(f"DiD First-Stage dP: {res.did_price_change_log:.4f}")
    if res.implied_did_elasticity is not None:
        print(f"Implied Elasticity: {res.implied_did_elasticity:.4f}")
    if "true_elasticity" in feat_df.columns:
        treated_true_el = feat_df.loc[feat_df["did_group"] == 1, "true_elasticity"].mean()
        print(f"True Treated Elast: {treated_true_el:.4f}")
    print(f"Pre-trend slope:    {res.pre_trend_diff_slope:.5f} (p={res.pre_trend_p_value:.4f})")
