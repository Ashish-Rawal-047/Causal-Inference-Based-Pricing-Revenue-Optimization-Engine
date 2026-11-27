"""Naive and controlled price elasticity estimation with regression diagnostics."""

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
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera

from config.loader import get_logger
from preprocessing.cleaning import clean_transaction_data
from preprocessing.feature_engineering import add_engineered_features
from simulation.data_generator import generate_synthetic_marketplace_data

logger = get_logger(__name__)


@dataclass
class ElasticityResult:
    """Structured result for an OLS/log-log price elasticity regression."""

    model_name: str
    elasticity: float
    std_error: float
    ci_low: float
    ci_high: float
    p_value: float
    r_squared: float
    adj_r_squared: float
    n_obs: int
    residual_std: float
    residual_mean: float
    bp_p_value: float | None = None
    jb_p_value: float | None = None
    max_vif: float | None = None
    notes: str = ""


def _build_design_matrix(
    df: pd.DataFrame,
    price_col: str,
    control_numeric: Sequence[str],
    control_categorical: Sequence[str],
    use_log_price: bool = True,
) -> pd.DataFrame:
    """Construct regression design matrix with constant, price term, and controls."""
    parts: list[pd.DataFrame] = []
    p_vals = pd.to_numeric(df[price_col], errors="coerce").to_numpy(dtype=float)
    if np.any(p_vals <= 0):
        raise ValueError("Price column contains non-positive values; cannot compute log-price elasticity.")

    price_feature = np.log(p_vals) if use_log_price else p_vals
    price_name = "log_price" if use_log_price else "price"
    parts.append(pd.DataFrame({price_name: price_feature}, index=df.index))

    avail_num = [c for c in control_numeric if c in df.columns]
    if avail_num:
        parts.append(df[avail_num].astype(float))

    avail_cat = [c for c in control_categorical if c in df.columns and df[c].nunique() > 1]
    if avail_cat:
        dummies = pd.get_dummies(df[avail_cat].astype(str), drop_first=True, dtype=float)
        parts.append(dummies)

    x_df = pd.concat(parts, axis=1)
    # Drop zero-variance columns before adding constant
    non_const = [c for c in x_df.columns if x_df[c].std() > 1e-12]
    x_df = x_df[non_const]
    return sm.add_constant(x_df, has_constant="add")


def _fit_ols_elasticity(
    df: pd.DataFrame,
    model_name: str,
    control_numeric: Sequence[str] = (),
    control_categorical: Sequence[str] = (),
    use_log: bool = True,
    compute_vif: bool = False,
) -> ElasticityResult:
    """Fit OLS elasticity regression with HC3 heteroskedasticity-robust standard errors."""
    if len(df) < 15:
        raise ValueError(f"Insufficient observations ({len(df)}) for elasticity estimation '{model_name}'.")
    if df["price"].nunique() < 2 or df["price"].std() < 1e-9:
        raise ValueError(f"Insufficient price variation for elasticity estimation '{model_name}'.")

    q_vals = pd.to_numeric(df["quantity"], errors="coerce").to_numpy(dtype=float)
    if np.any(q_vals <= 0):
        raise ValueError("Quantity column contains non-positive values; cannot take log(quantity).")

    y = np.log(q_vals) if use_log else q_vals
    X = _build_design_matrix(
        df,
        price_col="price",
        control_numeric=control_numeric,
        control_categorical=control_categorical,
        use_log_price=use_log,
    )
    price_key = "log_price" if use_log else "price"

    model = sm.OLS(y, X).fit(cov_type="HC3")
    raw_coef = float(model.params[price_key])
    se = float(model.bse[price_key])
    ci = model.conf_int().loc[price_key]
    p_val = float(model.pvalues[price_key])

    # For linear Q = a + b*P, convert slope b to point elasticity at sample means: b * (mean(P)/mean(Q))
    if use_log:
        elasticity = raw_coef
        se_el = se
        ci_low, ci_high = float(ci.iloc[0]), float(ci.iloc[1])
    else:
        scale = float(df["price"].mean() / df["quantity"].mean())
        elasticity = raw_coef * scale
        se_el = se * scale
        ci_low = float(ci.iloc[0]) * scale
        ci_high = float(ci.iloc[1]) * scale

    resid = np.asarray(model.resid, dtype=float)
    try:
        _, bp_p, _, _ = het_breuschpagan(resid, X)
        bp_p_val: float | None = float(bp_p)
    except Exception:
        bp_p_val = None

    try:
        _, jb_p, _, _ = jarque_bera(resid)
        jb_p_val: float | None = float(jb_p)
    except Exception:
        jb_p_val = None

    max_vif: float | None = None
    if compute_vif and X.shape[1] > 2:
        x_vals = X.to_numpy(dtype=float)
        vifs = []
        for idx in range(1, min(x_vals.shape[1], 25)):
            try:
                v = float(variance_inflation_factor(x_vals, idx))
                if np.isfinite(v):
                    vifs.append(v)
            except Exception:
                continue
        if vifs:
            max_vif = float(max(vifs))

    return ElasticityResult(
        model_name=model_name,
        elasticity=elasticity,
        std_error=se_el,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_val,
        r_squared=float(model.rsquared),
        adj_r_squared=float(model.rsquared_adj),
        n_obs=int(model.nobs),
        residual_std=float(np.std(resid, ddof=1)),
        residual_mean=float(np.mean(resid)),
        bp_p_value=bp_p_val,
        jb_p_value=jb_p_val,
        max_vif=max_vif,
        notes=(
            "Linear demand specification converted to elasticity at sample means; not causal."
            if not use_log
            else "OLS log-log specification; subject to omitted-variable / simultaneity bias if price is endogenous."
        ),
    )


def estimate_linear_elasticity(df: pd.DataFrame) -> ElasticityResult:
    """Estimate baseline linear demand regression Q = b0 + b1*P + e, converted to mean elasticity."""
    return _fit_ols_elasticity(df, model_name="Linear OLS (Mean Elasticity)", use_log=False)


def estimate_naive_loglog_elasticity(df: pd.DataFrame) -> ElasticityResult:
    """Estimate uncontrolled log-log demand regression: log(Q) = b0 + b1*log(P) + e."""
    return _fit_ols_elasticity(df, model_name="Naive Log-Log OLS", use_log=True)


def estimate_controlled_loglog_elasticity(df: pd.DataFrame, compute_vif: bool = True) -> ElasticityResult:
    """Estimate log-log regression with observable segment, calendar, and competition controls."""
    num_controls = [
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
    cat_controls = [c for c in ("region", "category", "customer_cohort") if c in df.columns]
    return _fit_ols_elasticity(
        df,
        model_name="Controlled Log-Log OLS",
        control_numeric=num_controls,
        control_categorical=cat_controls,
        use_log=True,
        compute_vif=compute_vif,
    )


def estimate_segment_elasticities(
    df: pd.DataFrame,
    segment_cols: Sequence[str] = ("region", "customer_cohort", "category", "order_frequency_group"),
    controlled: bool = True,
) -> pd.DataFrame:
    """Estimate within-segment price elasticity across requested grouping columns."""
    rows: list[dict[str, Any]] = []
    base_num_controls = [
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

    for seg_col in segment_cols:
        if seg_col not in df.columns:
            continue
        for seg_val, sub in df.groupby(seg_col, observed=True):
            if len(sub) < 30 or sub["price"].nunique() < 2:
                continue
            cat_ctrl = [c for c in ("region", "category", "customer_cohort") if c in sub.columns and c != seg_col]
            res = _fit_ols_elasticity(
                sub,
                model_name=f"{seg_col}={seg_val}",
                control_numeric=base_num_controls if controlled else (),
                control_categorical=cat_ctrl if controlled else (),
                use_log=True,
            )
            row: dict[str, Any] = {
                "segment_type": seg_col,
                "segment_value": str(seg_val),
                "elasticity": res.elasticity,
                "std_error": res.std_error,
                "ci_low": res.ci_low,
                "ci_high": res.ci_high,
                "p_value": res.p_value,
                "r_squared": res.r_squared,
                "n_obs": res.n_obs,
            }
            if "true_elasticity" in sub.columns:
                row["true_elasticity"] = float(sub["true_elasticity"].mean())
                row["bias"] = res.elasticity - float(sub["true_elasticity"].mean())
            rows.append(row)

    return pd.DataFrame(rows)


def run_elasticity_comparison(df: pd.DataFrame) -> dict[str, Any]:
    """Run all baseline elasticity specifications and segment breakdowns."""
    linear_res = estimate_linear_elasticity(df)
    naive_res = estimate_naive_loglog_elasticity(df)
    ctrl_res = estimate_controlled_loglog_elasticity(df)
    seg_df = estimate_segment_elasticities(df, controlled=True)
    seg_naive_df = estimate_segment_elasticities(df, controlled=False)

    summary_df = pd.DataFrame(
        [
            linear_res.__dict__,
            naive_res.__dict__,
            ctrl_res.__dict__,
        ]
    )
    if "true_elasticity" in df.columns:
        summary_df["true_avg_elasticity"] = float(df["true_elasticity"].mean())
        summary_df["bias_vs_true"] = summary_df["elasticity"] - summary_df["true_avg_elasticity"]

    return {
        "linear": linear_res,
        "naive_loglog": naive_res,
        "controlled_loglog": ctrl_res,
        "summary_table": summary_df,
        "segment_controlled": seg_df,
        "segment_naive": seg_naive_df,
    }


if __name__ == "__main__":
    raw_df = generate_synthetic_marketplace_data(n_rows=20_000, seed=42)
    cleaned_df, _ = clean_transaction_data(raw_df)
    feat_df = add_engineered_features(cleaned_df)
    res = run_elasticity_comparison(feat_df)
    print("=== Baseline Price Elasticity Comparison ===")
    cols = ["model_name", "elasticity", "std_error", "ci_low", "ci_high", "r_squared"]
    if "true_avg_elasticity" in res["summary_table"].columns:
        cols.extend(["true_avg_elasticity", "bias_vs_true"])
    print(res["summary_table"][cols].to_string(index=False))
    print("\n=== Controlled Segment Elasticities (Region) ===")
    print(res["segment_controlled"].query("segment_type == 'region'").to_string(index=False))
