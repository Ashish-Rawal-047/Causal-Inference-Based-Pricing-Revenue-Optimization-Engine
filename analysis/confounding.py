"""Confounding analysis, Simpson's Paradox demonstration, and causal DAG specification."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from analysis.elasticity import (
    estimate_controlled_loglog_elasticity,
    estimate_naive_loglog_elasticity,
    estimate_segment_elasticities,
)


def build_marketplace_dag() -> nx.DiGraph:
    """Construct the structural causal DAG for the marketplace pricing environment.

    Nodes capture observed confounders, unobserved demand shocks (`UnobservedDemand`),
    the excluded cost instrument (`SupplierCostShock`), `Price`, and `Demand`.
    """
    dag = nx.DiGraph()
    edges = [
        ("Region", "Price"),
        ("Region", "Demand"),
        ("Category", "Price"),
        ("Category", "Demand"),
        ("CustomerCohort", "Discount"),
        ("CustomerCohort", "Demand"),
        ("CalendarSeasonality", "Price"),
        ("CalendarSeasonality", "Demand"),
        ("PromotionFlag", "Discount"),
        ("PromotionFlag", "Demand"),
        ("SellerAvailability", "Price"),
        ("SellerAvailability", "Demand"),
        ("HistoricalOrders", "Discount"),
        ("HistoricalOrders", "Demand"),
        ("CompetitorPrice", "Demand"),
        ("UnobservedDemand", "Price"),
        ("UnobservedDemand", "Demand"),
        ("SupplierCostShock_IV", "Price"),
        ("Discount", "Price"),
        ("Price", "Demand"),
    ]
    dag.add_edges_from(edges)
    return dag


def analyze_correlations_and_confounders(df: pd.DataFrame) -> dict[str, Any]:
    """Identify candidate confounders via correlation with both price and demand.

    Note: Correlation with both treatment (price) and outcome (quantity) is a necessary
    screening signal for confounding, NOT proof that a variable is a valid confounder
    (e.g., colliders or mediators can also correlate with both).
    """
    candidate_num = [
        c
        for c in (
            "log_competitor_price",
            "promotion_flag",
            "discount",
            "is_weekend",
            "month",
            "hour_of_day",
            "log_historical_orders",
            "seller_availability",
            "festival_flag",
            "supplier_cost_shock",
        )
        if c in df.columns
    ]
    corr_cols = ["log_price", "log_quantity"] + candidate_num
    corr_matrix = df[corr_cols].corr()

    confounder_screen = []
    for col in candidate_num:
        r_price = float(corr_matrix.loc[col, "log_price"])
        r_qty = float(corr_matrix.loc[col, "log_quantity"])
        confounder_screen.append(
            {
                "variable": col,
                "corr_with_log_price": r_price,
                "corr_with_log_quantity": r_qty,
                "abs_product": abs(r_price * r_qty),
                "role_note": (
                    "Excluded cost instrument candidate (correlates with price, near-zero direct demand corr)"
                    if col == "supplier_cost_shock"
                    else "Potential confounder / covariate (requires domain DAG justification)"
                ),
            }
        )
    screen_df = pd.DataFrame(confounder_screen).sort_values("abs_product", ascending=False).reset_index(drop=True)

    # Predictive feature importance diagnostic (explicitly NOT causal importance)
    pred_cols = [c for c in candidate_num if c != "supplier_cost_shock"] + ["log_price"]
    X_pred = df[pred_cols].astype(float)
    y_pred = df["log_quantity"].to_numpy(dtype=float)
    gbr = GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=42)
    gbr.fit(X_pred, y_pred)
    feat_imp = (
        pd.DataFrame({"feature": pred_cols, "predictive_importance": gbr.feature_importances_})
        .sort_values("predictive_importance", ascending=False)
        .reset_index(drop=True)
    )

    naive_res = estimate_naive_loglog_elasticity(df)
    ctrl_res = estimate_controlled_loglog_elasticity(df)

    return {
        "correlation_matrix": corr_matrix,
        "confounder_screen": screen_df,
        "predictive_feature_importance": feat_imp,
        "before_controls_elasticity": naive_res.elasticity,
        "after_controls_elasticity": ctrl_res.elasticity,
        "control_shift": ctrl_res.elasticity - naive_res.elasticity,
        "diagnostic_warning": (
            "Predictive importance reflects statistical association in the training sample, "
            "not causal effect size. Unobserved demand shocks can still bias controlled OLS."
        ),
    }


def demonstrate_simpsons_paradox(
    df: pd.DataFrame,
    group_col: str = "region",
) -> dict[str, Any]:
    """Demonstrate Simpson's Paradox comparing pooled vs within-group price-demand slopes.

    In observational marketplaces where affluent/high-demand segments also pay higher prices,
    pooled log(Q) vs log(P) can appear flat or even positive while every individual segment
    (controlling for product mix/competitor reference prices) has a negative price elasticity.
    """
    pooled = estimate_naive_loglog_elasticity(df)
    within_uncontrolled = estimate_segment_elasticities(df, segment_cols=(group_col,), controlled=False)
    within_controlled = estimate_segment_elasticities(df, segment_cols=(group_col,), controlled=True)

    paradox_detected = bool(
        (pooled.elasticity > -0.25)
        and (within_controlled["elasticity"] < pooled.elasticity - 0.15).all()
        and (within_controlled["elasticity"] < 0).all()
    )
    sign_reversal = bool((pooled.elasticity >= 0) and (within_controlled["elasticity"] < 0).all())

    return {
        "group_col": group_col,
        "pooled_elasticity": pooled.elasticity,
        "pooled_ci": (pooled.ci_low, pooled.ci_high),
        "within_segment_elasticities": within_controlled,
        "within_segment_uncontrolled": within_uncontrolled,
        "paradox_detected": paradox_detected,
        "strict_sign_reversal": sign_reversal,
        "explanation": (
            f"Pooled naive log-log slope is {pooled.elasticity:+.3f}, whereas within-{group_col} slopes "
            f"(accounting for product reference/category mix) range from "
            f"{within_controlled['elasticity'].min():.3f} to {within_controlled['elasticity'].max():.3f}. "
            f"Aggregating across heterogeneous {group_col}s confounds price sensitivity with baseline demand levels."
        ),
    }
