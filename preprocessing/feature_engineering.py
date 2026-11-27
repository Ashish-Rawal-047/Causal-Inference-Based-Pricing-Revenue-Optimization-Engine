"""Leakage-free feature engineering for elasticity, causal inference, and uplift modeling."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from config.loader import get_logger
from preprocessing.validation import validate_no_leakage

logger = get_logger(__name__)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar, price, competitor, and strictly chronological historical features.

    All customer/product/region historical aggregates are computed strictly *prior* to each
    transaction timestamp (shifted expanding averages) to guarantee zero target leakage.
    """
    feat = df.copy()
    feat["timestamp"] = pd.to_datetime(feat["timestamp"])
    feat = feat.sort_values("timestamp").reset_index(drop=True)

    # Calendar / seasonality features
    feat["hour_of_day"] = feat["timestamp"].dt.hour
    feat["day_of_week"] = feat["timestamp"].dt.dayofweek
    feat["month"] = feat["timestamp"].dt.month
    feat["day_of_year"] = feat["timestamp"].dt.dayofyear
    feat["is_weekend"] = (feat["day_of_week"] >= 5).astype(int)
    feat["seasonality_sin"] = np.sin(2.0 * np.pi * feat["day_of_year"] / 365.0)
    feat["seasonality_cos"] = np.cos(2.0 * np.pi * feat["day_of_year"] / 365.0)

    # Price, discount, and competitor transformations
    feat["log_price"] = np.log(np.maximum(feat["price"].to_numpy(dtype=float), 1e-6))
    feat["log_quantity"] = np.log(np.maximum(feat["quantity"].to_numpy(dtype=float), 1e-6))
    feat["log_competitor_price"] = np.log(np.maximum(feat["competitor_price"].to_numpy(dtype=float), 1e-6))
    feat["price_to_competitor_ratio"] = feat["price"] / np.maximum(feat["competitor_price"], 1e-3)
    feat["competitor_price_diff"] = feat["price"] - feat["competitor_price"]
    feat["log_price_comp_gap"] = feat["log_price"] - feat["log_competitor_price"]
    feat["discount_pct"] = feat["discount"] * 100.0
    feat["has_discount"] = (feat["discount"] > 0.01).astype(int)

    # Order frequency bucket
    feat["order_frequency_group"] = pd.cut(
        feat["historical_orders"],
        bins=[-1, 1, 5, 12, 1_000_000],
        labels=["One-Time (0-1)", "Occasional (2-5)", "Regular (6-12)", "Power (13+)"],
    ).astype(str)
    feat["log_historical_orders"] = np.log1p(np.maximum(feat["historical_orders"].to_numpy(dtype=float), 0.0))

    # Strictly prior (leakage-free) expanding aggregates using shifted cumulative sums
    global_mean_qty = float(feat["quantity"].iloc[: max(1, min(200, len(feat) // 10))].mean())
    global_mean_price = float(feat["price"].iloc[: max(1, min(200, len(feat) // 10))].mean())

    for group_col, prefix in (("customer_id", "cust"), ("product_id", "prod"), ("region", "reg")):
        grp = feat.groupby(group_col, sort=False)
        prior_cnt = grp.cumcount()
        prior_qty_sum = grp["quantity"].cumsum() - feat["quantity"]
        prior_price_sum = grp["price"].cumsum() - feat["price"]

        feat[f"{prefix}_prior_tx_count"] = prior_cnt
        feat[f"{prefix}_prior_avg_quantity"] = np.where(
            prior_cnt > 0,
            prior_qty_sum / np.maximum(prior_cnt, 1),
            global_mean_qty,
        )
        feat[f"{prefix}_prior_avg_price"] = np.where(
            prior_cnt > 0,
            prior_price_sum / np.maximum(prior_cnt, 1),
            global_mean_price,
        )

    return feat


def build_feature_matrix(
    df: pd.DataFrame,
    include_price: bool = False,
    numeric_cols: Sequence[str] | None = None,
    categorical_cols: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build an encoded, leakage-validated feature matrix X for causal/uplift modeling.

    By default (`include_price=False`), `price` and `discount` are excluded when estimating
    treatment propensity or heterogeneous treatment effects of a pricing/discount treatment,
    because realized price/discount are downstream of treatment assignment.
    """
    default_num = [
        "historical_orders",
        "log_historical_orders",
        "competitor_price",
        "log_competitor_price",
        "hour_of_day",
        "day_of_week",
        "month",
        "is_weekend",
        "seasonality_sin",
        "seasonality_cos",
        "cust_prior_tx_count",
        "cust_prior_avg_quantity",
        "cust_prior_avg_price",
        "prod_prior_avg_quantity",
        "prod_prior_avg_price",
        "reg_prior_avg_quantity",
    ]
    if "seller_availability" in df.columns:
        default_num.append("seller_availability")
    if "festival_flag" in df.columns:
        default_num.append("festival_flag")
    if "base_price" in df.columns:
        default_num.append("base_price")
    if include_price:
        default_num.extend(["price", "log_price", "discount", "log_price_comp_gap"])

    num_cols = [c for c in (numeric_cols or default_num) if c in df.columns]
    cat_cols = [c for c in (categorical_cols or ["region", "category", "customer_cohort"]) if c in df.columns]

    validate_no_leakage(num_cols + cat_cols)

    x_num = df[num_cols].astype(float).copy()
    x_cat = pd.get_dummies(df[cat_cols].astype(str), drop_first=False, dtype=float)
    x_mat = pd.concat([x_num, x_cat], axis=1)
    validate_no_leakage(x_mat.columns)

    return x_mat, list(x_mat.columns)
