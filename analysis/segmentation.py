"""Customer and marketplace segmentation profiling."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_segment_profiles(
    df: pd.DataFrame,
    segment_col: str = "region",
) -> pd.DataFrame:
    """Profile segments across volume, pricing, discount intensity, and unit economics."""
    if segment_col not in df.columns:
        raise ValueError(f"Segment column '{segment_col}' not found in DataFrame.")

    agg_dict: dict[str, tuple[str, str]] = {
        "n_transactions": ("order_id", "count"),
        "n_customers": ("customer_id", "nunique"),
        "avg_price": ("price", "mean"),
        "avg_discount": ("discount", "mean"),
        "avg_quantity": ("quantity", "mean"),
        "avg_revenue": ("revenue", "mean"),
        "avg_margin": ("margin", "mean"),
        "total_revenue": ("revenue", "sum"),
        "total_margin": ("margin", "sum"),
        "promo_share": ("promotion_flag", "mean"),
    }
    if "true_elasticity" in df.columns:
        agg_dict["true_elasticity_mean"] = ("true_elasticity", "mean")
    if "true_cate_margin" in df.columns:
        agg_dict["true_cate_margin_mean"] = ("true_cate_margin", "mean")

    prof = df.groupby(segment_col, observed=True, as_index=False).agg(**agg_dict)
    prof["margin_pct"] = prof["total_margin"] / np.maximum(prof["total_revenue"], 1e-9)
    return prof.sort_values("total_revenue", ascending=False).reset_index(drop=True)
