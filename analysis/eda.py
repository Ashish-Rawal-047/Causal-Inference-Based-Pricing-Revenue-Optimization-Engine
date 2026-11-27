"""Exploratory Data Analysis (EDA) for marketplace pricing transactions."""

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
from preprocessing.feature_engineering import add_engineered_features
from simulation.data_generator import generate_synthetic_marketplace_data

logger = get_logger(__name__)


def run_eda(df: pd.DataFrame) -> dict[str, Any]:
    """Compute comprehensive exploratory statistics across transactions and segments."""
    if df.empty:
        raise ValueError("Cannot run EDA on an empty DataFrame.")

    numeric_cols = [
        c
        for c in (
            "price",
            "discount",
            "quantity",
            "revenue",
            "cost",
            "margin",
            "competitor_price",
            "historical_orders",
            "seller_availability",
        )
        if c in df.columns
    ]
    summary_stats = df[numeric_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]]
    summary_stats = summary_stats.rename(columns={"50%": "median"})

    by_region = (
        df.groupby("region", as_index=False)
        .agg(
            transactions=("order_id", "count"),
            avg_price=("price", "mean"),
            avg_discount=("discount", "mean"),
            avg_quantity=("quantity", "mean"),
            total_revenue=("revenue", "sum"),
            total_margin=("margin", "sum"),
            promo_rate=("promotion_flag", "mean"),
        )
        .sort_values("total_revenue", ascending=False)
    )

    by_cohort = (
        df.groupby("customer_cohort", as_index=False)
        .agg(
            transactions=("order_id", "count"),
            avg_price=("price", "mean"),
            avg_discount=("discount", "mean"),
            avg_quantity=("quantity", "mean"),
            total_revenue=("revenue", "sum"),
            total_margin=("margin", "sum"),
        )
        .sort_values("total_revenue", ascending=False)
    )

    by_category = (
        df.groupby("category", as_index=False)
        .agg(
            transactions=("order_id", "count"),
            avg_price=("price", "mean"),
            avg_discount=("discount", "mean"),
            avg_quantity=("quantity", "mean"),
            total_revenue=("revenue", "sum"),
            total_margin=("margin", "sum"),
        )
        .sort_values("total_revenue", ascending=False)
    )

    ts = pd.to_datetime(df["timestamp"])
    monthly = (
        df.assign(year_month=ts.dt.to_period("M").astype(str))
        .groupby("year_month", as_index=False)
        .agg(
            transactions=("order_id", "count"),
            avg_price=("price", "mean"),
            avg_quantity=("quantity", "mean"),
            total_revenue=("revenue", "sum"),
            total_margin=("margin", "sum"),
        )
    )

    missing_summary = {col: int(cnt) for col, cnt in df.isna().sum().items() if cnt > 0}

    return {
        "n_rows": int(len(df)),
        "n_customers": int(df["customer_id"].nunique()),
        "n_products": int(df["product_id"].nunique()),
        "date_min": str(ts.min()),
        "date_max": str(ts.max()),
        "total_revenue": float(df["revenue"].sum()),
        "total_margin": float(df["margin"].sum()),
        "overall_margin_pct": float(df["margin"].sum() / max(df["revenue"].sum(), 1e-9)),
        "missing_counts": missing_summary,
        "summary_stats": summary_stats,
        "by_region": by_region,
        "by_cohort": by_cohort,
        "by_category": by_category,
        "monthly_trends": monthly,
    }


if __name__ == "__main__":
    raw_df = generate_synthetic_marketplace_data(n_rows=20_000, seed=42)
    cleaned_df, _ = clean_transaction_data(raw_df)
    feat_df = add_engineered_features(cleaned_df)
    eda_results = run_eda(feat_df)
    print("=== EDA Summary ===")
    print(f"Rows: {eda_results['n_rows']}, Customers: {eda_results['n_customers']}, Products: {eda_results['n_products']}")
    print(f"Total Revenue: {eda_results['total_revenue']:,.2f}, Total Margin: {eda_results['total_margin']:,.2f}")
    print("\nBy Region:")
    print(eda_results["by_region"].to_string(index=False))
