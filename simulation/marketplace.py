"""Counterfactual marketplace simulator and public dataset (Olist) adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config.loader import get_logger
from simulation.ground_truth import DEFAULT_GROUND_TRUTH, compute_total_structural_log_q

logger = get_logger(__name__)

# Columns that represent ground truth or post-decision outcomes and MUST NOT be used as features
GROUND_TRUTH_COLUMNS = (
    "true_elasticity",
    "true_propensity",
    "true_q0",
    "true_q1",
    "true_cate_qty",
    "true_cate_revenue",
    "true_cate_margin",
    "exog_demand_index",
)


def evaluate_counterfactual_price_action(
    df: pd.DataFrame,
    price_change_pct: np.ndarray | float,
) -> pd.DataFrame:
    """Evaluate exact structural counterfactual outcomes on synthetic data for a price change.

    Strictly intended as an oracle benchmark for synthetic validation; real observational
    datasets lack `true_elasticity` and `exog_demand_index`.
    """
    required = {"price", "base_price", "competitor_price", "unit_cost", "true_elasticity", "exog_demand_index"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Counterfactual evaluation requires synthetic ground-truth columns: {sorted(missing)}")

    delta = np.asarray(price_change_pct, dtype=float)
    if delta.ndim == 0:
        delta = np.full(len(df), float(delta))

    new_price = np.maximum(df["price"].to_numpy(dtype=float) * (1.0 + delta), 1.0)
    # Promotional lift applies when price is cut by >= 5% or originally promoted
    promo = np.where(delta <= -0.05, 1, df.get("festival_flag", pd.Series(0, index=df.index)).to_numpy())

    log_q_cf = compute_total_structural_log_q(
        exogenous_demand_index=df["exog_demand_index"].to_numpy(dtype=float),
        price=new_price,
        reference_price=df["base_price"].to_numpy(dtype=float),
        competitor_price=df["competitor_price"].to_numpy(dtype=float),
        total_elasticity=df["true_elasticity"].to_numpy(dtype=float),
        promotion_flag=promo,
        spec=DEFAULT_GROUND_TRUTH,
    )
    ln_corr = float(np.exp(0.5 * (0.14**2)))
    expected_q = np.exp(log_q_cf) * ln_corr
    unit_cost = df["unit_cost"].to_numpy(dtype=float)
    expected_rev = new_price * expected_q
    expected_cost = unit_cost * expected_q
    expected_margin = expected_rev - expected_cost

    return pd.DataFrame(
        {
            "cf_price": new_price,
            "cf_quantity": expected_q,
            "cf_revenue": expected_rev,
            "cf_cost": expected_cost,
            "cf_margin": expected_margin,
        },
        index=df.index,
    )


def describe_olist_causal_support() -> dict[str, object]:
    """Document clearly which causal methods are supported by the public Olist dataset."""
    return {
        "dataset": "Olist Brazilian E-Commerce Dataset",
        "supported": [
            "Exploratory Data Analysis",
            "Naive / Controlled Price Elasticity Estimation (correlational)",
            "Confounding & Simpson's Paradox Analysis across Brazilian states/categories",
            "Chronological Backtesting of predictive pricing heuristics",
        ],
        "not_supported_without_external_data": [
            "Valid Randomized Uplift Evaluation (no randomized discount assignment column)",
            "Excluded Instrumental Variables (freight_value correlates with distance/product weight/demand)",
            "Sharp DiD Policy Evaluation (no documented exogenous regional pricing rollout date)",
        ],
    }


def load_olist_dataset(olist_dir: str | Path) -> pd.DataFrame:
    """Load and normalize the Olist Brazilian E-Commerce CSV files into the canonical schema.

    Expects standard Kaggle Olist files in `olist_dir`:
    - olist_orders_dataset.csv
    - olist_order_items_dataset.csv
    - olist_customers_dataset.csv
    - olist_products_dataset.csv
    """
    base = Path(olist_dir)
    required_files = [
        "olist_orders_dataset.csv",
        "olist_order_items_dataset.csv",
        "olist_customers_dataset.csv",
        "olist_products_dataset.csv",
    ]
    missing = [f for f in required_files if not (base / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Olist dataset files missing in {base}: {missing}. "
            "Provide the Kaggle Olist CSVs or use the synthetic marketplace generator."
        )

    orders = pd.read_csv(base / "olist_orders_dataset.csv")
    items = pd.read_csv(base / "olist_order_items_dataset.csv")
    customers = pd.read_csv(base / "olist_customers_dataset.csv")
    products = pd.read_csv(base / "olist_products_dataset.csv")

    merged = (
        items.merge(orders, on="order_id", how="inner")
        .merge(customers, on="customer_id", how="left")
        .merge(products, on="product_id", how="left")
    )
    merged["timestamp"] = pd.to_datetime(merged["order_purchase_timestamp"], errors="coerce")
    merged = merged.dropna(subset=["timestamp", "price"]).sort_values("timestamp").reset_index(drop=True)

    # Aggregate order-item rows to order-product level so quantity represents units per order line
    grouped = (
        merged.groupby(["order_id", "product_id"], as_index=False)
        .agg(
            customer_id=("customer_unique_id", "first"),
            region=("customer_state", "first"),
            timestamp=("timestamp", "first"),
            price=("price", "mean"),
            freight_value=("freight_value", "mean"),
            quantity=("order_item_id", "count"),
            category=("product_category_name", "first"),
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    grouped["category"] = grouped["category"].fillna("unknown")
    grouped["region"] = grouped["region"].fillna("unknown")
    grouped["historical_orders"] = grouped.groupby("customer_id").cumcount()
    grouped["customer_cohort"] = np.where(
        grouped["historical_orders"] == 0,
        "New",
        np.where(grouped["historical_orders"] < 3, "Returning", "Loyal"),
    )
    # Approximate proxy columns clearly flagged as observational approximations
    grouped["discount"] = 0.0
    grouped["promotion_flag"] = 0
    cat_median_price = grouped.groupby("category")["price"].transform("median")
    grouped["competitor_price"] = cat_median_price
    grouped["revenue"] = grouped["price"] * grouped["quantity"]
    grouped["unit_cost"] = 0.60 * grouped["price"] + grouped["freight_value"].fillna(0.0)
    grouped["cost"] = grouped["unit_cost"] * grouped["quantity"]
    grouped["margin"] = grouped["revenue"] - grouped["cost"]

    logger.info("Loaded Olist dataset with %d normalized order-product rows.", len(grouped))
    return grouped


def load_or_generate_dataset(
    source: str = "synthetic",
    n_rows: int = 50_000,
    seed: int = 42,
    raw_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Unified dataset loader supporting synthetic generation or Olist CSVs."""
    if source.lower() == "olist":
        return load_olist_dataset(Path(raw_dir) / "olist")
    from simulation.data_generator import generate_synthetic_marketplace_data

    return generate_synthetic_marketplace_data(n_rows=n_rows, seed=seed)
