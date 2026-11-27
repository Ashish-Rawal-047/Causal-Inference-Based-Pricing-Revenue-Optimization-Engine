"""Data cleaning, missing value handling, deduplication, and outlier control."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from config.loader import get_logger
from preprocessing.validation import DataValidationError, validate_transaction_schema

logger = get_logger(__name__)


@dataclass
class CleaningSummary:
    """Audit summary of cleaning operations."""

    initial_rows: int
    duplicates_removed: int
    invalid_rows_removed: int
    missing_values_imputed: int
    outliers_clipped: int
    final_rows: int


def clean_transaction_data(
    df: pd.DataFrame,
    outlier_quantile_low: float = 0.001,
    outlier_quantile_high: float = 0.999,
    strict_invalid: bool = False,
) -> tuple[pd.DataFrame, CleaningSummary]:
    """Clean raw transaction data while preserving valid causal structure.

    Steps:
    1. Validate schema presence.
    2. Parse timestamp and sort chronologically.
    3. Remove duplicate order_ids (keep earliest occurrence).
    4. Remove or raise on invalid prices (<= 0), quantities (<= 0), or discounts (< 0 or >= 1).
    5. Impute missing non-critical features (competitor_price, seller_availability, categoricals).
    6. Winsorize extreme continuous price/quantity outliers at configured quantiles.
    7. Recompute revenue, cost, margin consistently if needed.
    """
    if df is None or df.empty:
        raise DataValidationError("Cannot clean an empty DataFrame.")

    validate_transaction_schema(df, strict=False)
    cleaned = df.copy()
    initial_rows = len(cleaned)

    # Parse timestamps and numeric types
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce")
    for col in ("price", "discount", "quantity", "competitor_price", "historical_orders", "revenue", "cost", "margin"):
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # Sort chronologically first for deterministic deduplication & rolling features
    cleaned = cleaned.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    before_dedup = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=["order_id"], keep="first").reset_index(drop=True)
    duplicates_removed = before_dedup - len(cleaned)

    # Check invalid core numerical domains
    invalid_mask = (
        cleaned["price"].isna()
        | (cleaned["price"] <= 0)
        | cleaned["quantity"].isna()
        | (cleaned["quantity"] <= 0)
        | cleaned["discount"].isna()
        | (cleaned["discount"] < 0)
        | (cleaned["discount"] >= 1.0)
    )
    n_invalid = int(invalid_mask.sum())
    if strict_invalid and n_invalid > 0:
        raise DataValidationError(f"Found {n_invalid} invalid price/quantity/discount rows in strict mode.")

    cleaned = cleaned.loc[~invalid_mask].reset_index(drop=True)
    invalid_rows_removed = (initial_rows - duplicates_removed) - len(cleaned)
    if cleaned.empty:
        raise DataValidationError("All rows were removed during cleaning; no valid transactions remain.")

    # Impute missing non-key fields safely
    missing_imputed = 0
    for cat_col in ("region", "category", "customer_cohort"):
        if cat_col in cleaned.columns and cleaned[cat_col].isna().any():
            n_miss = int(cleaned[cat_col].isna().sum())
            cleaned[cat_col] = cleaned[cat_col].fillna("Unknown")
            missing_imputed += n_miss

    if "competitor_price" in cleaned.columns and cleaned["competitor_price"].isna().any():
        n_miss = int(cleaned["competitor_price"].isna().sum())
        cat_med = cleaned.groupby("category")["price"].transform("median")
        cleaned["competitor_price"] = cleaned["competitor_price"].fillna(cat_med).fillna(cleaned["price"].median())
        missing_imputed += n_miss

    for num_col in ("historical_orders", "promotion_flag", "seller_availability"):
        if num_col in cleaned.columns and cleaned[num_col].isna().any():
            n_miss = int(cleaned[num_col].isna().sum())
            fill_val = 0 if num_col in ("historical_orders", "promotion_flag") else float(cleaned[num_col].median())
            cleaned[num_col] = cleaned[num_col].fillna(fill_val)
            missing_imputed += n_miss

    # Winsorize extreme price & quantity tails without distorting core distribution
    outliers_clipped = 0
    if len(cleaned) >= 100 and 0.0 < outlier_quantile_low < outlier_quantile_high < 1.0:
        for col in ("price", "quantity"):
            lo, hi = cleaned[col].quantile([outlier_quantile_low, outlier_quantile_high])
            lo = max(float(lo), 1e-3)
            clipped_mask = (cleaned[col] < lo) | (cleaned[col] > hi)
            outliers_clipped += int(clipped_mask.sum())
            cleaned[col] = cleaned[col].clip(lower=lo, upper=float(hi))

    cleaned["promotion_flag"] = cleaned["promotion_flag"].astype(int)
    cleaned["historical_orders"] = cleaned["historical_orders"].astype(int)

    # Recompute revenue & margin if outliers were clipped or missing
    if outliers_clipped > 0 or cleaned["revenue"].isna().any() or cleaned["margin"].isna().any():
        cleaned["revenue"] = np.round(cleaned["price"] * cleaned["quantity"], 2)
        if "unit_cost" in cleaned.columns:
            cleaned["cost"] = np.round(cleaned["unit_cost"] * cleaned["quantity"], 2)
        elif cleaned["cost"].isna().any():
            cleaned["cost"] = np.round(0.55 * cleaned["price"] * cleaned["quantity"], 2)
        cleaned["margin"] = np.round(cleaned["revenue"] - cleaned["cost"], 2)

    summary = CleaningSummary(
        initial_rows=initial_rows,
        duplicates_removed=duplicates_removed,
        invalid_rows_removed=invalid_rows_removed,
        missing_values_imputed=missing_imputed,
        outliers_clipped=outliers_clipped,
        final_rows=len(cleaned),
    )
    logger.info(
        "Cleaned dataset: %d -> %d rows (dups=%d, invalid=%d, imputed=%d, clipped=%d)",
        summary.initial_rows,
        summary.final_rows,
        summary.duplicates_removed,
        summary.invalid_rows_removed,
        summary.missing_values_imputed,
        summary.outliers_clipped,
    )
    return cleaned, summary
