"""Schema, value, and target/temporal leakage validation for pricing datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "order_id",
    "customer_id",
    "product_id",
    "region",
    "timestamp",
    "price",
    "discount",
    "quantity",
    "category",
    "promotion_flag",
    "customer_cohort",
    "historical_orders",
    "competitor_price",
    "revenue",
    "cost",
    "margin",
)

# Post-decision outcomes or oracle columns that must never appear in decision-time features X
LEAKAGE_COLUMNS = frozenset(
    {
        "quantity",
        "log_quantity",
        "revenue",
        "cost",
        "margin",
        "true_elasticity",
        "true_propensity",
        "true_q0",
        "true_q1",
        "true_cate_qty",
        "true_cate_revenue",
        "true_cate_margin",
        "exog_demand_index",
    }
)


class DataValidationError(ValueError):
    """Raised when input pricing data fails structural, domain, or leakage checks."""


@dataclass
class ValidationReport:
    """Summary of validation checks performed on a transaction dataset."""

    n_rows: int
    n_missing: dict[str, int] = field(default_factory=dict)
    n_duplicates: int = 0
    n_non_positive_price: int = 0
    n_negative_quantity: int = 0
    n_invalid_discount: int = 0
    passed: bool = True
    errors: list[str] = field(default_factory=list)


def validate_transaction_schema(
    df: pd.DataFrame,
    required_columns: Sequence[str] = REQUIRED_COLUMNS,
    strict: bool = True,
) -> ValidationReport:
    """Validate required columns, data types, and fundamental domain invariants."""
    if df is None or df.empty:
        raise DataValidationError("Input DataFrame is empty or None.")

    errors: list[str] = []
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        if strict:
            raise DataValidationError(errors[0])

    n_missing = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()}
    n_duplicates = int(df["order_id"].duplicated().sum()) if "order_id" in df.columns else 0

    n_non_pos_price = int((pd.to_numeric(df["price"], errors="coerce") <= 0).sum()) if "price" in df.columns else 0
    n_neg_qty = int((pd.to_numeric(df["quantity"], errors="coerce") <= 0).sum()) if "quantity" in df.columns else 0
    n_bad_disc = (
        int(((pd.to_numeric(df["discount"], errors="coerce") < 0) | (pd.to_numeric(df["discount"], errors="coerce") >= 1.0)).sum())
        if "discount" in df.columns
        else 0
    )

    if n_non_pos_price > 0:
        errors.append(f"Found {n_non_pos_price} rows with non-positive price.")
    if n_neg_qty > 0:
        errors.append(f"Found {n_neg_qty} rows with non-positive quantity.")
    if n_bad_disc > 0:
        errors.append(f"Found {n_bad_disc} rows with discount outside [0, 1).")

    if strict and errors:
        raise DataValidationError(" | ".join(errors))

    return ValidationReport(
        n_rows=len(df),
        n_missing=n_missing,
        n_duplicates=n_duplicates,
        n_non_positive_price=n_non_pos_price,
        n_negative_quantity=n_neg_qty,
        n_invalid_discount=n_bad_disc,
        passed=len(errors) == 0,
        errors=errors,
    )


def validate_no_leakage(
    feature_columns: Iterable[str],
    forbidden_columns: Iterable[str] = LEAKAGE_COLUMNS,
) -> None:
    """Ensure decision-time feature matrices exclude post-outcome or ground-truth columns."""
    leaking = sorted(set(feature_columns).intersection(set(forbidden_columns)))
    if leaking:
        raise DataValidationError(
            f"Target/ground-truth leakage detected in feature columns: {leaking}"
        )


def validate_temporal_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> None:
    """Verify chronological split strictly prevents future observations in training data."""
    if train_df.empty or test_df.empty:
        raise DataValidationError("Both train and test splits must be non-empty.")

    train_max = pd.to_datetime(train_df[timestamp_col]).max()
    test_min = pd.to_datetime(test_df[timestamp_col]).min()
    if train_max > test_min:
        raise DataValidationError(
            f"Temporal leakage detected: train max timestamp ({train_max}) > test min timestamp ({test_min})."
        )
