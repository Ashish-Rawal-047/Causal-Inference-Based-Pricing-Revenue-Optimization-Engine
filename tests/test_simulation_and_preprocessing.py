"""Tests for synthetic data generation, reproducibility, validation, cleaning, and leakage prevention."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from preprocessing.cleaning import clean_transaction_data
from preprocessing.feature_engineering import add_engineered_features, build_feature_matrix
from preprocessing.validation import (
    DataValidationError,
    validate_no_leakage,
    validate_temporal_split,
    validate_transaction_schema,
)
from simulation.data_generator import generate_synthetic_marketplace_data


def test_synthetic_generation_schema_and_reproducibility() -> None:
    df1 = generate_synthetic_marketplace_data(n_rows=1000, seed=123)
    df2 = generate_synthetic_marketplace_data(n_rows=1000, seed=123)
    df3 = generate_synthetic_marketplace_data(n_rows=1000, seed=999)

    assert len(df1) == 1000
    pd.testing.assert_frame_equal(df1, df2)
    assert not df1["price"].equals(df3["price"])

    report = validate_transaction_schema(df1, strict=True)
    assert report.passed
    assert report.n_duplicates == 0


def test_validation_rejects_zero_price_and_negative_quantity() -> None:
    df = generate_synthetic_marketplace_data(n_rows=200, seed=42)
    bad_price = df.copy()
    bad_price.loc[0, "price"] = 0.0
    with pytest.raises(DataValidationError, match="non-positive price"):
        validate_transaction_schema(bad_price, strict=True)

    bad_qty = df.copy()
    bad_qty.loc[1, "quantity"] = -3.0
    with pytest.raises(DataValidationError, match="non-positive quantity"):
        validate_transaction_schema(bad_qty, strict=True)


def test_cleaning_handles_missing_duplicates_and_invalid_rows() -> None:
    df = generate_synthetic_marketplace_data(n_rows=300, seed=7)
    dirty = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)
    dirty.loc[5, "price"] = -10.0
    dirty.loc[6, "quantity"] = 0.0
    dirty.loc[7, "competitor_price"] = np.nan
    dirty.loc[8, "region"] = np.nan

    cleaned, summary = clean_transaction_data(dirty, strict_invalid=False)
    assert summary.duplicates_removed == 2
    assert summary.invalid_rows_removed == 2
    assert summary.missing_values_imputed >= 2
    assert (cleaned["price"] > 0).all()
    assert (cleaned["quantity"] > 0).all()
    assert not cleaned["competitor_price"].isna().any()


def test_feature_engineering_is_chronological_and_leakage_free() -> None:
    df = generate_synthetic_marketplace_data(n_rows=600, seed=11)
    cleaned, _ = clean_transaction_data(df)
    feat = add_engineered_features(cleaned)

    X_mat, feat_cols = build_feature_matrix(feat, include_price=False)
    assert len(X_mat) == len(feat)
    validate_no_leakage(feat_cols)

    with pytest.raises(DataValidationError, match="leakage"):
        validate_no_leakage(feat_cols + ["margin"])

    with pytest.raises(DataValidationError, match="leakage"):
        validate_no_leakage(feat_cols + ["true_elasticity"])

    # Verify chronological prior count starts at 0 for first appearance of a customer
    first_cust_rows = feat.groupby("customer_id", sort=False).head(1)
    assert (first_cust_rows["cust_prior_tx_count"] == 0).all()


def test_temporal_split_catches_future_leakage() -> None:
    df = generate_synthetic_marketplace_data(n_rows=200, seed=42)
    train_bad = df.iloc[100:].copy()
    test_bad = df.iloc[:100].copy()
    with pytest.raises(DataValidationError, match="Temporal leakage"):
        validate_temporal_split(train_bad, test_bad)
