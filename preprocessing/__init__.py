"""Preprocessing package."""

from preprocessing.cleaning import CleaningSummary, clean_transaction_data
from preprocessing.feature_engineering import add_engineered_features, build_feature_matrix
from preprocessing.validation import (
    LEAKAGE_COLUMNS,
    REQUIRED_COLUMNS,
    DataValidationError,
    ValidationReport,
    validate_no_leakage,
    validate_temporal_split,
    validate_transaction_schema,
)

__all__ = [
    "CleaningSummary",
    "DataValidationError",
    "LEAKAGE_COLUMNS",
    "REQUIRED_COLUMNS",
    "ValidationReport",
    "add_engineered_features",
    "build_feature_matrix",
    "clean_transaction_data",
    "validate_no_leakage",
    "validate_temporal_split",
    "validate_transaction_schema",
]
