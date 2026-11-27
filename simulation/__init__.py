"""Simulation and ground-truth marketplace data generation."""

from simulation.data_generator import generate_all_configured_sizes, generate_synthetic_marketplace_data
from simulation.ground_truth import (
    CATEGORIES,
    COHORTS,
    DEFAULT_GROUND_TRUTH,
    REGIONS,
    GroundTruthSpec,
    compute_true_elasticity,
)
from simulation.marketplace import (
    GROUND_TRUTH_COLUMNS,
    describe_olist_causal_support,
    evaluate_counterfactual_price_action,
    load_olist_dataset,
    load_or_generate_dataset,
)

__all__ = [
    "CATEGORIES",
    "COHORTS",
    "DEFAULT_GROUND_TRUTH",
    "GROUND_TRUTH_COLUMNS",
    "GroundTruthSpec",
    "REGIONS",
    "compute_true_elasticity",
    "describe_olist_causal_support",
    "evaluate_counterfactual_price_action",
    "generate_all_configured_sizes",
    "generate_synthetic_marketplace_data",
    "load_olist_dataset",
    "load_or_generate_dataset",
]
