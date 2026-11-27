"""Ground-truth structural parameters and potential outcomes for the synthetic marketplace."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


REGIONS = ("North", "South", "East", "West")
COHORTS = ("New", "Returning", "Loyal", "VIP")
CATEGORIES = ("Electronics", "Fashion", "Home", "Grocery", "Beauty")


@dataclass(frozen=True)
class GroundTruthSpec:
    """True structural parameters of the synthetic marketplace DGP.

    Elasticities are negative (log-log demand response to price).
    """

    region_elasticity: dict[str, float]
    cohort_elasticity_adj: dict[str, float]
    category_elasticity_adj: dict[str, float]
    region_base_demand: dict[str, float]
    region_base_price_mult: dict[str, float]
    category_base_price: dict[str, float]
    category_cost_ratio: dict[str, float]
    promo_demand_lift: float = 0.14
    competitor_cross_elasticity: float = 0.42
    weekend_demand_lift: float = 0.16
    festival_demand_lift: float = 0.28
    availability_demand_slope: float = 0.35


DEFAULT_GROUND_TRUTH = GroundTruthSpec(
    # West is affluent/inelastic + high demand + high price -> drives Simpson's Paradox
    region_elasticity={
        "North": -1.45,
        "East": -1.25,
        "South": -0.85,
        "West": -0.52,
    },
    cohort_elasticity_adj={
        "New": -0.35,
        "Returning": -0.10,
        "Loyal": 0.20,
        "VIP": 0.30,
    },
    category_elasticity_adj={
        "Grocery": -0.18,
        "Fashion": -0.15,
        "Beauty": -0.05,
        "Home": 0.08,
        "Electronics": 0.16,
    },
    region_base_demand={
        "North": 0.40,
        "East": 0.65,
        "South": 1.15,
        "West": 2.35,
    },
    region_base_price_mult={
        "North": 0.72,
        "East": 0.85,
        "South": 1.05,
        "West": 1.48,
    },
    category_base_price={
        "Grocery": 22.0,
        "Beauty": 38.0,
        "Fashion": 56.0,
        "Home": 88.0,
        "Electronics": 155.0,
    },
    category_cost_ratio={
        "Grocery": 0.64,
        "Beauty": 0.46,
        "Fashion": 0.52,
        "Home": 0.56,
        "Electronics": 0.68,
    },
)


def compute_true_elasticity(
    region: np.ndarray | pd.Series,
    cohort: np.ndarray | pd.Series,
    category: np.ndarray | pd.Series,
    historical_orders: np.ndarray | pd.Series,
    spec: GroundTruthSpec = DEFAULT_GROUND_TRUTH,
) -> np.ndarray:
    """Compute true structural price elasticity beta_i for each transaction.

    Elasticities are capped below -0.15 so every segment remains strictly downward sloping.
    """
    reg_eff = pd.Series(region).map(spec.region_elasticity).to_numpy(dtype=float)
    coh_eff = pd.Series(cohort).map(spec.cohort_elasticity_adj).to_numpy(dtype=float)
    cat_eff = pd.Series(category).map(spec.category_elasticity_adj).to_numpy(dtype=float)
    # Frequent buyers are slightly less price-sensitive
    freq_adj = 0.06 * np.log1p(np.asarray(historical_orders, dtype=float))
    elasticity = reg_eff + coh_eff + cat_eff + freq_adj
    return np.clip(elasticity, -2.4, -0.15)


def compute_structural_expected_quantity(
    base_demand_index: np.ndarray,
    price: np.ndarray,
    reference_price: np.ndarray,
    competitor_price: np.ndarray,
    elasticity: np.ndarray,
    promotion_flag: np.ndarray,
    spec: GroundTruthSpec = DEFAULT_GROUND_TRUTH,
) -> np.ndarray:
    """Deterministic expected quantity E[Q | X, P] under the structural log-log demand curve."""
    safe_price = np.maximum(price, 1e-3)
    safe_ref = np.maximum(reference_price, 1e-3)
    safe_comp = np.maximum(competitor_price, 1e-3)

    log_q = (
        base_demand_index
        + elasticity * np.log(safe_price / safe_ref)
        + spec.competitor_cross_elasticity * np.log(safe_comp / safe_price)
        + spec.promo_demand_lift * promotion_flag
    )
    # Note: log(safe_comp / safe_price) adds -competitor_cross_elasticity to effective price elasticity
    # when competitor_price is fixed. Wait: we want `elasticity` passed in to be the TOTAL structural
    # price elasticity d(log Q)/d(log P) holding competitor_price fixed!
    return np.exp(log_q)


def compute_total_structural_log_q(
    exogenous_demand_index: np.ndarray,
    price: np.ndarray,
    reference_price: np.ndarray,
    competitor_price: np.ndarray,
    total_elasticity: np.ndarray,
    promotion_flag: np.ndarray,
    spec: GroundTruthSpec = DEFAULT_GROUND_TRUTH,
) -> np.ndarray:
    """Compute structural log(E[Q]) where d(log Q)/d(log P) == total_elasticity exactly.

    Competitor prices enter via log(competitor_price / reference_price) so that
    total_elasticity is the exact structural derivative with respect to log(price).
    """
    safe_price = np.maximum(price, 1e-3)
    safe_ref = np.maximum(reference_price, 1e-3)
    safe_comp = np.maximum(competitor_price, 1e-3)

    return (
        exogenous_demand_index
        + total_elasticity * np.log(safe_price / safe_ref)
        + spec.competitor_cross_elasticity * np.log(safe_comp / safe_ref)
        + spec.promo_demand_lift * promotion_flag
    )
