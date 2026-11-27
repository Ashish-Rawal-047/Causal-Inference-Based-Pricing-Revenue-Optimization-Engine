"""Pricing optimization and policy simulation package."""

from optimization.constraints import PricingConstraints, enforce_coverage_budget, evaluate_action_feasibility
from optimization.pricing_policy import (
    DEFAULT_CANDIDATE_ACTIONS,
    CounterfactualOutcomeModel,
    FlatPolicy,
    PolicyDecision,
    RandomPolicy,
    SegmentElasticityPolicy,
    UpliftBasedPolicy,
    evaluate_candidate_scenarios,
)
from optimization.revenue import compute_policy_financials, compute_unit_economics, estimate_demand_response
from optimization.simulation import (
    build_segment_counterfactual_model,
    build_uplift_counterfactual_model,
    compare_pricing_policies,
)

__all__ = [
    "CounterfactualOutcomeModel",
    "DEFAULT_CANDIDATE_ACTIONS",
    "FlatPolicy",
    "PolicyDecision",
    "PricingConstraints",
    "RandomPolicy",
    "SegmentElasticityPolicy",
    "UpliftBasedPolicy",
    "build_segment_counterfactual_model",
    "build_uplift_counterfactual_model",
    "compare_pricing_policies",
    "compute_policy_financials",
    "compute_unit_economics",
    "enforce_coverage_budget",
    "estimate_demand_response",
    "evaluate_action_feasibility",
    "evaluate_candidate_scenarios",
]
