"""Analysis package (EDA, naive elasticity, segmentation, confounding)."""

from analysis.confounding import (
    analyze_correlations_and_confounders,
    build_marketplace_dag,
    demonstrate_simpsons_paradox,
)
from analysis.eda import run_eda
from analysis.elasticity import (
    ElasticityResult,
    estimate_controlled_loglog_elasticity,
    estimate_linear_elasticity,
    estimate_naive_loglog_elasticity,
    estimate_segment_elasticities,
    run_elasticity_comparison,
)
from analysis.segmentation import build_segment_profiles

__all__ = [
    "ElasticityResult",
    "analyze_correlations_and_confounders",
    "build_marketplace_dag",
    "build_segment_profiles",
    "demonstrate_simpsons_paradox",
    "estimate_controlled_loglog_elasticity",
    "estimate_linear_elasticity",
    "estimate_naive_loglog_elasticity",
    "estimate_segment_elasticities",
    "run_eda",
    "run_elasticity_comparison",
]
