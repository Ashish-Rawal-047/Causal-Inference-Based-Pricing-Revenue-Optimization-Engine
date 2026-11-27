"""Uplift modeling and heterogeneous treatment effect package."""

from uplift.base import BaseUpliftEstimator, SLearner, build_base_regressor
from uplift.evaluation import UpliftEvaluationResult, compute_uplift_curve, evaluate_uplift_model
from uplift.segmentation import summarize_segment_uplift
from uplift.t_learner import TLearner
from uplift.x_learner import XLearner

__all__ = [
    "BaseUpliftEstimator",
    "SLearner",
    "TLearner",
    "UpliftEvaluationResult",
    "XLearner",
    "build_base_regressor",
    "compute_uplift_curve",
    "evaluate_uplift_model",
    "summarize_segment_uplift",
]
