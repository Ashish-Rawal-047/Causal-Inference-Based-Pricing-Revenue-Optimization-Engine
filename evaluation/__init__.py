"""Evaluation, backtesting, experiments, and robustness package."""

from evaluation.backtest import chronological_train_test_split, run_chronological_backtest
from evaluation.experiments import (
    run_experiment_1_elasticity_comparison,
    run_experiment_2_segment_heterogeneity,
    run_experiment_3_uplift_models,
    run_experiment_4_pricing_policies,
    save_experiment_outputs,
)
from evaluation.metrics import mae, relative_bias, rmse
from evaluation.robustness import bootstrap_causal_elasticity, run_robustness_suite

__all__ = [
    "bootstrap_causal_elasticity",
    "chronological_train_test_split",
    "mae",
    "relative_bias",
    "rmse",
    "run_chronological_backtest",
    "run_experiment_1_elasticity_comparison",
    "run_experiment_2_segment_heterogeneity",
    "run_experiment_3_uplift_models",
    "run_experiment_4_pricing_policies",
    "run_robustness_suite",
    "save_experiment_outputs",
]
