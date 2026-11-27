"""Tests for T-Learner, X-Learner, uplift metrics, pricing constraints, optimization, and backtesting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal.identification import CausalAssumptionError
from evaluation.backtest import run_chronological_backtest
from optimization.constraints import PricingConstraints, evaluate_action_feasibility
from optimization.pricing_policy import CounterfactualOutcomeModel, UpliftBasedPolicy
from optimization.revenue import compute_unit_economics
from preprocessing.cleaning import clean_transaction_data
from preprocessing.feature_engineering import add_engineered_features, build_feature_matrix
from simulation.data_generator import generate_synthetic_marketplace_data
from uplift.base import SLearner
from uplift.evaluation import evaluate_uplift_model
from uplift.t_learner import TLearner
from uplift.x_learner import XLearner


@pytest.fixture(scope="module")
def prepared_df() -> pd.DataFrame:
    raw = generate_synthetic_marketplace_data(n_rows=6000, seed=42)
    cleaned, _ = clean_transaction_data(raw)
    return add_engineered_features(cleaned)


def test_t_learner_and_x_learner_recover_heterogeneity(prepared_df: pd.DataFrame) -> None:
    split = int(len(prepared_df) * 0.75)
    train_df = prepared_df.iloc[:split]
    test_df = prepared_df.iloc[split:]

    X_train, cols = build_feature_matrix(train_df, include_price=False)
    X_test, _ = build_feature_matrix(test_df, include_price=False)
    X_test = X_test.reindex(columns=cols, fill_value=0.0)

    t_train = train_df["treatment_flag"].to_numpy()
    y_train = train_df["margin"].to_numpy()
    t_test = test_df["treatment_flag"].to_numpy()
    y_test = test_df["margin"].to_numpy()
    true_cate = test_df["true_cate_margin"].to_numpy()

    s_mod = SLearner(learner_type="gradient_boosting", seed=42, n_estimators=60).fit(X_train, t_train, y_train)
    t_mod = TLearner(learner_type="gradient_boosting", seed=42, n_estimators=60).fit(X_train, t_train, y_train)
    x_mod = XLearner(learner_type="gradient_boosting", seed=42, n_estimators=60).fit(X_train, t_train, y_train)

    for m in (s_mod, t_mod, x_mod):
        preds = m.predict_cate(X_test)
        assert preds.shape == (len(test_df),)
        ev = evaluate_uplift_model(m.name, y_test, t_test, preds, true_cate=true_cate)
        assert ev.pehe is not None
        assert ev.cate_rank_corr is not None and ev.cate_rank_corr > 0.20

    # No control group raises clear error
    with pytest.raises(CausalAssumptionError, match="both treated and control"):
        TLearner().fit(X_train.iloc[:50], np.ones(50, dtype=int), y_train[:50])


def test_revenue_margin_and_constraint_enforcement(prepared_df: pd.DataFrame) -> None:
    econ = compute_unit_economics(price=[100.0], quantity=[2.0], unit_cost=[60.0])
    assert econ["revenue"][0] == pytest.approx(200.0)
    assert econ["cost"][0] == pytest.approx(120.0)
    assert econ["margin"][0] == pytest.approx(80.0)
    assert econ["margin_pct"][0] == pytest.approx(0.40)

    with pytest.raises(ValueError):
        compute_unit_economics(price=[-5.0], quantity=[1.0], unit_cost=[2.0])

    # Inelastic segment (-0.30): discount (-10%) increases orders slightly but REDUCES margin!
    sample = prepared_df.head(20).copy().reset_index(drop=True)
    sample["price"] = 100.0
    sample["unit_cost"] = 60.0
    sample["seller_availability"] = 0.80
    sample["base_price"] = 100.0
    om = CounterfactualOutcomeModel(
        baseline_quantity=np.full(20, 2.0),
        elasticity_estimate=np.full(20, -0.30),
        promo_lift_log=0.0,
    )
    decision = UpliftBasedPolicy().select_actions(
        sample,
        om,
        candidate_actions=(-0.10, -0.05, 0.0, 0.05, 0.10),
        constraints=PricingConstraints(min_margin_pct=0.08, max_price_change_pct=0.10),
    )
    # Optimizer must NOT select -10% or -5% for inelastic segment; it should choose +10%
    assert np.all(decision.chosen_price_change_pct >= 0.05)

    # Strict min_margin_pct constraint blocks low-margin price cuts
    tight = PricingConstraints(min_margin_pct=0.50)
    cand = om.evaluate_action(sample, -0.10)
    feas = evaluate_action_feasibility(sample, cand["price"], cand["margin_pct"], -0.10, tight)
    assert not np.any(feas)


def test_chronological_backtest_and_empty_segment_handling(prepared_df: pd.DataFrame) -> None:
    bt = run_chronological_backtest(prepared_df, train_fraction=0.75, seed=42)
    train_max = pd.to_datetime(bt["train_df"]["timestamp"]).max()
    test_min = pd.to_datetime(bt["test_df"]["timestamp"]).min()
    assert train_max <= test_min

    oracle_df = bt["oracle_comparison"]
    assert set(oracle_df["policy"]) == {
        "Flat Policy",
        "Random Policy",
        "Segment-Based Policy",
        "Uplift-Based Policy",
    }
    uplift_margin = float(oracle_df.loc[oracle_df["policy"] == "Uplift-Based Policy", "total_margin"].iloc[0])
    flat_margin = float(oracle_df.loc[oracle_df["policy"] == "Flat Policy", "total_margin"].iloc[0])
    rand_margin = float(oracle_df.loc[oracle_df["policy"] == "Random Policy", "total_margin"].iloc[0])
    assert uplift_margin > flat_margin
    assert uplift_margin > rand_margin

    # Empty segment DataFrame raises error in EDA/elasticity
    with pytest.raises(ValueError):
        clean_transaction_data(prepared_df.iloc[0:0])
