"""Unified policy simulation and fair comparison engine."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

if __name__ == "__main__" and str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from analysis.elasticity import estimate_segment_elasticities
from causal.iv import estimate_2sls_iv
from config.loader import get_logger
from optimization.constraints import PricingConstraints
from optimization.pricing_policy import (
    DEFAULT_CANDIDATE_ACTIONS,
    CounterfactualOutcomeModel,
    FlatPolicy,
    PolicyDecision,
    RandomPolicy,
    SegmentElasticityPolicy,
    UpliftBasedPolicy,
)
from optimization.revenue import compute_policy_financials
from preprocessing.cleaning import clean_transaction_data
from preprocessing.feature_engineering import add_engineered_features, build_feature_matrix
from simulation.data_generator import generate_synthetic_marketplace_data
from simulation.marketplace import evaluate_counterfactual_price_action
from uplift.x_learner import XLearner

logger = get_logger(__name__)


def build_uplift_counterfactual_model(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    promo_discount_depth: float = 0.15,
    learner_type: str = "gradient_boosting",
    seed: int = 42,
) -> tuple[CounterfactualOutcomeModel, np.ndarray, np.ndarray]:
    """Fit causal uplift & IV-anchored elasticity models on train_df to parameterize eval_df outcomes.

    Returns:
        (outcome_model_for_uplift, pred_cate_qty, pred_cate_margin)
    """
    X_train, feat_cols = build_feature_matrix(train_df, include_price=False)
    X_eval, _ = build_feature_matrix(eval_df, include_price=False)
    X_eval = X_eval.reindex(columns=feat_cols, fill_value=0.0)

    t_train = train_df["treatment_flag"].to_numpy(dtype=int)
    y_qty_train = train_df["quantity"].to_numpy(dtype=float)
    y_margin_train = train_df["margin"].to_numpy(dtype=float)

    xl_qty = XLearner(learner_type=learner_type, seed=seed)
    xl_qty.fit(X_train, t_train, y_qty_train)

    xl_margin = XLearner(learner_type=learner_type, seed=seed + 10)
    xl_margin.fit(X_train, t_train, y_margin_train)

    pred_cate_qty = xl_qty.predict_cate(X_eval)
    pred_cate_margin = xl_margin.predict_cate(X_eval)

    # Baseline expected quantity E[Q(0) | X] from X-Learner stage-1 control model
    mu0_qty = np.maximum(xl_qty.mu0.predict(X_eval.to_numpy(dtype=float)), 0.25)
    mu1_qty = np.maximum(mu0_qty + pred_cate_qty, 0.25)

    # Convert individual CATE on promo_discount_depth into implied structural elasticity
    # Accounting for promo lift (~0.14 in log units) so elasticity reflects pure price sensitivity
    log_price_ratio = np.log(max(1.0 - promo_discount_depth, 0.2))
    raw_log_lift = np.log(mu1_qty / mu0_qty) - 0.10
    implied_indiv_elast = np.clip(raw_log_lift / log_price_ratio, -2.5, -0.10)

    # Anchor overall mean elasticity to 2SLS IV estimate if supplier_cost_shock is available
    if "supplier_cost_shock" in train_df.columns:
        try:
            iv_res = estimate_2sls_iv(train_df)
            shift = iv_res.elasticity - float(np.mean(implied_indiv_elast))
            implied_indiv_elast = np.clip(implied_indiv_elast + shift, -2.5, -0.10)
        except Exception as exc:
            logger.warning("IV anchoring skipped: %s", exc)

    outcome_model = CounterfactualOutcomeModel(
        baseline_quantity=mu0_qty,
        elasticity_estimate=implied_indiv_elast,
        promo_lift_log=0.10,
    )
    return outcome_model, pred_cate_qty, pred_cate_margin


def build_segment_counterfactual_model(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> CounterfactualOutcomeModel:
    """Build segment-level elasticity counterfactual model (without individual CATE)."""
    seg_el = estimate_segment_elasticities(train_df, segment_cols=("region",), controlled=True)
    seg_map = dict(zip(seg_el["segment_value"], seg_el["elasticity"]))
    global_el = float(seg_el["elasticity"].mean()) if not seg_el.empty else -0.8
    eval_el = eval_df["region"].astype(str).map(seg_map).fillna(global_el).to_numpy(dtype=float)

    # Segment baseline quantity using training region-cohort mean
    seg_q = train_df.groupby(["region", "customer_cohort"], observed=True)["quantity"].mean()
    eval_q = (
        eval_df.set_index(["region", "customer_cohort"]).index.map(seg_q).to_numpy(dtype=float)
    )
    fallback_q = float(train_df["quantity"].mean())
    eval_q = np.where(np.isnan(eval_q), fallback_q, eval_q)

    return CounterfactualOutcomeModel(
        baseline_quantity=eval_q,
        elasticity_estimate=eval_el,
        promo_lift_log=0.08,
    )


def compare_pricing_policies(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    candidate_actions: Sequence[float] = DEFAULT_CANDIDATE_ACTIONS,
    constraints: PricingConstraints = PricingConstraints(),
    seed: int = 42,
) -> dict[str, Any]:
    """Compare Flat, Random, Segment-Based, and Uplift-Based policies on the exact same eval_df."""
    uplift_model, pred_cate_qty, pred_cate_margin = build_uplift_counterfactual_model(
        train_df, eval_df, seed=seed
    )
    segment_model = build_segment_counterfactual_model(train_df, eval_df)

    # Evaluate all policies using their decision rules
    flat_dec = FlatPolicy(uniform_delta=-0.05).select_actions(
        eval_df, segment_model, candidate_actions=candidate_actions, constraints=constraints
    )
    rand_dec = RandomPolicy(seed=seed).select_actions(
        eval_df, segment_model, candidate_actions=candidate_actions, constraints=constraints
    )
    seg_dec = SegmentElasticityPolicy().select_actions(
        eval_df, segment_model, candidate_actions=candidate_actions, constraints=constraints
    )
    uplift_dec = UpliftBasedPolicy().select_actions(
        eval_df, uplift_model, candidate_actions=candidate_actions, constraints=constraints
    )

    decisions: list[PolicyDecision] = [flat_dec, rand_dec, seg_dec, uplift_dec]

    # Model-estimated comparison (all scored consistently under the rich uplift outcome model)
    model_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    has_oracle = {"true_elasticity", "exog_demand_index", "base_price", "unit_cost"}.issubset(eval_df.columns)

    base_price = eval_df["price"].to_numpy(dtype=float)
    unit_cost = (
        eval_df["unit_cost"].to_numpy(dtype=float)
        if "unit_cost" in eval_df.columns
        else (eval_df["cost"] / np.maximum(eval_df["quantity"], 1e-6)).to_numpy(dtype=float)
    )

    if has_oracle:
        oracle_base = evaluate_counterfactual_price_action(eval_df, 0.0)
        oracle_q0 = oracle_base["cf_quantity"].to_numpy(dtype=float)

    for dec in decisions:
        # Score under unified model evaluator
        m_qty = np.zeros(len(eval_df))
        m_price = np.zeros(len(eval_df))
        for d_val in sorted(set(dec.chosen_price_change_pct.tolist())):
            mask = np.isclose(dec.chosen_price_change_pct, d_val)
            if np.any(mask):
                out = uplift_model.evaluate_action(eval_df, float(d_val))
                m_qty[mask] = out["quantity"][mask]
                m_price[mask] = out["price"][mask]

        m_fin = compute_policy_financials(
            baseline_price=base_price,
            baseline_quantity=uplift_model.baseline_quantity,
            unit_cost=unit_cost,
            new_price=m_price,
            new_quantity=m_qty,
        )
        m_fin["policy"] = dec.policy_name
        m_fin["evaluation_mode"] = "model_estimated"
        m_fin["avg_price_change_pct"] = float(np.mean(dec.chosen_price_change_pct))
        model_rows.append(m_fin)

        # Score under synthetic ground-truth oracle when available
        if has_oracle:
            cf = evaluate_counterfactual_price_action(eval_df, dec.chosen_price_change_pct)
            o_fin = compute_policy_financials(
                baseline_price=base_price,
                baseline_quantity=oracle_q0,
                unit_cost=unit_cost,
                new_price=cf["cf_price"].to_numpy(dtype=float),
                new_quantity=cf["cf_quantity"].to_numpy(dtype=float),
            )
            o_fin["policy"] = dec.policy_name
            o_fin["evaluation_mode"] = "synthetic_ground_truth"
            o_fin["avg_price_change_pct"] = float(np.mean(dec.chosen_price_change_pct))
            oracle_rows.append(o_fin)

    return {
        "model_comparison": pd.DataFrame(model_rows),
        "oracle_comparison": pd.DataFrame(oracle_rows) if oracle_rows else pd.DataFrame(),
        "decisions": {d.policy_name: d for d in decisions},
        "pred_cate_qty": pred_cate_qty,
        "pred_cate_margin": pred_cate_margin,
        "uplift_scenario_table": uplift_dec.scenario_table,
    }


if __name__ == "__main__":
    raw_df = generate_synthetic_marketplace_data(n_rows=20_000, seed=42)
    cleaned_df, _ = clean_transaction_data(raw_df)
    feat_df = add_engineered_features(cleaned_df)
    split_idx = int(len(feat_df) * 0.75)
    train_df = feat_df.iloc[:split_idx].reset_index(drop=True)
    eval_df = feat_df.iloc[split_idx:].reset_index(drop=True)

    res = compare_pricing_policies(train_df, eval_df, seed=42)
    print("=== Pricing Policy Comparison (Synthetic Ground-Truth Simulation) ===")
    cols = [
        "policy",
        "total_revenue",
        "total_margin",
        "incremental_revenue",
        "incremental_margin",
        "incremental_orders",
        "discount_expenditure",
        "treatment_coverage",
    ]
    print(res["oracle_comparison"][cols].to_string(index=False))
