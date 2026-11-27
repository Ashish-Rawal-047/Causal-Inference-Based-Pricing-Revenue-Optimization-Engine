"""Counterfactual outcome estimation and four pricing policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from optimization.constraints import PricingConstraints, enforce_coverage_budget, evaluate_action_feasibility
from optimization.revenue import compute_unit_economics, estimate_demand_response


DEFAULT_CANDIDATE_ACTIONS: tuple[float, ...] = (-0.10, -0.05, 0.0, 0.05, 0.10)


@dataclass
class PolicyDecision:
    """Decision output produced by a pricing policy on an evaluation dataset."""

    policy_name: str
    chosen_price_change_pct: np.ndarray
    recommended_price: np.ndarray
    expected_orders: np.ndarray
    expected_revenue: np.ndarray
    expected_cost: np.ndarray
    expected_margin: np.ndarray
    expected_incremental_revenue: np.ndarray
    expected_incremental_margin: np.ndarray
    scenario_table: pd.DataFrame | None = None


class CounterfactualOutcomeModel:
    """Decoupled counterfactual estimator predicting (orders, revenue, cost, margin) for candidate prices."""

    def __init__(
        self,
        baseline_quantity: np.ndarray,
        elasticity_estimate: np.ndarray,
        promo_lift_log: float = 0.08,
    ) -> None:
        self.baseline_quantity = np.maximum(np.asarray(baseline_quantity, dtype=float), 1e-3)
        self.elasticity_estimate = np.clip(np.asarray(elasticity_estimate, dtype=float), -2.6, -0.10)
        self.promo_lift_log = float(promo_lift_log)

    def evaluate_action(
        self,
        df: pd.DataFrame,
        price_change_pct: float,
    ) -> dict[str, np.ndarray]:
        """Estimate outcomes for every row under a candidate fractional price change."""
        base_price = df["price"].to_numpy(dtype=float)
        unit_cost = (
            df["unit_cost"].to_numpy(dtype=float)
            if "unit_cost" in df.columns
            else (df["cost"] / np.maximum(df["quantity"], 1e-6)).to_numpy(dtype=float)
        )
        new_price = np.maximum(base_price * (1.0 + price_change_pct), 0.01)
        new_qty = estimate_demand_response(
            baseline_quantity=self.baseline_quantity,
            price_change_pct=price_change_pct,
            elasticity=self.elasticity_estimate,
            promo_lift_log=self.promo_lift_log,
        )
        econ = compute_unit_economics(new_price, new_qty, unit_cost)
        base_econ = compute_unit_economics(base_price, self.baseline_quantity, unit_cost)

        return {
            "price": new_price,
            "quantity": new_qty,
            "revenue": econ["revenue"],
            "cost": econ["cost"],
            "margin": econ["margin"],
            "margin_pct": econ["margin_pct"],
            "incremental_revenue": econ["revenue"] - base_econ["revenue"],
            "incremental_margin": econ["margin"] - base_econ["margin"],
        }


def evaluate_candidate_scenarios(
    df: pd.DataFrame,
    outcome_model: CounterfactualOutcomeModel,
    candidate_actions: Sequence[float] = DEFAULT_CANDIDATE_ACTIONS,
    segment_col: str = "region",
) -> pd.DataFrame:
    """Evaluate all candidate pricing actions at the segment level for inspection."""
    rows: list[dict[str, Any]] = []
    for delta in candidate_actions:
        pred = outcome_model.evaluate_action(df, float(delta))
        tmp = pd.DataFrame(
            {
                "segment": df[segment_col].astype(str) if segment_col in df.columns else "All",
                "orders": pred["quantity"],
                "revenue": pred["revenue"],
                "cost": pred["cost"],
                "margin": pred["margin"],
                "inc_rev": pred["incremental_revenue"],
                "inc_margin": pred["incremental_margin"],
            }
        )
        for seg_val, grp in tmp.groupby("segment", observed=True):
            rows.append(
                {
                    "segment_col": segment_col,
                    "segment_value": str(seg_val),
                    "price_change_pct": float(delta),
                    "expected_orders": float(grp["orders"].sum()),
                    "expected_revenue": float(grp["revenue"].sum()),
                    "expected_cost": float(grp["cost"].sum()),
                    "expected_margin": float(grp["margin"].sum()),
                    "incremental_revenue": float(grp["inc_rev"].sum()),
                    "incremental_margin": float(grp["inc_margin"].sum()),
                }
            )
    return pd.DataFrame(rows)


def _assemble_decision(
    policy_name: str,
    df: pd.DataFrame,
    chosen_deltas: np.ndarray,
    outcome_model: CounterfactualOutcomeModel,
    candidate_actions: Sequence[float],
) -> PolicyDecision:
    """Compute unified predicted financials for any vector of row-level chosen deltas."""
    n = len(df)
    rec_price = np.zeros(n, dtype=float)
    exp_orders = np.zeros(n, dtype=float)
    exp_rev = np.zeros(n, dtype=float)
    exp_cost = np.zeros(n, dtype=float)
    exp_margin = np.zeros(n, dtype=float)
    inc_rev = np.zeros(n, dtype=float)
    inc_margin = np.zeros(n, dtype=float)

    for delta in sorted(set(chosen_deltas.tolist())):
        mask = np.isclose(chosen_deltas, delta)
        if not np.any(mask):
            continue
        pred = outcome_model.evaluate_action(df, float(delta))
        rec_price[mask] = pred["price"][mask]
        exp_orders[mask] = pred["quantity"][mask]
        exp_rev[mask] = pred["revenue"][mask]
        exp_cost[mask] = pred["cost"][mask]
        exp_margin[mask] = pred["margin"][mask]
        inc_rev[mask] = pred["incremental_revenue"][mask]
        inc_margin[mask] = pred["incremental_margin"][mask]

    return PolicyDecision(
        policy_name=policy_name,
        chosen_price_change_pct=chosen_deltas,
        recommended_price=rec_price,
        expected_orders=exp_orders,
        expected_revenue=exp_rev,
        expected_cost=exp_cost,
        expected_margin=exp_margin,
        expected_incremental_revenue=inc_rev,
        expected_incremental_margin=inc_margin,
        scenario_table=evaluate_candidate_scenarios(df, outcome_model, candidate_actions=candidate_actions),
    )


class RandomPolicy:
    """Assigns a random feasible candidate pricing action per transaction."""

    name = "Random Policy"

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def select_actions(
        self,
        df: pd.DataFrame,
        outcome_model: CounterfactualOutcomeModel,
        candidate_actions: Sequence[float] = DEFAULT_CANDIDATE_ACTIONS,
        constraints: PricingConstraints = PricingConstraints(),
    ) -> PolicyDecision:
        rng = np.random.default_rng(self.seed)
        n = len(df)
        actions = list(candidate_actions)
        feas_mat = np.column_stack(
            [
                evaluate_action_feasibility(
                    df,
                    outcome_model.evaluate_action(df, a)["price"],
                    outcome_model.evaluate_action(df, a)["margin_pct"],
                    a,
                    constraints,
                )
                for a in actions
            ]
        )
        chosen = np.zeros(n, dtype=float)
        for i in range(n):
            valid_idx = np.where(feas_mat[i])[0]
            if len(valid_idx) > 0:
                chosen[i] = float(actions[int(rng.choice(valid_idx))])
        # Enforce coverage limit
        est_inc = np.zeros(n, dtype=float)
        for a in actions:
            m = np.isclose(chosen, a)
            if np.any(m):
                est_inc[m] = outcome_model.evaluate_action(df, a)["incremental_margin"][m]
        chosen = enforce_coverage_budget(chosen, est_inc, constraints.max_treatment_coverage)
        return _assemble_decision(self.name, df, chosen, outcome_model, candidate_actions)


class FlatPolicy:
    """Applies a single uniform pricing action across all eligible transactions."""

    name = "Flat Policy"

    def __init__(self, uniform_delta: float = -0.05) -> None:
        self.uniform_delta = float(uniform_delta)

    def select_actions(
        self,
        df: pd.DataFrame,
        outcome_model: CounterfactualOutcomeModel,
        candidate_actions: Sequence[float] = DEFAULT_CANDIDATE_ACTIONS,
        constraints: PricingConstraints = PricingConstraints(),
    ) -> PolicyDecision:
        pred = outcome_model.evaluate_action(df, self.uniform_delta)
        feasible = evaluate_action_feasibility(
            df, pred["price"], pred["margin_pct"], self.uniform_delta, constraints
        )
        chosen = np.where(feasible, self.uniform_delta, 0.0)
        chosen = enforce_coverage_budget(chosen, pred["incremental_margin"], constraints.max_treatment_coverage)
        return _assemble_decision(self.name, df, chosen, outcome_model, candidate_actions)


class SegmentElasticityPolicy:
    """Segment-based policy maximizing expected incremental margin per (region, cohort)."""

    name = "Segment-Based Policy"

    def __init__(self, segment_cols: tuple[str, ...] = ("region", "customer_cohort")) -> None:
        self.segment_cols = segment_cols

    def select_actions(
        self,
        df: pd.DataFrame,
        outcome_model: CounterfactualOutcomeModel,
        candidate_actions: Sequence[float] = DEFAULT_CANDIDATE_ACTIONS,
        constraints: PricingConstraints = PricingConstraints(),
    ) -> PolicyDecision:
        n = len(df)
        actions = list(candidate_actions)
        preds = {a: outcome_model.evaluate_action(df, a) for a in actions}
        feas = {
            a: evaluate_action_feasibility(df, preds[a]["price"], preds[a]["margin_pct"], a, constraints)
            for a in actions
        }

        seg_keys = df[list(self.segment_cols)].astype(str).agg("|".join, axis=1)
        chosen = np.zeros(n, dtype=float)
        best_inc_margin = np.zeros(n, dtype=float)

        for _, idx in seg_keys.groupby(seg_keys).groups.items():
            idx_arr = np.asarray(list(idx), dtype=int)
            best_a = 0.0
            best_val = -np.inf
            for a in actions:
                if not np.any(feas[a][idx_arr]):
                    continue
                # Rows failing feasibility for action `a` fall back to 0.0 (no incremental margin)
                row_inc = np.where(feas[a][idx_arr], preds[a]["incremental_margin"][idx_arr], 0.0)
                total_inc = float(row_inc.sum())
                if total_inc > best_val:
                    best_val = total_inc
                    best_a = a
            chosen[idx_arr] = np.where(feas[best_a][idx_arr], best_a, 0.0)
            best_inc_margin[idx_arr] = np.where(feas[best_a][idx_arr], preds[best_a]["incremental_margin"][idx_arr], 0.0)

        chosen = enforce_coverage_budget(chosen, best_inc_margin, constraints.max_treatment_coverage)
        return _assemble_decision(self.name, df, chosen, outcome_model, candidate_actions)


class UpliftBasedPolicy:
    """Customer/segment-level causal uplift optimizer maximizing expected incremental margin."""

    name = "Uplift-Based Policy"

    def select_actions(
        self,
        df: pd.DataFrame,
        outcome_model: CounterfactualOutcomeModel,
        candidate_actions: Sequence[float] = DEFAULT_CANDIDATE_ACTIONS,
        constraints: PricingConstraints = PricingConstraints(),
    ) -> PolicyDecision:
        n = len(df)
        actions = list(candidate_actions)
        chosen = np.zeros(n, dtype=float)
        best_inc_margin = np.zeros(n, dtype=float)

        for a in actions:
            pred = outcome_model.evaluate_action(df, a)
            feasible = evaluate_action_feasibility(df, pred["price"], pred["margin_pct"], a, constraints)
            improves = feasible & (pred["incremental_margin"] > best_inc_margin)
            chosen = np.where(improves, float(a), chosen)
            best_inc_margin = np.where(improves, pred["incremental_margin"], best_inc_margin)

        chosen = enforce_coverage_budget(chosen, best_inc_margin, constraints.max_treatment_coverage)
        return _assemble_decision(self.name, df, chosen, outcome_model, candidate_actions)
