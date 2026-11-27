"""Revenue, cost, margin, and treatment cost calculations for pricing actions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_unit_economics(
    price: np.ndarray | pd.Series,
    quantity: np.ndarray | pd.Series,
    unit_cost: np.ndarray | pd.Series,
) -> dict[str, np.ndarray]:
    """Compute transaction-level revenue, cost, and gross margin arrays."""
    p = np.asarray(price, dtype=float)
    q = np.asarray(quantity, dtype=float)
    c = np.asarray(unit_cost, dtype=float)
    if np.any(p < 0) or np.any(q < 0) or np.any(c < 0):
        raise ValueError("Price, quantity, and unit_cost must be non-negative.")

    rev = p * q
    cost = c * q
    margin = rev - cost
    margin_pct = np.where(rev > 1e-9, margin / rev, 0.0)
    return {
        "revenue": rev,
        "cost": cost,
        "margin": margin,
        "margin_pct": margin_pct,
    }


def estimate_demand_response(
    baseline_quantity: np.ndarray,
    price_change_pct: np.ndarray | float,
    elasticity: np.ndarray,
    promo_lift_log: float = 0.08,
) -> np.ndarray:
    """Predict expected quantity under a fractional price change delta_pct.

    Q_new = Q_base * (1 + delta)^elasticity * exp(promo_lift_log * I[delta <= -0.05])
    """
    q0 = np.maximum(np.asarray(baseline_quantity, dtype=float), 1e-6)
    delta = np.asarray(price_change_pct, dtype=float)
    el = np.asarray(elasticity, dtype=float)
    price_ratio = np.maximum(1.0 + delta, 0.05)
    promo_mult = np.where(delta <= -0.05, np.exp(promo_lift_log), 1.0)
    return q0 * (price_ratio ** el) * promo_mult


def compute_policy_financials(
    baseline_price: np.ndarray,
    baseline_quantity: np.ndarray,
    unit_cost: np.ndarray,
    new_price: np.ndarray,
    new_quantity: np.ndarray,
) -> dict[str, float]:
    """Aggregate portfolio revenue, margin, incremental lift, and discount expenditure."""
    base_econ = compute_unit_economics(baseline_price, baseline_quantity, unit_cost)
    new_econ = compute_unit_economics(new_price, new_quantity, unit_cost)

    base_rev = float(base_econ["revenue"].sum())
    new_rev = float(new_econ["revenue"].sum())
    base_margin = float(base_econ["margin"].sum())
    new_margin = float(new_econ["margin"].sum())
    base_orders = float(np.sum(baseline_quantity))
    new_orders = float(np.sum(new_quantity))

    # Discount / treatment cost: revenue foregone from price reductions on realized volume
    price_drop = np.maximum(baseline_price - new_price, 0.0)
    discount_expenditure = float(np.sum(price_drop * new_quantity))
    treatment_coverage = float(np.mean(np.abs(new_price - baseline_price) > 1e-4))

    return {
        "total_revenue": new_rev,
        "total_cost": float(new_econ["cost"].sum()),
        "total_margin": new_margin,
        "total_orders": new_orders,
        "baseline_revenue": base_rev,
        "baseline_margin": base_margin,
        "baseline_orders": base_orders,
        "incremental_revenue": new_rev - base_rev,
        "incremental_margin": new_margin - base_margin,
        "incremental_orders": new_orders - base_orders,
        "discount_expenditure": discount_expenditure,
        "treatment_coverage": treatment_coverage,
    }
