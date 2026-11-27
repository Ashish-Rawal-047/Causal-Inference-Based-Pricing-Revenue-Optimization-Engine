"""Business guardrails and feasibility constraints for pricing optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PricingConstraints:
    """Configurable business guardrails enforced on candidate pricing actions."""

    min_margin_pct: float = 0.08
    max_discount: float = 0.35
    min_price: float = 5.0
    max_price: float = 2000.0
    max_price_change_pct: float = 0.10
    min_seller_availability: float = 0.25
    eligible_cohorts: tuple[str, ...] | None = None
    max_treatment_coverage: float = 1.0


def evaluate_action_feasibility(
    df: pd.DataFrame,
    candidate_price: np.ndarray,
    expected_margin_pct: np.ndarray,
    price_change_pct: float,
    constraints: PricingConstraints,
) -> np.ndarray:
    """Return boolean mask indicating which rows satisfy all pricing business constraints."""
    n = len(df)
    feasible = np.ones(n, dtype=bool)

    # Price bounds & max change guardrail
    if abs(price_change_pct) > constraints.max_price_change_pct + 1e-9:
        return np.zeros(n, dtype=bool)

    feasible &= (candidate_price >= constraints.min_price) & (candidate_price <= constraints.max_price)

    # Minimum margin guardrail (never allow zero-change baseline to become infeasible unless forced)
    if abs(price_change_pct) > 1e-9:
        feasible &= expected_margin_pct >= constraints.min_margin_pct

    # Maximum discount depth guardrail relative to base/list price if available
    if "base_price" in df.columns and price_change_pct < 0:
        eff_discount = 1.0 - (candidate_price / np.maximum(df["base_price"].to_numpy(dtype=float), 1e-6))
        feasible &= eff_discount <= constraints.max_discount

    # Seller availability guardrail for discount-driven demand increases
    if "seller_availability" in df.columns and price_change_pct < 0:
        feasible &= df["seller_availability"].to_numpy(dtype=float) >= constraints.min_seller_availability

    # Cohort eligibility guardrail for non-zero interventions
    if constraints.eligible_cohorts is not None and abs(price_change_pct) > 1e-9:
        feasible &= df["customer_cohort"].isin(constraints.eligible_cohorts).to_numpy()

    return feasible


def enforce_coverage_budget(
    chosen_deltas: np.ndarray,
    expected_incremental_margin: np.ndarray,
    max_coverage: float,
) -> np.ndarray:
    """If non-zero price interventions exceed `max_coverage`, keep top interventions by incremental margin."""
    if max_coverage >= 1.0 - 1e-9:
        return chosen_deltas
    if max_coverage <= 0.0:
        return np.zeros_like(chosen_deltas)

    out = chosen_deltas.copy()
    intervened_idx = np.where(np.abs(out) > 1e-9)[0]
    max_allowed = int(np.floor(max_coverage * len(out)))
    if len(intervened_idx) > max_allowed:
        # Rank intervened rows by expected incremental margin descending
        ranked = intervened_idx[np.argsort(-expected_incremental_margin[intervened_idx])]
        drop_idx = ranked[max_allowed:]
        out[drop_idx] = 0.0
    return out
