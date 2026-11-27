"""Causal identification specifications and assumption documentation."""

from __future__ import annotations

from dataclasses import dataclass


class CausalAssumptionError(ValueError):
    """Raised when required data support or causal identification preconditions fail."""


@dataclass(frozen=True)
class AssumptionAuditItem:
    """Documents a causal identification assumption, testability, and synthetic status."""

    method: str
    assumption: str
    testable_from_observational_data: bool
    synthetic_dgp_status: str
    explanation: str


def get_identification_assumptions_registry() -> list[AssumptionAuditItem]:
    """Return structured documentation of causal assumptions across estimators."""
    return [
        AssumptionAuditItem(
            method="IV (2SLS)",
            assumption="Relevance: Cov(Z, log P | X) != 0",
            testable_from_observational_data=True,
            synthetic_dgp_status="Satisfied by design (supplier_cost_shock shifts list_price)",
            explanation="Tested via first-stage partial F-statistic (rule of thumb F > 10).",
        ),
        AssumptionAuditItem(
            method="IV (2SLS)",
            assumption="Exclusion Restriction: Z affects Q only through P (given X)",
            testable_from_observational_data=False,
            synthetic_dgp_status="Satisfied by design (no direct path Z -> Q in structural equation)",
            explanation=(
                "Cannot be proven from observational data alone with a single instrument; "
                "requires institutional/domain justification."
            ),
        ),
        AssumptionAuditItem(
            method="IV (2SLS)",
            assumption="Independence: Z is independent of unobserved demand shocks U | X",
            testable_from_observational_data=False,
            synthetic_dgp_status="Satisfied by design (Z drawn independently of u_demand)",
            explanation=(
                "Observed covariate balance across Z quantiles provides partial falsification, "
                "but independence from unmeasured confounders remains untestable."
            ),
        ),
        AssumptionAuditItem(
            method="Difference-in-Differences",
            assumption="Parallel Trends: Treated & control groups share counterfactual post-period trend",
            testable_from_observational_data=False,
            synthetic_dgp_status="Satisfied by design (shared calendar trend across regions)",
            explanation=(
                "Pre-treatment trend checks and event-study leads can falsify divergence prior to treatment, "
                "but never prove parallel counterfactual trends after treatment."
            ),
        ),
        AssumptionAuditItem(
            method="Difference-in-Differences",
            assumption="No Anticipation / Sharp Policy Timing",
            testable_from_observational_data=False,
            synthetic_dgp_status="Satisfied by design (intervention activates at did_start_day)",
            explanation="Pre-treatment event-study leads near zero are consistent with no anticipation.",
        ),
        AssumptionAuditItem(
            method="CATE / Uplift (T- & X-Learner)",
            assumption="Conditional Ignorability / Unconfoundedness: (Y(0), Y(1)) _||_ T | X",
            testable_from_observational_data=False,
            synthetic_dgp_status="Partially violated when confounding_strength > 0 (u_demand in propensity)",
            explanation=(
                "Untestable from observational data; sensitivity analysis bounds how strong "
                "unobserved confounding must be to overturn conclusions."
            ),
        ),
        AssumptionAuditItem(
            method="CATE / Uplift / Policy",
            assumption="Positivity (Overlap): 0 < P(T=1 | X=x) < 1",
            testable_from_observational_data=True,
            synthetic_dgp_status="Satisfied by design (propensities clipped inside [0.08, 0.92])",
            explanation="Directly inspectable via estimated propensity distributions across treated/control.",
        ),
        AssumptionAuditItem(
            method="All Estimators",
            assumption="SUTVA: No interference between units & well-defined treatment versions",
            testable_from_observational_data=False,
            synthetic_dgp_status="Satisfied at transaction level; real marketplaces face inventory contention",
            explanation=(
                "In real two-sided marketplaces, discounting one user/product can cannibalize substitutes "
                "or deplete shared seller inventory."
            ),
        ),
    ]
