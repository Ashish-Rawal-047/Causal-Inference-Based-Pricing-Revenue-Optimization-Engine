"""Causal inference package (IV, DiD, identification, diagnostics)."""

from causal.diagnostics import (
    OverlapBalanceDiagnostics,
    SensitivityResult,
    check_sutva_and_interference_risk,
    check_treatment_overlap_and_balance,
    run_causal_diagnostics_report,
    run_omitted_variable_sensitivity,
)
from causal.did import DiDResult, estimate_difference_in_differences
from causal.identification import (
    AssumptionAuditItem,
    CausalAssumptionError,
    get_identification_assumptions_registry,
)
from causal.iv import IVResult, estimate_2sls_iv

__all__ = [
    "AssumptionAuditItem",
    "CausalAssumptionError",
    "DiDResult",
    "IVResult",
    "OverlapBalanceDiagnostics",
    "SensitivityResult",
    "check_sutva_and_interference_risk",
    "check_treatment_overlap_and_balance",
    "estimate_2sls_iv",
    "estimate_difference_in_differences",
    "get_identification_assumptions_registry",
    "run_causal_diagnostics_report",
    "run_omitted_variable_sensitivity",
]
