"""Tests for elasticity, confounding, 2SLS IV, DiD, and causal diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.confounding import demonstrate_simpsons_paradox
from analysis.elasticity import (
    estimate_controlled_loglog_elasticity,
    estimate_linear_elasticity,
    estimate_naive_loglog_elasticity,
    estimate_segment_elasticities,
)
from causal.diagnostics import check_treatment_overlap_and_balance, run_omitted_variable_sensitivity
from causal.did import estimate_difference_in_differences
from causal.identification import CausalAssumptionError
from causal.iv import estimate_2sls_iv
from preprocessing.cleaning import clean_transaction_data
from preprocessing.feature_engineering import add_engineered_features
from simulation.data_generator import generate_synthetic_marketplace_data


@pytest.fixture(scope="module")
def prepared_df() -> pd.DataFrame:
    raw = generate_synthetic_marketplace_data(n_rows=8000, seed=42)
    cleaned, _ = clean_transaction_data(raw)
    return add_engineered_features(cleaned)


def test_elasticity_estimators_and_simpsons_paradox(prepared_df: pd.DataFrame) -> None:
    lin = estimate_linear_elasticity(prepared_df)
    naive = estimate_naive_loglog_elasticity(prepared_df)
    ctrl = estimate_controlled_loglog_elasticity(prepared_df)
    seg = estimate_segment_elasticities(prepared_df, segment_cols=("region",), controlled=True)

    assert np.isfinite(lin.elasticity)
    assert ctrl.elasticity < naive.elasticity  # controls remove between-segment confounding
    assert len(seg) == 4
    # North should be substantially more elastic than West
    north_el = float(seg.loc[seg["segment_value"] == "North", "elasticity"].iloc[0])
    west_el = float(seg.loc[seg["segment_value"] == "West", "elasticity"].iloc[0])
    assert north_el < west_el

    simpson = demonstrate_simpsons_paradox(prepared_df, group_col="region")
    assert simpson["paradox_detected"] is True


def test_iv_recovers_ground_truth_elasticity(prepared_df: pd.DataFrame) -> None:
    iv_res = estimate_2sls_iv(prepared_df)
    true_mean = float(prepared_df["true_elasticity"].mean())

    assert iv_res.first_stage_f_stat > 100.0
    assert iv_res.weak_instrument_flag is False
    # 2SLS should recover true elasticity closely and much better than naive OLS
    assert abs(iv_res.elasticity - true_mean) < 0.12
    assert abs(iv_res.elasticity - true_mean) < abs(iv_res.ols_comparison_elasticity - true_mean)


def test_iv_flags_weak_instrument_and_fails_on_zero_variation(prepared_df: pd.DataFrame) -> None:
    weak_raw = generate_synthetic_marketplace_data(n_rows=2000, seed=9, instrument_strength=0.002)
    weak_cln, _ = clean_transaction_data(weak_raw)
    weak_feat = add_engineered_features(weak_cln)
    weak_iv = estimate_2sls_iv(weak_feat)
    assert weak_iv.weak_instrument_flag is True

    zero_var = prepared_df.head(200).copy()
    zero_var["supplier_cost_shock"] = 1.0
    with pytest.raises(CausalAssumptionError, match="zero variation"):
        estimate_2sls_iv(zero_var)


def test_did_estimation_and_edge_cases(prepared_df: pd.DataFrame) -> None:
    did_res = estimate_difference_in_differences(prepared_df)
    assert did_res.ate_log_quantity > 0.05
    assert did_res.implied_did_elasticity is not None
    assert did_res.implied_did_elasticity < -0.5
    # Pre-treatment trends in synthetic DGP should be plausible (not wildly diverging)
    assert did_res.pre_trend_p_value > 0.01
    assert len(did_res.event_study_df) >= 5

    # Missing control group must fail clearly
    no_ctrl = prepared_df.loc[prepared_df["did_group"] == 1].copy()
    with pytest.raises(CausalAssumptionError, match="control"):
        estimate_difference_in_differences(no_ctrl)

    # Small sample must fail clearly
    with pytest.raises(CausalAssumptionError):
        estimate_difference_in_differences(prepared_df.head(8))


def test_causal_diagnostics_overlap_and_sensitivity(prepared_df: pd.DataFrame) -> None:
    ov = check_treatment_overlap_and_balance(prepared_df)
    assert ov.common_support_fraction > 0.85
    assert ov.positivity_satisfied is True
    assert ov.max_ipw_smd <= ov.max_raw_smd + 0.05

    sens = run_omitted_variable_sensitivity(prepared_df)
    assert sens.robustness_value_rv_q1 > 0.0
    assert not sens.sensitivity_grid.empty

    # Insufficient overlap / no control group fails
    no_treat_var = prepared_df.head(100).copy()
    no_treat_var["treatment_flag"] = 1
    with pytest.raises(CausalAssumptionError, match="Binary treatment variation"):
        check_treatment_overlap_and_balance(no_treat_var)
