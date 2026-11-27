"""Four structured experiments specified in the project methodology."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.confounding import analyze_correlations_and_confounders, demonstrate_simpsons_paradox
from analysis.elasticity import run_elasticity_comparison
from causal.diagnostics import run_causal_diagnostics_report
from causal.did import estimate_difference_in_differences
from causal.iv import estimate_2sls_iv
from config.loader import get_logger, resolve_path
from evaluation.backtest import chronological_train_test_split, run_chronological_backtest
from optimization.constraints import PricingConstraints
from preprocessing.feature_engineering import build_feature_matrix
from uplift.base import SLearner
from uplift.evaluation import evaluate_uplift_model
from uplift.segmentation import summarize_segment_uplift
from uplift.t_learner import TLearner
from uplift.x_learner import XLearner

logger = get_logger(__name__)


def run_experiment_1_elasticity_comparison(df: pd.DataFrame) -> dict[str, Any]:
    """Experiment 1: Compare Naive OLS, Controlled OLS, 2SLS IV, and DiD elasticity."""
    base_res = run_elasticity_comparison(df)
    iv_res = estimate_2sls_iv(df)
    did_res = estimate_difference_in_differences(df)
    conf_res = analyze_correlations_and_confounders(df)
    simpson_res = demonstrate_simpsons_paradox(df, group_col="region")

    true_all = float(df["true_elasticity"].mean()) if "true_elasticity" in df.columns else None
    true_did = (
        float(df.loc[df["did_group"] == 1, "true_elasticity"].mean())
        if "true_elasticity" in df.columns and "did_group" in df.columns
        else None
    )

    rows = [
        {
            "method": "Linear OLS (Mean Elasticity)",
            "elasticity": base_res["linear"].elasticity,
            "std_error": base_res["linear"].std_error,
            "ci_low": base_res["linear"].ci_low,
            "ci_high": base_res["linear"].ci_high,
            "true_reference": true_all,
            "bias": (base_res["linear"].elasticity - true_all) if true_all is not None else None,
        },
        {
            "method": "Naive Log-Log OLS",
            "elasticity": base_res["naive_loglog"].elasticity,
            "std_error": base_res["naive_loglog"].std_error,
            "ci_low": base_res["naive_loglog"].ci_low,
            "ci_high": base_res["naive_loglog"].ci_high,
            "true_reference": true_all,
            "bias": (base_res["naive_loglog"].elasticity - true_all) if true_all is not None else None,
        },
        {
            "method": "Controlled Log-Log OLS",
            "elasticity": base_res["controlled_loglog"].elasticity,
            "std_error": base_res["controlled_loglog"].std_error,
            "ci_low": base_res["controlled_loglog"].ci_low,
            "ci_high": base_res["controlled_loglog"].ci_high,
            "true_reference": true_all,
            "bias": (base_res["controlled_loglog"].elasticity - true_all) if true_all is not None else None,
        },
        {
            "method": "Causal 2SLS IV",
            "elasticity": iv_res.elasticity,
            "std_error": iv_res.std_error,
            "ci_low": iv_res.ci_low,
            "ci_high": iv_res.ci_high,
            "true_reference": true_all,
            "bias": (iv_res.elasticity - true_all) if true_all is not None else None,
        },
        {
            "method": "Causal DiD (Implied Treated Elasticity)",
            "elasticity": did_res.implied_did_elasticity,
            "std_error": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "true_reference": true_did,
            "bias": (
                (did_res.implied_did_elasticity - true_did)
                if (did_res.implied_did_elasticity is not None and true_did is not None)
                else None
            ),
        },
    ]

    diagnostics = run_causal_diagnostics_report(df, iv_result=iv_res, did_result=did_res)

    return {
        "comparison_table": pd.DataFrame(rows),
        "segment_elasticities": base_res["segment_controlled"],
        "iv_result": iv_res,
        "did_result": did_res,
        "confounding": conf_res,
        "simpsons_paradox": simpson_res,
        "diagnostics": diagnostics,
    }


def run_experiment_2_segment_heterogeneity(
    df: pd.DataFrame,
    learner_type: str = "gradient_boosting",
    seed: int = 42,
) -> dict[str, Any]:
    """Experiment 2: Compare overall ATE vs segment-level CATE heterogeneity."""
    train_df, test_df = chronological_train_test_split(df, train_fraction=0.75)
    X_train, cols = build_feature_matrix(train_df, include_price=False)
    X_test, _ = build_feature_matrix(test_df, include_price=False)
    X_test = X_test.reindex(columns=cols, fill_value=0.0)

    t_train = train_df["treatment_flag"].to_numpy(dtype=int)
    xl_qty = XLearner(learner_type=learner_type, seed=seed).fit(X_train, t_train, train_df["quantity"])
    xl_margin = XLearner(learner_type=learner_type, seed=seed + 1).fit(X_train, t_train, train_df["margin"])

    cate_qty = xl_qty.predict_cate(X_test)
    cate_margin = xl_margin.predict_cate(X_test)

    seg_summary = summarize_segment_uplift(test_df, cate_qty=cate_qty, cate_margin=cate_margin)
    return {
        "ate_quantity": float(np.mean(cate_qty)),
        "ate_margin": float(np.mean(cate_margin)),
        "cate_qty_std": float(np.std(cate_qty)),
        "cate_margin_std": float(np.std(cate_margin)),
        "segment_summary": seg_summary,
        "cate_qty": cate_qty,
        "cate_margin": cate_margin,
        "test_df": test_df,
    }


def run_experiment_3_uplift_models(
    df: pd.DataFrame,
    learner_type: str = "gradient_boosting",
    seed: int = 42,
) -> dict[str, Any]:
    """Experiment 3: Benchmark S-Learner baseline, T-Learner, and X-Learner on margin uplift."""
    train_df, test_df = chronological_train_test_split(df, train_fraction=0.75)
    X_train, cols = build_feature_matrix(train_df, include_price=False)
    X_test, _ = build_feature_matrix(test_df, include_price=False)
    X_test = X_test.reindex(columns=cols, fill_value=0.0)

    t_train = train_df["treatment_flag"].to_numpy(dtype=int)
    y_train = train_df["margin"].to_numpy(dtype=float)

    t_test = test_df["treatment_flag"].to_numpy(dtype=int)
    y_test = test_df["margin"].to_numpy(dtype=float)
    true_cate = test_df["true_cate_margin"].to_numpy(dtype=float) if "true_cate_margin" in test_df.columns else None

    models = [
        SLearner(learner_type=learner_type, seed=seed),
        TLearner(learner_type=learner_type, seed=seed),
        XLearner(learner_type=learner_type, seed=seed),
    ]

    # Fit XLearner first to supply estimated propensity for IPW-adjusted observational uplift curve
    fitted: dict[str, Any] = {}
    for m in models:
        m.fit(X_train, t_train, y_train)
        fitted[m.name] = m

    propensity_test = fitted["X-Learner"].predict_propensity(X_test)

    eval_results: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for name, m in fitted.items():
        preds = m.predict_cate(X_test)
        ev = evaluate_uplift_model(
            model_name=name,
            y=y_test,
            treatment=t_test,
            cate_pred=preds,
            propensity=propensity_test,
            true_cate=true_cate,
        )
        eval_results[name] = ev
        summary_rows.append(
            {
                "model": name,
                "auuc": ev.auuc,
                "auuc_normalized": ev.auuc_normalized,
                "qini_coefficient": ev.qini_coefficient,
                "top_decile_uplift": ev.top_decile_uplift,
                "bottom_decile_uplift": ev.bottom_decile_uplift,
                "estimated_ate": ev.estimated_ate,
                "pehe": ev.pehe,
                "ate_abs_error": ev.ate_abs_error,
                "cate_rank_corr": ev.cate_rank_corr,
            }
        )

    return {
        "comparison_table": pd.DataFrame(summary_rows),
        "evaluations": eval_results,
        "test_df": test_df,
    }


def run_experiment_4_pricing_policies(
    df: pd.DataFrame,
    constraints: PricingConstraints = PricingConstraints(),
    seed: int = 42,
) -> dict[str, Any]:
    """Experiment 4: Chronological backtest comparing the four pricing policies."""
    return run_chronological_backtest(df, train_fraction=0.75, constraints=constraints, seed=seed)


def save_experiment_outputs(
    results: dict[str, Any],
    out_dir: str | Path = "reports/experiments",
) -> dict[str, Path]:
    """Persist machine-readable CSV and JSON outputs for all executed experiments."""
    base = resolve_path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    if "exp1" in results:
        p_csv = base / "exp1_elasticity_comparison.csv"
        results["exp1"]["comparison_table"].to_csv(p_csv, index=False)
        saved["exp1_comparison"] = p_csv

        p_seg = base / "exp1_segment_elasticities.csv"
        results["exp1"]["segment_elasticities"].to_csv(p_seg, index=False)
        saved["exp1_segments"] = p_seg

        p_es = base / "exp1_did_event_study.csv"
        results["exp1"]["did_result"].event_study_df.to_csv(p_es, index=False)
        saved["exp1_event_study"] = p_es

    if "exp2" in results:
        p_seg2 = base / "exp2_segment_heterogeneity.csv"
        results["exp2"]["segment_summary"].to_csv(p_seg2, index=False)
        saved["exp2_segments"] = p_seg2

    if "exp3" in results:
        p_up = base / "exp3_uplift_comparison.csv"
        results["exp3"]["comparison_table"].to_csv(p_up, index=False)
        saved["exp3_uplift"] = p_up

    if "exp4" in results:
        p_pol_m = base / "exp4_policy_model_comparison.csv"
        results["exp4"]["model_comparison"].to_csv(p_pol_m, index=False)
        saved["exp4_model"] = p_pol_m
        if not results["exp4"]["oracle_comparison"].empty:
            p_pol_o = base / "exp4_policy_ground_truth_comparison.csv"
            results["exp4"]["oracle_comparison"].to_csv(p_pol_o, index=False)
            saved["exp4_oracle"] = p_pol_o

    # Consolidated JSON summary for quick programmatic inspection
    summary_payload: dict[str, Any] = {}
    if "exp1" in results:
        summary_payload["exp1_elasticity"] = results["exp1"]["comparison_table"].to_dict(orient="records")
        summary_payload["exp1_iv_first_stage_f"] = results["exp1"]["iv_result"].first_stage_f_stat
        summary_payload["exp1_did_pretrend_p"] = results["exp1"]["did_result"].pre_trend_p_value
        summary_payload["exp1_simpsons_paradox"] = {
            "pooled": results["exp1"]["simpsons_paradox"]["pooled_elasticity"],
            "paradox_detected": results["exp1"]["simpsons_paradox"]["paradox_detected"],
        }
    if "exp2" in results:
        summary_payload["exp2_ate"] = {
            "ate_quantity": results["exp2"]["ate_quantity"],
            "ate_margin": results["exp2"]["ate_margin"],
            "cate_margin_std": results["exp2"]["cate_margin_std"],
        }
    if "exp3" in results:
        summary_payload["exp3_uplift"] = results["exp3"]["comparison_table"].to_dict(orient="records")
    if "exp4" in results:
        summary_payload["exp4_observed_test"] = results["exp4"]["observed_summary"]
        if not results["exp4"]["oracle_comparison"].empty:
            summary_payload["exp4_oracle_policies"] = results["exp4"]["oracle_comparison"].to_dict(orient="records")
        summary_payload["exp4_model_policies"] = results["exp4"]["model_comparison"].to_dict(orient="records")

    p_json = base / "experiments_summary.json"
    with p_json.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, default=str)
    saved["summary_json"] = p_json
    logger.info("Saved experiment reports to %s", base)
    return saved
