"""Interactive Streamlit Dashboard for the Causal Pricing Intelligence & Optimization Engine."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from analysis.eda import run_eda
from evaluation.experiments import (
    run_experiment_1_elasticity_comparison,
    run_experiment_2_segment_heterogeneity,
    run_experiment_3_uplift_models,
    run_experiment_4_pricing_policies,
)
from optimization.constraints import PricingConstraints
from preprocessing.cleaning import clean_transaction_data
from preprocessing.feature_engineering import add_engineered_features
from simulation.data_generator import generate_synthetic_marketplace_data
from simulation.marketplace import describe_olist_causal_support, load_olist_dataset


st.set_page_config(
    page_title="Causal Pricing Intelligence Engine",
    layout="wide",
)

st.title("Causal Pricing Intelligence & Revenue Optimization Engine")
st.caption(
    "End-to-end causal elasticity, heterogeneous uplift modeling, and margin-constrained pricing optimization. "
    "All outputs are computed live from the implemented pipeline."
)

# Sidebar controls
st.sidebar.header("Pipeline Configuration")
dataset_choice = st.sidebar.selectbox("Dataset", ["Synthetic Marketplace", "Olist (data/raw/olist)"])
n_rows = st.sidebar.select_slider("Synthetic Rows", options=[5000, 10000, 25000, 50000], value=10000)
seed = st.sidebar.number_input("Random Seed", min_value=1, max_value=9999, value=42)
confounding_strength = st.sidebar.slider("Confounding Strength", 0.0, 2.0, 1.0, 0.25)

st.sidebar.subheader("Pricing Constraints")
min_margin_pct = st.sidebar.slider("Min Margin %", 0.0, 0.30, 0.08, 0.02)
max_coverage = st.sidebar.slider("Max Intervention Coverage", 0.10, 1.0, 1.0, 0.10)
segment_filter = st.sidebar.multiselect("Filter Regions", ["North", "South", "East", "West"], default=["North", "South", "East", "West"])

@st.cache_data(show_spinner="Generating & preprocessing dataset...")
def load_prepared_data(source: str, rows: int, s: int, conf: float) -> pd.DataFrame:
    if source.startswith("Olist"):
        raw = load_olist_dataset("data/raw/olist")
    else:
        raw = generate_synthetic_marketplace_data(n_rows=rows, seed=s, confounding_strength=conf)
    cleaned, _ = clean_transaction_data(raw)
    return add_engineered_features(cleaned)

try:
    df_full = load_prepared_data(dataset_choice, int(n_rows), int(seed), float(confounding_strength))
    df = df_full.loc[df_full["region"].isin(segment_filter)].reset_index(drop=True) if segment_filter and not dataset_choice.startswith("Olist") else df_full
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("Olist support documentation:")
    st.json(describe_olist_causal_support())
    st.stop()

tabs = st.tabs([
    "Overview & EDA",
    "Elasticity & Confounding",
    "Causal Inference (IV & DiD)",
    "Uplift Modeling",
    "Pricing Optimization & Backtest",
    "Assumptions & Diagnostics",
])

with tabs[0]:
    eda = run_eda(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{eda['n_rows']:,}")
    c2.metric("Unique Customers", f"{eda['n_customers']:,}")
    c3.metric("Total Revenue", f"${eda['total_revenue']:,.0f}")
    c4.metric("Total Margin", f"${eda['total_margin']:,.0f} ({eda['overall_margin_pct']:.1%})")
    st.subheader("Regional Breakdown")
    st.dataframe(eda["by_region"], use_container_width=True)
    st.subheader("Cohort Breakdown")
    st.dataframe(eda["by_cohort"], use_container_width=True)

if dataset_choice.startswith("Olist"):
    st.warning("Olist is an observational public dataset; causal IV/DiD/randomized uplift experiments are disabled where identification assumptions lack support.")
    st.json(describe_olist_causal_support())
else:
    exp1 = run_experiment_1_elasticity_comparison(df)
    with tabs[1]:
        st.subheader("Baseline & Segment Elasticity")
        st.dataframe(exp1["comparison_table"], use_container_width=True)
        st.subheader("Simpson's Paradox Analysis")
        st.info(exp1["simpsons_paradox"]["explanation"])
        st.dataframe(exp1["simpsons_paradox"]["within_segment_elasticities"], use_container_width=True)
        st.subheader("Confounder Screening")
        st.dataframe(exp1["confounding"]["confounder_screen"], use_container_width=True)

    with tabs[2]:
        iv = exp1["iv_result"]
        did = exp1["did_result"]
        c1, c2, c3 = st.columns(3)
        c1.metric("2SLS IV Elasticity", f"{iv.elasticity:.3f}", f"F = {iv.first_stage_f_stat:.1f}")
        c2.metric("DiD Log(Q) ATE", f"{did.ate_log_quantity:.3f}", f"Pre-trend p = {did.pre_trend_p_value:.3f}")
        c3.metric("DiD Implied Elasticity", f"{did.implied_did_elasticity:.3f}" if did.implied_did_elasticity else "N/A")
        st.caption(iv.assumptions_note)
        st.caption(did.assumptions_note)
        st.subheader("DiD Event-Study Coefficients")
        st.dataframe(did.event_study_df, use_container_width=True)

    with tabs[3]:
        if st.button("Run Uplift & Heterogeneity Benchmark", key="run_uplift"):
            exp2 = run_experiment_2_segment_heterogeneity(df, seed=int(seed))
            exp3 = run_experiment_3_uplift_models(df, seed=int(seed))
            st.subheader("Uplift Model Benchmark (AUUC, Qini, PEHE)")
            st.dataframe(exp3["comparison_table"], use_container_width=True)
            st.subheader("Segment-Level CATE Summary")
            st.dataframe(exp2["segment_summary"], use_container_width=True)
        else:
            st.info("Click 'Run Uplift & Heterogeneity Benchmark' to train S-Learner, T-Learner, and X-Learner live.")

    with tabs[4]:
        if st.button("Run Pricing Optimization & Chronological Backtest", key="run_opt"):
            constraints = PricingConstraints(
                min_margin_pct=float(min_margin_pct),
                max_treatment_coverage=float(max_coverage),
            )
            exp4 = run_experiment_4_pricing_policies(df, constraints=constraints, seed=int(seed))
            st.subheader("Held-Out Test Period Observed Baseline")
            st.json(exp4["observed_summary"])
            st.subheader("Policy Comparison (Synthetic Ground-Truth Counterfactuals)")
            st.dataframe(exp4["oracle_comparison"], use_container_width=True)
            st.subheader("Policy Comparison (Model-Estimated Counterfactuals)")
            st.dataframe(exp4["model_comparison"], use_container_width=True)
        else:
            st.info("Click 'Run Pricing Optimization & Chronological Backtest' to evaluate all 4 policies out-of-time.")

    with tabs[5]:
        diag = exp1["diagnostics"]
        st.subheader("Positivity & Covariate Balance")
        st.write(
            f"Common support fraction [0.05, 0.95]: **{diag['overlap_balance'].common_support_fraction:.2%}** | "
            f"Max raw SMD: **{diag['overlap_balance'].max_raw_smd:.3f}** -> IPW SMD: **{diag['overlap_balance'].max_ipw_smd:.3f}**"
        )
        st.dataframe(diag["overlap_balance"].balance_table, use_container_width=True)
        st.subheader("Omitted-Variable Sensitivity (Cinelli & Hazlett RV)")
        st.info(diag["sensitivity"].interpretation)
        st.subheader("Testable vs. Untestable Causal Assumptions")
        st.dataframe(diag["assumptions_registry"], use_container_width=True)
