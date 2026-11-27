"""Visualization package generating all 14+ figures and interactive plots."""

from pathlib import Path
from typing import Any

import pandas as pd

from analysis.confounding import build_marketplace_dag
from config.loader import resolve_path
from visualization.causal_plots import (
    plot_confounding_analysis,
    plot_did_event_study,
    plot_simpsons_paradox,
)
from visualization.elasticity_plots import (
    plot_elasticity_by_segment,
    plot_naive_vs_causal_elasticity,
    plot_price_vs_demand,
)
from visualization.pricing_plots import (
    export_interactive_plotly_figures,
    plot_backtesting_results,
    plot_incremental_revenue_and_margin,
    plot_policy_metric_bar,
    plot_robustness_analysis,
)
from visualization.uplift_plots import (
    plot_auuc_comparison,
    plot_treatment_effect_distribution,
    plot_uplift_curves,
)


def generate_all_visualizations(
    df: pd.DataFrame,
    results: dict[str, Any],
    figures_dir: str | Path = "reports/figures",
) -> dict[str, Path]:
    """Generate all required static Matplotlib PNGs and interactive Plotly HTML charts."""
    fig_dir = resolve_path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, Path] = {}

    generated["01_price_vs_demand"] = plot_price_vs_demand(df, fig_dir / "01_price_vs_demand.png")

    if "exp1" in results:
        exp1 = results["exp1"]
        generated["02_naive_vs_causal_elasticity"] = plot_naive_vs_causal_elasticity(
            exp1["comparison_table"], fig_dir / "02_naive_vs_causal_elasticity.png"
        )
        generated["03_elasticity_by_region"] = plot_elasticity_by_segment(
            exp1["segment_elasticities"],
            "region",
            fig_dir / "03_elasticity_by_region.png",
            "Controlled Price Elasticity by Region vs. Ground Truth",
        )
        generated["04_elasticity_by_cohort"] = plot_elasticity_by_segment(
            exp1["segment_elasticities"],
            "customer_cohort",
            fig_dir / "04_elasticity_by_cohort.png",
            "Controlled Price Elasticity by Customer Cohort vs. Ground Truth",
        )
        generated["05_confounding_analysis"] = plot_confounding_analysis(
            exp1["confounding"], build_marketplace_dag(), fig_dir / "05_confounding_analysis.png"
        )
        generated["06_simpsons_paradox"] = plot_simpsons_paradox(
            df, exp1["simpsons_paradox"], fig_dir / "06_simpsons_paradox.png"
        )
        generated["15_did_event_study"] = plot_did_event_study(
            exp1["did_result"].event_study_df, fig_dir / "15_did_event_study.png"
        )

    if "exp2" in results:
        exp2 = results["exp2"]
        generated["07_treatment_effect_distribution"] = plot_treatment_effect_distribution(
            exp2["cate_qty"], exp2["cate_margin"], fig_dir / "07_treatment_effect_distribution.png"
        )

    if "exp3" in results:
        exp3 = results["exp3"]
        generated["08_uplift_curve"] = plot_uplift_curves(
            exp3["evaluations"], fig_dir / "08_uplift_curve.png"
        )
        generated["09_auuc_comparison"] = plot_auuc_comparison(
            exp3["comparison_table"], fig_dir / "09_auuc_comparison.png"
        )

    if "exp4" in results:
        exp4 = results["exp4"]
        pol_df = (
            exp4["oracle_comparison"]
            if not exp4["oracle_comparison"].empty
            else exp4["model_comparison"]
        )
        generated["10_policy_revenue_comparison"] = plot_policy_metric_bar(
            pol_df,
            "total_revenue",
            "Pricing Policy Total Revenue Comparison",
            "Total Revenue ($)",
            fig_dir / "10_policy_revenue_comparison.png",
        )
        generated["11_policy_margin_comparison"] = plot_policy_metric_bar(
            pol_df,
            "total_margin",
            "Pricing Policy Total Margin Comparison",
            "Total Margin ($)",
            fig_dir / "11_policy_margin_comparison.png",
        )
        generated["12_incremental_revenue_margin"] = plot_incremental_revenue_and_margin(
            pol_df, fig_dir / "12_incremental_revenue_margin.png"
        )
        generated["13_backtesting_results"] = plot_backtesting_results(
            exp4["trajectory_df"], fig_dir / "13_backtesting_results.png"
        )
        if "exp1" in results:
            html_paths = export_interactive_plotly_figures(
                pol_df, results["exp1"]["segment_elasticities"], fig_dir / "interactive"
            )
            for idx, hp in enumerate(html_paths, start=1):
                generated[f"interactive_{idx}"] = hp

    if "robustness" in results:
        generated["14_robustness_analysis"] = plot_robustness_analysis(
            results["robustness"], fig_dir / "14_robustness_analysis.png"
        )

    return generated
