"""Main CLI entry point for the Causal Pricing Intelligence & Optimization Engine."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.eda import run_eda
from analysis.elasticity import run_elasticity_comparison
from config.loader import ensure_directories, get_logger, load_config, resolve_path
from evaluation.experiments import (
    run_experiment_1_elasticity_comparison,
    run_experiment_2_segment_heterogeneity,
    run_experiment_3_uplift_models,
    run_experiment_4_pricing_policies,
    save_experiment_outputs,
)
from evaluation.robustness import bootstrap_causal_elasticity, run_robustness_suite
from optimization.constraints import PricingConstraints
from preprocessing.cleaning import clean_transaction_data
from preprocessing.feature_engineering import add_engineered_features
from simulation.data_generator import generate_synthetic_marketplace_data
from simulation.marketplace import load_olist_dataset
from visualization import generate_all_visualizations

logger = get_logger("main")


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavored Markdown table without external dependencies."""
    if df.empty:
        return "_No rows._"
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        vals = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:,.4f}" if abs(v) < 10 else f"{v:,.2f}")
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *rows])


def _write_markdown_report(
    results: dict[str, Any],
    eda_res: dict[str, Any],
    out_path: str | Path = "reports/experiment_report.md",
) -> Path:
    """Generate a concise Markdown summary of actual measured experimental outputs."""
    out = resolve_path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Causal Pricing Intelligence — Automated Experiment Report",
        "",
        "## Dataset Overview",
        f"- Transactions analyzed: **{eda_res['n_rows']:,}**",
        f"- Unique customers: **{eda_res['n_customers']:,}** | Unique products: **{eda_res['n_products']:,}**",
        f"- Time window: `{eda_res['date_min']}` to `{eda_res['date_max']}`",
        f"- Observed total revenue: **\\${eda_res['total_revenue']:,.2f}** | Total margin: **\\${eda_res['total_margin']:,.2f}** ({eda_res['overall_margin_pct']:.1%})",
        "",
    ]

    if "exp1" in results:
        lines.extend(
            [
                "## Experiment 1: Naive vs. Causal Elasticity",
                _df_to_markdown(results["exp1"]["comparison_table"]),
                "",
                f"- First-stage IV F-statistic: **{results['exp1']['iv_result'].first_stage_f_stat:.2f}** (Weak={results['exp1']['iv_result'].weak_instrument_flag})",
                f"- DiD pre-trend slope p-value: **{results['exp1']['did_result'].pre_trend_p_value:.4f}**",
                f"- Simpson's Paradox detected: **{results['exp1']['simpsons_paradox']['paradox_detected']}**",
                "",
            ]
        )

    if "exp2" in results:
        lines.extend(
            [
                "## Experiment 2: Segment Heterogeneity (ATE vs. CATE)",
                f"- Overall Quantity ATE: **{results['exp2']['ate_quantity']:.4f}** | Margin ATE: **\\${results['exp2']['ate_margin']:.2f}** (SD={results['exp2']['cate_margin_std']:.2f})",
                "",
                _df_to_markdown(results["exp2"]["segment_summary"].head(12)),
                "",
            ]
        )

    if "exp3" in results:
        lines.extend(
            [
                "## Experiment 3: Uplift Model Benchmark",
                _df_to_markdown(results["exp3"]["comparison_table"]),
                "",
            ]
        )

    if "exp4" in results:
        pol_df = (
            results["exp4"]["oracle_comparison"]
            if not results["exp4"]["oracle_comparison"].empty
            else results["exp4"]["model_comparison"]
        )
        lines.extend(
            [
                "## Experiment 4: Chronological Backtest & Pricing Policy Comparison",
                f"- Held-out test rows: **{results['exp4']['observed_summary']['test_rows']:,}**",
                f"- Observed historical test revenue: **\\${results['exp4']['observed_summary']['observed_test_revenue']:,.2f}** | margin: **\\${results['exp4']['observed_summary']['observed_test_margin']:,.2f}**",
                "",
                _df_to_markdown(
                    pol_df[
                        [
                            "policy",
                            "total_revenue",
                            "total_margin",
                            "incremental_revenue",
                            "incremental_margin",
                            "incremental_orders",
                            "discount_expenditure",
                            "treatment_coverage",
                        ]
                    ]
                ),
                "",
                "> Note: Counterfactual policy metrics on synthetic data reflect the synthetic marketplace ground-truth simulator, not real-world business impact.",
                "",
            ]
        )

    if "robustness" in results:
        lines.extend(
            [
                "## Robustness Analysis: Confounding Sensitivity",
                _df_to_markdown(results["robustness"]["confounding_sensitivity"]),
                "",
            ]
        )

    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote Markdown experiment report to %s", out)
    return out


def prepare_dataset(
    cfg: dict[str, Any],
    n_rows: int,
    seed: int,
    dataset_source: str = "synthetic",
) -> pd.DataFrame:
    """Generate/load, clean, validate, and engineer features for the pipeline."""
    if dataset_source == "olist":
        raw_df = load_olist_dataset(resolve_path(cfg["paths"]["raw_dir"]) / "olist")
    else:
        sim = cfg["simulation"]
        synth_path = resolve_path(cfg["paths"]["synthetic_dir"]) / f"marketplace_{n_rows}.csv"
        raw_df = generate_synthetic_marketplace_data(
            n_rows=n_rows,
            seed=seed,
            confounding_strength=float(sim["confounding_strength"]),
            instrument_strength=float(sim["instrument_strength"]),
            did_start_day=int(sim["did_start_day"]),
            did_price_drop=float(sim["did_price_drop"]),
            promo_discount_depth=float(sim["promo_discount_depth"]),
            treatment_assignment=str(sim["treatment_assignment"]),
            n_customers=int(sim["n_customers"]),
            n_products=int(sim["n_products"]),
            start_date=str(sim["start_date"]),
            n_days=int(sim["n_days"]),
            save_path=synth_path,
        )

    prep_cfg = cfg["preprocessing"]
    cleaned_df, _ = clean_transaction_data(
        raw_df,
        outlier_quantile_low=float(prep_cfg["outlier_quantile_low"]),
        outlier_quantile_high=float(prep_cfg["outlier_quantile_high"]),
    )
    feat_df = add_engineered_features(cleaned_df)
    proc_path = resolve_path(cfg["paths"]["processed_dir"]) / f"marketplace_{dataset_source}_{len(feat_df)}.csv"
    feat_df.to_csv(proc_path, index=False)
    logger.info("Prepared processed dataset (%d rows) at %s", len(feat_df), proc_path)
    return feat_df


def run_pipeline(
    experiment: str = "all",
    n_rows: int | None = None,
    seed: int | None = None,
    dataset_source: str = "synthetic",
    config_path: str | None = None,
) -> dict[str, Any]:
    """Execute requested stage(s) of the Causal Pricing Intelligence pipeline."""
    cfg = load_config(config_path)
    ensure_directories(cfg)

    seed_val = int(seed if seed is not None else cfg["project"]["seed"])
    rows_val = int(n_rows if n_rows is not None else cfg["project"]["default_n_rows"])
    df = prepare_dataset(cfg, n_rows=rows_val, seed=seed_val, dataset_source=dataset_source)

    eda_res = run_eda(df)
    results: dict[str, Any] = {"eda": eda_res}

    if experiment == "data":
        logger.info("Data preparation complete (%d rows).", len(df))
        return results

    if experiment in ("elasticity", "causal", "all"):
        if experiment == "elasticity":
            logger.info("Running baseline elasticity comparison...")
            results["elasticity"] = run_elasticity_comparison(df)
        else:
            logger.info("Running Experiment 1 (Naive vs. Causal Elasticity)...")
            results["exp1"] = run_experiment_1_elasticity_comparison(df)
            logger.info("Running Experiment 2 (Segment Heterogeneity)...")
            results["exp2"] = run_experiment_2_segment_heterogeneity(
                df, learner_type=cfg["uplift"]["base_learner"], seed=seed_val
            )

    if experiment in ("uplift", "all"):
        logger.info("Running Experiment 3 (Uplift Model Benchmark)...")
        results["exp3"] = run_experiment_3_uplift_models(
            df, learner_type=cfg["uplift"]["base_learner"], seed=seed_val
        )

    if experiment in ("optimization", "backtest", "all"):
        logger.info("Running Experiment 4 (Pricing Optimization & Chronological Backtest)...")
        c_cfg = cfg["optimization"]["constraints"]
        constraints = PricingConstraints(
            min_margin_pct=float(c_cfg["min_margin_pct"]),
            max_discount=float(c_cfg["max_discount"]),
            min_price=float(c_cfg["min_price"]),
            max_price=float(c_cfg["max_price"]),
            max_price_change_pct=float(c_cfg["max_price_change_pct"]),
            min_seller_availability=float(c_cfg["min_seller_availability"]),
            max_treatment_coverage=float(c_cfg["max_treatment_coverage"]),
        )
        results["exp4"] = run_experiment_4_pricing_policies(df, constraints=constraints, seed=seed_val)

    if experiment in ("robustness", "all"):
        logger.info("Running bootstrap uncertainty & robustness stress-test suite...")
        results["bootstrap"] = bootstrap_causal_elasticity(df, n_bootstrap=30, seed=seed_val)
        results["robustness"] = run_robustness_suite(seed=seed_val, n_rows=min(12_000, rows_val))

    exp_dir = cfg["paths"]["experiments_dir"]
    save_experiment_outputs(results, out_dir=exp_dir)
    if "robustness" in results:
        for name, rdf in results["robustness"].items():
            rdf.to_csv(resolve_path(exp_dir) / f"robustness_{name}.csv", index=False)
    if "bootstrap" in results:
        results["bootstrap"]["summary_df"].to_csv(
            resolve_path(exp_dir) / "bootstrap_elasticity_ci.csv", index=False
        )

    figures = generate_all_visualizations(df, results, figures_dir=cfg["paths"]["figures_dir"])
    report_path = _write_markdown_report(results, eda_res)
    results["figures"] = figures
    results["report_path"] = report_path
    logger.info("Pipeline '%s' finished. Generated %d figures and report at %s", experiment, len(figures), report_path)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Causal Pricing Intelligence & Optimization Engine")
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", "data", "elasticity", "causal", "uplift", "optimization", "backtest", "robustness"],
        help="Pipeline stage or experiment to execute.",
    )
    parser.add_argument("--n-rows", type=int, default=None, help="Number of synthetic transactions to generate.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=["synthetic", "olist"],
        help="Dataset source to use.",
    )
    parser.add_argument("--config", type=str, default=None, help="Optional path to YAML config file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        experiment=args.experiment,
        n_rows=args.n_rows,
        seed=args.seed,
        dataset_source=args.dataset,
        config_path=args.config,
    )
