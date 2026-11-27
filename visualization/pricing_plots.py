"""Visualizations for pricing policy comparison, backtesting, and robustness analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px


def plot_policy_metric_bar(
    policy_df: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    out_path: str | Path,
) -> Path:
    """Plots 10 & 11: Pricing policy revenue or margin bar comparison."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dfp = policy_df.sort_values(metric_col, ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    colors = ["#9aa0a6", "#ff7f0e", "#3367d6", "#0b8043"][: len(dfp)]
    bars = ax.barh(dfp["policy"], dfp[metric_col], color=colors, height=0.55)
    for bar in bars:
        w = bar.get_width()
        ax.text(w * 1.005, bar.get_y() + bar.get_height() / 2, f"${w:,.0f}", va="center", fontsize=9)
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0, float(dfp[metric_col].max()) * 1.15)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_incremental_revenue_and_margin(policy_df: pd.DataFrame, out_path: str | Path) -> Path:
    """Plot 12: Incremental revenue vs incremental margin across pricing policies."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(policy_df))
    w = 0.36

    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.bar(x - w / 2, policy_df["incremental_revenue"], width=w, label="Incremental Revenue ($)", color="#3367d6")
    ax.bar(x + w / 2, policy_df["incremental_margin"], width=w, label="Incremental Margin ($)", color="#0b8043")
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(policy_df["policy"], rotation=12)
    ax.set_ylabel("Incremental Financial Lift ($)")
    ax.set_title("Incremental Revenue vs. Incremental Margin by Policy (vs. Baseline)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_backtesting_results(trajectory_df: pd.DataFrame, out_path: str | Path) -> Path:
    """Plot 13: Chronological out-of-time cumulative margin trajectory across policies."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    palette = {
        "Flat Policy": "#7f7f7f",
        "Random Policy": "#ff7f0e",
        "Segment-Based Policy": "#3367d6",
        "Uplift-Based Policy": "#0b8043",
    }
    for pol, grp in trajectory_df.groupby("policy"):
        grp = grp.sort_values("week").copy()
        grp["cum_margin"] = grp["margin"].cumsum()
        ax.plot(
            np.arange(len(grp)),
            grp["cum_margin"],
            marker="o",
            markersize=3.5,
            linewidth=2.2,
            label=pol,
            color=palette.get(pol, "#222222"),
        )
    weeks = sorted(trajectory_df["week"].unique())
    step = max(1, len(weeks) // 6)
    ax.set_xticks(np.arange(0, len(weeks), step))
    ax.set_xticklabels(weeks[::step], rotation=20)
    ax.set_xlabel("Held-Out Test Week")
    ax.set_ylabel("Cumulative Margin ($)")
    ax.set_title("Chronological Backtest: Held-Out Cumulative Margin Trajectory")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_robustness_analysis(robust_res: dict[str, pd.DataFrame], out_path: str | Path) -> Path:
    """Plot 14: Estimator robustness across confounding strength and sample sizes."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    conf_df = robust_res["confounding_sensitivity"]
    size_df = robust_res["sample_size_scaling"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(conf_df["confounding_strength"], conf_df["naive_abs_error"], marker="o", label="Naive OLS Error", color="#d62728")
    axes[0].plot(conf_df["confounding_strength"], conf_df["controlled_abs_error"], marker="s", label="Controlled OLS Error", color="#ff7f0e")
    axes[0].plot(conf_df["confounding_strength"], conf_df["iv_abs_error"], marker="^", label="2SLS IV Error", color="#0b8043")
    axes[0].set_xlabel("Confounding Strength Multiplier")
    axes[0].set_ylabel("|Estimated Elasticity - True Elasticity|")
    axes[0].set_title("Elasticity Bias Under Increasing Unobserved Confounding")
    axes[0].legend(frameon=False)

    axes[1].errorbar(
        size_df["sample_size"],
        size_df["iv_2sls"],
        yerr=1.96 * size_df["iv_se"],
        marker="o",
        linewidth=2,
        capsize=4,
        color="#3367d6",
        label="2SLS IV +/- 1.96 SE",
    )
    axes[1].plot(size_df["sample_size"], size_df["true_elasticity"], linestyle="--", color="#111111", label="True Elasticity")
    axes[1].set_xlabel("Sample Size (Transactions)")
    axes[1].set_ylabel("Price Elasticity")
    axes[1].set_title("2SLS IV Convergence & Uncertainty Across Sample Sizes")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def export_interactive_plotly_figures(
    policy_df: pd.DataFrame,
    segment_df: pd.DataFrame,
    out_dir: str | Path,
) -> list[Path]:
    """Export interactive Plotly HTML figures for policy comparison and segment elasticity."""
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig1 = px.bar(
        policy_df,
        x="policy",
        y=["incremental_revenue", "incremental_margin"],
        barmode="group",
        title="Interactive Pricing Policy Comparison: Incremental Revenue & Margin",
    )
    p1 = base / "policy_comparison.html"
    fig1.write_html(str(p1), include_plotlyjs="cdn")
    paths.append(p1)

    if not segment_df.empty:
        fig2 = px.bar(
            segment_df,
            x="segment_value",
            y="elasticity",
            color="segment_type",
            barmode="group",
            title="Interactive Segment-Level Price Elasticity",
        )
        p2 = base / "segment_elasticity.html"
        fig2.write_html(str(p2), include_plotlyjs="cdn")
        paths.append(p2)

    return paths
