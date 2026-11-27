"""Visualizations for CATE distributions, uplift curves, and AUUC comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_treatment_effect_distribution(
    cate_qty: np.ndarray,
    cate_margin: np.ndarray,
    out_path: str | Path,
) -> Path:
    """Plot 7: Distribution of predicted heterogeneous treatment effects (Quantity & Margin)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axes[0].hist(cate_qty, bins=40, color="#3367d6", alpha=0.82, edgecolor="white")
    axes[0].axvline(float(np.mean(cate_qty)), color="#111111", linestyle="--", label=f"Mean ATE = {np.mean(cate_qty):.3f}")
    axes[0].set_title("Predicted Quantity CATE Distribution")
    axes[0].set_xlabel("Expected Incremental Units (CATE)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend(frameon=False)

    axes[1].hist(cate_margin, bins=40, color="#0b8043", alpha=0.82, edgecolor="white")
    axes[1].axvline(0.0, color="#d62728", linestyle=":", linewidth=1.8, label="Break-Even Margin (0)")
    axes[1].axvline(float(np.mean(cate_margin)), color="#111111", linestyle="--", label=f"Mean ATE = {np.mean(cate_margin):.2f}")
    axes[1].set_title("Predicted Margin CATE Distribution")
    axes[1].set_xlabel("Expected Incremental Margin ($)")
    axes[1].set_ylabel("Frequency")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_uplift_curves(evaluations: dict[str, Any], out_path: str | Path) -> Path:
    """Plot 8: Cumulative Uplift / Qini Curves comparing S-Learner, T-Learner, and X-Learner."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    colors = {"S-Learner": "#7f7f7f", "T-Learner": "#ff7f0e", "X-Learner": "#0b8043"}
    rand_plotted = False

    for name, ev in evaluations.items():
        cdf = ev.curve_df
        ax.plot(
            cdf["fraction_targeted"] * 100.0,
            cdf["cumulative_uplift"],
            linewidth=2.3,
            label=f"{name} (AUUC={ev.auuc:,.0f})",
            color=colors.get(name, "#3367d6"),
        )
        if not rand_plotted:
            ax.plot(
                cdf["fraction_targeted"] * 100.0,
                cdf["random_baseline_uplift"],
                linestyle="--",
                color="#222222",
                linewidth=1.5,
                label="Random Targeting Baseline",
            )
            rand_plotted = True

    ax.set_xlabel("Population Targeted (%)")
    ax.set_ylabel("Cumulative Incremental Margin ($)")
    ax.set_title("Cumulative Uplift Curves (IPW-Adjusted Evaluation)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_auuc_comparison(comparison_df: pd.DataFrame, out_path: str | Path) -> Path:
    """Plot 9: AUUC, Qini Coefficient, and PEHE comparison across uplift models."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    models = comparison_df["model"].tolist()
    x = np.arange(len(models))
    palette = ["#7f7f7f", "#ff7f0e", "#0b8043"][: len(models)]

    axes[0].bar(x, comparison_df["auuc_normalized"], color=palette, width=0.52)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].set_ylabel("Normalized Excess AUUC")
    axes[0].set_title("Uplift Ranking Performance (Normalized AUUC)")

    if comparison_df["pehe"].notna().any():
        axes[1].bar(x, comparison_df["pehe"], color=palette, width=0.52)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models)
        axes[1].set_ylabel("PEHE (RMSE vs. True Margin CATE)")
        axes[1].set_title("CATE Accuracy (Lower PEHE is Better)")
    else:
        axes[1].bar(x, comparison_df["qini_coefficient"], color=palette, width=0.52)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models)
        axes[1].set_ylabel("Qini Coefficient")
        axes[1].set_title("Qini Coefficient Comparison")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
