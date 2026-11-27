"""Publication-quality plots for price-demand relationships and elasticity comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update(
    {
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


def plot_price_vs_demand(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Plot 1: Price vs Demand scatter & binned trend across regions."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample = df.sample(min(4000, len(df)), random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"North": "#1f77b4", "East": "#ff7f0e", "South": "#2ca02c", "West": "#d62728"}
    for reg, grp in sample.groupby("region"):
        axes[0].scatter(
            grp["price"],
            grp["quantity"],
            alpha=0.28,
            s=14,
            label=reg,
            color=colors.get(reg, "#555555"),
        )
    axes[0].set_title("Transaction Price vs. Quantity by Region")
    axes[0].set_xlabel("Price ($)")
    axes[0].set_ylabel("Quantity")
    axes[0].legend(frameon=False)

    # Log-log binned means
    sample = sample.copy()
    sample["log_p_bin"] = pd.qcut(sample["log_price"], q=15, duplicates="drop")
    binned = sample.groupby("log_p_bin", observed=True)[["log_price", "log_quantity"]].mean()
    axes[1].plot(binned["log_price"], binned["log_quantity"], marker="o", linewidth=2.2, color="#2b5c8f")
    axes[1].set_title("Aggregate Binned Log(Price) vs. Log(Quantity)")
    axes[1].set_xlabel("log(Price)")
    axes[1].set_ylabel("Mean log(Quantity)")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_naive_vs_causal_elasticity(comparison_df: pd.DataFrame, out_path: str | Path) -> Path:
    """Plot 2: Naive OLS vs Controlled OLS vs 2SLS IV vs DiD elasticity comparison."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_df = comparison_df.dropna(subset=["elasticity"]).copy()

    fig, ax = plt.subplots(figsize=(9.5, 5))
    y_pos = np.arange(len(plot_df))
    palette = ["#9aa0a6", "#e67c73", "#f6bf26", "#0b8043", "#3367d6"][: len(plot_df)]

    ax.barh(y_pos, plot_df["elasticity"], color=palette, alpha=0.9, height=0.58)
    for i, row in enumerate(plot_df.itertuples()):
        if np.isfinite(row.ci_low) and np.isfinite(row.ci_high):
            ax.errorbar(
                row.elasticity,
                i,
                xerr=[[row.elasticity - row.ci_low], [row.ci_high - row.elasticity]],
                fmt="none",
                ecolor="#222222",
                capsize=4,
                linewidth=1.4,
            )
        ax.text(
            row.elasticity - 0.03 if row.elasticity < 0 else row.elasticity + 0.03,
            i,
            f"{row.elasticity:.3f}",
            va="center",
            ha="right" if row.elasticity < 0 else "left",
            fontsize=9,
            fontweight="bold",
        )

    if "true_reference" in plot_df.columns and plot_df["true_reference"].notna().any():
        true_val = float(plot_df["true_reference"].dropna().iloc[0])
        ax.axvline(true_val, color="#111111", linestyle="--", linewidth=1.8, label=f"True Mean Elasticity ({true_val:.3f})")
        ax.legend(frameon=False, loc="lower left")

    ax.axvline(0.0, color="#666666", linewidth=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["method"])
    ax.set_xlabel("Estimated Price Elasticity")
    ax.set_title("Naive Regression vs. Causal Price Elasticity Estimates")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_elasticity_by_segment(
    segment_df: pd.DataFrame,
    segment_type: str,
    out_path: str | Path,
    title: str,
) -> Path:
    """Plots 3 & 4: Elasticity across regions or customer cohorts with ground truth overlay."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub = segment_df.loc[segment_df["segment_type"] == segment_type].sort_values("elasticity").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(sub))
    width = 0.36

    ax.bar(x - width / 2, sub["elasticity"], width=width, label="Estimated Elasticity", color="#3367d6")
    ax.errorbar(
        x - width / 2,
        sub["elasticity"],
        yerr=1.96 * sub["std_error"],
        fmt="none",
        ecolor="#111111",
        capsize=4,
    )
    if "true_elasticity" in sub.columns:
        ax.bar(x + width / 2, sub["true_elasticity"], width=width, label="Synthetic Ground Truth", color="#0b8043", alpha=0.85)

    ax.axhline(0.0, color="#555555", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["segment_value"])
    ax.set_ylabel("Price Elasticity")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
