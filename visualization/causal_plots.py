"""Visualizations for confounding analysis, causal DAG, Simpson's Paradox, and DiD event study."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def plot_confounding_analysis(
    conf_res: dict[str, Any],
    dag: nx.DiGraph,
    out_path: str | Path,
) -> Path:
    """Plot 5: Confounder correlation profile and Marketplace Causal DAG."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    screen = conf_res["confounder_screen"].head(9).iloc[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    y = np.arange(len(screen))
    axes[0].barh(y - 0.18, screen["corr_with_log_price"], height=0.34, label="Corr with log(Price)", color="#d95f02")
    axes[0].barh(y + 0.18, screen["corr_with_log_quantity"], height=0.34, label="Corr with log(Quantity)", color="#1b9e77")
    axes[0].axvline(0.0, color="#444444", linewidth=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(screen["variable"])
    axes[0].set_xlabel("Pearson Correlation")
    axes[0].set_title("Candidate Confounder & Instrument Correlations")
    axes[0].legend(frameon=False)

    # Causal DAG layout
    pos = nx.spring_layout(dag, seed=42, k=1.4)
    node_colors = []
    for n in dag.nodes():
        if n in ("Price", "Demand"):
            node_colors.append("#d62728")
        elif n == "SupplierCostShock_IV":
            node_colors.append("#2ca02c")
        elif n == "UnobservedDemand":
            node_colors.append("#ff7f0e")
        else:
            node_colors.append("#aec7e8")

    nx.draw_networkx(
        dag,
        pos=pos,
        ax=axes[1],
        node_color=node_colors,
        node_size=1150,
        font_size=7.5,
        font_weight="bold",
        arrows=True,
        arrowsize=12,
        edge_color="#666666",
    )
    axes[1].set_title("Marketplace Causal DAG (Red: P->Q, Green: IV, Orange: Unobserved)")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_simpsons_paradox(df: pd.DataFrame, simpson_res: dict[str, Any], out_path: str | Path) -> Path:
    """Plot 6: Pooled vs. within-region log(Price)-log(Quantity) slopes (Simpson's Paradox)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    group_col = simpson_res["group_col"]
    sample = df.sample(min(3500, len(df)), random_state=42)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    colors = {"North": "#1f77b4", "East": "#ff7f0e", "South": "#2ca02c", "West": "#d62728"}
    within_map = dict(
        zip(
            simpson_res["within_segment_elasticities"]["segment_value"],
            simpson_res["within_segment_elasticities"]["elasticity"],
        )
    )
    # Plot residualized log(price) relative to category/competitor baseline vs log(quantity)
    sample = sample.copy()
    sample["rel_log_price"] = sample["log_price"] - sample.groupby("category")["log_price"].transform("mean") + sample["log_price"].mean()
    for reg, grp in sample.groupby(group_col):
        c = colors.get(reg, "#555555")
        ax.scatter(grp["rel_log_price"], grp["log_quantity"], alpha=0.18, s=12, color=c)
        slope = within_map.get(str(reg), np.polyfit(grp["rel_log_price"], grp["log_quantity"], 1)[0])
        x_mean = float(grp["rel_log_price"].mean())
        y_mean = float(grp["log_quantity"].mean())
        xs = np.linspace(grp["rel_log_price"].min(), grp["rel_log_price"].max(), 50)
        ys = y_mean + slope * (xs - x_mean)
        ax.plot(xs, ys, color=c, linewidth=2.3, label=f"{reg} within-slope = {slope:.2f}")

    # Pooled slope on raw log(price)
    p_slope = float(simpson_res["pooled_elasticity"])
    x_all_mean = float(sample["log_price"].mean())
    y_all_mean = float(sample["log_quantity"].mean())
    xs_all = np.linspace(sample["log_price"].min(), sample["log_price"].max(), 80)
    ax.plot(
        xs_all,
        y_all_mean + p_slope * (xs_all - x_all_mean),
        color="#111111",
        linestyle="--",
        linewidth=2.8,
        label=f"Pooled Naive Slope = {p_slope:+.2f}",
    )

    ax.set_xlabel("log(Price)")
    ax.set_ylabel("log(Quantity)")
    ax.set_title("Simpson's Paradox: Aggregated vs. Within-Region Demand Slopes")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_did_event_study(event_study_df: pd.DataFrame, out_path: str | Path) -> Path:
    """Extra Plot 15: Difference-in-Differences Event Study (Pre-trends & Post-policy lift)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    es = event_study_df.sort_values("relative_period")
    ax.errorbar(
        es["relative_period"],
        es["estimate"],
        yerr=[es["estimate"] - es["ci_low"], es["ci_high"] - es["estimate"]],
        fmt="o-",
        color="#1f77b4",
        ecolor="#444444",
        capsize=4,
        linewidth=2,
    )
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1)
    ax.axvline(-0.5, color="#d62728", linestyle=":", linewidth=1.8, label="Policy Rollout Boundary")
    ax.set_xlabel("Relative Period (Months from Policy Start, -1 = Reference)")
    ax.set_ylabel("Estimated Treated-Control Log(Quantity) Diff")
    ax.set_title("DiD Event-Study: Pre-Treatment Parallel Trends & Post-Intervention Lift")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
