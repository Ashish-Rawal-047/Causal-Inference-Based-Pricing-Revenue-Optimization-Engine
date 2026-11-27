"""Chronological backtesting framework preventing temporal and target leakage."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config.loader import get_logger
from optimization.constraints import PricingConstraints
from optimization.simulation import compare_pricing_policies
from preprocessing.validation import validate_temporal_split
from simulation.marketplace import evaluate_counterfactual_price_action

logger = get_logger(__name__)


def chronological_train_test_split(
    df: pd.DataFrame,
    train_fraction: float = 0.75,
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split transactions strictly by time boundary so training never sees future data."""
    if not 0.1 < train_fraction < 0.95:
        raise ValueError("train_fraction must lie strictly between 0.1 and 0.95.")

    ordered = df.sort_values(timestamp_col).reset_index(drop=True)
    split_ts = ordered[timestamp_col].quantile(train_fraction)
    train_df = ordered.loc[ordered[timestamp_col] < split_ts].reset_index(drop=True)
    test_df = ordered.loc[ordered[timestamp_col] >= split_ts].reset_index(drop=True)
    if train_df.empty or test_df.empty:
        idx = int(len(ordered) * train_fraction)
        train_df = ordered.iloc[:idx].reset_index(drop=True)
        test_df = ordered.iloc[idx:].reset_index(drop=True)

    validate_temporal_split(train_df, test_df, timestamp_col=timestamp_col)
    return train_df, test_df


def run_chronological_backtest(
    df: pd.DataFrame,
    train_fraction: float = 0.75,
    constraints: PricingConstraints = PricingConstraints(),
    seed: int = 42,
) -> dict[str, Any]:
    """Run an out-of-time backtest comparing all 4 pricing policies on the held-out period.

    Explicitly separates:
    1. Observed historical test-period outcomes (factual realized revenue/margin).
    2. Model-estimated counterfactual outcomes under each policy.
    3. Synthetic ground-truth oracle outcomes (when synthetic DGP metadata is present).
    """
    train_df, test_df = chronological_train_test_split(df, train_fraction=train_fraction)
    policy_res = compare_pricing_policies(
        train_df=train_df,
        eval_df=test_df,
        constraints=constraints,
        seed=seed,
    )

    observed_summary = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_start": str(pd.to_datetime(train_df["timestamp"]).min()),
        "train_end": str(pd.to_datetime(train_df["timestamp"]).max()),
        "test_start": str(pd.to_datetime(test_df["timestamp"]).min()),
        "test_end": str(pd.to_datetime(test_df["timestamp"]).max()),
        "observed_test_revenue": float(test_df["revenue"].sum()),
        "observed_test_margin": float(test_df["margin"].sum()),
        "observed_test_orders": float(test_df["quantity"].sum()),
    }

    # Build period-over-period test trajectory (weekly buckets) for backtest visualization
    test_ts = pd.to_datetime(test_df["timestamp"])
    week_labels = test_ts.dt.to_period("W").astype(str)
    has_oracle = {"true_elasticity", "exog_demand_index", "base_price", "unit_cost"}.issubset(test_df.columns)

    traj_frames: list[pd.DataFrame] = []
    for policy_name, dec in policy_res["decisions"].items():
        if has_oracle:
            cf = evaluate_counterfactual_price_action(test_df, dec.chosen_price_change_pct)
            margin_series = cf["cf_margin"].to_numpy(dtype=float)
            rev_series = cf["cf_revenue"].to_numpy(dtype=float)
            eval_source = "synthetic_ground_truth"
        else:
            margin_series = dec.expected_margin
            rev_series = dec.expected_revenue
            eval_source = "model_estimated"

        tmp = pd.DataFrame(
            {
                "week": week_labels,
                "policy": policy_name,
                "revenue": rev_series,
                "margin": margin_series,
                "evaluation_source": eval_source,
            }
        )
        agg = tmp.groupby(["week", "policy", "evaluation_source"], as_index=False).sum()
        traj_frames.append(agg)

    trajectory_df = pd.concat(traj_frames, ignore_index=True).sort_values(["week", "policy"]).reset_index(drop=True)

    return {
        "observed_summary": observed_summary,
        "model_comparison": policy_res["model_comparison"],
        "oracle_comparison": policy_res["oracle_comparison"],
        "trajectory_df": trajectory_df,
        "decisions": policy_res["decisions"],
        "test_df": test_df,
        "train_df": train_df,
    }
