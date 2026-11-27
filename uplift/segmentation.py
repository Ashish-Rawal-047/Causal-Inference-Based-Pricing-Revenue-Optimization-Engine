"""Segment-level uplift aggregation and uncertainty quantification."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def summarize_segment_uplift(
    df: pd.DataFrame,
    cate_qty: np.ndarray,
    cate_margin: np.ndarray,
    segment_cols: Sequence[str] = ("region", "customer_cohort", "category", "order_frequency_group"),
) -> pd.DataFrame:
    """Aggregate predicted quantity and margin CATEs across business segments."""
    work = df.copy()
    work["_pred_cate_qty"] = np.asarray(cate_qty, dtype=float)
    work["_pred_cate_margin"] = np.asarray(cate_margin, dtype=float)

    rows: list[pd.DataFrame] = []
    for col in segment_cols:
        if col not in work.columns:
            continue
        agg_spec: dict[str, tuple[str, str]] = {
            "n_obs": ("order_id", "count"),
            "pred_cate_qty_mean": ("_pred_cate_qty", "mean"),
            "pred_cate_qty_std": ("_pred_cate_qty", "std"),
            "pred_cate_margin_mean": ("_pred_cate_margin", "mean"),
            "pred_cate_margin_std": ("_pred_cate_margin", "std"),
            "positive_margin_uplift_share": ("_pred_cate_margin", lambda s: float((s > 0).mean())),
        }
        if "true_cate_qty" in work.columns:
            agg_spec["true_cate_qty_mean"] = ("true_cate_qty", "mean")
        if "true_cate_margin" in work.columns:
            agg_spec["true_cate_margin_mean"] = ("true_cate_margin", "mean")

        grp = work.groupby(col, observed=True, as_index=False).agg(**agg_spec)
        grp.insert(0, "segment_type", col)
        grp = grp.rename(columns={col: "segment_value"})
        grp["segment_value"] = grp["segment_value"].astype(str)
        rows.append(grp)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["segment_type", "pred_cate_margin_mean"], ascending=[True, False]).reset_index(drop=True)
