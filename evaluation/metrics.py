"""Shared evaluation metrics for elasticity, causal recovery, uplift, and pricing."""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(yt - yp)))


def relative_bias(estimate: float, ground_truth: float) -> float:
    """Relative estimation bias (estimate - truth) / |truth|."""
    denom = max(abs(ground_truth), 1e-9)
    return float((estimate - ground_truth) / denom)
