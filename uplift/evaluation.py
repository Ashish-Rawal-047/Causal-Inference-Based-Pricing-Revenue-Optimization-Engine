"""Uplift evaluation metrics: AUUC, Qini coefficient, uplift curves, and PEHE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class UpliftEvaluationResult:
    """Comprehensive evaluation summary for an uplift model."""

    model_name: str
    auuc: float
    auuc_normalized: float
    qini_coefficient: float
    top_decile_uplift: float
    bottom_decile_uplift: float
    estimated_ate: float
    pehe: float | None
    ate_abs_error: float | None
    cate_rank_corr: float | None
    curve_df: pd.DataFrame
    decile_df: pd.DataFrame
    estimator_note: str


def compute_uplift_curve(
    y: np.ndarray,
    treatment: np.ndarray,
    cate_pred: np.ndarray,
    propensity: np.ndarray | None = None,
    n_bins: int = 20,
) -> pd.DataFrame:
    """Compute cumulative uplift and Qini curve sorted descending by predicted CATE.

    For observational data where propensity e(x) = P(T=1|X=x) is provided, we compute
    an IPW-adjusted transformed outcome Z_i = Y_i * (T_i - e_i) / (e_i * (1 - e_i)) so
    that E[Z_i | X_i] = tau(X_i) under unconfoundedness.
    When propensity is None, standard randomized Qini/uplift curve:
        Qini(k) = sum_{i<=k, T=1} Y_i - (N_{T,k} / N_{C,k}) * sum_{i<=k, T=0} Y_i
    is computed directly.
    """
    y_arr = np.asarray(y, dtype=float)
    t_arr = np.asarray(treatment, dtype=int)
    pred = np.asarray(cate_pred, dtype=float)
    n = len(y_arr)
    if n < 10:
        raise ValueError("Need at least 10 observations to compute an uplift curve.")
    if (t_arr == 1).sum() == 0 or (t_arr == 0).sum() == 0:
        raise ValueError("Both treated and control samples are required for uplift evaluation.")

    order = np.argsort(-pred, kind="mergesort")
    y_s = y_arr[order]
    t_s = t_arr[order]

    if propensity is not None:
        e_s = np.clip(np.asarray(propensity, dtype=float)[order], 0.05, 0.95)
        z_s = y_s * (t_s - e_s) / (e_s * (1.0 - e_s))
    else:
        z_s = None

    cum_t = np.cumsum(t_s)
    cum_c = np.cumsum(1 - t_s)
    cum_y_t = np.cumsum(y_s * t_s)
    cum_y_c = np.cumsum(y_s * (1 - t_s))

    # Standard Qini gain: incremental total outcome relative to scaling control mean to treated count
    qini_raw = np.where(cum_c > 0, cum_y_t - cum_y_c * (cum_t / np.maximum(cum_c, 1)), 0.0)
    # Cumulative uplift curve (total incremental outcome across targeted fraction k):
    # U(k) = (mean_y_t(k) - mean_y_c(k)) * k when raw, or cumulative IPW transformed outcome sum_{i<=k} Z_i
    if z_s is not None:
        uplift_gain = np.cumsum(z_s)
    else:
        mean_y_t = np.where(cum_t > 0, cum_y_t / np.maximum(cum_t, 1), 0.0)
        mean_y_c = np.where(cum_c > 0, cum_y_c / np.maximum(cum_c, 1), 0.0)
        uplift_gain = (mean_y_t - mean_y_c) * np.arange(1, n + 1)

    total_gain = float(uplift_gain[-1])
    total_qini = float(qini_raw[-1])

    idx_grid = np.unique(np.linspace(max(1, n // n_bins), n - 1, n_bins, dtype=int))
    if idx_grid[-1] != n - 1:
        idx_grid = np.append(idx_grid, n - 1)

    rows = [
        {
            "fraction_targeted": 0.0,
            "n_targeted": 0,
            "cumulative_uplift": 0.0,
            "qini_gain": 0.0,
            "random_baseline_uplift": 0.0,
            "random_baseline_qini": 0.0,
        }
    ]
    for idx in idx_grid:
        frac = float(idx + 1) / float(n)
        rows.append(
            {
                "fraction_targeted": frac,
                "n_targeted": int(idx + 1),
                "cumulative_uplift": float(uplift_gain[idx]),
                "qini_gain": float(qini_raw[idx]),
                "random_baseline_uplift": frac * total_gain,
                "random_baseline_qini": frac * total_qini,
            }
        )
    return pd.DataFrame(rows)


def evaluate_uplift_model(
    model_name: str,
    y: np.ndarray,
    treatment: np.ndarray,
    cate_pred: np.ndarray,
    propensity: np.ndarray | None = None,
    true_cate: np.ndarray | None = None,
) -> UpliftEvaluationResult:
    """Evaluate an uplift model using AUUC, Qini coefficient, decile uplift, and PEHE (if ground truth available)."""
    curve = compute_uplift_curve(y=y, treatment=treatment, cate_pred=cate_pred, propensity=propensity, n_bins=25)
    fracs = curve["fraction_targeted"].to_numpy()
    uplift_vals = curve["cumulative_uplift"].to_numpy()
    qini_vals = curve["qini_gain"].to_numpy()
    rand_uplift = curve["random_baseline_uplift"].to_numpy()
    rand_qini = curve["random_baseline_qini"].to_numpy()

    auuc = float(np.trapz(uplift_vals, fracs))
    auuc_rand = float(np.trapz(rand_uplift, fracs))
    auuc_excess = auuc - auuc_rand
    denom = abs(float(uplift_vals[-1])) + 1e-9
    auuc_norm = float(auuc_excess / denom)

    qini_area = float(np.trapz(qini_vals - rand_qini, fracs))
    qini_norm = float(qini_area / (abs(float(qini_vals[-1])) + 1e-9))

    # Decile breakdown
    df_eval = pd.DataFrame(
        {
            "cate_pred": np.asarray(cate_pred, dtype=float),
            "y": np.asarray(y, dtype=float),
            "t": np.asarray(treatment, dtype=int),
        }
    )
    if propensity is not None:
        e = np.clip(np.asarray(propensity, dtype=float), 0.05, 0.95)
        df_eval["ipw_z"] = df_eval["y"] * (df_eval["t"] - e) / (e * (1.0 - e))
    if true_cate is not None:
        df_eval["true_cate"] = np.asarray(true_cate, dtype=float)

    df_eval["decile"] = pd.qcut(df_eval["cate_pred"].rank(method="first", ascending=False), q=10, labels=range(1, 11))
    dec_rows: list[dict[str, Any]] = []
    for dec, grp in df_eval.groupby("decile", observed=True):
        t1 = grp.loc[grp["t"] == 1, "y"]
        t0 = grp.loc[grp["t"] == 0, "y"]
        emp_uplift = float(t1.mean() - t0.mean()) if (len(t1) > 0 and len(t0) > 0) else np.nan
        row: dict[str, Any] = {
            "decile": int(dec),
            "n_obs": len(grp),
            "mean_pred_cate": float(grp["cate_pred"].mean()),
            "empirical_uplift": emp_uplift,
        }
        if "ipw_z" in grp.columns:
            row["ipw_uplift"] = float(grp["ipw_z"].mean())
        if "true_cate" in grp.columns:
            row["true_cate_mean"] = float(grp["true_cate"].mean())
        dec_rows.append(row)
    decile_df = pd.DataFrame(dec_rows).sort_values("decile").reset_index(drop=True)

    metric_col = "ipw_uplift" if "ipw_uplift" in decile_df.columns else "empirical_uplift"
    top_dec = float(decile_df.iloc[0][metric_col])
    bot_dec = float(decile_df.iloc[-1][metric_col])

    pehe = None
    ate_err = None
    rank_corr = None
    if true_cate is not None:
        tc = np.asarray(true_cate, dtype=float)
        pehe = float(np.sqrt(np.mean((cate_pred - tc) ** 2)))
        ate_err = float(abs(np.mean(cate_pred) - np.mean(tc)))
        rank_corr = float(pd.Series(cate_pred).corr(pd.Series(tc), method="spearman"))

    note = (
        "Evaluated with IPW-transformed outcome under estimated propensity (unconfoundedness assumed)."
        if propensity is not None
        else "Evaluated with standard subgroup Qini estimator (assumes conditional exchangeability within bins)."
    )

    return UpliftEvaluationResult(
        model_name=model_name,
        auuc=auuc,
        auuc_normalized=auuc_norm,
        qini_coefficient=qini_norm,
        top_decile_uplift=top_dec,
        bottom_decile_uplift=bot_dec,
        estimated_ate=float(np.mean(cate_pred)),
        pehe=pehe,
        ate_abs_error=ate_err,
        cate_rank_corr=rank_corr,
        curve_df=curve,
        decile_df=decile_df,
        estimator_note=note,
    )
