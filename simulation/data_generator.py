"""Synthetic two-sided marketplace transaction data generator with known causal ground truth."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.loader import get_logger, load_config, resolve_path
from simulation.ground_truth import (
    CATEGORIES,
    COHORTS,
    DEFAULT_GROUND_TRUTH,
    REGIONS,
    GroundTruthSpec,
    compute_total_structural_log_q,
    compute_true_elasticity,
)

logger = get_logger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def generate_synthetic_marketplace_data(
    n_rows: int = 50_000,
    seed: int = 42,
    confounding_strength: float = 1.0,
    instrument_strength: float = 0.32,
    did_start_day: int = 180,
    did_price_drop: float = 0.12,
    promo_discount_depth: float = 0.15,
    treatment_assignment: str = "observational",
    demand_shock_scale: float = 0.35,
    elasticity_shift: float = 0.0,
    n_customers: int = 4_000,
    n_products: int = 120,
    start_date: str = "2024-01-01",
    n_days: int = 365,
    spec: GroundTruthSpec = DEFAULT_GROUND_TRUTH,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """Generate a reproducible synthetic marketplace transaction dataset.

    Causal features of the DGP:
    1. Confounding: Unobserved demand intent (`u_demand`) and observed segment attributes
       shift BOTH price (surge/dynamic pricing) and quantity, creating positive omitted-variable
       bias in naive OLS elasticity and Simpson's Paradox across regions.
    2. Instrumental Variable (`supplier_cost_shock`): Exogenous logistics/supplier cost shock
       shifts list price (`alpha_Z = instrument_strength`) with zero direct path to consumer
       demand and zero correlation with `u_demand`.
    3. Difference-in-Differences: At `did_start_day`, `did_group == 1` (North, East) receives
       an exogenous regional fee/price reduction (`did_price_drop`), while `did_group == 0`
       (South, West) follows the same pre-treatment calendar trend.
    4. Heterogeneous Uplift (`treatment_flag`): Targeted promotional discount (`promo_discount_depth`)
       with ground-truth potential outcomes `true_q0`, `true_q1`, `true_cate_qty`,
       `true_cate_revenue`, and `true_cate_margin`.
    """
    if n_rows <= 0:
        raise ValueError("n_rows must be positive.")

    rng = np.random.default_rng(seed)

    # 1. Customer & Product latent attributes
    cust_ids = np.array([f"CUST_{i:05d}" for i in range(n_customers)])
    cust_regions = rng.choice(REGIONS, size=n_customers, p=[0.28, 0.25, 0.24, 0.23])
    cust_cohorts = rng.choice(COHORTS, size=n_customers, p=[0.30, 0.32, 0.24, 0.14])
    cust_base_freq = np.where(
        cust_cohorts == "New",
        rng.poisson(1.2, size=n_customers),
        np.where(
            cust_cohorts == "Returning",
            rng.poisson(4.0, size=n_customers),
            np.where(cust_cohorts == "Loyal", rng.poisson(9.0, size=n_customers), rng.poisson(16.0, size=n_customers)),
        ),
    )

    prod_ids = np.array([f"PROD_{j:04d}" for j in range(n_products)])
    prod_categories = rng.choice(CATEGORIES, size=n_products, p=[0.18, 0.24, 0.20, 0.22, 0.16])
    prod_base_prices = np.array([spec.category_base_price[c] for c in prod_categories]) * rng.lognormal(
        mean=0.0, sigma=0.22, size=n_products
    )
    prod_popularity = rng.normal(loc=0.0, scale=0.25, size=n_products)

    # 2. Sample transactions chronologically
    day_offsets = rng.integers(0, n_days, size=n_rows)
    hour_offsets = rng.integers(0, 24, size=n_rows)
    minute_offsets = rng.integers(0, 60, size=n_rows)
    base_ts = pd.Timestamp(start_date)
    timestamps = (
        base_ts
        + pd.to_timedelta(day_offsets, unit="D")
        + pd.to_timedelta(hour_offsets, unit="h")
        + pd.to_timedelta(minute_offsets, unit="m")
    )
    order_sort_idx = np.argsort(timestamps.to_numpy())
    timestamps = timestamps[order_sort_idx]
    day_offsets = day_offsets[order_sort_idx]

    c_idx = rng.integers(0, n_customers, size=n_rows)
    p_idx = rng.integers(0, n_products, size=n_rows)

    customer_id = cust_ids[c_idx]
    region = cust_regions[c_idx]
    customer_cohort = cust_cohorts[c_idx]
    product_id = prod_ids[p_idx]
    category = prod_categories[p_idx]

    # Chronological historical orders per customer prior to current transaction
    df_tmp = pd.DataFrame({"c_idx": c_idx, "base_freq": cust_base_freq[c_idx]})
    prior_in_window = df_tmp.groupby("c_idx").cumcount().to_numpy()
    historical_orders = cust_base_freq[c_idx] + prior_in_window

    # 3. Calendar, seasonality, festivals, seller availability
    day_of_week = timestamps.dayofweek.to_numpy()
    month = timestamps.month.to_numpy()
    is_weekend = (day_of_week >= 5).astype(float)
    # Smooth annual seasonality + festival spikes (e.g., Nov/Dec + mid-year sale)
    seasonality = 0.18 * np.sin(2.0 * np.pi * (day_offsets / 365.0) - np.pi / 4.0)
    festival_flag = np.isin(day_offsets, [45, 46, 110, 111, 200, 201, 320, 321, 322, 355, 356, 357]).astype(float)

    seller_availability = np.clip(
        0.78 - 0.08 * festival_flag + 0.05 * (region == "West") + rng.normal(0.0, 0.12, size=n_rows),
        0.15,
        1.0,
    )

    # 4. Unobserved demand shock u_demand (drives confounding!)
    u_demand = rng.normal(0.0, demand_shock_scale, size=n_rows)

    # 5. Valid Instrumental Variable Z: supplier / regional freight cost shock
    supplier_cost_shock = rng.normal(0.0, 1.0, size=n_rows)

    # 6. Difference-in-Differences indicator
    did_group = np.isin(region, ["North", "East"]).astype(int)
    post_period = (day_offsets >= did_start_day).astype(int)
    did_active = (did_group * post_period).astype(float)

    # 7. Reference & Competitor prices
    region_p_mult = pd.Series(region).map(spec.region_base_price_mult).to_numpy(dtype=float)
    reference_price = prod_base_prices[p_idx] * region_p_mult
    competitor_price = reference_price * np.exp(
        0.06 * seasonality + rng.normal(0.0, 0.07, size=n_rows)
    )

    # 8. Binary promotional treatment assignment (with realistic confounding)
    if treatment_assignment == "randomized":
        true_propensity = np.full(n_rows, 0.45)
    elif treatment_assignment == "skewed":
        logit_p = (
            -0.5
            + 0.75 * (region == "North")
            + 0.55 * (region == "East")
            - 0.85 * (region == "West")
            + 0.45 * (customer_cohort == "New")
            - 0.65 * (customer_cohort == "VIP")
            + 0.35 * is_weekend
        )
        true_propensity = np.clip(_sigmoid(logit_p), 0.08, 0.92)
    else:  # observational
        logit_p = (
            -0.35
            + 0.45 * (region == "North")
            + 0.30 * (region == "East")
            - 0.45 * (region == "West")
            + 0.35 * (customer_cohort == "New")
            - 0.35 * (customer_cohort == "VIP")
            + 0.25 * is_weekend
            + 0.40 * festival_flag
            - 0.30 * confounding_strength * u_demand  # marketer discounts more when demand is weak!
            + 0.15 * np.log1p(historical_orders) * (-0.3)
        )
        true_propensity = np.clip(_sigmoid(logit_p), 0.10, 0.90)

    treatment_flag = rng.binomial(1, true_propensity).astype(int)
    promotion_flag = ((treatment_flag == 1) | (festival_flag == 1)).astype(int)

    # Discount depth: base noise when untreated + targeted promo_discount_depth when treated + DiD drop
    base_discount = np.where(festival_flag == 1, 0.05, 0.0)
    discount = np.clip(
        base_discount + treatment_flag * promo_discount_depth + did_active * did_price_drop,
        0.0,
        0.40,
    )

    # Endogenous list price: driven by reference_price, IV supplier_cost_shock,
    # unobserved demand shock u_demand (dynamic surge pricing!), seasonality, and seller scarcity
    log_list_price = (
        np.log(reference_price)
        + instrument_strength * supplier_cost_shock
        + confounding_strength * 0.45 * u_demand
        + confounding_strength * 0.12 * seasonality
        + confounding_strength * 0.15 * festival_flag
        - 0.12 * (seller_availability - 0.78)
        + rng.normal(0.0, 0.05, size=n_rows)
    )
    list_price = np.exp(log_list_price)
    price = np.maximum(list_price * (1.0 - discount), 2.0)

    # Unit cost scales with category cost ratio and supplier cost shock
    cost_ratio = pd.Series(category).map(spec.category_cost_ratio).to_numpy(dtype=float)
    unit_cost = reference_price * cost_ratio * np.exp(0.10 * supplier_cost_shock)

    # 9. True structural elasticity & exogenous demand index
    true_elasticity = compute_true_elasticity(
        region=region,
        cohort=customer_cohort,
        category=category,
        historical_orders=historical_orders,
        spec=spec,
    )
    if elasticity_shift != 0.0:
        true_elasticity = np.clip(true_elasticity + elasticity_shift, -2.8, -0.10)

    region_dem = np.log(pd.Series(region).map(spec.region_base_demand).to_numpy(dtype=float))
    cohort_dem = np.where(
        customer_cohort == "VIP",
        0.38,
        np.where(customer_cohort == "Loyal", 0.22, np.where(customer_cohort == "Returning", 0.08, -0.10)),
    )
    exog_demand_index = (
        0.85
        + region_dem
        + cohort_dem
        + prod_popularity[p_idx]
        + seasonality
        + spec.weekend_demand_lift * is_weekend
        + spec.festival_demand_lift * festival_flag
        + spec.availability_demand_slope * (seller_availability - 0.78)
        + 0.05 * np.log1p(historical_orders)
        + confounding_strength * u_demand
    )

    # Observed quantity with log-normal noise + strictly positive continuous/integer representation
    log_q_expected = compute_total_structural_log_q(
        exogenous_demand_index=exog_demand_index,
        price=price,
        reference_price=reference_price,
        competitor_price=competitor_price,
        total_elasticity=true_elasticity,
        promotion_flag=promotion_flag,
        spec=spec,
    )
    # Also add small common calendar trend (identical across DiD groups) to test DiD trend removal
    common_time_trend = 0.08 * (day_offsets / n_days)
    eps_q = rng.normal(0.0, 0.14, size=n_rows)
    quantity = np.maximum(np.round(np.exp(log_q_expected + common_time_trend + eps_q), 2), 0.5)

    # 10. Ground-truth counterfactuals for treatment_flag = 0 vs treatment_flag = 1
    disc_0 = np.clip(base_discount + did_active * did_price_drop, 0.0, 0.40)
    disc_1 = np.clip(base_discount + promo_discount_depth + did_active * did_price_drop, 0.0, 0.40)
    price_0 = np.maximum(list_price * (1.0 - disc_0), 2.0)
    price_1 = np.maximum(list_price * (1.0 - disc_1), 2.0)
    promo_0 = (festival_flag == 1).astype(int)
    promo_1 = np.ones(n_rows, dtype=int)

    log_q0 = (
        compute_total_structural_log_q(
            exog_demand_index, price_0, reference_price, competitor_price, true_elasticity, promo_0, spec
        )
        + common_time_trend
    )
    log_q1 = (
        compute_total_structural_log_q(
            exog_demand_index, price_1, reference_price, competitor_price, true_elasticity, promo_1, spec
        )
        + common_time_trend
    )
    # Expected potential outcomes E[Q(t) | X, u] including lognormal mean correction exp(0.5 * sigma^2)
    ln_corr = float(np.exp(0.5 * (0.14**2)))
    true_q0 = np.exp(log_q0) * ln_corr
    true_q1 = np.exp(log_q1) * ln_corr
    true_cate_qty = true_q1 - true_q0

    true_rev0 = price_0 * true_q0
    true_rev1 = price_1 * true_q1
    true_cate_revenue = true_rev1 - true_rev0

    true_margin0 = (price_0 - unit_cost) * true_q0
    true_margin1 = (price_1 - unit_cost) * true_q1
    true_cate_margin = true_margin1 - true_margin0

    revenue = np.round(price * quantity, 2)
    cost = np.round(unit_cost * quantity, 2)
    margin = np.round(revenue - cost, 2)

    df = pd.DataFrame(
        {
            "order_id": [f"ORD_{i:07d}" for i in range(1, n_rows + 1)],
            "customer_id": customer_id,
            "product_id": product_id,
            "region": region,
            "timestamp": timestamps,
            "price": np.round(price, 2),
            "discount": np.round(discount, 4),
            "quantity": quantity,
            "category": category,
            "promotion_flag": promotion_flag,
            "customer_cohort": customer_cohort,
            "historical_orders": historical_orders,
            "competitor_price": np.round(competitor_price, 2),
            "revenue": revenue,
            "cost": cost,
            "margin": margin,
            # Auxiliary structural / experimental columns
            "base_price": np.round(reference_price, 2),
            "list_price": np.round(list_price, 2),
            "unit_cost": np.round(unit_cost, 2),
            "seller_availability": np.round(seller_availability, 4),
            "supplier_cost_shock": np.round(supplier_cost_shock, 5),
            "did_group": did_group,
            "post_period": post_period,
            "treatment_flag": treatment_flag,
            "festival_flag": festival_flag.astype(int),
            # Ground-truth evaluation columns (NEVER exposed as model features)
            "true_elasticity": np.round(true_elasticity, 5),
            "true_propensity": np.round(true_propensity, 5),
            "true_q0": np.round(true_q0, 4),
            "true_q1": np.round(true_q1, 4),
            "true_cate_qty": np.round(true_cate_qty, 4),
            "true_cate_revenue": np.round(true_cate_revenue, 4),
            "true_cate_margin": np.round(true_cate_margin, 4),
            "exog_demand_index": np.round(exog_demand_index + common_time_trend, 5),
        }
    )

    if save_path is not None:
        out = resolve_path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info("Saved synthetic dataset (%d rows) to %s", len(df), out)

    return df


def generate_all_configured_sizes(cfg: dict[str, Any] | None = None) -> dict[int, Path]:
    """Generate and persist synthetic CSVs for configured sizes or default size."""
    cfg = cfg or load_config()
    sim = cfg["simulation"]
    seed = int(cfg["project"]["seed"])
    out_dir = resolve_path(cfg["paths"]["synthetic_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[int, Path] = {}
    default_n = int(cfg["project"]["default_n_rows"])
    out_file = out_dir / f"marketplace_{default_n}.csv"
    generate_synthetic_marketplace_data(
        n_rows=default_n,
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
        save_path=out_file,
    )
    paths[default_n] = out_file
    return paths
