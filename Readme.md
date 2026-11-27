# Causal Pricing Intelligence & Optimization Engine

> An end-to-end causal inference, heterogeneous uplift modeling, and margin-constrained pricing optimization system for two-sided marketplaces.

---

## Overview

The **Causal Pricing Intelligence & Optimization Engine** answers a core marketplace decision question:

> **"If we change the price or discount for a particular customer segment, what will happen to demand, revenue, and margin?"**

Conventional pricing analysis often relies on correlation-based regression. In real marketplaces, observed prices are endogenous—influenced by demand shocks, geography, seasonality, promotions, seller availability, and customer cohorts. Consequently:

```text
Correlation != Causation
```

This repository implements the complete progression from exploratory analytics and naive regression to causal identification (2SLS Instrumental Variables and Difference-in-Differences), heterogeneous treatment effects (T-Learner and X-Learner), uplift ranking (AUUC / Qini / PEHE), constrained pricing optimization, chronological backtesting, and robustness stress testing.

---

## Business Problem

Suppose a marketplace increases a discount from **0% -> 15%**, or raises prices during peak demand. A naive regression can observe higher prices alongside higher quantities (or higher discounts alongside lower demand) because prices respond to unobserved demand shocks (`u_demand`) and regional income mix:

```text
Unobserved Demand / Region Mix
       │
       ├──────────────► Higher Prices
       │
       └──────────────► Higher Demand
```

Additionally, treating all customers uniformly destroys margin on price-inelastic cohorts (e.g., loyal or affluent buyers), even when total unit volume increases. The engine solves this by separating **causal demand sensitivity** from **confounding**, estimating **segment- and customer-level CATE**, and selecting feasible pricing actions that **maximize expected incremental margin** subject to business guardrails.

---

## System Architecture

```text
                       Transaction Data (Synthetic Ground-Truth or Olist)
                                              │
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │ Preprocessing & Leakage Validation           │
                       │ (cleaning.py, feature_engineering.py,        │
                       │  validation.py: chronological prior features)│
                       └──────────────────────┬───────────────────────┘
                                              │
                        ┌─────────────────────┴─────────────────────┐
                        ▼                                           ▼
         ┌─────────────────────────────┐             ┌─────────────────────────────┐
         │ Baseline & Confounding      │             │ Causal Identification       │
         │ • Linear & Log-Log OLS      │             │ • 2SLS IV (First/2nd Stage) │
         │ • Segment Elasticities      │             │ • DiD + Event Study         │
         │ • Simpson's Paradox & DAG   │             │ • Balance, Overlap & Sens.  │
         └──────────────┬──────────────┘             └──────────────┬──────────────┘
                        └─────────────────────┬─────────────────────┘
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │ Heterogeneous Treatment Effects & Uplift     │
                       │ • S-Learner Baseline, T-Learner, X-Learner   │
                       │ • AUUC, Qini Coefficient, PEHE, Segment CATE │
                       └──────────────────────┬───────────────────────┘
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │ Pricing Optimization & Business Guardrails   │
                       │ • Candidate Actions: [-10%, -5%, 0, +5, +10] │
                       │ • Objective: Max Expected Incremental Margin │
                       │ • Policies: Flat, Random, Segment, Uplift    │
                       └──────────────────────┬───────────────────────┘
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │ Chronological Backtesting & Robustness       │
                       │ • Out-of-time 75%/25% split (no leakage)     │
                       │ • Bootstrap CIs, Confounding & Shift Tests   │
                       └──────────────────────────────────────────────┘
```

---

## Repository Structure

```text
Causal-Inference-Based-Pricing-Revenue-Optimization-Engine/
├── config/
│   ├── settings.yaml              # Central hyperparameters, DGP config, constraints
│   └── loader.py                  # Config & logging utilities
├── simulation/
│   ├── ground_truth.py            # True structural elasticities & potential outcomes
│   ├── data_generator.py          # Configurable synthetic marketplace (10k–500k rows)
│   └── marketplace.py             # Oracle counterfactual evaluator & Olist adapter
├── preprocessing/
│   ├── validation.py              # Schema, domain, target & temporal leakage checks
│   ├── cleaning.py                # Missing values, duplicates, outlier winsorization
│   └── feature_engineering.py     # Calendar, competitor gaps, strict prior aggregates
├── analysis/
│   ├── eda.py                     # Exploratory summaries across regions/cohorts/categories
│   ├── elasticity.py              # Linear, naive log-log, controlled log-log, VIF/BP/JB
│   ├── segmentation.py            # Segment unit-economics profiling
│   └── confounding.py             # Confounder screen, Simpson's Paradox, NetworkX DAG
├── causal/
│   ├── identification.py          # CausalAssumptionError & testable/untestable registry
│   ├── iv.py                      # Two-Stage Least Squares (2SLS), F-stat, Wu-Hausman
│   ├── did.py                     # DiD interaction OLS, pre-trend test, event study
│   └── diagnostics.py             # Propensity overlap, IPW SMD balance, Cinelli-Hazlett RV
├── uplift/
│   ├── base.py                    # BaseUpliftEstimator, SLearner baseline, regressor factory
│   ├── t_learner.py               # T-Learner implementation
│   ├── x_learner.py               # X-Learner with imputed effects & propensity blending
│   ├── evaluation.py              # IPW-adjusted AUUC, Qini, decile uplift, PEHE
│   └── segmentation.py            # Segment-level CATE aggregation
├── optimization/
│   ├── revenue.py                 # Unit economics & policy financial aggregations
│   ├── constraints.py             # Margin, discount, price, availability & coverage caps
│   ├── pricing_policy.py          # CounterfactualOutcomeModel & 4 pricing policies
│   └── simulation.py              # Unified policy evaluation & oracle benchmarking
├── evaluation/
│   ├── metrics.py                 # RMSE, MAE, relative bias
│   ├── backtest.py                # Strict chronological out-of-time backtesting
│   ├── experiments.py             # Automated runner for Experiments 1–4 (CSV/JSON)
│   └── robustness.py              # Bootstrap CIs, confounding & shift stress tests
├── visualization/
│   ├── elasticity_plots.py        # Plots 1–4
│   ├── causal_plots.py            # Plots 5–6 & DiD event study
│   ├── uplift_plots.py            # Plots 7–9
│   └── pricing_plots.py           # Plots 10–14 & interactive Plotly HTML exports
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_elasticity.ipynb
│   ├── 03_causal_inference.ipynb
│   ├── 04_uplift_modeling.ipynb
│   └── 05_pricing_optimization.ipynb
├── tests/                         # Automated pytest suite (13 test suites incl. edge cases)
├── docs/
│   └── TECHNICAL_METHODOLOGY.md   # Mathematical & identification documentation
├── reports/
│   ├── experiment_report.md       # Auto-generated experiment report
│   ├── experiments/               # Machine-readable CSV & JSON experiment outputs
│   └── figures/                   # Publication-quality PNGs & Plotly HTMLs
├── app.py                         # Interactive Streamlit dashboard (live pipeline)
├── main.py                        # Unified CLI entry point
├── pyproject.toml
└── requirements.txt
```

---

## Methodology & Algorithms

For complete mathematical derivations and identification proofs, see [TECHNICAL_METHODOLOGY.md](docs/TECHNICAL_METHODOLOGY.md).

1. **Synthetic Marketplace (`simulation/`)**: Models heterogeneous elasticities across regions (`North`, `East`, `South`, `West`), cohorts (`New`, `Returning`, `Loyal`, `VIP`), categories, and order frequencies; seasonality, weekend/festival spikes, competitor pricing, seller availability, unobserved surge-pricing confounding (`u_demand`), a valid excluded cost instrument (`supplier_cost_shock`), a sharp Difference-in-Differences regional price rollout at day 180, and exact potential outcomes (`true_q0`, `true_q1`, `true_cate_qty`, `true_cate_margin`).
2. **Baseline Elasticity (`analysis/elasticity.py`)**: Estimates Linear OLS, Naive Log-Log OLS, Controlled Log-Log OLS, and within-segment elasticities with HC3 robust standard errors, VIF multicollinearity, Breusch-Pagan heteroskedasticity, and Jarque-Bera residual diagnostics.
3. **Causal Inference (`causal/`)**:
   - **2SLS IV (`causal/iv.py`)**: Implemented from first principles with Stage-1 partial $F$-statistic, weak-instrument detection ($F < 10$), Stage-2 asymptotic HC1 sandwich standard errors using structural residuals, Wu-Hausman endogeneity test, and covariate balance checks across instrument quantiles.
   - **DiD (`causal/did.py`)**: 2x2 interaction regression with HC3 SEs, implied Wald/DiD price elasticity, pre-treatment linear trend test, and dynamic event-study leads/lags.
   - **Causal Diagnostics (`causal/diagnostics.py`)**: Propensity common-support overlap, raw vs. IPW Standardized Mean Differences, SUTVA contention checks, and Cinelli-Hazlett omitted-variable Robustness Values.
4. **Uplift Modeling (`uplift/`)**: Implements `SLearner`, `TLearner`, and `XLearner` under a unified interface using `HistGradientBoostingRegressor` (plus `RandomForest`, `XGBoost`, and `Ridge` support), evaluated with IPW-adjusted AUUC, Qini coefficient, decile uplift, PEHE, and Spearman rank correlation.
5. **Pricing Optimization & Backtesting (`optimization/`, `evaluation/backtest.py`)**: Evaluates candidate actions `[-10%, -5%, 0%, +5%, +10%]` under `PricingConstraints` (`min_margin_pct >= 8%`, `max_discount <= 35%`, price bounds, seller availability, coverage limits) to maximize **Expected Incremental Margin**, comparing `Flat Policy`, `Random Policy`, `Segment-Based Policy`, and `Uplift-Based Policy` strictly out-of-time on the final 25% chronological test window.

---

## Installation & Execution

Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### Main Pipeline CLI

```bash
# Run complete end-to-end pipeline (50,000 synthetic rows, Experiments 1-4, robustness, figures, report)
python main.py --experiment all

# Run individual stages
python main.py --experiment data --n-rows 50000
python main.py --experiment elasticity
python main.py --experiment causal
python main.py --experiment uplift
python main.py --experiment optimization
python main.py --experiment backtest
python main.py --experiment robustness
```

### Standalone Module Commands

```bash
python analysis/eda.py
python analysis/elasticity.py
python causal/iv.py
python causal/did.py
python uplift/t_learner.py
python optimization/simulation.py
```

### Tests & Interactive Dashboard

```bash
pytest
streamlit run app.py
```

### Public Dataset Support (Olist)

To run exploratory, correlational elasticity, confounding, and predictive backtesting on the **Olist Brazilian E-Commerce Dataset**, place the Kaggle CSV files (`olist_orders_dataset.csv`, `olist_order_items_dataset.csv`, `olist_customers_dataset.csv`, `olist_products_dataset.csv`) under `data/raw/olist/` and run:

```bash
python main.py --experiment elasticity --dataset olist
```

> **Causal Scope Note:** Because Olist is purely observational without a documented exogenous pricing instrument, sharp policy rollout date, or randomized discount assignment, causal IV, DiD, and randomized uplift claims are not supported on raw Olist data alone.

---

## Actual Measured Experimental Results (50,000 Synthetic Transactions, Seed=42)

> **Disclaimer:** All metrics below were measured directly by executing `python main.py --experiment all --n-rows 50000 --seed 42` on the synthetic marketplace environment (`reports/experiments/`). They validate estimator and policy behavior against known synthetic counterfactuals and must not be interpreted as real-world business impact.

### Dataset Summary
* **Transactions:** 50,000 (`2024-01-01` to `2024-12-30`) | **Customers:** 4,000 | **Products:** 120
* **Observed Revenue:** \$17,386,171.04 | **Observed Margin:** \$6,793,384.25 (39.1%)

### Experiment 1 — Naive vs. Causal Elasticity

| Method | Estimated Elasticity | Std. Error | 95% CI | True Mean Reference | Bias |
| --- | ---: | ---: | --- | ---: | ---: |
| Linear OLS (Mean Elasticity) | +0.1717 | 0.0052 | [0.1615, 0.1820] | -0.9940 | +1.1657 |
| Naive Log-Log OLS | +0.1127 | 0.0042 | [0.1044, 0.1210] | -0.9940 | +1.1067 |
| Controlled Log-Log OLS | -0.5617 | 0.0061 | [-0.5736, -0.5498] | -0.9940 | +0.4323 |
| **Causal 2SLS IV** | **-0.9773** | **0.0072** | **[-0.9915, -0.9632]** | **-0.9940** | **+0.0166** |
| **Causal DiD (Treated Implied)** | **-1.3511** | — | — | **-1.3056** | **-0.0455** |

* **IV First-Stage $F$-statistic:** `152,064.66` (`weak_instrument = False`)
* **DiD Pre-Trend Differential Slope $p$-value:** `0.4315` (no statistically significant pre-treatment divergence)
* **Simpson's Paradox:** Pooled naive slope is **+0.1127**, whereas within-region controlled elasticities are strictly negative across every region: `North` (`-0.9319`), `East` (`-0.7500`), `South` (`-0.3853`), `West` (`-0.0766`).

### Experiment 2 — Segment Heterogeneity (ATE vs. CATE)

* **Overall Quantity ATE (15% promo intervention):** `+1.1257` units | **Overall Margin ATE:** `-$40.41` (SD `80.16`), showing that uniform promotional discounting destroys margin on average.
* Across cohorts, predicted quantity lift is higher for `New` (`+1.1931`) and `Returning` (`+1.2027`) customers than `VIP` (`+0.9212`), while `Electronics` suffers the largest margin erosion (`-$102.61`) under blanket discounting compared to `Beauty` (`-$19.25`) and `Grocery` (`-$19.93`).

### Experiment 3 — Uplift Model Benchmark (Held-Out Margin CATE)

| Model | Normalized Excess AUUC | Qini Coefficient | Top Decile Uplift | Bottom Decile Uplift | PEHE (RMSE vs. True CATE) | Spearman Rank Corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S-Learner (Baseline) | 0.2145 | 0.2467 | -$10.33 | -$95.60 | 37.77 | 0.7022 |
| T-Learner | 0.2426 | 0.2262 | -$5.98 | -$84.91 | 86.30 | 0.6492 |
| X-Learner | 0.1902 | 0.1784 | -$16.39 | -$56.43 | 52.96 | **0.7029** |

### Experiment 4 — Chronological Backtest & Pricing Policy Comparison (12,500 Held-Out Test Rows)

* **Observed Historical Test Period:** Revenue `\$3,768,079.40` | Margin `\$1,390,601.84`

| Policy | Total Revenue | Total Margin | Incremental Revenue | Incremental Margin | Incremental Orders | Discount Expenditure | Treatment Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Flat Policy (-5%) | \$3,933,313.49 | \$1,394,495.42 | +\$325,579.72 | +\$47,993.34 | +6,084.86 | \$172,549.84 | 68.91% |
| Random Policy | \$3,767,556.19 | \$1,423,924.81 | +\$159,822.42 | +\$77,422.72 | +1,590.46 | \$102,344.54 | 67.34% |
| Segment-Based Policy | \$3,754,464.05 | \$1,601,069.97 | +\$146,730.27 | +\$254,567.88 | -2,999.96 | \$0.00 | 89.16% |
| **Uplift-Based Policy** | **\$3,786,279.10** | **\$1,605,611.07** | **+\$178,545.33** | **+\$259,108.98** | **-1,948.57** | **\$13,435.85** | **89.16%** |

Key takeaway: **Flat -5% discounting** drives the highest unit volume (`+6,085` orders) and top-line revenue (`\$3.93M`), yet burns `\$172.5k` in discount expenditure and yields only `+$48.0k` incremental margin. **Uplift-Based Policy** targets discounts selectively (`\$13.4k` discount expenditure) while raising prices within guardrails on inelastic segments, achieving **`+$259.1k` incremental margin** (**5.4x** the incremental margin of Flat Policy, and outperforming Segment-Based Policy on both revenue and margin).

### Robustness Stress Test — Confounding Sensitivity (12,000 Rows per Level)

| Confounding Strength | True Elasticity | Naive OLS | Controlled OLS | 2SLS IV | Controlled OLS Abs Error | 2SLS IV Abs Error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | -1.0293 | +0.0655 | -1.0032 | -1.0227 | 0.0261 | 0.0066 |
| 0.5 | -1.0293 | +0.0735 | -0.8876 | -1.0202 | 0.1417 | 0.0091 |
| 1.0 | -1.0293 | +0.1024 | -0.6023 | -1.0183 | 0.4270 | 0.0110 |
| 1.8 | -1.0293 | +0.1792 | -0.0826 | -1.0125 | 0.9467 | 0.0168 |

---

## Generated Visualizations & Reports

All plots are saved under `reports/figures/` (plus interactive Plotly HTML files in `reports/figures/interactive/`):
1. `01_price_vs_demand.png` — Transaction price vs. demand & binned log-log curve
2. `02_naive_vs_causal_elasticity.png` — OLS vs. 2SLS IV vs. DiD comparison against truth
3. `03_elasticity_by_region.png` — Regional elasticity breakdown
4. `04_elasticity_by_cohort.png` — Customer cohort elasticity breakdown
5. `05_confounding_analysis.png` — Confounder correlations & Marketplace Causal DAG
6. `06_simpsons_paradox.png` — Pooled vs. within-region demand slopes
7. `07_treatment_effect_distribution.png` — Predicted quantity & margin CATE distributions
8. `08_uplift_curve.png` — Cumulative IPW-adjusted uplift curves
9. `09_auuc_comparison.png` — Normalized AUUC & PEHE comparison
10. `10_policy_revenue_comparison.png` — Total revenue by pricing policy
11. `11_policy_margin_comparison.png` — Total margin by pricing policy
12. `12_incremental_revenue_margin.png` — Incremental revenue vs. incremental margin lift
13. `13_backtesting_results.png` — Held-out weekly cumulative margin trajectories
14. `14_robustness_analysis.png` — Confounding sensitivity & sample-size convergence
15. `15_did_event_study.png` — Difference-in-Differences pre-trend & post-policy event study

---

## Causal Assumptions & Limitations

* **Unverified Observational Assumptions:** While IV relevance ($F$-stat) and DiD pre-treatment trend slopes can be tested, the **IV exclusion restriction**, **IV independence from unobserved confounders**, **post-treatment parallel trends**, and **conditional unconfoundedness** for uplift models cannot be proven from observational data alone.
* **Marketplace Equilibrium & SUTVA:** Real two-sided marketplaces exhibit cross-product cannibalization, seller inventory depletion, strategic customer waiting, and competitor price retaliation, which unit-level static counterfactuals do not capture.
* **Decision-Support Role:** Production rollout requires randomized online A/B pricing experiments with safety guardrails and post-deployment drift monitoring.

---

## Future Work

* Doubly robust (AIPW) & Causal Forest estimators for continuous multi-dose pricing treatments
* Dynamic inventory-constrained contextual bandits / reinforcement learning
* Cross-price cannibalization matrices across substitute SKUs
* Customer lifetime value (LTV)-aware multi-period pricing optimization

---

## Author

**Ashish Rawal** — Exploring **causal inference, heterogeneous uplift modeling, experimentation, and marketplace pricing optimization**.
