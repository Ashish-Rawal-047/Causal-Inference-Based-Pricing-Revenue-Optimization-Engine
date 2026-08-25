# Causal Pricing Intelligence & Optimization Engine

> A causal inference and uplift modeling framework for estimating price elasticity, identifying heterogeneous treatment effects, and optimizing pricing strategies for two-sided marketplaces.

## Overview

The **Causal Pricing Intelligence & Optimization Engine** is an end-to-end analytics and causal machine learning system designed to answer a fundamental marketplace question:

> **"If we change the price or discount for a particular customer segment, what will happen to demand, revenue, and margin?"**

Traditional pricing analysis often relies on correlation-based regression. However, observed prices are influenced by many factors including demand, geography, seasonality, promotions, customer behavior, and competitive conditions.

As a result:

```text
Correlation ≠ Causation
```

This project progressively moves from **naive elasticity estimation** to **causal inference** and finally to **heterogeneous treatment-effect modeling**, allowing pricing decisions to be evaluated based on estimated incremental impact rather than simple correlations.

---

## Key Objectives

The system is designed to:

* Estimate price elasticity of demand
* Correct for confounding in observed pricing data
* Estimate causal effects of price/discount changes
* Identify customer and geographic segments with different price sensitivity
* Estimate heterogeneous treatment effects
* Rank customers/segments by expected incremental revenue
* Compare targeted pricing against flat/random pricing strategies
* Backtest pricing strategies on a held-out time period
* Quantify uncertainty and potential business impact

---

## Business Problem

Consider a marketplace that increases a discount from **10% → 15%**.

A naive analysis might observe:

```text
Discount ↑
     ↓
Orders ↑
```

and conclude:

> "Increasing discounts increases demand."

But this may be incorrect.

The company may have increased discounts precisely when demand was expected to fall.

Other confounders may include:

* Weekend vs weekday
* Festival/holiday periods
* Region
* Customer cohort
* Product category
* Competitor pricing
* Seasonality
* Weather
* Seller availability
* Existing customer intent

Therefore, the observed relationship can be represented as:

```text
Price / Discount
       ↓
    Demand
       ↑
   Confounders
```

The goal of this project is to estimate:

```text
                 ┌─────────────────────┐
                 │   Pricing Decision   │
                 └──────────┬──────────┘
                            │
                            ▼
                  ┌──────────────────┐
                  │ Causal Estimator │
                  └────────┬─────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Treatment Effect     │
                │ / Elasticity         │
                └──────────┬───────────┘
                           │
                           ▼
                 ┌─────────────────────┐
                 │ Segment Optimization│
                 └──────────┬──────────┘
                            │
                            ▼
                  Revenue / Margin Lift
```

---

# System Architecture

```text
                         Transaction Data
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Cleaning & EDA │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    │                     │
                    │ • Price             │
                    │ • Discount          │
                    │ • Region            │
                    │ • Time              │
                    │ • Customer Cohort   │
                    │ • Product           │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
        ┌─────────────────┐         ┌──────────────────┐
        │ Naive Elasticity│         │ Causal Analysis  │
        │ Regression      │         │                  │
        └────────┬────────┘         │ IV / DiD         │
                 │                  └────────┬─────────┘
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Heterogeneous      │
                    │ Treatment Effects   │
                    │                     │
                    │ T-Learner /         │
                    │ X-Learner           │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Segment-Level       │
                    │ Uplift Estimation   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Pricing Optimization│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Backtesting &       │
                    │ Revenue Simulation  │
                    └─────────────────────┘
```

---

# Methodology

## 1. Baseline: Naive Price Elasticity

The first model estimates demand elasticity using regression.

A simple log-log specification is:

```text
log(Q) = β₀ + β₁ log(P) + γX + ε
```

where:

* `Q` = quantity/orders
* `P` = price
* `β₁` = estimated price elasticity
* `X` = control variables
* `ε` = error term

Interpretation:

```text
β₁ < 0  → demand decreases as price increases
β₁ ≈ -1 → unit elastic demand
|β₁| > 1 → elastic demand
|β₁| < 1 → inelastic demand
```

This provides a useful baseline but does not necessarily identify the causal effect of price.

---

# 2. Confounding Analysis

The system explicitly models potential confounders such as:

* Region
* Product category
* Customer cohort
* Time of day
* Day of week
* Month
* Promotional periods
* Historical demand
* Seller availability
* Order frequency

Example:

```text
High demand
     │
     ├──────────────► Higher prices
     │
     └──────────────► More orders
```

If demand itself influences price, ordinary regression can produce biased elasticity estimates.

---

# 3. Causal Inference

The next stage estimates the causal effect of pricing interventions.

Depending on the dataset, the project supports approaches such as:

### Instrumental Variables

An instrument `Z` is used when price is endogenous.

```text
Instrument Z
     │
     ▼
   Price
     │
     ▼
   Demand
```

A valid instrument should:

1. Be correlated with treatment/price
2. Affect the outcome primarily through treatment
3. Not be directly related to unobserved outcome determinants

The project evaluates these assumptions rather than treating IV as a black-box estimator.

---

## Difference-in-Differences

When a pricing or promotional intervention affects one group but not another:

```text
             Before        After

Treatment      Y₁            Y₂
Control        Y₃            Y₄
```

The estimated treatment effect is:

```text
(Y₂ - Y₁) - (Y₄ - Y₃)
```

This helps remove common time trends affecting both groups.

---

# 4. Heterogeneous Treatment Effects

A single average treatment effect is often insufficient for marketplace pricing.

For example:

```text
Segment A → Highly price sensitive
Segment B → Moderately price sensitive
Segment C → Almost price insensitive
```

Applying the same discount to all customers can therefore destroy margin unnecessarily.

The project estimates:

```text
τ(x) = E[Y(1) - Y(0) | X = x]
```

where `τ(x)` represents the treatment effect for a specific customer/segment profile.

---

# 5. Uplift Modeling

The project evaluates heterogeneous treatment-effect models such as:

* T-Learner
* X-Learner

The goal is to estimate the incremental effect of a pricing intervention for each segment.

Example:

| Segment                | Expected Demand Lift | Expected Margin Impact |
| ---------------------- | -------------------: | ---------------------: |
| New customers          |                 High |                   High |
| Frequent customers     |               Medium |                 Medium |
| Highly loyal customers |                  Low |               Negative |
| Price-sensitive region |                 High |                   High |

This allows the pricing engine to prioritize interventions where they are most valuable.

---

# 6. Pricing Optimization

Once treatment effects are estimated, the system simulates alternative pricing policies.

For each segment:

```text
Current Price
      │
      ├── Price - 10%
      ├── Price - 5%
      ├── Current Price
      ├── Price + 5%
      └── Price + 10%
```

For each scenario, estimate:

```text
Expected Orders
Expected Revenue
Expected Cost
Expected Margin
Incremental Revenue
Incremental Margin
```

The pricing policy can then be selected based on an objective such as:

```text
Maximize Expected Incremental Margin
```

subject to business constraints.

---

# Evaluation

## Primary Metrics

### Price Elasticity

Measure:

```text
Elasticity = %ΔDemand / %ΔPrice
```

Estimate elasticity overall and by segment.

---

### Incremental Revenue

Compare:

```text
Optimized Pricing
       vs.
Flat Pricing
```

Calculate:

```text
Incremental Revenue
=
Revenue(Optimized)
-
Revenue(Baseline)
```

---

### Incremental Margin

Revenue alone is insufficient when discounts are involved.

Therefore:

```text
Incremental Margin
=
Margin(Optimized)
-
Margin(Baseline)
```

is also evaluated.

---

### AUUC

The **Area Under the Uplift Curve (AUUC)** measures how effectively the uplift model ranks customers/segments according to their incremental treatment effect.

Higher AUUC indicates better targeting of pricing interventions.

---

# Backtesting

To avoid evaluating the model on the same period used for training:

```text
Historical Data
      │
      ├───────────────┐
      │               │
      ▼               ▼
 Training Period   Test Period
      │               │
      ▼               │
 Causal/Uplift       │
 Model               │
      │               │
      └───────┬───────┘
              ▼
        Backtest Policy
```

The final pricing policy is evaluated on a held-out time period.

---

# Pricing Strategy Comparison

The system compares multiple policies:

### Random Policy

Randomly assigns pricing interventions.

### Flat Policy

Applies the same pricing strategy across all customers.

### Segment-Based Policy

Uses estimated elasticity by segment.

### Uplift-Based Policy

Targets segments based on estimated heterogeneous treatment effects.

The primary objective is to determine whether causal targeting produces higher incremental revenue or margin than simpler pricing strategies.

---

# Experimental Design

## Experiment 1 — Naive vs Causal Elasticity

Compare:

```text
Naive Regression
        vs.
IV / DiD
```

Measure the difference in estimated elasticity.

---

## Experiment 2 — Segment Heterogeneity

Compare:

```text
Average Treatment Effect
        vs.
Conditional Treatment Effects
```

Determine whether pricing sensitivity varies significantly across:

* Regions
* Customer cohorts
* Product categories
* Order frequency groups

---

## Experiment 3 — Uplift Models

Compare:

```text
T-Learner
X-Learner
Baseline
```

using uplift metrics such as AUUC.

---

## Experiment 4 — Pricing Policies

Compare:

```text
Flat Pricing
Random Targeting
Elasticity-Based Pricing
Uplift-Based Pricing
```

using:

* Revenue
* Margin
* Incremental orders
* Treatment cost
* Customer coverage

---

# Dataset

The project can be evaluated using publicly available marketplace datasets such as:

* **Olist Brazilian E-Commerce Dataset**
* **Instacart Market Basket Analysis Dataset**
* Other public transaction-level datasets

Where real treatment variation is insufficient, a **synthetic marketplace environment** can be generated with controlled:

* Price elasticity
* Seasonality
* Regional effects
* Customer heterogeneity
* Promotional effects
* Demand shocks
* Confounding relationships

Synthetic simulation makes it possible to evaluate causal estimators against known ground-truth treatment effects.

---

# Example Dataset Schema

```text
order_id
customer_id
product_id
region
timestamp
price
discount
quantity
category
promotion_flag
customer_cohort
historical_orders
competitor_price
revenue
cost
margin
```

---

# Example Analysis

A hypothetical result might look like:

```text
Naive Elasticity:       -0.42
Causal Elasticity:      -0.71

Estimated bias:
Naive estimate understates price sensitivity
```

Segment-level analysis:

```text
Region A: -0.92
Region B: -0.63
Region C: -0.31
```

This suggests that applying the same pricing strategy across all regions may be suboptimal.

---

# Error & Validity Analysis

Causal inference requires more than simply fitting a model.

The project investigates:

### Confounding

Are important variables omitted from the model?

### Positivity

Does every relevant segment have sufficient treatment variation?

### SUTVA

Can one customer's treatment affect another customer's outcome?

### Instrument Validity

For IV estimation:

```text
Relevance
Exclusion Restriction
Independence
```

must be considered.

### Parallel Trends

For DiD:

```text
Treatment
Control
```

groups should exhibit sufficiently similar pre-treatment trends.

---

# Simpson's Paradox

The project also demonstrates how aggregated elasticity can differ substantially from segment-level elasticity.

Example:

```text
Overall:
Price ↑ → Demand appears stable

Within Region A:
Price ↑ → Demand ↓

Within Region B:
Price ↑ → Demand ↓
```

This highlights why marketplace pricing models should explicitly account for segmentation and confounding.

---

# Technology Stack

### Programming

* Python
* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* XGBoost
* Gradient Boosting

### Causal Inference

* EconML
* DoWhy
* Statsmodels

### Visualization

* Matplotlib
* Plotly

### Experimentation

* Jupyter
* Custom backtesting framework

---

# Project Structure

```text
Causal-Pricing-Intelligence/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
│
├── preprocessing/
│   ├── cleaning.py
│   └── feature_engineering.py
│
├── analysis/
│   ├── eda.py
│   ├── elasticity.py
│   └── segmentation.py
│
├── causal/
│   ├── iv.py
│   ├── did.py
│   └── diagnostics.py
│
├── uplift/
│   ├── t_learner.py
│   ├── x_learner.py
│   └── evaluation.py
│
├── optimization/
│   ├── pricing_policy.py
│   ├── revenue.py
│   └── simulation.py
│
├── evaluation/
│   ├── metrics.py
│   ├── backtest.py
│   └── experiments.py
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_elasticity.ipynb
│   ├── 03_causal_inference.ipynb
│   └── 04_uplift_modeling.ipynb
│
├── tests/
│
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/<repository-name>.git

cd <repository-name>
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate on Windows:

```bash
.venv\Scripts\activate
```

Activate on Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Analysis

Run the exploratory analysis:

```bash
python analysis/eda.py
```

Estimate baseline elasticity:

```bash
python analysis/elasticity.py
```

Run causal estimation:

```bash
python causal/iv.py
```

or:

```bash
python causal/did.py
```

Train uplift models:

```bash
python uplift/t_learner.py
```

Run pricing backtests:

```bash
python optimization/simulation.py
```

---

# Key Research Questions

The project investigates:

1. How different is causal price elasticity from naive regression elasticity?
2. How much does confounding bias pricing decisions?
3. Does price sensitivity vary significantly across customer segments?
4. Which causal estimator performs best under different data-generating conditions?
5. Can uplift models identify customers who benefit most from discounts?
6. Does targeted pricing outperform flat pricing?
7. What is the trade-off between incremental revenue and margin?
8. How robust are pricing recommendations under distribution shifts?
9. How can pricing experiments be deployed safely in a real marketplace?

---

# Limitations

Real-world pricing systems have additional challenges that may not be completely captured by public datasets:

* Competitive pricing
* Supply constraints
* Strategic customer behavior
* Cannibalization
* Long-term customer retention
* Interference between customers
* Inventory effects
* Dynamic marketplace equilibrium

Therefore, results should be interpreted as **decision-support estimates**, not guaranteed outcomes.

A production deployment would require controlled experimentation and appropriate business guardrails.

---

# Future Work

Potential extensions include:

* Doubly robust treatment-effect estimation
* Causal forests
* Double/debiased machine learning
* Bayesian hierarchical elasticity models
* Dynamic pricing under inventory constraints
* Reinforcement learning for pricing
* Multi-treatment uplift modeling
* Customer lifetime-value-aware pricing
* Competitive pricing features
* Online experimentation with safety guardrails
* Real-time pricing APIs
* Monitoring for treatment-effect drift

---

# Project Goal

Most pricing analytics systems answer:

> **"What happened when prices changed?"**

This project aims to answer a more useful question:

> **"What would happen if we changed the price, and for whom should we make that change?"**

The combination of **causal inference, elasticity modeling, heterogeneous treatment effects, and pricing optimization** transforms the project from a conventional analytics dashboard into a decision-support framework for marketplace pricing.

---

## Author

**Ashish Rawal**

Project exploring **causal inference, machine learning, pricing analytics, experimentation, and marketplace optimization**.
