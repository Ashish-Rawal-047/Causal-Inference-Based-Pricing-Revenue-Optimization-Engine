# Technical Methodology — Causal Pricing Intelligence & Optimization Engine

This document details the mathematical formulations, identification assumptions, algorithmic implementations, and evaluation boundaries of the engine.

---

## 1. Synthetic Marketplace Data-Generating Process (DGP)

To validate causal estimators against known counterfactuals (`simulation/data_generator.py`, `simulation/ground_truth.py`), transactions are generated chronologically from a structural log-log demand system:

$$\log Q_i = \mu(X_i) + \beta_i \log\left(\frac{P_i}{P_{\text{ref}, i}}\right) + \theta_{\text{comp}} \log\left(\frac{P^{\text{comp}}_i}{P_{\text{ref}, i}}\right) + \delta_{\text{promo}} \cdot \text{promo}_i + \gamma_t + \lambda u_{\text{demand}, i} + \varepsilon_{Q,i}$$

where:
- **Heterogeneous structural price elasticity** $\beta_i < 0$ varies across `region`, `customer_cohort`, `category`, and `historical_orders` (capped in $[-2.4, -0.15]$).
- **Endogenous pricing & confounding**: list prices depend on both segment demand differences and unobserved demand intent $u_{\text{demand},i}$ (e.g., surge/scarcity pricing when unobserved demand is high):
  $$\log P_{\text{list}, i} = \log P_{\text{ref}, i} + \alpha_Z Z_i + \alpha_u u_{\text{demand}, i} + \dots$$
  Because $u_{\text{demand}, i}$ raises *both* price and quantity, naive OLS regression of $\log Q$ on $\log P$ suffers positive omitted-variable bias (understating true price sensitivity).
- **Simpson's Paradox**: Affluent/high-demand segments (`West`) carry higher baseline prices and higher baseline quantities while being relatively price-inelastic ($\beta \approx -0.40$ before cohort/category adjustments), whereas `North` has lower baseline demand, lower prices, and high price sensitivity ($\beta \approx -1.45$). Pooled correlation mixes between-segment demand levels with within-segment price sensitivity.
- **Excluded Instrument ($Z$)**: `supplier_cost_shock` shifts $\log P_{\text{list}, i}$ with strength $\alpha_Z$, has zero direct causal path into $\log Q_i$, and is drawn independently of $u_{\text{demand}, i}$.
- **Difference-in-Differences Policy Rollout**: Starting at `did_start_day = 180`, `North` and `East` (`did_group = 1`) receive an exogenous `12%` regional fee/price reduction (`did_price_drop = 0.12`), while `South` and `West` (`did_group = 0`) share the identical calendar trend $\gamma_t$.
- **Ground-Truth Counterfactuals**: Every transaction records exact potential outcomes $Q(0), Q(1)$, revenue, margin, and `true_cate_qty`, `true_cate_margin` for the targeted promotional discount treatment (`treatment_flag`).

---

## 2. Leakage-Free Preprocessing & Feature Engineering

Implemented in `preprocessing/validation.py`, `preprocessing/cleaning.py`, and `preprocessing/feature_engineering.py`:
- **Chronological expanding features**: Customer, product, and region historical averages (`cust_prior_avg_quantity`, `prod_prior_avg_price`, etc.) use shifted cumulative sums (`cumsum - current`) after sorting by timestamp so row $t$ never uses outcomes at $\ge t$.
- **Strict leakage guards**: `validate_no_leakage` raises `DataValidationError` if post-decision outcomes (`quantity`, `revenue`, `cost`, `margin`) or oracle `true_*` columns enter feature matrices $X$. `validate_temporal_split` enforces $\max(t_{\text{train}}) \le \min(t_{\text{test}})$.

---

## 3. Baseline Price Elasticity & Confounding Analysis

`analysis/elasticity.py` implements four baseline estimators with HC3 heteroskedasticity-robust standard errors:
1. Linear OLS ($Q = \beta_0 + \beta_1 P + \varepsilon$), converted to point elasticity at sample means $\hat{\beta}_1 \cdot (\bar{P}/\bar{Q})$.
2. Naive Log-Log OLS ($\log Q = \beta_0 + \beta_1 \log P + \varepsilon$).
3. Controlled Log-Log OLS ($\log Q = \beta_0 + \beta_1 \log P + \gamma X + \varepsilon$) with VIF multicollinearity, Breusch-Pagan heteroskedasticity, and Jarque-Bera residual normality diagnostics.
4. Segment-level Log-Log OLS across regions, cohorts, categories, and order-frequency groups.

`analysis/confounding.py` quantifies correlation with both price and demand, demonstrates Simpson's Paradox, constructs the `networkx` causal DAG, and pairs predictive feature importance with an explicit caveat that predictive importance is not causal proof.

---

## 4. Causal Inference: 2SLS IV, DiD, and Diagnostics

### Two-Stage Least Squares (`causal/iv.py`)
1. **First stage**: $\widehat{\log P} = \hat{\pi}_0 + \hat{\pi}_1 Z + \hat{\Gamma} X$. Reports partial $F$-statistic and partial $R^2$ on excluded $Z$, flagging weak instruments ($F < 10$).
2. **Second stage**: $\hat{\beta}_{\text{2SLS}} = (\widehat{W}'\widehat{W})^{-1}\widehat{W}' y$.
3. **Asymptotic HC1 Covariance**: Structural residuals are computed using **actual** $\log P$ ($u_i = y_i - W_i' \hat{\beta}_{\text{2SLS}}$), avoiding invalid Stage-2 OLS standard errors.
4. **Endogeneity & Balance Tests**: Wu-Hausman test of first-stage residuals in the outcome equation, plus covariate SMD across $Z$ quantiles.

### Difference-in-Differences (`causal/did.py`)
Estimates $Y = \beta_0 + \beta_1 \text{Treated} + \beta_2 \text{Post} + \beta_3 (\text{Treated} \times \text{Post}) + \Gamma X + \varepsilon$ with HC3 SEs, computes implied Wald/DiD elasticity ($\hat{\beta}_{3,\log Q} / \hat{\beta}_{3,\log P}$), tests pre-treatment differential linear trends, and produces an event-study across relative monthly periods.

### Diagnostics (`causal/diagnostics.py`)
- **Positivity & Balance**: Logistic propensity common support `[0.05, 0.95]` and raw vs. IPW Standardized Mean Differences.
- **SUTVA Assessment**: Product-day order contention share highlighting real-world inventory interference risks.
- **Omitted-Variable Sensitivity**: Cinelli & Hazlett Robustness Value ($RV_{q=1}$) and partial-$R^2$ bias contour grid.

---

## 5. Heterogeneous Treatment Effects & Uplift Evaluation

`uplift/` implements a shared `BaseUpliftEstimator` interface across:
- **S-Learner** (single model baseline with treatment indicator $T$ as feature).
- **T-Learner**: $\hat{\tau}(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)$.
- **X-Learner**: Imputes $D_1 = Y_1 - \hat{\mu}_0(X_1)$ and $D_0 = \hat{\mu}_1(X_0) - Y_0$, fits second-stage $\hat{\tau}_1(x), \hat{\tau}_0(x)$, and blends via clipped propensity weights $\hat{\tau}(x) = g(x)\hat{\tau}_0(x) + (1-g(x))\hat{\tau}_1(x)$.

`uplift/evaluation.py` evaluates AUUC, Qini coefficient, decile uplift, and (on synthetic data) PEHE and Spearman rank correlation. For observational data, cumulative uplift curves use IPW-transformed outcomes $Z_i = Y_i (T_i - \hat{e}_i) / (\hat{e}_i (1 - \hat{e}_i))$, which rely on conditional unconfoundedness and positivity.

---

## 6. Margin-Maximizing Pricing Optimization & Backtesting

`optimization/` decouples `CounterfactualOutcomeModel` from policy selection (`RandomPolicy`, `FlatPolicy`, `SegmentElasticityPolicy`, `UpliftBasedPolicy`). For each candidate price change $\delta \in \{-10\%, -5\%, 0\%, +5\%, +10\%\}$, the optimizer checks `PricingConstraints` (minimum margin %, max discount, price bounds, seller availability, treatment coverage budget) and selects the feasible action maximizing **Expected Incremental Margin**, explicitly rejecting high-volume discounts that erode margin on inelastic segments.

`evaluation/backtest.py` enforces chronological train/test splitting so all policy comparisons occur strictly out-of-time, clearly separating observed historical outcomes, model-estimated counterfactuals, and synthetic ground-truth oracle evaluations.

---

## 7. Observational Data Limits (e.g., Olist)

On observational datasets such as the **Olist Brazilian E-Commerce Dataset**, correlation-based elasticity, confounding diagnostics, and predictive backtesting are supported, whereas valid IV, sharp DiD, and unconfounded uplift require documented exogenous price shocks, policy rollout dates, or randomized experiments not present in raw Olist tables.
