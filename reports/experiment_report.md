# Causal Pricing Intelligence — Automated Experiment Report

## Dataset Overview
- Transactions analyzed: **50,000**
- Unique customers: **4,000** | Unique products: **120**
- Time window: `2024-01-01 00:01:00` to `2024-12-30 23:51:00`
- Observed total revenue: **\$17,386,171.04** | Total margin: **\$6,793,384.25** (39.1%)

## Experiment 1: Naive vs. Causal Elasticity
| method | elasticity | std_error | ci_low | ci_high | true_reference | bias |
| --- | --- | --- | --- | --- | --- | --- |
| Linear OLS (Mean Elasticity) | 0.1717 | 0.0052 | 0.1615 | 0.1820 | -0.9940 | 1.1657 |
| Naive Log-Log OLS | 0.1127 | 0.0042 | 0.1044 | 0.1210 | -0.9940 | 1.1067 |
| Controlled Log-Log OLS | -0.5617 | 0.0061 | -0.5736 | -0.5498 | -0.9940 | 0.4323 |
| Causal 2SLS IV | -0.9773 | 0.0072 | -0.9915 | -0.9632 | -0.9940 | 0.0166 |
| Causal DiD (Implied Treated Elasticity) | -1.3511 | nan | nan | nan | -1.3056 | -0.0455 |

- First-stage IV F-statistic: **152064.66** (Weak=False)
- DiD pre-trend slope p-value: **0.4315**
- Simpson's Paradox detected: **True**

## Experiment 2: Segment Heterogeneity (ATE vs. CATE)
- Overall Quantity ATE: **1.1257** | Margin ATE: **\$-40.41** (SD=80.16)

| segment_type | segment_value | n_obs | pred_cate_qty_mean | pred_cate_qty_std | pred_cate_margin_mean | pred_cate_margin_std | positive_margin_uplift_share | true_cate_qty_mean | true_cate_margin_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| category | Beauty | 1921 | 1.4709 | 0.8143 | -19.25 | 13.59 | 0.0026 | 1.2875 | -10.66 |
| category | Grocery | 2735 | 1.2480 | 0.6552 | -19.93 | 13.31 | 0.0037 | 1.2328 | -9.3642 |
| category | Fashion | 3392 | 1.1005 | 0.5574 | -24.79 | 20.77 | 0.0021 | 1.1462 | -15.37 |
| category | Home | 1996 | 1.0512 | 0.5277 | -38.87 | 44.17 | 0.0060 | 1.0141 | -32.63 |
| category | Electronics | 2456 | 0.8149 | 0.5620 | -102.61 | 158.64 | 0.0041 | 0.9294 | -69.82 |
| customer_cohort | New | 3831 | 1.1931 | 0.6453 | -33.96 | 58.37 | 0.0018 | 1.2044 | -19.77 |
| customer_cohort | Returning | 4097 | 1.2027 | 0.6741 | -35.58 | 54.17 | 0.0041 | 1.1370 | -24.05 |
| customer_cohort | Loyal | 2979 | 1.0425 | 0.6186 | -45.53 | 90.88 | 0.0054 | 1.0078 | -32.60 |
| customer_cohort | VIP | 1593 | 0.9212 | 0.6276 | -58.78 | 136.76 | 0.0025 | 1.1081 | -39.83 |
| order_frequency_group | Occasional (2-5) | 75 | 1.2112 | 0.5768 | -33.50 | 48.09 | 0.0000 | 1.2415 | -22.38 |
| order_frequency_group | Regular (6-12) | 3251 | 1.2130 | 0.6481 | -34.27 | 61.12 | 0.0025 | 1.1760 | -20.97 |
| order_frequency_group | Power (13+) | 9174 | 1.0941 | 0.6543 | -42.64 | 85.99 | 0.0039 | 1.1035 | -28.89 |

## Experiment 3: Uplift Model Benchmark
| model | auuc | auuc_normalized | qini_coefficient | top_decile_uplift | bottom_decile_uplift | estimated_ate | pehe | ate_abs_error | cate_rank_corr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S-Learner | -83,636.65 | 0.2145 | 0.2467 | -10.33 | -95.60 | -27.85 | 37.77 | 1.0599 | 0.7022 |
| T-Learner | -75,429.13 | 0.2426 | 0.2262 | -5.9794 | -84.91 | -44.75 | 86.30 | 17.96 | 0.6492 |
| X-Learner | -90,763.79 | 0.1902 | 0.1784 | -16.39 | -56.43 | -35.23 | 52.96 | 8.4402 | 0.7029 |

## Experiment 4: Chronological Backtest & Pricing Policy Comparison
- Held-out test rows: **12,500**
- Observed historical test revenue: **\$3,768,079.40** | margin: **\$1,390,601.84**

| policy | total_revenue | total_margin | incremental_revenue | incremental_margin | incremental_orders | discount_expenditure | treatment_coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Flat Policy | 3,933,313.49 | 1,394,495.42 | 325,579.72 | 47,993.34 | 6,084.86 | 172,549.84 | 0.6891 |
| Random Policy | 3,767,556.19 | 1,423,924.81 | 159,822.42 | 77,422.72 | 1,590.46 | 102,344.54 | 0.6734 |
| Segment-Based Policy | 3,754,464.05 | 1,601,069.97 | 146,730.27 | 254,567.88 | -2,999.96 | 0.0000 | 0.8916 |
| Uplift-Based Policy | 3,786,279.10 | 1,605,611.07 | 178,545.33 | 259,108.98 | -1,948.57 | 13,435.85 | 0.8916 |

> Note: Counterfactual policy metrics on synthetic data reflect the synthetic marketplace ground-truth simulator, not real-world business impact.

## Robustness Analysis: Confounding Sensitivity
| confounding_strength | true_elasticity | naive_ols | controlled_ols | iv_2sls | did_implied | naive_abs_error | controlled_abs_error | iv_abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | -1.0293 | 0.0655 | -1.0032 | -1.0227 | -1.2194 | 1.0948 | 0.0261 | 0.0066 |
| 0.5000 | -1.0293 | 0.0735 | -0.8876 | -1.0202 | -1.3488 | 1.1028 | 0.1417 | 0.0091 |
| 1.0000 | -1.0293 | 0.1024 | -0.6023 | -1.0183 | -1.4923 | 1.1317 | 0.4270 | 0.0110 |
| 1.8000 | -1.0293 | 0.1792 | -0.0826 | -1.0125 | -1.7285 | 1.2085 | 0.9467 | 0.0168 |
