# PowerCo SME Churn — Results Report

*Fictional BCG Gamma case study. All numbers below are computed by
`src/run_pipeline.py`; nothing is hand-entered.*

## 1. Headline

- **Best model:** Random Forest — ROC-AUC **0.699** on holdout, **0.698 ± 0.008**
  in 5-fold cross-validation. LightGBM is a close, cheaper alternative
  (ROC-AUC 0.697, higher PR-AUC 0.306, better calibration).
- **Churn is hard to predict** from this data (AUC ~0.70), but the ranking is
  good enough to target a retention budget profitably.
- **A blanket 20% discount loses money.** Targeting it with the model
  (churn probability > 0.20 **and** positive margin) beats the blanket offer
  by **~248k per year** and beats doing nothing by **~23k per year**.

## 2. What was wrong before, and the fix

The original pipeline joined the ~12-row-per-customer price history directly
onto the customer table, exploding it ~12x and breaking the per-customer
aggregation (`std` always NaN, one-hot blowing up to ~497 columns). The new
`data_prep.build_abt` aggregates the price history per `id` **first**
(mean/std/min/max, year-change, peak–offpeak spread), then joins one row per
customer. Row count is asserted at one per id. Result: 67 clean features and
ROC-AUC up from ~0.68 to ~0.70 with far better precision/PR-AUC.

## 3. Model bake-off (holdout, sorted by ROC-AUC)

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|-------|---------|--------|-----|-----------|--------|
| Random Forest | 0.699 | 0.264 | 0.19 | 0.43 | 0.12 |
| LightGBM | 0.697 | 0.306 | 0.27 | 0.52 | 0.18 |
| XGBoost | 0.682 | 0.302 | 0.28 | 0.48 | 0.19 |
| SVC | 0.664 | 0.230 | 0.25 | 0.17 | 0.52 |
| Logistic Regression | 0.642 | 0.181 | 0.23 | 0.14 | 0.61 |
| AdaBoost | 0.633 | 0.155 | 0.24 | 0.18 | 0.33 |
| KNN | 0.620 | 0.150 | 0.23 | 0.15 | 0.44 |
| Naive Bayes | 0.599 | 0.150 | 0.19 | 0.10 | 0.88 |
| Decision Tree | 0.556 | 0.116 | 0.20 | 0.16 | 0.26 |
| Dummy | 0.524 | 0.104 | 0.18 | 0.11 | 0.54 |

## 4. Evaluation metric — why these

At a ~10% churn rate, **accuracy is useless** (predict "no churn" for all →
90% accurate, 0 churners caught). We rank models on **ROC-AUC** (threshold-
independent ranking quality) with **PR-AUC** as the tiebreaker (it focuses on
the rare positive class). For the money decision we then convert probabilities
to a **calibrated** scale (isotonic) and pick the operating point by
**expected value**, not by a default 0.5 cutoff.

## 5. Client questions

**Q1 — Top churn drivers** (permutation importance, Random Forest): power-
subscription margins (`margin_gross_pow_ele`, `margin_net_pow_ele`),
**sales channel**, recent-vs-average consumption ratio, margin per unit
consumption, customer tenure, and **fixed-power price change over 2015**
(`price_p1_fix_change`). Margin and price-change features dominate — consistent
with a price-driven churn story.

**Q2 — Subscribed power vs consumption:** positive and statistically
significant but only **moderate** (Pearson r = 0.10, Spearman rho = 0.40;
p < 1e-38). Higher subscribed power goes with higher consumption, but the two
are not interchangeable — `pow_max` (capacity) carries its own signal.

**Q3 — Sales channel vs churn:** **significant** (chi-square = 126, dof = 7,
p ≈ 4e-24). Churn varies from **5.6% to 12.5%** across channels; the largest
channel (`foosdfpf…`, 7,377 customers) has the highest rate at 12.5%.

## 6. The 20% discount — economics

Decision rule: offering a customer trades full margin for 80% of margin but
guarantees retention, so `EV(offer) − EV(no offer) = margin × (p − 0.20)`.
It pays to offer a **positive-margin** customer only when churn
probability **p > 0.20**.

| Strategy | Customers offered | Annual net margin | vs no-action |
|----------|-------------------|-------------------|--------------|
| No action | 0 | 3,095,984 | — |
| Blanket 20% to all | 16,096 | 2,871,001 | **−224,983** |
| **Targeted (p>0.20 & margin>0)** | 824 | **3,119,303** | **+23,319** |
| Oracle (perfect foresight) | 1,572 | 3,491,367 | +395,383 |

The blanket offer **destroys value**; the targeted rule adds value and, more
importantly, avoids the ~248k/year that the blanket offer wastes. Rank
customers by expected value `margin × (p − 0.20)`, not by probability alone.

## 7. Deliverable

`outputs/predictions/ml_case_test_output_filled.csv` — all 4,024 verification
customers scored and ranked by churn propensity, with a 0/1 flag
(625 flagged at the F1-optimal threshold of 0.14).

## 8. Caveats

- AUC ~0.70 means many individual predictions are uncertain; use the ranking
  and the margin filter, not the raw label.
- The dollar figures assume every offered customer accepts and stays for a
  year (per the brief) and use `net_margin` as the annual margin proxy.
- Isotonic calibration is coarse at the extremes (a few probabilities pin to
  1.0); tuning + SHAP are the recommended next steps.
