# PowerCo SME Churn — Results Report

*Fictional BCG Gamma case study. All numbers below are computed by
the pipeline scripts in `src/`; nothing is hand-entered.*

<!-- current:start -->
## Current status

**Deployed: XGB v2 (sigmoid calibration).** Holdout PR-AUC **0.360**, ROC-AUC 0.712. Targeted 20% discount (profit threshold 0.21, positive margin only): ~1,268 customers offered, ~528 churners retained if all accept (churn 9.9% → 6.6%), **~64,259/yr** vs no action and ~289,242/yr vs a blanket offer. If only half of would-be churners accept while every stayer does, uplift falls to ~14,842/yr. Verification set: 289 of 4,024 customers flagged at the all-data profit threshold 0.23. History: v1 bake-off (Random Forest) → v2 tuning (XGBoost) → v3 Tier-1 tests found no significant gain, so XGB v2 stays; its calibration moved to sigmoid.
<!-- current:end -->

## 1. Headline (v1 — superseded, see Current status)

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

## 7. Deliverable (v1 — superseded, see Current status)

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

<!-- v2-tuning:start -->
## 9. Hyperparameter tuning (v2)

**Method.** `RandomizedSearchCV` (5-fold stratified, scoring = PR-AUC) over
Random Forest (30 configs), LightGBM (60) and XGBoost (60). No SMOTE: class
imbalance is handled by the models' native weighting, and the weight itself is
searched (`class_weight` / `scale_pos_weight ∈ {1, 3, ~9}`), per the XGBoost
guidance to rebalance for ranking and *calibrate* for probability. The tuned
model is isotonic-calibrated on train out-of-fold predictions, the profit
threshold is picked on those train OOF probabilities, and all metrics are on the
untouched holdout. Ranking: PR-AUC → profit uplift → Brier → ROC-AUC.

| Model | PR-AUC v1 | PR-AUC tuned | Δ | ROC-AUC | Brier | Thr | Precision | Recall | F0.5 | Uplift vs no-action /yr |
|---|---|---|---|---|---|---|---|---|---|---|
| **XGBoost** | 0.302 | **0.360** | +0.058 | 0.712 | 0.077 | 0.16 | 0.41 | 0.34 | 0.39 | 67,121 |
| LightGBM | 0.306 | 0.355 | +0.049 | 0.705 | 0.077 | 0.17 | 0.40 | 0.30 | 0.37 | 44,713 |
| Random Forest | 0.265 | 0.320 | +0.056 | 0.724 | 0.079 | 0.16 | 0.31 | 0.36 | 0.32 | 50,623 |

**Selected model: XGBoost.** Best params: `subsample=1`, `scale_pos_weight=1`, `reg_lambda=2`, `reg_alpha=0`, `n_estimators=800`, `min_child_weight=1`, `max_depth=8`, `max_delta_step=1`, `learning_rate=0.05`, `gamma=0`, `colsample_bytree=0.6`.

All three models improved on their untuned v1 PR-AUC.
The winner lifts PR-AUC from 0.302 to **0.360** and raises the
targeted-discount uplift to **~67,121/yr** vs no-action
(v1 Random Forest: ~23,319/yr), while still beating the blanket offer by
~292,104/yr. Top drivers for the tuned model:
`margin_net_pow_ele`, `margin_gross_pow_ele`, `pow_max`, `months_to_end`, `months_to_renewal`, `net_margin`. Verification set re-scored with the
tuned calibrated model (275 flagged at threshold
0.18). Runtime 30.5 min.
<!-- v2-tuning:end -->

<!-- v3:start -->
## 10. v3 — Tier-1 improvements, tested honestly

**Candidates** (all with the v2-tuned hyperparameters): out-of-fold target
encoding of the full 419-category `activity_new` (`+TE`), native categorical
splits (`native`), and XGBoost + LightGBM rank-average blends.

**Pre-registered rule** (fixed before any result was seen): Deploy the top mean-CV-PR-AUC candidate only if it beats XGB v2 with Holm-adjusted p < 0.05 (Nadeau-Bengio corrected paired t-test, repeated 5x5 CV) and is not worse on the holdout.

| Config | CV PR-AUC (mean ± sd) | Δ vs XGB v2 | 95% CI (corrected) | Holm p | Holdout PR-AUC |
|---|---|---|---|---|---|
| Blend native | 0.3270 ± 0.0290 | +0.0071 | -0.0038 to +0.0180 | 1.00 | 0.3669 |
| XGB native | 0.3242 ± 0.0279 | +0.0043 | -0.0070 to +0.0155 | 1.00 | 0.3625 |
| Blend v2 | 0.3239 ± 0.0283 | +0.0040 | -0.0017 to +0.0097 | 1.00 | 0.3634 |
| Blend +TE | 0.3215 ± 0.0275 | +0.0016 | -0.0073 to +0.0105 | 1.00 | 0.3670 |
| **XGB v2** (deployed) | 0.3199 ± 0.0285 | — | — | — | 0.3602 |
| LGBM v2 | 0.3195 ± 0.0276 | -0.0005 | -0.0102 to +0.0093 | 1.00 | 0.3554 |
| LGBM native | 0.3178 ± 0.0289 | -0.0021 | -0.0145 to +0.0103 | 1.00 | 0.3584 |
| XGB +TE | 0.3173 ± 0.0283 | -0.0026 | -0.0121 to +0.0070 | 1.00 | 0.3668 |
| LGBM +TE | 0.3170 ± 0.0283 | -0.0029 | -0.0154 to +0.0096 | 1.00 | 0.3612 |
| RF v2 | 0.2932 ± 0.0288 | -0.0268 | -0.0441 to -0.0094 | 0.04 | 0.3201 |

Repeated 25-split CV on the training split; the corrected t-test
(Nadeau & Bengio) accounts for overlapping training folds, and Holm corrects for
testing 9 candidates at once.

**Decision:** Top CV candidate Blend native not adopted: its CV gain of +0.0071 is not significant (Holm p=1.00, 95% CI -0.0038 to +0.0180). The single holdout is a warning: several
candidates beat XGB v2 there, but the repeated CV shows those gains are noise.

**Calibration — a defect fixed.** v2 used isotonic calibration, whose step
function tied customers together. Calibration was chosen by cross-fitted profit on training data only: sigmoid 29,615 vs isotonic 28,856 (training-fold uplift). On the holdout the old isotonic version earns 2,862/yr more — the two sources disagree, both gaps are small, and the rule was fixed before the holdout was seen, so it stands. Sigmoid keeps every customer's score distinct (4,024 vs 48 under isotonic), so the ranked list the brief asks for is exact (holdout PR-AUC 0.360 vs 0.341).

| Holdout | XGB v2 (as deployed in v2, isotonic) | XGB v2 (sigmoid) |
|---|---|---|
| PR-AUC | 0.3412 | 0.3602 |
| ROC-AUC | 0.7094 | 0.7122 |
| Brier (lower = better calibrated) | 0.0765 | 0.0781 |
| Distinct scores (of 4,024) | 48 | 4,024 |
| Profit threshold | 0.16 | 0.21 |
| Customers offered | 1,316 | 1,268 |
| Churners retained (all accept) | 540 | 528 |
| Uplift vs no action /yr | 67,121 | 64,259 |
| Uplift vs blanket /yr | 292,104 | 289,242 |

## 11. What if customers don't all accept?

The brief assumes every offered customer accepts. Below, churners and stayers
accept at separate rates; the threshold is re-optimised on training data for
each scenario and evaluated on the holdout. The dangerous case is when loyal
customers take the discount more readily than customers who have already
decided to leave.

| Scenario | Break-even p | Threshold | Offered | Churners retained | Churn after | Uplift vs no action /yr |
|---|---|---|---|---|---|---|
| All accept (brief's assumption) | 0.20 | 0.21 | 1,268 | 528 | 6.6% | 64,259 |
| 75% of everyone accepts | 0.20 | 0.21 | 1,268 | 396 | 7.5% | 48,194 |
| 50% of everyone accepts | 0.20 | 0.21 | 1,268 | 264 | 8.3% | 32,130 |
| 25% of everyone accepts | 0.20 | 0.21 | 1,268 | 132 | 9.1% | 16,065 |
| Churners 75%, stayers 100% | 0.25 | 0.26 | 792 | 303 | 8.0% | 28,048 |
| Churners 50%, stayers 100% | 0.33 | 0.33 | 428 | 142 | 9.0% | 14,842 |
| Churners 25%, stayers 100% | 0.50 | 0.41 | 276 | 56 | 9.6% | 7,075 |

If every stayer accepts, targeting stops paying when only 5% of would-be churners accept; it stays profitable once at least ~10% of would-be churners accept. The threshold tightens as acceptance falls, so the
offer list shrinks — and the uplift shrinks with it. Thresholds are chosen empirically on training data, so they can sit slightly off the theoretical break-even.

## 12. Data roadmap — why the model plateaus, and what to collect

**Evidence.** Churn barely moves with contract timing (0-3 months: 10.6%, 3-6 months: 10.1%, 6-9 months: 9.1%, 9-12 months: 9.6%, 12+ months: 11.1%; chi-square p = 0.16, not significant), and the
dataset holds none of the behavioural signals that lift churn models elsewhere:

| Signal | Fields found |
|---|---|
| Churn reason (switch vs closure/move) | none |
| Payment behaviour | none |
| Customer contact / complaints | none |
| Smart-meter interval usage | none |
| Competitor price benchmark | none |
| Campaign / offer response | campaign_disc_ele (0% populated) |

**Priorities** (a recommendation — ranked by expected value and ease, a
judgement rather than a computed result):

| # | Data to collect | Why it matters here | Effort |
|---|---|---|---|
| 1 | Randomised pilot of the discount (offer vs hold-out control) | Measures real acceptance and *incremental* retention. The business case swings from ~64,259/yr (everyone accepts) to ~7,075/yr (a quarter of churners accept, all stayers do) on this one unknown. | Low |
| 2 | Churn reason at cancellation (switched supplier vs closed / moved) | A discount cannot save a closure. Reason codes let the model target voluntary switching only; the label today is a bare 0/1. | Low |
| 3 | Competitor price benchmark per customer profile | Tests the head of SME's price hypothesis directly; the best 2015 price-history feature ranks only #16 of 68 in the deployed model's importance. | Medium |
| 4 | Payment behaviour (late payments, direct-debit changes) | A standard early-warning signal; no payment field exists in the data. | Low–Medium |
| 5 | Customer contact and complaint log | Engagement and dissatisfaction signal; no contact field exists in the data. | Medium |
| 6 | Smart-meter interval usage | Sudden usage drops can reveal a business winding down or switching; the data holds only 12-month aggregates. | Medium–High |
<!-- v3:end -->
