# Modeling Customer Churn in an Energy Company

This project builds an end-to-end machine learning pipeline to model churn behavior among SME customers of a fictional energy provider, PowerCo. Based on a case study inspired by BCG Gamma, it explores EDA, predictive modeling, and business strategy.

## 📊 Problem Statement
PowerCo is facing a high churn rate. This project aims to:
- Predict churn probability
- Identify key churn drivers
- Recommend a strategy for offering a 20% retention discount

## 🧠 Pipeline (rebuilt)
1. **ABT** — aggregate 2015 price history per customer, then join one row per id (`src/data_prep.py`).
2. **Cleaning** — negative consumption to missing, drop empty columns, cap `activity_new` cardinality.
3. **Feature engineering** — tenure, months-to-renewal/end, price dynamics (year change, peak–offpeak spread), consumption/margin ratios.
4. **Leakage-safe preprocessing** — impute + scale + one-hot, with SMOTE applied on train folds only (`src/model_pipeline.py`).
5. **Model bake-off** — 10 classifiers compared, top models cross-validated.
6. **Business evaluation** — calibrated probabilities; ranked by ROC-AUC / PR-AUC.
7. **Discount economics** — expected-value targeting of the 20% offer.

## 🚀 Results (holdout, sorted by ROC-AUC)
| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|-------|---------|--------|-----|-----------|--------|
| **Random Forest** | **0.699** | 0.264 | 0.19 | 0.43 | 0.12 |
| LightGBM | 0.697 | **0.306** | 0.27 | 0.52 | 0.18 |
| XGBoost | 0.682 | 0.302 | 0.28 | 0.48 | 0.19 |
| SVC | 0.664 | 0.230 | 0.25 | 0.17 | 0.52 |
| Logistic Regression | 0.642 | 0.181 | 0.23 | 0.14 | 0.61 |

Random Forest wins on ROC-AUC (CV 0.698 ± 0.008); LightGBM is competitive
and leads on PR-AUC/calibration at a fraction of the cost. Full table:
`outputs/model_comparison.csv`.

**Discount economics (annual net margin, full population):** a **blanket
20% discount destroys ~225k** vs doing nothing, because most customers
would not have churned. A **targeted** offer (calibrated churn probability
> 0.20 **and** positive margin) offers to ~824 customers, retains ~260 real
churners, and beats the blanket offer by **~248k/year**. See
`outputs/economics.md`.

<!-- v2-tuning:start -->
## 🎯 Tuned results (v2 — RandomizedSearchCV, PR-AUC objective)

SMOTE was replaced by native class weighting (RF `class_weight`, GBM
`scale_pos_weight`) as a *tunable* hyperparameter; probabilities are isotonic-
calibrated on train out-of-fold predictions; the profit threshold is chosen on
train OOF; every number below is on the untouched 25% holdout. Models are
ranked by **PR-AUC → profit uplift → Brier → ROC-AUC**.

| Model | PR-AUC v1 | PR-AUC tuned | Δ | ROC-AUC | Brier | Thr | Precision | Recall | F0.5 | Uplift vs no-action /yr |
|---|---|---|---|---|---|---|---|---|---|---|
| **XGBoost** | 0.302 | **0.360** | +0.058 | 0.712 | 0.077 | 0.16 | 0.41 | 0.34 | 0.39 | 67,121 |
| LightGBM | 0.306 | 0.355 | +0.049 | 0.705 | 0.077 | 0.17 | 0.40 | 0.30 | 0.37 | 44,713 |
| Random Forest | 0.265 | 0.320 | +0.056 | 0.724 | 0.079 | 0.16 | 0.31 | 0.36 | 0.32 | 50,623 |

**Selected: XGBoost** — PR-AUC 0.360 (v1 0.302), ROC-AUC
0.712. At the profit-optimal threshold 0.16 it
offers the discount to ~1,316 customers, retains
~540 real churners, and adds
**~67,121/yr** over doing nothing
(**~292,104/yr** better than a blanket offer).
All three models improved on their v1 baseline.
Full table: `outputs/tuning_results.csv`. Re-run: `python src/run_tuning.py`.
<!-- v2-tuning:end -->

## 📁 Repository Structure
- `data/raw/`: source CSVs — `data/processed/`: generated ABT
- `docs/`: problem brief, feature definitions, rubric, plan
- `src/`:
  - `data_prep.py` — ABT (one row per customer), cleaning, features
  - `model_pipeline.py` — v1 preprocessing and the 10-model roster
  - `tuning.py`, `v3_models.py` — v2 search spaces, calibration, v3 encodings and ensemble
  - `economics.py` — discount economics, profit curve, acceptance sensitivity
  - `run_pipeline.py` → `run_tuning.py` → `run_v3.py` — v1, v2, v3 runners
  - `build_report.py`, `update_docs.py`, `report_content.py` — report and docs, generated from outputs
- `outputs/`: reports (`.md`), metrics (`.csv`), figures, predictions, `report.html`

## 🧪 How to Run
```bash
pip install -r requirements.txt
cd src
python run_pipeline.py   # v1: ABT, 10-model bake-off, client questions (~5 min)
python run_tuning.py     # v2: RandomizedSearchCV of RF / LightGBM / XGBoost (~30 min)
python run_v3.py         # v3: Tier-1 tests, calibration, acceptance sensitivity (~15 min)
python update_docs.py && python build_report.py   # refresh README, RESULTS.md, report.html
```

## 📌 Next steps
- Collect the data in the roadmap (`outputs/RESULTS.md` §12) — the model is data-limited, not tuning-limited
- Run the discount as a randomised pilot to measure real acceptance and incremental retention
- SHAP for per-customer driver explanations
- Deep-learning and GenAI approaches (rubric stages 5–6)
