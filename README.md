# Modeling Customer Churn in an Energy Company

This project builds an end-to-end machine learning pipeline to model churn behavior among SME customers of a fictional energy provider, PowerCo. Based on a case study inspired by BCG Gamma, it explores EDA, predictive modeling, and business strategy.

## 📊 Problem Statement
PowerCo is facing a high churn rate. This project aims to:
- Predict churn probability
- Identify key churn drivers
- Recommend a strategy for offering a 20% retention discount

## 🧠 Pipeline (rebuilt)
1. **ABT** — aggregate 2015 price history per customer, then join one row per id (`src/data_prep.py`).
2. **Cleaning** — negative consumption to missing, drop empty columns, cap `activity_new` cardinality, and clip *isolated* extreme consumption values (one row: `cons_12m` 16.1M and `cons_last_month` 4.5M, each far above the next distinct value and shared with no other customer).
3. **Feature engineering** — tenure, months-to-renewal/end, price dynamics (year change, peak–offpeak spread), consumption/margin ratios.
4. **Leakage-safe preprocessing** — impute + `signed_log1p` on the right-skewed columns + scale + one-hot, with SMOTE applied on train folds only (`src/model_pipeline.py`). The log step cuts mean |skew| from 7.86 to 0.83; on 5-fold CV it is metric-neutral, so it is kept for conditioning and interpretability, not for a score lift (toggle: `build_preprocessor(..., log_skewed=False)`).
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

## 📁 Repository Structure
- `data/raw/`: source CSVs — `data/processed/`: generated ABT
- `docs/`: problem brief, feature definitions, rubric, plan
- `src/`: `data_prep.py`, `model_pipeline.py`, `run_pipeline.py`
- `outputs/`: reports (`.md`), metrics (`.csv`), figures, predictions, `report.html`

## 🧪 How to Run
```bash
pip install -r requirements.txt
python src/run_pipeline.py   # builds ABT, runs bake-off, writes all outputs
```

## 📌 Next steps
- Hyperparameter tuning of the top 2 models (RF / LightGBM)
- SHAP for per-customer driver explanations
- Deep-learning and GenAI approaches (rubric stages 5–6)
