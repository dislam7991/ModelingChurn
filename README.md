# 🔌 Modeling Customer Churn for PowerCo (Energy Company)

## 📘 Overview
A machine learning pipeline to predict SME customer churn for PowerCo, based on a fictional BCG Gamma case study. The model helps identify high-risk clients and inform discount-based retention strategies.

## 🧠 Problem Statement
PowerCo is losing SME customers at above-average rates. We aim to:
- Predict churn risk
- Understand drivers of churn
- Evaluate if a 20% discount offer is financially sound

## 📊 Data
- `training_data.csv` — customer records (Jan 2016)
- `training_output.csv` — churn labels (by Mar 2016)
- `historical_data.csv` — 2015 price data

## 🔍 Approach
- Exploratory Data Analysis (EDA)
- Feature engineering + date parsing
- Handling class imbalance with SMOTE
- Benchmarking multiple classifiers
- Model evaluation via AUC, F1, Precision, Recall
- Business recommendations

## 📈 Model Performance
| Model               | AUC   | F1   | Accuracy | Precision | Recall |
|---------------------|-------|------|----------|-----------|--------|
| Random Forest       |       | ...  | ...      | ...       | ...    |
| Logistic Regression | ...   | ...  | ...      | ...       | ...    |
| XGBoost             | ...   | ...  | ...      | ...       | ...    |

## 🎯 Key Insights
- Top churn drivers: 
- Strong correlation between X and Y...
- Channel Z has highest churn rate...

## 💡 Business Recommendation
If the discount is offered to top % of customers by churn probability:
- Estimated retention: %
- Net revenue impact: Z dollars

## 🧪 How to Run
