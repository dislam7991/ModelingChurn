# 🔌 Modeling Customer Churn in an Energy Company

This project builds an end-to-end machine learning pipeline to model churn behavior among SME customers of a fictional energy provider, PowerCo. Based on a case study inspired by BCG Gamma, it explores EDA, predictive modeling, and business strategy.

## 📊 Problem Statement
PowerCo is facing a high churn rate. This project aims to:
- Predict churn probability
- Identify key churn drivers
- Recommend a strategy for offering a 20% retention discount

## 🧠 Project Steps
1. Data Cleaning & Merging
2. Exploratory Data Analysis (EDA)
3. Feature Engineering & Multicollinearity Reduction
4. Class Balancing (SMOTE)
5. Model Training: Logistic, Tree-based, SVM, XGBoost, etc.
6. Evaluation: AUC, F1, Precision, Recall
7. Strategic Recommendation

## 🚀 Current Results (Round 1)
| Model           | AUC   | F1-Score | Precision | Recall |
|----------------|-------|----------|-----------|--------|
| Random Forest  | 0.68  | 0.18     | 0.36      | 0.12   |
| XGBoost        | 0.67  | 0.21     | 0.38      | 0.15   |
| LightGBM       | 0.66  | 0.18     | 0.35      | 0.12   |

## 📁 Repository Structure
- `data/`: Raw & cleaned input CSVs
- `notebooks/`: Jupyter notebooks
- `src/`: Modular Python scripts
- `outputs/`: Plots and predictions

## 🧪 How to Run
```bash
pip install -r requirements.txt
python src/modeling.py
```

## 📌 To Do
- Visualizations (EDA, VIF, ROC)
- Feature selection
- Model tuning
- Revenue impact analysis
