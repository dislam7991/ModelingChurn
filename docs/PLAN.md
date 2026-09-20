# Project Plan — Modeling Churn for PowerCo (SME)

This plan links the BCG problem brief and the InnovatiCS rubric to concrete
work in this repository. See `docs/problem_statement.md`,
`docs/feature_definitions.md`, and `docs/rubric.md` for the imported source
material.

## 1. Current state of the repo

- `data/raw/` holds the training and test CSV files. They match the brief:
  - Training: 16,096 customers, 193,002 price-history rows (~12 per
    customer), one label file.
  - Test: 4,024 customers, 48,236 price-history rows, one output template.
- Class balance: churn = 1 for 1,595 of 16,096 customers (**~9.9%**). The
  data is imbalanced.
- `src/preprocessing.py`, `src/modeling.py`, `src/utils.py`, and
  `notebooks/03_modeling.ipynb` hold a first modeling pass.
- Best current result (README): AUC ~0.68, recall ~0.12 — the model rarely
  catches real churners.

## 2. Defects found in the current pipeline

These explain the weak results and must be fixed first.

1. **Broken price aggregation (the main bug).**
   `load_and_merge_data` left-joins the price history onto the customer
   table. Because the history has ~12 rows per customer, the merged table
   explodes to ~193k rows and every customer feature is duplicated 12 times.
   Then `clean_and_engineer` calls `df.groupby(df.index).agg(...)`. After the
   merge each row has a unique index, so each group is a single row: `std` is
   always `NaN` and each "mean" price is just that one row's value. The
   following `df.drop_duplicates(subset=df.index)` is also wrong — the index
   is not a column. Net effect: the price history is never truly aggregated
   per customer, and rows may still be duplicated. **Fix: aggregate the price
   history per `id` first, then join one row per customer onto the customer
   table (a proper ABT).**

2. **High-cardinality one-hot blow-up.**
   `activity_new` and other hashed codes have many distinct values.
   `pd.get_dummies` on them produces ~497 columns (the notebook trains "SVC
   on 497 features"). Most columns are sparse noise; they slow SVC to a crawl
   and hurt every model. **Fix: group rare categories into `Other`, or
   frequency/target-encode, and cap cardinality.**

3. **Strong signals thrown away.**
   `date_end`, `date_renewal`, and `date_modif_prod` are dropped, and the
   price *change* over 2015 is never computed. In this case study the price
   change (December vs January, and the peak/off-peak spread) is the classic
   churn driver. **Fix: engineer time-to-renewal, contract lengths, and price
   dynamics.**

4. **Business layer missing.**
   No discount economics and no scoring of the test set into the output
   template — both are required deliverables.

## 3. Target architecture

Build a clean, reproducible ABT and pipeline:

```
data/raw/            # unchanged source CSVs (already present)
data/processed/      # generated ABT (one row per customer) + scored output
src/
  data.py            # load raw files
  abt.py             # aggregate price history, join, build one-row-per-id ABT
  features.py        # feature engineering (tenure, renewal, price dynamics)
  preprocess.py      # impute, encode, scale, split, class-imbalance handling
  modeling.py        # train + compare models, cross-validated
  evaluate.py        # AUC/PR, calibration, confusion, feature importance
  economics.py       # 20% discount net-value analysis and targeting rule
  score.py           # score the test set into the output template
notebooks/           # narrative EDA + modeling for the presentations
outputs/             # figures, metrics tables, predictions
docs/                # imported brief, feature defs, rubric, this plan
```

## 4. Feature engineering (per customer)

- **Price dynamics** from the 2015 history, aggregated per `id`:
  mean and std of each `price_p*_var` / `price_p*_fix`; the change from the
  first to the last month of 2015; the max month-to-month jump; the
  peak/off-peak spread (`p1` vs `p2`/`p3`).
- **Contract timing** (reference date = 2016-01-01): tenure from
  `date_activ`; months to `date_end` and to `date_renewal`; time since
  `date_modif_prod`; total lifespan from `date_first_activ`.
- **Consumption**: keep `cons_*`; add per-product and per-kW ratios; flag
  negative consumption values as missing (a known data-quality issue).
- **Stickiness**: `has_gas` -> 0/1; `nb_prod_act`; dual-fuel flag.
- **Margins**: keep for the economics layer; add `net_margin` per unit
  consumption.

## 5. Modeling

- Split: stratified train/validation; keep a held-out set. Do all encoding,
  imputation, scaling, and resampling **inside** cross-validation folds to
  avoid leakage. Handle imbalance with class weights and/or SMOTE (fit on
  train folds only).
- Compare: logistic regression (baseline + interpretable), random forest,
  gradient boosting (XGBoost / LightGBM), and a calibrated best model.
- (Rubric stages 5–6) add a small neural-net baseline and a GenAI-assisted
  approach later.
- Select on **AUC and PR-AUC / recall at a business-set threshold**, not
  accuracy — accuracy is misleading at ~10% churn.
- Report the most explicative variables (client question 1) via model
  importance and SHAP.

## 6. Answering the three client questions

1. Most explicative variables — from importance + SHAP.
2. `pow_max` vs consumption — correlation and scatter.
3. `channel_sales` vs churn — churn rate by channel with a significance test.

## 7. Discount economics (the real recommendation)

- Expected value of offering the discount to a customer:
  `p(churn) * margin_saved  -  (1 - p(churn)) * discount_cost` (and the
  churner case net of the discounted margin).
- Rank customers by expected net value, not by churn probability alone.
- Recommend a targeting rule: offer the 20% discount only where predicted
  churn risk **and** margin are both high enough for positive expected value.
- Produce the scored test set in `ml_case_test_output_template.csv` format.

## 8. Suggested order of work

1. Build the ABT (`src/abt.py`) and validate: exactly one row per customer,
   no explosion. **(Unblocks everything; fixes the main bug.)**
2. Rewrite EDA as a notebook + a data-quality report (rubric stage 2).
3. Feature engineering (`src/features.py`).
4. Leakage-safe preprocessing and model comparison.
5. Evaluation, feature importance, client-question answers.
6. Discount economics + test-set scoring.
7. Package figures/tables for the presentations.

## 9. Open questions for the team

- Which presentation stage is next? That sets the immediate priority.
- Is a deep-learning and a GenAI model required for grading now (stages
  5–6), or is classical ML (stage 4) the current focus?
- Should the ABT reference date be 2016-01-01 for every customer (matches the
  brief), and should 2015 price history be treated as the full pre-period?
