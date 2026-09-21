"""Hyperparameter search spaces and RandomizedSearchCV builder.

Design (see docs/PLAN.md and the tuning plan):
- No SMOTE. Class imbalance is handled by the models' native weighting, and
  the weight is itself a *tunable* hyperparameter so CV decides how much
  rebalancing helps PR-AUC ranking:
    * sklearn RF   -> class_weight in {None, 'balanced', 'balanced_subsample'}
    * LightGBM/XGB -> scale_pos_weight in {1, 3, neg/pos ~ 9.1}
  XGBoost docs: use scale_pos_weight (start ~ neg/pos) when AUC ranking is the
  goal; if accurate probabilities are needed, do not rebalance -- calibrate.
  The runner calibrates the tuned model before the profit decision.
- LightGBM docs: num_leaves is the main complexity control (keep < 2^max_depth);
  min_data_in_leaf, lambda_l1/l2, bagging, feature_fraction curb overfitting.
- Scoring = average_precision (PR-AUC): the right ranking metric for a ~10%
  positive class and a costly intervention.
"""
from __future__ import annotations
from scipy.stats import randint
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from model_pipeline import build_preprocessor, RANDOM_STATE

NEG_POS = 14501 / 1595  # ~9.09, sum(neg)/sum(pos) in the training labels

# Models use n_jobs=1; the search parallelises across configurations instead
# (avoids nested-parallel oversubscription on a 4-core box).
SPACES = {
    "Random Forest": (
        RandomForestClassifier(n_jobs=1, random_state=RANDOM_STATE),
        {
            "model__n_estimators": randint(300, 801),
            "model__max_depth": [None, 8, 12, 16, 24],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__max_features": ["sqrt", "log2", 0.3, 0.5],
            "model__max_samples": [None, 0.7, 0.9],
            "model__class_weight": [None, "balanced", "balanced_subsample"],
        },
        30,
    ),
    "LightGBM": (
        LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=1,
                       subsample_freq=1),
        {
            "model__n_estimators": [300, 500, 800, 1200],
            "model__learning_rate": [0.01, 0.02, 0.05, 0.1],
            "model__num_leaves": [15, 31, 63, 127],
            "model__max_depth": [-1, 4, 6, 8, 12],
            "model__min_child_samples": [20, 50, 100, 200],
            "model__subsample": [0.7, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__reg_alpha": [0, 0.1, 1, 5],
            "model__reg_lambda": [0, 0.1, 1, 5],
            "model__min_split_gain": [0, 0.1],
            "model__scale_pos_weight": [1, 3, NEG_POS],
        },
        60,
    ),
    "XGBoost": (
        XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss",
                      n_jobs=1, tree_method="hist"),
        {
            "model__n_estimators": [300, 500, 800, 1200],
            "model__learning_rate": [0.01, 0.02, 0.05, 0.1],
            "model__max_depth": [3, 4, 6, 8, 10],
            "model__min_child_weight": [1, 3, 5, 10],
            "model__subsample": [0.7, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__gamma": [0, 0.1, 1],
            "model__reg_alpha": [0, 0.1, 1, 5],
            "model__reg_lambda": [0.5, 1, 2, 5],
            "model__max_delta_step": [0, 1],
            "model__scale_pos_weight": [1, 3, NEG_POS],
        },
        60,
    ),
}


def make_cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def build_search(name: str, X, n_iter: int | None = None) -> RandomizedSearchCV:
    model, space, default_iter = SPACES[name]
    pipe = Pipeline([("prep", build_preprocessor(X)), ("model", model)])
    return RandomizedSearchCV(
        pipe, space, n_iter=n_iter or default_iter,
        scoring="average_precision", cv=make_cv(), refit=True,
        n_jobs=-1, random_state=RANDOM_STATE, verbose=0, error_score="raise",
    )
