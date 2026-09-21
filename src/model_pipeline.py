"""Leakage-safe preprocessing + the model bake-off roster.

Preprocessing (impute, scale, one-hot) and SMOTE are wrapped inside an
imbalanced-learn Pipeline so they are fit only on training folds during
cross-validation -- no information leaks from validation/test data.
"""
from __future__ import annotations

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (StandardScaler, OneHotEncoder,
                                   FunctionTransformer)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from data_prep import SKEWED_COLS

RANDOM_STATE = 42
CATEGORICAL = ["activity_new", "channel_sales", "origin_up"]


def signed_log1p(X):
    """``sign(x) * log1p(|x|)`` -- a log transform that tolerates negatives.

    The consumption and forecast columns are right-skewed by a factor of 5 to 9,
    which hurts the scale-sensitive models (Logistic Regression, SVC, KNN) and
    leaves StandardScaler with a mean and variance dominated by a handful of
    very large customers. A log transform fixes that while keeping every
    customer distinct -- unlike a percentile cap, which would tie dozens of
    genuinely different customers to one value.

    For non-negative input this is exactly ``log1p``. The margin columns can be
    legitimately negative, so the sign is preserved instead of producing NaN.
    The tree ensembles are unaffected: the transform is monotonic.
    """
    X = np.asarray(X, dtype=float)
    return np.sign(X) * np.log1p(np.abs(X))


def build_preprocessor(feature_df, log_skewed: bool = True):
    """ColumnTransformer: median-impute + scale numerics, one-hot categoricals.

    With ``log_skewed`` (the default), the columns in ``SKEWED_COLS`` get a
    ``signed_log1p`` step between imputation and scaling. Doing it here rather
    than in the ABT keeps ``data/processed/`` in real units, so the discount
    economics and the client-question report still read in currency and kWh.

    Measured effect (5-fold CV, SMOTE on, seed 42): the transform cuts mean
    |skew| across those columns from 7.86 to 0.83, but it moves ROC-AUC and
    PR-AUC by less than one standard deviation for every model in the roster --
    it is **metric-neutral here**, not a lift. It is kept on because it makes
    the scaled feature space well-conditioned (StandardScaler no longer centres
    on a mean dragged by a few very large customers) and makes the Logistic
    Regression coefficients readable as elasticities. Pass ``log_skewed=False``
    to reproduce the untransformed baseline.
    """
    categorical = [c for c in CATEGORICAL if c in feature_df.columns]
    numeric = [c for c in feature_df.columns if c not in categorical]
    skewed = [c for c in numeric if c in SKEWED_COLS] if log_skewed else []
    plain = [c for c in numeric if c not in skewed]

    numeric_tf = SkPipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    skewed_tf = SkPipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    categorical_tf = SkPipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                                 sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_tf, plain),
        ("skew", skewed_tf, skewed),
        ("cat", categorical_tf, categorical),
    ])


def get_models():
    """The full bake-off roster (matches the team's notebook set)."""
    return {
        "Dummy Classifier": DummyClassifier(strategy="stratified",
                                            random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(max_iter=2000,
                                                   random_state=RANDOM_STATE),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "SVC": SVC(kernel="rbf", cache_size=1024, probability=True,
                   random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                                random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(eval_metric="logloss", n_jobs=-1,
                                 random_state=RANDOM_STATE),
        "LightGBM": LGBMClassifier(n_jobs=-1, random_state=RANDOM_STATE,
                                   verbose=-1),
    }


def make_pipeline(model, preprocessor, use_smote=True):
    """Wrap preprocessor (+ optional SMOTE) + model into one estimator.

    GaussianNB gets a dense array (OneHotEncoder output is sparse); other
    models handle sparse fine. SMOTE runs after preprocessing, on train only.
    """
    steps = [("prep", preprocessor)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("model", model))
    return ImbPipeline(steps)
