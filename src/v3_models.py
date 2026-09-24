"""v3 model variants: categorical-encoding options, tuned builders, ensemble.

Three ways to present the categorical columns to the tuned GBMs:

- ``v2``     : the current pipeline -- `activity_new` capped to top-10 +
               other/missing, one-hot for all categoricals (reuses
               ``model_pipeline.build_preprocessor``).
- ``te``     : out-of-fold **target encoding** of the *full* 419-category
               `activity_new` (the top-10 cap throws ~52% of its non-null rows
               into "other"). sklearn's ``TargetEncoder.fit_transform`` cross-
               fits internally, so inside a Pipeline each row's encoding comes
               from folds that exclude it -> no target leakage.
- ``native`` : no one-hot; categoricals passed as pandas ``category`` dtype to
               the GBMs' native categorical splitting (XGBoost
               ``enable_categorical=True``; LightGBM auto-detects).

``CalibratedEnsemble`` wraps one model (or a blend of several) and returns the
mean of each member's calibrated probability (isotonic or sigmoid). Averaging
*calibrated* probabilities keeps the blend batch-independent (a rank-average is
not: a customer's rank depends on who else is scored in the same batch).
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from model_pipeline import build_preprocessor, RANDOM_STATE, CATEGORICAL
from tuning import oof_proba, fit_calibrator

HERE = os.path.dirname(__file__)
TUNING_CSV = os.path.join(HERE, "..", "outputs", "tuning_results.csv")
RAW_ACT = "activity_raw"  # full-cardinality activity_new, added by add_raw_activity


def add_raw_activity(X: pd.DataFrame, ids: pd.Series, kind: str) -> pd.DataFrame:
    """Attach the uncapped `activity_new` (by customer id) as ``activity_raw``."""
    sub = "training_data" if kind == "train" else "test_data"
    fname = "ml_case_training_data.csv" if kind == "train" else "ml_case_test_data.csv"
    raw = pd.read_csv(os.path.join(HERE, "..", "data", "raw", sub, fname),
                      usecols=["id", "activity_new"])
    act = raw.set_index("id")["activity_new"].reindex(ids.values).values
    X = X.copy()
    X[RAW_ACT] = pd.Series(act, index=X.index, dtype="object")
    return X


class CategoricalCaster(BaseEstimator, TransformerMixin):
    """Cast columns to pandas ``category`` with categories frozen at fit time.

    Freezing the categories keeps the integer codes identical between train and
    scoring (XGBoost uses the codes); unseen or missing values become NaN.
    """

    def __init__(self, cat_cols=(), drop_cols=()):
        self.cat_cols = cat_cols
        self.drop_cols = drop_cols

    @staticmethod
    def _as_str(s: pd.Series) -> pd.Series:
        return s.astype(object).map(lambda v: v if pd.isna(v) else str(v))

    def fit(self, X, y=None):
        self.categories_ = {c: sorted(self._as_str(X[c]).dropna().unique())
                            for c in self.cat_cols}
        return self

    def transform(self, X):
        X = X.drop(columns=[c for c in self.drop_cols if c in X.columns]).copy()
        for c in self.cat_cols:
            X[c] = pd.Categorical(self._as_str(X[c]), categories=self.categories_[c])
        return X


def make_prep(kind: str, X: pd.DataFrame):
    """Preprocessor for an encoding variant (``X`` must include ``activity_raw``)."""
    if kind == "v2":
        # Identical to the v2 pipeline; activity_raw is dropped (remainder).
        return build_preprocessor(X.drop(columns=[RAW_ACT]))
    if kind == "te":
        num = [c for c in X.columns if c not in CATEGORICAL + [RAW_ACT]]
        return ColumnTransformer([
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), num),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                                                       sparse_output=False))]),
             ["channel_sales", "origin_up"]),
            ("te", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                             ("enc", TargetEncoder(target_type="binary", cv=5, shuffle=True,
                                                   random_state=RANDOM_STATE))]),
             [RAW_ACT]),
        ])
    if kind == "native":
        # GBMs handle NaN natively, so numerics pass through untouched.
        return CategoricalCaster(cat_cols=[RAW_ACT, "channel_sales", "origin_up"],
                                 drop_cols=["activity_new"])
    raise ValueError(kind)


def tuned_params(name: str) -> dict:
    r = pd.read_csv(TUNING_CSV, index_col=0)
    return json.loads(r.loc[name, "best_params"])


def make_model(name: str, native: bool = False):
    """The v2-tuned estimator (single-threaded; callers parallelise outside)."""
    p = tuned_params(name)
    if name == "XGBoost":
        return XGBClassifier(**p, eval_metric="logloss", tree_method="hist", n_jobs=1,
                             random_state=RANDOM_STATE, enable_categorical=native)
    if name == "LightGBM":
        return LGBMClassifier(**p, subsample_freq=1, verbose=-1, n_jobs=1,
                              random_state=RANDOM_STATE)
    if name == "Random Forest":
        return RandomForestClassifier(**p, n_jobs=1, random_state=RANDOM_STATE)
    raise ValueError(name)


def make_pipeline(model_name: str, kind: str, X: pd.DataFrame) -> Pipeline:
    return Pipeline([("prep", make_prep(kind, X)),
                     ("model", make_model(model_name, native=(kind == "native")))])


class CalibratedEnsemble(ClassifierMixin, BaseEstimator):
    """Mean of each member's calibrated churn probability.

    One member = a single calibrated model. Each member's calibrator
    (``method`` = "isotonic" or "sigmoid") is fit on that member's own train
    out-of-fold predictions. After fit, ``oof_raw_`` holds each member's raw
    OOF probability and ``oof_`` the ensemble's calibrated OOF probability,
    used to choose the profit threshold without touching evaluation data.
    """

    def __init__(self, members=(), method="sigmoid"):
        self.members = members
        self.method = method

    def fit(self, X, y):
        y = np.asarray(y)
        self.fitted_, self.cals_, self.oof_raw_ = [], [], []
        for est in self.members:
            raw = oof_proba(clone(est), X, y)
            self.cals_.append(fit_calibrator(raw, y, self.method))
            self.oof_raw_.append(raw)
            self.fitted_.append(clone(est).fit(X, y))
        self.oof_ = np.mean([c.predict(r) for c, r in zip(self.cals_, self.oof_raw_)], axis=0)
        self.classes_ = np.array([0, 1])
        return self

    def recalibrate(self, method, y):
        """Swap the calibration method, reusing the fitted members and OOF preds."""
        y = np.asarray(y)
        self.method = method
        self.cals_ = [fit_calibrator(r, y, method) for r in self.oof_raw_]
        self.oof_ = np.mean([c.predict(r) for c, r in zip(self.cals_, self.oof_raw_)], axis=0)
        return self

    def predict_proba(self, X):
        p = np.mean([c.predict(f.predict_proba(X)[:, 1])
                     for f, c in zip(self.fitted_, self.cals_)], axis=0)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
