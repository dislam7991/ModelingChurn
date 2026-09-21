"""Leakage-safe preprocessing + the model bake-off roster.

Preprocessing (impute, scale, one-hot) and SMOTE are wrapped inside an
imbalanced-learn Pipeline so they are fit only on training folds during
cross-validation -- no information leaks from validation/test data.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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

RANDOM_STATE = 42
CATEGORICAL = ["activity_new", "channel_sales", "origin_up"]


def build_preprocessor(feature_df):
    """ColumnTransformer: median-impute + scale numerics, one-hot categoricals."""
    categorical = [c for c in CATEGORICAL if c in feature_df.columns]
    numeric = [c for c in feature_df.columns if c not in categorical]

    numeric_tf = SkPipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_tf = SkPipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                                 sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_tf, numeric),
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
