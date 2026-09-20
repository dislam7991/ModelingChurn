"""Data preparation for the PowerCo SME churn model.

This module fixes the core bug in the original pipeline: the price history
(~12 rows per customer) must be *aggregated per customer first*, then joined
one-row-per-customer onto the customer table. The old code left-joined the
history directly, exploding the table ~12x and breaking every downstream step.

Public entry point: ``build_abt`` returns one row per customer.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

# Reference date: the training features describe customers as of Jan 2016.
REF_DATE = pd.Timestamp("2016-01-01")

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
TRAIN = os.path.join(RAW, "training_data")
TEST = os.path.join(RAW, "test_data")

PRICE_COLS = [
    "price_p1_var", "price_p2_var", "price_p3_var",
    "price_p1_fix", "price_p2_fix", "price_p3_fix",
]
DATE_COLS = ["date_activ", "date_end", "date_first_activ",
             "date_modif_prod", "date_renewal"]
# Consumption-type fields where a negative value is impossible (data error).
NONNEG_COLS = ["cons_12m", "cons_gas_12m", "cons_last_month", "imp_cons",
               "forecast_cons", "forecast_cons_12m", "forecast_cons_year"]


def aggregate_price_history(hist: pd.DataFrame) -> pd.DataFrame:
    """Collapse the 2015 monthly price history to one row per customer.

    Produces, per price column: mean, std, min, max, and the change from the
    first to the last available month of 2015 (a classic churn driver). Also
    the mean peak/off-peak energy-price spread.
    """
    hist = hist.copy()
    hist["price_date"] = pd.to_datetime(hist["price_date"], errors="coerce")
    hist = hist.sort_values(["id", "price_date"])

    agg = {c: ["mean", "std", "min", "max"] for c in PRICE_COLS}
    g = hist.groupby("id").agg(agg)
    g.columns = [f"{col}_{stat}" for col, stat in g.columns]

    # Change over the year: last month minus first month, per price column.
    first = hist.groupby("id")[PRICE_COLS].first()
    last = hist.groupby("id")[PRICE_COLS].last()
    change = (last - first).add_suffix("_change")
    g = g.join(change)

    # Peak vs off-peak energy-price spread (P1 minus P2), on the yearly mean.
    g["price_energy_peak_offpeak_spread"] = (
        g["price_p1_var_mean"] - g["price_p2_var_mean"]
    )
    # Max month-to-month volatility of the main energy price.
    vol = (hist.groupby("id")["price_p1_var"]
                .apply(lambda s: s.diff().abs().max()))
    g["price_p1_var_max_monthly_move"] = vol

    return g.reset_index()


def _clean_customer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop a column that is 100% missing in training.
    df = df.drop(columns=["campaign_disc_ele"], errors="ignore")

    # has_gas: t/f -> 1/0
    if df["has_gas"].dtype == object or str(df["has_gas"].dtype) == "str":
        df["has_gas"] = df["has_gas"].map({"t": 1, "f": 0}).astype("float")

    # Negative consumption is impossible -> mark missing (impute later).
    for c in NONNEG_COLS:
        if c in df.columns:
            df.loc[df[c] < 0, c] = np.nan

    # Parse dates.
    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add contract-timing, consumption, margin, and stickiness features."""
    df = df.copy()

    # Contract timing relative to the Jan-2016 reference date.
    df["tenure_years"] = (REF_DATE - df["date_activ"]).dt.days / 365.25
    df["months_to_end"] = (df["date_end"] - REF_DATE).dt.days / 30.44
    df["months_to_renewal"] = (df["date_renewal"] - REF_DATE).dt.days / 30.44
    df["months_since_modif"] = (REF_DATE - df["date_modif_prod"]).dt.days / 30.44
    # Total lifespan: use first-ever activation where present, else current.
    first_activ = df["date_first_activ"].fillna(df["date_activ"])
    df["lifespan_years"] = (REF_DATE - first_activ).dt.days / 365.25

    # Consumption ratios (guard against divide-by-zero).
    df["cons_per_product"] = df["cons_12m"] / df["nb_prod_act"].replace(0, np.nan)
    df["recent_cons_ratio"] = df["cons_last_month"] / (df["cons_12m"] / 12).replace(0, np.nan)
    df["net_margin_per_cons"] = df["net_margin"] / df["cons_12m"].replace(0, np.nan)

    # Stickiness.
    df["is_dual_fuel"] = ((df["has_gas"] == 1) & (df["cons_gas_12m"] > 0)).astype(int)
    df["has_forecast_bill"] = df["forecast_bill_12m"].notna().astype(int)

    # Drop raw date columns now that timing features are derived.
    df = df.drop(columns=DATE_COLS, errors="ignore")
    return df


def reduce_activity_cardinality(df: pd.DataFrame, top: pd.Index | None = None,
                                n_top: int = 10):
    """Cap the 419-category ``activity_new`` to top-N + 'other'/'missing'.

    When ``top`` is None (training), the top-N categories are learned and
    returned so the same mapping can be applied to the test set (no leakage).
    """
    df = df.copy()
    s = df["activity_new"].astype("object")
    if top is None:
        top = s.value_counts().head(n_top).index
    df["activity_new"] = np.where(
        s.isna(), "missing",
        np.where(s.isin(top), s, "other"),
    )
    return df, top


def build_abt(kind: str = "train", activity_top=None):
    """Build the analytics base table: one row per customer.

    kind='train' returns (abt_with_churn, ids, activity_top).
    kind='test'  returns (abt_without_label, ids, activity_top) and requires
    the activity_top learned on training.
    """
    if kind == "train":
        cust = pd.read_csv(os.path.join(TRAIN, "ml_case_training_data.csv"))
        hist = pd.read_csv(os.path.join(TRAIN, "ml_case_training_hist_data.csv"))
        out = pd.read_csv(os.path.join(TRAIN, "ml_case_training_output.csv"))
    elif kind == "test":
        cust = pd.read_csv(os.path.join(TEST, "ml_case_test_data.csv"))
        hist = pd.read_csv(os.path.join(TEST, "ml_case_test_hist_data.csv"))
        out = None
    else:
        raise ValueError("kind must be 'train' or 'test'")

    price = aggregate_price_history(hist)

    # One row per customer: exactly len(cust) rows after this join.
    df = cust.merge(price, on="id", how="left")
    assert len(df) == len(cust), "ABT join changed row count -- not 1 row/id!"

    df = _clean_customer(df)
    df = engineer_features(df)
    df, activity_top = reduce_activity_cardinality(df, top=activity_top)

    ids = df["id"].copy()
    df = df.drop(columns=["id"])

    if out is not None:
        y = cust[["id"]].merge(out, on="id", how="left")["churn"].values
        df["churn"] = y

    return df, ids, activity_top


if __name__ == "__main__":
    tr, ids, top = build_abt("train")
    te, tids, _ = build_abt("test", activity_top=top)
    print("TRAIN ABT:", tr.shape, "unique customers:", ids.nunique())
    print("TEST  ABT:", te.shape, "unique customers:", tids.nunique())
    print("churn rate:", round(tr["churn"].mean(), 4))
    print("n features:", tr.shape[1] - 1)
    print("dtypes:\n", tr.dtypes.value_counts())
    print("any object cols:", list(tr.select_dtypes('object').columns))
