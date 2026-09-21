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

# Heavily right-skewed columns (skew 5-9 on the raw training data). These are
# log-transformed inside the *modelling* pipeline (model_pipeline.signed_log1p),
# never here: the ABT deliberately keeps real units so the discount economics
# and the client-question report stay in currency/kWh terms.
# Selection rule: raw skew > +2 *and* the log measurably reduces |skew|. Every
# column below was checked against both; see the table in the data-quality
# report, which is regenerated from the data on each run.
SKEWED_COLS = [
    # raw, non-negative (raw skew 3.5 - 12)
    "cons_12m", "cons_gas_12m", "cons_last_month", "imp_cons",
    "forecast_cons", "forecast_cons_12m", "forecast_cons_year",
    "forecast_meter_rent_12m", "forecast_base_bill_ele",
    "forecast_base_bill_year", "forecast_bill_12m",
    # net margin: raw skew 21, and ~97 legitimately negative values, so the
    # transform has to be a *signed* log rather than a plain log1p.
    "net_margin",
    # engineered ratios derived from the columns above
    "cons_per_product", "recent_cons_ratio",
]

# Deliberately NOT log-transformed, each for a measured reason:
#   margin_gross_pow_ele  raw skew 1.05 -- not skewed; the log would push it to
#                         -2.16 because ~1,315 values are <= 0.
#   margin_net_pow_ele    raw skew -3.13 -- *left*-skewed, so a right-skew fix
#                         does not apply (the log only moves it to -2.37).
#   net_margin_per_cons   raw skew 110, but the values sit around 0.01, where
#                         log1p(x) ~= x and therefore does almost nothing. Its
#                         tail comes from a handful of customers with an
#                         implausibly small cons_12m denominator (3-47 kWh for a
#                         whole year), which is a separate data-quality question
#                         from the high-end outliers handled here.

# Isolated-extreme rule: a top value is treated as an error when it exceeds the
# next distinct value below it by more than this factor (see fit_extreme_caps).
EXTREME_GAP_FACTOR = 2.0


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


def fit_extreme_caps(df: pd.DataFrame, cols=None,
                     gap_factor: float = EXTREME_GAP_FACTOR) -> dict:
    """Learn a cap for values sitting in an isolated gap above the rest.

    Most very large consumption values are *shared* by many customers: e.g.
    ``cons_12m == 6_286_272`` appears in 11 training rows and is also the maximum
    of the test set, so it is a real (if repeated) record, not a typo. Capping at
    a fixed percentile would flatten dozens of genuinely distinct customers onto
    one value.

    One row is different. Customer ``2c2abbe8...`` reports cons_12m = 16,097,108
    and cons_last_month = 4,538,720 — 2.6x and 5.9x the next distinct value, and
    shared with no one else. That is the profile of a data error.

    This walks each column's distinct values downwards while every value is more
    than ``gap_factor`` times the one below it, and caps those isolated values to
    the first value that is *not* isolated. Fitted on training data only and
    passed to the test build, so nothing crosses the split.
    """
    cols = NONNEG_COLS if cols is None else cols
    caps: dict[str, float] = {}
    for c in cols:
        if c not in df.columns:
            continue
        vals = np.sort(df[c].dropna().unique())[::-1]
        if len(vals) < 2:
            continue
        i = 0
        while (i + 1 < len(vals) and vals[i + 1] > 0
               and vals[i] > gap_factor * vals[i + 1]):
            i += 1
        if i:
            caps[c] = float(vals[i])
    return caps


def apply_extreme_caps(df: pd.DataFrame, caps: dict) -> pd.DataFrame:
    """Clip the isolated extremes identified by :func:`fit_extreme_caps`."""
    df = df.copy()
    for c, cap in caps.items():
        if c in df.columns:
            df[c] = df[c].clip(upper=cap)
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


def build_abt(kind: str = "train", activity_top=None, caps=None):
    """Build the analytics base table: one row per customer.

    kind='train' returns (abt_with_churn, ids, activity_top, caps).
    kind='test'  returns (abt_without_label, ids, activity_top, caps) and
    requires the activity_top *and* caps learned on training, so that no
    test-set information influences the cleaning rules.
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
    # Cap isolated extreme consumption values before any ratio is derived from
    # them, so a single bad row cannot distort the engineered features.
    if caps is None:
        caps = fit_extreme_caps(df)
    df = apply_extreme_caps(df, caps)
    df = engineer_features(df)
    df, activity_top = reduce_activity_cardinality(df, top=activity_top)

    ids = df["id"].copy()
    df = df.drop(columns=["id"])

    if out is not None:
        y = cust[["id"]].merge(out, on="id", how="left")["churn"].values
        df["churn"] = y

    return df, ids, activity_top, caps


if __name__ == "__main__":
    tr, ids, top, caps = build_abt("train")
    te, tids, _, _ = build_abt("test", activity_top=top, caps=caps)
    print("TRAIN ABT:", tr.shape, "unique customers:", ids.nunique())
    print("TEST  ABT:", te.shape, "unique customers:", tids.nunique())
    print("churn rate:", round(tr["churn"].mean(), 4))
    print("n features:", tr.shape[1] - 1)
    print("dtypes:\n", tr.dtypes.value_counts())
    print("any object cols:", list(tr.select_dtypes('object').columns))
    print("extreme-value caps learned on training:", caps)
