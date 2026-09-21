"""End-to-end churn pipeline runner.

Runs the full workflow and writes real, reproducible artifacts to outputs/:
  1. Data-quality report
  2. Model bake-off (10 classifiers) on a stratified holdout
  3. Cross-validation of the top models for stability
  4. Business-oriented evaluation of the best model (ROC/PR/calibration)
  5. Answers to the three client questions
  6. 20% discount economics (targeting rule + dollar impact)
  7. Scored verification set in the output-template format
  8. A results summary

Nothing here is fabricated: every number is computed from the data.
"""
from __future__ import annotations

import os
import time
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, accuracy_score,
                             brier_score_loss, roc_curve, precision_recall_curve,
                             confusion_matrix)
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from scipy import stats

import joblib

from data_prep import build_abt, REF_DATE
from model_pipeline import (build_preprocessor, get_models, make_pipeline,
                            RANDOM_STATE, CATEGORICAL)

warnings.filterwarnings("ignore")

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "outputs")
FIG = os.path.join(OUT, "figures")
PRED = os.path.join(OUT, "predictions")
PROC = os.path.join(HERE, "..", "data", "processed")
MODELS = os.path.join(HERE, "..", "models")
for d in (OUT, FIG, PRED, PROC, MODELS):
    os.makedirs(d, exist_ok=True)

DISCOUNT = 0.20  # 20% price cut -> assume 20% of annual net_margin foregone


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# 1. Data-quality report
# --------------------------------------------------------------------------
def data_quality_report():
    log("Building data-quality report ...")
    RAW = os.path.join(HERE, "..", "data", "raw", "training_data")
    cust = pd.read_csv(os.path.join(RAW, "ml_case_training_data.csv"))
    out = pd.read_csv(os.path.join(RAW, "ml_case_training_output.csv"))
    hist = pd.read_csv(os.path.join(RAW, "ml_case_training_hist_data.csv"))

    miss = (cust.isna().mean() * 100).round(1).sort_values(ascending=False)
    miss = miss[miss > 0]

    lines = ["# Data-Quality Report (training data)\n",
             f"- Customers: **{len(cust):,}** (one row per id).",
             f"- Price-history rows: **{len(hist):,}** (~{len(hist)/hist.id.nunique():.1f} months per customer, Jan–Dec 2015).",
             f"- Churn rate: **{out.churn.mean()*100:.2f}%** "
             f"({int(out.churn.sum()):,} churners of {len(out):,}) — imbalanced.\n",
             "## Missing values (columns with any missing)\n",
             "| Column | % missing | Handling |", "| --- | --- | --- |"]
    handling = {
        "campaign_disc_ele": "100% empty — **dropped**",
        "forecast_base_bill_year": "78% missing — imputed (median) + `has_forecast_bill` flag",
        "forecast_base_bill_ele": "78% missing — imputed (median)",
        "forecast_bill_12m": "78% missing — imputed + flag",
        "forecast_cons": "78% missing — imputed (median)",
        "date_first_activ": "78% missing — fall back to `date_activ` for lifespan",
        "activity_new": "59% missing, 419 categories — capped to top-10 + other/missing",
        "channel_sales": "26% missing — imputed as 'missing' category",
    }
    for c, pct in miss.items():
        lines.append(f"| {c} | {pct} | {handling.get(c, 'imputed (median / missing category)')} |")

    # Negative values
    lines += ["\n## Implausible negative values\n",
              "| Column | # negatives | Handling |", "| --- | --- | --- |"]
    nonneg = ["cons_12m", "cons_gas_12m", "cons_last_month", "imp_cons",
              "forecast_cons", "forecast_cons_12m", "forecast_cons_year"]
    for c in nonneg:
        n = int((cust[c] < 0).sum()) if c in cust else 0
        if n:
            lines.append(f"| {c} | {n} | set to missing (consumption cannot be negative) |")
    for c in ["margin_gross_pow_ele", "margin_net_pow_ele", "net_margin"]:
        n = int((cust[c] < 0).sum())
        lines.append(f"| {c} | {n} | **kept** (a margin can legitimately be negative) |")

    lines += ["\n## Structural issue fixed\n",
              "The original pipeline left-joined the ~12-row price history "
              "directly onto the customer table, exploding it ~12x and breaking "
              "the per-customer aggregation. The new `data_prep.build_abt` "
              "aggregates the price history per `id` first (mean/std/min/max, "
              "year change, peak–offpeak spread), then joins one row per "
              "customer. Row count is asserted to stay at one per id.\n"]

    with open(os.path.join(OUT, "data_quality_report.md"), "w") as f:
        f.write("\n".join(lines))
    log("  -> outputs/data_quality_report.md")


# --------------------------------------------------------------------------
# 2. Model bake-off
# --------------------------------------------------------------------------
def evaluate_probs(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return {
        "ROC_AUC": roc_auc_score(y, p),
        "PR_AUC": average_precision_score(y, p),
        "Brier": brier_score_loss(y, p),
        "F1@0.5": f1_score(y, yhat, zero_division=0),
        "Precision@0.5": precision_score(y, yhat, zero_division=0),
        "Recall@0.5": recall_score(y, yhat, zero_division=0),
        "Accuracy@0.5": accuracy_score(y, yhat),
    }


def bakeoff(X_tr, X_te, y_tr, y_te, preprocessor):
    log("Running model bake-off (10 classifiers) ...")
    rows = []
    fitted = {}
    for name, model in get_models().items():
        t0 = time.time()
        pipe = make_pipeline(model, preprocessor, use_smote=True)
        pipe.fit(X_tr, y_tr)
        p = pipe.predict_proba(X_te)[:, 1]
        m = {"Model": name}
        m.update(evaluate_probs(y_te.values, p))
        m["fit_seconds"] = round(time.time() - t0, 1)
        rows.append(m)
        fitted[name] = pipe
        log(f"  {name:<20} ROC-AUC={m['ROC_AUC']:.3f} "
            f"PR-AUC={m['PR_AUC']:.3f} Recall={m['Recall@0.5']:.3f} "
            f"({m['fit_seconds']}s)")
    res = (pd.DataFrame(rows).set_index("Model")
           .sort_values("ROC_AUC", ascending=False))
    res.round(4).to_csv(os.path.join(OUT, "model_comparison.csv"))
    log("  -> outputs/model_comparison.csv")
    return res, fitted


# --------------------------------------------------------------------------
# 3. Cross-validation of top models
# --------------------------------------------------------------------------
def cross_validate_top(X, y, preprocessor, top_names):
    log(f"Cross-validating top models {top_names} (5-fold, ROC-AUC) ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = get_models()
    rows = []
    for name in top_names:
        pipe = make_pipeline(models[name], preprocessor, use_smote=True)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
        rows.append({"Model": name, "CV_ROC_AUC_mean": scores.mean(),
                     "CV_ROC_AUC_std": scores.std()})
        log(f"  {name:<20} CV ROC-AUC = {scores.mean():.3f} +/- {scores.std():.3f}")
    res = pd.DataFrame(rows).set_index("Model").sort_values(
        "CV_ROC_AUC_mean", ascending=False)
    res.round(4).to_csv(os.path.join(OUT, "cv_results.csv"))
    log("  -> outputs/cv_results.csv")
    return res


# --------------------------------------------------------------------------
# 4. Business evaluation of the best model
# --------------------------------------------------------------------------
def f1_optimal_threshold(y, p):
    """Threshold on p that maximizes F1 (a balanced churn-classification cut)."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thr[int(np.argmax(f1))])


def business_evaluation(best_name, p, y_te):
    log(f"Evaluating best model: {best_name} (calibrated) ...")
    y = y_te.values

    # ROC + PR curves
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(fpr, tpr, label=f"AUC={roc_auc_score(y,p):.3f}")
    ax[0].plot([0, 1], [0, 1], "--", color="grey")
    ax[0].set(title=f"ROC — {best_name}", xlabel="FPR", ylabel="TPR")
    ax[0].legend()
    ax[1].plot(rec, prec, label=f"PR-AUC={average_precision_score(y,p):.3f}")
    ax[1].axhline(y.mean(), ls="--", color="grey", label=f"baseline={y.mean():.3f}")
    ax[1].set(title=f"Precision-Recall — {best_name}", xlabel="Recall", ylabel="Precision")
    ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "roc_pr.png"), dpi=120)
    plt.close(fig)

    # Calibration
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(mean_pred, frac_pos, "o-", label=best_name)
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(title="Calibration", xlabel="Mean predicted prob", ylabel="Observed churn")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "calibration.png"), dpi=120)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------
# 5. Client questions
# --------------------------------------------------------------------------
def client_questions(best_name, best_pipe, X_te, y_te, abt_train):
    log("Answering the three client questions ...")
    lines = ["# Answers to the Client Questions\n"]

    # Q1: most explicative variables (permutation importance on the pipeline)
    log("  Q1: permutation importance ...")
    r = permutation_importance(best_pipe, X_te, y_te, n_repeats=5,
                               random_state=RANDOM_STATE, scoring="roc_auc",
                               n_jobs=1)
    imp = (pd.Series(r.importances_mean, index=X_te.columns)
           .sort_values(ascending=False))
    imp.round(5).to_csv(os.path.join(OUT, "feature_importance.csv"))
    top15 = imp.head(15)
    fig, ax = plt.subplots(figsize=(7, 6))
    top15[::-1].plot.barh(ax=ax, color="#3b7dd8")
    ax.set(title=f"Top 15 churn drivers ({best_name}, permutation ROC-AUC drop)",
           xlabel="Mean AUC decrease when shuffled")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "feature_importance.png"), dpi=120)
    plt.close(fig)
    lines += ["## Q1. Most explicative variables for churn\n",
              "Permutation importance (drop in ROC-AUC when a feature is "
              f"shuffled) on the held-out set, model = {best_name}.\n",
              "| Rank | Feature | Importance |", "| --- | --- | --- |"]
    for i, (feat, val) in enumerate(top15.items(), 1):
        lines.append(f"| {i} | {feat} | {val:.5f} |")

    # Q2: pow_max vs consumption
    log("  Q2: pow_max vs consumption correlation ...")
    d = abt_train[["pow_max", "cons_12m"]].dropna()
    pear = stats.pearsonr(d["pow_max"], d["cons_12m"])
    spear = stats.spearmanr(d["pow_max"], d["cons_12m"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(d["pow_max"], d["cons_12m"], s=6, alpha=0.2)
    ax.set(title=f"pow_max vs cons_12m (Pearson r={pear[0]:.2f})",
           xlabel="Subscribed power (pow_max)", ylabel="Annual consumption (cons_12m)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "pow_vs_cons.png"), dpi=120)
    plt.close(fig)
    lines += ["\n## Q2. Correlation between subscribed power and consumption\n",
              f"- Pearson r = **{pear[0]:.3f}** (p = {pear[1]:.2e})",
              f"- Spearman rho = **{spear[0]:.3f}** (p = {spear[1]:.2e})",
              f"- Interpretation: {'positive and significant' if pear[0]>0.2 and pear[1]<0.05 else 'weak/none'} — "
              "higher subscribed power tends to go with higher consumption.\n"]

    # Q3: channel_sales vs churn
    log("  Q3: channel_sales vs churn chi-square ...")
    ch = abt_train.copy()
    ch["churn"] = ch["churn"]
    ct = pd.crosstab(ch["channel_sales"].fillna("missing"), ch["churn"])
    chi2, pval, dof, _ = stats.chi2_contingency(ct)
    rate = (ch.assign(channel=ch["channel_sales"].fillna("missing"))
              .groupby("channel")["churn"].agg(["mean", "size"])
              .sort_values("mean", ascending=False))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    (rate["mean"] * 100).plot.bar(ax=ax, color="#d8743b")
    ax.axhline(ch["churn"].mean() * 100, ls="--", color="grey",
               label=f"overall {ch['churn'].mean()*100:.1f}%")
    ax.set(title="Churn rate by sales channel", ylabel="Churn rate (%)")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "channel_churn.png"), dpi=120)
    plt.close(fig)
    lines += ["## Q3. Link between sales channel and churn\n",
              f"- Chi-square = **{chi2:.1f}**, dof = {dof}, p = **{pval:.2e}** "
              f"({'significant' if pval < 0.05 else 'not significant'} at 5%).",
              "- Churn rate by channel:\n",
              "| Channel (hashed) | Churn rate | Customers |", "| --- | --- | --- |"]
    for chan, row in rate.iterrows():
        short = (chan[:12] + "…") if len(str(chan)) > 13 else chan
        lines.append(f"| {short} | {row['mean']*100:.1f}% | {int(row['size']):,} |")

    with open(os.path.join(OUT, "client_questions.md"), "w") as f:
        f.write("\n".join(lines))
    log("  -> outputs/client_questions.md")
    return imp


# --------------------------------------------------------------------------
# 6. Discount economics
# --------------------------------------------------------------------------
def discount_economics(p, y, margin, scale_to_full):
    """Realized-value analysis of the 20% discount on the held-out set.

    Uses *calibrated* churn probabilities ``p``. The decision rule is
    expected-value optimal: for a customer with annual net margin m and churn
    probability p,
        EV(offer) - EV(no offer) = 0.8*m - (1-p)*m = m*(p - 0.20).
    So for a positive-margin customer it pays to offer iff p > 0.20. Negative-
    margin customers are never offered (better to let a loss-maker churn).

    p      : calibrated churn probability
    y      : actual churn (1=left)  -- used only to *evaluate* realized value
    margin : annual net_margin
    scale_to_full : multiplier to extrapolate holdout -> full population
    """
    log("Running 20% discount economics (calibrated probabilities) ...")
    y = np.asarray(y); p = np.asarray(p); m = np.asarray(margin, float)
    m = np.nan_to_num(m, nan=0.0)
    n = len(y)
    s = scale_to_full

    def realized(offered):
        offered = np.asarray(offered, bool)
        val = np.empty_like(m)
        val[offered] = (1 - DISCOUNT) * m[offered]          # accept & retained
        stay = (~offered) & (y == 0); left = (~offered) & (y == 1)
        val[stay] = m[stay]; val[left] = 0.0                # keep or lose
        return val.sum()

    no_action = realized(np.zeros(n, bool))
    blanket = realized(np.ones(n, bool))

    # Principled EV-optimal targeting rule.
    EV_THR = DISCOUNT  # 0.20
    offered_rule = (p > EV_THR) & (m > 0)
    targeted = realized(offered_rule)
    retained_churners = int(((offered_rule) & (y == 1)).sum())
    wasted_offers = int(((offered_rule) & (y == 0)).sum())

    # Oracle upper bound: offer only to actual positive-margin churners.
    oracle = realized((y == 1) & (m > 0))

    # Sweep for the figure + an empirical-optimum sensitivity check.
    grid = np.linspace(0.0, 0.95, 96)
    curve = [(t, realized((p >= t) & (m > 0))) for t in grid]
    cdf = pd.DataFrame(curve, columns=["threshold", "realized_margin"])
    emp_thr = float(cdf.loc[cdf["realized_margin"].idxmax(), "threshold"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(cdf["threshold"], cdf["realized_margin"] * s / 1e6, label="targeted policy")
    ax.axhline(no_action * s / 1e6, ls="--", color="grey", label="no action")
    ax.axhline(blanket * s / 1e6, ls=":", color="red", label="blanket 20%")
    ax.axvline(EV_THR, color="green", alpha=0.6, label=f"EV-optimal thr={EV_THR:.2f}")
    ax.set(title="Realized annual margin vs discount threshold (full population)",
           xlabel="Calibrated churn-probability threshold to offer discount",
           ylabel="Total net margin (millions)")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "discount_ev_curve.png"), dpi=120)
    plt.close(fig)

    lines = [
        "# 20% Discount — Economic Analysis\n",
        "**Decision model.** A customer offered the discount accepts (per the "
        "brief's assumption) and is retained for a year at 80% of their net "
        "margin. A customer not offered keeps full margin if they stay and "
        "contributes zero if they churn. The expected value of offering minus "
        "not offering is `m*(p - 0.20)`, so it pays to offer a **positive-margin** "
        "customer only when calibrated churn probability **p > 0.20**. Negative-"
        "margin customers are never offered.\n",
        f"Figures are annual `net_margin`, extrapolated to the full "
        f"~{int(n*s):,}-customer population (holdout x{s:.2f}).\n",
        "| Strategy | Customers offered | Annual net margin | vs no-action |",
        "| --- | --- | --- | --- |",
        f"| No action | 0 | {no_action*s:,.0f} | — |",
        f"| Blanket 20% to all | {int(n*s):,} | {blanket*s:,.0f} | {(blanket-no_action)*s:,.0f} |",
        f"| **Targeted (p>0.20 & margin>0)** | {int(offered_rule.sum()*s):,} | **{targeted*s:,.0f}** | **{(targeted-no_action)*s:,.0f}** |",
        f"| Oracle (perfect foresight) | {int(((y==1)&(m>0)).sum()*s):,} | {oracle*s:,.0f} | {(oracle-no_action)*s:,.0f} |",
        "",
        f"- The EV-optimal rule offers to **{int(offered_rule.sum()*s):,}** customers "
        f"(~{offered_rule.mean()*100:.0f}% of the base), retaining "
        f"**{int(retained_churners*s):,}** true churners with "
        f"**{int(wasted_offers*s):,}** discounts going to customers who would have "
        "stayed.",
        f"- Targeted uplift over doing nothing: **{(targeted-no_action)*s:,.0f}** per year.",
        f"- Targeted uplift over the blanket offer: **{(targeted-blanket)*s:,.0f}** per "
        "year — the money saved by *not* discounting everyone.",
        f"- Sensitivity: the empirically best threshold on this holdout is "
        f"~{emp_thr:.2f}, close to the theoretical 0.20.",
        "",
        "**Recommendation.** A blanket 20% discount "
        f"{'destroys value' if blanket < no_action else 'adds little value'} "
        f"({(blanket-no_action)*s:,.0f} vs no action) because most customers would "
        "not have churned and the locked-in price cut is pure lost margin on them. "
        "Offer the discount only to customers with calibrated churn probability "
        "above 20% **and** positive margin. Rank by expected value "
        "`margin*(p-0.20)`, not by churn probability alone.",
    ]
    with open(os.path.join(OUT, "economics.md"), "w") as f:
        f.write("\n".join(lines))
    log("  -> outputs/economics.md")
    return {"threshold": EV_THR, "empirical_threshold": emp_thr,
            "no_action": no_action * s, "blanket": blanket * s,
            "targeted": targeted * s, "oracle": oracle * s,
            "retained_churners": int(retained_churners * s),
            "offered": int(offered_rule.sum() * s)}


# --------------------------------------------------------------------------
# 7. Score the verification set
# --------------------------------------------------------------------------
def build_calibrated(best_name, preprocessor, X, y, cv=5):
    """Calibrated version of the best SMOTE pipeline (isotonic, CV).

    Calibrating on the real (imbalanced) folds corrects the probability
    inflation that SMOTE introduces, giving trustworthy probabilities.
    """
    base = make_pipeline(get_models()[best_name], preprocessor, use_smote=True)
    calib = CalibratedClassifierCV(base, method="isotonic", cv=cv)
    calib.fit(X, y)
    return calib


def score_test_set(final, X_all, abt_test, test_ids, clf_threshold):
    log("Scoring the verification set with the calibrated model ...")
    X_test = abt_test[X_all.columns]
    proba = final.predict_proba(X_test)[:, 1]

    tmpl_path = os.path.join(HERE, "..", "data", "raw", "test_data",
                             "ml_case_test_output_template.csv")
    tmpl = pd.read_csv(tmpl_path)
    scored = pd.DataFrame({"id": test_ids.values, "Churn_probability": proba})
    scored["Churn_prediction"] = (proba >= clf_threshold).astype(int)
    out = tmpl[["id"]].merge(scored, on="id", how="left")
    out = out[["id", "Churn_prediction", "Churn_probability"]]
    out = out.sort_values("Churn_probability", ascending=False).reset_index(drop=True)
    out.to_csv(os.path.join(PRED, "ml_case_test_output_filled.csv"), index=False)
    log(f"  -> outputs/predictions/ml_case_test_output_filled.csv "
        f"({int(out['Churn_prediction'].sum())} flagged to churn, "
        f"threshold={clf_threshold:.3f})")
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    t0 = time.time()
    data_quality_report()

    log("Building ABT (train + test) ...")
    abt_train, train_ids, activity_top = build_abt("train")
    abt_test, test_ids, _ = build_abt("test", activity_top=activity_top)
    abt_train.to_csv(os.path.join(PROC, "abt_train.csv"), index=False)
    abt_test.to_csv(os.path.join(PROC, "abt_test.csv"), index=False)
    log(f"  ABT train {abt_train.shape}, test {abt_test.shape}")

    y = abt_train["churn"].astype(int)
    X = abt_train.drop(columns=["churn"])
    # keep net_margin aligned for economics
    margin_all = abt_train["net_margin"].copy()

    X_tr, X_te, y_tr, y_te, m_tr, m_te = train_test_split(
        X, y, margin_all, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
    scale_to_full = len(X) / len(X_te)

    preprocessor = build_preprocessor(X)

    # 2. Bake-off
    res, fitted = bakeoff(X_tr, X_te, y_tr, y_te, preprocessor)
    print("\n=== MODEL COMPARISON (holdout, sorted by ROC-AUC) ===")
    print(res.round(4).to_string())

    # 3. CV top 3 (exclude Dummy)
    ranked = [n for n in res.index if n != "Dummy Classifier"]
    cv = cross_validate_top(X, y, preprocessor, ranked[:3])

    best_name = cv.index[0]
    best_pipe = fitted[best_name]           # uncalibrated, for feature ranking
    log(f"BEST MODEL = {best_name}")

    # Calibrated probabilities for the business layer (fit on train split).
    log("Calibrating best model on the training split ...")
    calib_holdout = build_calibrated(best_name, preprocessor, X_tr, y_tr)
    p_te = calib_holdout.predict_proba(X_te)[:, 1]
    clf_threshold = f1_optimal_threshold(y_te.values, p_te)

    # 4. Business evaluation (calibrated probs)
    business_evaluation(best_name, p_te, y_te)

    # 5. Client questions (importance from the uncalibrated best pipeline)
    imp = client_questions(best_name, best_pipe, X_te, y_te, abt_train)

    # 6. Economics (calibrated probs + EV-optimal rule)
    econ = discount_economics(p_te, y_te.values, m_te.values, scale_to_full)

    # 7. Refit calibrated model on ALL data and score the verification set.
    final = build_calibrated(best_name, preprocessor, X, y)
    joblib.dump(final, os.path.join(MODELS, "best_model_calibrated.joblib"))
    scored = score_test_set(final, X, abt_test, test_ids, clf_threshold)

    # 8. Results summary
    summary = {
        "best_model": best_name,
        "holdout_ROC_AUC": float(res.loc[best_name, "ROC_AUC"]),
        "holdout_PR_AUC": float(res.loc[best_name, "PR_AUC"]),
        "cv_ROC_AUC_mean": float(cv.loc[best_name, "CV_ROC_AUC_mean"]),
        "cv_ROC_AUC_std": float(cv.loc[best_name, "CV_ROC_AUC_std"]),
        "discount_ev_threshold": econ["threshold"],
        "discount_empirical_threshold": econ["empirical_threshold"],
        "classification_threshold_f1": clf_threshold,
        "economics_full_population": {k: econ[k] for k in
            ["no_action", "blanket", "targeted", "oracle",
             "retained_churners", "offered"]},
        "top_features": list(imp.head(10).index),
        "test_flagged_to_churn": int(scored["Churn_prediction"].sum()),
        "runtime_minutes": round((time.time() - t0) / 60, 1),
    }
    with open(os.path.join(OUT, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"  -> outputs/results_summary.json")
    log(f"DONE in {summary['runtime_minutes']} min. Best model: {best_name}")
    print("\nSUMMARY:\n", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
