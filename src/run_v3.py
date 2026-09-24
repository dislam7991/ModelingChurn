"""v3: do the Tier-1 improvements beat the tuned XGBoost -- honestly?

Candidates (all using the v2-tuned hyperparameters, no re-search):
  * encodings: v2 one-hot | out-of-fold target encoding of the full
    `activity_new` | native categorical splits
  * models: XGBoost, LightGBM (+ Random Forest v2 as a reference)
  * blends: rank-average of XGBoost + LightGBM for each encoding

PRE-REGISTERED DECISION RULE (fixed before any result was seen):
  The deployed model changes only if the candidate with the highest mean
  PR-AUC in repeated 5x5 stratified CV (train split)
    (1) beats the deployed XGB v2 with Holm-adjusted p < 0.05 on the
        Nadeau-Bengio corrected paired t-test, and
    (2) is not worse than XGB v2 on the untouched 25% holdout.
  Otherwise XGB v2 stays deployed.

Calibration (isotonic vs sigmoid) is chosen by cross-fitted *profit* on the
train out-of-fold predictions -- the holdout stays untouched until the end.
Isotonic's step function ties customers (v2 holdout: 4,024 customers -> 48
distinct scores), which costs ranking; sigmoid is strictly monotone.

Usage: python run_v3.py [--repeats 5] [--no-rf] [--no-rescore]
"""
from __future__ import annotations
import os, sys, json, time, copy, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from joblib import Parallel, delayed
from scipy import stats
from scipy.stats import rankdata
from sklearn.model_selection import (train_test_split, RepeatedStratifiedKFold,
                                     StratifiedKFold)
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             brier_score_loss, log_loss, precision_score,
                             recall_score)
from sklearn.inspection import permutation_importance

from data_prep import build_abt
from model_pipeline import RANDOM_STATE
from tuning import fit_calibrator
from economics import (best_profit_threshold, policy_summary, realized_margin,
                       acceptance_sensitivity)
from v3_models import add_raw_activity, make_pipeline, CalibratedEnsemble

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "outputs"); FIG = os.path.join(OUT, "figures")
PRED = os.path.join(OUT, "predictions"); MODELS = os.path.join(HERE, "..", "models")
for d in (OUT, FIG, PRED, MODELS):
    os.makedirs(d, exist_ok=True)

BASE = "XGB v2"
CONFIGS = {
    "XGB v2": ("XGBoost", "v2"),
    "LGBM v2": ("LightGBM", "v2"),
    "RF v2": ("Random Forest", "v2"),
    "XGB +TE": ("XGBoost", "te"),
    "LGBM +TE": ("LightGBM", "te"),
    "XGB native": ("XGBoost", "native"),
    "LGBM native": ("LightGBM", "native"),
}
BLENDS = {
    "Blend v2": ("XGB v2", "LGBM v2"),
    "Blend +TE": ("XGB +TE", "LGBM +TE"),
    "Blend native": ("XGB native", "LGBM native"),
}
N_SPLITS = 5
ALPHA = 0.05

# Reference palette (validated: light surface, CVD dE 24.7, contrast >= 3:1).
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Repeated CV
# --------------------------------------------------------------------------
def _run_split(split_id, tr_idx, va_idx, X, y, configs):
    Xa, Xb = X.iloc[tr_idx], X.iloc[va_idx]
    ya, yb = y.iloc[tr_idx], y.iloc[va_idx]
    preds = {}
    for name, (model, kind) in configs.items():
        preds[name] = make_pipeline(model, kind, Xa).fit(Xa, ya).predict_proba(Xb)[:, 1]
    for bname, (a, b) in BLENDS.items():
        if a in preds and b in preds:
            preds[bname] = (rankdata(preds[a]) + rankdata(preds[b])) / 2
    return [dict(split=split_id, config=n,
                 pr_auc=average_precision_score(yb, p), roc_auc=roc_auc_score(yb, p))
            for n, p in preds.items()]


def _fit_predict(model, kind, Xa, ya, Xb):
    return make_pipeline(model, kind, Xa).fit(Xa, ya).predict_proba(Xb)[:, 1]


def corrected_ttest(d, test_train_ratio):
    """Nadeau & Bengio (2003) corrected resampled t-test for paired CV scores.

    Folds share training data, so the naive t-test is badly over-confident; the
    variance is inflated by (1/J + n_test/n_train).
    """
    d = np.asarray(d, float); J = len(d)
    m, v = d.mean(), d.var(ddof=1)
    se = np.sqrt((1 / J + test_train_ratio) * v)
    t = m / se if se > 0 else 0.0
    p = 2 * stats.t.sf(abs(t), J - 1)
    half = stats.t.ppf(0.975, J - 1) * se
    return m, t, p, m - half, m + half


def holm(pvals: pd.Series) -> pd.Series:
    order = pvals.sort_values()
    k = len(order); adj, running = {}, 0.0
    for i, (name, p) in enumerate(order.items()):
        running = max(running, min(1.0, (k - i) * p))
        adj[name] = running
    return pd.Series(adj).reindex(pvals.index)


# --------------------------------------------------------------------------
# Calibration choice (train data only)
# --------------------------------------------------------------------------
def choose_calibration(raw_oofs, y, margin):
    y = np.asarray(y); m = np.asarray(margin, float)
    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    no_action = realized_margin(np.zeros(len(y), bool), y, m)
    rows = []
    for method in ("isotonic", "sigmoid"):
        p_cf = np.zeros(len(y)); profit = 0.0
        for fit_i, held_i in skf.split(np.zeros(len(y)), y):
            cals = [fit_calibrator(r[fit_i], y[fit_i], method) for r in raw_oofs]
            p_fit = np.mean([c.predict(r[fit_i]) for c, r in zip(cals, raw_oofs)], axis=0)
            p_held = np.mean([c.predict(r[held_i]) for c, r in zip(cals, raw_oofs)], axis=0)
            thr, _ = best_profit_threshold(p_fit, y[fit_i], m[fit_i])
            offered = (p_held >= thr) & (np.nan_to_num(m[held_i]) > 0)
            profit += realized_margin(offered, y[held_i], m[held_i])
            p_cf[held_i] = p_held
        rows.append(dict(method=method,
                         crossfit_uplift_vs_no_action=profit - no_action,
                         brier=brier_score_loss(y, p_cf),
                         log_loss=log_loss(y, np.clip(p_cf, 1e-6, 1 - 1e-6)),
                         pr_auc=average_precision_score(y, p_cf),
                         distinct_scores=int(len(np.unique(np.round(p_cf, 12))))))
    df = pd.DataFrame(rows).set_index("method")
    best = df.sort_values(["crossfit_uplift_vs_no_action", "brier"],
                          ascending=[False, True]).index[0]
    return best, df


def holdout_eval(label, ens, X_te, y_te, m_te, scale):
    p = ens.predict_proba(X_te)[:, 1]
    thr, _ = best_profit_threshold(ens.oof_, ens._y_fit, ens._m_fit)
    pol = policy_summary(p, y_te.values, m_te.values, thr)
    yhat = ((p >= thr) & (np.nan_to_num(m_te.values) > 0)).astype(int)
    return p, dict(model=label, calibration=ens.method,
                   PR_AUC=average_precision_score(y_te, p), ROC_AUC=roc_auc_score(y_te, p),
                   Brier=brier_score_loss(y_te, p), distinct_scores=int(len(np.unique(p))),
                   profit_threshold=thr,
                   precision_offered=precision_score(y_te, yhat, zero_division=0),
                   recall_offered=recall_score(y_te, yhat, zero_division=0),
                   offered_full_pop=int(pol["offered"] * scale),
                   retained_churners_full_pop=int(pol["retained_churners"] * scale),
                   uplift_vs_no_action_full_pop=pol["uplift_vs_no_action"] * scale,
                   uplift_vs_blanket_full_pop=pol["uplift_vs_blanket"] * scale)


def fit_ensemble(config, method, X, y, margin):
    names = BLENDS[config] if config in BLENDS else (config,)
    members = [make_pipeline(*CONFIGS[n], X) for n in names]
    ens = CalibratedEnsemble(members, method=method).fit(X, y)
    ens._y_fit, ens._m_fit = np.asarray(y), np.asarray(margin, float)  # for thresholding
    return ens


# --------------------------------------------------------------------------
# Figures (reference palette; single series -> no legend box)
# --------------------------------------------------------------------------
def _style(ax, grid_axis="x"):
    ax.figure.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.title.set_color(INK); ax.xaxis.label.set_color(INK2); ax.yaxis.label.set_color(INK2)


def fig_cv_delta(cvt):
    d = cvt.drop(index=BASE).sort_values("delta_vs_base")
    fig, ax = plt.subplots(figsize=(8, 4.6)); _style(ax)
    yy = np.arange(len(d))
    ax.errorbar(d["delta_vs_base"], yy,
                xerr=[d["delta_vs_base"] - d["ci_low"], d["ci_high"] - d["delta_vs_base"]],
                fmt="o", color=BLUE, ms=6, lw=2, capsize=0)
    ax.axvline(0, color=INK2, ls="--", lw=1)
    ax.set_yticks(yy); ax.set_yticklabels(d.index, color=INK)
    for yi, (n, r) in zip(yy, d.iterrows()):
        ax.annotate(f"{r['delta_vs_base']:+.4f}  (Holm p={r['p_holm']:.2f})",
                    (r["ci_high"], yi), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK2)
    ax.set(title=f"PR-AUC change vs deployed XGB v2 — repeated 5x5 CV, corrected 95% CI",
           xlabel="Δ PR-AUC (right of the dashed line = better than XGB v2)")
    lo, hi = d["ci_low"].min(), d["ci_high"].max()
    ax.set_xlim(lo - 0.005, hi + (hi - lo) * 0.55)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "v3_cv_delta.png"), dpi=120)
    plt.close(fig)


def fig_acceptance(curve):
    fig, ax = plt.subplots(figsize=(7.5, 4.5)); _style(ax, grid_axis="y")
    series = [("symmetric", "Churners and stayers accept at the same rate", BLUE),
              ("stayers_all", "Stayers always accept (worst case)", ORANGE)]
    for (key, label, col), dy in zip(series, (10, -14)):
        c = curve[curve["family"] == key].set_index("a_churn")
        ax.plot(c.index, c["uplift_vs_no_action"] / 1e3, color=col, lw=2,
                marker="o", ms=5, label=label)
        # Direct label where the two lines are well apart (they meet at 1.0).
        v = c.loc[0.5, "uplift_vs_no_action"] / 1e3
        ax.annotate(f"{v:,.0f}k at 50%", (0.5, v), xytext=(0, dy), textcoords="offset points",
                    ha="center", fontsize=8, color=INK)
    end = curve[curve["family"] == "symmetric"].set_index("a_churn").loc[1.0, "uplift_vs_no_action"] / 1e3
    ax.annotate(f"{end:,.0f}k if all accept\n(the brief's assumption)", xy=(1.0, end),
                xytext=(0.58, end * 0.89), textcoords="data", ha="left", va="center",
                fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.axhline(0, color=INK2, ls="--", lw=1)
    ax.set(title="Targeted-discount uplift vs acceptance rate (deployed model, full population)",
           xlabel="Share of would-be churners who accept the offer",
           ylabel="Uplift vs no action (thousands / yr)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "v3_acceptance.png"), dpi=120)
    plt.close(fig)


def fig_contract(ct, base_rate):
    fig, ax = plt.subplots(figsize=(7, 4.2)); _style(ax, grid_axis="y")
    xs = np.arange(len(ct))
    ax.bar(xs, ct["churn_rate"] * 100, width=0.72, color=BLUE,
           edgecolor=SURFACE, linewidth=2)
    for x, (_, r) in zip(xs, ct.iterrows()):
        # Inside the bar, below the reference line -> no collisions with it.
        ax.annotate(f"{r['churn_rate']*100:.1f}%\nn={int(r['customers']):,}",
                    (x, r["churn_rate"] * 100), xytext=(0, -26), textcoords="offset points",
                    ha="center", va="top", fontsize=8, color="#ffffff")
    ax.axhline(base_rate * 100, color=INK2, ls="--", lw=1)
    ax.set_xlim(-0.6, len(ct) - 0.4 + 0.95)
    ax.annotate(f"overall\n{base_rate*100:.1f}%", (len(ct) - 0.4 + 0.08, base_rate * 100),
                ha="left", va="center", fontsize=8, color=INK2)
    ax.set_xticks(xs); ax.set_xticklabels(ct.index, color=INK)
    ax.set_ylim(0, max(ct["churn_rate"].max() * 100 * 1.25, base_rate * 100 * 1.35))
    ax.set(title="Churn rate by months until contract end (training data)",
           xlabel="Months until contract end (from 1 Jan 2016)", ylabel="Churn rate (%)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "v3_contract_timing.png"), dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Evidence for the data roadmap
# --------------------------------------------------------------------------
def contract_timing(abt):
    rows = {}
    for col, bins, labels in [
        ("months_to_end", [-np.inf, 3, 6, 9, 12, np.inf], ["0-3", "3-6", "6-9", "9-12", "12+"]),
        ("months_to_renewal", [-np.inf, -6, -3, 0, 3, np.inf],
         ["6+ ago", "3-6 ago", "0-3 ago", "0-3 ahead", "3+ ahead"]),
    ]:
        b = pd.cut(abt[col], bins, labels=labels)
        t = abt.groupby(b, observed=True)["churn"].agg(churn_rate="mean", customers="size")
        chi2, p, dof, _ = stats.chi2_contingency(pd.crosstab(b, abt["churn"]))
        t["lift_vs_overall"] = t["churn_rate"] / abt["churn"].mean()
        t["chi2_p_value"] = p
        rows[col] = t
    return rows


def data_audit():
    raw = os.path.join(HERE, "..", "data", "raw", "training_data")
    frames = [pd.read_csv(os.path.join(raw, f)) for f in
              ("ml_case_training_data.csv", "ml_case_training_hist_data.csv",
               "ml_case_training_output.csv")]
    nonnull = {c: f[c].notna().mean() for f in frames for c in f.columns}
    cols = set(nonnull)
    groups = {
        "Churn reason (switch vs closure/move)": ["reason", "move", "relocat", "voluntary", "cause"],
        "Payment behaviour": ["payment", "late", "arrear", "delinq", "overdue", "debit"],
        "Customer contact / complaints": ["contact", "call", "complaint", "ticket", "login", "portal", "email"],
        "Smart-meter interval usage": ["interval", "hourly", "daily", "load_profile", "meter_read"],
        "Competitor price benchmark": ["competitor", "market_price", "benchmark", "quote"],
        "Campaign / offer response": ["campaign", "offer", "response"],
    }
    rows = []
    for g, keys in groups.items():
        found = sorted(c for c in cols if any(k in c.lower() for k in keys))
        usable = [c for c in found if nonnull[c] > 0]
        rows.append(dict(
            signal=g,
            fields_found=(", ".join(f"{c} ({nonnull[c]:.0%} populated)" for c in found)
                          if found else "none"),
            usable=bool(usable)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--no-rf", action="store_true")
    ap.add_argument("--no-rescore", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    configs = {k: v for k, v in CONFIGS.items() if not (args.no_rf and k == "RF v2")}

    log("Building ABT ...")
    abt_train, train_ids, top = build_abt("train")
    abt_test, test_ids, _ = build_abt("test", activity_top=top)
    y = abt_train["churn"].astype(int)
    X = add_raw_activity(abt_train.drop(columns=["churn"]), train_ids, "train")
    X_test = add_raw_activity(abt_test[abt_train.drop(columns=["churn"]).columns], test_ids, "test")
    margin = abt_train["net_margin"]
    X_tr, X_te, y_tr, y_te, m_tr, m_te = train_test_split(
        X, y, margin, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
    scale = len(X) / len(X_te)

    # ---- 1. repeated CV ------------------------------------------------------
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=args.repeats,
                                   random_state=RANDOM_STATE)
    splits = list(rskf.split(X_tr, y_tr))
    log(f"Repeated CV: {len(configs)} configs + {len(BLENDS)} blends x {len(splits)} splits ...")
    res = Parallel(n_jobs=-1)(delayed(_run_split)(i, tr, va, X_tr, y_tr, configs)
                              for i, (tr, va) in enumerate(splits))
    folds = pd.DataFrame([r for rr in res for r in rr])
    folds.to_csv(os.path.join(OUT, "v3_cv_folds.csv"), index=False)
    pr = folds.pivot(index="split", columns="config", values="pr_auc")
    roc = folds.pivot(index="split", columns="config", values="roc_auc")

    ratio = 1 / (N_SPLITS - 1)  # n_test / n_train within each split
    rows = []
    for c in pr.columns:
        row = dict(config=c, cv_pr_mean=pr[c].mean(), cv_pr_std=pr[c].std(),
                   cv_roc_mean=roc[c].mean())
        if c == BASE:
            row.update(delta_vs_base=0.0, t=np.nan, p_raw=np.nan, ci_low=0.0, ci_high=0.0)
        else:
            m_, t_, p_, lo, hi = corrected_ttest(pr[c] - pr[BASE], ratio)
            row.update(delta_vs_base=m_, t=t_, p_raw=p_, ci_low=lo, ci_high=hi)
        rows.append(row)
    cvt = pd.DataFrame(rows).set_index("config")
    cvt["p_holm"] = holm(cvt["p_raw"].dropna()).reindex(cvt.index)
    log(f"  CV done ({(time.time()-t0)/60:.1f} min)")

    # ---- 2. holdout (raw ranking, same metric as CV) ------------------------
    log("Holdout ranking check for every config ...")
    hold_raw = dict(zip(configs, Parallel(n_jobs=-1)(
        delayed(_fit_predict)(mdl, kind, X_tr, y_tr, X_te) for mdl, kind in configs.values())))
    for bname, (a, b) in BLENDS.items():
        if a in hold_raw and b in hold_raw:
            hold_raw[bname] = (rankdata(hold_raw[a]) + rankdata(hold_raw[b])) / 2
    cvt["holdout_pr_auc"] = pd.Series({k: average_precision_score(y_te, v) for k, v in hold_raw.items()})
    cvt["holdout_roc_auc"] = pd.Series({k: roc_auc_score(y_te, v) for k, v in hold_raw.items()})
    cvt = cvt.sort_values("cv_pr_mean", ascending=False)
    cvt.round(5).to_csv(os.path.join(OUT, "v3_cv_results.csv"))
    print("\n=== v3 REPEATED-CV RESULTS (train split) + holdout ===")
    print(cvt[["cv_pr_mean", "cv_pr_std", "delta_vs_base", "ci_low", "ci_high",
               "p_raw", "p_holm", "cv_roc_mean", "holdout_pr_auc"]].round(4).to_string())

    # ---- 3. pre-registered decision ----------------------------------------
    cand = cvt.index[0]
    if cand == BASE:
        deploy, reason = BASE, "XGB v2 already has the highest mean CV PR-AUC."
    else:
        r = cvt.loc[cand]
        sig = (r["delta_vs_base"] > 0) and (r["p_holm"] < ALPHA)
        hold_ok = r["holdout_pr_auc"] >= cvt.loc[BASE, "holdout_pr_auc"]
        if sig and hold_ok:
            deploy = cand
            reason = (f"{cand} beats XGB v2 by {r['delta_vs_base']:+.4f} PR-AUC "
                      f"(Holm p={r['p_holm']:.3f}) and holds up on the holdout.")
        else:
            deploy = BASE
            why = []
            if not sig:
                why.append(f"its CV gain of {r['delta_vs_base']:+.4f} is not significant "
                           f"(Holm p={r['p_holm']:.2f}, 95% CI {r['ci_low']:+.4f} to {r['ci_high']:+.4f})")
            if not hold_ok:
                why.append("it is worse than XGB v2 on the holdout")
            reason = f"Top CV candidate {cand} not adopted: " + "; ".join(why) + "."
    log(f"DECISION: deploy {deploy}. {reason}")
    fig_cv_delta(cvt)

    # ---- 4. calibration choice (train OOF only) -----------------------------
    log(f"Fitting {deploy} on the train split and choosing calibration ...")
    ens = fit_ensemble(deploy, "sigmoid", X_tr, y_tr, m_tr)
    method, calt = choose_calibration(ens.oof_raw_, y_tr, m_tr)
    calt.round(5).to_csv(os.path.join(OUT, "v3_calibration.csv"))
    print("\n=== CALIBRATION (cross-fitted on train OOF) ===\n" + calt.round(4).to_string())
    if method != ens.method:
        ens.recalibrate(method, y_tr)
    log(f"  calibration chosen: {method}")

    # ---- 5. holdout: v2-as-deployed vs v3 deployed --------------------------
    base_ens = ens if deploy == BASE else fit_ensemble(BASE, method, X_tr, y_tr, m_tr)
    v2_as_deployed = copy.deepcopy(base_ens).recalibrate("isotonic", y_tr)
    v2_as_deployed._y_fit, v2_as_deployed._m_fit = base_ens._y_fit, base_ens._m_fit
    _, h_v2 = holdout_eval("XGB v2 (as deployed in v2, isotonic)", v2_as_deployed, X_te, y_te, m_te, scale)
    p_dep, h_v3 = holdout_eval(f"{deploy} ({method})", ens, X_te, y_te, m_te, scale)
    hold = pd.DataFrame([h_v2, h_v3]).set_index("model")
    hold.round(5).to_csv(os.path.join(OUT, "v3_holdout.csv"))
    print("\n=== HOLDOUT: v2 as deployed vs v3 deployed ===\n" + hold.round(4).T.to_string())
    # Consistency guard: our re-implementation must reproduce the v2 numbers.
    v2_ref = pd.read_csv(os.path.join(OUT, "tuning_results.csv"), index_col=0).loc["XGBoost"]
    assert abs(h_v2["uplift_vs_no_action_full_pop"] - v2_ref["uplift_vs_no_action_full_pop"]) < 1.0, \
        "v2-as-deployed does not reproduce the v2 tuning results"
    log("  v2-as-deployed reproduces the committed v2 uplift exactly")

    # ---- 6. acceptance sensitivity (deployed model) -------------------------
    log("Acceptance-rate sensitivity ...")
    scen = [("All accept (brief's assumption)", 1.0, 1.0),
            ("75% of everyone accepts", .75, .75), ("50% of everyone accepts", .5, .5),
            ("25% of everyone accepts", .25, .25),
            ("Churners 75%, stayers 100%", .75, 1.0), ("Churners 50%, stayers 100%", .5, 1.0),
            ("Churners 25%, stayers 100%", .25, 1.0)]
    sens = acceptance_sensitivity(ens.oof_, y_tr, m_tr, p_dep, y_te, m_te, scen, scale)
    sens.round(4).to_csv(os.path.join(OUT, "acceptance_sensitivity.csv"), index=False)
    grid = np.round(np.arange(0.05, 1.0001, 0.05), 2)
    curve = pd.concat([
        acceptance_sensitivity(ens.oof_, y_tr, m_tr, p_dep, y_te, m_te,
                               [(f"{a}", a, a) for a in grid], scale).assign(family="symmetric"),
        acceptance_sensitivity(ens.oof_, y_tr, m_tr, p_dep, y_te, m_te,
                               [(f"{a}", a, 1.0) for a in grid], scale).assign(family="stayers_all"),
    ])
    curve.round(4).to_csv(os.path.join(OUT, "acceptance_curve.csv"), index=False)
    # The threshold re-optimises per scenario (the policy shrinks the offer list
    # as acceptance falls), so "break-even" is where uplift stops being positive
    # on the holdout -- or nowhere in the tested range.
    worst = curve[curve["family"] == "stayers_all"].set_index("a_churn")
    unprofitable = worst[worst["uplift_vs_no_action"] <= 0]
    worst_case = dict(
        tested_range=[float(grid.min()), float(grid.max())],
        profitable_across_tested_range=bool(len(unprofitable) == 0),
        highest_unprofitable_churner_acceptance=(float(unprofitable.index.max())
                                                 if len(unprofitable) else None),
        uplift_at={f"{a:.2f}": float(worst.loc[a, "uplift_vs_no_action"]) for a in (0.25, 0.5, 0.75, 1.0)},
    )
    print("\n=== ACCEPTANCE SENSITIVITY ===\n" + sens[["scenario", "breakeven_p", "threshold",
          "offered", "expected_retained_churners", "churn_rate_after",
          "uplift_vs_no_action", "uplift_vs_blanket"]].round(3).to_string(index=False))
    log(f"  worst case (stayers always accept): {worst_case}")
    fig_acceptance(curve)

    # ---- 7. roadmap evidence ------------------------------------------------
    ct = contract_timing(abt_train)
    pd.concat(ct, names=["feature", "bucket"]).round(5).to_csv(os.path.join(OUT, "contract_timing.csv"))
    fig_contract(ct["months_to_end"], abt_train["churn"].mean())
    audit = data_audit(); audit.to_csv(os.path.join(OUT, "data_audit.csv"), index=False)
    print("\n=== DATA AUDIT ===\n" + audit.to_string(index=False))

    # ---- 8. drivers for the deployed model ---------------------------------
    log("Permutation importance (deployed model, holdout) ...")
    pi = permutation_importance(ens, X_te, y_te, n_repeats=5, scoring="average_precision",
                                random_state=RANDOM_STATE, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=X_te.columns).sort_values(ascending=False)
    imp.round(5).to_csv(os.path.join(OUT, "feature_importance_v3.csv"))

    # ---- 9. re-score the verification set -----------------------------------
    changed = not (deploy == BASE and method == "isotonic")
    thr_all, flagged = None, None
    if changed and not args.no_rescore:
        log("Deployed pipeline changed -> refit on ALL training data and re-score test ...")
        final = fit_ensemble(deploy, method, X, y, margin)
        thr_all, _ = best_profit_threshold(final.oof_, y, margin)
        p_test = final.predict_proba(X_test)[:, 1]
        tmpl = pd.read_csv(os.path.join(HERE, "..", "data", "raw", "test_data",
                                        "ml_case_test_output_template.csv"))
        scored = pd.DataFrame({"id": test_ids.values, "Churn_probability": p_test,
                               "Churn_prediction": (p_test >= thr_all).astype(int)})
        out = (tmpl[["id"]].merge(scored, on="id", how="left")
               [["id", "Churn_prediction", "Churn_probability"]]
               .sort_values("Churn_probability", ascending=False).reset_index(drop=True))
        out.to_csv(os.path.join(PRED, "ml_case_test_output_filled.csv"), index=False)
        joblib.dump({"model": final, "threshold": thr_all, "config": deploy, "calibration": method},
                    os.path.join(MODELS, "v3_deployed.joblib"))
        flagged = int(out["Churn_prediction"].sum())
        log(f"  -> predictions: {flagged} flagged at profit threshold {thr_all:.2f}, "
            f"{out['Churn_probability'].round(12).nunique()} distinct scores")

    summary = dict(
        decision_rule=("Deploy the top mean-CV-PR-AUC candidate only if it beats XGB v2 with "
                       "Holm-adjusted p < 0.05 (Nadeau-Bengio corrected paired t-test, repeated "
                       f"{N_SPLITS}x{args.repeats} CV) and is not worse on the holdout."),
        cv_splits=len(splits), top_cv_candidate=cand, deployed_config=deploy,
        decision_reason=reason, calibration=method,
        calibration_reason=("chosen by cross-fitted profit on train OOF predictions"),
        holdout_v2_as_deployed=h_v2, holdout_v3=h_v3,
        acceptance_worst_case=worst_case,
        acceptance_scenarios=sens.to_dict(orient="records"),
        top_features=list(imp.head(10).index),
        test_rescored=bool(changed and not args.no_rescore),
        test_flagged_to_churn=flagged, profit_threshold_all_data=thr_all,
        runtime_minutes=round((time.time() - t0) / 60, 1),
    )
    with open(os.path.join(OUT, "v3_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    log(f"DONE in {summary['runtime_minutes']} min. Deployed: {deploy} ({method})")


if __name__ == "__main__":
    main()
