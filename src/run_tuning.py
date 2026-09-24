"""Tune RF / LightGBM / XGBoost, re-rank by PR-AUC, pick the operating point
by profit, and re-score the verification set.

Leakage-safe design:
- preprocessing lives inside each pipeline -> fit per CV fold
- tuning uses the 75% train split only; the 25% holdout is untouched until
  final scoring
- calibration is fit on train out-of-fold predictions; the profit threshold
  is chosen on those train OOF calibrated probabilities, never on the holdout
"""
from __future__ import annotations
import os, sys, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, f1_score, precision_score,
                             recall_score, fbeta_score)

from data_prep import build_abt
from model_pipeline import RANDOM_STATE
from tuning import build_search, calibrate_oof, SPACES
from economics import (profit_curve, best_profit_threshold, policy_summary)

warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "outputs"); FIG = os.path.join(OUT, "figures")
PRED = os.path.join(OUT, "predictions"); MODELS = os.path.join(HERE, "..", "models")
for d in (OUT, FIG, PRED, MODELS):
    os.makedirs(d, exist_ok=True)

N_ITER_OVERRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else None  # smoke tests


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    t0 = time.time()
    log("Building ABT ...")
    abt_train, _, activity_top = build_abt("train")
    abt_test, test_ids, _ = build_abt("test", activity_top=activity_top)
    y = abt_train["churn"].astype(int); X = abt_train.drop(columns=["churn"])
    margin = abt_train["net_margin"]
    X_tr, X_te, y_tr, y_te, m_tr, m_te = train_test_split(
        X, y, margin, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
    scale = len(X) / len(X_te)
    log(f"  train {X_tr.shape}, holdout {X_te.shape}, scale x{scale:.2f}")

    # v1 baselines (untuned, SMOTE) for a like-for-like comparison.
    base = pd.read_csv(os.path.join(OUT, "model_comparison.csv")).set_index("Model")

    rows, curves, fitted = [], {}, {}
    for name in ["LightGBM", "XGBoost", "Random Forest"]:
        n_iter = N_ITER_OVERRIDE or SPACES[name][2]
        log(f"Tuning {name} (RandomizedSearchCV, n_iter={n_iter}, 5-fold, PR-AUC) ...")
        ts = time.time()
        search = build_search(name, X, n_iter=n_iter)
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        cv_mean = float(search.best_score_)
        cv_std = float(search.cv_results_["std_test_score"][search.best_index_])
        log(f"  {name}: CV PR-AUC {cv_mean:.4f} +/- {cv_std:.4f} "
            f"({time.time()-ts:.0f}s)  params={search.best_params_}")

        # Holdout, raw (threshold-free ranking metrics).
        p_raw = best.predict_proba(X_te)[:, 1]
        roc = roc_auc_score(y_te, p_raw); pr = average_precision_score(y_te, p_raw)

        # Calibrate on train OOF; choose profit threshold on train OOF.
        iso, p_tr_cal = calibrate_oof(best, X_tr, y_tr)
        thr, _ = best_profit_threshold(p_tr_cal, y_tr, m_tr)
        p_te_cal = iso.predict(p_raw)
        yhat = (p_te_cal >= thr).astype(int)
        pol = policy_summary(p_te_cal, y_te.values, m_te.values, thr)
        curves[name] = profit_curve(p_te_cal, y_te.values, m_te.values)
        fitted[name] = (best, iso, thr)

        rows.append({
            "Model": name,
            "CV_PR_AUC_mean": cv_mean, "CV_PR_AUC_std": cv_std,
            "ROC_AUC": roc, "PR_AUC": pr,
            "Brier_calibrated": brier_score_loss(y_te, p_te_cal),
            "profit_threshold": thr,
            "F1@thr": f1_score(y_te, yhat, zero_division=0),
            "Precision@thr": precision_score(y_te, yhat, zero_division=0),
            "Recall@thr": recall_score(y_te, yhat, zero_division=0),
            "Fbeta0.5@thr": fbeta_score(y_te, yhat, beta=0.5, zero_division=0),
            "offered_full_pop": int(pol["offered"] * scale),
            "retained_churners_full_pop": int(pol["retained_churners"] * scale),
            "uplift_vs_no_action_full_pop": pol["uplift_vs_no_action"] * scale,
            "uplift_vs_blanket_full_pop": pol["uplift_vs_blanket"] * scale,
            "baseline_ROC_AUC": float(base.loc[name, "ROC_AUC"]),
            "baseline_PR_AUC": float(base.loc[name, "PR_AUC"]),
            "best_params": json.dumps({k.replace("model__", ""): (v if isinstance(v, (int, float, str)) or v is None else str(v)) for k, v in search.best_params_.items()}),
            "search_seconds": round(time.time() - ts, 1),
        })

    res = pd.DataFrame(rows).set_index("Model")
    res["delta_PR_AUC_vs_baseline"] = res["PR_AUC"] - res["baseline_PR_AUC"]
    # Re-rank: PR-AUC -> profit uplift -> calibration -> ROC-AUC.
    res = res.sort_values(["PR_AUC", "uplift_vs_no_action_full_pop",
                           "Brier_calibrated", "ROC_AUC"],
                          ascending=[False, False, True, False])
    res.round(5).to_csv(os.path.join(OUT, "tuning_results.csv"))
    log("  -> outputs/tuning_results.csv")
    print("\n=== TUNED MODELS (holdout, ranked by PR-AUC) ===")
    print(res[["CV_PR_AUC_mean", "CV_PR_AUC_std", "ROC_AUC", "PR_AUC",
               "baseline_PR_AUC", "delta_PR_AUC_vs_baseline", "Brier_calibrated",
               "profit_threshold", "Precision@thr", "Recall@thr", "Fbeta0.5@thr",
               "uplift_vs_no_action_full_pop", "uplift_vs_blanket_full_pop"]]
          .round(4).to_string())

    winner = res.index[0]
    best, iso, thr = fitted[winner]
    log(f"SELECTED MODEL = {winner} (PR-AUC {res.loc[winner,'PR_AUC']:.4f})")

    # Figures: PR-AUC baseline vs tuned; profit curves on holdout.
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(res.index); xs = np.arange(len(names))
    ax.bar(xs - 0.2, res["baseline_PR_AUC"], 0.4, label="v1 (untuned, SMOTE)", color="#9aa4b2")
    ax.bar(xs + 0.2, res["PR_AUC"], 0.4, label="tuned (class-weighted)", color="#3b7dd8")
    ax.set_xticks(xs); ax.set_xticklabels(names)
    ax.axhline(y_te.mean(), ls="--", color="grey", label=f"no-skill {y_te.mean():.2f}")
    ax.set(title="PR-AUC on holdout: baseline vs tuned", ylabel="PR-AUC"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "tuning_pr_auc.png"), dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, c in curves.items():
        ax.plot(c["threshold"], c["realized_margin"] * scale / 1e6, label=name)
    pol0 = policy_summary(np.zeros(len(y_te)), y_te.values, m_te.values, 1.0)
    ax.axhline(pol0["no_action"] * scale / 1e6, ls="--", color="grey", label="no action")
    ax.axhline(pol0["blanket"] * scale / 1e6, ls=":", color="red", label="blanket 20%")
    ax.set(title="Profit curve on holdout (calibrated probabilities, full population)",
           xlabel="Churn-probability threshold to offer discount",
           ylabel="Annual net margin (millions)"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "tuning_profit_curves.png"), dpi=120); plt.close(fig)

    # Top drivers for the selected model (permutation importance, holdout).
    log("Permutation importance for the selected model ...")
    r = permutation_importance(best, X_te, y_te, n_repeats=5, scoring="average_precision",
                               random_state=RANDOM_STATE, n_jobs=-1)
    imp = pd.Series(r.importances_mean, index=X_te.columns).sort_values(ascending=False)
    imp.round(5).to_csv(os.path.join(OUT, "feature_importance_tuned.csv"))
    fig, ax = plt.subplots(figsize=(7, 6))
    imp.head(15)[::-1].plot.barh(ax=ax, color="#3b7dd8")
    ax.set(title=f"Top 15 churn drivers ({winner}, permutation PR-AUC drop)",
           xlabel="Mean PR-AUC decrease when shuffled")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "feature_importance_tuned.png"), dpi=120); plt.close(fig)

    # Refit winner on ALL training data, calibrate on all-data OOF, score test.
    log("Refitting selected model on all training data and scoring test set ...")
    final = clone(best)  # fresh, unfitted copy carrying the tuned params
    final.fit(X, y)
    iso_all, p_all_oof = calibrate_oof(final, X, y)
    thr_all, _ = best_profit_threshold(p_all_oof, y, margin)
    joblib.dump({"pipeline": final, "isotonic": iso_all, "threshold": thr_all},
                os.path.join(MODELS, "best_tuned_calibrated.joblib"))

    X_test = abt_test[X.columns]
    p_test = iso_all.predict(final.predict_proba(X_test)[:, 1])
    tmpl = pd.read_csv(os.path.join(HERE, "..", "data", "raw", "test_data",
                                    "ml_case_test_output_template.csv"))
    scored = pd.DataFrame({"id": test_ids.values, "Churn_probability": p_test,
                           "Churn_prediction": (p_test >= thr_all).astype(int)})
    out = (tmpl[["id"]].merge(scored, on="id", how="left")
           [["id", "Churn_prediction", "Churn_probability"]]
           .sort_values("Churn_probability", ascending=False).reset_index(drop=True))
    out.to_csv(os.path.join(PRED, "ml_case_test_output_filled.csv"), index=False)
    log(f"  -> predictions: {int(out['Churn_prediction'].sum())} flagged at "
        f"profit threshold {thr_all:.2f}")

    summary = {
        "selected_model": winner,
        "ranking": list(res.index),
        "holdout": res.loc[winner].drop(["best_params"]).to_dict(),
        "best_params": json.loads(res.loc[winner, "best_params"]),
        "profit_threshold_all_data": thr_all,
        "top_features": list(imp.head(10).index),
        "test_flagged_to_churn": int(out["Churn_prediction"].sum()),
        "runtime_minutes": round((time.time() - t0) / 60, 1),
    }
    with open(os.path.join(OUT, "tuning_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    log(f"DONE in {summary['runtime_minutes']} min. Selected: {winner}")


if __name__ == "__main__":
    main()
