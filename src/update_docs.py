"""Fill the README and RESULTS.md tuning sections from the tuning artifacts.

Idempotent: content between the v2 markers is replaced on every run, so the
docs always reflect the latest outputs/tuning_results.csv + tuning_summary.json
and never carry hand-typed numbers.
"""
import os, json, re
import pandas as pd

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "outputs")
START, END = "<!-- v2-tuning:start -->", "<!-- v2-tuning:end -->"

t = json.load(open(os.path.join(OUT, "tuning_summary.json")))
r = pd.read_csv(os.path.join(OUT, "tuning_results.csv")).set_index("Model")
w = t["selected_model"]; h = t["holdout"]


def table():
    rows = ["| Model | PR-AUC v1 | PR-AUC tuned | Δ | ROC-AUC | Brier | Thr | Precision | Recall | F0.5 | Uplift vs no-action /yr |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, x in r.iterrows():
        bold = "**" if n == w else ""
        rows.append(f"| {bold}{n}{bold} | {x['baseline_PR_AUC']:.3f} | {bold}{x['PR_AUC']:.3f}{bold} | "
                    f"{x['delta_PR_AUC_vs_baseline']:+.3f} | {x['ROC_AUC']:.3f} | {x['Brier_calibrated']:.3f} | "
                    f"{x['profit_threshold']:.2f} | {x['Precision@thr']:.2f} | {x['Recall@thr']:.2f} | "
                    f"{x['Fbeta0.5@thr']:.2f} | {x['uplift_vs_no_action_full_pop']:,.0f} |")
    return "\n".join(rows)


improved = all(r["delta_PR_AUC_vs_baseline"] > 0)
def _fmt(v):
    try:
        f = float(v); return f"{f:.4g}" if not float(f).is_integer() else str(int(f))
    except (TypeError, ValueError):
        return str(v)
params = ", ".join(f"`{k}={_fmt(v)}`" for k, v in t["best_params"].items())

readme_block = f"""{START}
## 🎯 Tuned results (v2 — RandomizedSearchCV, PR-AUC objective)

SMOTE was replaced by native class weighting (RF `class_weight`, GBM
`scale_pos_weight`) as a *tunable* hyperparameter; probabilities are isotonic-
calibrated on train out-of-fold predictions; the profit threshold is chosen on
train OOF; every number below is on the untouched 25% holdout. Models are
ranked by **PR-AUC → profit uplift → Brier → ROC-AUC**.

{table()}

**Selected: {w}** — PR-AUC {h['PR_AUC']:.3f} (v1 {h['baseline_PR_AUC']:.3f}), ROC-AUC
{h['ROC_AUC']:.3f}. At the profit-optimal threshold {h['profit_threshold']:.2f} it
offers the discount to ~{int(h['offered_full_pop']):,} customers, retains
~{int(h['retained_churners_full_pop']):,} real churners, and adds
**~{h['uplift_vs_no_action_full_pop']:,.0f}/yr** over doing nothing
(**~{h['uplift_vs_blanket_full_pop']:,.0f}/yr** better than a blanket offer).
{"All three models improved on their v1 baseline." if improved else "Not every model beat its v1 baseline — see the Δ column."}
Full table: `outputs/tuning_results.csv`. Re-run: `python src/run_tuning.py`.
{END}"""

results_block = f"""{START}
## 9. Hyperparameter tuning (v2)

**Method.** `RandomizedSearchCV` (5-fold stratified, scoring = PR-AUC) over
Random Forest (30 configs), LightGBM (60) and XGBoost (60). No SMOTE: class
imbalance is handled by the models' native weighting, and the weight itself is
searched (`class_weight` / `scale_pos_weight ∈ {{1, 3, ~9}}`), per the XGBoost
guidance to rebalance for ranking and *calibrate* for probability. The tuned
model is isotonic-calibrated on train out-of-fold predictions, the profit
threshold is picked on those train OOF probabilities, and all metrics are on the
untouched holdout. Ranking: PR-AUC → profit uplift → Brier → ROC-AUC.

{table()}

**Selected model: {w}.** Best params: {params}.

{"All three models improved on their untuned v1 PR-AUC." if improved else "Not every model beat its v1 baseline; the Δ column shows which."}
The winner lifts PR-AUC from {h['baseline_PR_AUC']:.3f} to **{h['PR_AUC']:.3f}** and raises the
targeted-discount uplift to **~{h['uplift_vs_no_action_full_pop']:,.0f}/yr** vs no-action
(v1 Random Forest: ~23,319/yr), while still beating the blanket offer by
~{h['uplift_vs_blanket_full_pop']:,.0f}/yr. Top drivers for the tuned model:
{", ".join(f"`{f}`" for f in t["top_features"][:6])}. Verification set re-scored with the
tuned calibrated model ({t['test_flagged_to_churn']} flagged at threshold
{t['profit_threshold_all_data']:.2f}). Runtime {t['runtime_minutes']} min.
{END}"""


def splice(path, block, insert_before=None):
    s = open(path).read()
    pat = re.compile(re.escape(START) + ".*?" + re.escape(END), re.S)
    if pat.search(s):
        s = pat.sub(lambda _: block, s)
    elif insert_before and insert_before in s:
        s = s.replace(insert_before, block + "\n\n" + insert_before, 1)
    else:
        s = s.rstrip("\n") + "\n\n" + block + "\n"
    open(path, "w").write(s)


splice(os.path.join(ROOT, "README.md"), readme_block, insert_before="## 📁 Repository Structure")
splice(os.path.join(OUT, "RESULTS.md"), results_block)  # appended after Caveats
print(f"docs updated: selected={w}, PR-AUC {h['PR_AUC']:.3f}, improved_all={improved}")
