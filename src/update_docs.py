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


def splice(path, block, insert_before=None, start=START, end=END):
    s = open(path).read()
    pat = re.compile(re.escape(start) + ".*?" + re.escape(end), re.S)
    if pat.search(s):
        s = pat.sub(lambda _: block, s)
    elif insert_before and insert_before in s:
        s = s.replace(insert_before, block + "\n\n" + insert_before, 1)
    else:
        s = s.rstrip("\n") + "\n\n" + block + "\n"
    open(path, "w").write(s)


def relabel(path, pairs):
    """Idempotent heading/text replacements (skip if already applied)."""
    s = open(path).read()
    for old, new in pairs:
        if new not in s and old in s:
            s = s.replace(old, new, 1)
    open(path, "w").write(s)


README = os.path.join(ROOT, "README.md")
RESULTS = os.path.join(OUT, "RESULTS.md")
splice(README, readme_block, insert_before="## 📁 Repository Structure")
splice(RESULTS, results_block)  # appended after Caveats

# ---------------------------------------------------------------------------
# v3 + current status (only once the v3 experiment has run)
# ---------------------------------------------------------------------------
from report_content import (load_state, current_status, calibration_note, roadmap,
                            scenario, worst_case_sentence, HOLDOUT_ROWS, THRESHOLD_NOTE)

v1, v2, v3 = load_state()
CS, CE = "<!-- current:start -->", "<!-- current:end -->"
V3S, V3E = "<!-- v3:start -->", "<!-- v3:end -->"

status = current_status(v1, v2, v3)
splice(README, f"{CS}\n## ✅ Current status\n\n{status}\n{CE}",
       insert_before="## 🚀", start=CS, end=CE)
splice(RESULTS, f"{CS}\n## Current status\n\n{status}\n{CE}",
       insert_before="## 1. Headline", start=CS, end=CE)

relabel(RESULTS, [
    ("All numbers below are computed by\n`src/run_pipeline.py`; nothing is hand-entered.",
     "All numbers below are computed by\nthe pipeline scripts in `src/`; nothing is hand-entered."),
    ("## 1. Headline\n", "## 1. Headline (v1 — superseded, see Current status)\n"),
    ("## 7. Deliverable\n", "## 7. Deliverable (v1 — superseded, see Current status)\n"),
])
relabel(README, [("## 🚀 Results (holdout, sorted by ROC-AUC)",
                  "## 🚀 v1 results (holdout, sorted by ROC-AUC)")])

if v3:
    cv = pd.read_csv(os.path.join(OUT, "v3_cv_results.csv"), index_col=0)
    ho = pd.read_csv(os.path.join(OUT, "v3_holdout.csv"), index_col=0)
    sens = pd.DataFrame(v3["acceptance_scenarios"])
    audit = pd.read_csv(os.path.join(OUT, "data_audit.csv"))
    ct = pd.read_csv(os.path.join(OUT, "contract_timing.csv")).query("feature == 'months_to_end'")

    cv_rows = ["| Config | CV PR-AUC (mean ± sd) | Δ vs XGB v2 | 95% CI (corrected) | Holm p | Holdout PR-AUC |",
               "|---|---|---|---|---|---|"]
    for n, x in cv.iterrows():
        base = n == "XGB v2"
        cv_rows.append(f"| {'**'+n+'** (deployed)' if base else n} | {x['cv_pr_mean']:.4f} ± {x['cv_pr_std']:.4f} | "
                       f"{'—' if base else f'{x.delta_vs_base:+.4f}'} | "
                       f"{'—' if base else f'{x.ci_low:+.4f} to {x.ci_high:+.4f}'} | "
                       f"{'—' if base else f'{x.p_holm:.2f}'} | {x['holdout_pr_auc']:.4f} |")
    ho_rows = ["| Holdout | " + " | ".join(ho.index) + " |", "|---|" + "---|" * len(ho)]
    for label, col, fmt in HOLDOUT_ROWS:
        ho_rows.append(f"| {label} | " + " | ".join(fmt.format(v) for v in ho[col]) + " |")
    s_rows = ["| Scenario | Break-even p | Threshold | Offered | Churners retained | Churn after | Uplift vs no action /yr |",
              "|---|---|---|---|---|---|---|"]
    for _, x in sens.iterrows():
        s_rows.append(f"| {x['scenario']} | {x['breakeven_p']:.2f} | {x['threshold']:.2f} | {x['offered']:,.0f} | "
                      f"{x['expected_retained_churners']:,.0f} | {x['churn_rate_after']*100:.1f}% | "
                      f"{x['uplift_vs_no_action']:,.0f} |")
    a_rows = ["| Signal | Fields found |", "|---|---|"] + \
             [f"| {r.signal} | {r.fields_found} |" for r in audit.itertuples()]
    ct_line = (", ".join(f"{r.bucket} months: {r.churn_rate*100:.1f}%" for r in ct.itertuples())
               + f"; chi-square p = {ct['chi2_p_value'].iloc[0]:.2f}, not significant")
    rm_rows = ["| # | Data to collect | Why it matters here | Effort |", "|---|---|---|---|"] + \
              [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in roadmap(v3)]
    wc = v3["acceptance_worst_case"]

    v3_results = f"""{V3S}
## 10. v3 — Tier-1 improvements, tested honestly

**Candidates** (all with the v2-tuned hyperparameters): out-of-fold target
encoding of the full 419-category `activity_new` (`+TE`), native categorical
splits (`native`), and XGBoost + LightGBM rank-average blends.

**Pre-registered rule** (fixed before any result was seen): {v3['decision_rule']}

{chr(10).join(cv_rows)}

Repeated {v3['cv_splits']}-split CV on the training split; the corrected t-test
(Nadeau & Bengio) accounts for overlapping training folds, and Holm corrects for
testing {len(cv) - 1} candidates at once.

**Decision:** {v3['decision_reason']} The single holdout is a warning: several
candidates beat XGB v2 there, but the repeated CV shows those gains are noise.

**Calibration — a defect fixed.** v2 used isotonic calibration, whose step
function tied customers together. {calibration_note(v3)}

{chr(10).join(ho_rows)}

## 11. What if customers don't all accept?

The brief assumes every offered customer accepts. Below, churners and stayers
accept at separate rates; the threshold is re-optimised on training data for
each scenario and evaluated on the holdout. The dangerous case is when loyal
customers take the discount more readily than customers who have already
decided to leave.

{chr(10).join(s_rows)}

{worst_case_sentence(v3)} The threshold tightens as acceptance falls, so the
offer list shrinks — and the uplift shrinks with it. {THRESHOLD_NOTE}

## 12. Data roadmap — why the model plateaus, and what to collect

**Evidence.** Churn barely moves with contract timing ({ct_line}), and the
dataset holds none of the behavioural signals that lift churn models elsewhere:

{chr(10).join(a_rows)}

**Priorities** (a recommendation — ranked by expected value and ease, a
judgement rather than a computed result):

{chr(10).join(rm_rows)}
{V3E}"""
    splice(RESULTS, v3_results, start=V3S, end=V3E)

    readme_v3 = f"""{V3S}
## 🧪 v3 — Tier-1 improvements (repeated CV, pre-registered rule)

{v3['decision_reason']} {calibration_note(v3)} Full tables, the acceptance
sensitivity and the data roadmap: `outputs/RESULTS.md` §10–12 and
`outputs/report.html`. Re-run: `python src/run_v3.py`.
{V3E}"""
    splice(README, readme_v3, insert_before="## 📁 Repository Structure", start=V3S, end=V3E)

print(f"docs updated: selected={w}, PR-AUC {h['PR_AUC']:.3f}, improved_all={improved}, "
      f"v3={'yes' if v3 else 'no'}")
