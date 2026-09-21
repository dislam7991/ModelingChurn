"""Build a self-contained visual HTML report from the pipeline outputs."""
import os, base64, json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "outputs")
FIG = os.path.join(OUT, "figures")

summary = json.load(open(os.path.join(OUT, "results_summary.json")))
comp = pd.read_csv(os.path.join(OUT, "model_comparison.csv"))

# Extra chart: model comparison (ROC-AUC + PR-AUC).
c = comp.sort_values("ROC_AUC")
fig, ax = plt.subplots(figsize=(8, 4.5))
y = range(len(c))
ax.barh([i + 0.2 for i in y], c["ROC_AUC"], height=0.4, label="ROC-AUC", color="#3b7dd8")
ax.barh([i - 0.2 for i in y], c["PR_AUC"], height=0.4, label="PR-AUC", color="#d8743b")
ax.set_yticks(list(y)); ax.set_yticklabels(c["Model"])
ax.axvline(0.5, ls="--", color="grey", alpha=0.6)
ax.set(title="Model bake-off: ROC-AUC vs PR-AUC (holdout)", xlabel="Score")
ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "model_comparison.png"), dpi=120)
plt.close(fig)


def b64(name):
    with open(os.path.join(FIG, name), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

econ = summary["economics_full_population"]

tune_path = os.path.join(OUT, "tuning_summary.json")
tuning = json.load(open(tune_path)) if os.path.exists(tune_path) else None
tres = (pd.read_csv(os.path.join(OUT, "tuning_results.csv")).set_index("Model")
        if tuning else None)
comp_rows = "".join(
    f"<tr><td>{r['Model']}</td><td>{r['ROC_AUC']:.3f}</td><td>{r['PR_AUC']:.3f}</td>"
    f"<td>{r['F1@0.5']:.2f}</td><td>{r['Precision@0.5']:.2f}</td>"
    f"<td>{r['Recall@0.5']:.2f}</td></tr>"
    for _, r in comp.iterrows())


def fig_card(title, desc, name):
    return (f'<figure><h3>{title}</h3><p>{desc}</p>'
            f'<img src="{b64(name)}" alt="{title}"></figure>')


def _tuned_rows():
    return "".join(
        f"<tr><td>{n}</td><td>{r['baseline_PR_AUC']:.3f}</td><td><b>{r['PR_AUC']:.3f}</b></td>"
        f"<td>{r['delta_PR_AUC_vs_baseline']:+.3f}</td><td>{r['ROC_AUC']:.3f}</td>"
        f"<td>{r['Brier_calibrated']:.3f}</td><td>{r['profit_threshold']:.2f}</td>"
        f"<td>{r['Precision@thr']:.2f}</td><td>{r['Recall@thr']:.2f}</td>"
        f"<td>{r['Fbeta0.5@thr']:.2f}</td><td>{r['uplift_vs_no_action_full_pop']:,.0f}</td></tr>"
        for n, r in tres.iterrows())

if tuning:
    w = tuning["selected_model"]; h = tuning["holdout"]
    tuning_section = f"""
<h2>5 · Hyperparameter tuning (v2)</h2>
<p>RandomizedSearchCV (5-fold stratified, scored on PR-AUC) over RF, LightGBM
and XGBoost. SMOTE dropped in favour of native class weighting as a tunable
knob; probabilities calibrated (isotonic) on train out-of-fold predictions;
profit threshold chosen on train OOF; all numbers on the untouched holdout.
Ranked by PR-AUC → profit → calibration → ROC-AUC.</p>
<div class="kpis">
  <div class="kpi"><div class="n">{w}</div><div class="l">Selected after tuning</div></div>
  <div class="kpi"><div class="n">{h['PR_AUC']:.3f}</div><div class="l">PR-AUC (was {h['baseline_PR_AUC']:.3f})</div></div>
  <div class="kpi"><div class="n">{h['ROC_AUC']:.3f}</div><div class="l">ROC-AUC (holdout)</div></div>
  <div class="kpi"><div class="n good">+{h['uplift_vs_no_action_full_pop']:,.0f}</div><div class="l">Targeted uplift vs no-action /yr</div></div>
  <div class="kpi"><div class="n good">+{h['uplift_vs_blanket_full_pop']:,.0f}</div><div class="l">Saved vs blanket /yr</div></div>
</div>
{fig_card("PR-AUC: baseline vs tuned", "Every model improves on its untuned v1 score; the no-skill line is the churn rate.", "tuning_pr_auc.png")}
<table><thead><tr><th>Model</th><th>PR-AUC v1</th><th>PR-AUC tuned</th><th>Δ</th>
<th>ROC-AUC</th><th>Brier</th><th>Thr</th><th>Prec</th><th>Recall</th><th>F0.5</th><th>Uplift /yr</th></tr></thead>
<tbody>{_tuned_rows()}</tbody></table>
{fig_card("Profit curves (calibrated, holdout)", "Realized annual net margin vs offer threshold for each tuned model; blanket (red) sits below no-action (grey).", "tuning_profit_curves.png")}
{fig_card(f"Top churn drivers — tuned {w}", "Permutation importance (PR-AUC drop) for the selected tuned model.", "feature_importance_tuned.png")}
"""
else:
    tuning_section = ""


if tuning:
    h = tuning["holdout"]; w = tuning["selected_model"]
    reco_note = f"""<div class="note">
Do <strong>not</strong> offer the 20% discount broadly — it destroys
~{abs(econ['blanket']-econ['no_action']):,.0f}/yr because most customers would
not have churned. Use the tuned <strong>{w}</strong>: offer the
~{int(h['offered_full_pop']):,} customers it scores above the profit-optimal
threshold ({h['profit_threshold']:.2f}, calibrated) <strong>and</strong> who carry
positive margin, ranked by expected value <code>margin × (p − 0.20)</code>.
That retains ~{int(h['retained_churners_full_pop']):,} real churners, adds
~{h['uplift_vs_no_action_full_pop']:,.0f}/yr over doing nothing, and beats the
blanket offer by ~{h['uplift_vs_blanket_full_pop']:,.0f}/yr. Scored verification
set: <code>outputs/predictions/ml_case_test_output_filled.csv</code>.
</div>"""
else:
    reco_note = f"""<div class="note">
Do <strong>not</strong> offer the 20% discount broadly — it destroys
~{abs(econ['blanket']-econ['no_action']):,.0f}/yr because most customers would
not have churned. Target the ~{econ['offered']:,} customers the model scores
above 20% churn probability <strong>and</strong> who carry positive margin;
rank them by expected value <code>margin × (p − 0.20)</code>. This retains
~{econ['retained_churners']} real churners and beats the blanket offer by
~{econ['targeted']-econ['blanket']:,.0f}/yr. The scored verification set is in
<code>outputs/predictions/ml_case_test_output_filled.csv</code>.
</div>"""

html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PowerCo Churn Results</title>
<style>
:root{{--bg:#ffffff;--fg:#1a1d24;--muted:#5b6472;--card:#f5f7fa;
--line:#e2e7ee;--accent:#3b7dd8;--good:#1f9d55;--bad:#d64545;}}
@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{
--bg:#12151c;--fg:#e6e9ef;--muted:#9aa4b2;--card:#1b1f28;--line:#2a303b;}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}}
.wrap{{max-width:960px;margin:0 auto;padding:32px 16px 64px}}
h1{{font-size:1.7rem;margin:0 0 4px}}
h2{{font-size:1.25rem;margin:40px 0 12px;border-bottom:1px solid var(--line);padding-bottom:6px}}
h3{{font-size:1rem;margin:0 0 4px}}
.sub{{color:var(--muted);margin:0 0 24px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}}
.kpi .n{{font-size:1.5rem;font-weight:700}}
.kpi .l{{color:var(--muted);font-size:.85rem}}
table{{width:100%;border-collapse:collapse;font-size:.9rem;margin:8px 0}}
th,td{{text-align:right;padding:7px 10px;border-bottom:1px solid var(--line)}}
th:first-child,td:first-child{{text-align:left}}
thead th{{color:var(--muted);font-weight:600}}
tbody tr:first-child td{{font-weight:700}}
figure{{margin:20px 0;background:var(--card);border:1px solid var(--line);
border-radius:12px;padding:16px}}
figure img{{width:100%;height:auto;border-radius:8px;background:#fff}}
figure p{{color:var(--muted);font-size:.88rem;margin:0 0 10px}}
.grid2{{display:grid;grid-template-columns:1fr;gap:0}}
@media(min-width:720px){{.grid2{{grid-template-columns:1fr 1fr;gap:16px}}
.grid2 figure{{margin:16px 0}}}}
.good{{color:var(--good)}}.bad{{color:var(--bad)}}
.note{{background:var(--card);border-left:3px solid var(--accent);
border-radius:6px;padding:12px 16px;color:var(--muted);font-size:.9rem}}
</style></head><body><div class="wrap">
<h1>PowerCo SME Churn — Results</h1>
<p class="sub">Fictional BCG Gamma case study · every figure computed by the pipeline</p>

<div class="kpis">
  <div class="kpi"><div class="n">{summary['best_model']}</div><div class="l">Best model</div></div>
  <div class="kpi"><div class="n">{summary['holdout_ROC_AUC']:.3f}</div><div class="l">ROC-AUC (holdout)</div></div>
  <div class="kpi"><div class="n">{summary['cv_ROC_AUC_mean']:.3f}</div><div class="l">ROC-AUC (5-fold CV)</div></div>
  <div class="kpi"><div class="n bad">−{abs(econ['blanket']-econ['no_action']):,.0f}</div><div class="l">Blanket discount vs no-action /yr</div></div>
  <div class="kpi"><div class="n good">+{econ['targeted']-econ['blanket']:,.0f}</div><div class="l">Targeting saves vs blanket /yr</div></div>
</div>

<h2>1 · Model bake-off</h2>
<p>Ten classifiers, stratified holdout, SMOTE on train folds only. Ranked by
ROC-AUC; PR-AUC breaks ties (it matters most at a ~10% churn rate, where
accuracy is misleading).</p>
{fig_card("ROC-AUC vs PR-AUC by model",
          "Random Forest leads on ROC-AUC; LightGBM/XGBoost lead on PR-AUC.",
          "model_comparison.png")}
<table><thead><tr><th>Model</th><th>ROC-AUC</th><th>PR-AUC</th><th>F1</th>
<th>Precision</th><th>Recall</th></tr></thead><tbody>{comp_rows}</tbody></table>

<h2>2 · Best model — quality of the scores</h2>
<div class="grid2">
{fig_card("ROC & Precision-Recall", "Ranking quality of the calibrated Random Forest.", "roc_pr.png")}
{fig_card("Calibration", "Predicted probability vs observed churn — used for the money decision.", "calibration.png")}
</div>

<h2>3 · Client questions</h2>
{fig_card("Q1 · Top churn drivers",
          "Permutation importance (AUC drop when a feature is shuffled). Margins, sales channel, recent-consumption ratio, tenure and the 2015 fixed-power price change dominate.",
          "feature_importance.png")}
<div class="grid2">
{fig_card("Q2 · Subscribed power vs consumption",
          "Positive but only moderate (Pearson r=0.10, Spearman rho=0.40). Capacity carries its own signal.",
          "pow_vs_cons.png")}
{fig_card("Q3 · Sales channel vs churn",
          "Significant (chi-square=126, p&approx;4e-24). Churn ranges 5.6%–12.5% by channel.",
          "channel_churn.png")}
</div>

<h2>4 · The 20% discount — economics</h2>
<p>Offering trades full margin for 80% of margin but guarantees retention, so
<code>EV(offer) − EV(no offer) = margin × (p − 0.20)</code>. Offer a
positive-margin customer only when churn probability p &gt; 0.20.</p>
{fig_card("Realized annual margin vs discount threshold",
          "Blanket (red) sits below no-action (grey): discounting everyone destroys margin. The targeted policy peaks near the EV-optimal 0.20 threshold.",
          "discount_ev_curve.png")}
<table><thead><tr><th>Strategy</th><th>Offered</th><th>Annual net margin</th><th>vs no-action</th></tr></thead>
<tbody>
<tr><td>No action</td><td>0</td><td>{econ['no_action']:,.0f}</td><td>—</td></tr>
<tr><td>Blanket 20% to all</td><td>16,096</td><td>{econ['blanket']:,.0f}</td><td class="bad">{econ['blanket']-econ['no_action']:,.0f}</td></tr>
<tr><td>Targeted (p&gt;0.20 &amp; margin&gt;0)</td><td>{econ['offered']:,}</td><td>{econ['targeted']:,.0f}</td><td class="good">+{econ['targeted']-econ['no_action']:,.0f}</td></tr>
<tr><td>Oracle (perfect foresight)</td><td>1,572</td><td>{econ['oracle']:,.0f}</td><td class="good">+{econ['oracle']-econ['no_action']:,.0f}</td></tr>
</tbody></table>

{tuning_section}
<h2>6 · Recommendation</h2>
{reco_note}


<p class="sub" style="margin-top:32px">Model AUC ~0.70: use the ranking plus the
margin filter, not individual raw labels. Dollar figures assume every offered
customer accepts and stays a year (per the brief) and use net_margin as the
annual margin proxy.</p>
</div></body></html>"""

with open(os.path.join(OUT, "report.html"), "w") as f:
    f.write(html)
print("wrote outputs/report.html", round(len(html)/1024, 1), "KB")
