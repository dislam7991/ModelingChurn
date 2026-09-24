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

# ---------------------------------------------------------------------------
# v3: header reflects what is deployed now; v3 / acceptance / roadmap sections
# ---------------------------------------------------------------------------
from report_content import (load_state, calibration_note, roadmap, scenario,
                            base_churn_rate, v3_history, worst_case_sentence,
                            HOLDOUT_ROWS, THRESHOLD_NOTE)

_, _, v3 = load_state()


def _kpi(n, label, cls=""):
    return f'<div class="kpi"><div class="n {cls}">{n}</div><div class="l">{label}</div></div>'


def _table(head, rows, cls="", left_cols=(0,)):
    th = "".join(f'<th class="{"l" if i in left_cols else ""}">{h}</th>' for i, h in enumerate(head))
    body = ""
    for r in rows:
        tr_cls, cells = (r[0], r[1]) if isinstance(r, tuple) else ("", r)
        tds = "".join(f'<td class="{"l" if i in left_cols else ""}">{c}</td>' for i, c in enumerate(cells))
        body += f'<tr class="{tr_cls}">{tds}</tr>'
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


if v3:
    h3 = v3["holdout_v3"]
    header_kpis = '<div class="kpis">' + "".join([
        _kpi(f"{v3['deployed_config']}", f"Deployed model ({v3['calibration']} calibration)"),
        _kpi(f"{h3['PR_AUC']:.3f}", "PR-AUC (holdout)"),
        _kpi(f"{h3['ROC_AUC']:.3f}", "ROC-AUC (holdout)"),
        _kpi(f"+{h3['uplift_vs_no_action_full_pop']:,.0f}", "Targeted uplift vs no action /yr", "good"),
        _kpi(f"+{h3['uplift_vs_blanket_full_pop']:,.0f}", "Saved vs blanket discount /yr", "good"),
    ]) + "</div>"
elif tuning:
    header_kpis = '<div class="kpis">' + "".join([
        _kpi(tuning["selected_model"], "Deployed model (v2)"),
        _kpi(f"{tuning['holdout']['PR_AUC']:.3f}", "PR-AUC (holdout)"),
        _kpi(f"+{tuning['holdout']['uplift_vs_no_action_full_pop']:,.0f}", "Uplift vs no action /yr", "good"),
    ]) + "</div>"
else:
    header_kpis = '<div class="kpis">' + _kpi(summary["best_model"], "Best model") + "</div>"

if v3:
    cv = pd.read_csv(os.path.join(OUT, "v3_cv_results.csv"), index_col=0)
    ho = pd.read_csv(os.path.join(OUT, "v3_holdout.csv"), index_col=0)
    cal = pd.read_csv(os.path.join(OUT, "v3_calibration.csv"), index_col=0)
    audit = pd.read_csv(os.path.join(OUT, "data_audit.csv"))
    sens = pd.DataFrame(v3["acceptance_scenarios"])
    wc = v3["acceptance_worst_case"]

    cv_rows = []
    for n, x in cv.iterrows():
        base = n == "XGB v2"
        cv_rows.append(("hl" if base else "", [
            n + (" (deployed)" if base else ""), f"{x['cv_pr_mean']:.4f} ± {x['cv_pr_std']:.4f}",
            "—" if base else f"{x['delta_vs_base']:+.4f}",
            "—" if base else f"{x['ci_low']:+.4f} to {x['ci_high']:+.4f}",
            "—" if base else f"{x['p_holm']:.2f}", f"{x['holdout_pr_auc']:.4f}"]))
    cv_table = _table(["Config", "CV PR-AUC (mean ± sd)", "Δ vs XGB v2", "95% CI (corrected)",
                       "Holm p", "Holdout PR-AUC"], cv_rows, cls="plain")

    ho_table = _table(["Holdout"] + list(ho.index),
                      [[lbl] + [f.format(v) for v in ho[col]] for lbl, col, f in HOLDOUT_ROWS],
                      cls="plain")
    cal_table = _table(["Method", "Training cross-fit uplift", "Brier", "Log-loss", "PR-AUC", "Distinct scores"],
                       [("hl" if m == v3["calibration"] else "",
                         [m + (" (chosen)" if m == v3["calibration"] else ""),
                          f"{r['crossfit_uplift_vs_no_action']:,.0f}", f"{r['brier']:.4f}",
                          f"{r['log_loss']:.4f}", f"{r['pr_auc']:.4f}", f"{r['distinct_scores']:,.0f}"])
                        for m, r in cal.iterrows()], cls="plain")
    sens_table = _table(["Scenario", "Break-even p", "Threshold", "Offered", "Churners retained",
                         "Churn after", "Uplift vs no action /yr"],
                        [[s["scenario"], f"{s['breakeven_p']:.2f}", f"{s['threshold']:.2f}",
                          f"{s['offered']:,.0f}", f"{s['expected_retained_churners']:,.0f}",
                          f"{s['churn_rate_after']*100:.1f}%", f"{s['uplift_vs_no_action']:,.0f}"]
                         for _, s in sens.iterrows()])
    audit_table = _table(["Signal", "Fields found in the data"],
                         [[r.signal, r.fields_found] for r in audit.itertuples()],
                         cls="plain", left_cols=(0, 1))
    rm_table = _table(["#", "Data to collect", "Why it matters here", "Effort"],
                      [[a, b, c.replace("*", ""), d] for a, b, c, d in roadmap(v3)],
                      cls="plain", left_cols=(1, 2))
    half = scenario(v3, "Churners 50%")["uplift_vs_no_action"]
    ct_p = pd.read_csv(os.path.join(OUT, "contract_timing.csv")).query(
        "feature == 'months_to_end'")["chi2_p_value"].iloc[0]

    v3_section = f"""
<h2>6 · v3 — Tier-1 improvements, tested honestly</h2>
<p>Candidates, all with the v2-tuned hyperparameters: out-of-fold target encoding
of the full 419-category <code>activity_new</code> (+TE), native categorical splits,
and XGBoost + LightGBM blends. <strong>Pre-registered rule</strong> (fixed before any
result was seen): {v3['decision_rule']}</p>
{fig_card("Change in PR-AUC vs the deployed XGB v2",
          f"Repeated {v3['cv_splits']}-split CV with a corrected paired t-test (Nadeau &amp; Bengio) and Holm correction. Every interval that crosses the dashed line is indistinguishable from no change.",
          "v3_cv_delta.png")}
{cv_table}
<div class="note"><strong>Decision:</strong> {v3['decision_reason']} On the single
holdout several candidates look better than XGB v2 — the repeated CV shows those
gains are noise, which is exactly why one split is not enough.</div>

<h3 style="margin-top:24px">Calibration — a defect fixed</h3>
<p>v2 used isotonic calibration, whose step function tied customers together.
{calibration_note(v3)}</p>
{cal_table}
{ho_table}

<h2>7 · What if customers don't all accept?</h2>
<p>The brief assumes every offered customer accepts. Here churners and stayers
accept at separate rates; the threshold is re-chosen on training data for each
scenario and evaluated on the holdout. The dangerous case is loyal customers
pocketing the discount while customers who have already decided to leave decline it.</p>
{fig_card("Uplift vs acceptance rate",
          f"Blue: everyone accepts at the same rate. Orange: every stayer accepts (worst case). {worst_case_sentence(v3)}",
          "v3_acceptance.png")}
{sens_table}
<p class="sub">{THRESHOLD_NOTE}</p>

<h2>8 · Data roadmap — why the model plateaus</h2>
<p>The model is limited by its data, not its tuning. Churn barely varies with
contract timing, and none of the behavioural signals that lift churn models
elsewhere exist in this dataset.</p>
{fig_card("Churn rate by months until contract end",
          f"Every bucket sits within about one point of the overall rate (chi-square p = {ct_p:.2f}, not significant) — contract expiry is not a churn trigger in this data.",
          "v3_contract_timing.png")}
{audit_table}
<p><strong>Priorities</strong> — a recommendation, ranked by expected value and ease
(judgement, not a computed result):</p>
{rm_table}
"""
    reco_num = 9
    reco_note = f"""<div class="note">
Do <strong>not</strong> offer the 20% discount broadly — a blanket offer loses money.
Use <strong>{v3['deployed_config']}</strong> ({v3['calibration']} calibration) and offer the
discount to the ~{h3['offered_full_pop']:,} customers above the profit threshold
({h3['profit_threshold']:.2f}) <strong>with positive margin</strong>, ranked by
<code>margin × (p − 0.20)</code>. If all accept, that retains ~{h3['retained_churners_full_pop']:,}
churners (churn {base_churn_rate()*100:.1f}% → {scenario(v3, 'All accept')['churn_rate_after']*100:.1f}%)
and adds ~{h3['uplift_vs_no_action_full_pop']:,.0f}/yr. If only half of would-be churners
accept while every stayer does, uplift falls to ~{half:,.0f}/yr — so
<strong>launch it as a randomised pilot</strong> and collect the roadmap data before
scaling. {v3_history(v3)[0].upper() + v3_history(v3)[1:]}.
</div>"""
else:
    v3_section, reco_num = "", 6

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
table.plain tbody tr:first-child td{{font-weight:400}}
@media(max-width:720px){{table{{display:block;overflow-x:auto;max-width:100%}}}}
tr.hl td{{font-weight:700}}
td.l,th.l{{text-align:left}}
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

{header_kpis}

<h2>1 · Model bake-off (v1)</h2>
<p>Ten classifiers, stratified holdout, SMOTE on train folds only. Ranked by
ROC-AUC; PR-AUC breaks ties (it matters most at a ~10% churn rate, where
accuracy is misleading).</p>
{fig_card("ROC-AUC vs PR-AUC by model",
          "Random Forest leads on ROC-AUC; LightGBM/XGBoost lead on PR-AUC.",
          "model_comparison.png")}
<table><thead><tr><th>Model</th><th>ROC-AUC</th><th>PR-AUC</th><th>F1</th>
<th>Precision</th><th>Recall</th></tr></thead><tbody>{comp_rows}</tbody></table>

<h2>2 · v1 best model — quality of the scores</h2>
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

<h2>4 · The 20% discount — economics (v1 model)</h2>
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
{v3_section}
<h2>{reco_num} · Recommendation</h2>
{reco_note}


<p class="sub" style="margin-top:32px">Model ROC-AUC ~0.71: use the ranking plus
the margin filter, not individual raw labels. Headline dollar figures follow the
brief's assumption that every offered customer accepts and stays a year (see the
acceptance sensitivity for what happens when they don't) and use net_margin as
the annual margin proxy.</p>
</div></body></html>"""

with open(os.path.join(OUT, "report.html"), "w") as f:
    f.write(html)
print("wrote outputs/report.html", round(len(html)/1024, 1), "KB")
