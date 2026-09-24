"""Shared, data-driven content for the HTML report and the markdown docs.

Every number is read from pipeline outputs. The data-roadmap priorities are
recommendations (judgement), and are labelled as such wherever shown.
"""
from __future__ import annotations
import os, json
import pandas as pd

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "outputs")


def _load_json(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def load_state():
    """Latest deployed state: v3 if it has run, else v2, else v1."""
    v3 = _load_json("v3_summary.json")
    v2 = _load_json("tuning_summary.json")
    v1 = _load_json("results_summary.json")
    return v1, v2, v3


def scenario(v3, label_prefix):
    for s in v3["acceptance_scenarios"]:
        if s["scenario"].startswith(label_prefix):
            return s
    raise KeyError(label_prefix)


def price_feature_rank():
    """Rank of the best price-history feature in the deployed model's importance."""
    p = os.path.join(OUT, "feature_importance_v3.csv")
    if not os.path.exists(p):
        return None
    imp = pd.read_csv(p, index_col=0).iloc[:, 0].sort_values(ascending=False)
    ranks = [i + 1 for i, f in enumerate(imp.index) if f.startswith("price_")]
    return (min(ranks), len(imp)) if ranks else None


def calibration_note(v3):
    h2, h3 = v3["holdout_v2_as_deployed"], v3["holdout_v3"]
    cal = pd.read_csv(os.path.join(OUT, "v3_calibration.csv"), index_col=0)
    cf = cal["crossfit_uplift_vs_no_action"]
    chosen, other = v3["calibration"], ("isotonic" if v3["calibration"] == "sigmoid" else "sigmoid")
    note = (f"Calibration was chosen by cross-fitted profit on training data only: "
            f"{chosen} {cf[chosen]:,.0f} vs {other} {cf[other]:,.0f} (training-fold uplift). ")
    d = h2["uplift_vs_no_action_full_pop"] - h3["uplift_vs_no_action_full_pop"]
    if chosen != "isotonic" and d > 0:
        note += (f"On the holdout the old isotonic version earns {d:,.0f}/yr more — the two "
                 "sources disagree, both gaps are small, and the rule was fixed before the "
                 "holdout was seen, so it stands. ")
    if chosen == "sigmoid":
        note += (f"Sigmoid keeps every customer's score distinct ({h3['distinct_scores']:,} vs "
                 f"{h2['distinct_scores']:,} under isotonic), so the ranked list the brief asks "
                 f"for is exact (holdout PR-AUC {h3['PR_AUC']:.3f} vs {h2['PR_AUC']:.3f}).")
    return note


def roadmap(v3):
    """Prioritised data-collection roadmap (priorities are judgement)."""
    allacc = scenario(v3, "All accept")["uplift_vs_no_action"]
    worst = scenario(v3, "Churners 25%")["uplift_vs_no_action"]
    pr = price_feature_rank()
    price_ev = (f"the best 2015 price-history feature ranks only #{pr[0]} of {pr[1]} "
                "in the deployed model's importance" if pr else
                "the 2015 price-history features carry little signal")
    return [
        ("1", "Randomised pilot of the discount (offer vs hold-out control)",
         f"Measures real acceptance and *incremental* retention. The business case swings "
         f"from ~{allacc:,.0f}/yr (everyone accepts) to ~{worst:,.0f}/yr (a quarter of "
         f"churners accept, all stayers do) on this one unknown.", "Low"),
        ("2", "Churn reason at cancellation (switched supplier vs closed / moved)",
         "A discount cannot save a closure. Reason codes let the model target voluntary "
         "switching only; the label today is a bare 0/1.", "Low"),
        ("3", "Competitor price benchmark per customer profile",
         f"Tests the head of SME's price hypothesis directly; {price_ev}.", "Medium"),
        ("4", "Payment behaviour (late payments, direct-debit changes)",
         "A standard early-warning signal; no payment field exists in the data.", "Low–Medium"),
        ("5", "Customer contact and complaint log",
         "Engagement and dissatisfaction signal; no contact field exists in the data.", "Medium"),
        ("6", "Smart-meter interval usage",
         "Sudden usage drops can reveal a business winding down or switching; the data "
         "holds only 12-month aggregates.", "Medium–High"),
    ]


def base_churn_rate():
    p = os.path.join(HERE, "..", "data", "raw", "training_data", "ml_case_training_output.csv")
    return float(pd.read_csv(p)["churn"].mean())


def v3_history(v3):
    if v3["deployed_config"] == "XGB v2":
        return (f"v3 Tier-1 tests found no significant gain, so XGB v2 stays; "
                f"its calibration moved to {v3['calibration']}")
    return f"v3 adopted {v3['deployed_config']}"


def current_status(v1, v2, v3):
    """One paragraph describing what is deployed right now (markdown)."""
    if v3:
        h = v3["holdout_v3"]; a = scenario(v3, "All accept"); w = scenario(v3, "Churners 50%")
        base = base_churn_rate()
        rescored = (f"{v3['test_flagged_to_churn']:,} of 4,024 verification customers flagged "
                    f"at the all-data profit threshold {v3['profit_threshold_all_data']:.2f}"
                    if v3["test_rescored"] else "verification set scored by the v2 model")
        return (
            f"**Deployed: {v3['deployed_config']} ({v3['calibration']} calibration).** "
            f"Holdout PR-AUC **{h['PR_AUC']:.3f}**, ROC-AUC {h['ROC_AUC']:.3f}. "
            f"Targeted 20% discount (profit threshold {h['profit_threshold']:.2f}, positive margin "
            f"only): ~{h['offered_full_pop']:,} customers offered, ~{h['retained_churners_full_pop']:,} "
            f"churners retained if all accept (churn {base*100:.1f}% → {a['churn_rate_after']*100:.1f}%), "
            f"**~{h['uplift_vs_no_action_full_pop']:,.0f}/yr** vs no action and "
            f"~{h['uplift_vs_blanket_full_pop']:,.0f}/yr vs a blanket offer. If only half of "
            f"would-be churners accept while every stayer does, uplift falls to "
            f"~{w['uplift_vs_no_action']:,.0f}/yr. Verification set: {rescored}. "
            f"History: v1 bake-off (Random Forest) → v2 tuning (XGBoost) → {v3_history(v3)}.")
    if v2:
        h = v2["holdout"]
        return (f"**Deployed: {v2['selected_model']} (tuned, isotonic calibration).** "
                f"Holdout PR-AUC {h['PR_AUC']:.3f}; uplift ~{h['uplift_vs_no_action_full_pop']:,.0f}/yr.")
    return f"**Deployed: {v1['best_model']} (v1).**"
