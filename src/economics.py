"""Discount economics shared by the v1 pipeline and the tuning runner.

From the brief: an offered customer accepts and is retained for a year at
(1 - DISCOUNT) of their annual net margin. A customer not offered keeps full
margin if they stay and contributes zero if they churn. Hence
    EV(offer) - EV(no offer) = margin * (p - DISCOUNT)
so a positive-margin customer is worth offering only when p > DISCOUNT.

The *profit curve* is realized annual net margin as a function of the offer
threshold, computed on held-out rows using actual churn ``y`` and actual
``net_margin`` -- our own business function, not a library metric.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

DISCOUNT = 0.20


def _m(margin):
    return np.nan_to_num(np.asarray(margin, float), nan=0.0)


def realized_margin(offered, y, margin, discount=DISCOUNT) -> float:
    offered = np.asarray(offered, bool); y = np.asarray(y); m = _m(margin)
    val = np.empty_like(m)
    val[offered] = (1 - discount) * m[offered]            # accept & retained
    stay = (~offered) & (y == 0); left = (~offered) & (y == 1)
    val[stay] = m[stay]; val[left] = 0.0                  # keep or lose
    return float(val.sum())


def profit_curve(p, y, margin, grid=None) -> pd.DataFrame:
    """Realized margin vs threshold for the rule: offer iff p>=t and margin>0."""
    grid = np.linspace(0.0, 0.95, 96) if grid is None else grid
    p = np.asarray(p); m = _m(margin)
    rows = []
    for t in grid:
        offered = (p >= t) & (m > 0)
        rows.append((t, realized_margin(offered, y, m), int(offered.sum())))
    return pd.DataFrame(rows, columns=["threshold", "realized_margin", "n_offered"])


def best_profit_threshold(p, y, margin):
    c = profit_curve(p, y, margin)
    i = int(c["realized_margin"].idxmax())
    return float(c.loc[i, "threshold"]), float(c.loc[i, "realized_margin"])


def policy_summary(p, y, margin, thr) -> dict:
    """No-action / blanket / targeted / oracle realized margins at ``thr``."""
    p = np.asarray(p); y = np.asarray(y); m = _m(margin); n = len(y)
    no_action = realized_margin(np.zeros(n, bool), y, m)
    blanket = realized_margin(np.ones(n, bool), y, m)
    offered = (p >= thr) & (m > 0)
    targeted = realized_margin(offered, y, m)
    oracle = realized_margin((y == 1) & (m > 0), y, m)
    return dict(
        threshold=float(thr), no_action=no_action, blanket=blanket,
        targeted=targeted, oracle=oracle, offered=int(offered.sum()),
        retained_churners=int((offered & (y == 1)).sum()),
        wasted_offers=int((offered & (y == 0)).sum()),
        uplift_vs_no_action=targeted - no_action,
        uplift_vs_blanket=targeted - blanket,
    )


# ---------------------------------------------------------------------------
# Acceptance-rate sensitivity
# ---------------------------------------------------------------------------
# The brief assumes every offered customer accepts. Relax that with separate
# acceptance rates for would-be churners (a_churn) and would-be stayers
# (a_stay). An offered customer who declines behaves as if not offered.
# Per customer:  EV(offer) - EV(no offer)
#   = m * [ (1-d) * p * a_churn  -  d * (1-p) * a_stay ]
# so it pays to offer iff  p > d*a_stay / ((1-d)*a_churn + d*a_stay).
# With a_churn = a_stay the break-even stays at d (0.20); the dangerous case is
# a_stay > a_churn (loyal customers happily pocket a discount while customers
# already signed with a competitor decline it).

def breakeven_probability(a_churn, a_stay, discount=DISCOUNT) -> float:
    den = (1 - discount) * a_churn + discount * a_stay
    return 1.0 if den <= 0 else discount * a_stay / den


def expected_margin(offered, y, margin, a_churn=1.0, a_stay=1.0,
                    discount=DISCOUNT) -> float:
    """Expected annual margin under partial acceptance (a=1 -> realized_margin)."""
    offered = np.asarray(offered, bool); y = np.asarray(y); m = _m(margin)
    base = np.where(y == 0, m, 0.0)                    # outcome if not offered
    a = np.where(y == 1, a_churn, a_stay)
    offer_val = a * (1 - discount) * m + (1 - a) * base
    return float(np.where(offered, offer_val, base).sum())


def acceptance_sensitivity(p_fit, y_fit, m_fit, p_eval, y_eval, m_eval,
                           scenarios, scale=1.0) -> pd.DataFrame:
    """Per scenario: pick the threshold on the *fit* data, evaluate on *eval*.

    ``scenarios`` is an iterable of (label, a_churn, a_stay).
    """
    grid = np.linspace(0.0, 0.95, 96)
    mf, me = _m(m_fit), _m(m_eval)
    y_eval = np.asarray(y_eval); n = len(y_eval)
    rows = []
    for label, a_c, a_s in scenarios:
        vals = [expected_margin((np.asarray(p_fit) >= t) & (mf > 0), y_fit, mf, a_c, a_s)
                for t in grid]
        thr = float(grid[int(np.argmax(vals))])
        offered = (np.asarray(p_eval) >= thr) & (me > 0)
        no_action = expected_margin(np.zeros(n, bool), y_eval, me, a_c, a_s)
        blanket = expected_margin(np.ones(n, bool), y_eval, me, a_c, a_s)
        targeted = expected_margin(offered, y_eval, me, a_c, a_s)
        retained = a_c * int((offered & (y_eval == 1)).sum())
        churners = int((y_eval == 1).sum())
        rows.append(dict(
            scenario=label, a_churn=a_c, a_stay=a_s,
            breakeven_p=breakeven_probability(a_c, a_s), threshold=thr,
            offered=int(offered.sum() * scale),
            expected_retained_churners=retained * scale,
            churn_rate_after=(churners - retained) / n,
            uplift_vs_no_action=(targeted - no_action) * scale,
            blanket_vs_no_action=(blanket - no_action) * scale,
            uplift_vs_blanket=(targeted - blanket) * scale,
        ))
    return pd.DataFrame(rows)
