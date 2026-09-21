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
