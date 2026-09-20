# 20% Discount — Economic Analysis

**Decision model.** A customer offered the discount accepts (per the brief's assumption) and is retained for a year at 80% of their net margin. A customer not offered keeps full margin if they stay and contributes zero if they churn. The expected value of offering minus not offering is `m*(p - 0.20)`, so it pays to offer a **positive-margin** customer only when calibrated churn probability **p > 0.20**. Negative-margin customers are never offered.

Figures are annual `net_margin`, extrapolated to the full ~16,096-customer population (holdout x4.00).

| Strategy | Customers offered | Annual net margin | vs no-action |
| --- | --- | --- | --- |
| No action | 0 | 3,095,984 | — |
| Blanket 20% to all | 16,096 | 2,871,001 | -224,983 |
| **Targeted (p>0.20 & margin>0)** | 824 | **3,119,303** | **23,319** |
| Oracle (perfect foresight) | 1,572 | 3,491,367 | 395,383 |

- The EV-optimal rule offers to **824** customers (~5% of the base), retaining **260** true churners with **564** discounts going to customers who would have stayed.
- Targeted uplift over doing nothing: **23,319** per year.
- Targeted uplift over the blanket offer: **248,302** per year — the money saved by *not* discounting everyone.
- Sensitivity: the empirically best threshold on this holdout is ~0.17, close to the theoretical 0.20.

**Recommendation.** A blanket 20% discount destroys value (-224,983 vs no action) because most customers would not have churned and the locked-in price cut is pure lost margin on them. Offer the discount only to customers with calibrated churn probability above 20% **and** positive margin. Rank by expected value `margin*(p-0.20)`, not by churn probability alone.