# Answers to the Client Questions

## Q1. Most explicative variables for churn

Permutation importance (drop in ROC-AUC when a feature is shuffled) on the held-out set, model = Random Forest.

| Rank | Feature | Importance |
| --- | --- | --- |
| 1 | margin_gross_pow_ele | 0.03175 |
| 2 | margin_net_pow_ele | 0.02725 |
| 3 | channel_sales | 0.01719 |
| 4 | recent_cons_ratio | 0.00834 |
| 5 | net_margin_per_cons | 0.00830 |
| 6 | num_years_antig | 0.00825 |
| 7 | price_p1_fix_change | 0.00813 |
| 8 | activity_new | 0.00798 |
| 9 | origin_up | 0.00659 |
| 10 | price_p1_fix_std | 0.00656 |
| 11 | lifespan_years | 0.00524 |
| 12 | cons_last_month | 0.00477 |
| 13 | cons_per_product | 0.00459 |
| 14 | months_to_end | 0.00457 |
| 15 | price_p1_fix_mean | 0.00365 |

## Q2. Correlation between subscribed power and consumption

- Pearson r = **0.103** (p = 5.45e-39)
- Spearman rho = **0.396** (p = 0.00e+00)
- Interpretation: weak/none — higher subscribed power tends to go with higher consumption.

## Q3. Link between sales channel and churn

- Chi-square = **126.1**, dof = 7, p = **4.13e-24** (significant at 5%).
- Churn rate by channel:

| Channel (hashed) | Churn rate | Customers |
| --- | --- | --- |
| foosdfpfkusa… | 12.5% | 7,377 |
| usilxuppasem… | 10.4% | 1,444 |
| ewpakwlliwis… | 8.5% | 966 |
| missing | 7.7% | 4,218 |
| lmkebamcaacl… | 5.6% | 2,073 |
| epumfxlbckes… | 0.0% | 4 |
| fixdbufsefwo… | 0.0% | 2 |
| sddiedcslfsl… | 0.0% | 12 |