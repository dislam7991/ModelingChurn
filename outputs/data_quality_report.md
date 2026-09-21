# Data-Quality Report (training data)

- Customers: **16,096** (one row per id).
- Price-history rows: **193,002** (~12.0 months per customer, Jan–Dec 2015).
- Churn rate: **9.91%** (1,595 churners of 16,096) — imbalanced.

## Missing values (columns with any missing)

| Column | % missing | Handling |
| --- | --- | --- |
| campaign_disc_ele | 100.0 | 100% empty — **dropped** |
| forecast_base_bill_year | 78.2 | 78% missing — imputed (median) + `has_forecast_bill` flag |
| forecast_bill_12m | 78.2 | 78% missing — imputed + flag |
| forecast_cons | 78.2 | 78% missing — imputed (median) |
| forecast_base_bill_ele | 78.2 | 78% missing — imputed (median) |
| date_first_activ | 78.2 | 78% missing — fall back to `date_activ` for lifespan |
| activity_new | 59.3 | 59% missing, 419 categories — capped to top-10 + other/missing |
| channel_sales | 26.2 | 26% missing — imputed as 'missing' category |
| date_modif_prod | 1.0 | imputed (median / missing category) |
| forecast_price_energy_p2 | 0.8 | imputed (median / missing category) |
| forecast_price_pow_p1 | 0.8 | imputed (median / missing category) |
| forecast_discount_energy | 0.8 | imputed (median / missing category) |
| forecast_price_energy_p1 | 0.8 | imputed (median / missing category) |
| origin_up | 0.5 | imputed (median / missing category) |
| date_renewal | 0.2 | imputed (median / missing category) |
| margin_net_pow_ele | 0.1 | imputed (median / missing category) |
| margin_gross_pow_ele | 0.1 | imputed (median / missing category) |
| net_margin | 0.1 | imputed (median / missing category) |

## Implausible negative values

| Column | # negatives | Handling |
| --- | --- | --- |
| cons_12m | 27 | set to missing (consumption cannot be negative) |
| cons_gas_12m | 6 | set to missing (consumption cannot be negative) |
| cons_last_month | 46 | set to missing (consumption cannot be negative) |
| imp_cons | 27 | set to missing (consumption cannot be negative) |
| forecast_cons_12m | 41 | set to missing (consumption cannot be negative) |
| forecast_cons_year | 25 | set to missing (consumption cannot be negative) |
| margin_gross_pow_ele | 1139 | **kept** (a margin can legitimately be negative) |
| margin_net_pow_ele | 1187 | **kept** (a margin can legitimately be negative) |
| net_margin | 97 | **kept** (a margin can legitimately be negative) |

## Structural issue fixed

The original pipeline left-joined the ~12-row price history directly onto the customer table, exploding it ~12x and breaking the per-customer aggregation. The new `data_prep.build_abt` aggregates the price history per `id` first (mean/std/min/max, year change, peak–offpeak spread), then joins one row per customer. Row count is asserted to stay at one per id.
