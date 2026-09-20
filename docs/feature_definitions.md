# Feature Definitions

> Imported from the team Google Drive spreadsheet
> "Churn_Features_definitions". Two data files share these fields:
> the customer file (`ml_case_*_data.csv`) and the price-history file
> (`ml_case_*_hist_data.csv`).

## Customer file (one row per customer)

| Field | Description |
| --- | --- |
| id | contact id (join key) |
| activity_new | category of the company's activity (hashed, high cardinality) |
| campaign_disc_ele | code of the last electricity campaign subscribed to |
| channel_sales | code of the sales channel (hashed) |
| cons_12m | electricity consumption over the past 12 months (kWh) |
| cons_gas_12m | gas consumption over the past 12 months (kWh) |
| cons_last_month | electricity consumption in the last month (kWh) |
| date_activ | activation date of the current contract |
| date_end | end date of the current contract |
| date_first_activ | date of the client's first-ever contract |
| date_modif_prod | date of the last product modification |
| date_renewal | date of the next contract renewal |
| forecast_base_bill_ele | forecasted electricity bill baseline, next month |
| forecast_base_bill_year | forecasted electricity bill baseline, calendar year |
| forecast_bill_12m | forecasted electricity bill baseline, 12 months |
| forecast_cons | forecasted electricity consumption, next month |
| forecast_cons_12m | forecasted electricity consumption, next 12 months |
| forecast_cons_year | forecasted electricity consumption, calendar year |
| forecast_discount_energy | forecasted value of the current discount |
| forecast_meter_rent_12m | forecasted meter-rental bill, next 12 months |
| forecast_price_energy_p1 | forecasted energy price, period 1 |
| forecast_price_energy_p2 | forecasted energy price, period 2 |
| forecast_price_pow_p1 | forecasted power price, period 1 |
| has_gas | flag: client is also a gas client (`t`/`f`) |
| imp_cons | current paid (last invoiced) consumption |
| margin_gross_pow_ele | gross margin on power subscription |
| margin_net_pow_ele | net margin on power subscription |
| nb_prod_act | number of active products and services |
| net_margin | total net margin |
| num_years_antig | client antiquity (tenure, in years) |
| origin_up | code of the first electricity campaign subscribed to |
| pow_max | subscribed power (kW) |

## Price-history file (about 12 rows per customer, monthly for 2015)

| Field | Description |
| --- | --- |
| id | contact id (join key) |
| price_date | reference month |
| price_p1_var | energy price, period 1 (per kWh) |
| price_p2_var | energy price, period 2 |
| price_p3_var | energy price, period 3 |
| price_p1_fix | power price, period 1 (per kW) |
| price_p2_fix | power price, period 2 |
| price_p3_fix | power price, period 3 |

## Label file

| Field | Description |
| --- | --- |
| id | contact id |
| churn | 1 = churned within the 3-month window, 0 = stayed |

## Industry insight (why features matter)

- **Consumption** (`cons_*`): a proxy for customer size and value. A sudden
  drop can signal a customer that is leaving or winding down.
- **has_gas / nb_prod_act**: dual-fuel and multi-product customers are
  "stickier" and churn less.
- **date_end / date_renewal**: the window before contract end or renewal is
  when customers shop around. Time-to-renewal is a strong signal.
- **num_years_antig**: longer-tenured customers are usually more loyal.
- **forecast_* bills**: a high expected bill can drive churn before the bill
  even arrives.
- **margin_* / net_margin**: needed for the discount economics — save
  profitable customers, not unprofitable ones.
- **pow_max**: subscribed capacity; large for machinery-heavy businesses.
- **price_p*_var / price_p*_fix**: time-of-use energy and power prices.
  Price level, the peak/off-peak spread, and the price **change** across
  2015 are the classic BCG churn drivers.
