# Problem Statement — Modeling Churn in an Energy Company

> Source: BCG Gamma / InnovatiCS "Project #6" brief.
> Imported from the team Google Drive folder
> "InnovatiCS Group 3 - Modeling Churn in an Energy Company".

## Scenario

The client is **PowerCo**, a large utility that supplies gas and electricity
to corporate, SME, and residential customers. After the European energy
market opened, PowerCo lost customers faster than the industry average. The
loss is worst in the **SME** division, so SME is the first priority.

The head of the SME division has one hypothesis: customers leave for cheaper
providers. The first planned action is to offer a **20% discount** to the
customers most likely to leave.

## Task

1. Build a model that predicts which SME customers will churn.
2. Recommend the commercial actions to take from the model output.
3. Answer three client questions:
   1. Which variables best explain churn?
   2. Is subscribed power (`pow_max`) correlated with consumption?
   3. Is there a link between sales channel (`channel_sales`) and churn?

## Data framing (important)

- Training features describe SME customers as of **January 2016**.
- The label says whether the customer churned by **March 2016**
  (a 3-month window).
- Price history from **2015** is provided for the same customers.
- The client cares a lot about *how the problem is framed* for training.

## Deliverable on the test set

- Score every customer in the verification (test) set.
- Rank them in descending order of churn propensity.
- Classify each customer: `1` = predicted to churn, `0` = predicted to stay.
- Fill the result template (`ml_case_test_output_template.csv`).

## Economics of the 20% discount

- Assume every customer who is offered the discount accepts it.
- By regulation, PowerCo cannot raise the price for one year after acceptance.
- So a discount given to a customer who would **not** have left is pure lost
  revenue. The discount must be targeted, not offered to everyone.
- The model output must be combined with **margin** data. It makes no sense
  to give a large discount to a low-margin customer.

## Notes from the brief

- Some text fields are hashed for privacy. Their business meaning is lost,
  but they may still have predictive power.
- Use descriptive statistics and visualization before modeling.
- Testing several algorithms is encouraged; describe each one simply.
- Any language except Excel-only is allowed (the data is large).
