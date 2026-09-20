# Grading Rubric — 7 Presentations

> Imported from the team Google Drive file "Team Project Rubric.xlsx".
> The course (InnovatiCS) grades the project as seven staged
> presentations. Each stage maps to a phase of the CRISP-DM cycle.

| # | Theme | What must be shown |
| --- | --- | --- |
| 1 | Framing the business problem | Team intro; problem statement; objective; as-is and to-be state; business and technical success measures; entity (unit of analysis); target; 3 descriptive + 2 predictive questions; an ABT (analytics base table); team communication channel |
| 2 | Data understanding & visualization | EDA; data visualization; statistical/descriptive analysis; answers to the P1 questions; a **data quality report** (all data problems + planned fixes); appendix |
| 3 | Data preparation | Solutions to the data problems; chosen modeling technique and why; how data was made model-ready; appendix |
| 4 | Machine learning | Which ML techniques and why; model comparison; chosen model and justification; appendix |
| 5 | Deep learning | A deep-learning approach; model comparison; chosen model; appendix |
| 6 | Generative AI | A GenAI approach; model comparison; chosen model; appendix |
| 7 | Insights, action plan, recommendations | End-to-end story; action plan; client recommendations with evidence; appendix |

## Implications for the codebase

- The project is not only a model. It must produce **evidence** for a
  business audience at each stage.
- Stage 1 needs a clean **ABT** (one row per customer) — this is exactly the
  merge step that the current code gets wrong.
- Stage 2 needs a repeatable EDA and a written data-quality report.
- Stages 4–6 expect model comparison across classical ML, deep learning, and
  a GenAI method.
- Stage 7 needs the discount economic analysis and the scored test set.
