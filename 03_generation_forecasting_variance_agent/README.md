
   # Power Generation Forecasting & Variance Explanation Agent

**Context:** Greenko operates complex renewable-energy assets and enterprise workflows where AI can support operations, maintenance, safety, analytics, and decision support.

## Problem Statement

Forecast renewable generation and explain variance against actual generation.

## Expected Outcome

A forecasting baseline with variance analysis and a lightweight agent that explains forecast miss drivers such as irradiation, wind speed, temperature, and outage flags.

## Skills Tested

forecasting, regression, feature engineering, variance analysis, LLM summary, Python API, frontend

## Candidate Submission Requirements

1. Build the Python backend or notebook pipeline.
2. Build a simple React or HTML/CSS frontend to demonstrate the workflow.
3. Use the bundled dummy dataset as the minimum runnable dataset.
4. Add clear documentation explaining design choices, assumptions, model logic, limitations, and evaluation results.
5. Deploy the completed app to Railway.

## Dataset Note

This package includes a small self-contained dummy dataset so the assignment can run without internet access. Candidates may optionally extend the dataset using public sources such as Hugging Face datasets, Kaggle, Open Power System Data, NREL examples, or other open energy/NLP datasets, but the core submission must run using the bundled data.

## Suggested Deliverables

- Working Python implementation
- API endpoint or executable script
- UI prototype
- Model / retrieval / classification output examples
- Evaluation results
- Short explanation of production-readiness gaps

# Evaluation Rubric

Score each submission out of 100 using the same Greenko AI/ML intern screening philosophy.

| Area | Weight |
|---|---:|
| Problem understanding and energy-domain framing | 10 |
| Python implementation quality | 12 |
| AI/ML or GenAI method selection | 14 |
| Data preprocessing and feature engineering | 10 |
| RAG / retrieval / agentic workflow quality, where applicable | 12 |
| Evaluation metrics and validation approach | 12 |
| API/backend design | 8 |
| Frontend usability and explanation clarity | 8 |
| Documentation and reproducibility | 8 |
| Deployment readiness, including Railway deployment step | 6 |

Strong submissions should clearly explain assumptions, limitations, failure cases, and next steps for productionization.
