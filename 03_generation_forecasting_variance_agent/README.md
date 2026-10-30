# Power Generation Forecasting & Variance Explanation Agent

## Project Overview

This system forecasts renewable energy generation and explains variance between forecasted and actual output using a five-component attribution waterfall. It combines a Gradient Boosting regression model with a lightweight retrieval-augmented agent backed by Greenko standard operating procedure knowledge entries.

The agent ingests daily telemetry (irradiance, wind speed, temperature, outage flag) and decomposes any forecast miss into quantified driver contributions: forced outage, solar irradiance deviation, wind resource variation, ambient temperature derating, and an unmodeled residual that captures inverter clipping or SLDC curtailment not reflected in the schedule.

## Quick Start

Install dependencies:

```
pip install -r requirements.txt
```

Start the server:

```
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open the UI at http://127.0.0.1:8000

On Windows you can also run `run_server.bat` from the project root.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/health | Service health check |
| GET | /api/demo | Case study metadata and summary statistics |
| GET | /api/data | All 60 records with variance decompositions |
| GET | /api/metrics | Model benchmarks, feature importances, and attribution methodology |
| POST | /api/predict | Point forecast given environmental inputs |
| POST | /api/analyze-variance | Variance waterfall for a specific date or custom scenario |
| POST | /api/run | Natural language agent query with reasoning steps and recommendations |

## Evaluation Results

Models were trained on the first 48 chronological records and evaluated on the final 12 held-out records.

| Metric | Operational Baseline | Ridge Linear | Gradient Boosting Train | Gradient Boosting Test |
|--------|---------------------|--------------|------------------------|------------------------|
| MAE (MWh) | see /api/metrics | see /api/metrics | see /api/metrics | see /api/metrics |
| RMSE (MWh) | see /api/metrics | see /api/metrics | see /api/metrics | see /api/metrics |
| R2 | see /api/metrics | see /api/metrics | see /api/metrics | see /api/metrics |

Live values are returned by the `/api/metrics` endpoint under `model_benchmarks.gradient_boosting_model` which exposes `train_mae`, `train_rmse`, `train_r2`, `test_mae`, `test_rmse`, `test_r2`, `cv_mae`, `cv_rmse`, and `cv_r2`.

## Assumptions and Limitations

- The dataset contains 60 synthetic daily records spanning August 2026 and is intended solely for demonstration. It does not reflect actual Greenko plant telemetry.
- All features are aggregated at daily granularity. Sub-hourly ramp events, intra-day cloud transients, and 15-minute scheduling block dynamics are not captured.
- The `outage_flag` is a binary indicator. It does not distinguish between a forced inverter trip, a scheduled maintenance window, and an SLDC grid curtailment instruction.
- The variance attribution coefficients (`solar_coeff`, `wind_coeff`) are derived from OLS regression on the non-outage subset of the training data. They represent average sensitivities across the dataset and may not generalise to individual asset types or seasonal conditions.
- StandardScaler normalisation is fitted exclusively on training rows (rows 0-47) and applied to both the test set and live inference to prevent data leakage.
- The knowledge base is loaded from `data/knowledge_base.json` at startup and extended with the four hard-coded SOP entries. Additional domain entries can be appended to that file without restarting the service.
- The deployment uses the `Procfile` and `railway.json` with `uvicorn` as the ASGI application server.

## Deployment

The application is deployed using the bundled `Procfile` and `railway.json` for one-click deployment to Railway:

```
web: uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}
```
