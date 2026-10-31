# Greenko Set 3: Power Generation Forecasting & Variance Explanation Agent

## Project Overview

This repository contains the complete implementation for **Greenko Set 3: Power Generation Forecasting & Variance Explanation Agent** located in [`03_generation_forecasting_variance_agent`](./03_generation_forecasting_variance_agent).

The system forecasts renewable energy generation and decomposes variance between scheduled and actual power output into physical and operational drivers:
1. **Operational Outages**: Inverter protection trips and SLDC grid curtailment.
2. **Solar Irradiance Variance**: Actual vs expected insolation response.
3. **Wind Resource Variations**: Cubic aerodynamic power curve dynamics ($P \propto v^3$).
4. **Thermal Derating**: Silicon efficiency degradation above Standard Test Conditions (25°C at -0.4%/°C).
5. **Unmodeled Residuals**: Inverter clipping and local grid curtailments.

---

## Directory Structure

```
Greenko-Set-3/
├── 03_generation_forecasting_variance_agent/
│   ├── backend/
│   │   ├── app.py                 # FastAPI application & REST endpoints with lifespan
│   │   ├── model_service.py       # ML Pipeline, StandardScaler & physics feature engineering
│   │   ├── variance_engine.py     # 5-driver waterfall decomposition engine with strict reconciliation
│   │   └── agent_service.py       # Offline RAG agent with TF-IDF retrieval & domain SOPs
│   ├── data/
│   │   ├── generation_forecast.csv # 60 daily renewable telemetry records
│   │   └── knowledge_base.json     # External domain knowledge entries (SLDC & clipping)
│   ├── frontend/
│   │   └── index.html             # Operational Dashboard UI (Chart.js & interactive simulator)
│   ├── tests/
│   │   ├── run_all_tests.py       # Automated test suite runner
│   │   ├── test_agent_service.py  # 13 RAG agent tests
│   │   ├── test_api.py            # 14 FastAPI endpoint tests
│   │   ├── test_model_service.py  # 14 ML & preprocessing tests
│   │   └── test_variance_engine.py# 10 variance attribution tests
│   ├── Procfile                   # Railway deployment process command
│   ├── railway.json               # Railway Nixpacks deployment configuration
│   ├── requirements.txt           # Python dependencies
│   ├── run_server.bat             # Windows one-click local server launcher
│   └── README.md                  # Comprehensive component documentation
└── README.md                      # Root repository documentation
```

---

## Quick Start

### 1. Install Dependencies
```bash
cd 03_generation_forecasting_variance_agent
pip install -r requirements.txt
```

### 2. Start Application Server
```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```


### 3. Run Automated Tests
```bash
python tests/run_all_tests.py
```


### 4. Access Dashboard UI
Open your browser to: **`http://127.0.0.1:8000`**

---

## REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Service health status and dataset record count |
| `GET` | `/api/demo` | Problem statement overview & summary statistics |
| `GET` | `/api/data` | All 60 records with 5-component waterfall breakdowns |
| `GET` | `/api/metrics` | Model benchmarks (Train/Test MAE, RMSE, R²), feature importances & methodology |
| `POST` | `/api/predict` | Generation forecast inference for custom weather parameters across all models |
| `POST` | `/api/analyze-variance` | Quantitative variance breakdown for a specific date or custom scenario |
| `POST` | `/api/run` | Natural language AI agent query returning reasoning traces, citations & recommendations |

---

## Model Evaluation Benchmarks

Evaluation is conducted using a chronological date-grouped train/test split (first 24 unique dates for training across both assets = 48 records, held-out 6 dates = 12 records for independent out-of-sample evaluation) and 4-fold TimeSeriesSplit cross-validation inside `sklearn.pipeline.Pipeline` to strictly prevent look-ahead bias and data leakage:

| Metric | Operational Baseline | **Linear Ridge (Selected ML)** | Random Forest Regressor | Gradient Boosting Regressor |
|---|:---:|:---:|:---:|:---:|
| **Test MAE** | 4.604 MWh | **6.286 MWh** | 6.582 MWh | 6.601 MWh |
| **Test RMSE** | 5.169 MWh | **7.895 MWh** | 7.631 MWh | 7.823 MWh |
| **Test R²** | 0.859 | **0.670** | 0.692 | 0.676 |
| **Test MAPE** | 7.92% | **11.61%** | 10.74% | 11.55% |
| **Train MAE** | 5.106 MWh | **4.986 MWh** | 2.980 MWh | 1.079 MWh |
| **CV MAE** | N/A | **6.466 MWh** | 10.064 MWh | 9.058 MWh |

**Model selection:** Ridge Linear Regressor achieves the lowest held-out Test MAE and the lowest CV MAE, indicating superior generalisation on this 48-record training sample. Random Forest and Gradient Boosting show high CV variance ($> 9\text{ MWh}$), consistent with ensemble instability at small data scale.
