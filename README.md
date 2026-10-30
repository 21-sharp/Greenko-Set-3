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
│   │   ├── app.py                 # FastAPI application & REST endpoints
│   │   ├── model_service.py       # ML Pipeline, StandardScaler & feature engineering
│   │   ├── variance_engine.py     # 5-driver waterfall decomposition engine
│   │   └── agent_service.py       # 4-step RAG agent with domain SOPs
│   ├── data/
│   │   ├── generation_forecast.csv # 60 daily renewable telemetry records
│   │   └── knowledge_base.json     # External domain knowledge entries (SLDC & clipping)
│   ├── frontend/
│   │   └── index.html             # Operational Dashboard UI (Chart.js & interactive simulator)
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
*On Windows, you can also double-click `run_server.bat` inside `03_generation_forecasting_variance_agent`.*

### 3. Access Dashboard UI
Open your browser to: **`http://127.0.0.1:8000`**

---

## REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Service health status and dataset record count |
| `GET` | `/api/demo` | Problem statement overview & summary statistics |
| `GET` | `/api/data` | All 60 records with 5-component waterfall breakdowns |
| `GET` | `/api/metrics` | Model benchmarks (Train/Test MAE, RMSE, R²), feature importances & methodology |
| `POST` | `/api/predict` | Generation forecast inference for custom weather parameters |
| `POST` | `/api/analyze-variance` | Quantitative variance breakdown for a specific date or custom scenario |
| `POST` | `/api/run` | Natural language AI agent query returning reasoning traces & recommendations |

---

## Model Evaluation Benchmarks

Evaluation is conducted using a chronological train/test split (first 48 records for training, held-out 12 records for independent evaluation) to prevent look-ahead bias:

| Metric | Operational Baseline | Linear Ridge Baseline | Gradient Boosting (Train) | Gradient Boosting (Held-out Test) |
|---|:---:|:---:|:---:|:---:|
| **MAE** | 5.01 MWh | 5.36 MWh | 1.55 MWh | Evaluated via `/api/metrics` |
| **RMSE** | 6.84 MWh | 7.12 MWh | 1.98 MWh | Evaluated via `/api/metrics` |
| **R²** | 0.889 | 0.858 | 0.988 | Evaluated via `/api/metrics` |

---

## Production Deployment (Railway)

The application is fully configured for continuous deployment on Railway using:
- **`Procfile`**: `web: uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}`
- **`railway.json`**: Configured with NIXPACKS builder and automatic restarts on failure.
