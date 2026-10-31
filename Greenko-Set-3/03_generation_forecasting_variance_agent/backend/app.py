import sys
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("greenko.app")

BASE_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / 'frontend'

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    from backend.model_service import ModelService
    from backend.variance_engine import VarianceEngine
    from backend.agent_service import AgentService
except ImportError:
    from model_service import ModelService
    from variance_engine import VarianceEngine
    from agent_service import AgentService

logger.info("Initializing ModelService...")
model_service = ModelService()
logger.info("Initializing VarianceEngine with training calibration...")
variance_engine = VarianceEngine(model_service.df, model_service.train_dates)
logger.info("Initializing AgentService...")
agent_service = AgentService(model_service, variance_engine)
logger.info("All services initialized successfully.")


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    logger.info("=" * 60)
    logger.info("Greenko Power Generation Forecasting & Variance Agent")
    logger.info("Status: Operational | Version: 2.2.0")
    logger.info("Total Records: %d across %d unique dates", len(model_service.df), model_service.df['date'].nunique())
    logger.info("Selected ML Model: %s", model_service.selected_model_name)
    logger.info("Knowledge Base SOPs: %d documents indexed", len(agent_service.retriever.documents))
    logger.info("=" * 60)
    yield


app = FastAPI(
    title="Power Generation Forecasting & Variance Explanation Agent",
    description="Greenko renewable energy forecasting and five-driver variance attribution system.",
    version="2.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    # Same-origin deployment works without CORS. Add explicit browser origins
    # through ALLOWED_ORIGINS (comma-separated); wildcard + credentials is unsafe.
    allow_origins=[origin.strip() for origin in os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
    ).split(",") if origin.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check server logs for details."}
    )


class Query(BaseModel):
    text: str = Field(default='', description="Natural language query about variance or generation")


class PredictionRequest(BaseModel):
    irradiance_wm2: float = Field(..., ge=0.0, le=1400.0, description="Global Horizontal Irradiance (W/m²)")
    wind_speed_ms: float = Field(..., ge=0.0, le=25.0, description="Hub-height wind speed (m/s)")
    temperature_c: float = Field(..., ge=-10.0, le=55.0, description="Ambient temperature (°C)")
    outage_flag: int = Field(..., ge=0, le=1, description="0 = Normal, 1 = Outage / Trip")
    asset_id: Optional[str] = Field(default="Asset_1", description="Asset identifier (Asset_1 or Asset_2)")

    @field_validator('asset_id')
    @classmethod
    def validate_asset(cls, value):
        if value not in (None, "Asset_1", "Asset_2"):
            raise ValueError("asset_id must be Asset_1 or Asset_2")
        return value

    @field_validator('outage_flag')
    @classmethod
    def validate_outage(cls, v):
        if v not in (0, 1):
            raise ValueError("outage_flag must be 0 or 1")
        return v


class VarianceRequest(BaseModel):
    date: Optional[str] = Field(None, description="Target date in YYYY-MM-DD format")
    asset_id: Optional[str] = Field(None, description="Optional asset filter")
    irradiance_wm2: Optional[float] = Field(None, ge=0.0, le=1400.0)
    wind_speed_ms: Optional[float] = Field(None, ge=0.0, le=25.0)
    temperature_c: Optional[float] = Field(None, ge=-10.0, le=55.0)
    outage_flag: Optional[int] = Field(None, ge=0, le=1)
    forecast_mwh: Optional[float] = Field(None, ge=0.0)
    actual_mwh: Optional[float] = Field(None, ge=0.0)


@app.get('/api/health', tags=["System"])
def health():
    ready = (
        model_service.selected_pipeline is not None and
        model_service.df is not None and
        len(model_service.df) > 0
    )
    return {
        'status': 'healthy' if ready else 'degraded',
        'model_ready': ready,
        'records_loaded': len(model_service.df) if model_service.df is not None else 0,
        'unique_dates': int(model_service.df['date'].nunique()) if model_service.df is not None else 0,
        'models_trained': ['Ridge Linear Baseline', 'Random Forest Regressor', 'Gradient Boosting Regressor'],
        'selected_model': model_service.selected_model_name,
        'version': '2.2.0'
    }


@app.get('/api/demo', tags=["Data"])
def demo():
    return {
        'case_study': 'Power Generation Forecasting & Variance Explanation Agent',
        'status': 'active',
        'summary': model_service.get_summary_stats(),
        'selected_model': model_service.selected_model_name,
        'models_available': ['Operational Baseline', 'Ridge Linear Baseline', 'Random Forest Regressor', 'Gradient Boosting Regressor'],
        'variance_drivers': ['Outage', 'Solar Irradiance', 'Wind Speed', 'Temperature', 'Residual'],
        'rag_retriever': 'TF-IDF Vectorizer with n-grams (1,2) and Cosine Similarity'
    }


@app.get('/api/data', tags=["Data"])
def get_data():
    try:
        records = variance_engine.get_all_decompositions()
        return {
            'count': len(records),
            'unique_dates': int(model_service.df['date'].nunique()),
            'records': records
        }
    except Exception as e:
        logger.error("Error computing decompositions: %s", e)
        raise HTTPException(status_code=500, detail="Failed to compute decompositions")


@app.get('/api/metrics', tags=["Model"])
def get_metrics():
    return {
        'summary_statistics': model_service.get_summary_stats(),
        'model_benchmarks': model_service.metrics_summary,
        'feature_importances': model_service.feature_importances,
        'driver_summary': variance_engine.get_aggregated_driver_summary(),
        'variance_attribution_methodology': (
            "Variance is decomposed into five components with strict mathematical reconciliation: "
            "(1) Outage impact: estimated generation capacity lost when outage_flag=1. "
            "(2) Solar irradiance: training OLS sensitivity applied to irradiance deviation from mean. "
            "(3) Wind speed: training OLS sensitivity in cubic velocity space (v^3). "
            "(4) Temperature derate: -0.24 MWh/°C above 25°C STC (-0.4%/°C crystalline Si standard). "
            "(5) Unmodeled residual: captures inverter clipping, SLDC grid curtailment, and sensor drift. "
            "Sum of all five components strictly equals total generation variance (Actual - Forecast)."
        ),
        'rag_info': {
            'retrieval_method': 'TF-IDF with n-grams (1,2) and Cosine Similarity',
            'knowledge_base_documents': len(agent_service.retriever.documents),
            'retrieval_evaluation': agent_service.retriever.evaluate_retrieval_benchmark(),
            'top_k': 3,
            'fallback': 'Deterministic structured response grounded in available data',
            'llm_explanation': {
                'enabled': bool(os.getenv('OPENAI_API_KEY', '').strip()),
                'provider': 'OpenAI-compatible Chat Completions API',
                'model': os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
                'role': 'Narrative synthesis only; numerical tool outputs remain deterministic'
            }
        }
    }


@app.post('/api/predict', tags=["Model"])
def predict_generation(req: PredictionRequest):
    try:
        return model_service.predict(
            irradiance_wm2=req.irradiance_wm2,
            wind_speed_ms=req.wind_speed_ms,
            temperature_c=req.temperature_c,
            outage_flag=req.outage_flag,
            asset_id=req.asset_id or "Asset_1"
        )
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail="Prediction failed; inspect server logs using the request timestamp.")


@app.post('/api/analyze-variance', tags=["Variance"])
def analyze_variance(req: VarianceRequest):
    if req.date:
        matches = model_service.df[model_service.df['date'] == req.date]
        if req.asset_id:
            matches = matches[matches['asset_id'] == req.asset_id]

        if matches.empty:
            available = sorted(model_service.df['date'].unique().tolist())
            raise HTTPException(
                status_code=404,
                detail={
                    "message": f"Date '{req.date}' not found in dataset",
                    "available_dates_sample": available[:6],
                    "total_available_dates": len(available)
                }
            )

        decompositions = [
            variance_engine.decompose_record(record)
            for record in matches.to_dict(orient='records')
        ]
        if len(decompositions) == 1:
            return decompositions[0]
        return {
            'date': req.date,
            'count': len(decompositions),
            'decompositions': decompositions
        }

    if req.forecast_mwh is not None and req.actual_mwh is not None:
        custom_record = {
            'date': 'Custom Scenario',
            'asset_id': req.asset_id or 'Custom_Asset',
            'irradiance_wm2': req.irradiance_wm2 if req.irradiance_wm2 is not None else 600.0,
            'wind_speed_ms': req.wind_speed_ms if req.wind_speed_ms is not None else 7.0,
            'temperature_c': req.temperature_c if req.temperature_c is not None else 30.0,
            'outage_flag': req.outage_flag if req.outage_flag is not None else 0,
            'forecast_mwh': req.forecast_mwh,
            'actual_mwh': req.actual_mwh
        }
        return variance_engine.decompose_record(custom_record)

    raise HTTPException(
        status_code=400,
        detail="Either 'date' or both 'forecast_mwh' and 'actual_mwh' must be provided"
    )


@app.post('/api/run', tags=["Agent"])
def run_agent(query: Query):
    if not query.text.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty")

    try:
        agent_output = agent_service.query(query.text)
        return {
            'input': query.text,
            'result': agent_output.get('summary', ''),
            'category': agent_output.get('category', ''),
            'target_date': agent_output.get('target_date', ''),
            'drivers': agent_output.get('drivers', {}),
            'waterfall': agent_output.get('waterfall', []),
            'retrieval_evidence': agent_output.get('retrieval_evidence', []),
            'reasoning_steps': agent_output.get('reasoning_steps', []),
            'recommendations': agent_output.get('recommendations', []),
            'limitations': agent_output.get('limitations', []),
            'confidence': agent_output.get('confidence', 'medium'),
            'selected_model': model_service.selected_model_name,
            'evaluation_metrics': model_service.metrics_summary
        }
    except Exception as e:
        logger.error("Agent query error: %s", e)
        raise HTTPException(status_code=500, detail="Agent query failed; inspect server logs using the request timestamp.")


if FRONTEND_DIR.exists():
    app.mount('/static', StaticFiles(directory=str(FRONTEND_DIR)), name='static')

    @app.get('/', tags=["Frontend"])
    def read_root():
        return FileResponse(str(FRONTEND_DIR / 'index.html'))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
