import sys
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

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

app = FastAPI(title='Power Generation Forecasting & Variance Explanation Agent')

model_service = ModelService()
variance_engine = VarianceEngine(model_service.df)
agent_service = AgentService(model_service, variance_engine)

class Query(BaseModel):
    text: str = ''

class PredictionRequest(BaseModel):
    irradiance_wm2: float
    wind_speed_ms: float
    temperature_c: float
    outage_flag: int

class VarianceRequest(BaseModel):
    date: Optional[str] = None
    irradiance_wm2: Optional[float] = None
    wind_speed_ms: Optional[float] = None
    temperature_c: Optional[float] = None
    outage_flag: Optional[int] = None
    forecast_mwh: Optional[float] = None
    actual_mwh: Optional[float] = None

@app.get('/api/health')
def health():
    return {
        'status': 'healthy',
        'records_loaded': len(model_service.df),
        'version': '1.0.0'
    }

@app.get('/api/demo')
def demo():
    return {
        'case_study': 'Power Generation Forecasting & Variance Explanation Agent',
        'status': 'active',
        'expected_outcome': 'A forecasting baseline with variance analysis and a lightweight agent that explains forecast miss drivers such as irradiation, wind speed, temperature, and outage flags.',
        'summary': model_service.get_summary_stats()
    }

@app.get('/api/data')
def get_data():
    records = variance_engine.get_all_decompositions()
    return {
        'count': len(records),
        'records': records
    }

@app.get('/api/metrics')
def get_metrics():
    return {
        'summary_statistics': model_service.get_summary_stats(),
        'model_benchmarks': model_service.metrics_summary,
        'feature_importances': model_service.feature_importances
    }

@app.post('/api/predict')
def predict_generation(req: PredictionRequest):
    return model_service.predict(
        irradiance_wm2=req.irradiance_wm2,
        wind_speed_ms=req.wind_speed_ms,
        temperature_c=req.temperature_c,
        outage_flag=req.outage_flag
    )

@app.post('/api/analyze-variance')
def analyze_variance(req: VarianceRequest):
    if req.date:
        matches = model_service.df[model_service.df['date'] == req.date]
        if matches.empty:
            raise HTTPException(status_code=404, detail=f"Date {req.date} not found")
        row = matches.iloc[0].to_dict()
        return variance_engine.decompose_record(row)
    
    if req.forecast_mwh is not None and req.actual_mwh is not None:
        mock_record = {
            'date': 'Custom Scenario',
            'irradiance_wm2': req.irradiance_wm2 or 600.0,
            'wind_speed_ms': req.wind_speed_ms or 7.0,
            'temperature_c': req.temperature_c or 30.0,
            'outage_flag': req.outage_flag or 0,
            'forecast_mwh': req.forecast_mwh,
            'actual_mwh': req.actual_mwh
        }
        return variance_engine.decompose_record(mock_record)

    raise HTTPException(status_code=400, detail="Either 'date' or complete custom telemetry must be provided")

@app.post('/api/run')
def run_agent(query: Query):
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
        'evaluation_metrics': model_service.metrics_summary
    }

if FRONTEND_DIR.exists():
    app.mount('/static', StaticFiles(directory=str(FRONTEND_DIR)), name='static')

    @app.get('/')
    def read_root():
        return FileResponse(str(FRONTEND_DIR / 'index.html'))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
