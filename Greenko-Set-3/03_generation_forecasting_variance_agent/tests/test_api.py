import pytest
from pathlib import Path
import sys
from pydantic import ValidationError
from fastapi import HTTPException

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.app import (
    app, health, demo, get_data, get_metrics,
    predict_generation, analyze_variance, run_agent,
    PredictionRequest, VarianceRequest, Query
)


class TestAPIEndpoints:
    def test_health_check(self):
        res = health()
        assert res['status'] == 'healthy'
        assert res['model_ready'] is True
        assert res['records_loaded'] == 60
        assert res['unique_dates'] == 30
        assert res['selected_model'] == 'Ridge Linear Baseline'
        assert 'version' in res

    def test_demo_endpoint(self):
        res = demo()
        assert res['status'] == 'active'
        assert 'case_study' in res
        assert 'summary' in res
        assert res['summary']['total_records'] == 60
        assert res['selected_model'] == 'Ridge Linear Baseline'
        assert len(res['models_available']) == 4

    def test_get_data_endpoint(self):
        res = get_data()
        assert res['count'] == 60
        assert res['unique_dates'] == 30
        assert len(res['records']) == 60
        first = res['records'][0]
        assert 'drivers' in first
        assert 'waterfall' in first
        assert 'driver_reconciliation' in first
        assert 'asset_id' in first
        assert first['driver_reconciliation']['reconciled'] is True

    def test_get_metrics_endpoint(self):
        res = get_metrics()
        assert 'summary_statistics' in res
        assert 'model_benchmarks' in res
        assert 'feature_importances' in res
        assert 'driver_summary' in res
        assert 'variance_attribution_methodology' in res
        assert 'rag_info' in res
        assert res['model_benchmarks']['model_selection']['selected_model'] == 'Ridge Linear Baseline'
        assert 'residual_analysis' in res['model_benchmarks']['ridge_linear_baseline']['test']

    def test_predict_endpoint_valid(self):
        req = PredictionRequest(
            irradiance_wm2=650.0,
            wind_speed_ms=7.5,
            temperature_c=32.0,
            outage_flag=0,
            asset_id='Asset_1'
        )
        res = predict_generation(req)
        assert 'predicted_mwh' in res
        assert 'random_forest_mwh' in res
        assert 'gradient_boosting_mwh' in res
        assert 'baseline_linear_mwh' in res
        assert res['predicted_mwh'] > 0
        assert res['outage_active'] is False
        assert res['selected_model'] == 'Ridge Linear Baseline'

    def test_predict_boundary_validation(self):
        with pytest.raises(ValidationError):
            PredictionRequest(
                irradiance_wm2=2500.0,
                wind_speed_ms=7.5,
                temperature_c=32.0,
                outage_flag=0
            )

        with pytest.raises(ValidationError):
            PredictionRequest(
                irradiance_wm2=600.0,
                wind_speed_ms=45.0,
                temperature_c=32.0,
                outage_flag=0
            )

        with pytest.raises(ValidationError):
            PredictionRequest(
                irradiance_wm2=600.0,
                wind_speed_ms=7.0,
                temperature_c=30.0,
                outage_flag=3
            )

    def test_analyze_variance_by_date(self):
        req = VarianceRequest(date="2026-08-15")
        res = analyze_variance(req)
        if 'decompositions' in res:
            assert len(res['decompositions']) == 2  # 2 assets on 2026-08-15
        else:
            assert 'outage_mwh' in res['drivers']

    def test_analyze_variance_with_asset_id(self):
        req = VarianceRequest(date="2026-08-15", asset_id="Asset_1")
        res = analyze_variance(req)
        assert res['asset_id'] == 'Asset_1'
        assert res['drivers']['outage_mwh'] < 0

    def test_analyze_variance_date_not_found(self):
        req = VarianceRequest(date="1999-01-01")
        with pytest.raises(HTTPException) as excinfo:
            analyze_variance(req)
        assert excinfo.value.status_code == 404

    def test_analyze_variance_custom_scenario(self):
        req = VarianceRequest(
            forecast_mwh=65.0,
            actual_mwh=50.0,
            irradiance_wm2=500.0,
            wind_speed_ms=6.0,
            temperature_c=35.0,
            outage_flag=0,
            asset_id='Custom_Block'
        )
        res = analyze_variance(req)
        assert res['total_variance_mwh'] == -15.0
        assert res['driver_reconciliation']['reconciled'] is True

    def test_analyze_variance_empty_request(self):
        req = VarianceRequest()
        with pytest.raises(HTTPException) as excinfo:
            analyze_variance(req)
        assert excinfo.value.status_code == 400

    def test_run_agent_endpoint(self):
        q = Query(text="Explain the variance on 2026-08-15")
        res = run_agent(q)
        assert 'result' in res
        assert 'reasoning_steps' in res
        assert 'recommendations' in res
        assert 'retrieval_evidence' in res
        assert len(res['reasoning_steps']) > 0

    def test_run_agent_empty_query(self):
        q = Query(text="   ")
        with pytest.raises(HTTPException) as excinfo:
            run_agent(q)
        assert excinfo.value.status_code == 400

    def test_app_metadata(self):
        assert app.title == "Power Generation Forecasting & Variance Explanation Agent"
        assert app.version == "2.2.0"


def test_prediction_rejects_unknown_asset_id():
    with pytest.raises(ValidationError, match="asset_id must be Asset_1 or Asset_2"):
        PredictionRequest(
            irradiance_wm2=500, wind_speed_ms=7, temperature_c=30,
            outage_flag=0, asset_id="unverified_asset"
        )
