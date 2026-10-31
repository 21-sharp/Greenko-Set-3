import pytest
import pandas as pd
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.model_service import ModelService
from backend.variance_engine import VarianceEngine


@pytest.fixture(scope="module")
def engine():
    ms = ModelService()
    return VarianceEngine(ms.df, ms.train_dates)


class TestVarianceEngine:
    def test_strict_reconciliation_all_records(self, engine):
        all_decomps = engine.get_all_decompositions()
        assert len(all_decomps) == 60

        for d in all_decomps:
            drivers = d['drivers']
            driver_sum = round(
                drivers['outage_mwh'] +
                drivers['irradiance_mwh'] +
                drivers['wind_mwh'] +
                drivers['temperature_mwh'] +
                drivers['residual_mwh'],
                2
            )
            total_var = round(d['total_variance_mwh'], 2)
            assert abs(driver_sum - total_var) < 0.05, f"Reconciliation failed on {d['date']}"
            assert d['driver_reconciliation']['reconciled'] is True

    def test_training_only_calibration(self, engine):
        assert engine.solar_coeff is not None
        assert engine.wind_coeff is not None
        assert engine.mean_irradiance > 0
        assert engine.mean_wind_cubed > 0

    def test_temperature_driver_uses_calibrated_coefficient(self, engine):
        sample = {
            'date': '2026-08-15', 'irradiance_wm2': engine.mean_irradiance,
            'wind_speed_ms': 8.0, 'temperature_c': 35.0, 'outage_flag': 0,
            'forecast_mwh': 50.0, 'actual_mwh': 50.0
        }
        result = engine.decompose_record(sample)
        expected = round(engine.temp_coeff * 10.0, 2)
        assert result['drivers']['temperature_mwh'] == expected
        metadata = engine.get_aggregated_driver_summary()['calibration_metadata']
        assert metadata['temp_coefficient_per_degC'] == round(engine.temp_coeff, 5)

    def test_normal_operation_zero_outage(self, engine):
        sample = {
            'date': '2026-08-01',
            'irradiance_wm2': 633.7,
            'wind_speed_ms': 10.81,
            'temperature_c': 31.5,
            'outage_flag': 0,
            'forecast_mwh': 71.93,
            'actual_mwh': 71.23,
            'asset_id': 'Asset_1'
        }
        res = engine.decompose_record(sample)
        assert res['drivers']['outage_mwh'] == 0.0
        assert res['total_variance_mwh'] == -0.7
        assert res['driver_reconciliation']['reconciled'] is True

    def test_outage_event_attribution(self, engine):
        sample = {
            'date': '2026-08-15',
            'irradiance_wm2': 572.9,
            'wind_speed_ms': 4.59,
            'temperature_c': 35.4,
            'outage_flag': 1,
            'forecast_mwh': 21.36,
            'actual_mwh': 4.19,
            'asset_id': 'Asset_1'
        }
        res = engine.decompose_record(sample)
        assert res['drivers']['outage_mwh'] < 0.0
        assert 'Outage' in res['category']
        assert res['driver_reconciliation']['reconciled'] is True

    def test_negative_variance_edge_cases(self, engine):
        sample = {
            'date': '2026-08-99',
            'irradiance_wm2': 200.0,
            'wind_speed_ms': 2.0,
            'temperature_c': 42.0,
            'outage_flag': 0,
            'forecast_mwh': 80.0,
            'actual_mwh': 20.0
        }
        res = engine.decompose_record(sample)
        assert res['total_variance_mwh'] == -60.0
        assert res['drivers']['temperature_mwh'] < 0.0
        assert res['driver_reconciliation']['reconciled'] is True

    def test_positive_variance_edge_cases(self, engine):
        sample = {
            'date': '2026-08-99',
            'irradiance_wm2': 950.0,
            'wind_speed_ms': 13.0,
            'temperature_c': 24.0,
            'outage_flag': 0,
            'forecast_mwh': 40.0,
            'actual_mwh': 85.0
        }
        res = engine.decompose_record(sample)
        assert res['total_variance_mwh'] == 45.0
        assert res['drivers']['wind_mwh'] > 0.0
        assert res['drivers']['temperature_mwh'] == 0.0
        assert res['driver_reconciliation']['reconciled'] is True

    def test_cubic_wind_consistency(self, engine):
        rec_low_wind = {'irradiance_wm2': 600, 'wind_speed_ms': 5.0, 'temperature_c': 25, 'outage_flag': 0, 'forecast_mwh': 50, 'actual_mwh': 50}
        rec_high_wind = {'irradiance_wm2': 600, 'wind_speed_ms': 10.0, 'temperature_c': 25, 'outage_flag': 0, 'forecast_mwh': 50, 'actual_mwh': 50}
        d_low = engine.decompose_record(rec_low_wind)
        d_high = engine.decompose_record(rec_high_wind)
        assert d_high['drivers']['wind_mwh'] > d_low['drivers']['wind_mwh']

    def test_confidence_scoring_levels(self, engine):
        # High confidence: small residual
        rec_high = {'irradiance_wm2': engine.mean_irradiance, 'wind_speed_ms': 7.0, 'temperature_c': 25, 'outage_flag': 0, 'forecast_mwh': 50, 'actual_mwh': 50.5}
        d_high = engine.decompose_record(rec_high)
        assert d_high['confidence'] in ['high', 'medium']

    def test_top_misses_sorted(self, engine):
        top5 = engine.get_top_misses(5)
        assert len(top5) == 5
        for i in range(len(top5) - 1):
            assert abs(top5[i]['total_variance_mwh']) >= abs(top5[i + 1]['total_variance_mwh'])

    def test_aggregated_driver_summary(self, engine):
        summary = engine.get_aggregated_driver_summary()
        assert 'total_outage_impact_mwh' in summary
        assert 'total_solar_impact_mwh' in summary
        assert 'total_wind_impact_mwh' in summary
        assert 'total_temperature_impact_mwh' in summary
        assert 'total_residual_mwh' in summary
        assert summary['outage_events_count'] == 3
        assert 'calibration_metadata' in summary


    def test_decomposition_cache_is_reused_and_defensive(self, engine):
        first = engine.get_all_decompositions()
        cached = engine._decomposition_cache
        assert cached is not None
        first[0]['drivers']['outage_mwh'] = 999999.0
        second = engine.get_all_decompositions()
        assert engine._decomposition_cache is cached
        assert second[0]['drivers']['outage_mwh'] != 999999.0
