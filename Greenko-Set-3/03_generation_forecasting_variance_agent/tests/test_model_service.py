import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.model_service import ModelService, safe_mape


@pytest.fixture(scope="module")
def model_service():
    return ModelService()


class TestModelService:
    def test_initialization(self, model_service):
        assert model_service.df is not None
        assert len(model_service.df) == 60
        assert model_service.pipeline_rf is not None
        assert model_service.pipeline_gb is not None
        assert model_service.pipeline_ridge is not None
        assert model_service.selected_model_name == 'Ridge Linear Baseline'

    def test_required_columns_present(self, model_service):
        required_cols = [
            'date', 'irradiance_wm2', 'wind_speed_ms', 'temperature_c',
            'outage_flag', 'forecast_mwh', 'actual_mwh', 'variance_mwh',
            'ml_forecast_mwh', 'asset_id', 'asset_code'
        ]
        for col in required_cols:
            assert col in model_service.df.columns, f"Missing column: {col}"

    def test_asset_identification(self, model_service):
        df = model_service.df
        assert (df['asset_id'] == 'Asset_1').sum() == 30
        assert (df['asset_id'] == 'Asset_2').sum() == 30
        assert df['date'].nunique() == 30

    def test_grouped_date_chronological_split(self, model_service):
        assert len(model_service.train_dates) == 24
        assert len(model_service.test_dates) == 6
        assert len(set(model_service.train_dates) & set(model_service.test_dates)) == 0
        assert model_service.train_mask.sum() == 48
        assert model_service.test_mask.sum() == 12

    def test_data_leakage_prevention_pipeline(self, model_service):
        train_df = model_service.df[model_service.train_mask]
        test_df = model_service.df[model_service.test_mask]
        scaler = model_service.pipeline_rf.named_steps['scaler']

        train_feats = model_service.engineer_features(train_df)[model_service.engineered_cols].values
        expected_means = np.mean(train_feats, axis=0)

        assert np.allclose(scaler.mean_, expected_means, atol=1e-3)
        assert not np.allclose(scaler.mean_, model_service.engineer_features(model_service.df)[model_service.engineered_cols].values.mean(axis=0), atol=1e-3)

    def test_feature_engineering_physics(self, model_service):
        sample_df = pd.DataFrame([{
            'date': '2026-08-15',
            'irradiance_wm2': 800.0,
            'wind_speed_ms': 10.0,
            'temperature_c': 35.0,
            'outage_flag': 0,
            'asset_code': 0
        }])
        feats = model_service.engineer_features(sample_df)

        assert pytest.approx(feats['wind_cubed'].iloc[0], rel=1e-3) == 1.0
        assert pytest.approx(feats['wind_squared'].iloc[0], rel=1e-3) == 1.0
        assert pytest.approx(feats['temp_derate'].iloc[0], rel=1e-3) == 10.0
        assert pytest.approx(feats['irr_kw'].iloc[0], rel=1e-3) == 0.8
        assert pytest.approx(feats['solar_effective'].iloc[0], rel=1e-3) == 0.768
        assert feats['day_of_month'].iloc[0] == 15

    def test_model_metrics_structure(self, model_service):
        metrics = model_service.metrics_summary
        assert 'operational_baseline' in metrics
        assert 'ridge_linear_baseline' in metrics
        assert 'random_forest_model' in metrics
        assert 'gradient_boosting_model' in metrics
        assert 'model_selection' in metrics
        assert 'evaluation_protocol' in metrics

        rf = metrics['random_forest_model']
        assert rf['test']['mae'] <= 7.0
        assert rf['test']['r2'] >= 0.60
        assert 'residual_analysis' in rf['test']
        assert 'mean_error' in rf['test']['residual_analysis']

        ridge = metrics['ridge_linear_baseline']
        assert ridge['test']['mae'] <= 6.5
        assert 'residual_analysis' in ridge['test']

    def test_model_selection_consistency(self, model_service):
        sel = model_service.metrics_summary['model_selection']
        assert sel['selected_model'] == 'Ridge Linear Baseline'
        assert sel['selected_model'] == model_service.selected_model_name
        # Model selection must be driven by training-only CV, never the held-out test set.
        cv_maes = sel['cv_mae_comparison']
        ml_cv = {k: v for k, v in cv_maes.items() if k != 'Operational Baseline'}
        assert sel['selection_metric'].startswith('Lowest 4-fold chronological CV MAE')
        assert sel['selected_model'] == min(ml_cv, key=ml_cv.get)
        assert model_service.metrics_summary['operational_baseline']['cv_mae'] > 0
        maes = sel['test_mae_comparison']
        assert maes['Ridge Linear Baseline'] <= maes['Random Forest Regressor']
        assert maes['Ridge Linear Baseline'] <= maes['Gradient Boosting Regressor']

    def test_baseline_comparison_honesty(self, model_service):
        sel = model_service.metrics_summary['model_selection']
        maes = sel['test_mae_comparison']
        assert maes['Operational Baseline'] < maes['Ridge Linear Baseline']
        assert 'baseline_comparison_analysis' in sel
        assert len(sel['baseline_comparison_analysis']) > 50

    def test_prediction_output(self, model_service):
        pred = model_service.predict(
            irradiance_wm2=700.0,
            wind_speed_ms=8.0,
            temperature_c=30.0,
            outage_flag=0,
            asset_id='Asset_1'
        )
        assert 'predicted_mwh' in pred
        assert 'random_forest_mwh' in pred
        assert 'gradient_boosting_mwh' in pred
        assert 'baseline_linear_mwh' in pred
        assert pred['predicted_mwh'] > 0
        assert pred['outage_active'] is False
        assert pred['selected_model'] == 'Ridge Linear Baseline'

    def test_outage_prediction(self, model_service):
        pred_normal = model_service.predict(700.0, 8.0, 30.0, 0)
        pred_outage = model_service.predict(700.0, 8.0, 30.0, 1)
        assert pred_outage['outage_active'] is True
        assert pred_outage['predicted_mwh'] < pred_normal['predicted_mwh']

    def test_safe_mape_zeros(self):
        y_true = np.array([10.0, 20.0, 0.0, 0.05, 50.0])
        y_pred = np.array([11.0, 18.0, 5.0, 2.0, 48.0])
        val = safe_mape(y_true, y_pred)
        assert not np.isnan(val)
        assert not np.isinf(val)
        assert val > 0

    def test_missing_sensor_values_use_training_only_imputation(self, tmp_path):
        source = pd.read_csv(BASE_DIR / 'data' / 'generation_forecast.csv')
        source.loc[2, 'irradiance_wm2'] = np.nan
        source.loc[50, 'wind_speed_ms'] = np.nan  # held-out period
        source.loc[4, 'temperature_c'] = np.nan
        dirty_path = tmp_path / 'telemetry_with_gaps.csv'
        source.to_csv(dirty_path, index=False)

        fitted = ModelService(data_path=dirty_path)
        quality = fitted.metrics_summary['data_quality']
        assert quality['missing_sensor_values_imputed_from_training_medians'] == {
            'irradiance_wm2': 1, 'wind_speed_ms': 1, 'temperature_c': 1
        }
        assert fitted.df[['irradiance_wm2', 'wind_speed_ms', 'temperature_c']].isna().sum().sum() == 0
        assert 'imputer' in fitted.pipeline_ridge.named_steps
        assert fitted.metrics_summary['evaluation_protocol']['leakage_prevention'].startswith('Median imputation')

    def test_missing_targets_fail_fast(self, tmp_path):
        source = pd.read_csv(BASE_DIR / 'data' / 'generation_forecast.csv')
        source.loc[0, 'actual_mwh'] = np.nan
        dirty_path = tmp_path / 'missing_target.csv'
        source.to_csv(dirty_path, index=False)
        with pytest.raises(ValueError, match='Missing target values'):
            ModelService(data_path=dirty_path)

    def test_outlier_capping(self):
        vals = pd.Series([-50.0, 500.0, 2500.0])
        clipped = vals.clip(lower=0.0, upper=1400.0)
        assert clipped.iloc[0] == 0.0
        assert clipped.iloc[1] == 500.0
        assert clipped.iloc[2] == 1400.0

    def test_prediction_interval_is_ordered_and_nonnegative(self, model_service):
        pred = model_service.predict(700.0, 8.0, 30.0, 0)
        interval = pred['prediction_interval_90_mwh']
        assert interval['lower'] >= 0.0
        assert interval['upper'] >= interval['lower']
        assert interval['lower'] <= pred['predicted_mwh'] <= interval['upper']
        assert interval['nominal_coverage'] == 0.90

    def test_prediction_interval_calibration_uses_training_oof_only(self, model_service):
        calibration = model_service.interval_calibration
        assert calibration['calibration_observations'] > 0
        assert calibration['nominal_coverage'] == 0.90
        assert 'training dates only' in calibration['method']
        assert calibration['radius_mwh'] > 0
