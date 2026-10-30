from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / 'data' / 'generation_forecast.csv'

class ModelService:
    def __init__(self, data_path: Path = DATA_PATH):
        self.data_path = data_path
        self.df = None
        self.feature_cols = ['irradiance_wm2', 'wind_speed_ms', 'temperature_c', 'outage_flag']
        self.engineered_cols = ['irradiance_wm2', 'wind_speed_ms', 'temperature_c', 'outage_flag', 'wind_cubed', 'temp_derate', 'irr_kw']
        self.baseline_ridge = None
        self.gb_model = None
        self.scaler = None
        self.metrics_summary = {}
        self.feature_importances = {}
        self.initialize()

    def engineer_features(self, df_input: pd.DataFrame) -> pd.DataFrame:
        df = df_input.copy()
        df['wind_cubed'] = (df['wind_speed_ms'] / 10.0) ** 3
        df['temp_derate'] = np.maximum(0.0, df['temperature_c'] - 25.0)
        df['irr_kw'] = df['irradiance_wm2'] / 1000.0
        return df

    def initialize(self):
        self.df = pd.read_csv(self.data_path)
        self.df['variance_mwh'] = np.round(self.df['actual_mwh'] - self.df['forecast_mwh'], 2)
        self.df['abs_variance_mwh'] = np.round(np.abs(self.df['variance_mwh']), 2)
        self.df['variance_pct'] = np.round((self.df['variance_mwh'] / np.maximum(self.df['forecast_mwh'], 0.1)) * 100, 2)

        df_feats = self.engineer_features(self.df)
        X_all = df_feats[self.engineered_cols].values
        y_all = self.df['actual_mwh'].values

        train_size = 48
        X_train = X_all[:train_size]
        X_test = X_all[train_size:]
        y_train = y_all[:train_size]
        y_test = y_all[train_size:]

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        X_all_scaled = self.scaler.transform(X_all)

        baseline_preds = self.df['forecast_mwh'].values
        y_series = self.df['actual_mwh']

        self.baseline_ridge = Ridge(alpha=1.0)
        self.baseline_ridge.fit(X_train_scaled, y_train)
        ridge_preds = self.baseline_ridge.predict(X_all_scaled)

        self.gb_model = GradientBoostingRegressor(n_estimators=80, max_depth=3, learning_rate=0.08, random_state=42)
        self.gb_model.fit(X_train_scaled, y_train)
        gb_preds_train = self.gb_model.predict(X_train_scaled)
        gb_preds_test = self.gb_model.predict(X_test_scaled)
        gb_preds_all = self.gb_model.predict(X_all_scaled)
        self.df['ml_forecast_mwh'] = np.round(gb_preds_all, 2)

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_validate(self.gb_model, X_train_scaled, y_train, cv=cv, scoring=['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2'])

        actual_mean = float(y_series.mean())
        self.metrics_summary = {
            'operational_baseline': {
                'mae': round(float(mean_absolute_error(y_series, baseline_preds)), 3),
                'rmse': round(float(root_mean_squared_error(y_series, baseline_preds)), 3),
                'r2': round(float(r2_score(y_series, baseline_preds)), 3),
                'mape': round(float(np.mean(np.abs((y_series - baseline_preds) / np.maximum(y_series, 0.1))) * 100), 2),
                'nmae': round(float((mean_absolute_error(y_series, baseline_preds) / actual_mean) * 100), 2)
            },
            'ridge_linear_baseline': {
                'mae': round(float(mean_absolute_error(y_series, ridge_preds)), 3),
                'rmse': round(float(root_mean_squared_error(y_series, ridge_preds)), 3),
                'r2': round(float(r2_score(y_series, ridge_preds)), 3),
                'mape': round(float(np.mean(np.abs((y_series - ridge_preds) / np.maximum(y_series, 0.1))) * 100), 2),
                'nmae': round(float((mean_absolute_error(y_series, ridge_preds) / actual_mean) * 100), 2)
            },
            'gradient_boosting_model': {
                'train_mae': round(float(mean_absolute_error(y_train, gb_preds_train)), 3),
                'train_rmse': round(float(root_mean_squared_error(y_train, gb_preds_train)), 3),
                'train_r2': round(float(r2_score(y_train, gb_preds_train)), 3),
                'test_mae': round(float(mean_absolute_error(y_test, gb_preds_test)), 3),
                'test_rmse': round(float(root_mean_squared_error(y_test, gb_preds_test)), 3),
                'test_r2': round(float(r2_score(y_test, gb_preds_test)), 3),
                'cv_mae': round(float(-cv_results['test_neg_mean_absolute_error'].mean()), 3),
                'cv_rmse': round(float(-cv_results['test_neg_root_mean_squared_error'].mean()), 3),
                'cv_r2': round(float(cv_results['test_r2'].mean()), 3),
                'nmae': round(float((-cv_results['test_neg_mean_absolute_error'].mean() / actual_mean) * 100), 2)
            }
        }

        importances = self.gb_model.feature_importances_
        total = float(np.sum(importances))
        self.feature_importances = {col: round(float(val / total) * 100, 2) for col, val in zip(self.engineered_cols, importances)}

    def predict(self, irradiance_wm2: float, wind_speed_ms: float, temperature_c: float, outage_flag: int) -> dict:
        row = pd.DataFrame([{
            'irradiance_wm2': float(irradiance_wm2),
            'wind_speed_ms': float(wind_speed_ms),
            'temperature_c': float(temperature_c),
            'outage_flag': int(outage_flag)
        }])
        row_feat = self.engineer_features(row)[self.engineered_cols]
        row_scaled = self.scaler.transform(row_feat)
        pred_ml = float(self.gb_model.predict(row_scaled)[0])
        pred_ridge = float(self.baseline_ridge.predict(row_scaled)[0])

        solar_capacity_factor = min(1.0, max(0.0, float(irradiance_wm2) / 1000.0))
        wind_capacity_factor = min(1.0, max(0.0, ((float(wind_speed_ms) / 11.5) ** 3)))
        temp_loss_pct = max(0.0, (float(temperature_c) - 25.0) * 0.4)

        return {
            'predicted_mwh': round(max(0.0, pred_ml), 2),
            'baseline_linear_mwh': round(max(0.0, pred_ridge), 2),
            'solar_capacity_factor': round(solar_capacity_factor, 3),
            'wind_capacity_factor': round(wind_capacity_factor, 3),
            'temp_loss_percent': round(temp_loss_pct, 2),
            'outage_active': bool(outage_flag == 1)
        }

    def get_records(self) -> list:
        return self.df.to_dict(orient='records')

    def get_summary_stats(self) -> dict:
        return {
            'total_records': int(len(self.df)),
            'total_actual_mwh': round(float(self.df['actual_mwh'].sum()), 2),
            'total_forecast_mwh': round(float(self.df['forecast_mwh'].sum()), 2),
            'net_variance_mwh': round(float(self.df['variance_mwh'].sum()), 2),
            'mean_abs_variance_mwh': round(float(self.df['abs_variance_mwh'].mean()), 2),
            'outage_days_count': int((self.df['outage_flag'] == 1).sum()),
            'avg_irradiance_wm2': round(float(self.df['irradiance_wm2'].mean()), 2),
            'avg_wind_speed_ms': round(float(self.df['wind_speed_ms'].mean()), 2),
            'avg_temperature_c': round(float(self.df['temperature_c'].mean()), 2)
        }
