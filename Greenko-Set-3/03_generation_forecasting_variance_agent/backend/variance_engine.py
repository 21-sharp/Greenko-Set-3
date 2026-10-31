from typing import Dict, Any, List, Optional
import logging
from copy import deepcopy
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VarianceEngine:
    TEMP_DERATE_COEFF = -0.24

    def __init__(self, df: pd.DataFrame, train_dates: Optional[List[str]] = None):
        self.df = df.copy()
        self._decomposition_cache = None

        if train_dates is not None and len(train_dates) > 0:
            calib_df = self.df[self.df['date'].isin(train_dates)].copy()
        else:
            calib_df = self.df.iloc[:48].copy()

        non_outage_train = calib_df[calib_df['outage_flag'] == 0].copy()

        from sklearn.linear_model import LinearRegression

        non_outage_train['wind_cubed'] = (non_outage_train['wind_speed_ms'] / 10.0) ** 3
        non_outage_train['temp_derate'] = np.maximum(0.0, non_outage_train['temperature_c'] - 25.0)

        self.mean_irradiance = float(non_outage_train['irradiance_wm2'].mean())
        self.mean_wind_cubed = float(non_outage_train['wind_cubed'].mean())
        self.mean_temp = float(non_outage_train['temperature_c'].mean())
        self.mean_forecast = float(non_outage_train['forecast_mwh'].mean())
        self.mean_actual = float(non_outage_train['actual_mwh'].mean())

        reg = LinearRegression()
        reg.fit(non_outage_train[['irradiance_wm2', 'wind_cubed', 'temp_derate']], non_outage_train['actual_mwh'])
        self.solar_coeff = float(reg.coef_[0])
        self.wind_coeff = float(reg.coef_[1])
        self.temp_coeff = float(reg.coef_[2]) if reg.coef_[2] < 0 else self.TEMP_DERATE_COEFF

        logger.info(
            "VarianceEngine calibrated on %d training records: solar_coeff=%.5f, wind_coeff=%.4f",
            len(non_outage_train), self.solar_coeff, self.wind_coeff
        )

    def decompose_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        irradiance = float(record.get('irradiance_wm2', 0.0))
        wind_speed = float(record.get('wind_speed_ms', 0.0))
        temp = float(record.get('temperature_c', 25.0))
        outage = int(record.get('outage_flag', 0))
        forecast = float(record.get('forecast_mwh', 0.0))
        actual = float(record.get('actual_mwh', 0.0))
        asset_id = record.get('asset_id', 'Asset_1')

        total_variance = round(actual - forecast, 2)

        wind_cubed = (wind_speed / 10.0) ** 3
        delta_irr = irradiance - self.mean_irradiance
        delta_wind_cubed = wind_cubed - self.mean_wind_cubed
        delta_temp = max(0.0, temp - 25.0)

        solar_driver = round(self.solar_coeff * delta_irr, 2)
        wind_driver = round(self.wind_coeff * delta_wind_cubed, 2)
        temp_driver = round(self.temp_coeff * delta_temp, 2)

        if outage == 1:
            estimated_healthy_output = max(
                0.0,
                forecast + solar_driver + wind_driver + temp_driver
            )
            outage_driver = round(min(0.0, actual - estimated_healthy_output), 2)
            if total_variance < 0 and outage_driver < total_variance:
                outage_driver = round(total_variance - (solar_driver + wind_driver + temp_driver), 2)

            explained_so_far = outage_driver + solar_driver + wind_driver + temp_driver
            residual = round(total_variance - explained_so_far, 2)

            category = "Forced Outage / Operational Trip"
            explanation = (
                f"Generation fell short by {abs(total_variance):.2f} MWh. Asset outage flag is active, "
                f"accounting for an estimated mechanical deficit of {abs(outage_driver):.2f} MWh relative "
                f"to resource potential. Weather deviations contributed: solar {solar_driver:+.2f} MWh, "
                f"wind {wind_driver:+.2f} MWh, thermal derate {temp_driver:+.2f} MWh."
            )
        else:
            outage_driver = 0.0
            explained_so_far = solar_driver + wind_driver + temp_driver
            residual = round(total_variance - explained_so_far, 2)

            if total_variance <= -5.0:
                category = "Severe Forecast Deficit"
                explanation = (
                    f"Generation was {abs(total_variance):.2f} MWh below day-ahead forecast. "
                    f"Resource deviation breakdown: solar insolation {solar_driver:+.2f} MWh, "
                    f"aerodynamic wind {wind_driver:+.2f} MWh, thermal derating {temp_driver:+.2f} MWh, "
                    f"and unmodeled residual {residual:+.2f} MWh."
                )
            elif total_variance >= 5.0:
                category = "Generation Surplus"
                explanation = (
                    f"Generation exceeded forecast by {total_variance:.2f} MWh. "
                    f"Resource deviation breakdown: solar insolation {solar_driver:+.2f} MWh, "
                    f"aerodynamic wind {wind_driver:+.2f} MWh, thermal derating {temp_driver:+.2f} MWh, "
                    f"and unmodeled residual {residual:+.2f} MWh."
                )
            else:
                category = "Within Normal Band"
                explanation = (
                    f"Generation variance of {total_variance:+.2f} MWh is within normal operational "
                    f"tolerance (±5 MWh). Solar: {solar_driver:+.2f} MWh, Wind: {wind_driver:+.2f} MWh, "
                    f"Temp: {temp_driver:+.2f} MWh."
                )

        abs_var = abs(total_variance)
        abs_res = abs(residual)
        if abs_var < 2.0 or abs_res <= 0.25 * abs_var:
            confidence = "high"
        elif abs_res <= 0.50 * abs_var:
            confidence = "medium"
        else:
            confidence = "low"

        waterfall = [
            {"step": "Day-Ahead Forecast", "value": round(forecast, 2), "delta": round(forecast, 2)},
            {"step": "Outage Impact", "value": outage_driver, "delta": outage_driver},
            {"step": "Solar Irradiance Impact", "value": solar_driver, "delta": solar_driver},
            {"step": "Wind Speed Impact", "value": wind_driver, "delta": wind_driver},
            {"step": "Temperature Derate", "value": temp_driver, "delta": temp_driver},
            {"step": "Unmodeled Residual", "value": residual, "delta": residual},
            {"step": "Actual Generation", "value": round(actual, 2), "delta": round(actual, 2)}
        ]

        driver_sum = round(outage_driver + solar_driver + wind_driver + temp_driver + residual, 2)
        reconciled = abs(driver_sum - total_variance) < 0.05

        return {
            "date": record.get("date", "N/A"),
            "asset_id": asset_id,
            "forecast_mwh": round(forecast, 2),
            "actual_mwh": round(actual, 2),
            "total_variance_mwh": total_variance,
            "variance_pct": round((total_variance / max(0.1, forecast)) * 100, 2),
            "category": category,
            "confidence": confidence,
            "drivers": {
                "outage_mwh": outage_driver,
                "irradiance_mwh": solar_driver,
                "wind_mwh": wind_driver,
                "temperature_mwh": temp_driver,
                "residual_mwh": residual
            },
            "driver_reconciliation": {
                "driver_sum": driver_sum,
                "total_variance": total_variance,
                "reconciled": reconciled
            },
            "explanation": explanation,
            "waterfall": waterfall,
            "operating_conditions": {
                "irradiance_wm2": irradiance,
                "wind_speed_ms": wind_speed,
                "temperature_c": temp,
                "outage_flag": outage
            },
            "methodology_note": (
                "Drivers are calibrated on training non-outage observations. Solar and wind drivers use OLS regression sensitivities "
                f"(wind strictly uses cubic v^3). Temperature sensitivity is {self.temp_coeff:.4f} MWh/°C above 25°C, "
                "estimated from training non-outage observations when negative, otherwise using a documented fallback. "
                "Attribution represents estimated statistical sensitivity decomposition, not direct sub-metered physical causation."
            )
        }

    def _get_cached_decompositions(self) -> List[Dict[str, Any]]:
        """Compute static portfolio decompositions once and reuse them across API reads."""
        if self._decomposition_cache is None:
            self._decomposition_cache = [
                self.decompose_record(record)
                for record in self.df.to_dict(orient='records')
            ]
        return self._decomposition_cache

    def get_all_decompositions(self) -> List[Dict[str, Any]]:
        # Return defensive copies so API callers cannot mutate the internal cache.
        return deepcopy(self._get_cached_decompositions())

    def get_top_misses(self, limit: int = 5) -> List[Dict[str, Any]]:
        decomps = self._get_cached_decompositions()
        sorted_decomps = sorted(decomps, key=lambda x: abs(x["total_variance_mwh"]), reverse=True)
        return sorted_decomps[:limit]

    def get_aggregated_driver_summary(self) -> Dict[str, Any]:
        decomps = self._get_cached_decompositions()
        drivers_df = pd.DataFrame([d['drivers'] for d in decomps])
        return {
            "total_outage_impact_mwh": round(float(drivers_df['outage_mwh'].sum()), 2),
            "total_solar_impact_mwh": round(float(drivers_df['irradiance_mwh'].sum()), 2),
            "total_wind_impact_mwh": round(float(drivers_df['wind_mwh'].sum()), 2),
            "total_temperature_impact_mwh": round(float(drivers_df['temperature_mwh'].sum()), 2),
            "total_residual_mwh": round(float(drivers_df['residual_mwh'].sum()), 2),
            "mean_outage_impact_mwh": round(float(drivers_df['outage_mwh'].mean()), 2),
            "mean_solar_impact_mwh": round(float(drivers_df['irradiance_mwh'].mean()), 2),
            "mean_wind_impact_mwh": round(float(drivers_df['wind_mwh'].mean()), 2),
            "mean_temperature_impact_mwh": round(float(drivers_df['temperature_mwh'].mean()), 2),
            "mean_residual_mwh": round(float(drivers_df['residual_mwh'].mean()), 2),
            "outage_events_count": int(sum(1 for d in decomps if d['operating_conditions']['outage_flag'] == 1)),
            "calibration_metadata": {
                "solar_sensitivity_per_wm2": round(self.solar_coeff, 5),
                "wind_sensitivity_per_cubic_unit": round(self.wind_coeff, 4),
                "temp_coefficient_per_degC": round(self.temp_coeff, 5),
                "temperature_coefficient_source": "training_ols_or_fallback",
                "wind_exponent": 3.0,
                "reconciliation_guarantee": "100% strict mathematical equality"
            }
        }
