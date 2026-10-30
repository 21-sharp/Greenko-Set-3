from typing import Dict, Any, List
import numpy as np
import pandas as pd

class VarianceEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.mean_irradiance = float(self.df['irradiance_wm2'].mean())
        self.mean_wind = float(self.df['wind_speed_ms'].mean())
        self.mean_temp = float(self.df['temperature_c'].mean())
        self.mean_forecast = float(self.df['forecast_mwh'].mean())
        self.mean_actual = float(self.df['actual_mwh'].mean())

        non_outage = self.df[self.df['outage_flag'] == 0]
        self.solar_coeff = float(np.polyfit(non_outage['irradiance_wm2'], non_outage['actual_mwh'], 1)[0]) * 0.45
        self.wind_coeff = float(np.polyfit(non_outage['wind_speed_ms'], non_outage['actual_mwh'], 1)[0]) * 0.55

    def decompose_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        irradiance = float(record.get('irradiance_wm2', 0.0))
        wind_speed = float(record.get('wind_speed_ms', 0.0))
        temp = float(record.get('temperature_c', 25.0))
        outage = int(record.get('outage_flag', 0))
        forecast = float(record.get('forecast_mwh', 0.0))
        actual = float(record.get('actual_mwh', 0.0))
        asset_id = record.get('asset_id', None)

        total_variance = round(actual - forecast, 2)

        if outage == 1:
            weather_potential = max(10.0, (irradiance / 1000.0) * 45.0 + ((wind_speed / 11.5) ** 3) * 50.0 - max(0.0, (temp - 25.0) * 0.25))
            outage_driver = round(min(total_variance, -(forecast - actual)), 2)
            if outage_driver > -1.0:
                outage_driver = round(-(weather_potential * 0.7), 2)
            weather_residual = round(total_variance - outage_driver, 2)
            solar_driver = round(weather_residual * 0.45, 2)
            wind_driver = round(weather_residual * 0.45, 2)
            temp_driver = round(weather_residual * 0.10, 2)
            residual = round(total_variance - (outage_driver + solar_driver + wind_driver + temp_driver), 2)
            category = "Forced Outage / Operational Trip"
            explanation = f"Generation fell short by {abs(total_variance):.2f} MWh. Outage flag is active (curtailment or inverter trip), causing the dominant deficit of {abs(outage_driver):.2f} MWh."
        else:
            outage_driver = 0.0
            delta_irr = irradiance - self.mean_irradiance
            solar_driver = round((delta_irr / 100.0) * self.solar_coeff * 2.5, 2)

            delta_wind = (wind_speed ** 2.2) - (self.mean_wind ** 2.2)
            wind_driver = round((delta_wind / 20.0) * self.wind_coeff * 1.5, 2)

            delta_temp = temp - 25.0
            temp_driver = round(-max(0.0, delta_temp * 0.22), 2)

            explained_sum = solar_driver + wind_driver + temp_driver
            residual = round(total_variance - explained_sum, 2)

            if total_variance <= -5.0:
                category = "Severe Forecast Deficit"
                explanation = f"Generation was {abs(total_variance):.2f} MWh below forecast due to combined resource underperformance and thermal derating."
            elif total_variance >= 5.0:
                category = "Generation Surplus"
                explanation = f"Generation exceeded forecast by {total_variance:.2f} MWh due to favorable weather conditions exceeding forecast assumptions."
            else:
                category = "Within Normal Band"
                explanation = f"Generation variance of {total_variance:.2f} MWh is within acceptable operational tolerance (+/- 5 MWh)."

        waterfall = [
            {"step": "Day-Ahead Forecast", "value": round(forecast, 2), "delta": round(forecast, 2)},
            {"step": "Outage Impact", "value": outage_driver, "delta": outage_driver},
            {"step": "Solar Irradiance Impact", "value": solar_driver, "delta": solar_driver},
            {"step": "Wind Speed Impact", "value": wind_driver, "delta": wind_driver},
            {"step": "Temperature Derate", "value": temp_driver, "delta": temp_driver},
            {"step": "Unmodeled Residual", "value": residual, "delta": residual},
            {"step": "Actual Generation", "value": round(actual, 2), "delta": round(actual, 2)}
        ]

        result = {
            "date": record.get("date", "N/A"),
            "forecast_mwh": round(forecast, 2),
            "actual_mwh": round(actual, 2),
            "total_variance_mwh": total_variance,
            "variance_pct": round((total_variance / max(0.1, forecast)) * 100, 2),
            "category": category,
            "drivers": {
                "outage_mwh": outage_driver,
                "irradiance_mwh": solar_driver,
                "wind_mwh": wind_driver,
                "temperature_mwh": temp_driver,
                "residual_mwh": residual
            },
            "explanation": explanation,
            "waterfall": waterfall,
            "operating_conditions": {
                "irradiance_wm2": irradiance,
                "wind_speed_ms": wind_speed,
                "temperature_c": temp,
                "outage_flag": outage
            }
        }
        if asset_id is not None:
            result["asset_id"] = asset_id
        return result

    def get_all_decompositions(self) -> List[Dict[str, Any]]:
        results = []
        for _, row in self.df.iterrows():
            results.append(self.decompose_record(row.to_dict()))
        return results

    def get_top_misses(self, limit: int = 5) -> List[Dict[str, Any]]:
        decomps = self.get_all_decompositions()
        sorted_decomps = sorted(decomps, key=lambda x: abs(x["total_variance_mwh"]), reverse=True)
        return sorted_decomps[:limit]
