# Model Card — Greenko Generation Forecasting Demo

## Intended use
This is an internship case-study prototype for comparing an operational day-ahead schedule with a small weather-driven ML challenger and explaining forecast-to-actual variance. It is not approved for real-time dispatch, grid bidding, safety decisions, or financial settlement.

## Data
- Bundled dataset: 60 synthetic/demo records, 30 dates, and two assumed assets.
- The CSV does not contain an authoritative asset identifier. The demo's positional mapping (first 30 rows / next 30 rows) is an explicit assumption and must be replaced with a source-system asset key for production.
- Six final dates are held out chronologically; all records sharing a date stay in the same split.

## Evaluation
- Model selection: lowest 4-fold chronological CV MAE among ML candidates, using training dates only.
- Final holdout: 12 records across six dates.
- Operational baseline holdout MAE: 4.604 MWh.
- Selected Ridge challenger holdout MAE: 6.286 MWh.
- Relative skill: -36.52% versus the operational baseline (negative means worse).
- Decision: retain the operational schedule as primary and treat ML as a challenger pending more data and prospective validation.

## Preprocessing and leakage controls
- Missing actual/forecast targets fail fast; targets are not imputed.
- Sensor features use median imputation inside each model pipeline. Each CV fold fits its own imputer and scaler using only that fold's training dates.
- For the variance dashboard's complete-record requirement, missing sensor values are filled from medians computed on the initial training window only. Counts and policy are exposed in `/api/metrics`.
- No full-series interpolation, backward fill, or test-set-fitted preprocessing is used.

## Limitations and risks
- Sample size is too small to support claims of generalization to other plants, seasons, or asset types.
- Weather inputs are limited; forecast lead time, NWP provenance, curtailment dispatch, inverter state, availability, and sub-hourly telemetry are absent.
- The variance waterfall is a reconciled statistical decomposition, not causal proof. The residual can contain omitted operational factors.
- Outage impact is inferred from a binary flag and a counterfactual; it is not a directly metered loss.
- Temperature sensitivity is estimated from a small non-outage training subset and falls back to a documented coefficient if the fitted sign is non-physical.

## Production requirements
Use authoritative asset IDs and timestamps, validate telemetry units and provenance, collect multiple seasons and plants, compare against persistence and operational baselines, run rolling-origin backtests, add prediction intervals and drift monitoring, and obtain domain-owner review before operational use.
