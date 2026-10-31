# Improvement Log — Revision 2

## Verified changes
- Removed full-series interpolation/backfill to avoid future-to-past leakage.
- Added fold-local median imputation and scaling to every ML/CV pipeline.
- Fail fast when target columns (`actual_mwh`, `forecast_mwh`) contain missing values.
- Impute sensor gaps for variance reporting from training-window medians only and expose counts/policy through metrics.
- Keep ML model selection strictly on training-only chronological CV MAE.
- Expose held-out ML-vs-operational-baseline skill and a deployment recommendation.
- Apply the calibrated temperature coefficient consistently in the variance waterfall and report it in calibration metadata.
- Correct variance methodology documentation and add a model card.
- Added regression tests for missing sensor data, missing targets, and calibrated temperature attribution.

## Test evidence
Run from this project directory:

```bash
python -m pytest -q
python tests/run_all_tests.py
```

Latest verified result: **55 passed**.

## Important remaining limitation
The bundled 60-row demo dataset is not sufficient to demonstrate production-grade forecasting generalization. On the chronological holdout, the operational baseline MAE is 4.604 MWh and the selected Ridge challenger MAE is 6.286 MWh. The operational schedule must remain the primary forecast until prospective validation shows the ML model consistently beats it.
