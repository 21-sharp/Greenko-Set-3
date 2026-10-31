# Deployment and Release Readiness (v2.2.0)

## What changed in this release

- Added a nominal 90% prediction interval to `/api/predict`.
- Interval radius is calibrated from chronological out-of-fold absolute residuals on training dates only. The final six-date holdout is not used to set the interval.
- Exposed interval calibration method, sample count, radius, and caveat in `/api/metrics` under `prediction_interval_calibration`.
- Restricted prediction `asset_id` to the two supported demo identifiers.
- Removed wildcard CORS with credentials. Same-origin Railway hosting works without CORS; explicitly allow additional browser origins with `ALLOWED_ORIGINS`.
- Added Railway health-check configuration for `/api/health`.
- Added an optional OpenAI-compatible LLM narrative layer. It consumes deterministic analysis and retrieved evidence; it cannot change numeric driver/waterfall outputs. Without credentials or on provider failure, the offline deterministic response is retained.
- Avoided returning raw exception text from prediction and agent API failures.

## Run locally

```bash
cd 03_generation_forecasting_variance_agent
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/` for the dashboard and
`http://127.0.0.1:8000/docs` for the API contract.

## Railway deployment

1. Push the repository to a Git provider.
2. In Railway, create a project from the repository.
3. Set the service root directory to `Greenko-Set-3/03_generation_forecasting_variance_agent` if deploying this subproject as its own service. If deploying from the repository root, configure the service root directory accordingly.
4. Railway uses `railway.json` and the configured start command:
   `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`
5. Wait for the `/api/health` health check to pass.
6. Verify all of these against the public service domain:
   - `GET /api/health` returns HTTP 200 and `status: healthy`.
   - `GET /api/metrics` includes model benchmarks and `prediction_interval_calibration`.
   - `POST /api/predict` returns a point prediction and ordered 90% interval.
   - `POST /api/run` returns a grounded investigation with retrieval evidence.
   - `/` serves the dashboard and `/docs` serves OpenAPI documentation.
7. If the frontend is hosted on a different origin, set `ALLOWED_ORIGINS` to a comma-separated list of exact trusted origins, including scheme and host. Do not use `*`.

## Optional grounded LLM explanation

The application runs without an LLM key. To enable narrative synthesis, configure:

- `OPENAI_API_KEY`: provider API key (required to enable)
- `OPENAI_MODEL`: optional model name; defaults to `gpt-4o-mini`
- `OPENAI_BASE_URL`: optional OpenAI-compatible API base URL; defaults to `https://api.openai.com/v1`

The deterministic intent router, retrieval, variance calculations, and recommendations run first. The LLM receives only the structured result and top retrieved knowledge-base evidence. It rewrites the narrative summary only; the numeric `drivers`, `waterfall`, `category`, and recommendations remain deterministic. Provider errors fall back to deterministic output. Do not send confidential plant data to an external provider without organizational approval.

## Prediction interval interpretation

The interval is an empirical uncertainty band around the selected ML point forecast. It uses rolling chronological out-of-fold residuals from the training period, with a finite-sample upper quantile. The current dataset is only 60 synthetic/demo records; therefore the interval is indicative, not a certified 90% coverage guarantee. Recalibrate with several months of timestamped plant telemetry and monitor realized coverage before operational use.

## Evidence and limitations

- Local automated test suite: run `python -m pytest -q`; release validation for this archive recorded 60 passing tests.
- A live Railway URL is intentionally not claimed or embedded. Deployment requires the owner's Railway account and repository authorization.
- No LLM API key is required: the agent remains deterministic and offline-capable with hybrid lexical retrieval. An optional LLM narrative layer can be enabled with credentials; it is not described as autonomous tool-using reasoning.
- The supplied data's asset assignment convention remains synthetic. Replace it with authoritative plant asset IDs before production.
- Do not treat the demo's five-driver attribution as directly metered physical causation; residual remains an explicit unmodeled bucket.
