# Snow Map Dashboard

## Requirements

- Python 3.10+ (recommended)
- `pip` or another Python package manager

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install streamlit pandas requests pydeck
```

## Repository layout

- `scripts/`: runnable entrypoints (Streamlit app, prediction pipeline, and snapshot wrapper).
- `snow_map_dashboard/`: the core Python package (including the snapshot module).
- `data/`: generated artifacts and snapshot datasets committed by automation.

## Pipelines

### Streamlit app

The current Streamlit entrypoint is `scripts/snowplow_app.py` (the older `scripts/snow_map.py` is
optional/legacy).

```bash
streamlit run scripts/snowplow_app.py
```

If you want to use the Predictions tab, make sure the latest artifacts are available at
`data/artifacts_snow/predictions_latest_prob.csv`.

### Snapshot job

The snapshot job is implemented in `snow_map_dashboard.snapshot`. It pulls the latest snow route
data from the ArcGIS feature service, writes a fresh GeoJSON file for the dashboard, and records
changes over time.

Run it with:

```bash
python -m snow_map_dashboard.snapshot
```

Generated files (all under `data/snapshot_snow_routes/`):

- `latest_routes.geojson`: the most recent full GeoJSON snapshot for map rendering.
- `snapshots.csv`: an append-only history of route attribute changes.
- `state.json`: a lightweight fingerprint store used to detect which features have changed since the
  last run.

### Prediction pipeline

The prediction pipeline is implemented in `scripts/snow_predict.py` and produces the latest model
outputs for the dashboard.

```bash
python scripts/snow_predict.py
```

Generated files (all under `data/artifacts_snow/`):

- `predictions_latest_prob.csv`: latest per-segment probability predictions.
- `model_metrics_prob.json`: model evaluation metrics.
- `weather_data_sources.json`: weather feed metadata for the training window.

## Automation

GitHub Actions workflows run the snapshot and prediction jobs on a schedule in
`.github/workflows/snapshot.yml` and `.github/workflows/predict.yml`.

## Weather data sources

The training pipeline (`snow_predict.py`) enriches each snapshot with hourly weather
observations and alert context for Syracuse, NY.

- NOAA/NWS hourly observations: `api.weather.gov/stations/KSYR/observations` (nearest-hour
  join on `snapshot_ts`, engineered snowfall rate, temperature, wind, and freezing rain flag).
- NWS alerts: `api.weather.gov/alerts?point=43.0481,-76.1474` (hourly alert indicator/count).

When the model job runs, it writes `data/artifacts_snow/weather_data_sources.json` with the
observed time coverage and row counts for each weather feed.

### NWS alerts log dataset

The alerts collection step appends new alert records to a dedicated dataset at
`data/artifacts_snow/nws_alerts_log.csv`. Records are de-duplicated by `alert_id` so the
file grows incrementally over time.

Schema:

- `alert_id`: NWS alert identifier (de-duplication key).
- `event`: event name (e.g., Winter Storm Warning).
- `start_ts`: alert start timestamp (UTC).
- `end_ts`: alert end timestamp (UTC).
- `severity`: NWS severity classification.
- `source_url`: source alert URL from `api.weather.gov/alerts`.
