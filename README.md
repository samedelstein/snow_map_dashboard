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

## Run the Streamlit app

The Streamlit app lives in `snow_map.py`.

```bash
streamlit run snow_map.py
```

## Snapshot job

The snapshot job is implemented in `snapshot_snow_routes/snapshot_snow_routes.py`. It pulls the latest
snow route data from the ArcGIS feature service, writes a fresh GeoJSON file for the dashboard, and
records changes over time.

Run it with:

```bash
python snapshot_snow_routes/snapshot_snow_routes.py
```

Generated files (all under `snapshot_snow_routes/`):

- `latest_routes.geojson`: the most recent full GeoJSON snapshot for map rendering.
- `snapshots.csv`: an append-only history of route attribute changes.
- `state.json`: a lightweight fingerprint store used to detect which features have changed since the
  last run.

## Automation

A GitHub Actions workflow runs the snapshot job on a schedule in
`.github/workflows/snapshot.yml`.

## Weather data sources

The training pipeline (`snow_predict.py`) enriches each snapshot with hourly weather
observations and alert context for Syracuse, NY.

- NOAA/NWS hourly observations: `api.weather.gov/stations/KSYR/observations` (nearest-hour
  join on `snapshot_ts`, engineered snowfall rate, temperature, wind, and freezing rain flag).
- NWS alerts: `api.weather.gov/alerts?point=43.0481,-76.1474` (hourly alert indicator/count).

When the model job runs, it writes `artifacts_snow/weather_data_sources.json` with the
observed time coverage and row counts for each weather feed.
