Files:
- snow_ranking.py: computes per-event ranks and cross-event stability (median/IQR).
- streamlit_app.py: single-page Streamlit UI with search + map hover tooltips + narrative.

How to run locally:
  pip install streamlit pandas numpy pydeck
  streamlit run streamlit_app.py

Repo placement:
- Put both files in your repo root (same level as snapshots.csv and winter_operations_snow_routes_layer0.geojson).
- Ensure snapshots.csv exists (or update SNAPSHOTS_PATH in streamlit_app.py).
