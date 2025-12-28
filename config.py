BASE_URL = (
    "https://services6.arcgis.com/"
    "bdPqSfflsdgFRVVM/arcgis/rest/services/"
    "Winter_Operations_Snow_Routes/FeatureServer/0/query"
)

BUCKET_ORDER = [
    "Never plowed (before cutoff)",
    "> 24 hours",
    "12–24 hours",
    "6–12 hours",
    "1–6 hours",
    "< 1 hour",
]

BUCKET_DISPLAY_LABELS = {
    "Never plowed (before cutoff)": "⬜ Never plowed (before cutoff)",
    "> 24 hours": "🟥 > 24 hours",
    "12–24 hours": "🟥 12–24 hours",
    "6–12 hours": "🟧 6–12 hours",
    "1–6 hours": "🟨 1–6 hours",
    "< 1 hour": "🟩 < 1 hour",
}

BUCKET_COLORS = {
    "Never plowed (before cutoff)": [200, 200, 200],  # gray
    "> 24 hours": [128, 0, 38],                      # dark red
    "12–24 hours": [227, 26, 28],                    # red
    "6–12 hours": [253, 141, 60],                    # orange
    "1–6 hours": [255, 237, 111],                    # yellow
    "< 1 hour": [44, 162, 95],                       # green-ish
}

STATE_PATH = "snapshot_snow_routes/state.json"
LATEST_GEOJSON_PATH = "snapshot_snow_routes/latest_routes.geojson"
SNAPSHOTS_CSV_PATH = "snapshot_snow_routes/snapshots.csv"
