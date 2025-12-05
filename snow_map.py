import math
import requests
import pandas as pd
import streamlit as st
import pydeck as pdk

BASE_URL = (
    "https://services6.arcgis.com/"
    "bdPqSfflsdgFRVVM/arcgis/rest/services/"
    "Winter_Operations_Snow_Routes/FeatureServer/0/query"
)

# “This storm” cutoff – anything before this is considered never plowed for this event
CUTOFF_DATE = pd.to_datetime("2025-12-01", utc=True)


def web_mercator_to_wgs84(x, y):
    """
    Convert Web Mercator (EPSG:3857/102100) to lon/lat in WGS84.
    """
    R = 6378137.0
    lon = (x / R) * 180.0 / math.pi
    lat = (2 * math.atan(math.exp(y / R)) - math.pi / 2.0) * 180.0 / math.pi
    return lon, lat


@st.cache_data(show_spinner=True)
def fetch_data(page_size=1000):
    """
    Fetch all snow route segments with attributes + geometry and
    pre-compute:
      - lastserviced_dt (UTC datetime)
      - bucket (time since last plow)
      - miles (segment length in miles, from Shape__Length in meters)
      - path (lon/lat coordinates for pydeck PathLayer)
    """
    offset = 0
    attr_rows = []
    geom_rows = []

    while True:
        params = {
            "f": "json",
            "where": "1=1",
            "outFields": "*",
            "resultOffset": offset,
            "resultRecordCount": page_size,
            "returnGeometry": "true",
        }
        r = requests.get(BASE_URL, params=params)
        r.raise_for_status()
        data = r.json()

        features = data.get("features", [])
        if not features:
            break

        for feat in features:
            attrs = feat["attributes"]
            geom = feat.get("geometry", {}) or {}
            paths = geom.get("paths", [])

            attr_rows.append(attrs)

            # Convert each path from Web Mercator -> WGS84
            paths_wgs84 = []
            for path in paths:
                path_wgs = []
                for x, y in path:
                    lon, lat = web_mercator_to_wgs84(x, y)
                    path_wgs.append([lon, lat])
                paths_wgs84.append(path_wgs)

            geom_rows.append(
                {
                    "OBJECTID": attrs.get("OBJECTID"),
                    "paths_wgs84": paths_wgs84,
                }
            )

        if len(features) < page_size or not data.get("exceededTransferLimit", False):
            break

        offset += page_size

    df_attr = pd.DataFrame(attr_rows)
    df_geom = pd.DataFrame(geom_rows)
    df = df_attr.merge(df_geom, on="OBJECTID", how="left")

    # --- Time handling ---
    df["lastserviced_dt"] = pd.to_datetime(
        df["lastserviced"], unit="ms", utc=True, errors="coerce"
    )

    # Completely ignore NaT/invalid timestamps
    df = df.dropna(subset=["lastserviced_dt"])

    # --- Miles from Shape__Length (meters) ---
    if "Shape__Length" in df.columns:
        df["miles"] = df["Shape__Length"] / 1609.344
    elif "segmentlength" in df.columns:
        df["miles"] = df["segmentlength"]
    else:
        df["miles"] = 0.0

    # --- Bucket classification ---
    now = pd.Timestamp.now(tz="UTC")

    def classify_row(row):
        dt = row["lastserviced_dt"]

        if dt < CUTOFF_DATE:
            return "Never plowed (before 12/1)"

        hours = (now - dt).total_seconds() / 3600.0

        if hours < 1:
            return "< 1 hour"
        elif hours < 6:
            return "1–6 hours"
        elif hours < 12:
            return "6–12 hours"
        elif hours < 24:
            return "12–24 hours"
        else:
            return "> 24 hours"

    df["bucket"] = df.apply(classify_row, axis=1)

    # Geometry for map: use first path
    df["path"] = df["paths_wgs84"].str[0]

    return df


# --- Streamlit UI ---

st.title("Syracuse Snow Routes – Plowing Status Dashboard")

st.caption(
    "Buckets are based on `lastserviced` timestamps. "
    "Segments with no service since before 2025-12-01 are tagged as "
    "`Never plowed (before 12/1)` for this storm. "
    "Segments with no valid timestamp are ignored."
)

# ---- Custom CSS for nicer KPI cards ----
st.markdown("""
<style>
.kpi-card {
    background-color: #ffffff;
    padding: 1.0rem 1.2rem;
    border-radius: 12px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.06);
    border: 1px solid #e5e5e5;
    margin-bottom: 0.8rem;
}

.kpi-card-total {
    background-color: #ffffff;
    padding: 1.1rem 1.4rem;
    border-radius: 12px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
    border: 1px solid #e5e5e5;
    margin-bottom: 0.8rem;
}

.kpi-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #555555;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.kpi-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #222222;
    margin-top: 0.25rem;
}

.kpi-gauge {
    margin-top: 0.5rem;
    height: 6px;
    border-radius: 999px;
    background-color: #f0f0f0;
    overflow: hidden;
}

.kpi-gauge-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #2b8cff, #68d391);
}

.kpi-gauge-label {
    margin-top: 0.25rem;
    font-size: 0.75rem;
    color: #666666;
}
</style>
""", unsafe_allow_html=True)


df = fetch_data()

# --- Miles by bucket (for metrics) ---

bucket_order = [
    "Never plowed (before 12/1)",
    "> 24 hours",
    "12–24 hours",
    "6–12 hours",
    "1–6 hours",
    "< 1 hour",
]

bucket_miles = (
    df.groupby("bucket")["miles"]
    .sum()
    .reindex(bucket_order)
    .fillna(0)
)

bucket_miles_df = bucket_miles.reset_index()
bucket_miles_df.columns = ["bucket", "miles"]

total_miles = bucket_miles_df["miles"].sum()

st.subheader("Miles plowed by recency bucket")

# Add percentage of total for each bucket
bucket_miles_df["pct"] = (
    (bucket_miles_df["miles"] / total_miles * 100).round(1)
    if total_miles > 0 else 0
)

# --- Total miles card (full-width) ---
st.markdown(f"""
<div class="kpi-card-total">
    <div class="kpi-title">Total miles in snow route system (valid timestamps)</div>
    <div class="kpi-value">{total_miles:.2f} miles</div>
</div>
""", unsafe_allow_html=True)

# --- Bucket cards laid out with Streamlit columns ---
cols = st.columns(3)  # 3 cards per row; Streamlit will wrap on smaller screens

for i, row in bucket_miles_df.iterrows():
    bucket = row["bucket"]
    miles = row["miles"]
    pct = row["pct"] if total_miles > 0 else 0

    card_html = f"""
    <div class="kpi-card">
        <div class="kpi-title">{bucket}</div>
        <div class="kpi-value">{miles:.2f} miles</div>
        <div class="kpi-gauge">
            <div class="kpi-gauge-fill" style="width: {pct}%;"></div>
        </div>
        <div class="kpi-gauge-label">{pct:.1f}% of total miles</div>
    </div>
    """

    cols[i % 3].markdown(card_html, unsafe_allow_html=True)

with st.expander("Bucket miles data"):
    st.dataframe(bucket_miles_df, use_container_width=True)


# --- Map ---

st.subheader("Map of snow routes by time since last plow")

buckets = bucket_miles_df["bucket"].tolist()
selected_buckets = st.multiselect(
    "Show buckets:",
    options=buckets,
    default=buckets,
)

map_df = df[df["bucket"].isin(selected_buckets)].copy()
map_df = map_df.dropna(subset=["lastserviced_dt"])
map_df = map_df.dropna(subset=["path"])
map_df = map_df[map_df["path"].map(lambda p: isinstance(p, list) and len(p) > 0)]

bucket_colors = {
    "Never plowed (before 12/1)": [200, 200, 200],
    "> 24 hours": [128, 0, 38],
    "12–24 hours": [227, 26, 28],
    "6–12 hours": [253, 141, 60],
    "1–6 hours": [255, 237, 111],
    "< 1 hour": [44, 162, 95],
}

map_df["color"] = map_df["bucket"].map(
    lambda b: bucket_colors.get(b, [0, 0, 0])
)

if not map_df.empty:
    all_points = [pt for path in map_df["path"] for pt in path]
    mean_lon = sum(p[0] for p in all_points) / len(all_points)
    mean_lat = sum(p[1] for p in all_points) / len(all_points)
else:
    mean_lon, mean_lat = -76.15, 43.05

layer = pdk.Layer(
    "PathLayer",
    data=map_df,
    get_path="path",
    get_color="color",
    width_scale=2,
    width_min_pixels=2,
    pickable=True,
)

view_state = pdk.ViewState(
    longitude=mean_lon,
    latitude=mean_lat,
    zoom=12,
    pitch=0,
)

st.pydeck_chart(
    pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={
            "text": "Road: {roadname}\nBucket: {bucket}\nMiles: {miles:.3f}"
        },
    )
)

with st.expander("Raw data (sample)"):
    st.write(df.head())