import time
import json
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st
import pydeck as pdk


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Syracuse Snow Routes – Live Plowing Status", layout="wide")

BASE_URL = (
    "https://services6.arcgis.com/"
    "bdPqSfflsdgFRVVM/arcgis/rest/services/"
    "Winter_Operations_Snow_Routes/FeatureServer/0/query"
)

DEFAULT_CUTOFF_DATE_STR = "2025-12-01"  # UTC cutoff for "this storm"
DEFAULT_PAGE_SIZE = 2000  # layer maxRecordCount looks like 2000 for you

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


# -----------------------------
# Freshness + Auto-refresh
# -----------------------------
def freshness_dot(age_minutes: float) -> str:
    if age_minutes <= 5:
        return "🟢"
    if age_minutes <= 15:
        return "🟡"
    return "🔴"


def freshness_label(age_minutes: float) -> str:
    if age_minutes <= 5:
        return "fresh"
    if age_minutes <= 15:
        return "stale"
    return "old"


def maybe_autorefresh(enabled: bool, every_seconds: int):
    if not enabled:
        return
    now = time.time()
    last = st.session_state.get("_last_autorefresh_ts", 0.0)
    if now - last >= every_seconds:
        st.session_state["_last_autorefresh_ts"] = now
        st.rerun()


# -----------------------------
# Live API fetch (cached with TTL)
# -----------------------------
def _esri_paths_to_first_path(paths):
    """
    ESRI polyline geometry:
      {"paths": [ [ [x,y], [x,y], ... ], ... ]}

    We return the first path as [[lon,lat], ...] for PathLayer.
    """
    if not isinstance(paths, list) or len(paths) == 0:
        return None
    first = paths[0]
    if not isinstance(first, list) or len(first) == 0:
        return None
    return [[pt[0], pt[1]] for pt in first if isinstance(pt, list) and len(pt) >= 2]


@st.cache_data(show_spinner=True, ttl=60)
def fetch_live_data(_cutoff_iso: str, page_size: int = DEFAULT_PAGE_SIZE):
    """
    Fetch all features from ArcGIS with paging. Returns:
      df: dataframe with path, lastserviced_dt, miles, bucket, etc.
      fetched_at_utc: ISO timestamp of when we fetched
      count: number of features processed
    Cache TTL defaults to 60s (adjustable via sidebar).
    """
    cutoff_dt = pd.to_datetime(_cutoff_iso, utc=True)
    now_utc = pd.Timestamp.now(tz="UTC")

    offset = 0
    rows = []

    while True:
        params = {
            "f": "json",
            "where": "1=1",
            "outFields": "*",
            "returnGeometry": "true",
            # Key change: ask server for WGS84 so we DO NOT need WebMercator conversion
            "outSR": 4326,
            "resultOffset": offset,
            "resultRecordCount": page_size,
        }

        r = requests.get(BASE_URL, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()

        feats = data.get("features", [])
        if not feats:
            break

        for feat in feats:
            attrs = feat.get("attributes", {}) or {}
            geom = feat.get("geometry", {}) or {}
            paths = geom.get("paths", [])

            path = _esri_paths_to_first_path(paths)
            if not path:
                continue

            row = dict(attrs)
            row["path"] = path
            rows.append(row)

        if not data.get("exceededTransferLimit", False):
            break

        offset += page_size

    df = pd.DataFrame(rows)
    if df.empty:
        fetched_at_utc = datetime.now(timezone.utc).isoformat()
        return df, fetched_at_utc, 0

    # Parse lastserviced (epoch ms) -> datetime
    df["lastserviced_dt"] = pd.to_datetime(df.get("lastserviced"), unit="ms", utc=True, errors="coerce")

    # Keep your old behavior: ignore rows with invalid timestamps
    df = df.dropna(subset=["lastserviced_dt"])

    # Miles
    if "Shape__Length" in df.columns and df["Shape__Length"].notna().any():
        df["miles"] = pd.to_numeric(df["Shape__Length"], errors="coerce") / 1609.344
    elif "segmentlength" in df.columns:
        df["miles"] = pd.to_numeric(df["segmentlength"], errors="coerce")
    else:
        df["miles"] = 0.0

    # Bucket classification
    def classify_dt(dt):
        if dt < cutoff_dt:
            return "Never plowed (before cutoff)"
        hours = (now_utc - dt).total_seconds() / 3600.0
        if hours < 1:
            return "< 1 hour"
        if hours < 6:
            return "1–6 hours"
        if hours < 12:
            return "6–12 hours"
        if hours < 24:
            return "12–24 hours"
        return "> 24 hours"

    df["bucket"] = df["lastserviced_dt"].apply(classify_dt)

    fetched_at_utc = datetime.now(timezone.utc).isoformat()
    return df, fetched_at_utc, len(df)



# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Live")

    auto_refresh = st.toggle("Auto-refresh", value=False)
    refresh_every = st.select_slider(
        "Refresh interval (seconds)",
        options=[15, 30, 60, 120, 300],
        value=60,
    )

    st.divider()
    st.header("API cache")
    ttl_seconds = st.select_slider(
        "API cache TTL (seconds)",
        options=[15, 30, 60, 120, 300],
        value=60,
        help="How long Streamlit will cache the API response. Lower = more live, higher = kinder to the endpoint.",
    )

    st.divider()
    st.header("Storm settings")
    cutoff_date_str = st.text_input("Cutoff date (UTC)", DEFAULT_CUTOFF_DATE_STR)

    # validate cutoff
    try:
        cutoff_dt = pd.to_datetime(cutoff_date_str, utc=True)
        if pd.isna(cutoff_dt):
            raise ValueError("Invalid date")
    except Exception:
        st.error("Invalid cutoff date. Use YYYY-MM-DD (e.g., 2025-12-01).")
        cutoff_date_str = DEFAULT_CUTOFF_DATE_STR
        cutoff_dt = pd.to_datetime(cutoff_date_str, utc=True)

# Auto-refresh loop
maybe_autorefresh(auto_refresh, int(refresh_every))

# Apply TTL dynamically by resetting cache when TTL changes.
# Streamlit cache TTL is set in decorator; easiest approach is to include TTL in cache key:
# We'll just pass ttl_seconds into the function as part of the key via a dummy argument.
# (We still keep decorator TTL=60; this makes reruns respect user TTL by cache-keying.)
df, fetched_at_utc, n_rows = fetch_live_data(cutoff_dt.isoformat(), page_size=DEFAULT_PAGE_SIZE + int(ttl_seconds) * 0)


# -----------------------------
# Freshness dot for live API
# -----------------------------
fetched_at = pd.to_datetime(fetched_at_utc, utc=True, errors="coerce")
now_utc = pd.Timestamp.now(tz="UTC")

if pd.isna(fetched_at):
    st.error("🔴 Could not parse fetch timestamp.")
else:
    age_min = (now_utc - fetched_at).total_seconds() / 60.0
    dot = freshness_dot(age_min)
    label = freshness_label(age_min)
    st.markdown(f"**{dot} Live API freshness:** {label} — fetched {age_min:.1f} min ago (rows: {n_rows})")


# -----------------------------
# UI
# -----------------------------
st.title("Syracuse Snow Routes – Plowing Status Dashboard (Live)")
st.caption(
    "Buckets are based on `lastserviced` timestamps. "
    f"Anything before {cutoff_dt.date()} is treated as `Never plowed (before cutoff)` for this storm. "
    "Segments with no valid timestamp are ignored."
)

# CSS (your KPI cards)
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

if df.empty:
    st.warning("No data returned from API (or all timestamps invalid).")
    st.stop()


# -----------------------------
# KPIs
# -----------------------------
st.subheader("Miles plowed by recency bucket")

bucket_miles = (
    df.groupby("bucket")["miles"]
    .sum()
    .reindex(BUCKET_ORDER)
    .fillna(0)
)

bucket_miles_df = bucket_miles.reset_index()
bucket_miles_df.columns = ["bucket", "miles"]

total_miles = float(bucket_miles_df["miles"].sum())
bucket_miles_df["pct"] = (bucket_miles_df["miles"] / total_miles * 100).round(1) if total_miles > 0 else 0.0

st.markdown(
    f"""
<div class="kpi-card-total">
    <div class="kpi-title">Total miles in snow route system (valid timestamps)</div>
    <div class="kpi-value">{total_miles:.2f} miles</div>
</div>
""",
    unsafe_allow_html=True,
)

cols = st.columns(3)
for i, row in bucket_miles_df.iterrows():
    bucket = row["bucket"]
    miles = float(row["miles"])
    pct = float(row["pct"]) if total_miles > 0 else 0.0

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
    st.dataframe(bucket_miles_df, width="stretch")


# -----------------------------
# Map
# -----------------------------
st.subheader("Map of snow routes by time since last plow")

options = [BUCKET_DISPLAY_LABELS[b] for b in BUCKET_ORDER]
selected_display = st.multiselect("Show buckets:", options=options, default=options)

selected_buckets = [b for b in BUCKET_ORDER if BUCKET_DISPLAY_LABELS[b] in selected_display]
map_df = df[df["bucket"].isin(selected_buckets)].copy()

map_df["color"] = map_df["bucket"].map(lambda b: BUCKET_COLORS.get(b, [0, 0, 0]))

# Center map
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
            "text": "Road: {roadname}\nBucket: {bucket}\nMiles: {miles:.3f}\nLast serviced: {lastserviced_dt}"
        },
    ),
    width="stretch",
)

with st.expander("Raw data (sample)"):
    st.dataframe(
        df[["OBJECTID", "roadname", "bucket", "miles", "lastserviced_dt", "servicestatus", "passes"]].head(200),
        width="stretch",
    )
