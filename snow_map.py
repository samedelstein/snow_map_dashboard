import time
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import pydeck as pdk


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Syracuse Snow Routes – Plowing Status",
    layout="wide",
)


# -----------------------------
# Paths + constants
# -----------------------------
LATEST_GEOJSON_PATH = Path("snapshot_snow_routes/latest_routes.geojson")
SNAPSHOTS_CSV_PATH = Path("snapshot_snow_routes/snapshots.csv")

DEFAULT_CUTOFF_DATE_STR = "2025-12-01"  # UTC cutoff for "this storm"

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
# Freshness + Auto-refresh utils
# -----------------------------
def file_mtime_utc(p: Path):
    """Return file modified time as a UTC Timestamp, or None if missing."""
    if not p.exists():
        return None
    return pd.to_datetime(p.stat().st_mtime, unit="s", utc=True)


def freshness_dot(age_minutes: float) -> str:
    # adjust thresholds if you want (these are sensible for 5-min snapshots)
    if age_minutes <= 10:
        return "🟢"
    if age_minutes <= 30:
        return "🟡"
    return "🔴"


def freshness_label(age_minutes: float) -> str:
    if age_minutes <= 10:
        return "fresh"
    if age_minutes <= 30:
        return "stale"
    return "old"


def maybe_autorefresh(enabled: bool, every_seconds: int):
    """
    If enabled, rerun the app every N seconds.
    Uses Streamlit rerun; no external deps.
    """
    if not enabled:
        return

    now = time.time()
    last = st.session_state.get("_last_autorefresh_ts", 0.0)

    if now - last >= every_seconds:
        st.session_state["_last_autorefresh_ts"] = now
        st.rerun()


# -----------------------------
# Cache-busting key
# -----------------------------
def mtime_key(p: Path) -> float:
    """Float key used to bust st.cache_data when a file changes."""
    return p.stat().st_mtime if p.exists() else 0.0


# -----------------------------
# Data loading (cached)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_history(_mtime_key: float) -> pd.DataFrame:
    """Load snapshots.csv and parse key timestamps."""
    if not SNAPSHOTS_CSV_PATH.exists():
        return pd.DataFrame()

    hist = pd.read_csv(SNAPSHOTS_CSV_PATH)
    hist["snapshot_ts"] = pd.to_datetime(hist.get("snapshot_ts"), utc=True, errors="coerce")

    # ESRI epoch milliseconds -> datetime
    if "lastserviced" in hist.columns:
        hist["lastserviced_dt"] = pd.to_datetime(hist["lastserviced"], unit="ms", utc=True, errors="coerce")
    else:
        hist["lastserviced_dt"] = pd.NaT

    return hist


@st.cache_data(show_spinner=True)
def load_geojson_df(_mtime_key: float) -> tuple[pd.DataFrame, dict]:
    """
    Load GeoJSON and build a dataframe with:
      - properties
      - path (for pydeck PathLayer)
      - lastserviced_dt
      - miles
      - bucket
    Also returns the loaded FeatureCollection dict for optional metadata display.
    """
    if not LATEST_GEOJSON_PATH.exists():
        return pd.DataFrame(), {}

    with open(LATEST_GEOJSON_PATH, "r", encoding="utf-8") as f:
        fc = json.load(f)

    features = fc.get("features", [])
    rows = []

    for feat in features:
        props = feat.get("properties", {}) or {}
        geom = feat.get("geometry", {}) or {}

        coords = geom.get("coordinates", [])
        path = coords[0] if isinstance(coords, list) and len(coords) > 0 else None

        rows.append({**props, "path": path})

    df = pd.DataFrame(rows)
    if df.empty:
        return df, fc

    # Time handling
    df["lastserviced_dt"] = pd.to_datetime(df.get("lastserviced"), unit="ms", utc=True, errors="coerce")

    # Miles
    if "Shape__Length" in df.columns and df["Shape__Length"].notna().any():
        df["miles"] = pd.to_numeric(df["Shape__Length"], errors="coerce") / 1609.344
    elif "segmentlength" in df.columns:
        df["miles"] = pd.to_numeric(df["segmentlength"], errors="coerce")
    else:
        df["miles"] = 0.0

    # Remove missing/empty paths
    df = df.dropna(subset=["path"])
    df = df[df["path"].map(lambda p: isinstance(p, list) and len(p) > 0)]

    return df, fc


def bucketize(df: pd.DataFrame, cutoff_dt: pd.Timestamp) -> pd.DataFrame:
    """Add/overwrite df['bucket'] based on df['lastserviced_dt']."""
    now = pd.Timestamp.now(tz="UTC")

    def classify(dt: pd.Timestamp) -> str:
        if pd.isna(dt):
            # You previously dropped these. Keep dropping behavior by returning None,
            # and we will dropna later.
            return None
        if dt < cutoff_dt:
            return "Never plowed (before cutoff)"
        hours = (now - dt).total_seconds() / 3600.0
        if hours < 1:
            return "< 1 hour"
        if hours < 6:
            return "1–6 hours"
        if hours < 12:
            return "6–12 hours"
        if hours < 24:
            return "12–24 hours"
        return "> 24 hours"

    df = df.copy()
    df["bucket"] = df["lastserviced_dt"].apply(classify)
    df = df.dropna(subset=["bucket"])  # keeps your existing behavior: ignore missing timestamps
    return df


# -----------------------------
# Sidebar controls (refresh + cutoff)
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
    st.header("Storm settings")
    cutoff_date_str = st.text_input("Cutoff date (UTC)", DEFAULT_CUTOFF_DATE_STR)

    try:
        CUTOFF_DATE = pd.to_datetime(cutoff_date_str, utc=True)
        if pd.isna(CUTOFF_DATE):
            raise ValueError("Invalid date")
    except Exception:
        st.error("Invalid cutoff date. Use YYYY-MM-DD (e.g., 2025-12-01).")
        CUTOFF_DATE = pd.to_datetime(DEFAULT_CUTOFF_DATE_STR, utc=True)


# Trigger rerun loop if enabled
maybe_autorefresh(auto_refresh, int(refresh_every))


# -----------------------------
# Freshness banner (🟢🟡🔴)
# -----------------------------
now_utc = pd.Timestamp.now(tz="UTC")
geo_mtime = file_mtime_utc(LATEST_GEOJSON_PATH)
csv_mtime = file_mtime_utc(SNAPSHOTS_CSV_PATH)

available = [t for t in [geo_mtime, csv_mtime] if t is not None]
if not available:
    st.error("🔴 Data files missing — check Streamlit Cloud is pulling the latest repo.")
else:
    newest = max(available)
    age_min = (now_utc - newest).total_seconds() / 60.0
    dot = freshness_dot(age_min)
    label = freshness_label(age_min)

    detail = ""
    if geo_mtime is not None and csv_mtime is not None:
        geo_age = (now_utc - geo_mtime).total_seconds() / 60.0
        csv_age = (now_utc - csv_mtime).total_seconds() / 60.0
        detail = f" (geojson: {geo_age:.1f}m, csv: {csv_age:.1f}m)"

    st.markdown(f"**{dot} Data freshness:** {label} — {age_min:.1f} min old{detail}")


# -----------------------------
# Load data (cache busting via mtime)
# -----------------------------
geo_df, fc = load_geojson_df(mtime_key(LATEST_GEOJSON_PATH))
hist = load_history(mtime_key(SNAPSHOTS_CSV_PATH))

# Optional: show “generated_at_utc” if you add it to the geojson in the snapshot script
generated_at = fc.get("generated_at_utc") if isinstance(fc, dict) else None
latest_snapshot_ts = None
if not hist.empty and "snapshot_ts" in hist.columns:
    latest_snapshot_ts = hist["snapshot_ts"].max()

cols = st.columns(3)
cols[0].metric("GeoJSON mtime (UTC)", str(geo_mtime) if geo_mtime is not None else "missing")
cols[1].metric("CSV mtime (UTC)", str(csv_mtime) if csv_mtime is not None else "missing")
cols[2].metric("Latest snapshot_ts (CSV)", str(latest_snapshot_ts) if latest_snapshot_ts is not None else "missing")

if generated_at:
    st.caption(f"GeoJSON generated_at_utc: {generated_at}")

if geo_df.empty:
    st.error(f"Missing/empty GeoJSON: {LATEST_GEOJSON_PATH}")
    st.stop()

# Bucketize after loading
df = bucketize(geo_df, CUTOFF_DATE)


# -----------------------------
# UI header
# -----------------------------
st.title("Syracuse Snow Routes – Plowing Status Dashboard")
st.caption(
    "Buckets are based on `lastserviced` timestamps (epoch ms). "
    f"Anything before {CUTOFF_DATE.date()} is treated as `Never plowed (before cutoff)` for this storm. "
    "Segments with no valid timestamp are ignored (dropped)."
)

# -----------------------------
# CSS (your existing KPI styles)
# -----------------------------
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


# -----------------------------
# KPI section
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
    st.dataframe(bucket_miles_df, use_container_width=True)


# -----------------------------
# Map section
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
            "text": "Road: {roadname}\nBucket: {bucket}\nMiles: {miles}"
        },
    ),
    use_container_width=True,
)

with st.expander("Raw data (sample)"):
    st.dataframe(df.head(200), use_container_width=True)
