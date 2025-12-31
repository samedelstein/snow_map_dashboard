import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

from bucket_copy import BUCKET_EXPLAINER_TEXT

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Syracuse Snow Routes – Live + Predictions", layout="wide")

BASE_URL = (
    "https://services6.arcgis.com/"
    "bdPqSfflsdgFRVVM/arcgis/rest/services/"
    "Winter_Operations_Snow_Routes/FeatureServer/0/query"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = DATA_DIR / "artifacts_snow"
PREDICTIONS_PROB_PATH = ARTIFACTS_DIR / "predictions_latest_prob.csv"
METRICS_PATH = ARTIFACTS_DIR / "model_metrics_prob.json"
GEOJSON_PATH = DATA_DIR / "snapshot_snow_routes/latest_routes.geojson"
SNAPSHOT_PATH = DATA_DIR / "snapshot_snow_routes/snapshots.csv"
NWS_ALERTS_PATH = ARTIFACTS_DIR / "nws_alerts_log.csv"

DEFAULT_CUTOFF_DATE_STR = "2025-12-01"  # UTC cutoff for "this storm"
DEFAULT_PAGE_SIZE = 2000  # layer maxRecordCount looks like 2000 for you

BUCKET_ORDER = [
    "Never plowed (before cutoff)",
    "Unknown timestamp",
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
    "Unknown timestamp": [110, 110, 110],  # dark gray
    "> 24 hours": [128, 0, 38],  # dark red
    "12–24 hours": [227, 26, 28],  # red
    "6–12 hours": [253, 141, 60],  # orange
    "1–6 hours": [255, 237, 111],  # yellow
    "< 1 hour": [44, 162, 95],  # green-ish
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


@st.cache_data(show_spinner=True, ttl=3600)
def fetch_live_data(_cutoff_iso: str, page_size: int = DEFAULT_PAGE_SIZE, cache_bust: int = 0):
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

    # Keep rows with invalid timestamps; we'll bucket them as 'Unknown timestamp'

    # Miles
    if "Shape__Length" in df.columns and df["Shape__Length"].notna().any():
        df["miles"] = pd.to_numeric(df["Shape__Length"], errors="coerce") / 1609.344
    elif "segmentlength" in df.columns:
        df["miles"] = pd.to_numeric(df["segmentlength"], errors="coerce")
    else:
        df["miles"] = 0.0

    # Bucket classification
    def classify_dt(dt):
        if dt is None or pd.isna(dt):
            return "Unknown timestamp"
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

    if "snowroutesegmentid" in df.columns:
        df["snowroutesegmentid"] = df["snowroutesegmentid"].astype(str)

    fetched_at_utc = datetime.now(timezone.utc).isoformat()
    return df, fetched_at_utc, len(df)


@st.cache_data(show_spinner=False)
def load_latest_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        return df

    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], utc=True, errors="coerce")
    latest_ts = df["snapshot_ts"].max()
    if pd.isna(latest_ts):
        return pd.DataFrame()

    latest = df[df["snapshot_ts"] == latest_ts].copy()
    if "snowroutesegmentid" in latest.columns:
        latest["snowroutesegmentid"] = latest["snowroutesegmentid"].astype(str)
    keep_cols = [
        "OBJECTID",
        "snowroutesegmentid",
        "p_1h",
        "p_2h",
        "p_4h",
        "p_8h",
        "p_ml_1h",
        "p_ml_2h",
        "p_ml_4h",
        "p_ml_8h",
        "p_bucket_1h",
        "p_bucket_2h",
        "p_bucket_4h",
        "p_bucket_8h",
        "eta_bucket_hours",
        "current_bucket_priority",
        "bucket_priority_id",
        "eta_hours_60",
        "eta_ts_60",
        "eta_hours_pred",
    ]
    return latest[[c for c in keep_cols if c in latest.columns]]


@st.cache_data(show_spinner=False)
def load_nws_alerts(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        return df

    df["start_ts"] = pd.to_datetime(df.get("start_ts"), utc=True, errors="coerce")
    df["end_ts"] = pd.to_datetime(df.get("end_ts"), utc=True, errors="coerce")
    return df


@st.cache_data(ttl=300)
def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PROB_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(PREDICTIONS_PROB_PATH)

    for c in ["snapshot_ts", "lastserviced", "eta_ts_60"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")

    prob_cols = [
        "p_1h",
        "p_2h",
        "p_4h",
        "p_8h",
        "p_bucket_1h",
        "p_bucket_2h",
        "p_bucket_4h",
        "p_bucket_8h",
        "p_ml_1h",
        "p_ml_2h",
        "p_ml_4h",
        "p_ml_8h",
    ]
    for c in prob_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if {"p_1h", "p_2h", "p_4h", "p_8h"}.issubset(df.columns):
        df["p_2h"] = df[["p_1h", "p_2h"]].max(axis=1)
        df["p_4h"] = df[["p_2h", "p_4h"]].max(axis=1)
        df["p_8h"] = df[["p_4h", "p_8h"]].max(axis=1)

    if "snowroutesegmentid" in df.columns:
        df["snowroutesegmentid"] = df["snowroutesegmentid"].astype(str)

    if "bucket_priority_id" not in df.columns and "routepriority" in df.columns:
        df["bucket_priority_id"] = df["routepriority"].apply(derive_bucket_priority_id)

    return df


@st.cache_data(ttl=300)
def load_model_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data(ttl=3600)
def load_geojson() -> dict:
    if not GEOJSON_PATH.exists():
        return {}
    return json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))


@st.cache_data(ttl=300)
def load_snapshot_event_ranges() -> pd.DataFrame:
    if not SNAPSHOT_PATH.exists():
        return pd.DataFrame()

    s = pd.read_csv(SNAPSHOT_PATH, usecols=["eventid", "snapshot_ts"])
    s["snapshot_ts"] = pd.to_datetime(s["snapshot_ts"], utc=True, errors="coerce")
    s = s.dropna(subset=["eventid", "snapshot_ts"])

    g = s.groupby("eventid")["snapshot_ts"].agg(["min", "max", "count"]).reset_index()
    g = g.sort_values("max")
    return g


def attach_predictions_to_geojson(geojson_obj: dict, preds: pd.DataFrame, eventid: str, color_by: str):
    """
    Adds prediction fields + precomputed _line_color into each feature's properties.

    If eventid == "AUTO", uses each feature's own properties.eventid to select predictions.
    Otherwise uses the selected eventid for all features.
    """
    if not geojson_obj or preds.empty:
        return geojson_obj

    preds2 = preds.copy()
    preds2["snowroutesegmentid"] = preds2["snowroutesegmentid"].astype(str)
    pred_index = preds2.set_index(["eventid", "snowroutesegmentid"])

    def prob_to_color(prob):
        if prob is None or (isinstance(prob, float) and pd.isna(prob)):
            return [120, 120, 120, 160]
        prob = float(prob)
        prob = min(max(prob, 0.0), 1.0)
        r = int(255 * (1 - prob))
        g = int(255 * prob)
        return [r, g, 60, 220]

    tier_colors = {
        "Now (≤1h)": [44, 162, 95, 220],
        "Soon (1–4h)": [253, 187, 132, 220],
        "Later today (4–8h)": [252, 141, 89, 220],
        "Overnight (8–12h)": [227, 74, 51, 220],
        "Next day (12–24h)": [179, 0, 0, 220],
        "Later (24h+)": [84, 39, 143, 220],
        "Unknown": [120, 120, 120, 160],
    }

    fields = [
        "prediction_status",
        "bucket_priority_id",
        "bucket_id",
        "current_bucket_priority",
        "current_bucket_id",
        "eta_bucket_hours",
        "p_bucket_1h",
        "p_bucket_2h",
        "p_bucket_4h",
        "p_bucket_8h",
        "p_ml_1h",
        "p_ml_2h",
        "p_ml_4h",
        "p_ml_8h",
        "eta_hours_60",
        "p_1h",
        "p_2h",
        "p_4h",
        "p_8h",
        "routepriority",
        "snowrouteid",
        "roadname",
    ]

    for feat in geojson_obj.get("features", []):
        props = feat.get("properties", {}) or {}
        seg = str(props.get("snowroutesegmentid"))
        feat_event = props.get("eventid")

        lookup_event = feat_event if eventid == "AUTO" else eventid

        props["prediction_status"] = props.get("prediction_status") or "NO_PRED_INSUFFICIENT_DATA"
        props["_line_color"] = [120, 120, 120, 160]

        key = (lookup_event, seg)
        if key in pred_index.index:
            row = pred_index.loc[key]
            for f in fields:
                if f in row.index:
                    val = row[f]
                    if pd.isna(val):
                        val = None
                    elif hasattr(val, "isoformat"):
                        val = val.isoformat()
                    props[f] = val

            status = props.get("prediction_status")
            if status == "NO_PRED_UNTRACKED":
                props["_line_color"] = [160, 160, 160, 180]
            else:
                if color_by == "bucket_eta_tier":
                    eta_value = props.get("eta_bucket_hours") or props.get("eta_hours_60")
                    tier = bucket_eta_tier(eta_value)
                    props["_line_color"] = tier_colors.get(tier, tier_colors["Unknown"])
                else:
                    props["_line_color"] = prob_to_color(row.get(color_by))

        props["route_label"] = format_route_label(props.get("snowrouteid"))
        props["p_1h_pct"] = format_probability(props.get("p_1h"))
        props["p_2h_pct"] = format_probability(props.get("p_2h"))
        props["p_4h_pct"] = format_probability(props.get("p_4h"))
        props["p_8h_pct"] = format_probability(props.get("p_8h"))
        props["p_bucket_1h_pct"] = format_probability(props.get("p_bucket_1h"))
        props["p_bucket_2h_pct"] = format_probability(props.get("p_bucket_2h"))
        props["p_bucket_4h_pct"] = format_probability(props.get("p_bucket_4h"))
        props["p_bucket_8h_pct"] = format_probability(props.get("p_bucket_8h"))
        props["p_ml_1h_pct"] = format_probability(props.get("p_ml_1h"))
        props["p_ml_2h_pct"] = format_probability(props.get("p_ml_2h"))
        props["p_ml_4h_pct"] = format_probability(props.get("p_ml_4h"))
        props["p_ml_8h_pct"] = format_probability(props.get("p_ml_8h"))
        props["bucket_priority_label"] = format_bucket_priority(props.get("bucket_priority_id"))
        props["current_bucket_priority_label"] = format_bucket_priority(
            props.get("current_bucket_priority")
        )
        props["eta_bucket_label"] = format_eta_hours(props.get("eta_bucket_hours"))
        props["bucket_eta_tier"] = bucket_eta_tier(
            props.get("eta_bucket_hours") or props.get("eta_hours_60")
        )
        props["eta_60_label"] = format_eta_hours(props.get("eta_hours_60"))
        feat["properties"] = props

    return geojson_obj


def format_probability(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.0f}%"


def format_eta_hours(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.1f} hours"


def format_route_label(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    try:
        return f"Route {int(value)}"
    except (TypeError, ValueError):
        return str(value)


def format_bucket_priority(value: str | float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    label = str(value)
    if label.lower() in {"nan", "none", ""}:
        return "N/A"
    if label.startswith("P") and label[1:].isdigit():
        return f"Priority {label[1:]}"
    return label


def bucket_eta_tier(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    hours = float(value)
    if hours <= 1:
        return "Now (≤1h)"
    if hours <= 4:
        return "Soon (1–4h)"
    if hours <= 8:
        return "Later today (4–8h)"
    if hours <= 12:
        return "Overnight (8–12h)"
    if hours <= 24:
        return "Next day (12–24h)"
    return "Later (24h+)"


def format_storm_start(start_ts: pd.Timestamp | None) -> str:
    if start_ts is None or pd.isna(start_ts):
        return "N/A"
    return pd.to_datetime(start_ts).strftime("%Y-%m-%d %H:%M UTC")


def derive_bucket_priority_id(value: str | float | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value)
    if text.startswith("P") and text[1:].isdigit():
        return text
    digits = "".join(ch for ch in text if ch.isdigit())
    return f"P{digits}" if digits else None



# -----------------------------
# Command Center UI (single view)
# -----------------------------

st.set_page_config(page_title="Snow Operations Command Center", layout="wide")

# Seed session state for manual refresh nonce
if "_force_refresh_nonce" not in st.session_state:
    st.session_state["_force_refresh_nonce"] = 0

# -----------------------------
# Sidebar: global controls
# -----------------------------
with st.sidebar:
    st.title("❄️ Snow Ops")

    view_mode = st.radio(
        "Mode",
        options=["Live recency", "Predictions"],
        index=0,
        help="Live uses ArcGIS last-serviced timestamps. Predictions uses the latest model outputs.",
    )

    st.divider()

    auto_refresh = st.toggle("Auto-refresh", value=False)
    refresh_every = st.select_slider(
        "Refresh interval (seconds)",
        options=[15, 30, 60, 120, 300],
        value=60,
    )

    cols = st.columns([1, 1])
    with cols[0]:
        if st.button("Refresh now", use_container_width=True):
            st.session_state["_force_refresh_nonce"] += 1
            st.rerun()
    with cols[1]:
        st.caption(" ")

    with st.expander("Advanced", expanded=False):
        st.markdown("**Live data**")
        ttl_seconds = st.select_slider(
            "Cache window (seconds)",
            options=[15, 30, 60, 120, 300],
            value=60,
            help="We re-fetch ArcGIS when this window rolls over, or when you click Refresh now.",
        )
        cutoff_date_str = st.text_input(
            "Storm cutoff date (UTC)",
            value=DEFAULT_CUTOFF_DATE_STR,
            help="Anything before this is treated as 'Never plowed (before cutoff)'.",
        )

        st.divider()
        st.markdown("**Predictions**")
        color_by = st.selectbox(
            "Color by",
            options=[
                "p_ml_1h",
                "p_ml_2h",
                "p_ml_4h",
                "p_ml_8h",
                "bucket_eta_tier",
            ],
            index=2,
            help="Controls the prediction map colors.",
        )
        min_prob = st.slider(
            "Min probability (ranked list)",
            0.0, 1.0, 0.0, 0.05,
            help="Filters the ranked list and selection menu in Predictions mode.",
        )

# Auto-refresh loop
maybe_autorefresh(auto_refresh, int(refresh_every))

# Parse cutoff date safely
try:
    cutoff_dt = pd.to_datetime(cutoff_date_str, utc=True)
except Exception:
    cutoff_dt = pd.to_datetime(DEFAULT_CUTOFF_DATE_STR, utc=True)
    st.sidebar.warning("Invalid cutoff date; using default.")

# -----------------------------
# Header
# -----------------------------
st.title("Snow Operations Command Center")
st.caption("Live plow activity + model-based next-service probabilities (single view).")

# -----------------------------
# Layout: filters row + map + details drawer
# -----------------------------
filters = st.container()
main_left, main_right = st.columns([3.2, 1.3], gap="large")

# -----------------------------
# Helper: build a consistent set of filters (road / priority / route)
# -----------------------------
def apply_common_filters(df_in: pd.DataFrame, road_q: str, pri_sel: list, route_sel: list) -> pd.DataFrame:
    out = df_in.copy()
    if "routepriority" in out.columns and pri_sel:
        out = out[out["routepriority"].isin(pri_sel)]
    if "snowrouteid" in out.columns and route_sel:
        out = out[out["snowrouteid"].isin(route_sel)]
    if road_q and "roadname" in out.columns:
        out = out[out["roadname"].astype(str).str.contains(road_q, case=False, na=False)]
    return out

# -----------------------------
# LIVE MODE
# -----------------------------
if view_mode == "Live recency":
    # Cache-bust value changes each ttl window, plus manual refresh nonce.
    cache_bust = int(time.time() // int(ttl_seconds)) + int(st.session_state["_force_refresh_nonce"])
    df, fetched_at_utc, n_rows = fetch_live_data(
        cutoff_dt.isoformat(),
        page_size=DEFAULT_PAGE_SIZE,
        cache_bust=cache_bust,
    )

    # Attach latest predictions if available (optional)
    predictions_df = load_latest_predictions(PREDICTIONS_PROB_PATH)
    if not predictions_df.empty:
        if "OBJECTID" in df.columns and "OBJECTID" in predictions_df.columns:
            df = df.merge(predictions_df, on="OBJECTID", how="left")
        elif "snowroutesegmentid" in df.columns and "snowroutesegmentid" in predictions_df.columns:
            df = df.merge(predictions_df, on="snowroutesegmentid", how="left")

    if df.empty:
        st.warning("No live data returned from ArcGIS.")
        st.stop()

    # Common filter options
    pri_opts = sorted([p for p in df.get("routepriority", pd.Series(dtype=object)).dropna().unique()])
    route_opts = sorted([r for r in df.get("snowrouteid", pd.Series(dtype=object)).dropna().unique()])

    # Filter row (consistent with predictions)
    with filters:
        c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.1, 1.6], gap="small")
        with c1:
            pri_sel = st.multiselect("Priority", options=pri_opts, default=pri_opts)
        with c2:
            route_sel = st.multiselect("Route", options=route_opts, default=[])
        with c3:
            road_q = st.text_input("Road search", value="")
        with c4:
            # Bucket filter lives here (not sidebar)
            bucket_opts = [BUCKET_DISPLAY_LABELS[b] for b in BUCKET_ORDER]
            selected_display = st.multiselect("Buckets", options=bucket_opts, default=bucket_opts)

    selected_buckets = [b for b in BUCKET_ORDER if BUCKET_DISPLAY_LABELS[b] in selected_display] or BUCKET_ORDER[:]
    view_df = apply_common_filters(df, road_q, pri_sel, route_sel)
    map_df = view_df[view_df["bucket"].isin(selected_buckets)].copy()

    # KPIs
    total_miles = float(view_df.get("miles", pd.Series([0.0])).fillna(0).sum())
    st_metrics = st.container()
    with st_metrics:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Segments (filtered)", f"{len(view_df):,}")
        k2.metric("Miles (filtered)", f"{total_miles:,.1f}")
        k3.metric("Last fetch", pd.to_datetime(fetched_at_utc).strftime("%H:%M:%S UTC"))
        # Coverage is % with a usable timestamp
        ts_cov = 100.0 * (1.0 - float(view_df["lastserviced_dt"].isna().mean())) if "lastserviced_dt" in view_df.columns else 0.0
        k4.metric("Timestamp coverage", f"{ts_cov:.1f}%")

    # Enrich for map tooltip fields (reuse existing formatting helpers)
    for col in [
        "p_1h", "p_2h", "p_4h", "p_8h",
        "p_bucket_1h", "p_bucket_2h", "p_bucket_4h", "p_bucket_8h",
        "p_ml_1h", "p_ml_2h", "p_ml_4h", "p_ml_8h",
        "eta_bucket_hours", "current_bucket_priority", "bucket_priority_id",
        "eta_hours_60", "eta_ts_60", "eta_hours_pred",
    ]:
        if col not in map_df.columns:
            map_df[col] = None

    map_df["route_label"] = map_df.get("snowrouteid", "").apply(format_route_label) if "snowrouteid" in map_df.columns else "N/A"
    map_df["p_bucket_4h_pct"] = map_df["p_bucket_4h"].apply(format_probability)
    map_df["p_ml_4h_pct"] = map_df["p_ml_4h"].apply(format_probability)
    map_df["eta_bucket_label"] = map_df["eta_bucket_hours"].apply(format_eta_hours)
    map_df["bucket_priority_label"] = map_df["bucket_priority_id"].apply(format_bucket_priority)
    map_df["current_bucket_priority_label"] = map_df["current_bucket_priority"].apply(format_bucket_priority)
    map_df["bucket_eta_tier"] = map_df["eta_bucket_hours"].apply(bucket_eta_tier)

    if "lastserviced_dt" in map_df.columns:
        map_df["lastserviced_label"] = map_df["lastserviced_dt"].dt.strftime("%Y-%m-%d %H:%M UTC")
    else:
        map_df["lastserviced_label"] = "N/A"

    map_df["color"] = map_df["bucket"].map(lambda b: BUCKET_COLORS.get(b, [0, 0, 0]))

    # Selection mechanism (table/selectbox) + optional highlight
    # Build a compact label to avoid huge select lists.
    sel_candidates = map_df.copy()
    # Prefer showing the most stale / never / unknown first
    bucket_rank = {b: i for i, b in enumerate(BUCKET_ORDER)}
    sel_candidates["_bucket_rank"] = sel_candidates["bucket"].map(bucket_rank).fillna(9999)
    sel_candidates = sel_candidates.sort_values(["_bucket_rank", "roadname"], ascending=[True, True]).head(400)

    def live_label(row) -> str:
        rn = str(row.get("roadname", ""))
        rt = str(row.get("route_label", ""))
        b = str(row.get("bucket", ""))
        return f"{rn} • {rt} • {b}"

    sel_candidates["_label"] = sel_candidates.apply(live_label, axis=1)
    selection_options = sel_candidates["_label"].tolist()

    with main_right:
        st.subheader("Details")
        if not selection_options:
            st.info("No segments match your filters.")
            selected_label = None
        else:
            selected_label = st.selectbox("Select a segment", options=selection_options, index=0)

        if selected_label:
            row = sel_candidates[sel_candidates["_label"] == selected_label].iloc[0].to_dict()
            st.metric("Bucket", row.get("bucket", "—"))
            st.metric("Last serviced", row.get("lastserviced_label", "—"))
            st.metric("Bucket ETA", row.get("eta_bucket_label", "—"))
            st.metric("Chance ≤4h (Bucket)", row.get("p_bucket_4h_pct", "—"))
            st.metric("Chance ≤4h (Model)", row.get("p_ml_4h_pct", "—"))
            st.caption(f"Priority: {row.get('bucket_priority_label','—')} • Current: {row.get('current_bucket_priority_label','—')}")
        with st.expander("NWS alerts", expanded=False):
            alerts_df = load_nws_alerts(NWS_ALERTS_PATH)
            if alerts_df.empty:
                st.caption("No alerts loaded yet.")
            else:
                now_utc = pd.Timestamp.now(tz="UTC")
                active_alerts = alerts_df[
                    (alerts_df["start_ts"].isna() | (alerts_df["start_ts"] <= now_utc))
                    & (alerts_df["end_ts"].isna() | (alerts_df["end_ts"] >= now_utc))
                ].copy()
                display_df = active_alerts if not active_alerts.empty else alerts_df
                display_df = display_df.sort_values("start_ts", ascending=False).head(10)
                st.dataframe(display_df[["event", "severity", "start_ts", "end_ts", "source_url"]], width="stretch")

    with main_left:
        st.subheader("Map")
        if map_df.empty:
            st.warning("No segments match your filters/buckets.")
        else:
            layers = [
                pdk.Layer(
                    "PathLayer",
                    data=map_df,
                    get_path="path",
                    get_color="color",
                    width_scale=15,
                    width_min_pixels=2,
                    pickable=True,
                )
            ]

            # Highlight selected
            if selected_label:
                sel_row = sel_candidates[sel_candidates["_label"] == selected_label].iloc[0]
                highlight_df = pd.DataFrame([sel_row])
                layers.append(
                    pdk.Layer(
                        "PathLayer",
                        data=highlight_df,
                        get_path="path",
                        get_color=[0, 0, 0],
                        width_scale=60,
                        width_min_pixels=5,
                        pickable=False,
                    )
                )

            # View state: center on Syracuse
            view_state = pdk.ViewState(
                latitude=43.0481,
                longitude=-76.1474,
                zoom=11.6,
                pitch=0,
            )

            st.pydeck_chart(
                pdk.Deck(
                    layers=layers,
                    initial_view_state=view_state,
                    tooltip={
                        "text": (
                            "{roadname}\n"
                            "Route: {route_label}\n"
                            "Bucket: {bucket}\n"
                            "Last plowed at: {lastserviced_label}\n"
                            "Bucket (typical): {bucket_priority_label}\n"
                            "Current plow bucket: {current_bucket_priority_label}\n"
                            "Bucket ETA: {eta_bucket_label}\n"
                            "Bucket chance ≤4h: {p_bucket_4h_pct}\n"
                            "Model chance ≤4h: {p_ml_4h_pct}\n"
                        )
                    },
                ),
                width="stretch",
            )

        with st.expander("Ranked list (filtered)", expanded=False):
            cols = [c for c in [
                "roadname", "route_label", "bucket", "lastserviced_label", "miles",
                "bucket_priority_label", "current_bucket_priority_label",
                "eta_bucket_label", "p_bucket_4h_pct", "p_ml_4h_pct"
            ] if c in map_df.columns]
            st.dataframe(map_df[cols].sort_values(["bucket"], ascending=True).head(800), width="stretch")

# -----------------------------
# PREDICTIONS MODE
# -----------------------------
else:
    preds = load_predictions()
    metrics = load_model_metrics()
    geojson_obj = load_geojson()

    if preds.empty:
        st.warning("No predictions file found. Expected at: data/artifacts_snow/predictions_latest_prob.csv")
        st.stop()

    # Event selection (kept, but not as a separate page)
    event_ranges = load_snapshot_event_ranges()
    if event_ranges.empty:
        event_opts = sorted([e for e in preds["eventid"].dropna().unique()])
        selected_event = st.selectbox("Event", options=event_opts, index=len(event_opts) - 1 if event_opts else 0)
        storm_start_label = "—"
    else:
        event_ranges = event_ranges.sort_values("start_ts")
        event_opts = event_ranges["eventid"].tolist()
        default_idx = len(event_opts) - 1 if event_opts else 0
        selected_event = st.selectbox("Event", options=event_opts, index=default_idx)
        storm_start = event_ranges[event_ranges["eventid"] == selected_event]["start_ts"].iloc[0]
        storm_start_label = format_storm_start(storm_start)

    # Filter options
    pri_opts = sorted([p for p in preds.get("routepriority", pd.Series(dtype=object)).dropna().unique()])
    route_opts = sorted([r for r in preds.get("snowrouteid", pd.Series(dtype=object)).dropna().unique()])

    with filters:
        c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.1, 1.6], gap="small")
        with c1:
            pri_sel = st.multiselect("Priority", options=pri_opts, default=pri_opts)
        with c2:
            route_sel = st.multiselect("Route", options=route_opts, default=[])
        with c3:
            road_q = st.text_input("Road search", value="")
        with c4:
            horizon = st.selectbox("Horizon", options=["1h", "2h", "4h", "8h"], index=2)

    view = preds[preds["eventid"] == selected_event].copy()
    view = apply_common_filters(view, road_q, pri_sel, route_sel)

    # Select the probability column based on horizon
    prob_col = {"1h": "p_ml_1h", "2h": "p_ml_2h", "4h": "p_ml_4h", "8h": "p_ml_8h"}[horizon]
    view["_prob"] = pd.to_numeric(view.get(prob_col), errors="coerce")
    view = view[view["_prob"].fillna(0) >= float(min_prob)].copy()

    # Attach predictions to geojson for map layer coloring
    geo_for_map = None
    match_summary = {"matched": 0, "total": 0, "pct": 0.0}
    if geojson_obj:
        geo_copy = copy.deepcopy(geojson_obj)
        geo_for_map = attach_predictions_to_geojson(geo_copy, preds, selected_event, color_by=color_by)
        features = geo_for_map.get("features", [])
        match_summary["total"] = len(features)
        match_summary["matched"] = sum(
            1 for feat in features
            if (feat.get("properties", {}) or {}).get("prediction_status") != "NO_PRED_INSUFFICIENT_DATA"
        )
        match_summary["pct"] = 100.0 * match_summary["matched"] / max(1, match_summary["total"])

    # KPI strip
    with st.container():
        a, b, c, d = st.columns(4)
        a.metric("Event start", storm_start_label)
        b.metric("Segments (filtered)", f"{len(view):,}")
        # Untracked for this event (in full event slice)
        event_slice = preds[preds["eventid"] == selected_event].copy()
        untracked = int((event_slice.get("prediction_status", "") == "NO_PRED_INSUFFICIENT_DATA").sum()) if "prediction_status" in event_slice.columns else 0
        c.metric("No prediction", f"{untracked:,}")
        if metrics and str(4) in metrics and "auc" in metrics[str(4)]:
            d.metric("Model AUC (≤4h)", f"{metrics[str(4)]['auc']:.3f}")
        else:
            d.metric("Model AUC (≤4h)", "—")
        st.caption(f"GeoJSON match rate: {match_summary['pct']:.1f}% (grey = untracked/no prediction).")

    # Build a selection menu based on highest probability
    sel_candidates = view.sort_values("_prob", ascending=False).head(400).copy()

    def pred_label(row) -> str:
        rn = str(row.get("roadname", ""))
        rt = format_route_label(row.get("snowrouteid"))
        pr = str(row.get("routepriority", ""))
        p = row.get("_prob")
        ptxt = f"{p:.0%}" if pd.notna(p) else "—"
        return f"{rn} • {rt} • P{pr} • {horizon}:{ptxt}"

    sel_candidates["_label"] = sel_candidates.apply(pred_label, axis=1)
    selection_options = sel_candidates["_label"].tolist()

    # Details drawer
    with main_right:
        st.subheader("Details")
        if not selection_options:
            st.info("No segments match your filters.")
            selected_label = None
        else:
            selected_label = st.selectbox("Select a segment", options=selection_options, index=0)

        if selected_label:
            row = sel_candidates[sel_candidates["_label"] == selected_label].iloc[0]
            st.metric(f"Model p(≤{horizon})", f"{float(row.get(prob_col, 0.0)):.0%}" if pd.notna(row.get(prob_col)) else "—")
            st.metric("Bucket ETA tier", bucket_eta_tier(row.get("eta_bucket_hours") or row.get("eta_hours_60")))
            st.caption(f"Route: {format_route_label(row.get('snowrouteid'))} • Priority: {row.get('routepriority','—')}")
            # Show all horizons compactly
            st.markdown("**All horizons (model):**")
            st.write({
                "1h": format_probability(row.get("p_ml_1h")),
                "2h": format_probability(row.get("p_ml_2h")),
                "4h": format_probability(row.get("p_ml_4h")),
                "8h": format_probability(row.get("p_ml_8h")),
            })
        with st.expander("What does this mean?", expanded=False):
            st.write(BUCKET_EXPLAINER_TEXT)

    # Map
    with main_left:
        st.subheader("Map")
        if not geo_for_map:
            st.warning("No GeoJSON available to render prediction map.")
        else:
            layers = [
                pdk.Layer(
                    "GeoJsonLayer",
                    data=geo_for_map,
                    stroked=True,
                    filled=False,
                    get_line_color="properties._line_color",
                    get_line_width=3,
                    line_width_min_pixels=2,
                    pickable=True,
                )
            ]

            # Highlight selection if possible (find feature by segment id)
            if selected_label and geo_for_map:
                sel_row = sel_candidates[sel_candidates["_label"] == selected_label].iloc[0]
                seg = str(sel_row.get("snowroutesegmentid"))
                feat_match = None
                for feat in geo_for_map.get("features", []):
                    props = feat.get("properties", {}) or {}
                    if str(props.get("snowroutesegmentid")) == seg:
                        feat_match = feat
                        break
                if feat_match:
                    sel_geo = {"type": "FeatureCollection", "features": [feat_match]}
                    layers.append(
                        pdk.Layer(
                            "GeoJsonLayer",
                            data=sel_geo,
                            stroked=True,
                            filled=False,
                            get_line_color=[0, 0, 0],
                            get_line_width=8,
                            line_width_min_pixels=5,
                            pickable=False,
                        )
                    )

            view_state = pdk.ViewState(latitude=43.0481, longitude=-76.1474, zoom=11.6, pitch=0)

            st.pydeck_chart(
                pdk.Deck(
                    layers=layers,
                    initial_view_state=view_state,
                    tooltip={
                        "html": (
                            "<b>{properties.roadname}</b><br/>"
                            "Route: {properties.route_label}<br/>"
                            f"p(≤{horizon}): {{properties.{prob_col}}}<br/>"
                            "Tier: {properties.bucket_eta_tier}<br/>"
                            "Status: {properties.prediction_status}"
                        )
                    },
                ),
                width="stretch",
            )

        with st.expander("Ranked list (filtered)", expanded=False):
            show_cols = [c for c in [
                "roadname", "snowrouteid", "routepriority",
                "p_ml_1h", "p_ml_2h", "p_ml_4h", "p_ml_8h",
                "eta_bucket_hours", "eta_hours_60",
                "bucket_priority_id", "current_bucket_priority",
                "prediction_status",
            ] if c in view.columns]
            st.dataframe(
                view.sort_values("_prob", ascending=False)[show_cols].head(800),
                width="stretch",
            )

        with st.expander("Model health", expanded=False):
            st.write(metrics if metrics else "No metrics loaded.")

