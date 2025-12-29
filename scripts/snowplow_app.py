# streamlit_app.py
import time
import json
import copy
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import pydeck as pdk
import sys

# ============================================================
# Config
# ============================================================
st.set_page_config(page_title="Syracuse Snow Routes – Live + Predictions", layout="wide")

BASE_URL = (
    "https://services6.arcgis.com/"
    "bdPqSfflsdgFRVVM/arcgis/rest/services/"
    "Winter_Operations_Snow_Routes/FeatureServer/0/query"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "snow_map_dashboard"))
from bucket_copy import BUCKET_EXPLAINER_TEXT
DATA_DIR = REPO_ROOT / "data"
ARTIFACT_DIR = DATA_DIR / "artifacts_snow"
PRED_PATH = ARTIFACT_DIR / "predictions_latest_prob.csv"
METRICS_PATH = ARTIFACT_DIR / "model_metrics_prob.json"
GEOJSON_PATH = (
    DATA_DIR / "snapshot_snow_routes/latest_routes.geojson"
)  # updated by scripts/snapshot_snow_routes.py
SNAPSHOT_PATH = DATA_DIR / "snapshot_snow_routes" / "snapshots.csv"

DEFAULT_CUTOFF_DATE_STR = "2025-12-01"  # UTC cutoff for "this storm"
DEFAULT_PAGE_SIZE = 2000

# Buckets for LIVE map (time since lastserviced)
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
    "Never plowed (before cutoff)": [200, 200, 200],
    "> 24 hours": [128, 0, 38],
    "12–24 hours": [227, 26, 28],
    "6–12 hours": [253, 141, 60],
    "1–6 hours": [255, 237, 111],
    "< 1 hour": [44, 162, 95],
}


# ============================================================
# Freshness + Auto-refresh
# ============================================================
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


# ============================================================
# ESRI helpers
# ============================================================
def _esri_paths_to_first_path(paths):
    if not isinstance(paths, list) or len(paths) == 0:
        return None
    first = paths[0]
    if not isinstance(first, list) or len(first) == 0:
        return None
    return [[pt[0], pt[1]] for pt in first if isinstance(pt, list) and len(pt) >= 2]


@st.cache_data(show_spinner=True, ttl=60)
def fetch_live_data(_cutoff_iso: str, page_size: int = DEFAULT_PAGE_SIZE):
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

    df["lastserviced_dt"] = pd.to_datetime(df.get("lastserviced"), unit="ms", utc=True, errors="coerce")
    df = df.dropna(subset=["lastserviced_dt"])

    # Miles
    if "Shape__Length" in df.columns and df["Shape__Length"].notna().any():
        df["miles"] = pd.to_numeric(df["Shape__Length"], errors="coerce") / 1609.344
    elif "segmentlength" in df.columns:
        df["miles"] = pd.to_numeric(df["segmentlength"], errors="coerce")
    else:
        df["miles"] = 0.0

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


# ============================================================
# Predictions loaders
# ============================================================
@st.cache_data(ttl=300)
def load_predictions():
    if not PRED_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(PRED_PATH)

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

    # enforce monotonicity across horizons
    if {"p_1h", "p_2h", "p_4h", "p_8h"}.issubset(df.columns):
        df["p_2h"] = df[["p_1h", "p_2h"]].max(axis=1)
        df["p_4h"] = df[["p_2h", "p_4h"]].max(axis=1)
        df["p_8h"] = df[["p_4h", "p_8h"]].max(axis=1)

    # ensure segment id is string for joins
    if "snowroutesegmentid" in df.columns:
        df["snowroutesegmentid"] = df["snowroutesegmentid"].astype(str)

    if "bucket_priority_id" not in df.columns and "routepriority" in df.columns:
        df["bucket_priority_id"] = df["routepriority"].apply(derive_bucket_priority_id)

    return df


@st.cache_data(ttl=300)
def load_model_metrics():
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data(ttl=3600)
def load_geojson():
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

        props["p_1h_pct"] = format_probability(props.get("p_1h"))
        props["p_2h_pct"] = format_probability(props.get("p_2h"))
        props["p_4h_pct"] = format_probability(props.get("p_4h"))
        props["p_8h_pct"] = format_probability(props.get("p_8h"))
        props["p_bucket_4h_pct"] = format_probability(props.get("p_bucket_4h"))
        props["p_ml_4h_pct"] = format_probability(props.get("p_ml_4h"))
        props["eta_60_label"] = format_eta_hours(props.get("eta_hours_60"))
        props["eta_bucket_label"] = format_eta_hours(props.get("eta_bucket_hours"))
        props["bucket_priority_label"] = format_bucket_priority(props.get("bucket_priority_id"))
        props["current_bucket_priority_label"] = format_bucket_priority(
            props.get("current_bucket_priority")
        )
        props["bucket_eta_tier"] = bucket_eta_tier(
            props.get("eta_bucket_hours") or props.get("eta_hours_60")
        )
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


# ============================================================
# Sidebar (defines variables BEFORE we use them)
# ============================================================
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
        cutoff_dt = pd.to_datetime(cutoff_date_str, utc=True)
        if pd.isna(cutoff_dt):
            raise ValueError("Invalid date")
    except Exception:
        st.error("Invalid cutoff date. Use YYYY-MM-DD.")
        cutoff_date_str = DEFAULT_CUTOFF_DATE_STR
        cutoff_dt = pd.to_datetime(cutoff_date_str, utc=True)

    st.divider()
    st.header("Predictions")
    color_by = st.selectbox(
        "Color predictions by",
        [
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
            "bucket_eta_tier",
        ],
        index=2,
    )
    min_prob = st.slider("Min probability (for table)", 0.0, 1.0, 0.0, 0.05)

# Auto-refresh MUST come after sidebar definitions
maybe_autorefresh(auto_refresh, int(refresh_every))


# ============================================================
# Tabs
# ============================================================
tab_live, tab_pred = st.tabs(["🗺️ Live status", "🔮 Predictions"])


# ============================================================
# Tab: LIVE
# ============================================================
with tab_live:
    df_live, fetched_at_utc, n_rows = fetch_live_data(cutoff_dt.isoformat(), page_size=DEFAULT_PAGE_SIZE)
    fetched_at = pd.to_datetime(fetched_at_utc, utc=True, errors="coerce")
    now_utc = pd.Timestamp.now(tz="UTC")

    max_last = df_live["lastserviced_dt"].max() if not df_live.empty else pd.NaT

    if not pd.isna(fetched_at):
        age_min = (now_utc - fetched_at).total_seconds() / 60.0
        dot = freshness_dot(age_min)
        label = freshness_label(age_min)
        fetched_str = fetched_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        max_last_str = max_last.strftime("%Y-%m-%d %H:%M:%S UTC") if not pd.isna(max_last) else "—"

        st.markdown(
            f"**{dot} Live API freshness:** {label} — fetched {age_min:.1f} min ago (rows: {n_rows})  \n"
            f"**Fetched at:** {fetched_str}  \n"
            f"**Newest lastserviced in response:** {max_last_str}"
        )

    st.title("Syracuse Snow Routes – Live Plowing Status")
    st.caption(
        "Buckets are based on `lastserviced` timestamps. "
        f"Anything before {cutoff_dt.date()} is treated as `Never plowed (before cutoff)` for this storm."
    )

    # CSS for KPI cards
    st.markdown(
        """
<style>
.kpi-card {background:#fff;padding:1.0rem 1.2rem;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,0.06);border:1px solid #e5e5e5;margin-bottom:0.8rem;}
.kpi-card-total {background:#fff;padding:1.1rem 1.4rem;border-radius:12px;box-shadow:0 2px 10px rgba(0,0,0,0.08);border:1px solid #e5e5e5;margin-bottom:0.8rem;}
.kpi-title {font-size:0.8rem;font-weight:600;color:#555;text-transform:uppercase;letter-spacing:.04em;}
.kpi-value {font-size:1.5rem;font-weight:700;color:#222;margin-top:.25rem;}
.kpi-gauge {margin-top:.5rem;height:6px;border-radius:999px;background:#f0f0f0;overflow:hidden;}
.kpi-gauge-fill {height:100%;border-radius:999px;background:linear-gradient(90deg,#2b8cff,#68d391);}
.kpi-gauge-label {margin-top:.25rem;font-size:.75rem;color:#666;}
</style>
""",
        unsafe_allow_html=True,
    )

    if df_live.empty:
        st.warning("No data returned from API (or all timestamps invalid).")
        st.stop()

    st.subheader("Miles plowed by recency bucket")
    bucket_miles = df_live.groupby("bucket")["miles"].sum().reindex(BUCKET_ORDER).fillna(0)
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
        cols[i % 3].markdown(
            f"""
<div class="kpi-card">
  <div class="kpi-title">{bucket}</div>
  <div class="kpi-value">{miles:.2f} miles</div>
  <div class="kpi-gauge"><div class="kpi-gauge-fill" style="width:{pct}%;"></div></div>
  <div class="kpi-gauge-label">{pct:.1f}% of total miles</div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.subheader("Map of snow routes by time since last plow")
    options = [BUCKET_DISPLAY_LABELS[b] for b in BUCKET_ORDER]
    selected_display = st.multiselect("Show buckets:", options=options, default=options, key="live_bucket_sel")
    selected_buckets = [b for b in BUCKET_ORDER if BUCKET_DISPLAY_LABELS[b] in selected_display]
    if not selected_buckets:
        selected_buckets = BUCKET_ORDER[:]
    map_df = df_live[df_live["bucket"].isin(selected_buckets)].copy()
    map_df["color"] = map_df["bucket"].map(lambda b: BUCKET_COLORS.get(b, [0, 0, 0]))

    if "lastserviced_dt" in map_df.columns:
        map_df["lastserviced_label"] = map_df["lastserviced_dt"].dt.strftime("%Y-%m-%d %H:%M UTC")
    else:
        map_df["lastserviced_label"] = "N/A"

    if not map_df.empty:
        all_points = [pt for path in map_df["path"] for pt in path]
        mean_lon = sum(p[0] for p in all_points) / len(all_points)
        mean_lat = sum(p[1] for p in all_points) / len(all_points)
    else:
        mean_lon, mean_lat = -76.15, 43.05

    live_layer = pdk.Layer(
        "PathLayer",
        data=map_df,
        get_path="path",
        get_color="color",
        width_scale=2,
        width_min_pixels=2,
        pickable=True,
    )

    view_state = pdk.ViewState(longitude=mean_lon, latitude=mean_lat, zoom=12, pitch=0)

    st.pydeck_chart(
        pdk.Deck(
            layers=[live_layer],
            initial_view_state=view_state,
            tooltip={
                "text": (
                    "Road: {roadname}\n"
                    "Bucket: {bucket}\n"
                    "Miles: {miles:.3f}\n"
                    "Last serviced: {lastserviced_label}"
                )
            },
        ),
        width="stretch",
    )


# ============================================================
# Tab: PREDICTIONS
# ============================================================
with tab_pred:
    st.title("Predicted next service (probabilities)")

    preds = load_predictions()
    metrics = load_model_metrics()
    geojson_obj = load_geojson()

    if preds.empty:
        st.warning(
            "No predictions file found. Expected at: data/artifacts_snow/predictions_latest_prob.csv"
        )
        st.stop()

    # Human-friendly event selector
    event_ranges = load_snapshot_event_ranges()

    def short(eid: str) -> str:
        return str(eid)[:8]

    pred_event_ids = set(preds["eventid"].dropna().unique())
    if not event_ranges.empty:
        event_ranges = event_ranges[event_ranges["eventid"].isin(pred_event_ids)].copy()

    if not event_ranges.empty:
        event_ranges["label"] = event_ranges.apply(
            lambda r: (
                f"{pd.to_datetime(r['min']).strftime('%Y-%m-%d %H:%M')} → "
                f"{pd.to_datetime(r['max']).strftime('%Y-%m-%d %H:%M')}  "
                f"({int(r['count']):,} snapshots)  [{short(r['eventid'])}]"
            ),
            axis=1,
        )
        label_to_id = dict(zip(event_ranges["label"], event_ranges["eventid"]))
        labels = list(event_ranges["label"])
        selected_label = st.selectbox("Storm window (event)", options=labels, index=len(labels) - 1)
        selected_event = label_to_id[selected_label]
    else:
        event_ids = sorted(list(pred_event_ids))
        selected_event = st.selectbox("Event ID", options=event_ids, index=len(event_ids) - 1)

    # Map join mode toggle
    mode = st.radio(
        "Map join mode",
        ["AUTO (use GeoJSON eventid)", "Selected storm only"],
        horizontal=True,
    )
    join_event = "AUTO" if mode.startswith("AUTO") else selected_event

    AUTO_MATCH_THRESHOLD_PCT = 1.0
    NO_PRED_STATUS = "NO_PRED_INSUFFICIENT_DATA"
    join_mode_label = mode
    match_summary = {"matched": 0, "total": 0, "pct": 0.0}
    geo_for_map = None

    if geojson_obj:
        geo_copy = copy.deepcopy(geojson_obj)
        geo_for_map = attach_predictions_to_geojson(geo_copy, preds, join_event, color_by=color_by)
        features = geo_for_map.get("features", [])
        match_summary["total"] = len(features)
        match_summary["matched"] = sum(
            1
            for feat in features
            if feat.get("properties", {}).get("prediction_status") != NO_PRED_STATUS
        )
        match_summary["pct"] = (
            match_summary["matched"] / match_summary["total"] * 100.0
            if match_summary["total"] > 0
            else 0.0
        )

        if join_event == "AUTO" and match_summary["pct"] < AUTO_MATCH_THRESHOLD_PCT:
            st.warning(
                "AUTO join matched ~0% of segments. Switching to Selected storm only "
                "so predictions can be applied."
            )
            join_event = selected_event
            join_mode_label = "Selected storm only (auto fallback from AUTO)"
            geo_copy = copy.deepcopy(geojson_obj)
            geo_for_map = attach_predictions_to_geojson(geo_copy, preds, join_event, color_by=color_by)
            features = geo_for_map.get("features", [])
            match_summary["total"] = len(features)
            match_summary["matched"] = sum(
                1
                for feat in features
                if feat.get("properties", {}).get("prediction_status") != NO_PRED_STATUS
            )
            match_summary["pct"] = (
                match_summary["matched"] / match_summary["total"] * 100.0
                if match_summary["total"] > 0
                else 0.0
            )
        elif join_event == "AUTO":
            join_mode_label = "AUTO (use GeoJSON eventid)"
        else:
            join_mode_label = "Selected storm only"

    st.caption(f"Prediction join mode in use: {join_mode_label}")

    # Filters
    c1, c2, c3 = st.columns(3)
    with c1:
        pri_opts = sorted([p for p in preds["routepriority"].dropna().unique()])
        pri_sel = st.multiselect("Route priority", options=pri_opts, default=pri_opts)
    with c2:
        route_opts = sorted([r for r in preds["snowrouteid"].dropna().unique()])
        route_sel = st.multiselect("Route", options=route_opts, default=[])
    with c3:
        road_q = st.text_input("Road name contains", value="")

    view = preds[preds["eventid"] == selected_event].copy()
    view = view[view["routepriority"].isin(pri_sel)]
    if route_sel:
        view = view[view["snowrouteid"].isin(route_sel)]
    if road_q:
        view = view[view["roadname"].astype(str).str.contains(road_q, case=False, na=False)]

    ok = (view["prediction_status"] == "OK").sum()
    untracked = (view["prediction_status"] != "OK").sum()

    cA, cB, cC, cD, cE = st.columns(5)
    cA.metric("Segments (filtered)", f"{len(view):,}")
    cB.metric("Predicted (OK)", f"{ok:,}")
    cC.metric("No prediction", f"{untracked:,}")
    if metrics and str(4) in metrics and "auc" in metrics[str(4)]:
        cD.metric("Model AUC (≤4h)", f"{metrics[str(4)]['auc']:.3f}")
    else:
        cD.metric("Model AUC (≤4h)", "—")
    if match_summary["total"] > 0:
        cE.metric("Segments matched", f"{match_summary['pct']:.1f}%")
    else:
        cE.metric("Segments matched", "—")

    st.caption("Color scale uses probabilities. Grey segments are untracked/no prediction.")

    event_slice = preds[preds["eventid"] == selected_event].copy()
    if "bucket_priority_id" not in event_slice.columns and "routepriority" in event_slice.columns:
        event_slice["bucket_priority_id"] = event_slice["routepriority"].apply(derive_bucket_priority_id)
    if "bucket_priority_id" not in view.columns and "routepriority" in view.columns:
        view["bucket_priority_id"] = view["routepriority"].apply(derive_bucket_priority_id)

    current_bucket = None
    if "current_bucket_priority" in event_slice.columns:
        bucket_mode = event_slice["current_bucket_priority"].dropna().mode()
        current_bucket = bucket_mode.iloc[0] if not bucket_mode.empty else None
    if current_bucket is None and "bucket_priority_id" in event_slice.columns:
        likely_now = event_slice[event_slice.get("p_1h", 0).fillna(0) >= 0.6]
        source = likely_now if not likely_now.empty else event_slice
        bucket_mode = source["bucket_priority_id"].dropna().mode()
        current_bucket = bucket_mode.iloc[0] if not bucket_mode.empty else None

    typical_bucket = None
    if "bucket_priority_id" in view.columns:
        typical_mode = view["bucket_priority_id"].dropna().mode()
        typical_bucket = typical_mode.iloc[0] if not typical_mode.empty else None

    st.subheader("Bucket-based schedule estimate")
    st.caption(BUCKET_EXPLAINER_TEXT)
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Current storm bucket", format_bucket_priority(current_bucket))
    b2.metric("Typical bucket (filtered)", format_bucket_priority(typical_bucket))
    eta_bucket_median = None
    if "eta_bucket_hours" in view.columns and view["eta_bucket_hours"].notna().any():
        eta_bucket_median = view["eta_bucket_hours"].median()
    elif "eta_hours_60" in view.columns and view["eta_hours_60"].notna().any():
        eta_bucket_median = view["eta_hours_60"].median()
    b3.metric("Median bucket ETA", format_eta_hours(eta_bucket_median))

    bucket_prob_median = None
    if "p_bucket_4h" in view.columns and view["p_bucket_4h"].notna().any():
        bucket_prob_median = view["p_bucket_4h"].median()
    elif "p_4h" in view.columns and view["p_4h"].notna().any():
        bucket_prob_median = view["p_4h"].median()
    b4.metric("Bucket chance ≤4h", format_probability(bucket_prob_median))

    storm_start = None
    if "storm_operational_start" in event_slice.columns:
        storm_start = event_slice["storm_operational_start"].dropna().max()
    storm_start_label = format_storm_start(storm_start)
    if storm_start is not None and not pd.isna(storm_start):
        hours_into = (pd.Timestamp.now(tz="UTC") - pd.to_datetime(storm_start, utc=True)).total_seconds() / 3600
        st.caption(f"Storm started: {storm_start_label} ({hours_into:.1f} hours ago).")
    else:
        st.caption("Storm start time is unavailable for this event.")

    tier_value = bucket_eta_tier(eta_bucket_median)
    st.caption(f"Typical service timing narrative: **{tier_value}** for this filtered set.")

    # Map
    if not geojson_obj:
        st.warning(f"GeoJSON not found at {GEOJSON_PATH.name}. Add it to render the prediction map.")
    else:
        pred_layer = pdk.Layer(
            "GeoJsonLayer",
            geo_for_map,
            stroked=True,
            filled=False,
            get_line_width=5,
            line_width_min_pixels=2,
            pickable=True,
            auto_highlight=True,
            get_line_color="properties._line_color",
        )

        pred_view_state = pdk.ViewState(latitude=43.0481, longitude=-76.1474, zoom=12, pitch=0)
        tooltip = {
            "html": """
            <b>{roadname}</b><br/>
            Priority: {routepriority}<br/>
            Route: {snowrouteid}<br/>
            Typical bucket: {bucket_priority_label}<br/>
            Current storm bucket: {current_bucket_priority_label}<br/>
            Bucket ETA: {eta_bucket_label}<br/>
            Status: {prediction_status}<br/>
            Bucket ETA tier: {bucket_eta_tier}<br/>
            p(≤1h): {p_1h_pct}<br/>
            p(≤2h): {p_2h_pct}<br/>
            p(≤4h): {p_4h_pct}<br/>
            p(≤8h): {p_8h_pct}<br/>
            Bucket p(≤4h): {p_bucket_4h_pct}<br/>
            Model p(≤4h): {p_ml_4h_pct}<br/>
            ETA@60%: {eta_60_label}
            """,
            "style": {"backgroundColor": "white", "color": "black"},
        }

        st.pydeck_chart(pdk.Deck(layers=[pred_layer], initial_view_state=pred_view_state, tooltip=tooltip), width="stretch")

    # Table
    st.subheader("Ranked list (most likely within selected horizon)")
    show_untracked = st.toggle("Include NO_PRED_UNTRACKED in table", value=False)

    table = view.copy()
    if not show_untracked:
        table = table[table["prediction_status"] == "OK"]

    if color_by in table.columns:
        table = table[table[color_by].fillna(0) >= float(min_prob)]
        table = table.sort_values(color_by, ascending=False)

    show_cols = [
        c
        for c in [
            "roadname",
            "routepriority",
            "snowrouteid",
            "bucket_priority_id",
            "current_bucket_priority",
            "eta_bucket_hours",
            "bucket_eta_tier",
            "p_bucket_1h",
            "p_bucket_2h",
            "p_bucket_4h",
            "p_bucket_8h",
            "p_ml_1h",
            "p_ml_2h",
            "p_ml_4h",
            "p_ml_8h",
            "p_1h",
            "p_2h",
            "p_4h",
            "p_8h",
            "eta_hours_60",
            "prediction_status",
            "snowroutesegmentid",
        ]
        if c in table.columns
    ]

    st.dataframe(table[show_cols].head(500), use_container_width=True)

    with st.expander("Show no-prediction segments (untracked)"):
        un = view[view["prediction_status"] != "OK"].copy()
        st.dataframe(
            un[["roadname", "routepriority", "snowrouteid", "prediction_status", "snowroutesegmentid"]].head(500),
            use_container_width=True,
        )
