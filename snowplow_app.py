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


# ============================================================
# Config
# ============================================================
st.set_page_config(page_title="Syracuse Snow Routes – Live + Predictions", layout="wide")

BASE_URL = (
    "https://services6.arcgis.com/"
    "bdPqSfflsdgFRVVM/arcgis/rest/services/"
    "Winter_Operations_Snow_Routes/FeatureServer/0/query"
)

REPO_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = REPO_ROOT / "artifacts_snow"
PRED_PATH = ARTIFACT_DIR / "predictions_latest_prob.csv"
METRICS_PATH = ARTIFACT_DIR / "model_metrics_prob.json"
GEOJSON_PATH = REPO_ROOT / "snapshot_snow_routes/latest_routes.geojson"  # updated by snapshot_snow_routes/snapshot_snow_routes.py
SNAPSHOT_PATH = REPO_ROOT / "snapshot_snow_routes" / "snapshots.csv"

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

    for c in ["p_1h", "p_2h", "p_4h", "p_8h"]:
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

    fields = [
        "prediction_status",
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
                props["_line_color"] = prob_to_color(row.get(color_by))

        feat["properties"] = props

    return geojson_obj


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
    color_by = st.selectbox("Color predictions by", ["p_1h", "p_2h", "p_4h", "p_8h"], index=2)
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
    map_df = df_live[df_live["bucket"].isin(selected_buckets)].copy()
    map_df["color"] = map_df["bucket"].map(lambda b: BUCKET_COLORS.get(b, [0, 0, 0]))

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
            tooltip={"text": "Road: {roadname}\nBucket: {bucket}\nMiles: {miles:.3f}\nLast serviced: {lastserviced_dt}"},
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
        st.warning("No predictions file found. Expected at: artifacts_snow/predictions_latest_prob.csv")
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

    cA, cB, cC, cD = st.columns(4)
    cA.metric("Segments (filtered)", f"{len(view):,}")
    cB.metric("Predicted (OK)", f"{ok:,}")
    cC.metric("No prediction", f"{untracked:,}")
    if metrics and str(4) in metrics and "auc" in metrics[str(4)]:
        cD.metric("Model AUC (≤4h)", f"{metrics[str(4)]['auc']:.3f}")
    else:
        cD.metric("Model AUC (≤4h)", "—")

    st.caption("Color scale uses probabilities. Grey segments are untracked/no prediction.")

    # Map
    if not geojson_obj:
        st.warning(f"GeoJSON not found at {GEOJSON_PATH.name}. Add it to render the prediction map.")
    else:
        # IMPORTANT: deep copy so we don't mutate cached geojson across reruns/events
        geo_copy = copy.deepcopy(geojson_obj)
        geo_for_map = attach_predictions_to_geojson(geo_copy, preds, join_event, color_by=color_by)

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
            Status: {prediction_status}<br/>
            p(≤1h): {p_1h}<br/>
            p(≤2h): {p_2h}<br/>
            p(≤4h): {p_4h}<br/>
            p(≤8h): {p_8h}<br/>
            ETA@60%: {eta_hours_60}h
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

    show_cols = [c for c in [
        "roadname",
        "routepriority",
        "snowrouteid",
        "p_1h",
        "p_2h",
        "p_4h",
        "p_8h",
        "eta_hours_60",
        "prediction_status",
        "snowroutesegmentid",
    ] if c in table.columns]

    st.dataframe(table[show_cols].head(500), use_container_width=True)

    with st.expander("Show no-prediction segments (untracked)"):
        un = view[view["prediction_status"] != "OK"].copy()
        st.dataframe(
            un[["roadname", "routepriority", "snowrouteid", "prediction_status", "snowroutesegmentid"]].head(500),
            use_container_width=True,
        )
