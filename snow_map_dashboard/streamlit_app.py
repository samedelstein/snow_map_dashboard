# streamlit_app.py
# Syracuse Snow Routes – Storm Progress (Work + Coverage) + Live Recency Map
#
# What this app does:
# - Lets user pick a "storm episode" from NWS alerts (grouped into operational storms)
# - Computes:
#   (A) Lane-miles plowed (counts repeat passes) in last 1/6/12/24h + since storm start
#   (B) Coverage (unique road-miles plowed at least once) + new coverage in last 1/6/12/24h
#   (C) Hourly bar chart of lane-miles plowed (ET) over last 24h
# - Keeps your live ArcGIS map colored by time-since-lastserviced buckets
#
# Notes:
# - All display + bucketing is in US/Eastern (ET). Raw timestamps remain UTC.
# - Coverage counts each segment once (first time it is plowed during the storm). Caps at 100%.
# - Lane-miles counts every pass. Can exceed total road-miles.

import time
from datetime import datetime, timezone
from pathlib import Path
import plotly.express as px
import textwrap

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
from zoneinfo import ZoneInfo

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Syracuse Snow Routes – Storm Progress", layout="wide")

LOCAL_TZ = ZoneInfo("America/New_York")  # ET (handles EST/EDT)

BASE_URL = (
    "https://services6.arcgis.com/"
    "bdPqSfflsdgFRVVM/arcgis/rest/services/"
    "Winter_Operations_Snow_Routes/FeatureServer/0/query"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = DATA_DIR / "artifacts_snow"
SNAPSHOT_PATH = DATA_DIR / "snapshot_snow_routes" / "snapshots.csv"

DEFAULT_PAGE_SIZE = 2000

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
    "> 24 hours": [128, 0, 38],  # dark red
    "12–24 hours": [227, 26, 28],  # red
    "6–12 hours": [253, 141, 60],  # orange
    "1–6 hours": [255, 237, 111],  # yellow
    "< 1 hour": [44, 162, 95],  # green-ish
}

st.markdown(
    """
<style>
.kpi-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.75rem; }
.kpi-card {
    background: #ffffff;
    padding: 0.9rem 1.0rem;
    border-radius: 14px;
    border: 1px solid #eaeaea;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
}
.kpi-title { font-size: 0.78rem; font-weight: 700; color: #555; letter-spacing: 0.03em; text-transform: uppercase; }
.kpi-value { font-size: 1.9rem; font-weight: 800; color: #111; margin-top: 0.25rem; line-height: 1.0; }
.kpi-sub  { font-size: 0.82rem; color: #666; margin-top: 0.35rem; }
.kpi-chip { display: inline-block; font-size: 0.74rem; padding: 0.15rem 0.45rem; border-radius: 999px; background: #f3f4f6; color: #374151; margin-left: 0.4rem; }
.section-note { color: #666; font-size: 0.9rem; margin-top: -0.25rem; }
</style>
""",
    unsafe_allow_html=True,
)

def kpi_card(title: str, value: str, sub: str = "", chip: str | None = None) -> str:
    chip_html = f'<span class="kpi-chip">{chip}</span>' if chip else ""
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    html = f"""
    <div class="kpi-card">
      <div class="kpi-title">{title}{chip_html}</div>
      <div class="kpi-value">{value}</div>
      {sub_html}
    </div>
    """
    return textwrap.dedent(html).strip()



# NWS alerts URL (public)
NWS_ALERTS_URL = (
    "https://raw.githubusercontent.com/samedelstein/snow_map_dashboard/"
    "refs/heads/main/data/artifacts_snow/nws_alerts_log.csv"
)

EVENT_FAMILY = {
    "Winter Storm Watch": "Winter Storm",
    "Winter Storm Warning": "Winter Storm",
    "Winter Weather Advisory": "Winter Weather",
    "Wind Advisory": "Wind",
    "Special Weather Statement": "Special",
}
SEVERITY_ORDER = ["Minor", "Moderate", "Severe", "Extreme"]


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
            "outSR": 4326,  # ask server for WGS84
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

    if "snowroutesegmentid" in df.columns:
        df["snowroutesegmentid"] = df["snowroutesegmentid"].astype(str)

    fetched_at_utc = datetime.now(timezone.utc).isoformat()
    return df, fetched_at_utc, len(df)


# -----------------------------
# NWS storms (cached)
# -----------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def load_nws_alerts_url(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    df = df.drop_duplicates(subset=["event", "start_ts", "end_ts", "severity"])
    df["start_ts"] = pd.to_datetime(df.get("start_ts"), utc=True, errors="coerce")
    df["end_ts"] = pd.to_datetime(df.get("end_ts"), utc=True, errors="coerce")
    df = df.dropna(subset=["start_ts", "end_ts"])
    df["event_family"] = df["event"].map(EVENT_FAMILY).fillna(df["event"])
    return df


def build_op_storms(alerts: pd.DataFrame, gap_hours: int = 6) -> pd.DataFrame:
    df = alerts.copy().sort_values(["event_family", "start_ts"]).reset_index(drop=True)

    df["prev_end"] = df.groupby("event_family")["end_ts"].shift()
    df["gap_hours"] = (df["start_ts"] - df["prev_end"]).dt.total_seconds() / 3600.0
    df["new_storm"] = df["prev_end"].isna() | (df["gap_hours"] > gap_hours)
    df["storm_id"] = df.groupby("event_family")["new_storm"].cumsum()

    storms = (
        df.groupby(["event_family", "storm_id"], as_index=False)
        .agg(
            storm_start=("start_ts", "min"),
            storm_end=("end_ts", "max"),
            alerts=("alert_id", "count") if "alert_id" in df.columns else ("event", "count"),
            max_severity=("severity", "max"),
        )
        .sort_values(["storm_start"])
        .reset_index(drop=True)
    )

    storms_ops = storms.sort_values("storm_start").reset_index(drop=True).copy()
    storms_ops["storm_end_running"] = storms_ops["storm_end"].cummax()
    storms_ops["prev_storm_end_running"] = storms_ops["storm_end_running"].shift()
    storms_ops["gap_hours"] = (
        (storms_ops["storm_start"] - storms_ops["prev_storm_end_running"]).dt.total_seconds() / 3600.0
    )
    storms_ops["new_op_storm"] = storms_ops["prev_storm_end_running"].isna() | (storms_ops["gap_hours"] > gap_hours)
    storms_ops["op_storm_id"] = storms_ops["new_op_storm"].cumsum()

    storms_ops["_severity_cat"] = pd.Categorical(
        storms_ops["max_severity"], categories=SEVERITY_ORDER, ordered=True
    )

    op_storms = (
        storms_ops.groupby("op_storm_id", as_index=False)
        .agg(
            op_storm_start=("storm_start", "min"),
            op_storm_end=("storm_end", "max"),
            event_families=("event_family", lambda x: ", ".join(sorted(set(x)))),
            alert_storms=("event_family", "count"),
            peak_severity=("_severity_cat", "max"),
        )
    )

    # Ops buffers (optional but helpful)
    op_storms["ops_start"] = op_storms["op_storm_start"] - pd.Timedelta(hours=24)
    op_storms["ops_end"] = op_storms["op_storm_end"] + pd.Timedelta(hours=48)

    op_storms["op_storm_key"] = "op_storm_" + op_storms["op_storm_id"].astype(int).astype(str).str.zfill(2)

    # Windows-safe label formatting (no %- flags)
    start_et = op_storms["op_storm_start"].dt.tz_convert(LOCAL_TZ.key)
    end_et = op_storms["op_storm_end"].dt.tz_convert(LOCAL_TZ.key)
    op_storms["op_storm_name"] = (
        "Storm: "
        + start_et.dt.strftime("%b %d, %I:%M %p ET")
        + " → "
        + end_et.dt.strftime("%b %d, %I:%M %p ET")
        + "  ·  "
        + op_storms["event_families"].astype(str)
        + "  ·  "
        + op_storms["peak_severity"].astype(str)
    )

    return op_storms.sort_values("op_storm_start").reset_index(drop=True)


# -----------------------------
# Snapshot-based storm metrics (Work + Coverage)
# -----------------------------
@st.cache_data(ttl=300, show_spinner=False)
def compute_storm_work_and_coverage(
    snapshots_path_str: str,
    storm_start_utc: pd.Timestamp,
    eval_now_utc: pd.Timestamp | None,
):
    """
    snapshots_path_str: must be str for caching stability
    storm_start_utc: UTC timestamp for baseline attribution
    eval_now_utc: evaluate metrics as-of this time (UTC). If None, uses latest snapshot.

    Returns dicts + hourly series in ET:
      metrics_work, metrics_cov, hourly_work (hour_et,lane_miles), hourly_cov_new (hour_et,new_coverage_miles)
    """
    s = pd.read_csv(snapshots_path_str)

    # Required columns
    need = ["snapshot_ts", "snowroutesegmentid", "passes", "segmentlength"]
    s = s[[c for c in need if c in s.columns]].copy()

    s["snapshot_ts"] = pd.to_datetime(s["snapshot_ts"], utc=True, errors="coerce")
    s = s.dropna(subset=["snapshot_ts", "snowroutesegmentid", "passes", "segmentlength"]).copy()

    s["snowroutesegmentid"] = s["snowroutesegmentid"].astype(str)

    # De-dupe segment+timestamp
    s = s.drop_duplicates(subset=["snowroutesegmentid", "snapshot_ts"], keep="last")

    if eval_now_utc is None:
        eval_now_utc = s["snapshot_ts"].max()
    eval_now_utc = pd.to_datetime(eval_now_utc, utc=True)

    # Only keep rows up to eval_now_utc
    s = s[s["snapshot_ts"] <= eval_now_utc].copy()

    # Baseline passes before storm start (per segment)
    baseline = (
        s[s["snapshot_ts"] < storm_start_utc]
        .sort_values(["snowroutesegmentid", "snapshot_ts"])
        .groupby("snowroutesegmentid", as_index=False)
        .last()[["snowroutesegmentid", "passes"]]
        .rename(columns={"passes": "baseline_passes"})
    )

    df = (
        s[s["snapshot_ts"] >= storm_start_utc]
        .sort_values(["snowroutesegmentid", "snapshot_ts"])
        .merge(baseline, on="snowroutesegmentid", how="left")
    )
    df["baseline_passes"] = df["baseline_passes"].fillna(0)

    # Previous pass: previous storm snapshot else baseline
    df["prev_passes"] = df.groupby("snowroutesegmentid")["passes"].shift(1)
    df["prev_passes"] = df["prev_passes"].fillna(df["baseline_passes"])

    df["delta_passes"] = (df["passes"] - df["prev_passes"]).clip(lower=0)
    df["delta_miles_work"] = df["delta_passes"] * df["segmentlength"]

    # ET timestamps
    df["snapshot_ts_et"] = df["snapshot_ts"].dt.tz_convert(LOCAL_TZ.key)
    df["hour_et"] = df["snapshot_ts_et"].dt.floor("h")

    storm_start_et = storm_start_utc.tz_convert(LOCAL_TZ.key)
    now_et = eval_now_utc.tz_convert(LOCAL_TZ.key)

    # Total road-miles in system (unique segments)
    seg_miles_all = s.groupby("snowroutesegmentid")["segmentlength"].max()
    total_road_miles = float(seg_miles_all.sum())


    # Work (lane-miles) windows
    def work_in_last(hours: int) -> float:
        cutoff = now_et - pd.Timedelta(hours=hours)
        return float(
            df.loc[(df["snapshot_ts_et"] > cutoff) & (df["snapshot_ts_et"] <= now_et), "delta_miles_work"].sum()
        )

    metrics_work = {
        "past_1h": work_in_last(1),
        "past_6h": work_in_last(6),
        "past_12h": work_in_last(12),
        "past_24h": work_in_last(24),
        "since_storm_start": float(df["delta_miles_work"].sum()),
        "total_road_miles": total_road_miles,
        "storm_start_et": storm_start_et,
        "now_et": now_et,
    }

    # Hourly work series (last 24h)
    last24 = df[(df["snapshot_ts_et"] > now_et - pd.Timedelta(hours=24)) & (df["snapshot_ts_et"] <= now_et)].copy()
    hourly_work = (
        last24.groupby("hour_et", as_index=False)["delta_miles_work"]
        .sum()
        .rename(columns={"delta_miles_work": "lane_miles"})
        .sort_values("hour_et")
    )



    # Coverage: first touch (first time any delta_passes>0 during storm)
    touches = df[df["delta_passes"] > 0].copy()
    if touches.empty:
        covered_miles_total = 0.0
        covered_pct_total = 0.0
        first_touch = pd.DataFrame(columns=["snowroutesegmentid", "first_touch_et", "segment_miles"])
    else:
        first_touch = (
            touches.sort_values(["snowroutesegmentid", "snapshot_ts_et"])
            .groupby("snowroutesegmentid", as_index=False)
            .first()[["snowroutesegmentid", "snapshot_ts_et"]]
            .rename(columns={"snapshot_ts_et": "first_touch_et"})
        )
        first_touch = first_touch.merge(
            seg_miles_all.rename("segment_miles").reset_index(),
            on="snowroutesegmentid",
            how="left",
        )
        covered_miles_total = float(first_touch.loc[first_touch["first_touch_et"] <= now_et, "segment_miles"].sum())
        covered_pct_total = (covered_miles_total / total_road_miles * 100.0) if total_road_miles > 0 else 0.0

    def new_coverage_in_last(hours: int) -> float:
        cutoff = now_et - pd.Timedelta(hours=hours)
        if first_touch.empty:
            return 0.0
        return float(
            first_touch.loc[(first_touch["first_touch_et"] > cutoff) & (first_touch["first_touch_et"] <= now_et), "segment_miles"].sum()
        )

    metrics_cov = {
        "new_cov_1h_miles": new_coverage_in_last(1),
        "new_cov_6h_miles": new_coverage_in_last(6),
        "new_cov_12h_miles": new_coverage_in_last(12),
        "new_cov_24h_miles": new_coverage_in_last(24),
        "cov_total_miles": covered_miles_total,
        "cov_total_pct": covered_pct_total,
        "total_road_miles": total_road_miles,
    }

    # Hourly new coverage series (last 24h)
    if first_touch.empty:
        hourly_cov_new = pd.DataFrame(columns=["hour_et", "new_coverage_miles"])
    else:
        ft_last24 = first_touch[
            (first_touch["first_touch_et"] > now_et - pd.Timedelta(hours=24))
            & (first_touch["first_touch_et"] <= now_et)
        ].copy()
        ft_last24["hour_et"] = ft_last24["first_touch_et"].dt.floor("h")
        hourly_cov_new = (
            ft_last24.groupby("hour_et", as_index=False)["segment_miles"]
            .sum()
            .rename(columns={"segment_miles": "new_coverage_miles"})
            .sort_values("hour_et")
        )

    return metrics_work, metrics_cov, hourly_work, hourly_cov_new


# -----------------------------
# Plot: NYT-ish hourly bar chart (ET)
# -----------------------------
def plot_hourly_lane_miles(hourly_work: pd.DataFrame, storm_start_et: pd.Timestamp, now_et: pd.Timestamp):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12.5, 4.3), layout="constrained")

    if hourly_work.empty:
        ax.text(0.5, 0.5, "No hourly data available.", transform=ax.transAxes, ha="center", va="center")
        return fig

    # Ensure datetime + tz-aware
    hw = hourly_work.copy()
    hw["hour_et"] = pd.to_datetime(hw["hour_et"])
    if hw["hour_et"].dt.tz is None:
        # If somehow tz-naive, assume it's ET
        hw["hour_et"] = hw["hour_et"].dt.tz_localize("America/New_York")

    hw = hw.sort_values("hour_et")

    # Create continuous hourly index WITHOUT passing tz= (prevents AssertionError)
    start = hw["hour_et"].min().floor("h")
    end = hw["hour_et"].max().floor("h")
    all_hours = pd.date_range(start=start, end=end, freq="h")  # tz inferred from start/end

    hw = (
        hw.set_index("hour_et")
          .reindex(all_hours, fill_value=0)
          .rename_axis("hour_et")
          .reset_index()
    )

    ax.bar(hw["hour_et"], hw["lane_miles"], width=0.03)

    ax.set_title("Lane-miles plowed per hour (last 24 hours)", loc="left", fontsize=14, pad=6)

    # Subtitle (figure-level, avoids overlap)
    fig.text(
        0.01, 0.96,
        f"Storm start: {storm_start_et:%b %d, %I:%M %p ET} • Latest snapshot: {now_et:%b %d, %I:%M %p ET}",
        ha="left", va="top", fontsize=10
    )

    ax.set_ylabel("Lane-miles", fontsize=11)
    ax.set_xlabel("")

    ax.yaxis.grid(True, linewidth=0.6, alpha=0.25)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0, labelsize=9)

    # Use the tz from the data itself (avoid tz mismatches)
    tzinfo = hw["hour_et"].dt.tz

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3, tz=tzinfo))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%I %p\n%b %d", tz=tzinfo))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    # Peak annotation
    peak_idx = hw["lane_miles"].idxmax()
    peak_row = hw.loc[peak_idx]
    ax.annotate(
        f"Peak: {peak_row['lane_miles']:.0f}",
        xy=(peak_row["hour_et"], peak_row["lane_miles"]),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center", va="bottom", fontsize=9
    )

    fig.text(
        0.01, 0.02,
        "Lane-miles computed from pass-count deltas × segment length; includes repeat passes (maintenance).",
        ha="left", va="bottom", fontsize=9, color="0.35"
    )

    return fig



# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Storm selection")

    alerts = load_nws_alerts_url(NWS_ALERTS_URL)
    if alerts.empty:
        st.warning("No NWS alerts found; using manual storm start/end.")
        storm_start_utc = pd.to_datetime(
            st.text_input("Storm start (UTC)", "2025-12-28 14:01:00+00:00"),
            utc=True, errors="coerce"
        )
        storm_end_utc = pd.to_datetime(
            st.text_input("Storm end (UTC)", "2025-12-29 23:59:00+00:00"),
            utc=True, errors="coerce"
        )
        if pd.isna(storm_start_utc) or pd.isna(storm_end_utc) or storm_end_utc <= storm_start_utc:
            st.error("Invalid storm start/end.")
            st.stop()
        storm_label = "Manual storm"
    else:
        gap = st.select_slider("Gap threshold (hours)", options=[3, 6, 9, 12], value=6)
        op_storms = build_op_storms(alerts, gap_hours=int(gap)).copy()

        # Make labels stable + clickable
        # Use op_storm_key as the value, show label as display
        options = op_storms[["op_storm_key", "op_storm_name", "op_storm_start", "op_storm_end"]].copy()
        options = options.sort_values("op_storm_start").reset_index(drop=True)

        default_idx = len(options) - 1
        chosen_key = st.selectbox(
            "Choose storm",
            options=options["op_storm_key"].tolist(),
            index=default_idx,
            format_func=lambda k: options.loc[options["op_storm_key"] == k, "op_storm_name"].iloc[0],
        )

        sel = options.loc[options["op_storm_key"] == chosen_key].iloc[0]
        storm_start_utc = sel["op_storm_start"]
        storm_end_utc = sel["op_storm_end"]
        storm_label = options.loc[options["op_storm_key"] == chosen_key, "op_storm_name"].iloc[0]

    st.divider()
    st.subheader("Replay mode")

    replay_mode = st.radio(
        "Compute metrics as-of",
        ["Storm end (default)", "Storm start + N hours"],
        index=0
    )

    if replay_mode == "Storm start + N hours":
        hours = st.slider("Hours since storm start", 0, 72, 12, 1)
        eval_now_utc = storm_start_utc + pd.Timedelta(hours=hours)
        # clamp to storm end so you can't go beyond it
        eval_now_utc = min(eval_now_utc, storm_end_utc)
    else:
        eval_now_utc = storm_end_utc

    # Clear, explicit summary (this is what removes confusion)
    st.caption(
        f"**Selected:** {storm_label}\n\n"
        f"Storm window (UTC): {storm_start_utc:%Y-%m-%d %H:%M} → {storm_end_utc:%Y-%m-%d %H:%M}\n\n"
        f"Replay as-of (UTC): {eval_now_utc:%Y-%m-%d %H:%M}"
    )



# -----------------------------
# Main layout
# -----------------------------
st.title("Syracuse Snow Routes — Storm Progress")
st.caption(
    "Two complementary measures:\n"
    "- **Lane-miles plowed** = work done (includes repeat passes; can exceed total road-miles)\n"
    "- **Coverage** = unique road-miles plowed at least once (caps at 100%)"
)

# -----------------------------
# Metrics from snapshots.csv (storm selection drives it)
# -----------------------------
if not SNAPSHOT_PATH.exists():
    st.error(f"Missing snapshots file: {SNAPSHOT_PATH}")
    st.stop()

# Evaluate “as-of” time: clamp to selected ops window end (makes storm replay stable)
# Use latest snapshot time, capped by ops_end_utc
snap_ts = pd.read_csv(SNAPSHOT_PATH, usecols=["snapshot_ts"])
snap_ts["snapshot_ts"] = pd.to_datetime(snap_ts["snapshot_ts"], utc=True, errors="coerce")
latest_snapshot_utc = snap_ts["snapshot_ts"].dropna().max()

eval_now_utc = min(latest_snapshot_utc, storm_end_utc) if pd.notna(latest_snapshot_utc) else storm_end_utc

metrics_work, metrics_cov, hourly_work, hourly_cov_new = compute_storm_work_and_coverage(
    str(SNAPSHOT_PATH),
    storm_start_utc=storm_start_utc,
    eval_now_utc=eval_now_utc,
)

# -----------------------------
# KPI rows
# -----------------------------
st.subheader("Storm progress")

# WORK (lane-miles)
st.markdown('<div class="section-note"><b>Lane-miles plowed</b> = work done (repeat passes count; can exceed system miles)</div>', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="kpi-grid">
      {kpi_card("Last 1h",  f"{metrics_work['past_1h']:.0f}",  "lane-miles")}
      {kpi_card("Last 6h",  f"{metrics_work['past_6h']:.0f}",  "lane-miles")}
      {kpi_card("Last 12h", f"{metrics_work['past_12h']:.0f}", "lane-miles")}
      {kpi_card("Last 24h", f"{metrics_work['past_24h']:.0f}", "lane-miles")}
      {kpi_card("Storm total", f"{metrics_work['since_storm_start']:.0f}", "lane-miles")}
    </div>
    """,
    unsafe_allow_html=True,
)

# COVERAGE (unique)
covered_pct = (metrics_cov["cov_total_miles"] / metrics_cov["total_road_miles"] * 100.0) if metrics_cov["total_road_miles"] > 0 else 0.0
st.markdown('<div class="section-note"><b>Coverage</b> = unique road-miles plowed at least once (caps at 100%)</div>', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="kpi-grid">
      {kpi_card("New coverage 1h",  f"{metrics_cov['new_cov_1h_miles']:.0f}",  "miles")}
      {kpi_card("New coverage 6h",  f"{metrics_cov['new_cov_6h_miles']:.0f}",  "miles")}
      {kpi_card("New coverage 12h", f"{metrics_cov['new_cov_12h_miles']:.0f}", "miles")}
      {kpi_card("New coverage 24h", f"{metrics_cov['new_cov_24h_miles']:.0f}", "miles")}
      {kpi_card("Total coverage", f"{covered_pct:.1f}%", f"{metrics_cov['cov_total_miles']:.0f} / {metrics_cov['total_road_miles']:.0f} mi")}
    </div>
    """,
    unsafe_allow_html=True,
)


total_road_miles = metrics_work["total_road_miles"]
covered_miles = metrics_cov["cov_total_miles"]
st.caption(
    f"**Coverage** counts each road-mile once (first plow during this storm). "
    f"Covered so far: {covered_miles:.0f} / {total_road_miles:.0f} road-miles."
)

# Optional: “intensity” helper
if total_road_miles > 0:
    avg_passes_per_mile = (metrics_work["since_storm_start"] / metrics_work["total_road_miles"]) if metrics_work["total_road_miles"] > 0 else 0.0
    st.caption(f"Average passes per road-mile so far: **{avg_passes_per_mile:.1f}×** (lane-miles ÷ road-miles).")

# -----------------------------
# Hourly bar chart
# -----------------------------
st.subheader("Work tempo (ET)")

if hourly_work.empty:
    st.info("No hourly data available for this storm window.")
else:
    fig = px.bar(
        hourly_work.sort_values("hour_et"),
        x="hour_et",
        y="lane_miles",
        labels={"hour_et": "Hour (ET)", "lane_miles": "Lane-miles"},
        title="Lane-miles plowed per hour (last 24 hours)",
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(tickformat="%I %p<br>%b %d"),
        hovermode="x unified",
    )
    fig.update_traces(hovertemplate="%{x|%b %d, %I:%M %p ET}<br>Lane-miles: %{y:.0f}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Show hourly tables (debug)"):
    st.write("Hourly lane-miles:")
    st.dataframe(hourly_work, use_container_width=True)

    st.write("Hourly new coverage miles:")
    st.dataframe(hourly_cov_new, use_container_width=True)

# -----------------------------
# Live API recency map (your existing style)
# -----------------------------

# Use storm_start_utc as the cutoff for “Never plowed (before cutoff)”
df_live, fetched_at_utc, n_rows = fetch_live_data(storm_start_utc.isoformat(), page_size=DEFAULT_PAGE_SIZE)


fetched_at = pd.to_datetime(fetched_at_utc, utc=True, errors="coerce")
now_utc = pd.Timestamp.now(tz="UTC")

if df_live.empty:
    st.warning("No data returned from API (or all timestamps invalid).")
    st.stop()

if pd.notna(fetched_at):
    age_min = (now_utc - fetched_at).total_seconds() / 60.0
    dot = freshness_dot(age_min)
    label = freshness_label(age_min)
    st.markdown(f"**{dot} Live API freshness:** {label} — fetched {age_min:.1f} min ago (rows: {n_rows})")

# Bucket miles breakdown (recency)

# Ensure miles is numeric and buckets are clean
df_live["miles"] = pd.to_numeric(df_live.get("miles"), errors="coerce").fillna(0.0)
df_live["bucket"] = df_live.get("bucket").fillna("Unknown").astype(str)

bucket_miles = (
    df_live.groupby("bucket", dropna=False)["miles"]
          .sum()
          .reindex(BUCKET_ORDER, fill_value=0.0)
)

bucket_miles_df = bucket_miles.reset_index()
bucket_miles_df.columns = ["bucket", "miles"]

total_miles_live = float(bucket_miles_df["miles"].sum())

# Always create a column (never a scalar)
if total_miles_live > 0:
    bucket_miles_df["pct"] = (bucket_miles_df["miles"] / total_miles_live * 100.0).round(1)
else:
    bucket_miles_df["pct"] = 0.0

# Build and render cards
# Build and render cards (same mechanism as top KPIs)
st.subheader("Live recency buckets")

# compute bucket_miles_df, total_miles_live, etc. (keep your existing calculation)

#cards_html = []
cards_html = [kpi_card("Total miles", f"{total_miles_live:.0f}", "live system miles")]
for _, r in bucket_miles_df.iterrows():
    b = str(r["bucket"])
    cards_html.append(
        kpi_card(
            BUCKET_DISPLAY_LABELS.get(b, b),
            f"{float(r['miles']):.0f}",
            f"{float(r['pct']):.1f}% of total",
        )
    )

cards_grid = (
    "<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:0.75rem;'>"
    + "".join(cards_html)
    + "</div>"
)

# IMPORTANT: only render THIS variable, and nowhere else
st.markdown(cards_grid, unsafe_allow_html=True)



# Map filters

st.markdown("### Live map — time since last plow (recency buckets)")

options = [BUCKET_DISPLAY_LABELS[b] for b in BUCKET_ORDER]
selected_display = st.multiselect("Show buckets", options=options, default=options)

selected_buckets = [b for b in BUCKET_ORDER if BUCKET_DISPLAY_LABELS[b] in selected_display]
if not selected_buckets:
    selected_buckets = BUCKET_ORDER[:]

map_df = df_live[df_live["bucket"].isin(selected_buckets)].copy()


map_df["color"] = map_df["bucket"].map(lambda b: BUCKET_COLORS.get(b, [0, 0, 0]))

# Labels in ET
map_df["lastserviced_label"] = map_df["lastserviced_dt"].dt.tz_convert(LOCAL_TZ).dt.strftime("%b %d, %I:%M %p ET")

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

view_state = pdk.ViewState(longitude=mean_lon, latitude=mean_lat, zoom=12, pitch=0)

st.pydeck_chart(
    pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={
            "text": (
                "Road name: {roadname}\n"
                "Priority level: {routepriority}\n"
                "Current status: {servicestatus}\n"
                "Time-since-plow bucket: {bucket}\n"
                "Segment length (miles): {miles:.3f}\n"
                "Last plowed at: {lastserviced_label}\n"
            )
        },
    ),
    use_container_width=True,
)


with st.expander("Raw live rows (sample)"):
    st.dataframe(
        df_live[
            ["OBJECTID", "roadname", "bucket", "miles", "lastserviced_dt", "servicestatus", "passes"]
        ].head(500),
        width="stretch",
    )
