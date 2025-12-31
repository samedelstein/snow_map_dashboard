import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from snow_ranking import (
    EventRankingConfig,
    answer_for_segment,
    compute_event_start,
    load_snapshots,
    rank_segments_within_events,
    summarize_across_events,
    write_ranked_geojson,
)

# ------------------------------------------------------------
# Paths (update to your local repo paths)
# ------------------------------------------------------------
SNAPSHOTS_PATH = Path(r"C:\Users\samie\snow_map\data\snapshot_snow_routes\snapshots.csv")
GEOJSON_PATH   = Path(r"C:\Users\samie\snow_map\winter_operations_snow_routes_layer0.geojson")
ARTIFACT_DIR   = Path(r"C:\Users\samie\snow_map\data\artifacts_snow")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Customize per event if you want a precise storm start.
EVENT_START_OVERRIDES = {
    # Example:
    # "0a4138c6-cc5c-4cba-a7b0-3c06695345ef": "2025-12-26 18:18:15+00:00"
}


st.set_page_config(page_title="Snow Route Ranking", layout="wide")


@st.cache_data(show_spinner=False)
def load_all():
    df = load_snapshots(SNAPSHOTS_PATH)

    cfg = EventRankingConfig(
        default_event_start_strategy="min_snapshot_ts",
        priority_lambda=0.06,
        n_buckets=5,
    )
    events = compute_event_start(df, cfg, overrides=EVENT_START_OVERRIDES)
    ranked = rank_segments_within_events(df, events, cfg)
    stats = summarize_across_events(ranked)

    return df, events, ranked, stats, cfg


@st.cache_data(show_spinner=False)
def load_geojson():
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- Map visuals ----------
BUCKET_PALETTE = {
    "Bucket 1": ([27, 94, 32, 220], "Very early"),
    "Bucket 2": ([102, 187, 106, 220], "Early"),
    "Bucket 3": ([255, 202, 40, 220], "Middle"),
    "Bucket 4": ([255, 112, 67, 220], "Late"),
    "Bucket 5": ([183, 28, 28, 220], "Very late"),
}


def bucket_color(bucket_label: str):
    return BUCKET_PALETTE.get(str(bucket_label), ([140, 140, 140, 140], "Unknown"))[0]


def render_legend():
    # Small inline legend with colored swatches
    items = []
    for k, (rgba, label) in BUCKET_PALETTE.items():
        r, g, b, a = rgba
        items.append(
            f"""
            <div style="display:flex;align-items:center;margin:2px 0;">
              <div style="width:14px;height:14px;border-radius:3px;background:rgba({r},{g},{b},{a/255});margin-right:8px;"></div>
              <div style="font-size:13px;"><b>{label}</b> ({k})</div>
            </div>
            """
        )
    st.markdown(
        "<div style='padding:10px 12px;border:1px solid rgba(255,255,255,0.08);border-radius:10px;'>"
        "<div style='font-weight:700;margin-bottom:6px;'>Service order legend</div>"
        + "".join(items)
        + "</div>",
        unsafe_allow_html=True,
    )


def _coords_centroid(geom: dict):
    """
    Return (lon, lat) centroid for LineString/MultiLineString/Point/Polygon-ish.
    Very forgiving — averages all coordinate points it can find.
    """
    if not geom:
        return None

    gtype = geom.get("type")
    coords = geom.get("coordinates")

    pts = []

    def add_points(obj):
        if obj is None:
            return
        if isinstance(obj, (list, tuple)):
            # point
            if len(obj) == 2 and all(isinstance(x, (int, float)) for x in obj):
                pts.append(obj)
                return
            # recurse
            for it in obj:
                add_points(it)

    add_points(coords)

    if not pts:
        return None

    arr = np.array(pts, dtype=float)
    lon = float(np.nanmean(arr[:, 0]))
    lat = float(np.nanmean(arr[:, 1]))
    if np.isnan(lon) or np.isnan(lat):
        return None
    return lon, lat


def _feature_lookup_by_segment(gj: dict):
    """
    Build lookup: snowroutesegmentid -> (feature, centroid)
    """
    lookup = {}
    for feat in gj.get("features", []):
        props = feat.get("properties", {}) or {}
        seg_id = props.get("snowroutesegmentid")
        if seg_id is None:
            continue
        seg_id = str(seg_id)
        centroid = _coords_centroid(feat.get("geometry", {}))
        lookup[seg_id] = (feat, centroid)
    return lookup


def build_map_geojson(base_geojson: dict, ranked_ev: pd.DataFrame, use_priority_adjusted: bool):
    """
    Add ranking fields into a copy of the geojson so PyDeck can color by bucket.
    """
    gj = json.loads(json.dumps(base_geojson))  # deep copy

    if use_priority_adjusted:
        bucket_field = "bucket_priority_adjusted"
    else:
        bucket_field = "bucket"

    # Fields to inject into GeoJSON properties for tooltips
    inject_fields = [
        "rank",
        "rank_pct",
        "minutes_from_start",
        "snowrouteid",
        "routepriority",
        bucket_field,
    ]
    inject_fields = [c for c in inject_fields if c in ranked_ev.columns]

    lookup = ranked_ev.set_index("snowroutesegmentid")[inject_fields].to_dict("index")

    for feat in gj.get("features", []):
        props = feat.get("properties", {}) or {}
        seg_id = str(props.get("snowroutesegmentid", ""))
        if seg_id in lookup:
            props.update(lookup[seg_id])
            props["__color__"] = bucket_color(props.get(bucket_field))
            props["__has_rank__"] = True
        else:
            props["__color__"] = [200, 200, 200, 40]
            props["__has_rank__"] = False
        feat["properties"] = props

    return gj, bucket_field


def compute_route_progress(df_event: pd.DataFrame, ranked_ev: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """
    Route-level progress = % of segments in the event's universe for that route
    that have been serviced (first_serviced_event <= now).
    Universe = distinct segments seen in snapshots for this event (after enrichment).
    """
    # total segments per route seen in snapshots for this event
    universe = (
        df_event.dropna(subset=["snowroutesegmentid", "snowrouteid"])
               .drop_duplicates(subset=["snowroutesegmentid", "snowrouteid"])
               [["snowroutesegmentid", "snowrouteid"]]
    )

    totals = universe.groupby("snowrouteid", as_index=False).agg(total_segments=("snowroutesegmentid", "count"))

    serviced = (
        ranked_ev.loc[ranked_ev["first_serviced_event"] <= now]
                .groupby("snowrouteid", as_index=False)
                .agg(serviced_segments=("snowroutesegmentid", "count"))
    )

    out = totals.merge(serviced, on="snowrouteid", how="left").fillna({"serviced_segments": 0})
    out["serviced_segments"] = out["serviced_segments"].astype(int)
    out["total_segments"] = out["total_segments"].astype(int)
    out["pct_complete"] = out["serviced_segments"] / out["total_segments"].clip(lower=1)
    return out.sort_values("pct_complete", ascending=False)


def main():
    st.title("Syracuse Snow Plow — Street Rank + Progress")

    df, events, ranked, stats, cfg = load_all()
    base_gj = load_geojson()
    feature_lookup = _feature_lookup_by_segment(base_gj)

    # Controls row
    colA, colB, colC, colD = st.columns([1.3, 1.0, 1.0, 1.2])
    with colA:
        eventid = st.selectbox(
            "Event",
            options=sorted(ranked["eventid"].unique().tolist()),
            index=0,
            help="Ranking is computed per eventid using first in-event service time.",
        )
    with colB:
        use_priority_adjusted = st.toggle("Priority-adjusted view", value=True)
    with colC:
        show_table = st.toggle("Show ranked table", value=False)
    with colD:
        resident_mode = st.toggle("Resident-safe phrasing", value=True, help="Hide internal IDs, show simpler language + ranges.")

    ranked_ev = ranked[ranked["eventid"] == str(eventid)].copy()
    ranked_ev = ranked_ev.sort_values("rank").reset_index(drop=True)

    # Per-event snapshot slice for route progress universe
    df_event = df[df["eventid"] == str(eventid)].copy()

    now = ranked_ev["event_end"].max()
    serviced_so_far = int((ranked_ev["first_serviced_event"] <= now).sum())
    total_serviced = int(ranked_ev["n_segments_serviced"].max())

    # top status line
    st.caption(
        f"Event start: {ranked_ev['event_start'].min().isoformat()} • "
        f"Now: {now.isoformat()} • "
        f"Serviced so far: {serviced_so_far:,} / {total_serviced:,}"
    )

    # Layout: left narrative, right map
    left, right = st.columns([0.45, 0.55], gap="large")

    with left:
        st.subheader("Find your street")

        q = st.text_input("Search road name or segment id", value="", placeholder="e.g., James St or SNW-479")
        candidates = ranked_ev

        if q.strip():
            qn = q.strip().lower()
            candidates = candidates[
                candidates["snowroutesegmentid"].str.lower().str.contains(qn, na=False)
                | candidates["roadname"].fillna("").str.lower().str.contains(qn, na=False)
            ]

        candidates = candidates.head(2500)
        pick = st.selectbox(
            "Select a segment",
            options=candidates["snowroutesegmentid"].tolist(),
            format_func=lambda sid: f"{sid} — {candidates.loc[candidates['snowroutesegmentid']==sid,'roadname'].iloc[0]}",
        )

        # Route progress bar for picked route
        picked_route = candidates.loc[candidates["snowroutesegmentid"] == pick, "snowrouteid"].iloc[0]
        route_prog = compute_route_progress(df_event, ranked_ev, now=now)
        rp = route_prog.loc[route_prog["snowrouteid"] == picked_route]
        if not rp.empty:
            pct = float(rp["pct_complete"].iloc[0])
            serviced_cnt = int(rp["serviced_segments"].iloc[0])
            total_cnt = int(rp["total_segments"].iloc[0])
            st.markdown("### Route progress")
            st.progress(pct)
            st.caption(f"Route **{picked_route}** is **{pct*100:.0f}%** complete ({serviced_cnt}/{total_cnt} segments serviced so far).")

        payload = answer_for_segment(ranked, stats, eventid=eventid, snowroutesegmentid=pick, now=now)

        st.markdown("### What to tell a resident" if resident_mode else "### Internal detail (ops view)")
        if not payload.get("found"):
            st.warning(payload.get("reason", "Not found"))
        else:
            rank = payload["rank"]
            N = payload["total_serviced_segments_in_event"]
            Z = payload["serviced_so_far"]
            remaining = payload["remaining_before_you_est"]

            road = payload.get("roadname") or "(unknown road)"
            routeid = payload.get("snowrouteid", "")
            priority = payload.get("routepriority", "")

            eta_h = payload["eta_minutes_from_now_est"] / 60.0
            eta_low = max(0.0, eta_h * 0.6)
            eta_high = eta_h * 1.4

            # Resident-safe phrasing
            if resident_mode:
                st.write(f"**{road}**")
                st.write(f"• Your street segment is typically **#{rank} out of {N}** in this event’s service order.")
                st.write(f"• So far **{Z} segments** have been serviced — you have roughly **{max(remaining, 0)}** segments ahead of you.")
                st.write(f"• Based on the current pace, estimate **~{eta_low:.1f}–{eta_high:.1f} hours** from now (rough).")
                if routeid:
                    st.write(f"• You’re on **{routeid}** — streets on the same route tend to be serviced close together.")
                if priority:
                    st.write(f"• This route is **{priority}** (priority routes are usually addressed earlier).")
            else:
                st.write(
                    f"**{road}** (segment **{payload['segment']}**, {routeid}, {priority})"
                )
                st.write(f"rank={rank}/{N}, serviced_so_far={Z}, remaining_before_you≈{max(remaining, 0)}")
                st.write(f"minutes_from_start_actual={payload['minutes_from_start_actual']:.1f} min")
                st.write(f"eta_minutes_from_now_est={payload['eta_minutes_from_now_est']:.1f} min")

            # Cross-event stability (uncertainty)
            if payload.get("n_events_history", 0) > 1:
                st.markdown("### Consistency across storms")
                st.write(f"• This segment is **{payload.get('consistency_label','unknown')}** across **{payload['n_events_history']} events**.")
                st.write(
                    f"• Typical position: **{payload['rank_pct_median']*100:.0f}%** (IQR ± **{payload['rank_pct_iqr']*100:.0f}%**)."
                )
                st.write(
                    f"• Typical time to first service: **{payload['minutes_median']/60:.1f}h** (IQR ± **{payload['minutes_iqr']/60:.1f}h**)."
                )
            else:
                st.caption("Not enough event history yet to compute median/IQR uncertainty.")

        st.markdown("### Legend")
        render_legend()

    with right:
        st.subheader("Map")

        gj_for_map, bucket_field = build_map_geojson(base_gj, ranked_ev, use_priority_adjusted)

        # Zoom-to-selected segment (fallback to Syracuse center)
        centroid = feature_lookup.get(str(pick), (None, None))[1]
        if centroid:
            lon, lat = centroid
            view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=14, pitch=0)
        else:
            view_state = pdk.ViewState(latitude=43.0481, longitude=-76.1474, zoom=11, pitch=0)

        layer = pdk.Layer(
            "GeoJsonLayer",
            gj_for_map,
            pickable=True,
            stroked=True,
            filled=False,
            get_line_color="properties.__color__",
            get_line_width=4,
            line_width_min_pixels=1,
        )

        # Tooltip: resident-safe vs ops
        if resident_mode:
            tooltip = {
                "html": (
                    "<b>{roadname}</b><br/>"
                    "Route: {snowrouteid}<br/>"
                    "Priority: {routepriority}<br/>"
                    "Service order rank: {rank}<br/>"
                    f"Bucket: {{{bucket_field}}}<br/>"
                )
            }
        else:
            tooltip = {
                "html": (
                    "<b>{roadname}</b><br/>"
                    "Segment: {snowroutesegmentid}<br/>"
                    "Route: {snowrouteid}<br/>"
                    "Priority: {routepriority}<br/>"
                    "Rank: {rank}<br/>"
                    f"Bucket: {{{bucket_field}}}<br/>"
                    "Minutes from start: {minutes_from_start}"
                )
            }

        st.pydeck_chart(
            pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip),
            use_container_width=True,
        )

        st.caption("Hover segments for details. Use the left search/select to zoom to a specific segment.")

        with st.expander("Route progress leaderboard"):
            route_prog = compute_route_progress(df_event, ranked_ev, now=now)
            route_prog_disp = route_prog.copy()
            route_prog_disp["pct_complete"] = (route_prog_disp["pct_complete"] * 100).round(1)
            st.dataframe(route_prog_disp, use_container_width=True, height=380)

    if show_table:
        st.subheader("Ranked segments (this event)")
        cols = [
            "rank", "snowroutesegmentid", "roadname", "snowrouteid", "routepriority",
            "first_serviced_event", "minutes_from_start", "bucket", "bucket_priority_adjusted"
        ]
        cols = [c for c in cols if c in ranked_ev.columns]
        show = ranked_ev[cols].copy()
        st.dataframe(show, use_container_width=True, height=520)

    # Optional: write enriched GeoJSON artifact for this event
    with st.expander("Export ranked GeoJSON for this event"):
        out_path = ARTIFACT_DIR / f"ranked_{eventid}.geojson"
        if st.button("Write ranked GeoJSON"):
            write_ranked_geojson(GEOJSON_PATH, ranked_ev, out_path)
            st.success(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
