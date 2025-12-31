from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def parse_priority(val) -> Optional[int]:
    """Turn 'Priority 1' / 'Priority 2' into 1/2. Returns None if unknown."""
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    if s.isdigit():
        return int(s)
    s = s.replace("-", " ")
    for tok in s.split():
        if tok.isdigit():
            return int(tok)
    return None


def iqr(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) == 0:
        return np.nan
    return float(x.quantile(0.75) - x.quantile(0.25))


def ensure_utc_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def parse_lastserviced(series: pd.Series) -> pd.Series:
    """
    Handles numeric epoch-ms (common in ArcGIS) OR ISO strings.
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="ms", utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


@dataclass
class EventRankingConfig:
    """
    priority_lambda:
      - 0.00 disables priority calibration.
      - ~0.03–0.08 is a gentle nudge earlier for higher priority.
    """
    default_event_start_strategy: str = "min_snapshot_ts"  # or "min_first_serviced"
    priority_lambda: float = 0.06
    n_buckets: int = 5


# ------------------------------------------------------------
# Load
# ------------------------------------------------------------
def load_snapshots(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # normalize timestamps
    df["snapshot_ts"] = ensure_utc_ts(df["snapshot_ts"])
    df["lastserviced"] = parse_lastserviced(df["lastserviced"])

    # normalize IDs as strings (prevents numeric quirks / NaN issues)
    for col in ["eventid", "snowroutesegmentid", "snowrouteid"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # priority
    if "routepriority" in df.columns:
        df["routepriority_num"] = df["routepriority"].apply(parse_priority)
    else:
        df["routepriority_num"] = np.nan

    return df


# ------------------------------------------------------------
# Event start per event
# ------------------------------------------------------------
def compute_event_start(
    df: pd.DataFrame,
    cfg: EventRankingConfig,
    overrides: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Returns one row per event:
      eventid, event_start, event_end, snapshots_min/max
    """
    overrides = overrides or {}

    events = df.groupby("eventid", as_index=False).agg(
        snapshots_min=("snapshot_ts", "min"),
        snapshots_max=("snapshot_ts", "max"),
    )

    if cfg.default_event_start_strategy == "min_snapshot_ts":
        events["event_start"] = events["snapshots_min"]
    else:
        tmp = (
            df.dropna(subset=["lastserviced"])
              .groupby("eventid", as_index=False)["lastserviced"]
              .min()
              .rename(columns={"lastserviced": "event_start"})
        )
        events = events.merge(tmp, on="eventid", how="left")
        events["event_start"] = events["event_start"].fillna(events["snapshots_min"])

    # apply overrides
    if overrides:
        o = pd.Series(overrides, name="override").rename_axis("eventid").reset_index()
        o["override"] = pd.to_datetime(o["override"], utc=True, errors="coerce")
        events = events.merge(o, on="eventid", how="left")
        events["event_start"] = events["override"].combine_first(events["event_start"])
        events = events.drop(columns=["override"])

    events["event_end"] = events["snapshots_max"]
    return events


# ------------------------------------------------------------
# Rank within events
# ------------------------------------------------------------
def rank_segments_within_events(
    df: pd.DataFrame,
    events: pd.DataFrame,
    cfg: EventRankingConfig,
) -> pd.DataFrame:
    """
    Output: one row per (eventid, snowroutesegmentid) for segments serviced in-event.
    """
    base = df.merge(events[["eventid", "event_start", "event_end"]], on="eventid", how="left")

    serviced = (
        base.dropna(subset=["lastserviced"])
            .loc[base["lastserviced"] > base["event_start"]]
            .groupby(["eventid", "snowroutesegmentid"], as_index=False)
            .agg(
                first_serviced_event=("lastserviced", "min"),
                snowrouteid=("snowrouteid", "first"),
                routepriority=("routepriority", "first"),
                routepriority_num=("routepriority_num", "first"),
                roadname=("roadname", "first"),
                event_start=("event_start", "first"),
                event_end=("event_end", "first"),
            )
    )

    serviced["first_serviced_event"] = pd.to_datetime(serviced["first_serviced_event"], utc=True, errors="coerce")

    serviced = serviced.sort_values(["eventid", "first_serviced_event"]).reset_index(drop=True)
    serviced["rank"] = serviced.groupby("eventid").cumcount() + 1
    serviced["n_segments_serviced"] = serviced.groupby("eventid")["snowroutesegmentid"].transform("count")
    serviced["rank_pct"] = serviced["rank"] / serviced["n_segments_serviced"]

    serviced["minutes_from_start"] = (
        serviced["first_serviced_event"] - serviced["event_start"]
    ).dt.total_seconds() / 60.0

    # Priority calibration (soft nudge)
    pr = serviced["routepriority_num"].fillna(99).astype(float)
    pr_shift = (pr - 1.0) * float(cfg.priority_lambda)
    serviced["rank_pct_priority_adjusted"] = (serviced["rank_pct"] + pr_shift).clip(0, 1)

    # Buckets for UX
    labels = [f"Bucket {i+1}" for i in range(cfg.n_buckets)]
    serviced["bucket"] = pd.qcut(serviced["rank_pct"], q=cfg.n_buckets, labels=labels, duplicates="drop")
    serviced["bucket_priority_adjusted"] = pd.qcut(
        serviced["rank_pct_priority_adjusted"], q=cfg.n_buckets, labels=labels, duplicates="drop"
    )

    return serviced


# ------------------------------------------------------------
# Cross-event stability
# ------------------------------------------------------------
def summarize_across_events(ranked: pd.DataFrame) -> pd.DataFrame:
    """
    One row per snowroutesegmentid:
      - typical percentile rank (median) and uncertainty (IQR)
      - typical minutes-to-first-service (median) and uncertainty (IQR)
    """
    def classify(median_rank_pct: float) -> str:
        if pd.isna(median_rank_pct):
            return "unknown"
        if median_rank_pct <= 0.20:
            return "consistently early"
        if median_rank_pct >= 0.80:
            return "consistently late"
        if median_rank_pct <= 0.40:
            return "often early"
        if median_rank_pct >= 0.60:
            return "often late"
        return "middle"

    agg = (
        ranked.groupby("snowroutesegmentid", as_index=False)
              .agg(
                  roadname=("roadname", "first"),
                  snowrouteid=("snowrouteid", "first"),
                  routepriority=("routepriority", "first"),
                  routepriority_num=("routepriority_num", "first"),
                  n_events=("eventid", "nunique"),
                  rank_pct_median=("rank_pct", "median"),
                  rank_pct_iqr=("rank_pct", iqr),
                  minutes_median=("minutes_from_start", "median"),
                  minutes_iqr=("minutes_from_start", iqr),
              )
    )
    agg["consistency_label"] = agg["rank_pct_median"].apply(classify)
    return agg


# ------------------------------------------------------------
# Resident-facing answer (per event)
# ------------------------------------------------------------
def answer_for_segment(
    ranked: pd.DataFrame,
    stats: pd.DataFrame,
    eventid: str,
    snowroutesegmentid: str,
    now: Optional[pd.Timestamp] = None,
) -> dict:
    """
    Returns a dict you can render in Streamlit.
    """
    eventid = str(eventid)
    snowroutesegmentid = str(snowroutesegmentid)

    ev = ranked.loc[ranked["eventid"] == eventid].copy()
    if ev.empty:
        return {"found": False, "reason": f"No ranked data for eventid={eventid}"}

    if now is None:
        now = ev["event_end"].max()
    now = pd.to_datetime(now, utc=True, errors="coerce")

    seg = ev.loc[ev["snowroutesegmentid"] == snowroutesegmentid]
    if seg.empty:
        return {"found": False, "reason": f"Segment not serviced in-event (or not present): {snowroutesegmentid}"}
    seg = seg.iloc[0]

    Z = int((ev["first_serviced_event"] <= now).sum())
    N = int(ev["n_segments_serviced"].max())

    elapsed_min = float((now - seg["event_start"]).total_seconds() / 60.0)
    avg_min_per_seg = elapsed_min / max(Z, 1)
    eta_from_now_min = float(max(seg["rank"] - Z, 0) * avg_min_per_seg)

    hist = stats.loc[stats["snowroutesegmentid"] == snowroutesegmentid]
    hist_row = hist.iloc[0].to_dict() if not hist.empty else {}

    return {
        "found": True,
        "eventid": eventid,
        "now": now.isoformat(),
        "segment": snowroutesegmentid,
        "roadname": seg.get("roadname"),
        "snowrouteid": seg.get("snowrouteid"),
        "routepriority": seg.get("routepriority"),

        "rank": int(seg["rank"]),
        "total_serviced_segments_in_event": N,
        "serviced_so_far": Z,
        "remaining_before_you_est": int(max(seg["rank"] - Z - 1, 0)),

        "event_start": seg["event_start"].isoformat(),
        "first_serviced_event": seg["first_serviced_event"].isoformat(),
        "minutes_from_start_actual": float(seg["minutes_from_start"]),
        "eta_minutes_from_now_est": eta_from_now_min,

        "bucket": str(seg.get("bucket")),
        "bucket_priority_adjusted": str(seg.get("bucket_priority_adjusted")),

        "n_events_history": int(hist_row.get("n_events", 0)) if hist_row else 0,
        "consistency_label": hist_row.get("consistency_label"),
        "rank_pct_median": float(hist_row.get("rank_pct_median", np.nan)) if hist_row else np.nan,
        "rank_pct_iqr": float(hist_row.get("rank_pct_iqr", np.nan)) if hist_row else np.nan,
        "minutes_median": float(hist_row.get("minutes_median", np.nan)) if hist_row else np.nan,
        "minutes_iqr": float(hist_row.get("minutes_iqr", np.nan)) if hist_row else np.nan,
    }


# ------------------------------------------------------------
# GeoJSON merge helper (adds rank/bucket fields to features)
# ------------------------------------------------------------
def write_ranked_geojson(
    geojson_path: Path,
    ranked_event: pd.DataFrame,
    out_path: Path,
    fields: Optional[list] = None,
) -> None:
    fields = fields or [
        "rank",
        "rank_pct",
        "bucket",
        "bucket_priority_adjusted",
        "minutes_from_start",
        "snowrouteid",
        "routepriority",
    ]

    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    # lookup by segment id
    keep = [c for c in fields if c in ranked_event.columns]
    lookup = ranked_event.set_index("snowroutesegmentid")[keep].to_dict("index")

    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        seg_id = str(props.get("snowroutesegmentid", ""))
        if seg_id in lookup:
            # ensure JSON-serializable
            upd = {}
            for k, v in lookup[seg_id].items():
                if pd.isna(v):
                    upd[k] = None
                elif hasattr(v, "item"):
                    upd[k] = v.item()
                else:
                    upd[k] = v
            props.update(upd)
            feat["properties"] = props

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gj, f)
