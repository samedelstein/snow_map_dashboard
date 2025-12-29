#!/usr/bin/env python
"""Export static site artifacts from the latest snapshot + predictions."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SNAPSHOT_GEOJSON = DATA_DIR / "snapshot_snow_routes" / "latest_routes.geojson"
PREDICTIONS_CSV = DATA_DIR / "artifacts_snow" / "predictions_latest_prob.csv"
STATIC_SITE_DIR = REPO_ROOT / "static_site" / "data"

DEFAULT_CUTOFF_DATE_STR = "2025-12-01"

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


def format_probability(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.0f}%"


def format_eta_hours(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.1f} hours"


def format_route_label(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    try:
        return f"Route {int(value)}"
    except (TypeError, ValueError):
        return str(value)


def eta_tier(value: float | None) -> tuple[str, int]:
    if value is None or pd.isna(value):
        return "Unknown", -1
    value = float(value)
    if value < 1:
        return "< 1 hour", 0
    if value < 2:
        return "1–2 hours", 1
    if value < 4:
        return "2–4 hours", 2
    if value < 8:
        return "4–8 hours", 3
    return "8+ hours", 4


def classify_bucket(lastserviced_ts: pd.Timestamp | None, snapshot_ts: pd.Timestamp) -> str | None:
    if lastserviced_ts is None or pd.isna(lastserviced_ts):
        return None
    cutoff_dt = pd.to_datetime(DEFAULT_CUTOFF_DATE_STR, utc=True, errors="coerce")
    if pd.isna(cutoff_dt):
        cutoff_dt = pd.Timestamp.min.tz_localize("UTC")
    if lastserviced_ts < cutoff_dt:
        return BUCKET_ORDER[0]
    hours = (snapshot_ts - lastserviced_ts).total_seconds() / 3600.0
    if hours < 1:
        return BUCKET_ORDER[5]
    if hours < 6:
        return BUCKET_ORDER[4]
    if hours < 12:
        return BUCKET_ORDER[3]
    if hours < 24:
        return BUCKET_ORDER[2]
    return BUCKET_ORDER[1]


def load_geojson(path: Path) -> dict:
    if not path.exists():
        return {"type": "FeatureCollection", "features": []}
    return json.loads(path.read_text(encoding="utf-8"))


def build_snapshot_index(geojson_obj: dict) -> tuple[dict[tuple[str, str], dict], dict[str, dict]]:
    event_index: dict[tuple[str, str], dict] = {}
    seg_index: dict[str, dict] = {}
    for feat in geojson_obj.get("features", []):
        props = feat.get("properties", {}) or {}
        seg = str(props.get("snowroutesegmentid"))
        eventid = props.get("eventid")
        if eventid is not None:
            event_index[(eventid, seg)] = feat
        if seg not in seg_index:
            seg_index[seg] = feat
    return event_index, seg_index


def build_predictions_geojson(snapshot_geojson: dict, predictions_df: pd.DataFrame) -> dict:
    if predictions_df.empty:
        return {"type": "FeatureCollection", "features": []}

    predictions = predictions_df.copy()
    if "snapshot_ts" in predictions.columns:
        predictions["snapshot_ts"] = pd.to_datetime(predictions["snapshot_ts"], utc=True, errors="coerce")
        latest_ts = predictions["snapshot_ts"].max()
        if not pd.isna(latest_ts):
            predictions = predictions[predictions["snapshot_ts"] == latest_ts]
    if "snowroutesegmentid" in predictions.columns:
        predictions["snowroutesegmentid"] = predictions["snowroutesegmentid"].astype(str)

    event_index, seg_index = build_snapshot_index(snapshot_geojson)

    features = []
    for _, row in predictions.iterrows():
        seg = str(row.get("snowroutesegmentid"))
        eventid = row.get("eventid")
        base_feat = event_index.get((eventid, seg)) or seg_index.get(seg)
        if base_feat is None:
            continue

        snapshot_props = base_feat.get("properties", {}) or {}
        props = dict(snapshot_props)

        for field in [
            "OBJECTID",
            "snowroutesegmentid",
            "eventid",
            "snowrouteid",
            "roadname",
            "routepriority",
            "prediction_status",
            "p_1h",
            "p_2h",
            "p_4h",
            "p_8h",
            "eta_hours_60",
            "eta_ts_60",
            "eta_hours_pred",
        ]:
            if field in row.index and pd.notna(row[field]):
                value = row[field]
                if isinstance(value, pd.Timestamp):
                    value = value.isoformat()
                props[field] = value

        props.setdefault("prediction_status", "NO_PRED_INSUFFICIENT_DATA")

        snapshot_ts = row.get("snapshot_ts")
        if not isinstance(snapshot_ts, pd.Timestamp) or pd.isna(snapshot_ts):
            snapshot_ts = pd.Timestamp.now(tz="UTC")
        lastserviced = row.get("lastserviced")
        lastserviced_ts = pd.to_datetime(lastserviced, utc=True, unit="ms", errors="coerce")
        bucket = classify_bucket(lastserviced_ts, snapshot_ts)
        props["bucket"] = bucket
        props["bucket_label"] = BUCKET_DISPLAY_LABELS.get(bucket, "Unknown")

        props["route_label"] = format_route_label(props.get("snowrouteid"))
        props["p_1h_pct"] = format_probability(props.get("p_1h"))
        props["p_2h_pct"] = format_probability(props.get("p_2h"))
        props["p_4h_pct"] = format_probability(props.get("p_4h"))
        props["p_8h_pct"] = format_probability(props.get("p_8h"))
        props["eta_60_label"] = format_eta_hours(props.get("eta_hours_60"))
        eta_label, eta_rank = eta_tier(props.get("eta_hours_60"))
        props["eta_tier"] = eta_label
        props["eta_tier_rank"] = eta_rank

        features.append({"type": "Feature", "geometry": base_feat.get("geometry"), "properties": props})

    return {"type": "FeatureCollection", "features": features}


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    STATIC_SITE_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_geojson = load_geojson(SNAPSHOT_GEOJSON)
    write_json(STATIC_SITE_DIR / "live_snapshot.geojson", snapshot_geojson)

    if PREDICTIONS_CSV.exists():
        preds = pd.read_csv(PREDICTIONS_CSV)
    else:
        preds = pd.DataFrame()

    predictions_geojson = build_predictions_geojson(snapshot_geojson, preds)
    write_json(STATIC_SITE_DIR / "predictions_latest.geojson", predictions_geojson)

    now = datetime.now(timezone.utc).isoformat()
    meta = {
        "generated_at": now,
        "live_snapshot": {"generated_at": now},
        "predictions": {"generated_at": now},
    }
    write_json(STATIC_SITE_DIR / "meta.json", meta)


if __name__ == "__main__":
    main()
