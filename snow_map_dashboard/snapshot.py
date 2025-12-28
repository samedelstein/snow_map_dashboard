import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_QUERY_URL = (
    "https://services6.arcgis.com/bdPqSfflsdgFRVVM/arcgis/rest/services/"
    "Winter_Operations_Snow_Routes/FeatureServer/0/query"
)

PAGE_SIZE = 2000  # layer maxRecordCount
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "snapshot_snow_routes"
STATE_PATH = DATA_DIR / "state.json"
LATEST_GEOJSON_PATH = DATA_DIR / "latest_routes.geojson"
SNAPSHOTS_CSV_PATH = DATA_DIR / "snapshots.csv"

# Only changes to these fields create a new historical row.
CHANGE_FIELDS = [
    "lastserviced",
    "lastserviceleft",
    "lastserviceright",
    "passes",
    "passesleft",
    "passesright",
    "servicestatus",
    "eventid",
    "eventphaseid",
]

# What you store in the CSV history (can be more than CHANGE_FIELDS)
CSV_FIELDS = [
    "snapshot_ts",
    "OBJECTID",
    "snowrouteid",
    "snowroutesegmentid",
    "roadname",
    "routepriority",
    "roadclass",
    "segmentlength",
    "servicestatus",
    "lastserviced",
    "lastserviceleft",
    "lastserviceright",
    "passes",
    "passesleft",
    "passesright",
    "eventid",
    "eventphaseid",
]


def fetch_all_features_esrijson():
    """
    Fetch ALL features with reliable paging using ESRI JSON (f=json).
    Returns a list of ESRI features: {"attributes": {...}, "geometry": {...}}.
    """
    features_all = []
    offset = 0

    while True:
        params = {
            "f": "json",
            "where": "1=1",
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": 4326,  # server returns lon/lat so we don't need WebMercator conversion
            "resultOffset": offset,
            "resultRecordCount": PAGE_SIZE,
        }
        r = requests.get(BASE_QUERY_URL, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()

        batch = data.get("features", [])
        if not batch:
            break

        features_all.extend(batch)

        # If server says we exceeded transfer limit, keep paging.
        if not data.get("exceededTransferLimit", False):
            break

        offset += PAGE_SIZE

    return features_all


def esri_features_to_geojson_featurecollection(esri_features):
    """
    Convert ESRI polyline features to GeoJSON FeatureCollection.

    ESRI geometry for polylines uses:
      geometry: { "paths": [ [ [x,y], [x,y], ... ], ... ] }

    GeoJSON equivalent:
      MultiLineString with coordinates: [ [ [lon,lat], ... ], [ ... ] ]
    """
    geo_features = []

    for feat in esri_features:
        props = feat.get("attributes", {}) or {}
        geom = feat.get("geometry", {}) or {}
        paths = geom.get("paths", [])

        # Some rows might be missing geometry; skip safely
        if not paths:
            continue

        coordinates = []
        for path in paths:
            # each path is a list of [x,y]
            coordinates.append([[pt[0], pt[1]] for pt in path])

        geo_features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {
                    "type": "MultiLineString",
                    "coordinates": coordinates,
                },
            }
        )

    return {"type": "FeatureCollection", "features": geo_features}


def write_latest_geojson(fc, path=LATEST_GEOJSON_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f)


def load_state(path=STATE_PATH):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state, path=STATE_PATH):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    tmp.replace(path)


def feature_fingerprint(props: dict) -> str:
    """
    Hash only the fields we care about for "did it change?"
    """
    payload = {k: props.get(k) for k in CHANGE_FIELDS}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def append_changed_rows_from_geojson(fc, state, csv_path=SNAPSHOTS_CSV_PATH):
    """
    Append a row per *changed* feature (by OBJECTID fingerprint).
    Updates state in-memory; caller must save_state().
    """
    snapshot_ts = datetime.now(timezone.utc).isoformat()
    file_exists = csv_path.exists()
    changed_count = 0

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            w.writeheader()

        for feat in fc["features"]:
            props = feat.get("properties", {}) or {}
            objectid = props.get("OBJECTID")
            if objectid is None:
                continue

            oid_key = str(objectid)
            fp = feature_fingerprint(props)
            prev_fp = state.get(oid_key)

            if prev_fp == fp:
                continue  # unchanged

            # changed (or first time) -> append snapshot row
            row = {k: props.get(k) for k in CSV_FIELDS if k != "snapshot_ts"}
            row["snapshot_ts"] = snapshot_ts
            w.writerow(row)

            state[oid_key] = fp
            changed_count += 1

    return changed_count


def main():
    esri_features = fetch_all_features_esrijson()
    print(f"Fetched ESRI features: {len(esri_features)}")

    fc = esri_features_to_geojson_featurecollection(esri_features)
    print(f"GeoJSON features (with geometry): {len(fc['features'])}")

    # Always refresh latest for your dashboard
    write_latest_geojson(fc, LATEST_GEOJSON_PATH)

    # Append history only when something changed
    state = load_state(STATE_PATH)
    changed = append_changed_rows_from_geojson(fc, state, SNAPSHOTS_CSV_PATH)
    save_state(state, STATE_PATH)

    print(f"Changed features appended this run: {changed}")


if __name__ == "__main__":
    main()
