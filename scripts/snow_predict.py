"""
Snowplow probability prediction model (robust, geometry optional).
Uses local snapshot + GeoJSON files by default, with optional CLI/env overrides
for remote snapshot training.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as sk_metrics
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import BallTree

# -----------------------------
# Config (defaults to local snapshot/geojson files)
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SNAPSHOT_PATH = str(DATA_DIR / "snapshot_snow_routes" / "snapshots.csv")
REMOTE_SNAPSHOT_URL = (
    "https://raw.githubusercontent.com/samedelstein/snow_map_dashboard/"
    "refs/heads/main/data/snapshot_snow_routes/snapshots.csv"
)
GEOJSON_PATH = str(REPO_ROOT / "winter_operations_snow_routes_layer0.geojson")

EVENT = "eventid"
SEG = "snowroutesegmentid"
TS = "snapshot_ts"

HORIZONS = [1, 2, 4, 8]
ETA_THRESHOLD_DEFAULT = 0.60
CALIBRATION_MIN_ROWS_ISOTONIC = 500
RANKING_KS = [10, 25, 50, 100]
ARTIFACT_DIR = DATA_DIR / "artifacts_snow"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
ALERT_LOG_PATH = str(ARTIFACT_DIR / "nws_alerts_log.csv")
BUCKET_LADDER_PATH = ARTIFACT_DIR / "bucket_ladder.csv"
BUCKET_LADDER_PRIORITY_PATH = ARTIFACT_DIR / "bucket_ladder_priority.csv"

NOAA_STATION_ID = "KSYR"
NWS_POINT = "43.0481,-76.1474"
NWS_USER_AGENT = "snow_map_dashboard (https://github.com/samedelstein/snow_map_dashboard)"

# -----------------------------
# Storm windows (alerts + operational activity)
# -----------------------------
SNOW_ALERT_EVENTS = {
    "Winter Storm Warning",
    "Winter Storm Watch",
    "Winter Weather Advisory",
}

NWS_ALERTS_LOG_LOCAL = str(ARTIFACT_DIR / "nws_alerts_log.csv")
NWS_ALERTS_LOG_REMOTE = (
    "https://raw.githubusercontent.com/samedelstein/snow_map_dashboard/main/"
    "data/artifacts_snow/nws_alerts_log.csv"
)

# envelope expansion (tunable)
ALERT_START_PAD_H = 6
ALERT_END_PAD_H = 24

# operational start/end detection (tunable)
OPS_BUCKET_MIN = "15min"
OPS_MIN_SERVICES_PER_BUCKET = 10   # tune based on your segment count
OPS_SUSTAIN_BUCKETS = 2            # require sustained activity

# bucket baseline parameters
BUCKET_CURRENT_WINDOW = "30min"
BUCKET_SOFTNESS_HOURS = 1.75
BUCKET_BLEND_MIN_CITY_15M = 10
BUCKET_BLEND_LOW_CITY_15M = 2
BUCKET_BLEND_WEIGHT_HIGH = 0.75
BUCKET_BLEND_WEIGHT_MID = 0.6
BUCKET_BLEND_WEIGHT_LOW = 0.45


def load_alert_log(source: str | None = None) -> pd.DataFrame:
    """
    Load alerts log from local path or raw GitHub URL.
    Returns df with columns: event, start_ts, end_ts, severity
    """
    src = source or (NWS_ALERTS_LOG_LOCAL if os.path.exists(NWS_ALERTS_LOG_LOCAL) else NWS_ALERTS_LOG_REMOTE)
    try:
        a = pd.read_csv(src)
    except Exception:
        return pd.DataFrame(columns=["event", "start_ts", "end_ts", "severity"])

    if a.empty:
        return pd.DataFrame(columns=["event", "start_ts", "end_ts", "severity"])

    a["start_ts"] = pd.to_datetime(a["start_ts"], utc=True, errors="coerce")
    a["end_ts"] = pd.to_datetime(a["end_ts"], utc=True, errors="coerce")
    a["event"] = a["event"].astype(str)
    a["severity"] = a.get("severity", "").astype(str)
    a = a[a["start_ts"].notna() & a["end_ts"].notna()].copy()
    return a


def build_storm_envelopes_from_alerts(alerts: pd.DataFrame) -> pd.DataFrame:
    """
    Merge overlapping snow-related alerts into storm envelopes.
    Output columns:
      storm_id, storm_envelope_start, storm_envelope_end, severity_max
    """
    if alerts.empty:
        return pd.DataFrame(columns=["storm_id", "storm_envelope_start", "storm_envelope_end", "severity_max"])

    a = alerts[alerts["event"].isin(SNOW_ALERT_EVENTS)].copy()
    if a.empty:
        return pd.DataFrame(columns=["storm_id", "storm_envelope_start", "storm_envelope_end", "severity_max"])

    # pad windows
    a["storm_envelope_start"] = a["start_ts"] - pd.Timedelta(hours=ALERT_START_PAD_H)
    a["storm_envelope_end"] = a["end_ts"] + pd.Timedelta(hours=ALERT_END_PAD_H)

    # severity ordering
    sev_rank = {"Minor": 1, "Moderate": 2, "Severe": 3, "Extreme": 4}
    a["sev_rank"] = a["severity"].map(sev_rank).fillna(0).astype(int)

    # merge overlaps by time
    a = a.sort_values("storm_envelope_start").reset_index(drop=True)
    merged = []
    cur_start = None
    cur_end = None
    cur_sev = 0

    for row in a.itertuples(index=False):
        s = row.storm_envelope_start
        e = row.storm_envelope_end
        sev = int(row.sev_rank)


        if cur_start is None:
            cur_start, cur_end, cur_sev = s, e, sev
            continue

        if s <= cur_end:  # overlap
            cur_end = max(cur_end, e)
            cur_sev = max(cur_sev, sev)
        else:
            merged.append((cur_start, cur_end, cur_sev))
            cur_start, cur_end, cur_sev = s, e, sev

    if cur_start is not None:
        merged.append((cur_start, cur_end, cur_sev))

    inv_sev = {v: k for k, v in sev_rank.items()}
    out = pd.DataFrame(
        [
            {
                "storm_id": f"storm_{i+1:03d}",
                "storm_envelope_start": s,
                "storm_envelope_end": e,
                "severity_max": inv_sev.get(sev, "Unknown"),
            }
            for i, (s, e, sev) in enumerate(merged)
        ]
    )
    return out


def derive_city_service_events(labeled: pd.DataFrame) -> pd.DataFrame:
    """
    Identify observed service events from lastserviced changes.
    Returns a DataFrame of (snapshot_ts) timestamps where any segment was serviced.
    """
    if labeled.empty:
        return pd.DataFrame(columns=[TS, EVENT])

    tmp = labeled.sort_values([EVENT, SEG, TS]).copy()
    tmp["prev_last"] = tmp.groupby([EVENT, SEG])["lastserviced"].shift(1)
    tmp["lastserviced_changed"] = (tmp["lastserviced"].notna()) & (tmp["lastserviced"] != tmp["prev_last"])
    svc = tmp.loc[tmp["lastserviced_changed"], [EVENT, TS]].copy()
    return svc


def refine_operational_windows(
    storms: pd.DataFrame,
    service_events: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each storm envelope, find operational start/end based on sustained city service counts.
    Output columns:
      storm_id, storm_envelope_start, storm_envelope_end,
      storm_operational_start, storm_operational_end, severity_max
    """
    if storms.empty:
        return pd.DataFrame(columns=[
            "storm_id","storm_envelope_start","storm_envelope_end",
            "storm_operational_start","storm_operational_end","severity_max"
        ])

    if service_events.empty:
        out = storms.copy()
        out["storm_operational_start"] = pd.NaT
        out["storm_operational_end"] = pd.NaT
        return out

    svc = service_events.copy()
    svc["bucket"] = svc[TS].dt.floor(OPS_BUCKET_MIN)

    out_rows = []
    for st in storms.itertuples(index=False):
        s0 = st.storm_envelope_start
        s1 = st.storm_envelope_end

        ssvc = svc[(svc[TS] >= s0) & (svc[TS] <= s1)]
        if ssvc.empty:
            out_rows.append({
                "storm_id": st.storm_id,
                "storm_envelope_start": s0,
                "storm_envelope_end": s1,
                "storm_operational_start": pd.NaT,
                "storm_operational_end": pd.NaT,
                "severity_max": st.severity_max,
            })
            continue

        counts = (
            ssvc.groupby("bucket")
                .size()
                .rename("services")
                .reset_index()
                .sort_values("bucket")
        )

        # sustained activity mask
        active = counts["services"] >= OPS_MIN_SERVICES_PER_BUCKET

        # find first run of OPS_SUSTAIN_BUCKETS active buckets
        op_start = pd.NaT
        op_end = pd.NaT
        if active.any():
            # rolling window to find sustained True blocks
            sustain = active.rolling(OPS_SUSTAIN_BUCKETS, min_periods=OPS_SUSTAIN_BUCKETS).sum() >= OPS_SUSTAIN_BUCKETS
            if sustain.any():
                first_idx = int(np.argmax(sustain.to_numpy()))
                op_start = counts.iloc[first_idx]["bucket"]

                # op_end = last active bucket time (or last sustained)
                last_active_idx = int(np.where(active.to_numpy())[0].max())
                op_end = counts.iloc[last_active_idx]["bucket"] + pd.Timedelta(OPS_BUCKET_MIN)

        out_rows.append({
            "storm_id": st.storm_id,
            "storm_envelope_start": s0,
            "storm_envelope_end": s1,
            "storm_operational_start": op_start,
            "storm_operational_end": op_end,
            "severity_max": st.severity_max,
        })

    return pd.DataFrame(out_rows)


def attach_storm_context(df: pd.DataFrame, storms_ops: pd.DataFrame) -> pd.DataFrame:
    """
    Attach storm_id and storm operational start/end to each snapshot row.
    If multiple storms overlap (rare after merging), pick the one whose envelope contains TS
    with the closest operational start (or first).
    """
    if df.empty or storms_ops.empty:
        out = df.copy()
        out["storm_id"] = "no_storm"
        out["storm_operational_start"] = pd.NaT
        out["storm_operational_end"] = pd.NaT
        out["storm_severity_max"] = "Unknown"
        return out

    storms = storms_ops.copy()
    storms["storm_operational_start"] = pd.to_datetime(storms["storm_operational_start"], utc=True, errors="coerce")
    storms["storm_operational_end"] = pd.to_datetime(storms["storm_operational_end"], utc=True, errors="coerce")

    out = df.copy()
    out["storm_id"] = "no_storm"

    # Force proper dtypes up-front (prevents object dtype warnings/errors)
    out["storm_operational_start"] = pd.to_datetime(pd.Series([pd.NaT] * len(out)), utc=True)
    out["storm_operational_end"] = pd.to_datetime(pd.Series([pd.NaT] * len(out)), utc=True)

    out["storm_severity_max"] = "Unknown"


    # simple interval assignment: loop storms (should be few)
    for st in storms.itertuples(index=False):
        mask = (out[TS] >= st.storm_envelope_start) & (out[TS] <= st.storm_envelope_end)
        out.loc[mask, "storm_id"] = st.storm_id
        out.loc[mask, "storm_operational_start"] = st.storm_operational_start
        out.loc[mask, "storm_operational_end"] = st.storm_operational_end
        out.loc[mask, "storm_severity_max"] = st.severity_max

    return out


class _PrefitCalibrator:
    def __init__(self, estimator: HistGradientBoostingClassifier, method: str) -> None:
        self.estimator = estimator
        self.method = method
        self.calibrator: IsotonicRegression | LogisticRegression | None = None
        self.classes_ = np.array([0, 1])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_PrefitCalibrator":
        p = self.estimator.predict_proba(X)[:, 1]
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(p, y)
        else:
            self.calibrator = LogisticRegression(solver="lbfgs")
            self.calibrator.fit(p.reshape(-1, 1), y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.calibrator is None:
            raise ValueError("Calibrator has not been fitted.")
        p = self.estimator.predict_proba(X)[:, 1]
        if isinstance(self.calibrator, IsotonicRegression):
            p_cal = self.calibrator.predict(p)
        else:
            p_cal = self.calibrator.predict_proba(p.reshape(-1, 1))[:, 1]
        p_cal = np.clip(p_cal, 0.0, 1.0)
        return np.column_stack([1 - p_cal, p_cal])


def _fit_prefit_calibrator(
    estimator: HistGradientBoostingClassifier,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    method: str,
) -> CalibratedClassifierCV | _PrefitCalibrator:
    try:
        calibrator = CalibratedClassifierCV(estimator, cv="prefit", method=method)
        calibrator.fit(X_calib, y_calib)
        return calibrator
    except ValueError:
        calibrator = _PrefitCalibrator(estimator, method)
        calibrator.fit(X_calib, y_calib)
        return calibrator


# -----------------------------
# Load snapshots
# -----------------------------
def load_snapshots(source: str) -> pd.DataFrame:
    df = pd.read_csv(source)
    df[TS] = pd.to_datetime(df[TS], utc=True, errors="coerce")

    for c in ["lastserviced", "lastserviceleft", "lastserviceright"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], unit="ms", utc=True, errors="coerce")

    df["routepriority"] = df.get("routepriority", "Unknown").fillna("Unknown")
    df["snowrouteid"] = df.get("snowrouteid", "Unknown").fillna("Unknown")
    df["roadname"] = df.get("roadname", "Unknown").fillna("Unknown")

    for c in ["passes", "passesleft", "passesright"]:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0)

    df["segmentlength"] = pd.to_numeric(df.get("segmentlength"), errors="coerce")
    df = df[df[TS].notna() & df[EVENT].notna() & df[SEG].notna()].copy()
    return df


# -----------------------------
# Geometry helpers
# -----------------------------
def _flatten_multilinestring(coords: Any) -> np.ndarray:
    if not coords:
        return np.empty((0, 2))
    arr = np.array(coords, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr
    if arr.ndim == 3:
        return arr.reshape(-1, 2)
    return np.empty((0, 2))


def load_geojson_features(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=[SEG, EVENT, "lon", "lat", "passes_phase"])

    with open(path, "r", encoding="utf-8") as f:
        geo = json.load(f)

    rows = []
    for feat in geo.get("features", []):
        props = feat.get("properties", {}) or {}
        coords = _flatten_multilinestring(feat.get("geometry", {}).get("coordinates"))
        if coords.size == 0:
            continue
        rows.append(
            {
                SEG: str(props.get(SEG)),
                EVENT: props.get(EVENT),
                "lon": float(np.nanmean(coords[:, 0])),
                "lat": float(np.nanmean(coords[:, 1])),
                "passes_phase": props.get("passes_phase"),
            }
        )

    geo_df = pd.DataFrame(rows)
    if not geo_df.empty:
        geo_df["passes_phase"] = pd.to_numeric(geo_df["passes_phase"], errors="coerce")
    return geo_df


def build_neighbor_lookup(centroids: pd.DataFrame, k: int = 12) -> dict[str, list[str]]:
    if centroids.empty:
        return {}
    rad = np.deg2rad(centroids[["lat", "lon"]].to_numpy())
    tree = BallTree(rad, metric="haversine")

    _, idx = tree.query(rad, k=min(k, len(centroids)))
    segs = centroids[SEG].to_numpy()
    return {
        str(segs[i]): [str(segs[j]) for j in idx[i] if segs[j] != segs[i]]
        for i in range(len(segs))
    }


def _safe_value(payload: Any) -> float | None:
    if isinstance(payload, dict):
        return payload.get("value")
    return None


def _collect_paginated(url: str, params: dict[str, str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    next_url = url
    next_params = params
    headers = {"User-Agent": NWS_USER_AGENT, "Accept": "application/geo+json"}

    while next_url:
        try:
            resp = requests.get(
                next_url, params=next_params, headers=headers, timeout=30
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"Weather API request failed: {exc}")
            break

        data = resp.json()
        results.extend(data.get("features", []))
        next_url = data.get("pagination", {}).get("next")
        next_params = None

    return results


def load_noaa_observations(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    url = f"https://api.weather.gov/stations/{NOAA_STATION_ID}/observations"
    params = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "limit": "500",
    }
    features = _collect_paginated(url, params)
    rows: list[dict[str, Any]] = []
    for feat in features:
        props = feat.get("properties", {}) or {}
        ts = pd.to_datetime(props.get("timestamp"), utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        present_weather = props.get("presentWeather")
        rows.append(
            {
                "obs_ts": ts,
                "temp_c": _safe_value(props.get("temperature")),
                "wind_speed_mps": _safe_value(props.get("windSpeed")),
                "wind_gust_mps": _safe_value(props.get("windGust")),
                "precip_mm": _safe_value(props.get("precipitationLastHour")),
                "snowfall_mm": _safe_value(props.get("snowfallLastHour")),
                "present_weather": json.dumps(present_weather),
                "text_description": props.get("textDescription"),
            }
        )
    return pd.DataFrame(rows)


def build_weather_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if df.empty:
        return pd.DataFrame(), {"observations": {}, "station_id": NOAA_STATION_ID}

    start = df[TS].min().floor("h") - pd.Timedelta(hours=1)
    end = df[TS].max().ceil("h") + pd.Timedelta(hours=1)

    obs = load_noaa_observations(start, end)
    if obs.empty:
        return pd.DataFrame(), {
            "observations": {"start": str(start), "end": str(end), "rows": 0},
            "station_id": NOAA_STATION_ID,
        }

    obs = obs.sort_values("obs_ts")
    obs["hour_bucket"] = obs["obs_ts"].dt.floor("h")
    hourly = obs.groupby("hour_bucket", as_index=False).tail(1).copy()

    hourly["snowfall_rate_mmhr"] = hourly["snowfall_mm"]
    temp_c = hourly["temp_c"]
    precip_mm = hourly["precip_mm"].fillna(0)
    hourly.loc[hourly["snowfall_rate_mmhr"].isna(), "snowfall_rate_mmhr"] = (
        np.where(temp_c <= 0, precip_mm, np.nan)
    )

    present = hourly["present_weather"].fillna("").str.lower()
    desc = hourly["text_description"].fillna("").str.lower()
    hourly["freezing_rain"] = (
        (temp_c <= 0)
        & (precip_mm > 0)
        & (present.str.contains("fzra") | desc.str.contains("freezing"))
    ).astype(int)

    hourly = hourly.sort_values("hour_bucket").copy()
    for lag in [1, 2, 3]:
        hourly[f"temp_c_lag{lag}"] = hourly["temp_c"].shift(lag)
        hourly[f"snowfall_rate_mmhr_lag{lag}"] = hourly["snowfall_rate_mmhr"].shift(lag)
        hourly[f"wind_speed_mps_lag{lag}"] = hourly["wind_speed_mps"].shift(lag)
        hourly[f"wind_gust_mps_lag{lag}"] = hourly["wind_gust_mps"].shift(lag)

    weather_meta = {
        "station_id": NOAA_STATION_ID,
        "observations": {
            "start": str(hourly["obs_ts"].min()),
            "end": str(hourly["obs_ts"].max()),
            "rows": int(hourly.shape[0]),
        },
    }

    return (
        hourly[
            [
                "hour_bucket",
                "temp_c",
                "wind_speed_mps",
                "wind_gust_mps",
                "snowfall_rate_mmhr",
                "freezing_rain",
                "temp_c_lag1",
                "temp_c_lag2",
                "temp_c_lag3",
                "snowfall_rate_mmhr_lag1",
                "snowfall_rate_mmhr_lag2",
                "snowfall_rate_mmhr_lag3",
                "wind_speed_mps_lag1",
                "wind_speed_mps_lag2",
                "wind_speed_mps_lag3",
                "wind_gust_mps_lag1",
                "wind_gust_mps_lag2",
                "wind_gust_mps_lag3",
            ]
        ],
        weather_meta,
    )


def load_nws_alerts(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    url = "https://api.weather.gov/alerts"
    params = {"point": NWS_POINT, "start": start.isoformat(), "end": end.isoformat()}
    features = _collect_paginated(url, params)
    rows: list[dict[str, Any]] = []
    for feat in features:
        props = feat.get("properties", {}) or {}
        onset = pd.to_datetime(props.get("onset"), utc=True, errors="coerce")
        effective = pd.to_datetime(props.get("effective"), utc=True, errors="coerce")
        ends = pd.to_datetime(props.get("ends"), utc=True, errors="coerce")
        expires = pd.to_datetime(props.get("expires"), utc=True, errors="coerce")
        start_ts = onset if pd.notna(onset) else effective
        end_ts = ends if pd.notna(ends) else expires
        if pd.isna(start_ts) or pd.isna(end_ts):
            continue
        alert_id = props.get("id") or props.get("identifier") or props.get("@id")
        source_url = props.get("@id") or props.get("uri") or props.get("id")
        rows.append(
            {
                "alert_id": alert_id,
                "event": props.get("event"),
                "severity": props.get("severity"),
                "start_ts": start_ts,
                "end_ts": end_ts,
                "source_url": source_url,
            }
        )
    return pd.DataFrame(rows)


def _alert_window(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = df[TS].min().floor("h") - pd.Timedelta(hours=1)
    end = df[TS].max().ceil("h") + pd.Timedelta(hours=1)
    return start, end


def persist_alert_log(
    alerts: pd.DataFrame, path: str = ALERT_LOG_PATH
) -> dict[str, Any]:
    columns = ["alert_id", "event", "start_ts", "end_ts", "severity", "source_url"]
    if alerts.empty and not os.path.exists(path):
        return {"path": path, "rows": 0, "added": 0}

    log_df = alerts.reindex(columns=columns).copy()
    if not log_df.empty:
        log_df["start_ts"] = pd.to_datetime(log_df["start_ts"], utc=True)
        log_df["end_ts"] = pd.to_datetime(log_df["end_ts"], utc=True)

    if os.path.exists(path):
        existing = pd.read_csv(path)
        combined = pd.concat([existing, log_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["alert_id"], keep="last")
        added = len(combined) - len(existing)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        combined = log_df.drop_duplicates(subset=["alert_id"], keep="last")
        added = len(combined)

    if not combined.empty:
        combined = combined.sort_values("start_ts")
    combined.to_csv(path, index=False)
    return {"path": path, "rows": int(len(combined)), "added": int(added)}


def build_alert_features(
    df: pd.DataFrame,
    alerts: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if df.empty:
        return pd.DataFrame(), {"alerts": {}, "point": NWS_POINT}

    start, end = _alert_window(df)
    if alerts is None:
        alerts = load_nws_alerts(start, end)

    if alerts.empty:
        return pd.DataFrame(), {
            "alerts": {"start": str(start), "end": str(end), "rows": 0},
            "point": NWS_POINT,
        }

    hours = pd.date_range(start=start, end=end, freq="h", tz="UTC")
    alert_counts = pd.Series(0, index=hours)

    for row in alerts.itertuples(index=False):
        active_hours = hours[(hours >= row.start_ts) & (hours <= row.end_ts)]
        alert_counts.loc[active_hours] += 1

    alerts_hourly = (
        alert_counts.rename_axis("hour_bucket")
        .reset_index(name="nws_alert_count")
        .copy()
    )
    alerts_hourly["nws_alert_active"] = (alerts_hourly["nws_alert_count"] > 0).astype(
        int
    )

    alert_meta = {
        "point": NWS_POINT,
        "alerts": {
            "start": str(alerts["start_ts"].min()),
            "end": str(alerts["end_ts"].max()),
            "rows": int(alerts.shape[0]),
        },
    }

    return alerts_hourly, alert_meta


# -----------------------------
# Event + label building
# -----------------------------
def build_events(df: pd.DataFrame) -> pd.DataFrame:
    s = df.sort_values([EVENT, SEG, TS]).copy()
    s["prev_last"] = s.groupby([EVENT, SEG])["lastserviced"].shift(1)
    ev = s[
        (s["lastserviced"].notna()) & (s["lastserviced"] != s["prev_last"])
    ][[EVENT, SEG, TS, "lastserviced"]].copy()
    ev = ev.rename(columns={TS: "observed_at", "lastserviced": "serviced_at"})
    return ev.sort_values([EVENT, SEG, "serviced_at"])


def label_next_service(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    svc_times = {
        (e, s): g["serviced_at"].values
        for (e, s), g in events.groupby([EVENT, SEG])
    }

    labeled = df.sort_values([EVENT, SEG, TS]).copy()
    next_times = []
    for row in labeled[[EVENT, SEG, TS]].itertuples(index=False):
        times = svc_times.get((row.eventid, row.snowroutesegmentid))
        if times is None or len(times) == 0:
            next_times.append(pd.NaT)
            continue
        times64 = pd.to_datetime(times, utc=True).to_numpy(dtype="datetime64[ns]")
        idx = np.searchsorted(times64, row.snapshot_ts.to_datetime64(), side="right")
        next_times.append(pd.to_datetime(times64[idx], utc=True) if idx < len(times64) else pd.NaT)

    labeled["next_serviced_at"] = pd.to_datetime(next_times, utc=True)
    labeled["hours_to_next_service"] = (
        labeled["next_serviced_at"] - labeled[TS]
    ).dt.total_seconds() / 3600.0
    return labeled


# -----------------------------
# Feature engineering
# -----------------------------
def add_features(
    df: pd.DataFrame,
    events: pd.DataFrame,
    neighbor_lookup: dict[str, list[str]],
    weather_hourly: pd.DataFrame | None = None,
    alerts_hourly: pd.DataFrame | None = None,
) -> pd.DataFrame:
    f = df.sort_values([EVENT, SEG, TS]).copy()
    for c, default in [
        ("storm_id", "no_storm"),
        ("storm_operational_start", pd.NaT),
        ("storm_operational_end", pd.NaT),
        ("storm_severity_max", "Unknown"),
    ]:
        if c not in f.columns:
            f[c] = default

    f["priority_num"] = f["routepriority"].str.extract(r"(\d+)").astype(float)
    route_label = (
        f["snowrouteid"]
        .fillna("UNK")
        .astype(str)
        .str.strip()
        .replace({"": "UNK", "nan": "UNK", "None": "UNK"})
    )
    route_key = route_label.str.replace(r"\s+", "", regex=True).str.replace(
        r"[^A-Za-z0-9_-]", "", regex=True
    )
    priority_label = f["priority_num"].apply(
        lambda x: f"P{int(x)}" if pd.notna(x) else "PUNK"
    )
    f["route_label"] = route_label
    f["route_key"] = route_key
    f["bucket_priority_id"] = priority_label
    f["bucket_id"] = priority_label + "-R" + route_key
    f["hour"] = f[TS].dt.hour
    f["dow"] = f[TS].dt.weekday
    f["hours_since_last_service"] = (f[TS] - f["lastserviced"]).dt.total_seconds() / 3600

    denom = f["segmentlength"].replace(0, np.nan)
    f["passes_per_len"] = f["passes_event"] / denom

    f["prev_last"] = f.groupby([EVENT, SEG])["lastserviced"].shift(1)
    f["lastserviced_changed"] = (
    f["lastserviced"].notna() & (f["prev_last"].notna()) & (f["lastserviced"] != f["prev_last"])).astype(int)

    f["hour_bucket"] = f[TS].dt.floor("h")
    # phase signal (free operational mode)
    if "eventphaseid" in f.columns:
        phase_num = f["eventphaseid"].astype(str).str.extract(r"-(\d+)$")[0]
        f["phase_num"] = pd.to_numeric(phase_num, errors="coerce").fillna(0)
    else:
        f["phase_num"] = 0

    # storm clock features
    f["in_storm"] = (
        f["storm_operational_start"].notna()
        & f["storm_operational_end"].notna()
        & (f[TS] >= f["storm_operational_start"])
        & (f[TS] <= f["storm_operational_end"])
    ).astype(int)

    f["hours_since_storm_start"] = (
        (f[TS] - f["storm_operational_start"]).dt.total_seconds() / 3600.0
    )
    f["hours_until_storm_end"] = (
        (f["storm_operational_end"] - f[TS]).dt.total_seconds() / 3600.0
    )

    f["hours_since_storm_start"] = f["hours_since_storm_start"].clip(lower=0).fillna(-1)
    f["hours_until_storm_end"] = f["hours_until_storm_end"].clip(lower=0).fillna(-1)

    # finer buckets (5-minute snapshots -> 15m works well)
    f["bucket_15m"] = f[TS].dt.floor("15min")

    # service events derived from lastserviced changes
    svc = f.loc[f["lastserviced_changed"] == 1, [EVENT, "snowrouteid", TS]].copy()
    svc["bucket_15m"] = svc[TS].dt.floor("15min")

    # city tempo (services per 15m)
    city_15m = (
        svc.groupby([EVENT, "bucket_15m"])
        .size()
        .rename("city_services_15m")
        .reset_index()
    )
    f = f.merge(city_15m, on=[EVENT, "bucket_15m"], how="left")
    f["city_services_15m"] = f["city_services_15m"].fillna(0)

    # route tempo (services per 15m)
    route_15m = (
        svc.groupby([EVENT, "snowrouteid", "bucket_15m"])
        .size()
        .rename("route_services_15m")
        .reset_index()
    )
    f = f.merge(route_15m, on=[EVENT, "snowrouteid", "bucket_15m"], how="left")
    f["route_services_15m"] = f["route_services_15m"].fillna(0)

    # route completion: share of segments in route that have had ANY service in this event so far
    f["_route_served_once"] = f.groupby([EVENT, "snowrouteid", SEG])["lastserviced_changed"].cummax()
    route_completion = (
        f.groupby([EVENT, "snowrouteid", "hour_bucket"])["_route_served_once"]
        .mean()
        .rename("route_completion_60m")
        .reset_index()
    )
    f = f.merge(route_completion, on=[EVENT, "snowrouteid", "hour_bucket"], how="left")
    f["route_completion_60m"] = f["route_completion_60m"].fillna(0)
    f = f.drop(columns=["_route_served_once"])

    city = (
        events.assign(hour_bucket=events["serviced_at"].dt.floor("h"))
        .groupby([EVENT, "hour_bucket"])
        .size()
        .rename("city_services_last_hour")
        .reset_index()
    )
    f = f.merge(city, on=[EVENT, "hour_bucket"], how="left")
    f["city_services_last_hour"] = f["city_services_last_hour"].fillna(0)

    route = (
        f.groupby([EVENT, "snowrouteid", "hour_bucket"])["lastserviced_changed"]
        .sum()
        .rename("route_services_last_hour")
        .reset_index()
    )
    f = f.merge(route, on=[EVENT, "snowrouteid", "hour_bucket"], how="left")
    f["route_services_last_hour"] = f["route_services_last_hour"].fillna(0)

    def rolling_sum_by_group(
        data: pd.DataFrame, group_cols: list[str], value_col: str, window: int
    ) -> pd.Series:
        ordered = data.sort_values(group_cols + ["hour_bucket"]).copy()
        rolled = (
            ordered.set_index("hour_bucket")
            .groupby(group_cols, sort=False)[value_col]
            .rolling(f"{window}h", min_periods=1)
            .sum()
            .reset_index(level=group_cols, drop=True)
        )
        rolled.index = ordered.index
        return rolled.reindex(data.index)

    def rolling_delta_ratio_by_group(
        data: pd.DataFrame, group_cols: list[str], value_col: str, window: int, prefix: str
    ) -> pd.DataFrame:
        ordered = data.sort_values(group_cols + ["hour_bucket"]).copy()
        series = (
            ordered.set_index("hour_bucket")
            .groupby(group_cols, sort=False)[value_col]
        )
        rolling_obj = series.rolling(f"{window}h", min_periods=1)
        rolling_min = rolling_obj.min().reset_index(level=group_cols, drop=True)
        rolling_max = rolling_obj.max().reset_index(level=group_cols, drop=True)
        delta = (rolling_max - rolling_min).to_numpy()
        ratio = (rolling_max / rolling_min.replace(0, np.nan)).to_numpy()
        out = pd.DataFrame(
            {
                f"{prefix}_delta_{window}h": delta,
                f"{prefix}_ratio_{window}h": ratio,
            },
            index=ordered.index,
        )
        return out.reindex(data.index)

    def group_apply_no_groups(df: pd.DataFrame, by: list[str], func, **kwargs):
        grouped = df.groupby(by, group_keys=False)
        try:
            return grouped.apply(func, include_groups=False, **kwargs)
        except TypeError:
            return grouped.apply(func, **kwargs)

    for window in [3, 6]:
        f[f"seg_services_{window}h"] = rolling_sum_by_group(
            f, [EVENT, SEG], "lastserviced_changed", window
        ).fillna(0)
        f[f"route_services_{window}h"] = rolling_sum_by_group(
            f, [EVENT, "snowrouteid"], "lastserviced_changed", window
        ).fillna(0)

        seg_roll = rolling_delta_ratio_by_group(
            f, [EVENT, SEG], "passes_event", window, "passes_event"
        ).fillna(0)
        f = f.join(seg_roll)

        route_roll = rolling_delta_ratio_by_group(
            f, [EVENT, "snowrouteid"], "passes_event", window, "route_passes_event"
        ).fillna(0)
        f = f.join(route_roll)

    if neighbor_lookup:
        seg_hour = (
            f.groupby([EVENT, SEG, "hour_bucket"])["lastserviced_changed"]
            .sum()
            .rename("seg_services_last_hour")
        )

        def neighbor_sum(row: pd.Series) -> int:
            return sum(
                seg_hour.get((row[EVENT], n, row["hour_bucket"]), 0)
                for n in neighbor_lookup.get(str(row[SEG]), [])
            )

        f["neighbor_services_last_hour"] = f.apply(neighbor_sum, axis=1)
    else:
        f["neighbor_services_last_hour"] = 0

    if weather_hourly is not None and not weather_hourly.empty:
        f = f.merge(weather_hourly, on="hour_bucket", how="left")
    else:
        f["temp_c"] = np.nan
        f["wind_speed_mps"] = np.nan
        f["wind_gust_mps"] = np.nan
        f["snowfall_rate_mmhr"] = np.nan
        f["freezing_rain"] = 0
        for lag in [1, 2, 3]:
            f[f"temp_c_lag{lag}"] = np.nan
            f[f"snowfall_rate_mmhr_lag{lag}"] = np.nan
            f[f"wind_speed_mps_lag{lag}"] = np.nan
            f[f"wind_gust_mps_lag{lag}"] = np.nan

    if alerts_hourly is not None and not alerts_hourly.empty:
        f = f.merge(alerts_hourly, on="hour_bucket", how="left")
    else:
        f["nws_alert_count"] = 0
        f["nws_alert_active"] = 0

    return f


# -----------------------------
# Bucket baseline model
# -----------------------------
def _build_bucket_ladder(
    bucket_starts: pd.DataFrame,
    bucket_col: str,
) -> pd.DataFrame:
    if bucket_starts.empty:
        return pd.DataFrame(
            columns=[
                "bucket_from",
                "bucket_to",
                "delta_hours_median",
                "delta_hours_p10",
                "delta_hours_p90",
                "n_events",
                "n_pairs",
            ]
        )

    pairs = bucket_starts.merge(bucket_starts, on=EVENT, suffixes=("_from", "_to"))
    pairs["delta_hours"] = (
        pairs[f"{TS}_to"] - pairs[f"{TS}_from"]
    ).dt.total_seconds() / 3600.0
    pairs = pairs[pairs["delta_hours"].notna()]
    pairs = pairs[pairs["delta_hours"] >= 0]

    summary = (
        pairs.groupby([f"{bucket_col}_from", f"{bucket_col}_to"])
        .agg(
            delta_hours_median=("delta_hours", "median"),
            delta_hours_p10=("delta_hours", lambda x: x.quantile(0.1)),
            delta_hours_p90=("delta_hours", lambda x: x.quantile(0.9)),
            n_events=(EVENT, "nunique"),
            n_pairs=("delta_hours", "size"),
        )
        .reset_index()
        .rename(
            columns={
                f"{bucket_col}_from": "bucket_from",
                f"{bucket_col}_to": "bucket_to",
            }
        )
    )
    return summary


def build_bucket_ladders(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        empty = pd.DataFrame(
            columns=[
                "bucket_from",
                "bucket_to",
                "delta_hours_median",
                "delta_hours_p10",
                "delta_hours_p90",
                "n_events",
                "n_pairs",
            ]
        )
        return empty.copy(), empty.copy()

    service_events = df.loc[df["lastserviced_changed"] == 1, [EVENT, TS, "bucket_id", "bucket_priority_id"]].copy()
    service_events = service_events.dropna(subset=[EVENT, TS])
    service_events["bucket_id"] = service_events["bucket_id"].fillna("PUNK-RUNK")
    service_events["bucket_priority_id"] = service_events["bucket_priority_id"].fillna("PUNK")

    bucket_starts = (
        service_events.groupby([EVENT, "bucket_id"])[TS]
        .min()
        .reset_index()
    )
    priority_starts = (
        service_events.groupby([EVENT, "bucket_priority_id"])[TS]
        .min()
        .reset_index()
    )

    bucket_ladder = _build_bucket_ladder(bucket_starts, "bucket_id")
    priority_ladder = _build_bucket_ladder(priority_starts, "bucket_priority_id")
    return bucket_ladder, priority_ladder


def _bucket_lookup_map(ladder: pd.DataFrame) -> dict[tuple[str, str], float]:
    return {
        (row.bucket_from, row.bucket_to): row.delta_hours_median
        for row in ladder.itertuples(index=False)
    }


def _derive_current_bucket(
    service_events: pd.DataFrame,
    latest_event_times: pd.DataFrame,
    bucket_col: str,
    window: str,
) -> dict[str, str | None]:
    current = {}
    for row in latest_event_times.itertuples(index=False):
        event_id = row.eventid
        now_ts = row.snapshot_ts
        window_start = now_ts - pd.Timedelta(window)
        mask = (
            (service_events[EVENT] == event_id)
            & (service_events[TS] >= window_start)
            & (service_events[TS] <= now_ts)
        )
        counts = service_events.loc[mask].groupby(bucket_col).size()
        current[event_id] = counts.idxmax() if not counts.empty else None
    return current


def _sigmoid(x: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def attach_bucket_baseline(
    latest: pd.DataFrame,
    full_df: pd.DataFrame,
    bucket_ladder: pd.DataFrame,
    priority_ladder: pd.DataFrame,
) -> pd.DataFrame:
    if latest.empty:
        return latest

    service_events = full_df.loc[
        full_df["lastserviced_changed"] == 1,
        [EVENT, TS, "bucket_id", "bucket_priority_id"],
    ].copy()
    service_events = service_events.dropna(subset=[EVENT, TS])
    if service_events.empty:
        for h in HORIZONS:
            latest[f"p_bucket_{h}h"] = np.nan
        latest["current_bucket_id"] = None
        latest["current_bucket_priority"] = None
        latest["eta_bucket_hours"] = np.nan
        return latest

    latest_event_times = (
        latest.groupby(EVENT, as_index=False)[TS].max().rename(columns={TS: "snapshot_ts"})
    )
    current_bucket_id = _derive_current_bucket(
        service_events,
        latest_event_times,
        "bucket_id",
        BUCKET_CURRENT_WINDOW,
    )
    current_bucket_priority = _derive_current_bucket(
        service_events,
        latest_event_times,
        "bucket_priority_id",
        BUCKET_CURRENT_WINDOW,
    )

    bucket_starts = (
        service_events.groupby([EVENT, "bucket_id"])[TS]
        .min()
        .to_dict()
    )
    priority_starts = (
        service_events.groupby([EVENT, "bucket_priority_id"])[TS]
        .min()
        .to_dict()
    )

    bucket_lookup = _bucket_lookup_map(bucket_ladder)
    priority_lookup = _bucket_lookup_map(priority_ladder)

    def compute_eta(row: pd.Series) -> float:
        event_id = row[EVENT]
        now_ts = row[TS]
        current_bucket = current_bucket_id.get(event_id)
        current_priority = current_bucket_priority.get(event_id)

        delta = None
        if current_bucket is not None:
            delta = bucket_lookup.get((current_bucket, row["bucket_id"]))

        if delta is None and current_priority is not None:
            delta = priority_lookup.get((current_priority, row["bucket_priority_id"]))

        if delta is None:
            return np.nan

        if current_bucket is not None:
            current_start = bucket_starts.get((event_id, current_bucket))
        else:
            current_start = None

        if current_start is None and current_priority is not None:
            current_start = priority_starts.get((event_id, current_priority))

        if current_start is None:
            return np.nan

        hours_since = (now_ts - current_start).total_seconds() / 3600.0
        hours_since = max(0.0, hours_since)
        return max(0.0, delta - hours_since)

    latest["current_bucket_id"] = latest[EVENT].map(current_bucket_id)
    latest["current_bucket_priority"] = latest[EVENT].map(current_bucket_priority)
    latest["eta_bucket_hours"] = latest.apply(compute_eta, axis=1)

    for h in HORIZONS:
        latest[f"p_bucket_{h}h"] = _sigmoid(
            (h - latest["eta_bucket_hours"]) / BUCKET_SOFTNESS_HOURS
        )

    return latest


def compute_blend_weight(row: pd.Series) -> float:
    if row.get("in_storm", 0) == 1 and row.get("city_services_15m", 0) >= BUCKET_BLEND_MIN_CITY_15M:
        return BUCKET_BLEND_WEIGHT_HIGH
    if row.get("in_storm", 0) == 1 and row.get("city_services_15m", 0) >= BUCKET_BLEND_LOW_CITY_15M:
        return BUCKET_BLEND_WEIGHT_MID
    return BUCKET_BLEND_WEIGHT_LOW


def mark_untracked(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    growth = out.groupby([EVENT, SEG])["passes_event"].agg(["min", "max"]).reset_index()
    growth["growth"] = growth["max"] - growth["min"]
    out = out.merge(growth[[EVENT, SEG, "growth"]], on=[EVENT, SEG], how="left")

    out["prediction_status"] = "OK"
    out.loc[
        (out["growth"] == 0) | (out["snowrouteid"].str.lower() == "unknown"),
        "prediction_status",
    ] = "NO_PRED_UNTRACKED"
    return out


def add_horizon_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for h in HORIZONS:
        out[f"y_{h}h"] = (
            out["hours_to_next_service"].notna()
            & (out["hours_to_next_service"] <= h)
        ).astype(int)
    return out


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = bin_ids == i
        if not np.any(mask):
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += np.abs(bin_acc - bin_conf) * (mask.sum() / len(y_true))
    return float(ece)

def ranking_metrics_at_k(
    y_true: np.ndarray, y_prob: np.ndarray, ks: list[int]
) -> list[dict[str, float | int | None]]:
    mask = np.isfinite(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    if y_true.size == 0:
        return [
            {"k": int(k), "precision": None, "recall": None, "total_positives": 0}
            for k in ks
        ]

    order = np.argsort(-y_prob)
    total_pos = int(y_true.sum())
    results = []
    for k in ks:
        k_use = int(min(k, y_true.size))
        top_idx = order[:k_use]
        hits = int(y_true[top_idx].sum())
        precision = hits / k_use if k_use > 0 else None
        recall = hits / total_pos if total_pos > 0 else None
        results.append(
            {
                "k": int(k),
                "precision": float(precision) if precision is not None else None,
                "recall": float(recall) if recall is not None else None,
                "total_positives": total_pos,
            }
        )
    return results


def derive_eta_from_probs(prob_df: pd.DataFrame, thresholds: dict[int, float] | None = None) -> pd.Series:
    thr = thresholds or {}
    def eta(row: pd.Series) -> float:
        for h in HORIZONS:
            t = float(thr.get(h, ETA_THRESHOLD_DEFAULT))
            if float(row.get(f"p_{h}h", np.nan)) >= t:
                return float(h)
        return np.nan
    return prob_df.apply(eta, axis=1)



def train_eta_regression(
    df: pd.DataFrame,
    X: pd.DataFrame,
    train_mask: pd.Series,
    models: dict[int, HistGradientBoostingClassifier | None],
    calibrated: dict[int, CalibratedClassifierCV | None],
    thresholds: dict[int, float] | None = None,
) -> tuple[HistGradientBoostingRegressor | None, dict[str, Any]]:
    valid = df["hours_to_next_service"].notna()
    train_idx = train_mask & valid
    test_idx = (~train_mask) & valid

    metrics: dict[str, Any] = {
        "rows_train": int(train_idx.sum()),
        "rows_test": int(test_idx.sum()),
        "mae": None,
        "rmse": None,
        "eta_class_mae": None,
        "eta_class_rmse": None,
    }

    if train_idx.sum() < 10 or test_idx.sum() == 0:
        metrics["note"] = "insufficient training data"
        return None, metrics

    reg = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.08,
        max_iter=350,
        random_state=42,
    )
    reg.fit(X[train_idx], df.loc[train_idx, "hours_to_next_service"])

    preds = reg.predict(X[test_idx])
    y_true = df.loc[test_idx, "hours_to_next_service"]
    metrics["mae"] = float(mean_absolute_error(y_true, preds))
    if hasattr(sk_metrics, "root_mean_squared_error"):
        metrics["rmse"] = float(sk_metrics.root_mean_squared_error(y_true, preds))
    else:
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, preds)))

    prob_df = pd.DataFrame(index=df.loc[test_idx].index)
    for h in HORIZONS:
        model = calibrated.get(h) or models.get(h)
        if model is None:
            prob_df[f"p_{h}h"] = np.nan
        else:
            prob_df[f"p_{h}h"] = model.predict_proba(X[test_idx])[:, 1]
    eta_series = derive_eta_from_probs(prob_df, thresholds)
    valid_eta = eta_series.notna()
    if valid_eta.any():
        eta_true = y_true[valid_eta]
        eta_pred = eta_series[valid_eta]
        metrics["eta_class_mae"] = float(mean_absolute_error(eta_true, eta_pred))
        if hasattr(sk_metrics, "root_mean_squared_error"):
            metrics["eta_class_rmse"] = float(
                sk_metrics.root_mean_squared_error(eta_true, eta_pred)
            )
        else:
            metrics["eta_class_rmse"] = float(
                np.sqrt(mean_squared_error(eta_true, eta_pred))
            )

    return reg, metrics


# -----------------------------
# Train models
# -----------------------------
def train_models(
    df: pd.DataFrame, features: list[str]
) -> tuple[
    dict[int, HistGradientBoostingClassifier | None],
    dict[int, CalibratedClassifierCV | None],
    dict[str, dict[int, dict]],
    pd.Series,
    HistGradientBoostingRegressor | None,
    dict[int, list[dict[str, float]]],
    dict[int, float],
]:
    if df.empty:
        return {}, {}, {}, pd.Series(dtype=float), None, {}, {}

    train_cutoffs = df.groupby(EVENT)[TS].transform(lambda s: s.quantile(0.8))
    train_mask = df[TS] <= train_cutoffs

    X = df[features].replace([np.inf, -np.inf], np.nan)
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)

    models: dict[int, HistGradientBoostingClassifier | None] = {}
    calibrated: dict[int, CalibratedClassifierCV | None] = {}
    metrics: dict[str, dict[int, dict]] = {
        "within_event_time": {},
        "groupkfold_event": {},
        "calibration": {},
        "ranking": {},
    }
    feature_importance: dict[int, list[dict[str, float]]] = {}
    threshold_values: dict[int, float] = {}
    threshold_meta: dict[int, dict[str, float | int | None]] = {}

    calib_cutoffs = df.loc[train_mask].groupby(EVENT)[TS].transform(
        lambda s: s.quantile(0.8)
    )
    calib_mask = pd.Series(False, index=df.index)
    calib_mask.loc[train_mask] = df.loc[train_mask, TS] > calib_cutoffs
    base_train_mask = train_mask & ~calib_mask

    for h in HORIZONS:
        y = df[f"y_{h}h"].astype(int)
        X_train, y_train = X[base_train_mask], y[base_train_mask]
        X_calib, y_calib = X[calib_mask], y[calib_mask]
        X_test, y_test = X[~train_mask], y[~train_mask]

        if X_train.empty or y_train.nunique() < 2:
            models[h] = None
            calibrated[h] = None
            metrics["within_event_time"][h] = {
                "rows_train": int(X_train.shape[0]),
                "rows_test": int(X_test.shape[0]),
                "auc": None,
                "brier": None,
                "pr_auc": None,
                "event_count": int(df[EVENT].nunique()),
                "note": "insufficient training data",
            }
            metrics["calibration"][h] = {
                "rows": int(X_calib.shape[0]),
                "brier": None,
                "ece": None,
            }
            continue

        pos = y_train.sum()
        neg = len(y_train) - pos
        pos_rate = float(pos / len(y_train)) if len(y_train) else 0.0
        if pos > 0 and neg > 0 and (pos_rate < 0.2 or pos_rate > 0.8):
            pos_weight = neg / pos
            sample_weight = np.where(y_train == 1, pos_weight, 1.0)
            class_strategy = "weighted"
        else:
            sample_weight = None
            class_strategy = "none"

        clf = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.08,
            max_iter=350,
            random_state=42,
        )
        clf.fit(X_train, y_train, sample_weight=sample_weight)
        models[h] = clf

        if not X_calib.empty and y_calib.nunique() >= 2:
            if len(X_calib) >= CALIBRATION_MIN_ROWS_ISOTONIC:
                method = "isotonic"
            else:
                method = "sigmoid"
            calibrated[h] = _fit_prefit_calibrator(clf, X_calib, y_calib, method)
        else:
            calibrated[h] = None

        if X_test.empty:
            auc = None
            brier = None
            pr_auc = None
            p = None
        else:
            model_for_eval = calibrated[h] or clf
            p = model_for_eval.predict_proba(X_test)[:, 1]
            if y_test.nunique() < 2:
                auc = None
                brier = None
                pr_auc = None
            else:
                auc = float(roc_auc_score(y_test, p))
                brier = float(brier_score_loss(y_test, p))
                pr_auc = float(average_precision_score(y_test, p))

        metrics["within_event_time"][h] = {
            "rows_train": int(X_train.shape[0]),
            "rows_test": int(X_test.shape[0]),
            "auc": auc,
            "brier": brier,
            "pr_auc": pr_auc,
            "event_count": int(df[EVENT].nunique()),
            "train_ratio": float(train_mask.mean()),
            "pos_rate_train": pos_rate,
            "class_strategy": class_strategy,
        }

        if p is None or X_test.empty:
            metrics["ranking"][h] = {
                "rows_test": int(X_test.shape[0]),
                "overall": [],
                "priority_1": [],
                "other": [],
            }
        else:
            test_meta = df.loc[X_test.index]
            priority_1_mask = test_meta["priority_num"].fillna(-1) == 1
            metrics["ranking"][h] = {
                "rows_test": int(X_test.shape[0]),
                "overall": ranking_metrics_at_k(
                    y_test.to_numpy(), p, RANKING_KS
                ),
                "priority_1": ranking_metrics_at_k(
                    y_test[priority_1_mask].to_numpy(),
                    p[priority_1_mask.to_numpy()],
                    RANKING_KS,
                ),
                "other": ranking_metrics_at_k(
                    y_test[~priority_1_mask].to_numpy(),
                    p[~priority_1_mask.to_numpy()],
                    RANKING_KS,
                ),
            }

        if calibrated[h] is not None and not X_calib.empty and y_calib.nunique() >= 2:
            p_calib = calibrated[h].predict_proba(X_calib)[:, 1]
            calibration_method = calibrated[h].method
        elif not X_calib.empty and y_calib.nunique() >= 2:
            p_calib = clf.predict_proba(X_calib)[:, 1]

        if p_calib is not None:
            metrics["calibration"][h] = {
                "rows": int(X_calib.shape[0]),
                "brier": float(brier_score_loss(y_calib, p_calib)),
                "ece": expected_calibration_error(y_calib.to_numpy(), p_calib),
                "method": calibration_method,
            }
        else:
            metrics["calibration"][h] = {
                "rows": int(X_calib.shape[0]),
                "brier": None,
                "ece": None,
                "method": None,
            }

        if p_calib is not None and y_calib.nunique() >= 2:
            candidate_thresholds = np.unique(
                np.clip(
                    np.quantile(p_calib, np.linspace(0.05, 0.95, 19)),
                    0.05,
                    0.95,
                )
            )
            if candidate_thresholds.size == 0:
                candidate_thresholds = np.array([ETA_THRESHOLD_DEFAULT])

            f1_scores = []
            for threshold in candidate_thresholds:
                preds = (p_calib >= threshold).astype(int)
                f1_scores.append(sk_metrics.f1_score(y_calib, preds))
            best_idx = int(np.argmax(f1_scores))
            best_threshold = float(candidate_thresholds[best_idx])
            threshold_values[h] = best_threshold
            threshold_meta[h] = {
                "threshold": best_threshold,
                "f1": float(f1_scores[best_idx]),
                "rows": int(X_calib.shape[0]),
                "pos_rate": float(y_calib.mean()),
                "metric": "f1",
            }
        else:
            threshold_values[h] = ETA_THRESHOLD_DEFAULT
            threshold_meta[h] = {
                "threshold": ETA_THRESHOLD_DEFAULT,
                "f1": None,
                "rows": int(X_calib.shape[0]),
                "pos_rate": float(y_calib.mean()) if len(y_calib) else None,
                "metric": "f1",
            }

        if not X_test.empty and y_test.nunique() >= 2:
            perm = permutation_importance(
                clf,
                X_test,
                y_test,
                n_repeats=10,
                random_state=42,
                scoring="roc_auc",
            )
            ranked = sorted(
                [
                    {
                        "feature": features[i],
                        "importance_mean": float(perm.importances_mean[i]),
                        "importance_std": float(perm.importances_std[i]),
                    }
                    for i in range(len(features))
                ],
                key=lambda x: x["importance_mean"],
                reverse=True,
            )
            feature_importance[h] = ranked

    n_events = int(df[EVENT].nunique())
    if n_events < 2:
        for h in HORIZONS:
            metrics["groupkfold_event"][h] = {
                "rows_train": int(df.shape[0]),
                "rows_test": 0,
                "auc": None,
                "brier": None,
                "pr_auc": None,
                "event_count": n_events,
                "folds": 0,
                "note": "insufficient training data",
            }
        reg_model, reg_metrics = train_eta_regression(
            df, X, train_mask, models, calibrated, thresholds=threshold_values
        )
        metrics["eta_regression"] = reg_metrics
        metrics["thresholds"] = threshold_meta
        return (
            models,
            calibrated,
            metrics,
            medians,
            reg_model,
            feature_importance,
            threshold_values,
        )

    n_splits = min(5, n_events)
    gkf = GroupKFold(n_splits=n_splits)
    groups = df[EVENT]

    for h in HORIZONS:
        y = df[f"y_{h}h"].astype(int)
        aucs: list[float] = []
        briers: list[float] = []
        pr_aucs: list[float] = []
        rows_train = 0
        rows_test = 0
        folds = 0

        for train_idx, test_idx in gkf.split(X, y, groups):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            rows_train += int(X_train.shape[0])
            rows_test += int(X_test.shape[0])

            if X_train.empty or y_train.nunique() < 2 or y_test.nunique() < 2:
                continue

            pos = y_train.sum()
            neg = len(y_train) - pos
            pos_rate = float(pos / len(y_train)) if len(y_train) else 0.0
            if pos > 0 and neg > 0 and (pos_rate < 0.2 or pos_rate > 0.8):
                pos_weight = neg / pos
                sample_weight = np.where(y_train == 1, pos_weight, 1.0)
            else:
                sample_weight = None

            clf = HistGradientBoostingClassifier(
                max_depth=6,
                learning_rate=0.08,
                max_iter=350,
                random_state=42,
            )
            clf.fit(X_train, y_train, sample_weight=sample_weight)
            p = clf.predict_proba(X_test)[:, 1]
            aucs.append(float(roc_auc_score(y_test, p)))
            briers.append(float(brier_score_loss(y_test, p)))
            pr_aucs.append(float(average_precision_score(y_test, p)))
            folds += 1

        metrics["groupkfold_event"][h] = {
            "rows_train": rows_train,
            "rows_test": rows_test,
            "auc": float(np.mean(aucs)) if aucs else None,
            "brier": float(np.mean(briers)) if briers else None,
            "pr_auc": float(np.mean(pr_aucs)) if pr_aucs else None,
            "event_count": n_events,
            "folds": folds,
        }

    reg_model, reg_metrics = train_eta_regression(
        df, X, train_mask, models, calibrated, thresholds=threshold_values
    )
    metrics["eta_regression"] = reg_metrics
    metrics["thresholds"] = threshold_meta

    return (
        models,
        calibrated,
        metrics,
        medians,
        reg_model,
        feature_importance,
        threshold_values,
    )


# -----------------------------
# Predict latest per segment
# -----------------------------
def predict_latest(
    df: pd.DataFrame,
    models: dict[int, HistGradientBoostingClassifier | None],
    calibrated: dict[int, CalibratedClassifierCV | None],
    medians: pd.Series,
    features: list[str],
    reg_model: HistGradientBoostingRegressor | None,
    bucket_ladder: pd.DataFrame | None = None,
    priority_ladder: pd.DataFrame | None = None,
    thresholds: dict[int, float] | None = None,
) -> pd.DataFrame:
    latest = (
        df.sort_values([EVENT, SEG, TS])
        .groupby([EVENT, SEG], as_index=False)
        .tail(1)
        .copy()
    )

    if latest.empty:
        return latest

    X_now = latest[features].replace([np.inf, -np.inf], np.nan)
    if not medians.empty:
        X_now = X_now.fillna(medians)
    else:
        X_now = X_now.fillna(0)

    for h in HORIZONS:
        model = calibrated.get(h) or models.get(h)
        if model is None:
            latest[f"p_ml_{h}h"] = np.nan
        else:
            latest[f"p_ml_{h}h"] = model.predict_proba(X_now)[:, 1]

    bucket_ladder = bucket_ladder if bucket_ladder is not None else pd.DataFrame()
    priority_ladder = priority_ladder if priority_ladder is not None else pd.DataFrame()
    latest = attach_bucket_baseline(latest, df, bucket_ladder, priority_ladder)
    latest["p_blend_weight"] = latest.apply(compute_blend_weight, axis=1)

    for h in HORIZONS:
        p_ml = latest[f"p_ml_{h}h"]
        p_bucket = latest[f"p_bucket_{h}h"]
        blended = (
            latest["p_blend_weight"] * p_ml
            + (1 - latest["p_blend_weight"]) * p_bucket
        )
        latest[f"p_{h}h"] = np.where(
            p_ml.isna() & p_bucket.isna(),
            np.nan,
            np.where(p_ml.isna(), p_bucket, np.where(p_bucket.isna(), p_ml, blended)),
        )

    latest["p_2h"] = latest[["p_1h", "p_2h"]].max(axis=1)
    latest["p_4h"] = latest[["p_2h", "p_4h"]].max(axis=1)
    latest["p_8h"] = latest[["p_4h", "p_8h"]].max(axis=1)

    latest["eta_hours_60"] = derive_eta_from_probs(latest, thresholds)
    latest["eta_ts_60"] = latest[TS] + pd.to_timedelta(latest["eta_hours_60"], unit="h")

    if reg_model is not None:
        latest["eta_hours_pred"] = reg_model.predict(X_now)
    else:
        latest["eta_hours_pred"] = np.nan

    for h in HORIZONS:
        latest.loc[latest["prediction_status"] != "OK", f"p_{h}h"] = np.nan
        latest.loc[latest["prediction_status"] != "OK", f"p_ml_{h}h"] = np.nan
    latest.loc[
        latest["prediction_status"] != "OK",
        ["eta_hours_60", "eta_ts_60", "eta_hours_pred"],
    ] = np.nan

    return latest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train snowplow service prediction models from snapshots."
    )
    parser.add_argument(
        "--snapshot-source",
        default=None,
        help=(
            "Optional CSV path/URL override for snapshots. Defaults to the local "
            f"file ({SNAPSHOT_PATH})."
        ),
    )
    parser.add_argument(
        "--use-remote",
        action="store_true",
        help="Use the remote snapshot URL instead of the local file.",
    )
    args = parser.parse_args()

    env_source = os.getenv("SNOW_SNAPSHOT_SOURCE")
    snapshot_source = (
        args.snapshot_source
        or env_source
        or (REMOTE_SNAPSHOT_URL if args.use_remote else SNAPSHOT_PATH)
    )

    snapshots = load_snapshots(snapshot_source)
    if snapshots.empty:
        print("No snapshots loaded; exiting without training.")
        return

    geo_df = load_geojson_features(GEOJSON_PATH)
    if "passes_phase" in snapshots.columns:
        snapshots["passes_event"] = snapshots["passes_phase"].fillna(
            snapshots["passes"]
        )
    else:
        snapshots["passes_event"] = snapshots["passes"]

    centroids = geo_df.dropna(subset=["lat", "lon"]).drop_duplicates(SEG)
    neighbor_lookup = build_neighbor_lookup(centroids, k=12)

    events = build_events(snapshots)
    labeled = label_next_service(snapshots, events)

    # --- NEW: storms from alerts + operational activity ---
    alerts_log = load_alert_log()  # local if present, else GitHub raw
    storms_env = build_storm_envelopes_from_alerts(alerts_log)

    service_events = derive_city_service_events(labeled)  # based on lastserviced changes
    storms_ops = refine_operational_windows(storms_env, service_events)
    labeled = attach_storm_context(labeled, storms_ops)

    # optional debug once
    print("storm cols present?", {"storm_operational_start","storm_operational_end"}.issubset(labeled.columns))

    weather_hourly, weather_meta = build_weather_features(labeled)
    alert_start, alert_end = _alert_window(labeled)
    alert_rows = load_nws_alerts(alert_start, alert_end)
    alerts_log_meta = persist_alert_log(alert_rows)
    alerts_hourly, alerts_meta = build_alert_features(labeled, alert_rows)
    featured = add_features(
        labeled, events, neighbor_lookup, weather_hourly, alerts_hourly
    )
    bucket_ladder, priority_ladder = build_bucket_ladders(featured)
    featured = mark_untracked(featured)
    featured = add_horizon_labels(featured)

    feature_cols = [
        "priority_num",
        "passes_event",
        "passes_per_len",
        "hours_since_last_service",
        "hour",
        "dow",
        "city_services_last_hour",
        "route_services_last_hour",
        "seg_services_3h",
        "seg_services_6h",
        "route_services_3h",
        "route_services_6h",
        "passes_event_delta_3h",
        "passes_event_ratio_3h",
        "passes_event_delta_6h",
        "passes_event_ratio_6h",
        "route_passes_event_delta_3h",
        "route_passes_event_ratio_3h",
        "route_passes_event_delta_6h",
        "route_passes_event_ratio_6h",
        "neighbor_services_last_hour",
        "temp_c",
        "wind_speed_mps",
        "wind_gust_mps",
        "snowfall_rate_mmhr",
        "freezing_rain",
        "temp_c_lag1",
        "temp_c_lag2",
        "temp_c_lag3",
        "snowfall_rate_mmhr_lag1",
        "snowfall_rate_mmhr_lag2",
        "snowfall_rate_mmhr_lag3",
        "wind_speed_mps_lag1",
        "wind_speed_mps_lag2",
        "wind_speed_mps_lag3",
        "wind_gust_mps_lag1",
        "wind_gust_mps_lag2",
        "wind_gust_mps_lag3",
        "nws_alert_count",
        "nws_alert_active",
        "phase_num",
        "in_storm",
        "hours_since_storm_start",
        "hours_until_storm_end",
        "city_services_15m",
        "route_services_15m",
        "route_completion_60m",

    ]

    train_df = featured[featured["prediction_status"] == "OK"].copy()
    (
        models,
        calibrated,
        metrics,
        medians,
        reg_model,
        feature_importance,
        threshold_values,
    ) = train_models(train_df, feature_cols)

    pred_latest = predict_latest(
        featured,
        models,
        calibrated,
        medians,
        feature_cols,
        reg_model,
        bucket_ladder=bucket_ladder,
        priority_ladder=priority_ladder,
        thresholds=threshold_values,
    )
    pred_path = ARTIFACT_DIR / "predictions_latest_prob.csv"
    pred_latest.to_csv(pred_path, index=False)

    bucket_ladder.to_csv(BUCKET_LADDER_PATH, index=False)
    priority_ladder.to_csv(BUCKET_LADDER_PRIORITY_PATH, index=False)

    metrics_path = ARTIFACT_DIR / "model_metrics_prob.json"
    metrics["feature_importance_path"] = os.path.join(
        ARTIFACT_DIR / "feature_importance.json"
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    feature_path = ARTIFACT_DIR / "feature_importance.json"
    with open(feature_path, "w", encoding="utf-8") as f:
        json.dump(feature_importance, f, indent=2)

    weather_path = ARTIFACT_DIR / "weather_data_sources.json"
    with open(weather_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "noaa_observations": weather_meta,
                "nws_alerts": alerts_meta,
                "nws_alerts_log": alerts_log_meta,
            },
            f,
            indent=2,
        )

    print(f"Saved: {pred_path} | rows: {len(pred_latest)}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {weather_path}")

    if not pred_latest.empty:
        untracked_share = (pred_latest["prediction_status"] != "OK").mean()
        print(f"Untracked share: {untracked_share:.3f}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
