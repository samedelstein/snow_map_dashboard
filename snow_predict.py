"""
Snowplow probability prediction model (robust, geometry optional).
Uses local snapshot + GeoJSON files by default, with optional CLI/env overrides
for remote snapshot training.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
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
SNAPSHOT_PATH = "snapshot_snow_routes/snapshots.csv"
REMOTE_SNAPSHOT_URL = (
    "https://raw.githubusercontent.com/samedelstein/snow_map_dashboard/"
    "refs/heads/main/snapshot_snow_routes/snapshots.csv"
)
GEOJSON_PATH = "winter_operations_snow_routes_layer0.geojson"

EVENT = "eventid"
SEG = "snowroutesegmentid"
TS = "snapshot_ts"

HORIZONS = [1, 2, 4, 8]
ETA_THRESHOLD = 0.60
ARTIFACT_DIR = "artifacts_snow"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

NOAA_STATION_ID = "KSYR"
NWS_POINT = "43.0481,-76.1474"
NWS_USER_AGENT = "snow_map_dashboard (https://github.com/samedelstein/snow_map_dashboard)"


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
        rows.append(
            {
                "alert_id": props.get("id") or props.get("@id"),
                "event": props.get("event"),
                "severity": props.get("severity"),
                "start_ts": start_ts,
                "end_ts": end_ts,
            }
        )
    return pd.DataFrame(rows)


def build_alert_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if df.empty:
        return pd.DataFrame(), {"alerts": {}, "point": NWS_POINT}

    start = df[TS].min().floor("h") - pd.Timedelta(hours=1)
    end = df[TS].max().ceil("h") + pd.Timedelta(hours=1)
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
        idx = np.searchsorted(times, row.snapshot_ts.to_datetime64(), side="right")
        next_times.append(times[idx] if idx < len(times) else pd.NaT)

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

    f["priority_num"] = f["routepriority"].str.extract(r"(\d+)").astype(float)
    f["hour"] = f[TS].dt.hour
    f["dow"] = f[TS].dt.weekday
    f["hours_since_last_service"] = (f[TS] - f["lastserviced"]).dt.total_seconds() / 3600

    denom = f["segmentlength"].replace(0, np.nan)
    f["passes_per_len"] = f["passes_event"] / denom

    f["prev_last"] = f.groupby([EVENT, SEG])["lastserviced"].shift(1)
    f["lastserviced_changed"] = (f["lastserviced"] != f["prev_last"]).astype(int)
    f["hour_bucket"] = f[TS].dt.floor("h")

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

    def rolling_sum(group: pd.DataFrame, value_col: str, window: int) -> pd.Series:
        g = group.sort_values("hour_bucket")
        series = g.set_index("hour_bucket")[value_col]
        rolled = series.rolling(f"{window}h", min_periods=1).sum()
        rolled.index = g.index
        return rolled

    def rolling_delta_ratio(
        group: pd.DataFrame, value_col: str, window: int, prefix: str
    ) -> pd.DataFrame:
        g = group.sort_values("hour_bucket")
        series = g.set_index("hour_bucket")[value_col]
        rolling_min = series.rolling(f"{window}h", min_periods=1).min()
        rolling_max = series.rolling(f"{window}h", min_periods=1).max()
        delta = (rolling_max - rolling_min).to_numpy()
        ratio = (rolling_max / rolling_min.replace(0, np.nan)).to_numpy()
        return pd.DataFrame(
            {
                f"{prefix}_delta_{window}h": delta,
                f"{prefix}_ratio_{window}h": ratio,
            },
            index=g.index,
        )

    for window in [3, 6]:
        f[f"seg_services_{window}h"] = (
            f.groupby([EVENT, SEG], group_keys=False)
            .apply(rolling_sum, value_col="lastserviced_changed", window=window)
            .fillna(0)
        )
        f[f"route_services_{window}h"] = (
            f.groupby([EVENT, "snowrouteid"], group_keys=False)
            .apply(rolling_sum, value_col="lastserviced_changed", window=window)
            .fillna(0)
        )

        seg_roll = (
            f.groupby([EVENT, SEG], group_keys=False)
            .apply(
                rolling_delta_ratio,
                value_col="passes_event",
                window=window,
                prefix="passes_event",
            )
            .fillna(0)
        )
        f = f.join(seg_roll)

        route_roll = (
            f.groupby([EVENT, "snowrouteid"], group_keys=False)
            .apply(
                rolling_delta_ratio,
                value_col="passes_event",
                window=window,
                prefix="route_passes_event",
            )
            .fillna(0)
        )
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


def derive_eta_from_probs(prob_df: pd.DataFrame) -> pd.Series:
    def eta(row: pd.Series) -> float:
        for h in HORIZONS:
            if row.get(f"p_{h}h", np.nan) >= ETA_THRESHOLD:
                return float(h)
        return np.nan

    return prob_df.apply(eta, axis=1)


def train_eta_regression(
    df: pd.DataFrame,
    X: pd.DataFrame,
    train_mask: pd.Series,
    models: dict[int, HistGradientBoostingClassifier | None],
    calibrated: dict[int, CalibratedClassifierCV | None],
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
    metrics["rmse"] = float(mean_squared_error(y_true, preds, squared=False))

    prob_df = pd.DataFrame(index=df.loc[test_idx].index)
    for h in HORIZONS:
        model = calibrated.get(h) or models.get(h)
        if model is None:
            prob_df[f"p_{h}h"] = np.nan
        else:
            prob_df[f"p_{h}h"] = model.predict_proba(X[test_idx])[:, 1]
    eta_series = derive_eta_from_probs(prob_df)
    valid_eta = eta_series.notna()
    if valid_eta.any():
        eta_true = y_true[valid_eta]
        eta_pred = eta_series[valid_eta]
        metrics["eta_class_mae"] = float(mean_absolute_error(eta_true, eta_pred))
        metrics["eta_class_rmse"] = float(
            mean_squared_error(eta_true, eta_pred, squared=False)
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
]:
    if df.empty:
        return {}, {}, {}, pd.Series(dtype=float), None, {}

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
    }
    feature_importance: dict[int, list[dict[str, float]]] = {}

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
            calibrator = CalibratedClassifierCV(clf, cv="prefit", method="sigmoid")
            calibrator.fit(X_calib, y_calib)
            calibrated[h] = calibrator
        else:
            calibrated[h] = None

        if X_test.empty or y_test.nunique() < 2:
            auc = None
            brier = None
            pr_auc = None
        else:
            model_for_eval = calibrated[h] or clf
            p = model_for_eval.predict_proba(X_test)[:, 1]
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

        if calibrated[h] is not None and not X_calib.empty and y_calib.nunique() >= 2:
            p_calib = calibrated[h].predict_proba(X_calib)[:, 1]
            metrics["calibration"][h] = {
                "rows": int(X_calib.shape[0]),
                "brier": float(brier_score_loss(y_calib, p_calib)),
                "ece": expected_calibration_error(
                    y_calib.to_numpy(), p_calib
                ),
            }
        else:
            metrics["calibration"][h] = {
                "rows": int(X_calib.shape[0]),
                "brier": None,
                "ece": None,
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
            df, X, train_mask, models, calibrated
        )
        metrics["eta_regression"] = reg_metrics
        return models, calibrated, metrics, medians, reg_model, feature_importance

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
        df, X, train_mask, models, calibrated
    )
    metrics["eta_regression"] = reg_metrics

    return models, calibrated, metrics, medians, reg_model, feature_importance


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
            latest[f"p_{h}h"] = np.nan
        else:
            latest[f"p_{h}h"] = model.predict_proba(X_now)[:, 1]

    latest["p_2h"] = latest[["p_1h", "p_2h"]].max(axis=1)
    latest["p_4h"] = latest[["p_2h", "p_4h"]].max(axis=1)
    latest["p_8h"] = latest[["p_4h", "p_8h"]].max(axis=1)

    latest["eta_hours_60"] = derive_eta_from_probs(latest)
    latest["eta_ts_60"] = latest[TS] + pd.to_timedelta(latest["eta_hours_60"], unit="h")

    if reg_model is not None:
        latest["eta_hours_pred"] = reg_model.predict(X_now)
    else:
        latest["eta_hours_pred"] = np.nan

    for h in HORIZONS:
        latest.loc[latest["prediction_status"] != "OK", f"p_{h}h"] = np.nan
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
    if not geo_df.empty:
        snapshots = snapshots.merge(
            geo_df[[SEG, EVENT, "passes_phase"]], on=[SEG, EVENT], how="left"
        )
        snapshots["passes_event"] = snapshots["passes_phase"].fillna(snapshots["passes"])
    else:
        snapshots["passes_event"] = snapshots["passes"]

    centroids = geo_df.dropna(subset=["lat", "lon"]).drop_duplicates(SEG)
    neighbor_lookup = build_neighbor_lookup(centroids, k=12)

    events = build_events(snapshots)
    labeled = label_next_service(snapshots, events)
    weather_hourly, weather_meta = build_weather_features(labeled)
    alerts_hourly, alerts_meta = build_alert_features(labeled)
    featured = add_features(
        labeled, events, neighbor_lookup, weather_hourly, alerts_hourly
    )
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
    ]

    train_df = featured[featured["prediction_status"] == "OK"].copy()
    models, calibrated, metrics, medians, reg_model, feature_importance = train_models(
        train_df, feature_cols
    )

    pred_latest = predict_latest(
        featured, models, calibrated, medians, feature_cols, reg_model
    )
    pred_path = os.path.join(ARTIFACT_DIR, "predictions_latest_prob.csv")
    pred_latest.to_csv(pred_path, index=False)

    metrics_path = os.path.join(ARTIFACT_DIR, "model_metrics_prob.json")
    metrics["feature_importance_path"] = os.path.join(
        ARTIFACT_DIR, "feature_importance.json"
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    feature_path = os.path.join(ARTIFACT_DIR, "feature_importance.json")
    with open(feature_path, "w", encoding="utf-8") as f:
        json.dump(feature_importance, f, indent=2)

    weather_path = os.path.join(ARTIFACT_DIR, "weather_data_sources.json")
    with open(weather_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "noaa_observations": weather_meta,
                "nws_alerts": alerts_meta,
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
