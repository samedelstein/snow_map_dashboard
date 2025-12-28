"""
Snowplow probability prediction model (robust, geometry optional).
Fetches snapshots, engineers features, trains horizon models, and writes predictions.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.neighbors import BallTree

# -----------------------------
# Config
# -----------------------------
SNAPSHOT_URL = (
    "https://raw.githubusercontent.com/samedelstein/snow_map_dashboard/"
    "refs/heads/main/snapshot_snow_routes/snapshots.csv"
)
GEOJSON_PATH = "snapshot_snow_routes/latest_routes.geojson"

EVENT = "eventid"
SEG = "snowroutesegmentid"
TS = "snapshot_ts"

HORIZONS = [1, 2, 4, 8]
ETA_THRESHOLD = 0.60
ARTIFACT_DIR = "artifacts_snow"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


# -----------------------------
# Load snapshots
# -----------------------------
def load_snapshots(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
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
    last_snapshot = labeled.groupby([EVENT, SEG])[TS].max().rename("last_snapshot_ts")
    labeled = labeled.merge(last_snapshot, on=[EVENT, SEG], how="left")
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
    labeled["hours_to_censor"] = (
        labeled["last_snapshot_ts"] - labeled[TS]
    ).dt.total_seconds() / 3600.0
    labeled["event_observed"] = labeled["hours_to_next_service"].notna().astype(int)
    labeled["duration_hours"] = labeled["hours_to_next_service"].where(
        labeled["event_observed"].eq(1), labeled["hours_to_censor"]
    )
    return labeled


# -----------------------------
# Feature engineering
# -----------------------------
def add_features(
    df: pd.DataFrame, events: pd.DataFrame, neighbor_lookup: dict[str, list[str]]
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


# -----------------------------
# Train models
# -----------------------------
def train_models(
    df: pd.DataFrame, features: list[str]
) -> tuple[dict[int, HistGradientBoostingClassifier | None], dict[int, dict], pd.Series]:
    if df.empty:
        return {}, {}, pd.Series(dtype=float)

    cutoff = df[TS].quantile(0.8)
    train_mask = df[TS] <= cutoff

    X = df[features].replace([np.inf, -np.inf], np.nan)
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)

    models: dict[int, HistGradientBoostingClassifier | None] = {}
    metrics: dict[int, dict] = {}

    for h in HORIZONS:
        y = df[f"y_{h}h"].astype(int)
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[~train_mask], y[~train_mask]

        if X_train.empty or y_train.nunique() < 2:
            models[h] = None
            metrics[h] = {
                "rows_train": int(X_train.shape[0]),
                "rows_test": int(X_test.shape[0]),
                "auc": None,
                "brier": None,
                "cutoff_utc": str(cutoff),
                "note": "insufficient training data",
            }
            continue

        clf = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.08,
            max_iter=350,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        models[h] = clf

        if X_test.empty or y_test.nunique() < 2:
            auc = None
            brier = None
        else:
            p = clf.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, p))
            brier = float(brier_score_loss(y_test, p))

        metrics[h] = {
            "rows_train": int(X_train.shape[0]),
            "rows_test": int(X_test.shape[0]),
            "auc": auc,
            "brier": brier,
            "cutoff_utc": str(cutoff),
        }

    return models, metrics, medians


# -----------------------------
# Train survival model
# -----------------------------
def train_survival_model(
    df: pd.DataFrame, features: list[str]
) -> tuple[CoxPHFitter | None, pd.Series, dict]:
    if df.empty:
        return None, pd.Series(dtype=float), {}

    work = df.copy()
    work = work[work["duration_hours"].notna()].copy()
    work = work[work["duration_hours"] > 0]
    if work.empty:
        return None, pd.Series(dtype=float), {}

    X = work[features].replace([np.inf, -np.inf], np.nan)
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)

    model_df = X.copy()
    model_df["duration_hours"] = work["duration_hours"]
    model_df["event_observed"] = work["event_observed"].astype(int)

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(model_df, duration_col="duration_hours", event_col="event_observed")

    metrics = {
        "rows_train": int(model_df.shape[0]),
        "concordance": float(cph.concordance_index_),
    }
    return cph, medians, metrics


# -----------------------------
# Predict latest per segment
# -----------------------------
def predict_latest(
    df: pd.DataFrame,
    models: dict[int, HistGradientBoostingClassifier | None],
    medians: pd.Series,
    features: list[str],
    survival_model: CoxPHFitter | None,
    survival_medians: pd.Series,
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
        model = models.get(h)
        if model is None:
            latest[f"p_{h}h"] = np.nan
        else:
            latest[f"p_{h}h"] = model.predict_proba(X_now)[:, 1]

    latest["p_2h"] = latest[["p_1h", "p_2h"]].max(axis=1)
    latest["p_4h"] = latest[["p_2h", "p_4h"]].max(axis=1)
    latest["p_8h"] = latest[["p_4h", "p_8h"]].max(axis=1)

    def eta(row: pd.Series) -> float:
        for h in HORIZONS:
            if row.get(f"p_{h}h", np.nan) >= ETA_THRESHOLD:
                return float(h)
        return np.nan

    latest["eta_hours_60"] = latest.apply(eta, axis=1)
    latest["eta_ts_60"] = latest[TS] + pd.to_timedelta(latest["eta_hours_60"], unit="h")

    if survival_model is None:
        latest["eta_hours_pred"] = np.nan
        latest["eta_hours_ci_low"] = np.nan
        latest["eta_hours_ci_high"] = np.nan
    else:
        X_surv = latest[features].replace([np.inf, -np.inf], np.nan)
        if not survival_medians.empty:
            X_surv = X_surv.fillna(survival_medians)
        else:
            X_surv = X_surv.fillna(0)

        eta_median = survival_model.predict_median(X_surv)
        eta_low = survival_model.predict_percentile(X_surv, p=0.9)
        eta_high = survival_model.predict_percentile(X_surv, p=0.1)

        latest["eta_hours_pred"] = eta_median.to_numpy()
        latest["eta_hours_ci_low"] = eta_low.to_numpy()
        latest["eta_hours_ci_high"] = eta_high.to_numpy()

    latest["eta_ts_pred"] = latest[TS] + pd.to_timedelta(latest["eta_hours_pred"], unit="h")
    latest["eta_ts_ci_low"] = latest[TS] + pd.to_timedelta(latest["eta_hours_ci_low"], unit="h")
    latest["eta_ts_ci_high"] = latest[TS] + pd.to_timedelta(latest["eta_hours_ci_high"], unit="h")

    for h in HORIZONS:
        latest.loc[latest["prediction_status"] != "OK", f"p_{h}h"] = np.nan
    latest.loc[
        latest["prediction_status"] != "OK",
        [
            "eta_hours_60",
            "eta_ts_60",
            "eta_hours_pred",
            "eta_hours_ci_low",
            "eta_hours_ci_high",
            "eta_ts_pred",
            "eta_ts_ci_low",
            "eta_ts_ci_high",
        ],
    ] = np.nan

    return latest


def main() -> None:
    snapshots = load_snapshots(SNAPSHOT_URL)
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
    featured = add_features(labeled, events, neighbor_lookup)
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
        "neighbor_services_last_hour",
    ]

    train_df = featured[featured["prediction_status"] == "OK"].copy()
    models, metrics, medians = train_models(train_df, feature_cols)
    survival_model, survival_medians, survival_metrics = train_survival_model(
        train_df, feature_cols
    )

    pred_latest = predict_latest(
        featured,
        models,
        medians,
        feature_cols,
        survival_model,
        survival_medians,
    )
    pred_path = os.path.join(ARTIFACT_DIR, "predictions_latest_prob.csv")
    pred_latest.to_csv(pred_path, index=False)

    metrics_path = os.path.join(ARTIFACT_DIR, "model_metrics_prob.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    survival_metrics_path = os.path.join(ARTIFACT_DIR, "model_metrics_survival.json")
    with open(survival_metrics_path, "w", encoding="utf-8") as f:
        json.dump(survival_metrics, f, indent=2)

    print(f"Saved: {pred_path} | rows: {len(pred_latest)}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {survival_metrics_path}")

    if not pred_latest.empty:
        untracked_share = (pred_latest["prediction_status"] != "OK").mean()
        print(f"Untracked share: {untracked_share:.3f}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
