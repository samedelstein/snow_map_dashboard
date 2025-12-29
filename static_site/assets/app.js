/*
  Live ArcGIS Feature Service endpoint (same as snow_map_dashboard/app.py BASE_URL):
  https://services6.arcgis.com/bdPqSfflsdgFRVVM/arcgis/rest/services/Winter_Operations_Snow_Routes/FeatureServer/0/query

  Fallback snapshot GeoJSON (written by Actions):
  static_site/data/latest_routes.geojson

  Prediction artifact GeoJSON (static build output):
  static_site/data/predictions_latest.geojson
*/

const LIVE_API_URL =
  "https://services6.arcgis.com/bdPqSfflsdgFRVVM/arcgis/rest/services/Winter_Operations_Snow_Routes/FeatureServer/0/query";
const FALLBACK_GEOJSON_PATH = "data/latest_routes.geojson";
const PREDICTIONS_GEOJSON_PATH = "data/predictions_latest.geojson";
const DEFAULT_CUTOFF_ISO = "2025-12-01T00:00:00Z";
const PAGE_SIZE = 2000;

const BUCKETS = [
  "Never plowed (before cutoff)",
  "> 24 hours",
  "12–24 hours",
  "6–12 hours",
  "1–6 hours",
  "< 1 hour",
];

const BUCKET_COLORS = {
  "Never plowed (before cutoff)": "#c8c8c8",
  "> 24 hours": "#800026",
  "12–24 hours": "#e31a1c",
  "6–12 hours": "#fd8d3c",
  "1–6 hours": "#ffed6f",
  "< 1 hour": "#2ca25f",
};

const statusEl = document.getElementById("status");
const detailsEl = document.getElementById("details");
const summaryStatusEl = document.getElementById("summary-status");
const summaryEtaEl = document.getElementById("summary-eta");
const summarySeverityEl = document.getElementById("summary-severity");

const map = L.map("map", { zoomControl: true }).setView([43.0481, -76.1474], 12);

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

function classifyBucket(lastServicedMs, cutoffIso) {
  if (!lastServicedMs) {
    return "Never plowed (before cutoff)";
  }
  const cutoff = new Date(cutoffIso);
  const serviced = new Date(lastServicedMs);
  if (Number.isNaN(serviced.getTime())) {
    return "Never plowed (before cutoff)";
  }
  if (serviced < cutoff) {
    return "Never plowed (before cutoff)";
  }
  const hours = (Date.now() - serviced.getTime()) / 36e5;
  if (hours < 1) return "< 1 hour";
  if (hours < 6) return "1–6 hours";
  if (hours < 12) return "6–12 hours";
  if (hours < 24) return "12–24 hours";
  return "> 24 hours";
}

async function fetchArcGisGeojson() {
  let offset = 0;
  const features = [];

  while (true) {
    const params = new URLSearchParams({
      f: "geojson",
      where: "1=1",
      outFields: "*",
      returnGeometry: "true",
      outSR: "4326",
      resultOffset: offset.toString(),
      resultRecordCount: PAGE_SIZE.toString(),
    });

    const response = await fetch(`${LIVE_API_URL}?${params.toString()}`);
    if (!response.ok) {
      throw new Error(`ArcGIS request failed: ${response.status}`);
    }
    const data = await response.json();
    if (!data.features || data.features.length === 0) {
      break;
    }
    features.push(...data.features);
    if (!data.exceededTransferLimit && data.features.length < PAGE_SIZE) {
      break;
    }
    offset += PAGE_SIZE;
  }

  return { type: "FeatureCollection", features };
}

async function fetchGeojson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load ${path}`);
  }
  return response.json();
}

function indexPredictions(predictions) {
  const bySegment = new Map();
  const byObjectId = new Map();

  predictions.features.forEach((feature) => {
    const props = feature.properties || {};
    if (props.snowroutesegmentid) {
      bySegment.set(String(props.snowroutesegmentid), feature);
    }
    if (props.OBJECTID) {
      byObjectId.set(String(props.OBJECTID), feature);
    }
  });

  return { bySegment, byObjectId };
}

function mergePredictionProps(liveFeature, predictionIndex) {
  const props = liveFeature.properties || {};
  const segmentId = props.snowroutesegmentid ? String(props.snowroutesegmentid) : null;
  const objectId = props.OBJECTID ? String(props.OBJECTID) : null;

  let predictionFeature = null;
  if (segmentId && predictionIndex.bySegment.has(segmentId)) {
    predictionFeature = predictionIndex.bySegment.get(segmentId);
  } else if (objectId && predictionIndex.byObjectId.has(objectId)) {
    predictionFeature = predictionIndex.byObjectId.get(objectId);
  }

  if (!predictionFeature) {
    return props;
  }

  const predictionProps = predictionFeature.properties || {};
  return {
    ...props,
    prediction_status: predictionProps.prediction_status || "–",
    eta_hours_pred: predictionProps.eta_hours_pred ?? predictionProps.eta_bucket_hours ?? null,
    storm_severity_max: predictionProps.storm_severity_max || "–",
    predicted_snapshot_ts: predictionProps.snapshot_ts || "–",
    predicted_route_priority: predictionProps.routepriority || props.routepriority,
  };
}

function formatServiceTime(value) {
  if (!value) return "–";
  const date = new Date(value);
  if (!Number.isNaN(date.getTime())) {
    return date.toLocaleString();
  }
  const numeric = Number(value);
  if (!Number.isNaN(numeric)) {
    const fromEpoch = new Date(numeric);
    return Number.isNaN(fromEpoch.getTime()) ? String(value) : fromEpoch.toLocaleString();
  }
  return String(value);
}

function updateStatus(text, tone = "info") {
  statusEl.textContent = text;
  statusEl.className = `status status-${tone}`;
}

function renderDetails(feature) {
  const props = feature.properties || {};
  const bucket = props.bucket || "–";

  detailsEl.innerHTML = `
    <div class="detail-row"><strong>Road</strong> ${props.roadname || "Unknown"}</div>
    <div class="detail-row"><strong>Route priority</strong> ${props.routepriority || "–"}</div>
    <div class="detail-row"><strong>Segment ID</strong> ${props.snowroutesegmentid || "–"}</div>
    <div class="detail-row"><strong>Last serviced</strong> ${formatServiceTime(props.lastserviced)}</div>
    <div class="detail-row"><strong>Service bucket</strong> ${bucket}</div>
    <div class="detail-row"><strong>Prediction snapshot</strong> ${props.predicted_snapshot_ts || "–"}</div>
  `;

  summaryStatusEl.textContent = props.prediction_status || "–";
  summaryEtaEl.textContent =
    props.eta_hours_pred !== null && props.eta_hours_pred !== undefined
      ? `${Number(props.eta_hours_pred).toFixed(1)} hrs`
      : "–";
  summarySeverityEl.textContent = props.storm_severity_max || "–";
}

function addGeojsonToMap(geojson) {
  const layer = L.geoJSON(geojson, {
    style: (feature) => ({
      color: BUCKET_COLORS[feature.properties.bucket] || "#94a3b8",
      weight: 4,
      opacity: 0.85,
    }),
    onEachFeature: (feature, leafletLayer) => {
      leafletLayer.on("click", () => renderDetails(feature));
      leafletLayer.bindTooltip(
        `${feature.properties.roadname || "Unknown"}<br />${feature.properties.bucket}`,
        { sticky: true }
      );
    },
  }).addTo(map);

  if (layer.getBounds && layer.getBounds().isValid()) {
    map.fitBounds(layer.getBounds(), { padding: [20, 20] });
  }
}

async function initialize() {
  updateStatus("Loading live snow routes…", "info");

  let liveGeojson;
  let sourceLabel = "live";

  try {
    liveGeojson = await fetchArcGisGeojson();
  } catch (error) {
    console.warn("Live API fetch failed, falling back", error);
    liveGeojson = await fetchGeojson(FALLBACK_GEOJSON_PATH);
    sourceLabel = "snapshot";
  }

  const predictionsGeojson = await fetchGeojson(PREDICTIONS_GEOJSON_PATH);
  const predictionIndex = indexPredictions(predictionsGeojson);

  liveGeojson.features = liveGeojson.features.map((feature) => {
    const mergedProps = mergePredictionProps(feature, predictionIndex);
    const lastServiced = mergedProps.lastserviced ?? mergedProps.lastserviced_dt;
    const bucket = mergedProps.bucket || classifyBucket(lastServiced, DEFAULT_CUTOFF_ISO);
    return {
      ...feature,
      properties: {
        ...mergedProps,
        bucket,
      },
    };
  });

  addGeojsonToMap(liveGeojson);

  updateStatus(
    `Loaded ${liveGeojson.features.length} segments (${sourceLabel} data). Click a segment for details.`,
    "ready"
  );
}

initialize().catch((error) => {
  console.error("Failed to initialize map", error);
  updateStatus("Unable to load map data.", "error");
});
