const map = L.map("map", {
  zoomControl: false,
  preferCanvas: true,
}).setView([43.04, -76.15], 12);

L.control.zoom({ position: "topright" }).addTo(map);

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

const routeDetails = document.querySelector("#route-details");
const toggleRoutes = document.querySelector("#toggle-routes");
const opacityRange = document.querySelector("#opacity-range");
const bucketControls = document.querySelector("#bucket-controls");
const metricTotal = document.querySelector("#metric-total");
const metricNever = document.querySelector("#metric-never");
const metricNeverBar = document.querySelector("#metric-never-bar");
const metricNeverFoot = document.querySelector("#metric-never-foot");
const metricGt24 = document.querySelector("#metric-gt24");
const metricGt24Bar = document.querySelector("#metric-gt24-bar");
const metricGt24Foot = document.querySelector("#metric-gt24-foot");
const metric12 = document.querySelector("#metric-12-24");
const metric12Bar = document.querySelector("#metric-12-24-bar");
const metric12Foot = document.querySelector("#metric-12-24-foot");
const metric6 = document.querySelector("#metric-6-12");
const metric6Bar = document.querySelector("#metric-6-12-bar");
const metric6Foot = document.querySelector("#metric-6-12-foot");
const metric1 = document.querySelector("#metric-1-6");
const metric1Bar = document.querySelector("#metric-1-6-bar");
const metric1Foot = document.querySelector("#metric-1-6-foot");
const metricLt1 = document.querySelector("#metric-lt1");
const metricLt1Bar = document.querySelector("#metric-lt1-bar");
const metricLt1Foot = document.querySelector("#metric-lt1-foot");
const predictionCurrent = document.querySelector("#prediction-current");
const predictionTypical = document.querySelector("#prediction-typical");
const predictionEta = document.querySelector("#prediction-eta");
const predictionChance = document.querySelector("#prediction-chance");
const predictionNote = document.querySelector("#prediction-note");

let routesLayer;
let geojsonData;

const bucketColors = {
  never: "#ef4444",
  gt24: "#f97316",
  "12-24": "#f59e0b",
  "6-12": "#eab308",
  "1-6": "#84cc16",
  lt1: "#22c55e",
};

const bucketLabels = {
  never: "Never plowed",
  gt24: "> 24 hours",
  "12-24": "12–24 hours",
  "6-12": "6–12 hours",
  "1-6": "1–6 hours",
  lt1: "< 1 hour",
};

const bucketOrder = ["never", "gt24", "12-24", "6-12", "1-6", "lt1"];

const getBucket = (feature) => {
  const lastServiced = feature?.properties?.lastserviced;
  if (!lastServiced) return "never";
  const hours = (Date.now() - lastServiced) / (1000 * 60 * 60);
  if (hours > 24) return "gt24";
  if (hours > 12) return "12-24";
  if (hours > 6) return "6-12";
  if (hours > 1) return "1-6";
  return "lt1";
};

const styleRoutes = (feature) => {
  const bucket = getBucket(feature);
  const selectedBuckets = getSelectedBuckets();
  const isFiltered = !selectedBuckets.has(bucket);
  const opacity = parseFloat(opacityRange.value);

  return {
    color: bucketColors[bucket] || "#94a3b8",
    weight: isFiltered ? 1.3 : 3.5,
    opacity: isFiltered ? opacity * 0.25 : opacity,
  };
};

const formatTimestamp = (value) => {
  if (!value) return "Unavailable";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Unavailable" : date.toLocaleString();
};

const updateContextPanel = (feature) => {
  const props = feature.properties || {};
  const bucket = getBucket(feature);
  routeDetails.innerHTML = `
    <h4>${props.roadname || "Unnamed Route"}</h4>
    <p class="muted">${props.snowrouteid || "Route ID N/A"}</p>
    <dl>
      <div>
        <dt>Priority</dt>
        <dd>${props.routepriority || "Priority 3"}</dd>
      </div>
      <div>
        <dt>Recency bucket</dt>
        <dd>${bucketLabels[bucket]}</dd>
      </div>
      <div>
        <dt>Status</dt>
        <dd>${props.servicestatus || "Unknown"}</dd>
      </div>
      <div>
        <dt>Last serviced</dt>
        <dd>${formatTimestamp(props.lastserviced)}</dd>
      </div>
      <div>
        <dt>Segment length</dt>
        <dd>${(props.segmentlength || 0).toFixed(2)} mi</dd>
      </div>
    </dl>
  `;
};

const getSelectedBuckets = () => {
  const selected = new Set();
  if (!bucketControls) return selected;
  bucketControls.querySelectorAll("input[type='checkbox']").forEach((input) => {
    if (input.checked) selected.add(input.value);
  });
  return selected;
};

const getMiles = (feature) => {
  const length = feature?.properties?.segmentlength;
  return Number.isFinite(length) ? length : 0;
};

const formatMiles = (value) => `${value.toFixed(2)} miles`;

const updateMetrics = () => {
  if (!geojsonData) return;
  const buckets = Object.fromEntries(bucketOrder.map((key) => [key, 0]));
  let total = 0;

  geojsonData.features.forEach((feature) => {
    const miles = getMiles(feature);
    const bucket = getBucket(feature);
    buckets[bucket] += miles;
    if (feature?.properties?.lastserviced) total += miles;
  });

  const totalMiles = Math.max(total, 1);
  const setMetric = (valueEl, barEl, footEl, miles, color) => {
    valueEl.textContent = formatMiles(miles);
    barEl.style.width = `${(miles / totalMiles) * 100}%`;
    barEl.style.background = color;
    footEl.textContent = `${((miles / totalMiles) * 100).toFixed(1)}% of total miles`;
  };

  metricTotal.textContent = formatMiles(total);
  setMetric(metricNever, metricNeverBar, metricNeverFoot, buckets.never, bucketColors.never);
  setMetric(metricGt24, metricGt24Bar, metricGt24Foot, buckets.gt24, bucketColors.gt24);
  setMetric(metric12, metric12Bar, metric12Foot, buckets["12-24"], bucketColors["12-24"]);
  setMetric(metric6, metric6Bar, metric6Foot, buckets["6-12"], bucketColors["6-12"]);
  setMetric(metric1, metric1Bar, metric1Foot, buckets["1-6"], bucketColors["1-6"]);
  setMetric(metricLt1, metricLt1Bar, metricLt1Foot, buckets.lt1, bucketColors.lt1);

  updatePredictions(buckets, totalMiles);
};

const updatePredictions = (buckets, totalMiles) => {
  const sorted = bucketOrder
    .map((key) => ({ key, miles: buckets[key] }))
    .sort((a, b) => b.miles - a.miles);
  const current = sorted[0]?.key || "never";
  const selectedBuckets = [...getSelectedBuckets()];
  const typical = selectedBuckets[0] || current;

  const medianIndex = Math.floor(bucketOrder.length / 2);
  const etaLookup = {
    never: "24+ hours",
    gt24: "18–24 hours",
    "12-24": "12–18 hours",
    "6-12": "6–12 hours",
    "1-6": "2–6 hours",
    lt1: "<1 hour",
  };
  const chanceLookup = {
    never: "8%",
    gt24: "21%",
    "12-24": "34%",
    "6-12": "56%",
    "1-6": "78%",
    lt1: "94%",
  };

  predictionCurrent.textContent = bucketLabels[current];
  predictionTypical.textContent = bucketLabels[typical];
  predictionEta.textContent = etaLookup[bucketOrder[medianIndex]];
  predictionChance.textContent = chanceLookup[typical];
  predictionNote.textContent = `Based on ${totalMiles.toFixed(
    1
  )} miles of valid timestamp data.`;
};

const onEachFeature = (feature, layer) => {
  layer.on({
    click: () => updateContextPanel(feature),
    mouseover: () => layer.setStyle({ weight: 5 }),
    mouseout: () => routesLayer.resetStyle(layer),
  });
};

const renderRoutes = () => {
  if (!geojsonData) return;
  if (routesLayer) {
    routesLayer.remove();
  }

  routesLayer = L.geoJSON(geojsonData, {
    style: styleRoutes,
    onEachFeature,
  });

  if (toggleRoutes.checked) {
    routesLayer.addTo(map);
  }

  if (!map._loaded && routesLayer.getLayers().length) {
    map.fitBounds(routesLayer.getBounds().pad(0.1));
  }
};

const setRoutesVisibility = () => {
  if (!routesLayer) return;
  if (toggleRoutes.checked) {
    routesLayer.addTo(map);
  } else {
    routesLayer.remove();
  }
};

const bindControls = () => {
  toggleRoutes.addEventListener("change", setRoutesVisibility);
  opacityRange.addEventListener("input", () => renderRoutes());
  if (bucketControls) {
    bucketControls
      .querySelectorAll("input[type='checkbox']")
      .forEach((input) => {
        input.addEventListener("change", () => {
          renderRoutes();
          updateMetrics();
        });
      });
  }
};

const loadRoutes = async () => {
  try {
    const response = await fetch(
      "../../winter_operations_snow_routes_layer0.geojson"
    );
    if (!response.ok) throw new Error("Failed to load GeoJSON");
    geojsonData = await response.json();
    renderRoutes();
    updateMetrics();
  } catch (error) {
    routeDetails.innerHTML = `
      <p class="muted">Unable to load GeoJSON data. Check the file path.</p>
    `;
    console.error(error);
  }
};

bindControls();
loadRoutes();
