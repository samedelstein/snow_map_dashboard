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
const priorityFilter = document.querySelector("#priority-filter");

let routesLayer;
let geojsonData;

const priorityColors = {
  "1": "#ef4444",
  "2": "#f97316",
  "3": "#3b82f6",
};

const getPriorityValue = (feature) => {
  const raw = feature?.properties?.routepriority || "";
  const match = raw.match(/\d/);
  return match ? match[0] : "3";
};

const styleRoutes = (feature) => {
  const priority = getPriorityValue(feature);
  const isFiltered =
    priorityFilter.value !== "all" &&
    priorityFilter.value !== priority;
  const opacity = parseFloat(opacityRange.value);

  return {
    color: priorityColors[priority] || priorityColors["3"],
    weight: isFiltered ? 1.5 : 3.5,
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
  routeDetails.innerHTML = `
    <h4>${props.roadname || "Unnamed Route"}</h4>
    <p class="muted">${props.snowrouteid || "Route ID N/A"}</p>
    <dl>
      <div>
        <dt>Priority</dt>
        <dd>${props.routepriority || "Priority 3"}</dd>
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
  priorityFilter.addEventListener("change", () => renderRoutes());
};

const loadRoutes = async () => {
  try {
    const response = await fetch(
      "../../winter_operations_snow_routes_layer0.geojson"
    );
    if (!response.ok) throw new Error("Failed to load GeoJSON");
    geojsonData = await response.json();
    renderRoutes();
  } catch (error) {
    routeDetails.innerHTML = `
      <p class="muted">Unable to load GeoJSON data. Check the file path.</p>
    `;
    console.error(error);
  }
};

bindControls();
loadRoutes();
