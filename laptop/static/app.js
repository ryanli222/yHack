const dashboardState = {
  snapshot: null,
  ws: null,
  reconnectTimer: null,
  configRenderedHash: "",
  mapMode: "goal",
  foxgloveEmbedUrl: "",
  keyboardVector: { vx: 0, vy: 0, omega: 0 },
  keyboardInterval: null,
  joystick: {
    active: false,
    pointerId: null,
    vx: 0,
    vy: 0,
    interval: null,
  },
  visibleLogs: [],
};

const elements = {};

document.addEventListener("DOMContentLoaded", () => {
  cacheElements();
  bindEvents();
  connectWebSocket();
  refreshWaypoints();
  refreshConfig();
  initCameraStream();
  resizeCanvases();
  window.addEventListener("resize", () => {
    resizeCanvases();
    renderAll();
  });
});

function cacheElements() {
  const ids = [
    "robot-name",
    "sidebar-robot-name",
    "breadcrumb-root",
    "breadcrumb-leaf",
    "connection-pill",
    "sidebar-live-pill",
    "snapshot-rate",
    "robot-dot",
    "robot-status-text",
    "subsystems-tree",
    "modules-list",
    "pose-x",
    "pose-y",
    "pose-theta",
    "pose-frame",
    "pose-cov-xy",
    "pose-cov-theta",
    "vel-vx",
    "vel-vy",
    "vel-omega",
    "nav-mode",
    "nav-state",
    "nav-estop",
    "imu-status",
    "imu-gyro-z",
    "imu-age",
    "waypoint-select",
    "arm-voice-pill",
    "arm-voice-detail",
    "arm-voice-start-button",
    "arm-voice-stop-button",
    "pickup-pill",
    "pickup-detail",
    "pickup-refresh-button",
    "pickup-start-button",
    "pickup-stop-button",
    "pickup-label",
    "pickup-confidence",
    "pickup-range",
    "pickup-lateral",
    "pickup-zone",
    "pickup-attempts",
    "goal-summary",
    "command-queue-note",
    "map-canvas",
    "map-mode-label",
    "map-cursor-coords",
    "camera-stream",
    "camera-overlay",
    "foxglove-status-pill",
    "foxglove-note",
    "foxglove-ws-url",
    "foxglove-frame",
    "obstacles-list",
    "landmarks-list",
    "voice-transcript",
    "voice-response",
    "log-viewer",
    "config-form",
    "config-note",
    "joystick-pad",
    "joystick-knob",
    "joystick-readout",
  ];
  ids.forEach((id) => {
    elements[id] = document.getElementById(id);
  });
  elements.tabButtons = Array.from(document.querySelectorAll(".tab-button"));
  elements.tabContents = Array.from(document.querySelectorAll(".tab-content"));
  elements.mapModeButtons = Array.from(document.querySelectorAll(".map-mode-button"));
  elements.teleopButtons = Array.from(document.querySelectorAll(".teleop-button"));
}

function bindEvents() {
  elements.tabButtons.forEach((button) => {
    button.addEventListener("click", () => activateTab(button.dataset.tab));
  });

  elements.mapModeButtons.forEach((button) => {
    button.addEventListener("click", () => {
      dashboardState.mapMode = button.dataset.mapMode;
      elements.mapModeButtons.forEach((candidate) => {
        candidate.classList.toggle("active", candidate === button);
      });
      updateMapModeLabel();
    });
  });

  document.getElementById("goto-button").addEventListener("click", async () => {
    const waypoint = elements["waypoint-select"].value;
    if (!waypoint) {
      return;
    }
    await postJson("/command", { cmd: "goto", waypoint });
  });

  document.getElementById("reload-waypoints-button").addEventListener("click", refreshWaypoints);
  document.getElementById("refresh-config-button").addEventListener("click", refreshConfig);
  document.getElementById("save-config-button").addEventListener("click", saveConfig);
  document.getElementById("foxglove-reload-button").addEventListener("click", reloadFoxgloveFrame);
  document.getElementById("foxglove-open-button").addEventListener("click", openFoxgloveExternal);
  document.getElementById("stop-button").addEventListener("click", () => postJson("/command", { cmd: "stop" }));
  document.getElementById("arm-voice-start-button").addEventListener("click", startArmVoiceTest);
  document.getElementById("arm-voice-stop-button").addEventListener("click", stopArmVoiceTest);
  document.getElementById("pickup-refresh-button").addEventListener("click", refreshPickupDetection);
  document.getElementById("pickup-start-button").addEventListener("click", startPickup);
  document.getElementById("pickup-stop-button").addEventListener("click", stopPickup);
  document.getElementById("clear-log-view-button").addEventListener("click", () => {
    dashboardState.visibleLogs = [];
    renderLogs();
  });

  elements.teleopButtons.forEach((button) => {
    const vector = teleopPreset(button.dataset.teleop);
    const start = () => startTeleopPulse(vector);
    const stop = () => stopTeleopPulse();
    button.addEventListener("mousedown", start);
    button.addEventListener("mouseup", stop);
    button.addEventListener("mouseleave", stop);
    button.addEventListener("touchstart", (event) => {
      event.preventDefault();
      start();
    }, { passive: false });
    button.addEventListener("touchend", stop);
  });

  setupJoystick();
  setupKeyboardTeleop();
  setupMapInteractions();
}

function connectWebSocket() {
  if (dashboardState.ws) {
    dashboardState.ws.close();
  }
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${protocol}://${window.location.host}/ws`);
  dashboardState.ws = ws;

  ws.addEventListener("open", () => {
    window.clearTimeout(dashboardState.reconnectTimer);
  });

  ws.addEventListener("message", (event) => {
    try {
      const message = JSON.parse(event.data);
      handleMessage(message);
    } catch (error) {
      console.error("Invalid WebSocket payload", error);
    }
  });

  ws.addEventListener("close", () => {
    dashboardState.reconnectTimer = window.setTimeout(connectWebSocket, 1500);
  });
}

function handleMessage(message) {
  if (message.type !== "snapshot") {
    return;
  }
  dashboardState.snapshot = message.data;
  dashboardState.visibleLogs = Array.isArray(message.data.logs) ? message.data.logs.slice(-MAX_LOGS) : [];
  renderAll();
}

function renderAll() {
  renderMeta();
  renderSubsystems();
  renderModules();
  renderTelemetry();
  renderWaypoints();
  renderGoal();
  renderArmVoice();
  renderPickup();
  renderObstacles();
  renderLandmarks();
  renderVoice();
  renderLogs();
  renderConfig();
  renderMap();
  renderFoxglove();
  renderCameraOverlay();
}

function renderMeta() {
  const snapshot = dashboardState.snapshot;
  if (!snapshot) {
    return;
  }
  const meta = snapshot.meta || {};
  const breadcrumbs = meta.breadcrumbs || ["Wheelchair", "main"];
  elements["breadcrumb-root"].textContent = breadcrumbs[0] || "Wheelchair";
  elements["breadcrumb-leaf"].textContent = breadcrumbs[1] || "main";
  elements["robot-name"].textContent = meta.robot_name || "Autonomous Wheelchair";
  elements["sidebar-robot-name"].textContent = meta.robot_name || "Autonomous Wheelchair";
  elements["snapshot-rate"].textContent = `${formatNumber(meta.snapshot_rate_hz || 0, 0)} Hz`;

  const live = Boolean(meta.live);
  const label = meta.connection_label || (live ? "Live" : "Offline");
  setPillState(elements["connection-pill"], live ? "status-live" : "status-offline", label);
  setPillState(elements["sidebar-live-pill"], live ? "status-live" : "status-offline", label);
  elements["robot-dot"].classList.toggle("live", live);
  elements["robot-status-text"].textContent = live
    ? "Serial connected and telemetry streaming"
    : "Waiting for live robot connection";
}

function renderSubsystems() {
  const snapshot = dashboardState.snapshot;
  const container = elements["subsystems-tree"];
  if (!snapshot) {
    return;
  }
  const subsystems = snapshot.subsystems || {};
  const markup = Object.entries(subsystems).map(([name, subsystem]) => {
    const statusClass = statusClassFor(subsystem.status);
    return `
      <div class="tree-item">
        <span class="tree-icon">${escapeHtml(subsystem.icon || name.slice(0, 2).toUpperCase())}</span>
        <div>
          <div class="tree-title">${escapeHtml(name)}</div>
          <div class="tree-detail">${escapeHtml(subsystem.detail || "No detail")}</div>
        </div>
        <span class="status-pill ${statusClass}">${escapeHtml(humanizeStatus(subsystem.status))}</span>
      </div>
    `;
  }).join("");
  container.innerHTML = markup || emptyState("No subsystems reported.");
}

function renderModules() {
  const snapshot = dashboardState.snapshot;
  const container = elements["modules-list"];
  if (!snapshot) {
    return;
  }
  const modules = snapshot.modules || [];
  const markup = modules.map((module) => `
    <div class="module-item">
      <div>
        <div class="tree-title">${escapeHtml(module.name)}</div>
        <div class="tree-detail">${escapeHtml(humanizeStatus(module.status))}</div>
      </div>
      <div class="module-status">
        <span class="module-status-dot ${module.connected ? "connected" : "idle"}"></span>
        <span class="status-pill ${statusClassFor(module.status)}">${module.connected ? "Active" : "Idle"}</span>
      </div>
    </div>
  `).join("");
  container.innerHTML = markup || emptyState("No active modules.");
}

function renderTelemetry() {
  const snapshot = dashboardState.snapshot;
  if (!snapshot) {
    return;
  }
  const pose = snapshot.pose || {};
  const velocity = snapshot.velocity || {};
  const nav = snapshot.nav_state || {};
  const imu = snapshot.imu || {};

  elements["pose-x"].textContent = `${formatNumber(pose.x)} m`;
  elements["pose-y"].textContent = `${formatNumber(pose.y)} m`;
  elements["pose-theta"].textContent = `${formatNumber(pose.theta)} rad`;
  elements["pose-frame"].textContent = pose.frame || "world";
  const covariance = Array.isArray(pose.covariance) ? pose.covariance : null;
  elements["pose-cov-xy"].textContent = covariance ? formatNumber(covariance[0]?.[0] ?? 0) : "n/a";
  elements["pose-cov-theta"].textContent = covariance ? formatNumber(covariance[2]?.[2] ?? 0) : "n/a";
  elements["vel-vx"].textContent = `${formatNumber(velocity.vx)} m/s`;
  elements["vel-vy"].textContent = `${formatNumber(velocity.vy)} m/s`;
  elements["vel-omega"].textContent = `${formatNumber(velocity.omega)} rad/s`;
  elements["nav-mode"].textContent = nav.mode || "manual";
  elements["nav-state"].textContent = nav.state || "Idle";
  elements["nav-estop"].textContent = nav.estop ? "ACTIVE" : "Clear";
  elements["imu-status"].textContent = imu.receiving ? "receiving" : "idle";
  elements["imu-gyro-z"].textContent = `${formatNumber(imu.gyro_z)} rad/s`;
  elements["imu-age"].textContent = imu.packet_age_s == null ? "n/a" : `${formatNumber(imu.packet_age_s)} s`;
}

function renderWaypoints() {
  const snapshot = dashboardState.snapshot;
  const select = elements["waypoint-select"];
  if (!snapshot || !select) {
    return;
  }
  const waypoints = snapshot.waypoints || {};
  const current = select.value;
  select.innerHTML = Object.entries(waypoints).map(([name, coords]) =>
    `<option value="${escapeAttribute(name)}">${escapeHtml(name)} (${formatNumber(coords[0])}, ${formatNumber(coords[1])})</option>`
  ).join("");
  if (current && waypoints[current]) {
    select.value = current;
  }
}

function renderGoal() {
  const snapshot = dashboardState.snapshot;
  if (!snapshot) {
    return;
  }
  const goal = snapshot.goal;
  const nav = snapshot.nav_state || {};
  const meta = snapshot.meta || {};
  if (!goal) {
    elements["goal-summary"].textContent = "No active goal.";
  } else {
    const label = goal.label ? `${goal.label} ` : "";
    elements["goal-summary"].textContent = `${label}x ${formatNumber(goal.x)} | y ${formatNumber(goal.y)} | state ${nav.state || "Idle"}`;
  }
  elements["command-queue-note"].textContent = `Queued for controller: ${meta.pending_commands || 0} commands | ${meta.pending_landmarks || 0} landmark requests.`;
}

function renderArmVoice() {
  const snapshot = dashboardState.snapshot;
  if (!snapshot) {
    return;
  }
  const armVoice = snapshot.arm_voice || {};
  const status = armVoice.status || "idle";
  const running = Boolean(armVoice.running);
  const detail = armVoice.detail || "Press Start to run the arm voice script.";

  setPillState(elements["arm-voice-pill"], statusClassFor(status), humanizeStatus(status));
  elements["arm-voice-detail"].textContent = detail;
  elements["arm-voice-start-button"].disabled = running || status === "starting";
  elements["arm-voice-stop-button"].disabled = !(running || status === "starting" || status === "stopping");
}

function renderPickup() {
  const snapshot = dashboardState.snapshot;
  if (!snapshot) {
    return;
  }

  const pickup = snapshot.pickup || {};
  const detection = pickup.last_detection || null;
  const target = pickup.last_target || null;
  const status = pickup.status || "idle";
  const running = Boolean(pickup.running);

  setPillState(elements["pickup-pill"], statusClassFor(status), humanizeStatus(status));
  elements["pickup-detail"].textContent = pickup.detail || "Pickup controller idle.";
  elements["pickup-refresh-button"].disabled = running;
  elements["pickup-start-button"].disabled = running;
  elements["pickup-stop-button"].disabled = !running;
  elements["pickup-label"].textContent = detection && detection.label ? detection.label : "n/a";
  elements["pickup-confidence"].textContent = detection
    ? `${Math.round((Number(detection.confidence) || 0) * 100)}%`
    : "n/a";
  elements["pickup-range"].textContent = target && target.range_m != null
    ? `${formatNumber(target.range_m)} m`
    : "n/a";
  elements["pickup-lateral"].textContent = target && target.lateral_m != null
    ? `${formatNumber(target.lateral_m)} m`
    : "n/a";
  elements["pickup-zone"].textContent = !target
    ? "No target"
    : target.in_pickup_zone
      ? (target.centered ? "Ready" : "Off center")
      : "Outside zone";
  elements["pickup-attempts"].textContent = String(pickup.attempts || 0);
}

function renderObstacles() {
  const snapshot = dashboardState.snapshot;
  const container = elements["obstacles-list"];
  if (!snapshot) {
    return;
  }
  const obstacles = snapshot.obstacles || [];
  if (!obstacles.length) {
    container.innerHTML = emptyState("No obstacles detected.");
    return;
  }
  container.innerHTML = obstacles.map((obstacle) => `
    <div class="list-item">
      <div>
        <div class="list-title">Obstacle</div>
        <div class="list-detail">dist ${formatNumber(obstacle.distance_m)} m | angle ${formatNumber(obstacle.angle_deg)} deg</div>
      </div>
      <div class="list-metric">${Math.round((obstacle.confidence || 0) * 100)}%</div>
    </div>
  `).join("");
}

function renderLandmarks() {
  const snapshot = dashboardState.snapshot;
  const container = elements["landmarks-list"];
  if (!snapshot) {
    return;
  }
  const landmarks = snapshot.landmarks || [];
  if (!landmarks.length) {
    container.innerHTML = emptyState("No landmarks registered.");
    return;
  }
  container.innerHTML = landmarks.map((landmark) => `
    <div class="list-item">
      <div>
        <div class="list-title">${escapeHtml(landmark.name || "landmark")}</div>
        <div class="list-detail">x ${formatNumber(landmark.x)} | y ${formatNumber(landmark.y)} | ${escapeHtml(landmark.source || "map")} | ${Math.round((landmark.confidence || 0) * 100)}%</div>
      </div>
      <div class="list-metric">${escapeHtml((landmark.descriptor_status || "unknown").replaceAll("_", " "))}</div>
    </div>
  `).join("");
}

function renderVoice() {
  const snapshot = dashboardState.snapshot;
  if (!snapshot) {
    return;
  }
  const voice = snapshot.voice || {};
  elements["voice-transcript"].textContent = voice.transcript || "Waiting for command...";
  elements["voice-response"].textContent = voice.response || "No response yet.";
}

function renderFoxglove() {
  const snapshot = dashboardState.snapshot;
  if (!snapshot) {
    return;
  }

  const foxglove = snapshot.foxglove || {};
  const available = Boolean(foxglove.available);
  const status = foxglove.status || "disconnected";
  const bridgeUrl = buildFoxgloveWsUrl(foxglove);
  const embedUrl = buildFoxgloveEmbedUrl(foxglove, bridgeUrl);
  const frame = elements["foxglove-frame"];
  dashboardState.foxgloveEmbedUrl = embedUrl;

  elements["foxglove-ws-url"].textContent = bridgeUrl;
  setPillState(elements["foxglove-status-pill"], statusClassFor(status), humanizeStatus(status));

  let note = foxglove.detail || "Foxglove bridge status unknown.";
  if (!available && foxglove.install_command) {
    note = `${note} Install with: ${foxglove.install_command}`;
  }
  elements["foxglove-note"].textContent = note;

  if (available) {
    if (frame.dataset.loadedSrc !== embedUrl) {
      frame.src = embedUrl;
      frame.dataset.loadedSrc = embedUrl;
    }
  } else if (frame.dataset.loadedSrc) {
    frame.removeAttribute("src");
    frame.dataset.loadedSrc = "";
  }
}

function renderLogs() {
  const container = elements["log-viewer"];
  const logs = dashboardState.visibleLogs || [];
  if (!logs.length) {
    container.innerHTML = emptyState("No log messages yet.");
    return;
  }
  const nearBottom = container.scrollTop + container.clientHeight >= container.scrollHeight - 24;
  container.innerHTML = logs.map((entry) => `
    <div class="log-entry">
      <span>${escapeHtml(entry.ts || "")}</span>
      <span class="log-level" style="color:${escapeAttribute(entry.color || "#1f6feb")}">${escapeHtml(entry.level || "INFO")}</span>
      <span>${escapeHtml(entry.source || "system")}</span>
      <span>${escapeHtml(entry.message || "")}</span>
    </div>
  `).join("");
  if (nearBottom) {
    container.scrollTop = container.scrollHeight;
  }
}

async function startArmVoiceTest() {
  elements["arm-voice-detail"].textContent = "Starting arm voice test...";
  try {
    await postJson("/arm/voice", { action: "start", mode: "local_fake" });
  } catch (error) {
    console.error("Failed to start arm voice test", error);
    setPillState(elements["arm-voice-pill"], "status-error", "Error");
    elements["arm-voice-detail"].textContent = String(error.message || error);
  }
}

async function stopArmVoiceTest() {
  elements["arm-voice-detail"].textContent = "Stopping arm voice test...";
  try {
    await postJson("/arm/voice", { action: "stop" });
  } catch (error) {
    console.error("Failed to stop arm voice test", error);
    setPillState(elements["arm-voice-pill"], "status-error", "Error");
    elements["arm-voice-detail"].textContent = String(error.message || error);
  }
}

async function refreshPickupDetection() {
  elements["pickup-detail"].textContent = "Refreshing VIAM pickup detection...";
  try {
    await postJson("/pickup", { action: "refresh" });
  } catch (error) {
    console.error("Failed to refresh pickup detection", error);
    setPillState(elements["pickup-pill"], "status-error", "Error");
    elements["pickup-detail"].textContent = String(error.message || error);
  }
}

async function startPickup() {
  elements["pickup-detail"].textContent = "Starting pickup sequence...";
  try {
    await postJson("/pickup", { action: "start" });
  } catch (error) {
    console.error("Failed to start pickup", error);
    setPillState(elements["pickup-pill"], "status-error", "Error");
    elements["pickup-detail"].textContent = String(error.message || error);
  }
}

async function stopPickup() {
  elements["pickup-detail"].textContent = "Stopping pickup sequence...";
  try {
    await postJson("/pickup", { action: "stop" });
  } catch (error) {
    console.error("Failed to stop pickup", error);
    setPillState(elements["pickup-pill"], "status-error", "Error");
    elements["pickup-detail"].textContent = String(error.message || error);
  }
}

function renderConfig() {
  const snapshot = dashboardState.snapshot;
  const container = elements["config-form"];
  if (!snapshot) {
    return;
  }
  const config = snapshot.config || {};
  const configMeta = snapshot.config_meta || {};
  if (elements["config-note"]) {
    elements["config-note"].textContent = configMeta.save_note || "Saving updates the dashboard snapshot only.";
  }
  const hash = JSON.stringify(config);
  if (hash === dashboardState.configRenderedHash) {
    return;
  }
  dashboardState.configRenderedHash = hash;

  const preferredOrder = ["serial", "lidar", "slam", "planner", "safety", "camera", "pickup", "ekf", "phone_imu"];
  const sections = Object.keys(config).sort((a, b) => {
    const aIdx = preferredOrder.indexOf(a);
    const bIdx = preferredOrder.indexOf(b);
    if (aIdx === -1 && bIdx === -1) {
      return a.localeCompare(b);
    }
    if (aIdx === -1) {
      return 1;
    }
    if (bIdx === -1) {
      return -1;
    }
    return aIdx - bIdx;
  });
  container.innerHTML = sections.map((sectionName) => renderConfigSection(sectionName, config[sectionName] || {})).join("");
}

function renderConfigSection(sectionName, sectionConfig) {
  const fields = Object.entries(sectionConfig).map(([key, value]) => {
    const label = `
      <div class="config-label-row">
        <div>
          <div class="config-key">${escapeHtml(key)}</div>
          <div class="config-helper">${escapeHtml(sectionName)}</div>
        </div>
      </div>
    `;
    if (typeof value === "boolean") {
      return `
        <div class="config-field" data-config-section="${escapeAttribute(sectionName)}" data-config-key="${escapeAttribute(key)}">
          ${label}
          <div class="switch-row">
            <label class="switch">
              <input type="checkbox" class="config-toggle" ${value ? "checked" : ""} />
              <span class="switch-slider"></span>
            </label>
            <span>${value ? "Enabled" : "Disabled"}</span>
          </div>
        </div>
      `;
    }
    const inputType = typeof value === "number" ? "number" : "text";
    return `
      <label class="config-field" data-config-section="${escapeAttribute(sectionName)}" data-config-key="${escapeAttribute(key)}">
        ${label}
        <input class="config-input" type="${inputType}" value="${escapeAttribute(String(value))}" ${inputType === "number" ? "step=any" : ""} />
      </label>
    `;
  }).join("");

  return `
    <section class="config-section">
      <div class="config-section-header">
        <div>
          <div class="section-title">${escapeHtml(capitalize(sectionName))}</div>
          <div class="section-subtitle">Live runtime parameters</div>
        </div>
      </div>
      <div class="config-section-body">${fields || emptyState("No fields")}</div>
    </section>
  `;
}

function renderMap() {
  const snapshot = dashboardState.snapshot;
  const canvas = elements["map-canvas"];
  if (!snapshot || !canvas) {
    return;
  }
  const ctx = canvas.getContext("2d");
  const width = canvas.clientWidth || canvas.width;
  const height = canvas.clientHeight || canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(0, 0, width, height);

  const map = snapshot.occupancy_grid || {};
  drawOccupancyGrid(ctx, canvas, map);
  drawPath(ctx, canvas, map, snapshot.path || []);
  drawLidar(ctx, canvas, map, snapshot.pose || {}, snapshot.lidar_scan || {});
  drawObstacles(ctx, canvas, map, snapshot.pose || {}, snapshot.obstacles || []);
  drawLandmarks(ctx, canvas, map, snapshot.pose || {}, snapshot.landmarks || []);
  drawGoal(ctx, canvas, map, snapshot.goal);
  drawRobot(ctx, canvas, map, snapshot.pose || {});
}

function drawOccupancyGrid(ctx, canvas, map) {
  const grid = map.occupancy || [];
  if (!grid.length || !grid[0] || !grid[0].length) {
    return;
  }
  const rows = grid.length;
  const cols = grid[0].length;
  const cellWidth = (canvas.clientWidth || canvas.width) / cols;
  const cellHeight = (canvas.clientHeight || canvas.height) / rows;

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const value = Number(grid[row][col]);
      const shade = Math.max(110, Math.min(248, Math.round(252 - value * 160)));
      ctx.fillStyle = `rgb(${shade}, ${shade}, ${shade})`;
      ctx.fillRect(col * cellWidth, row * cellHeight, cellWidth + 1, cellHeight + 1);
    }
  }
}

function drawPath(ctx, canvas, map, path) {
  if (!path.length) {
    return;
  }
  ctx.save();
  ctx.strokeStyle = "#1f6feb";
  ctx.lineWidth = 3;
  ctx.beginPath();
  path.forEach((point, index) => {
    const pixel = worldToCanvas(point.x, point.y, map, canvas);
    if (index === 0) {
      ctx.moveTo(pixel.x, pixel.y);
    } else {
      ctx.lineTo(pixel.x, pixel.y);
    }
  });
  ctx.stroke();
  ctx.restore();
}

function drawLidar(ctx, canvas, map, pose, lidarScan) {
  const points = (lidarScan && lidarScan.xy_points) || [];
  if (!points.length) {
    return;
  }
  ctx.save();
  ctx.fillStyle = "rgba(40, 167, 69, 0.9)";
  points.forEach((point) => {
    const worldPoint = robotToWorld(point.x, point.y, pose);
    const pixel = worldToCanvas(worldPoint.x, worldPoint.y, map, canvas);
    ctx.beginPath();
    ctx.arc(pixel.x, pixel.y, 2, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.restore();
}

function drawObstacles(ctx, canvas, map, pose, obstacles) {
  if (!obstacles.length) {
    return;
  }
  ctx.save();
  ctx.strokeStyle = "#d1242f";
  ctx.lineWidth = 2;
  ctx.fillStyle = "rgba(209, 36, 47, 0.08)";
  obstacles.forEach((obstacle) => {
    const worldPoint = robotToWorld(obstacle.x, obstacle.y, pose);
    const pixel = worldToCanvas(worldPoint.x, worldPoint.y, map, canvas);
    const radiusPx = Math.max(6, obstacle.radius / Math.max(map.resolution_m || 0.1, 0.01));
    ctx.beginPath();
    ctx.arc(pixel.x, pixel.y, radiusPx, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  });
  ctx.restore();
}

function drawLandmarks(ctx, canvas, map, pose, landmarks) {
  if (!landmarks.length) {
    return;
  }
  ctx.save();
  ctx.fillStyle = "#111111";
  ctx.strokeStyle = "#111111";
  ctx.font = "12px Inter, system-ui, sans-serif";
  landmarks.forEach((landmark) => {
    const point = landmark.frame === "robot"
      ? robotToWorld(landmark.x, landmark.y, pose)
      : { x: landmark.x, y: landmark.y };
    const pixel = worldToCanvas(point.x, point.y, map, canvas);
    ctx.beginPath();
    ctx.moveTo(pixel.x, pixel.y - 10);
    ctx.lineTo(pixel.x - 7, pixel.y + 6);
    ctx.lineTo(pixel.x + 7, pixel.y + 6);
    ctx.closePath();
    ctx.fill();
    ctx.fillText(landmark.name || "landmark", pixel.x + 10, pixel.y + 2);
  });
  ctx.restore();
}

function drawGoal(ctx, canvas, map, goal) {
  if (!goal) {
    return;
  }
  const pixel = worldToCanvas(goal.x, goal.y, map, canvas);
  ctx.save();
  ctx.strokeStyle = "#1f6feb";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(pixel.x, pixel.y, 10, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(pixel.x - 12, pixel.y);
  ctx.lineTo(pixel.x + 12, pixel.y);
  ctx.moveTo(pixel.x, pixel.y - 12);
  ctx.lineTo(pixel.x, pixel.y + 12);
  ctx.stroke();
  ctx.restore();
}

function drawRobot(ctx, canvas, map, pose) {
  const pixel = worldToCanvas(pose.x || 0, pose.y || 0, map, canvas);
  const theta = Number(pose.theta || 0);
  ctx.save();
  ctx.translate(pixel.x, pixel.y);
  ctx.rotate(-theta);
  ctx.fillStyle = "#111111";
  ctx.beginPath();
  ctx.moveTo(16, 0);
  ctx.lineTo(-12, -10);
  ctx.lineTo(-6, 0);
  ctx.lineTo(-12, 10);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function setupMapInteractions() {
  const canvas = elements["map-canvas"];
  canvas.addEventListener("mousemove", (event) => {
    const snapshot = dashboardState.snapshot;
    if (!snapshot) {
      return;
    }
    const map = snapshot.occupancy_grid || {};
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const world = canvasToWorld(x, y, map, canvas);
    elements["map-cursor-coords"].textContent = `x ${formatNumber(world.x)} | y ${formatNumber(world.y)}`;
  });

  canvas.addEventListener("click", async (event) => {
    const snapshot = dashboardState.snapshot;
    if (!snapshot) {
      return;
    }
    const map = snapshot.occupancy_grid || {};
    const rect = canvas.getBoundingClientRect();
    const world = canvasToWorld(event.clientX - rect.left, event.clientY - rect.top, map, canvas);
    if (dashboardState.mapMode === "goal") {
      await postJson("/command", { cmd: "goto", x: world.x, y: world.y, label: "map goal" });
    } else {
      const name = window.prompt("Landmark name");
      if (!name) {
        return;
      }
      await postJson("/landmark", { name, x: world.x, y: world.y, frame: "world", source: "map", descriptor_status: "pending_capture" });
    }
  });

  updateMapModeLabel();
}

function renderCameraOverlay() {
  const snapshot = dashboardState.snapshot;
  const overlay = elements["camera-overlay"];
  const image = elements["camera-stream"];
  if (!snapshot || !overlay || !image) {
    return;
  }
  const width = image.clientWidth || 640;
  const height = image.clientHeight || 360;
  const ratio = window.devicePixelRatio || 1;
  overlay.width = Math.max(1, Math.floor(width * ratio));
  overlay.height = Math.max(1, Math.floor(height * ratio));
  overlay.style.width = `${width}px`;
  overlay.style.height = `${height}px`;
  const ctx = overlay.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = "#28a745";
  ctx.fillStyle = "#28a745";
  ctx.lineWidth = 2;
  ctx.font = "12px Inter, system-ui, sans-serif";

  const obstacles = snapshot.obstacles || [];
  obstacles.forEach((obstacle, index) => {
    const distance = Math.max(0.3, Number(obstacle.distance_m || Math.hypot(obstacle.x, obstacle.y)));
    const angle = Number(obstacle.angle_deg || (Math.atan2(obstacle.y, obstacle.x) * 180 / Math.PI));
    const centerX = (0.5 - (angle / 60)) * width;
    const boxWidth = Math.max(28, 140 / distance);
    const boxHeight = Math.max(40, 180 / distance);
    const x = clamp(centerX - boxWidth / 2, 0, width - boxWidth);
    const y = clamp(height * 0.68 - boxHeight + index * 4, 20, height - boxHeight - 8);
    ctx.strokeRect(x, y, boxWidth, boxHeight);
    ctx.fillText(`${formatNumber(distance)}m`, x, Math.max(14, y - 6));
  });

  const pickup = snapshot.pickup || {};
  const detection = pickup.last_detection || null;
  const target = pickup.last_target || null;
  if (!detection || !detection.image_width || !detection.image_height) {
    return;
  }

  const scaleX = width / Number(detection.image_width || width);
  const scaleY = height / Number(detection.image_height || height);
  const x = Number(detection.x_min || 0) * scaleX;
  const y = Number(detection.y_min || 0) * scaleY;
  const boxWidth = Math.max(1, Number((detection.x_max || 0) - (detection.x_min || 0)) * scaleX);
  const boxHeight = Math.max(1, Number((detection.y_max || 0) - (detection.y_min || 0)) * scaleY);
  const centerX = x + boxWidth / 2;
  const centerY = y + boxHeight / 2;
  const ready = pickup.status === "ready" || pickup.status === "success";
  const stroke = ready ? "#1f6feb" : "#f59e0b";

  ctx.save();
  ctx.strokeStyle = stroke;
  ctx.fillStyle = stroke;
  ctx.lineWidth = 3;
  ctx.strokeRect(x, y, boxWidth, boxHeight);
  ctx.beginPath();
  ctx.arc(centerX, centerY, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(centerX - 12, centerY);
  ctx.lineTo(centerX + 12, centerY);
  ctx.moveTo(centerX, centerY - 12);
  ctx.lineTo(centerX, centerY + 12);
  ctx.stroke();
  const label = detection.label || "ball";
  const confidence = `${Math.round((Number(detection.confidence) || 0) * 100)}%`;
  const rangeText = target && target.range_m != null ? `${formatNumber(target.range_m)}m` : "range n/a";
  ctx.fillText(`${label} ${confidence} | ${rangeText}`, x, Math.max(16, y - 8));
  ctx.restore();
}

function setupJoystick() {
  const pad = elements["joystick-pad"];
  const knob = elements["joystick-knob"];

  const updateJoystick = (clientX, clientY) => {
    const rect = pad.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    const radius = rect.width / 2 - 32;
    let dx = clientX - centerX;
    let dy = clientY - centerY;
    const distance = Math.hypot(dx, dy);
    if (distance > radius) {
      dx = (dx / distance) * radius;
      dy = (dy / distance) * radius;
    }
    knob.style.transform = `translate(${dx}px, ${dy}px)`;
    dashboardState.joystick.vx = Number((-dy / radius) * 0.5);
    dashboardState.joystick.vy = Number((dx / radius) * 0.25);
    elements["joystick-readout"].textContent = `vx ${formatNumber(dashboardState.joystick.vx)} | vy ${formatNumber(dashboardState.joystick.vy)}`;
  };

  const resetJoystick = () => {
    dashboardState.joystick.active = false;
    dashboardState.joystick.pointerId = null;
    dashboardState.joystick.vx = 0;
    dashboardState.joystick.vy = 0;
    knob.style.transform = "translate(0px, 0px)";
    elements["joystick-readout"].textContent = "vx 0.00 | vy 0.00";
    stopJoystickPulse();
    postJson("/command", { cmd: "teleop", vx: 0, vy: 0, omega: 0 });
  };

  pad.addEventListener("pointerdown", (event) => {
    dashboardState.joystick.active = true;
    dashboardState.joystick.pointerId = event.pointerId;
    pad.setPointerCapture(event.pointerId);
    updateJoystick(event.clientX, event.clientY);
    startJoystickPulse();
  });

  pad.addEventListener("pointermove", (event) => {
    if (!dashboardState.joystick.active || dashboardState.joystick.pointerId !== event.pointerId) {
      return;
    }
    updateJoystick(event.clientX, event.clientY);
  });

  ["pointerup", "pointercancel", "lostpointercapture"].forEach((eventName) => {
    pad.addEventListener(eventName, resetJoystick);
  });
}

function startJoystickPulse() {
  stopJoystickPulse();
  sendTeleop(dashboardState.joystick.vx, dashboardState.joystick.vy, 0);
  dashboardState.joystick.interval = window.setInterval(() => {
    sendTeleop(dashboardState.joystick.vx, dashboardState.joystick.vy, 0);
  }, 120);
}

function stopJoystickPulse() {
  if (dashboardState.joystick.interval) {
    window.clearInterval(dashboardState.joystick.interval);
    dashboardState.joystick.interval = null;
  }
}

function setupKeyboardTeleop() {
  const pressed = new Set();

  const updateVector = () => {
    const vx = (pressed.has("w") ? 0.45 : 0) + (pressed.has("s") ? -0.45 : 0);
    const vy = (pressed.has("a") ? 0.22 : 0) + (pressed.has("d") ? -0.22 : 0);
    dashboardState.keyboardVector = { vx, vy, omega: 0 };
    if (vx || vy) {
      if (!dashboardState.keyboardInterval) {
        sendTeleop(vx, vy, 0);
        dashboardState.keyboardInterval = window.setInterval(() => {
          sendTeleop(dashboardState.keyboardVector.vx, dashboardState.keyboardVector.vy, 0);
        }, 120);
      }
    } else if (dashboardState.keyboardInterval) {
      window.clearInterval(dashboardState.keyboardInterval);
      dashboardState.keyboardInterval = null;
      sendTeleop(0, 0, 0);
    }
  };

  window.addEventListener("keydown", (event) => {
    const key = event.key.toLowerCase();
    if (!["w", "a", "s", "d"].includes(key)) {
      return;
    }
    pressed.add(key);
    updateVector();
  });

  window.addEventListener("keyup", (event) => {
    const key = event.key.toLowerCase();
    if (!["w", "a", "s", "d"].includes(key)) {
      return;
    }
    pressed.delete(key);
    updateVector();
  });
}

function startTeleopPulse(vector) {
  stopTeleopPulse();
  sendTeleop(vector.vx, vector.vy, vector.omega || 0);
  dashboardState.keyboardInterval = window.setInterval(() => {
    sendTeleop(vector.vx, vector.vy, vector.omega || 0);
  }, 120);
}

function stopTeleopPulse() {
  if (dashboardState.keyboardInterval) {
    window.clearInterval(dashboardState.keyboardInterval);
    dashboardState.keyboardInterval = null;
  }
  sendTeleop(0, 0, 0);
}

function teleopPreset(name) {
  const presets = {
    forward: { vx: 0.45, vy: 0, omega: 0 },
    back: { vx: -0.45, vy: 0, omega: 0 },
    left: { vx: 0, vy: 0.22, omega: 0 },
    right: { vx: 0, vy: -0.22, omega: 0 },
  };
  return presets[name] || { vx: 0, vy: 0, omega: 0 };
}

function sendTeleop(vx, vy, omega) {
  postJson("/command", { cmd: "teleop", vx, vy, omega }).catch(() => {});
}

async function refreshWaypoints() {
  try {
    const response = await fetch("/waypoints");
    const data = await response.json();
    if (!dashboardState.snapshot) {
      dashboardState.snapshot = { waypoints: data.data || {} };
    } else {
      dashboardState.snapshot.waypoints = data.data || {};
    }
    renderWaypoints();
  } catch (error) {
    console.error("Failed to load waypoints", error);
  }
}

async function refreshConfig() {
  try {
    const response = await fetch("/config");
    const data = await response.json();
    if (!dashboardState.snapshot) {
      dashboardState.snapshot = { config: data.data || {}, config_meta: data.meta || {} };
    } else {
      dashboardState.snapshot.config = data.data || {};
      dashboardState.snapshot.config_meta = data.meta || {};
    }
    dashboardState.configRenderedHash = "";
    renderConfig();
  } catch (error) {
    console.error("Failed to load config", error);
  }
}

async function saveConfig() {
  const payload = collectConfigValues();
  await postJson("/config", payload);
  dashboardState.configRenderedHash = "";
  await refreshConfig();
}

function collectConfigValues() {
  const payload = {};
  document.querySelectorAll("[data-config-section][data-config-key]").forEach((field) => {
    const section = field.dataset.configSection;
    const key = field.dataset.configKey;
    if (!payload[section]) {
      payload[section] = {};
    }
    const toggle = field.querySelector(".config-toggle");
    if (toggle) {
      payload[section][key] = toggle.checked;
      return;
    }
    const input = field.querySelector(".config-input");
    if (!input) {
      return;
    }
    payload[section][key] = input.type === "number" ? Number(input.value) : input.value;
  });
  return payload;
}

function initCameraStream() {
  elements["camera-stream"].src = `/camera?stream=${Date.now()}`;
}

function reloadFoxgloveFrame() {
  const frame = elements["foxglove-frame"];
  if (!dashboardState.foxgloveEmbedUrl) {
    return;
  }
  const reloaded = `${dashboardState.foxgloveEmbedUrl}${dashboardState.foxgloveEmbedUrl.includes("?") ? "&" : "?"}reload=${Date.now()}`;
  frame.src = reloaded;
  frame.dataset.loadedSrc = dashboardState.foxgloveEmbedUrl;
}

function openFoxgloveExternal() {
  if (!dashboardState.foxgloveEmbedUrl) {
    return;
  }
  window.open(dashboardState.foxgloveEmbedUrl, "_blank", "noopener,noreferrer");
}

function resizeCanvases() {
  resizeCanvas(elements["map-canvas"]);
  resizeOverlayCanvas(elements["camera-overlay"]);
}

function resizeCanvas(canvas) {
  if (!canvas) {
    return;
  }
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function resizeOverlayCanvas(canvas) {
  if (!canvas) {
    return;
  }
  const image = elements["camera-stream"];
  const ratio = window.devicePixelRatio || 1;
  const width = image.clientWidth || canvas.clientWidth || 320;
  const height = image.clientHeight || 180;
  canvas.width = Math.max(1, Math.floor(width * ratio));
  canvas.height = Math.max(1, Math.floor(height * ratio));
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function activateTab(tabName) {
  elements.tabButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tabName);
  });
  elements.tabContents.forEach((content) => {
    content.classList.toggle("active", content.id === `tab-${tabName}`);
  });
}

function updateMapModeLabel() {
  elements["map-mode-label"].textContent = `Click mode: ${dashboardState.mapMode}`;
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return response.json();
}

function buildFoxgloveWsUrl(foxglove) {
  const hostname = window.location.hostname || "127.0.0.1";
  const port = foxglove.port || 8765;
  return `ws://${hostname}:${port}`;
}

function buildFoxgloveEmbedUrl(foxglove, bridgeUrl) {
  const base = foxglove.viewer_url || "https://app.foxglove.dev/";
  const url = new URL(base);
  url.searchParams.set("ds", "foxglove-websocket");
  url.searchParams.set("ds.url", bridgeUrl);
  return url.toString();
}

function worldToCanvas(x, y, map, canvas) {
  const rows = map.shape ? map.shape[0] : 0;
  const cols = map.shape ? map.shape[1] : 0;
  const resolution = Number(map.resolution_m || 0.1);
  const origin = Number(map.origin_cell || 0);
  const width = canvas.clientWidth || canvas.width;
  const height = canvas.clientHeight || canvas.height;
  if (!rows || !cols) {
    return { x: width / 2, y: height / 2 };
  }
  const col = x / resolution + origin;
  const row = rows - (y / resolution + origin);
  return {
    x: (col / cols) * width,
    y: (row / rows) * height,
  };
}

function canvasToWorld(px, py, map, canvas) {
  const rows = map.shape ? map.shape[0] : 0;
  const cols = map.shape ? map.shape[1] : 0;
  const resolution = Number(map.resolution_m || 0.1);
  const origin = Number(map.origin_cell || 0);
  const width = canvas.clientWidth || canvas.width;
  const height = canvas.clientHeight || canvas.height;
  if (!rows || !cols) {
    return { x: 0, y: 0 };
  }
  const col = (px / width) * cols;
  const row = (py / height) * rows;
  return {
    x: (col - origin) * resolution,
    y: ((rows - row) - origin) * resolution,
  };
}

function robotToWorld(x, y, pose) {
  const theta = Number(pose.theta || 0);
  return {
    x: Number(pose.x || 0) + Math.cos(theta) * x - Math.sin(theta) * y,
    y: Number(pose.y || 0) + Math.sin(theta) * x + Math.cos(theta) * y,
  };
}

function setPillState(element, className, label) {
  if (!element) {
    return;
  }
  element.classList.remove("status-live", "status-offline", "status-connected", "status-disconnected", "status-error", "status-warn");
  element.classList.add(className);
  element.textContent = label;
}

function statusClassFor(status) {
  const normalized = String(status || "disconnected").toLowerCase();
  if (normalized === "connected" || normalized === "running" || normalized === "starting" || normalized === "ready" || normalized === "success") {
    return "status-connected";
  }
  if (normalized === "error") {
    return "status-error";
  }
  if (normalized === "warning" || normalized === "warn" || normalized === "searching") {
    return "status-warn";
  }
  return "status-disconnected";
}

function humanizeStatus(status) {
  const normalized = String(status || "disconnected").toLowerCase();
  if (normalized === "connected") {
    return "Connected";
  }
  if (normalized === "running") {
    return "Running";
  }
  if (normalized === "starting") {
    return "Starting";
  }
  if (normalized === "stopping") {
    return "Stopping";
  }
  if (normalized === "idle") {
    return "Idle";
  }
  if (normalized === "error") {
    return "Error";
  }
  if (normalized === "ready") {
    return "Ready";
  }
  if (normalized === "success") {
    return "Success";
  }
  if (normalized === "searching") {
    return "Searching";
  }
  if (normalized === "stopped") {
    return "Stopped";
  }
  if (normalized === "warning" || normalized === "warn") {
    return "Warning";
  }
  return "Disconnected";
}

function formatNumber(value, digits = 2) {
  return Number(value || 0).toFixed(digits);
}

function capitalize(value) {
  return value ? value.charAt(0).toUpperCase() + value.slice(1) : "";
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function emptyState(text) {
  return `<div class="empty-state">${escapeHtml(text)}</div>`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function escapeAttribute(value) {
  return escapeHtml(value);
}

const MAX_LOGS = 250;
