const MAX_LOGS = 120;
const MAP_EMPTY_SHADE = 228;
const MAP_OBSTACLE_SHADE = 72;

const dashboardState = {
  snapshot: null,
  ws: null,
  reconnectTimer: null,
  cameraRetryTimer: null,
  mapMode: "goal",
  selectedWaypoint: "",
};

const elements = {};

document.addEventListener("DOMContentLoaded", () => {
  cacheElements();
  bindEvents();
  initCameraStream();
  connectWebSocket();
  refreshWaypoints();
  resizeCanvases();
  renderAll();
  window.addEventListener("resize", () => {
    resizeCanvases();
    renderAll();
  });
});

function cacheElements() {
  [
    "robot-name",
    "connection-pill",
    "snapshot-rate",
    "nav-state-summary",
    "nav-mode-summary",
    "goal-summary",
    "object-target-label",
    "object-target-detail",
    "listen-target-button",
    "set-target-button",
    "clear-target-button",
    "planner-banner",
    "waypoint-select",
    "goto-button",
    "reload-waypoints-button",
    "stop-button",
    "map-mode-chip",
    "map-mode-goal-button",
    "map-mode-waypoint-button",
    "map-canvas",
    "map-cursor-coords",
    "camera-stream",
    "lidar-canvas",
    "lidar-pill",
    "lidar-points",
    "lidar-obstacles",
    "lidar-map-shape",
    "lidar-detail",
    "pid-command",
    "pid-pose",
    "pid-estop",
    "pid-note",
    "imu-pill",
    "imu-status",
    "imu-gyro",
    "imu-accel",
    "imu-age",
    "imu-note",
    "arm-voice-pill",
    "arm-voice-detail",
    "arm-voice-start-button",
    "arm-voice-stop-button",
    "pickup-pill",
    "pickup-detail",
    "pickup-refresh-button",
    "pickup-start-button",
    "pickup-stop-button",
    "subsystems-list",
    "log-viewer",
  ].forEach((id) => {
    elements[id] = document.getElementById(id);
  });
}

function bindEvents() {
  elements["goto-button"].addEventListener("click", async () => {
    const waypoint = dashboardState.selectedWaypoint || elements["waypoint-select"].value;
    if (!waypoint) {
      return;
    }
    dashboardState.selectedWaypoint = waypoint;
    await postJson("/command", { cmd: "goto", waypoint });
  });

  elements["waypoint-select"].addEventListener("change", () => {
    dashboardState.selectedWaypoint = elements["waypoint-select"].value || "";
  });

  elements["reload-waypoints-button"].addEventListener("click", () => {
    refreshWaypoints();
  });

  elements["listen-target-button"].addEventListener("click", () => {
    listenForTarget();
  });

  elements["set-target-button"].addEventListener("click", () => {
    setTargetLabel();
  });

  elements["clear-target-button"].addEventListener("click", () => {
    clearTargetLabel();
  });

  elements["map-mode-goal-button"].addEventListener("click", () => {
    setMapMode("goal");
  });

  elements["map-mode-waypoint-button"].addEventListener("click", () => {
    setMapMode("waypoint");
  });

  elements["stop-button"].addEventListener("click", () => {
    postJson("/command", { cmd: "stop" }).catch(console.error);
  });

  elements["arm-voice-start-button"].addEventListener("click", () => {
    startArmVoiceTest();
  });
  elements["arm-voice-stop-button"].addEventListener("click", () => {
    stopArmVoiceTest();
  });

  elements["pickup-refresh-button"].addEventListener("click", () => {
    refreshPickupDetection();
  });
  elements["pickup-start-button"].addEventListener("click", () => {
    startPickup();
  });
  elements["pickup-stop-button"].addEventListener("click", () => {
    stopPickup();
  });

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
    restartCameraStream();
  });

  ws.addEventListener("message", (event) => {
    try {
      const message = JSON.parse(event.data);
      if (message.type !== "snapshot") {
        return;
      }
      dashboardState.snapshot = message.data;
      renderAll();
    } catch (error) {
      console.error("Invalid WebSocket payload", error);
    }
  });

  ws.addEventListener("close", () => {
    scheduleCameraStreamRetry(1500);
    dashboardState.reconnectTimer = window.setTimeout(connectWebSocket, 1500);
  });
}

function renderAll() {
  renderMeta();
  renderSummary();
  renderPlannerBanner();
  renderWaypoints();
  renderTargetCard();
  renderMapModeControls();
  renderLidarCard();
  renderPidCard();
  renderImuCard();
  renderArmCard();
  renderSystems();
  renderLogs();
  renderMap();
  renderLidarCanvas();
}

function renderMeta() {
  const snapshot = dashboardState.snapshot;
  if (!snapshot) {
    return;
  }
  const meta = snapshot.meta || {};
  elements["robot-name"].textContent = meta.robot_name || "Wheelchair Dashboard";
  elements["snapshot-rate"].textContent = `${formatNumber(meta.snapshot_rate_hz || 0, 0)} Hz`;
  const live = Boolean(meta.live);
  const label = meta.connection_label || (live ? "Live" : "Offline");
  setPillState(elements["connection-pill"], live ? "status-live" : "status-offline", label);
}

function renderSummary() {
  const snapshot = dashboardState.snapshot;
  if (!snapshot) {
    return;
  }
  const nav = snapshot.nav_state || {};
  const goal = snapshot.goal || null;
  elements["nav-state-summary"].textContent = nav.state || "Idle";
  elements["nav-mode-summary"].textContent = `${nav.mode || "manual"} · E-stop ${nav.estop ? "ACTIVE" : "clear"}`;
  elements["goal-summary"].textContent = goal
    ? `${goal.label ? `${goal.label} · ` : ""}${formatNumber(goal.x)}, ${formatNumber(goal.y)}`
    : "No goal";
}

function renderPlannerBanner() {
  const snapshot = dashboardState.snapshot;
  const banner = elements["planner-banner"];
  if (!snapshot || !banner) {
    return;
  }

  const nav = snapshot.nav_state || {};
  const logs = snapshot.logs || [];
  const goal = snapshot.goal || null;
  const path = snapshot.path || [];
  const latest = [...logs].reverse().find((entry) => ["planner", "safety"].includes(String(entry.source || "")));
  let tone = "idle";
  let text = "Idle · click the map or press Go.";

  if (nav.estop) {
    tone = "error";
    text = "E-stop active. Commands are blocked.";
  } else if (latest) {
    const message = String(latest.message || "");
    const lower = message.toLowerCase();
    if (lower.includes("goal cell is blocked")) {
      tone = "warn";
      text = "Goal blocked by inflated costmap.";
    } else if (lower.includes("could not find a path") || lower.includes("no path")) {
      tone = "warn";
      text = "No path found from the current local map.";
    } else if (lower.includes("arrived")) {
      tone = "success";
      text = "Arrived at goal.";
    } else if (lower.includes("emergency stop")) {
      tone = "error";
      text = message;
    } else if (lower.includes("passed tracked obstacle cluster") || lower.includes("refresh")) {
      tone = "active";
      text = "Refreshing scan after obstacle pass or obstruction.";
    } else if (lower.includes("planned")) {
      tone = "success";
      text = message;
    } else if (lower.includes("preview-only mode")) {
      tone = "success";
      text = "Path preview ready. No motion commands are being sent.";
    } else if (lower.includes("goal received")) {
      tone = "active";
      text = "Goal accepted. Planning path...";
    } else if (lower.includes("scanning")) {
      tone = "active";
      text = message;
    }
  }

  if (nav.state === "Planning") {
    tone = "active";
    text = "Planning A* path...";
  } else if (nav.state === "Target Search") {
    tone = "active";
    text = "Rotating to center the visible target in the camera box.";
  } else if (nav.state === "Path Ready") {
    tone = "success";
    text = path.length > 1
      ? `Path preview ready · ${path.length} waypoints`
      : "Path preview ready.";
  } else if (nav.state === "Moving") {
    tone = "success";
    text = path.length > 1
      ? `Following path · ${path.length} waypoints`
      : "Moving toward goal.";
  } else if (nav.state && nav.state.toLowerCase().includes("scanning")) {
    tone = "active";
    text = `${nav.state}. Building local map...`;
  } else if (goal && path.length > 1) {
    tone = "success";
    text = `Path ready · ${path.length} waypoints`;
  } else if (goal && tone === "idle") {
    tone = "active";
    text = `Goal set · ${formatNumber(goal.x)}, ${formatNumber(goal.y)}`;
  }

  banner.className = `planner-banner planner-banner--${tone}`;
  banner.textContent = text;
}

function renderWaypoints() {
  const snapshot = dashboardState.snapshot;
  const select = elements["waypoint-select"];
  if (!snapshot || !select) {
    return;
  }
  const waypoints = snapshot.waypoints || {};
  const current = dashboardState.selectedWaypoint || select.value;
  const entries = Object.entries(waypoints).sort(([left], [right]) => left.localeCompare(right));
  if (!entries.length) {
    select.innerHTML = `<option value="">No waypoints</option>`;
    select.disabled = true;
    elements["goto-button"].disabled = true;
    dashboardState.selectedWaypoint = "";
    return;
  }

  select.disabled = false;
  elements["goto-button"].disabled = false;
  select.innerHTML = entries.map(([name, coords]) =>
    `<option value="${escapeAttribute(name)}">${escapeHtml(name)} (${formatNumber(coords[0])}, ${formatNumber(coords[1])})</option>`,
  ).join("");
  if (current && waypoints[current]) {
    select.value = current;
  } else {
    select.value = entries[0][0];
  }
  dashboardState.selectedWaypoint = select.value || "";
}

function renderTargetCard() {
  const snapshot = dashboardState.snapshot;
  if (!snapshot) {
    return;
  }

  const objectTarget = snapshot.object_target || {};
  const label = String(objectTarget.label || "").trim();
  const detection = objectTarget.detection || null;
  const status = String(objectTarget.status || "idle");
  const listening = Boolean(objectTarget.listening);
  let detail = objectTarget.detail || "Click Listen or Set to choose an object.";

  if (detection) {
    const angle = Number(detection.angle_deg || 0);
    detail = `${detection.label} ${formatNumber(detection.confidence)} at ${formatNumber(angle, 0)} deg`;
  } else if (objectTarget.transcript) {
    detail = `${detail} Heard: "${objectTarget.transcript}"`;
  }
  if (objectTarget.waypoint_name) {
    detail = `${detail} Saved waypoint: ${objectTarget.waypoint_name}`;
  }

  elements["object-target-label"].textContent = label || "No target";
  elements["object-target-detail"].textContent = detail;
  elements["listen-target-button"].disabled = listening;
  elements["set-target-button"].disabled = listening;
  elements["clear-target-button"].disabled = !label && status === "idle";
}

function renderMapModeControls() {
  const chip = elements["map-mode-chip"];
  const goalButton = elements["map-mode-goal-button"];
  const waypointButton = elements["map-mode-waypoint-button"];
  const isWaypoint = dashboardState.mapMode === "waypoint";

  if (chip) {
    chip.textContent = isWaypoint ? "Waypoint" : "Goal";
    chip.className = `status-pill ${isWaypoint ? "status-warn" : "status-connected"}`;
  }
  if (goalButton) {
    goalButton.classList.toggle("button-active", !isWaypoint);
  }
  if (waypointButton) {
    waypointButton.classList.toggle("button-active", isWaypoint);
  }
}

function renderLidarCard() {
  const snapshot = dashboardState.snapshot;
  if (!snapshot) {
    return;
  }

  const lidarSubsystem = (snapshot.subsystems || {})["LiDAR"] || {};
  const lidarScan = snapshot.lidar_scan || {};
  const map = snapshot.occupancy_grid || {};
  const obstacles = snapshot.obstacles || [];
  const points = lidarScan.xy_points || [];

  setPillState(
    elements["lidar-pill"],
    statusClassFor(lidarSubsystem.status || "disconnected"),
    humanizeStatus(lidarSubsystem.status || "disconnected"),
  );
  elements["lidar-points"].textContent = String(points.length);
  elements["lidar-obstacles"].textContent = String(obstacles.length);
  elements["lidar-map-shape"].textContent = map.shape ? `${map.shape[0]} x ${map.shape[1]}` : "0 x 0";
  elements["lidar-detail"].textContent = lidarSubsystem.detail || "Waiting for scan data.";
}

function renderPidCard() {
  const snapshot = dashboardState.snapshot;
  if (!snapshot) {
    return;
  }
  const velocity = snapshot.velocity || {};
  const pose = snapshot.pose || {};
  const nav = snapshot.nav_state || {};

  elements["pid-command"].textContent = `${formatNumber(velocity.vx)}, ${formatNumber(velocity.vy)}, ${formatNumber(velocity.omega)}`;
  elements["pid-pose"].textContent = `${formatNumber(pose.x)}, ${formatNumber(pose.y)}, ${formatNumber(pose.theta)}`;
  elements["pid-estop"].textContent = nav.estop ? "ACTIVE" : "Clear";
  elements["pid-note"].textContent = nav.estop
    ? "E-stop is active. PID slot reserved for wheel tuning and encoder feedback."
    : "Slot reserved for wheel PID, encoder feedback, and tuning plots.";
}

function renderImuCard() {
  const snapshot = dashboardState.snapshot;
  if (!snapshot) {
    return;
  }

  const subsystem = (snapshot.subsystems || {})["IMU"] || {};
  const imu = snapshot.imu || {};
  const receiving = Boolean(imu.receiving);
  const age = imu.packet_age_s;

  setPillState(
    elements["imu-pill"],
    statusClassFor(subsystem.status || (receiving ? "connected" : "disconnected")),
    humanizeStatus(subsystem.status || (receiving ? "connected" : "disconnected")),
  );

  elements["imu-status"].textContent = receiving ? "Receiving packets" : "Waiting for packets";
  elements["imu-gyro"].textContent = `${formatNumber(imu.gyro_x)}, ${formatNumber(imu.gyro_y)}, ${formatNumber(imu.gyro_z)}`;
  elements["imu-accel"].textContent = `${formatNumber(imu.accel_x)}, ${formatNumber(imu.accel_y)}, ${formatNumber(imu.accel_z)}`;
  elements["imu-age"].textContent = age == null ? "n/a" : `${formatNumber(age, 2)} s`;
  elements["imu-note"].textContent = subsystem.detail || "Waiting for HyperIMU UDP on port 5555.";
}

function renderArmCard() {
  const snapshot = dashboardState.snapshot;
  if (!snapshot) {
    return;
  }

  const armVoice = snapshot.arm_voice || {};
  const armStatus = armVoice.status || "idle";
  const armRunning = Boolean(armVoice.running);
  setPillState(elements["arm-voice-pill"], statusClassFor(armStatus), humanizeStatus(armStatus));
  elements["arm-voice-detail"].textContent = armVoice.detail || "Arm voice idle.";
  elements["arm-voice-start-button"].disabled = armRunning || armStatus === "starting";
  elements["arm-voice-stop-button"].disabled = !(armRunning || armStatus === "starting" || armStatus === "stopping");

  const pickup = snapshot.pickup || {};
  const pickupStatus = pickup.status || "idle";
  const pickupRunning = Boolean(pickup.running);
  setPillState(elements["pickup-pill"], statusClassFor(pickupStatus), humanizeStatus(pickupStatus));
  elements["pickup-detail"].textContent = pickup.detail || "Pickup idle.";
  elements["pickup-refresh-button"].disabled = pickupRunning;
  elements["pickup-start-button"].disabled = pickupRunning;
  elements["pickup-stop-button"].disabled = !pickupRunning;
}

function renderSystems() {
  const snapshot = dashboardState.snapshot;
  const container = elements["subsystems-list"];
  if (!snapshot || !container) {
    return;
  }

  const subsystems = snapshot.subsystems || {};
  const rows = Object.entries(subsystems).map(([name, subsystem]) => `
    <div class="system-row">
      <div>
        <div class="system-name">${escapeHtml(name)}</div>
        <div class="system-detail">${escapeHtml(subsystem.detail || "")}</div>
      </div>
      <span class="status-pill ${statusClassFor(subsystem.status)}">${escapeHtml(humanizeStatus(subsystem.status))}</span>
    </div>
  `).join("");
  container.innerHTML = rows || emptyState("No subsystem data.");
}

function renderLogs() {
  const container = elements["log-viewer"];
  if (!container) {
    return;
  }
  const logs = ((dashboardState.snapshot || {}).logs || []).slice(-MAX_LOGS);
  if (!logs.length) {
    container.innerHTML = emptyState("No logs yet.");
    return;
  }
  container.innerHTML = logs.map((entry) => `
    <div class="log-entry">
      <span>${escapeHtml(entry.ts || "")}</span>
      <span style="color:${escapeAttribute(entry.color || "#60a5fa")}; font-weight:700;">${escapeHtml(entry.level || "INFO")}</span>
      <span>${escapeHtml(entry.source || "")}</span>
      <span>${escapeHtml(entry.message || "")}</span>
    </div>
  `).join("");
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
  ctx.fillStyle = `rgb(${MAP_EMPTY_SHADE}, ${MAP_EMPTY_SHADE}, ${MAP_EMPTY_SHADE})`;
  ctx.fillRect(0, 0, width, height);

  const map = snapshot.occupancy_grid || {};
  drawOccupancyGrid(ctx, canvas, map);
  drawPath(ctx, canvas, map, snapshot.path || []);
  drawLidarOnMap(ctx, canvas, map, snapshot.pose || {}, snapshot.lidar_scan || {});
  drawObstaclesOnMap(ctx, canvas, map, snapshot.obstacles || []);
  drawVoiceTargetOnMap(ctx, canvas, map, snapshot.pose || {}, snapshot.object_target || {});
  drawWaypoints(ctx, canvas, map, snapshot.waypoints || {});
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
  const shadeRange = MAP_EMPTY_SHADE - MAP_OBSTACLE_SHADE;

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const value = clamp(Number(grid[row][col]) || 0, 0, 1);
      const shade = Math.round(MAP_EMPTY_SHADE - value * shadeRange);
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
  ctx.strokeStyle = "#2563eb";
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

function drawLidarOnMap(ctx, canvas, map, pose, lidarScan) {
  const points = (lidarScan && lidarScan.xy_points) || [];
  const frame = String((lidarScan && lidarScan.frame) || "robot").toLowerCase();
  if (!points.length) {
    return;
  }
  ctx.save();
  ctx.fillStyle = "rgba(16, 185, 129, 0.85)";
  points.forEach((point) => {
    const worldPoint = frame === "map"
      ? { x: Number(point.x || 0), y: Number(point.y || 0) }
      : robotToWorld(point.x, point.y, pose);
    const pixel = worldToCanvas(worldPoint.x, worldPoint.y, map, canvas);
    ctx.beginPath();
    ctx.arc(pixel.x, pixel.y, 2, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.restore();
}

function drawObstaclesOnMap(ctx, canvas, map, obstacles) {
  if (!obstacles.length) {
    return;
  }
  ctx.save();
  ctx.strokeStyle = "#b91c1c";
  ctx.fillStyle = "rgba(185, 28, 28, 0.16)";
  obstacles.forEach((obstacle) => {
    const pixel = worldToCanvas(obstacle.x, obstacle.y, map, canvas);
    const radiusPx = Math.max(6, Number(obstacle.radius || 0.2) / Math.max(Number(map.resolution_m || 0.1), 0.01));
    ctx.beginPath();
    ctx.arc(pixel.x, pixel.y, radiusPx, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  });
  ctx.restore();
}

function drawWaypoints(ctx, canvas, map, waypoints) {
  const entries = Object.entries(waypoints || {});
  if (!entries.length) {
    return;
  }

  ctx.save();
  ctx.font = "12px Inter, system-ui, sans-serif";
  ctx.textBaseline = "top";
  entries.forEach(([name, coords]) => {
    const pixel = worldToCanvas(coords[0], coords[1], map, canvas);
    ctx.fillStyle = "#7c3aed";
    ctx.beginPath();
    ctx.arc(pixel.x, pixel.y, 6, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "rgba(124, 58, 237, 0.14)";
    ctx.fillRect(pixel.x + 10, pixel.y - 2, Math.max(54, name.length * 7), 18);
    ctx.fillStyle = "#ede9fe";
    ctx.fillText(name, pixel.x + 14, pixel.y + 1);
  });
  ctx.restore();
}

function drawVoiceTargetOnMap(ctx, canvas, map, pose, objectTarget) {
  const target = objectTarget && objectTarget.lidar_target;
  if (!target) {
    return;
  }

  const frame = String(target.frame || "map").toLowerCase();
  const worldPoint = frame === "map"
    ? { x: Number(target.x || 0), y: Number(target.y || 0) }
    : robotToWorld(target.x, target.y, pose);
  const pixel = worldToCanvas(worldPoint.x, worldPoint.y, map, canvas);
  const radiusPx = Math.max(8, Number(target.radius || 0.2) / Math.max(Number(map.resolution_m || 0.1), 0.01));

  ctx.save();
  ctx.strokeStyle = "#d946ef";
  ctx.fillStyle = "rgba(217, 70, 239, 0.18)";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(pixel.x, pixel.y, radiusPx, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(pixel.x, pixel.y, radiusPx + 10, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

function drawGoal(ctx, canvas, map, goal) {
  if (!goal) {
    return;
  }
  const pixel = worldToCanvas(goal.x, goal.y, map, canvas);
  ctx.save();
  ctx.strokeStyle = "#0ea5e9";
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
  ctx.fillStyle = "#111827";
  ctx.beginPath();
  ctx.moveTo(16, 0);
  ctx.lineTo(-12, -10);
  ctx.lineTo(-6, 0);
  ctx.lineTo(-12, 10);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function renderLidarCanvas() {
  const snapshot = dashboardState.snapshot;
  const canvas = elements["lidar-canvas"];
  if (!snapshot || !canvas) {
    return;
  }

  const ctx = canvas.getContext("2d");
  const width = canvas.clientWidth || canvas.width;
  const height = canvas.clientHeight || canvas.height;
  const centerX = width / 2;
  const centerY = height / 2;
  const lidarScan = snapshot.lidar_scan || {};
  const points = (lidarScan.xy_points || []);
  const frame = String(lidarScan.frame || "robot").toLowerCase();
  const pose = snapshot.pose || {};
  const map = snapshot.occupancy_grid || {};
  const halfSpanM = Math.max(3, ((Number(map.grid_size || 120) * Number(map.resolution_m || 0.1)) / 2));
  const scale = (Math.min(width, height) * 0.42) / halfSpanM;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#0e1116";
  ctx.fillRect(0, 0, width, height);

  ctx.save();
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.lineWidth = 1;
  for (let ring = 1; ring <= 3; ring += 1) {
    ctx.beginPath();
    ctx.arc(centerX, centerY, ring * (Math.min(width, height) * 0.12), 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.beginPath();
  ctx.moveTo(centerX, 0);
  ctx.lineTo(centerX, height);
  ctx.moveTo(0, centerY);
  ctx.lineTo(width, centerY);
  ctx.stroke();
  ctx.restore();

  ctx.save();
  ctx.fillStyle = "#34d399";
  points.forEach((point) => {
    const pointXY = frame === "map"
      ? worldToRobot(point.x, point.y, pose)
      : { x: Number(point.x || 0), y: Number(point.y || 0) };
    const px = centerX + Number(pointXY.y || 0) * scale;
    const py = centerY - Number(pointXY.x || 0) * scale;
    ctx.beginPath();
    ctx.arc(px, py, 1.8, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.restore();

  ctx.save();
  ctx.translate(centerX, centerY);
  ctx.fillStyle = "#e5e7eb";
  ctx.beginPath();
  ctx.moveTo(0, -12);
  ctx.lineTo(-8, 10);
  ctx.lineTo(0, 5);
  ctx.lineTo(8, 10);
  ctx.closePath();
  ctx.fill();
  ctx.restore();

  drawVoiceTargetOnLidarCanvas(ctx, centerX, centerY, scale, pose, snapshot.object_target || {});
}

function drawVoiceTargetOnLidarCanvas(ctx, centerX, centerY, scale, pose, objectTarget) {
  const target = objectTarget && objectTarget.lidar_target;
  if (!target) {
    return;
  }

  const frame = String(target.frame || "map").toLowerCase();
  const pointXY = frame === "map"
    ? worldToRobot(target.x, target.y, pose)
    : { x: Number(target.x || 0), y: Number(target.y || 0) };
  const px = centerX + Number(pointXY.y || 0) * scale;
  const py = centerY - Number(pointXY.x || 0) * scale;

  ctx.save();
  ctx.strokeStyle = "#d946ef";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(px, py, 9, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(px - 12, py);
  ctx.lineTo(px + 12, py);
  ctx.moveTo(px, py - 12);
  ctx.lineTo(px, py + 12);
  ctx.stroke();
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
    const world = canvasToWorld(event.clientX - rect.left, event.clientY - rect.top, map, canvas);
    elements["map-cursor-coords"].textContent = `x ${formatNumber(world.x)} · y ${formatNumber(world.y)}`;
  });

  canvas.addEventListener("click", async (event) => {
    const snapshot = dashboardState.snapshot;
    if (!snapshot) {
      return;
    }
    const map = snapshot.occupancy_grid || {};
    const rect = canvas.getBoundingClientRect();
    const world = canvasToWorld(event.clientX - rect.left, event.clientY - rect.top, map, canvas);
    if (dashboardState.mapMode === "waypoint") {
      const suggested = `wp-${Date.now().toString().slice(-4)}`;
      const name = window.prompt("Waypoint name?", suggested);
      if (!name || !name.trim()) {
        return;
      }
      const savedName = name.trim();
      dashboardState.selectedWaypoint = savedName;
      await postJson("/waypoints", { name: savedName, x: world.x, y: world.y });
      await refreshWaypoints(savedName);
      setMapMode("goal");
      return;
    }
    await postJson("/command", { cmd: "goto", x: world.x, y: world.y, label: "map goal" });
  });
}

function initCameraStream() {
  const img = elements["camera-stream"];
  if (!img) {
    return;
  }
  img.addEventListener("error", () => {
    scheduleCameraStreamRetry(1000);
  });
  img.addEventListener("load", () => {
    if (dashboardState.cameraRetryTimer) {
      window.clearTimeout(dashboardState.cameraRetryTimer);
      dashboardState.cameraRetryTimer = null;
    }
  });
  restartCameraStream();
}

function restartCameraStream() {
  const img = elements["camera-stream"];
  if (!img) {
    return;
  }
  img.src = `/camera?stream=${Date.now()}`;
}

function scheduleCameraStreamRetry(delayMs = 1000) {
  if (dashboardState.cameraRetryTimer) {
    window.clearTimeout(dashboardState.cameraRetryTimer);
  }
  dashboardState.cameraRetryTimer = window.setTimeout(() => {
    dashboardState.cameraRetryTimer = null;
    restartCameraStream();
  }, delayMs);
}

function resizeCanvases() {
  resizeCanvas(elements["map-canvas"]);
  resizeCanvas(elements["lidar-canvas"]);
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

async function refreshWaypoints(preferredWaypoint = "") {
  if (preferredWaypoint) {
    dashboardState.selectedWaypoint = String(preferredWaypoint).trim();
  }
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

async function listenForTarget() {
  elements["object-target-detail"].textContent = "Listening for target object...";
  try {
    await postJson("/target-object", { action: "listen" });
  } catch (error) {
    elements["object-target-detail"].textContent = String(error.message || error);
  }
}

async function setTargetLabel() {
  const current = (((dashboardState.snapshot || {}).object_target || {}).label || "").trim();
  const label = window.prompt("Object to detect?", current || "chair");
  if (!label || !label.trim()) {
    return;
  }
  try {
    await postJson("/target-object", { action: "set", label: label.trim() });
  } catch (error) {
    elements["object-target-detail"].textContent = String(error.message || error);
  }
}

async function clearTargetLabel() {
  try {
    await postJson("/target-object", { action: "clear" });
  } catch (error) {
    elements["object-target-detail"].textContent = String(error.message || error);
  }
}

function setMapMode(mode) {
  dashboardState.mapMode = mode === "waypoint" ? "waypoint" : "goal";
  renderMapModeControls();
}

async function startArmVoiceTest() {
  elements["arm-voice-detail"].textContent = "Starting...";
  try {
    await postJson("/arm/voice", { action: "start", mode: "local_fake" });
  } catch (error) {
    elements["arm-voice-detail"].textContent = String(error.message || error);
  }
}

async function stopArmVoiceTest() {
  elements["arm-voice-detail"].textContent = "Stopping...";
  try {
    await postJson("/arm/voice", { action: "stop" });
  } catch (error) {
    elements["arm-voice-detail"].textContent = String(error.message || error);
  }
}

async function refreshPickupDetection() {
  elements["pickup-detail"].textContent = "Refreshing...";
  try {
    await postJson("/pickup", { action: "refresh" });
  } catch (error) {
    elements["pickup-detail"].textContent = String(error.message || error);
  }
}

async function startPickup() {
  elements["pickup-detail"].textContent = "Starting...";
  try {
    await postJson("/pickup", { action: "start" });
  } catch (error) {
    elements["pickup-detail"].textContent = String(error.message || error);
  }
}

async function stopPickup() {
  elements["pickup-detail"].textContent = "Stopping...";
  try {
    await postJson("/pickup", { action: "stop" });
  } catch (error) {
    elements["pickup-detail"].textContent = String(error.message || error);
  }
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
  return { x: (col / cols) * width, y: (row / rows) * height };
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

function worldToRobot(x, y, pose) {
  const theta = Number(pose.theta || 0);
  const dx = Number(x || 0) - Number(pose.x || 0);
  const dy = Number(y || 0) - Number(pose.y || 0);
  return {
    x: Math.cos(theta) * dx + Math.sin(theta) * dy,
    y: -Math.sin(theta) * dx + Math.cos(theta) * dy,
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
  if (["connected", "running", "starting", "ready", "success"].includes(normalized)) {
    return "status-connected";
  }
  if (normalized === "error") {
    return "status-error";
  }
  if (["warning", "warn", "searching"].includes(normalized)) {
    return "status-warn";
  }
  return "status-disconnected";
}

function humanizeStatus(status) {
  const normalized = String(status || "disconnected").toLowerCase();
  const labels = {
    connected: "OK",
    disconnected: "Idle",
    running: "Run",
    starting: "Start",
    stopping: "Stop",
    idle: "Idle",
    error: "Err",
    ready: "Ready",
    success: "OK",
    searching: "Scan",
    warning: "Warn",
    warn: "Warn",
  };
  return labels[normalized] || "Idle";
}

function formatNumber(value, digits = 2) {
  return Number(value || 0).toFixed(digits);
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
