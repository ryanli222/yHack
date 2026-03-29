/**
 * Floating windows: drag title bars, minimize / maximize, layout persisted in localStorage.
 */

const WINDOW_IDS = ["map", "camera", "telemetry", "systems", "world", "ops", "config", "logs"];
const SECONDARY_IDS = ["systems", "world", "ops", "config", "logs"];
const STORAGE_WINDOWS = "wheelchairFloatWindows";

let windowLayout = {};
let zCounter = 30;
let dragState = null;
/** Saved normal rect when maximizing (per window id). */
const maxRestore = {};

const dashboardState = {
  snapshot: null,
  ws: null,
  reconnectTimer: null,
  mapMode: "goal",
  configRenderedHash: "",
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
const MAX_LOGS = 250;

document.addEventListener("DOMContentLoaded", () => {
  cacheElements();
  initFloatingWindows();
  bindEvents();
  connectWebSocket();
  refreshWaypoints();
  refreshConfig();
  initCameraStream();
  requestAnimationFrame(() => {
    resizeCanvases();
    renderAll();
  });
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
    "map-canvas",
    "map-mode-label",
    "map-cursor-coords",
    "camera-stream",
    "camera-overlay",
    "pose-xyz",
    "pose-vel-line",
    "nav-line",
    "nav-estop",
    "imu-gyro-z",
    "joystick-pad",
    "joystick-knob",
    "joystick-readout",
    "waypoint-select",
    "goal-summary",
    "arm-voice-pill",
    "arm-voice-detail",
    "arm-voice-start-button",
    "arm-voice-stop-button",
    "pickup-pill",
    "pickup-detail",
    "pickup-refresh-button",
    "pickup-start-button",
    "pickup-stop-button",
    "voice-transcript",
    "voice-response",
    "subsystems-tree",
    "obstacles-list",
    "landmarks-list",
    "config-form",
    "config-note",
    "log-viewer",
  ].forEach((id) => {
    elements[id] = document.getElementById(id);
  });
  elements.mapModeButtons = Array.from(document.querySelectorAll("[data-map-mode]"));
  elements.teleopButtons = Array.from(document.querySelectorAll(".pad-btn[data-teleop]"));
}

function defaultWindowLayout() {
  return {
    map: { x: 0.02, y: 0.05, w: 0.71, h: 0.44, min: false, max: false, z: 10 },
    camera: { x: 0.02, y: 0.51, w: 0.71, h: 0.42, min: false, max: false, z: 11 },
    telemetry: { x: 0.745, y: 0.05, w: 0.235, h: 0.88, min: false, max: false, z: 12 },
    systems: {
      x: 0.05,
      y: 0.48,
      w: 0.32,
      h: 0.4,
      min: false,
      max: false,
      z: 13,
      norm: { x: 0.05, y: 0.48, w: 0.32, h: 0.4 },
      opened: false,
    },
    world: {
      x: 0.37,
      y: 0.48,
      w: 0.32,
      h: 0.4,
      min: false,
      max: false,
      z: 14,
      norm: { x: 0.37, y: 0.48, w: 0.32, h: 0.4 },
      opened: false,
    },
    ops: {
      x: 0.52,
      y: 0.48,
      w: 0.32,
      h: 0.4,
      min: false,
      max: false,
      z: 15,
      norm: { x: 0.52, y: 0.48, w: 0.32, h: 0.4 },
      opened: false,
    },
    config: {
      x: 0.18,
      y: 0.52,
      w: 0.34,
      h: 0.36,
      min: false,
      max: false,
      z: 16,
      norm: { x: 0.18, y: 0.52, w: 0.34, h: 0.36 },
      opened: false,
    },
    logs: {
      x: 0.48,
      y: 0.52,
      w: 0.34,
      h: 0.36,
      min: false,
      max: false,
      z: 17,
      norm: { x: 0.48, y: 0.52, w: 0.34, h: 0.36 },
      opened: false,
    },
  };
}

function loadWindowLayout() {
  const base = defaultWindowLayout();
  try {
    const raw = localStorage.getItem(STORAGE_WINDOWS);
    if (!raw) {
      return base;
    }
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") {
      return base;
    }
    const merged = { ...base };
    for (const id of WINDOW_IDS) {
      if (parsed[id] && typeof parsed[id] === "object") {
        merged[id] = { ...base[id], ...parsed[id] };
        for (const k of ["x", "y", "w", "h", "z"]) {
          if (typeof merged[id][k] !== "number" || Number.isNaN(merged[id][k])) {
            merged[id][k] = base[id][k];
          }
        }
        merged[id].min = Boolean(merged[id].min);
        merged[id].max = Boolean(merged[id].max);
        if (SECONDARY_IDS.includes(id)) {
          if (typeof merged[id].opened !== "boolean") {
            merged[id].opened = !merged[id].min;
          }
          merged[id].min = false;
        }
        if (merged[id].norm && typeof merged[id].norm === "object") {
          const n = merged[id].norm;
          for (const k of ["x", "y", "w", "h"]) {
            if (typeof n[k] !== "number" || Number.isNaN(n[k])) {
              merged[id].norm = base[id].norm ? { ...base[id].norm } : undefined;
              break;
            }
          }
        } else if (base[id].norm) {
          merged[id].norm = { ...base[id].norm };
        }
      }
    }
    return merged;
  } catch {
    return base;
  }
}

function saveWindowLayout() {
  try {
    localStorage.setItem(STORAGE_WINDOWS, JSON.stringify(windowLayout));
  } catch {
    /* ignore quota */
  }
}

function scheduleCanvasResize() {
  requestAnimationFrame(() => {
    resizeCanvases();
    renderMap();
    renderCameraOverlay();
  });
}

function bringWindowToFront(id) {
  const L = windowLayout[id];
  if (!L) {
    return;
  }
  zCounter += 1;
  L.z = zCounter;
  const el = document.querySelector(`[data-window="${id}"]`);
  if (el) {
    el.style.zIndex = String(L.z);
  }
  saveWindowLayout();
}

function applyWindowLayout() {
  WINDOW_IDS.forEach((id) => {
    const el = document.querySelector(`[data-window="${id}"]`);
    const L = windowLayout[id];
    if (!el || !L) {
      return;
    }
    if (SECONDARY_IDS.includes(id) && L.opened === false) {
      el.classList.add("float-window--closed");
      el.setAttribute("aria-hidden", "true");
      return;
    }
    el.classList.remove("float-window--closed");
    el.setAttribute("aria-hidden", "false");

    el.classList.toggle("minimized", L.min);
    el.classList.toggle("maximized", L.max);
    el.style.zIndex = String(L.z || 10);

    if (L.max) {
      el.style.left = "6px";
      el.style.top = "6px";
      el.style.width = "calc(100% - 12px)";
      el.style.height = "calc(100% - 12px)";
      return;
    }

    el.style.left = `${L.x * 100}%`;
    el.style.top = `${L.y * 100}%`;
    el.style.width = `${L.w * 100}%`;
    if (L.min) {
      el.style.height = "auto";
    } else {
      el.style.height = `${L.h * 100}%`;
    }
  });
  scheduleCanvasResize();
}

function toggleMinimize(id) {
  const L = windowLayout[id];
  if (!L) {
    return;
  }
  if (SECONDARY_IDS.includes(id)) {
    if (!L.opened) {
      return;
    }
    if (L.max) {
      const r = maxRestore[id];
      if (r) {
        L.x = r.x;
        L.y = r.y;
        L.w = r.w;
        L.h = r.h;
        L.max = false;
        delete maxRestore[id];
      }
    }
    L.norm = { x: L.x, y: L.y, w: L.w, h: L.h };
    L.opened = false;
    L.min = false;
    L.max = false;
    saveWindowLayout();
    applyWindowLayout();
    return;
  }

  if (L.min) {
    L.min = false;
    L.max = false;
    if (L.norm) {
      L.x = L.norm.x;
      L.y = L.norm.y;
      L.w = L.norm.w;
      L.h = L.norm.h;
    }
  } else {
    if (L.max) {
      const r = maxRestore[id];
      if (r) {
        L.x = r.x;
        L.y = r.y;
        L.w = r.w;
        L.h = r.h;
        L.max = false;
        delete maxRestore[id];
      }
    }
    L.norm = { x: L.x, y: L.y, w: L.w, h: L.h };
    L.min = true;
    L.max = false;
  }
  saveWindowLayout();
  applyWindowLayout();
}

function toggleMaximize(id) {
  const L = windowLayout[id];
  if (!L) {
    return;
  }
  if (L.max) {
    L.max = false;
    const r = maxRestore[id];
    if (r) {
      L.x = r.x;
      L.y = r.y;
      L.w = r.w;
      L.h = r.h;
      delete maxRestore[id];
    }
    L.min = false;
  } else {
    maxRestore[id] = { x: L.x, y: L.y, w: L.w, h: L.h };
    L.max = true;
    L.min = false;
  }
  saveWindowLayout();
  applyWindowLayout();
}

function restoreAndFocus(id) {
  const L = windowLayout[id];
  if (!L) {
    return;
  }
  if (SECONDARY_IDS.includes(id)) {
    L.opened = true;
    if (L.norm) {
      L.x = L.norm.x;
      L.y = L.norm.y;
      L.w = L.norm.w;
      L.h = L.norm.h;
    }
  } else if (L.norm && L.min) {
    L.x = L.norm.x;
    L.y = L.norm.y;
    L.w = L.norm.w;
    L.h = L.norm.h;
  }
  L.min = false;
  L.max = false;
  bringWindowToFront(id);
  saveWindowLayout();
  applyWindowLayout();
}

function initFloatingWindows() {
  windowLayout = loadWindowLayout();
  zCounter = Math.max(
    30,
    ...WINDOW_IDS.map((id) => windowLayout[id]?.z || 0),
  );
  applyWindowLayout();

  const workspace = document.getElementById("window-workspace");
  if (!workspace) {
    return;
  }

  workspace.querySelectorAll(".float-window").forEach((winEl) => {
    winEl.addEventListener("pointerdown", () => {
      const id = winEl.dataset.window;
      if (id) {
        bringWindowToFront(id);
      }
    });

    const head = winEl.querySelector("[data-drag-handle]");
    if (!head) {
      return;
    }

    head.addEventListener("pointerdown", (e) => {
      if (e.target.closest(".win-btn")) {
        return;
      }
      const id = winEl.dataset.window;
      if (!id || !windowLayout[id]) {
        return;
      }
      bringWindowToFront(id);
      const L = windowLayout[id];
      if (L.max) {
        toggleMaximize(id);
      }
      const rect = workspace.getBoundingClientRect();
      dragState = {
        id,
        startX: e.clientX,
        startY: e.clientY,
        origX: L.x,
        origY: L.y,
        wsW: rect.width,
        wsH: rect.height,
      };
      head.setPointerCapture(e.pointerId);
    });

    head.addEventListener("pointermove", (e) => {
      if (!dragState || dragState.id !== winEl.dataset.window) {
        return;
      }
      const L = windowLayout[dragState.id];
      if (!L || L.max) {
        return;
      }
      const rect = workspace.getBoundingClientRect();
      const dx = (e.clientX - dragState.startX) / rect.width;
      const dy = (e.clientY - dragState.startY) / rect.height;
      let nx = dragState.origX + dx;
      let ny = dragState.origY + dy;
      const boxH = L.min ? 0.04 : L.h;
      nx = clamp(nx, 0, Math.max(0, 1 - L.w));
      ny = clamp(ny, 0, Math.max(0, 1 - boxH));
      L.x = nx;
      L.y = ny;
      applyWindowLayout();
    });

    head.addEventListener("pointerup", (e) => {
      if (dragState && dragState.id === winEl.dataset.window) {
        saveWindowLayout();
        dragState = null;
      }
      try {
        head.releasePointerCapture(e.pointerId);
      } catch {
        /* ignore */
      }
    });

    head.addEventListener("pointercancel", () => {
      if (dragState && dragState.id === winEl.dataset.window) {
        saveWindowLayout();
        dragState = null;
      }
    });

    winEl.querySelector(".win-min")?.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleMinimize(winEl.dataset.window);
    });
    winEl.querySelector(".win-max")?.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleMaximize(winEl.dataset.window);
    });
  });

  document.querySelectorAll("[data-launch]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = btn.dataset.launch;
      if (id && WINDOW_IDS.includes(id)) {
        restoreAndFocus(id);
      }
    });
  });

  document.getElementById("reset-windows-btn")?.addEventListener("click", () => {
    windowLayout = defaultWindowLayout();
    for (const k of Object.keys(maxRestore)) {
      delete maxRestore[k];
    }
    saveWindowLayout();
    applyWindowLayout();
  });

  if (typeof ResizeObserver !== "undefined") {
    const ro = new ResizeObserver(() => {
      scheduleCanvasResize();
    });
    ro.observe(workspace);
  }
}

function bindEvents() {
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
    const preset = button.dataset.teleop || button.getAttribute("data-teleop");
    const vector = teleopPreset(preset);
    const start = () => startTeleopPulse(vector);
    const stop = () => stopTeleopPulse();
    button.addEventListener("mousedown", start);
    button.addEventListener("mouseup", stop);
    button.addEventListener("mouseleave", stop);
    button.addEventListener("touchstart", (e) => {
      e.preventDefault();
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
      if (message.type !== "snapshot") {
        return;
      }
      dashboardState.snapshot = message.data;
      dashboardState.visibleLogs = Array.isArray(message.data.logs)
        ? message.data.logs.slice(-MAX_LOGS)
        : [];
      renderAll();
    } catch (error) {
      console.error("Invalid WebSocket payload", error);
    }
  });

  ws.addEventListener("close", () => {
    dashboardState.reconnectTimer = window.setTimeout(connectWebSocket, 1500);
  });
}

function renderAll() {
  renderMeta();
  renderTelemetry();
  renderSubsystems();
  renderObstacles();
  renderLandmarks();
  renderVoice();
  renderGoal();
  renderArmVoice();
  renderPickup();
  renderConfig();
  renderLogs();
  renderMap();
  renderCameraOverlay();
}

function renderMeta() {
  const snapshot = dashboardState.snapshot;
  if (!snapshot) {
    return;
  }
  const meta = snapshot.meta || {};
  elements["robot-name"].textContent = meta.robot_name || "Wheelchair";
  elements["snapshot-rate"].textContent = `${formatNumber(meta.snapshot_rate_hz || 0, 0)} Hz`;
  const live = Boolean(meta.live);
  const label = meta.connection_label || (live ? "Live" : "Offline");
  setPillState(elements["connection-pill"], live ? "status-live" : "status-offline", label);
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
  elements["pose-xyz"].textContent = `${formatNumber(pose.x)}, ${formatNumber(pose.y)}, ${formatNumber(pose.theta)}`;
  elements["pose-vel-line"].textContent = `${formatNumber(velocity.vx)}, ${formatNumber(velocity.vy)}, ${formatNumber(velocity.omega)}`;
  elements["nav-line"].textContent = `${nav.state || "Idle"} · ${nav.mode || "manual"}`;
  elements["nav-estop"].textContent = nav.estop ? "ACTIVE" : "Clear";
  elements["imu-gyro-z"].textContent = `${formatNumber(imu.gyro_z)} rad/s`;
}

function renderSubsystems() {
  const snapshot = dashboardState.snapshot;
  const container = elements["subsystems-tree"];
  if (!snapshot || !container) {
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
          <div class="tree-detail">${escapeHtml(subsystem.detail || "")}</div>
        </div>
        <span class="status-pill ${statusClass}">${escapeHtml(humanizeStatus(subsystem.status))}</span>
      </div>
    `;
  }).join("");
  container.innerHTML = markup || emptyState("No subsystems.");
}

function renderObstacles() {
  const snapshot = dashboardState.snapshot;
  const container = elements["obstacles-list"];
  if (!snapshot || !container) {
    return;
  }
  const obstacles = snapshot.obstacles || [];
  if (!obstacles.length) {
    container.innerHTML = emptyState("None.");
    return;
  }
  container.innerHTML = obstacles.map((obstacle) => `
    <div class="list-item">
      <div>
        <div class="list-title">Obstacle</div>
        <div class="list-detail">${formatNumber(obstacle.distance_m)} m · ${formatNumber(obstacle.angle_deg)}°</div>
      </div>
      <div class="list-metric">${Math.round((obstacle.confidence || 0) * 100)}%</div>
    </div>
  `).join("");
}

function renderLandmarks() {
  const snapshot = dashboardState.snapshot;
  const container = elements["landmarks-list"];
  if (!snapshot || !container) {
    return;
  }
  const landmarks = snapshot.landmarks || [];
  if (!landmarks.length) {
    container.innerHTML = emptyState("None.");
    return;
  }
  container.innerHTML = landmarks.map((landmark) => `
    <div class="list-item">
      <div>
        <div class="list-title">${escapeHtml(landmark.name || "—")}</div>
        <div class="list-detail">${formatNumber(landmark.x)}, ${formatNumber(landmark.y)}</div>
      </div>
      <div class="list-metric">${Math.round((landmark.confidence || 0) * 100)}%</div>
    </div>
  `).join("");
}

function renderVoice() {
  const snapshot = dashboardState.snapshot;
  if (!snapshot) {
    return;
  }
  const voice = snapshot.voice || {};
  elements["voice-transcript"].textContent = voice.transcript || "—";
  elements["voice-response"].textContent = voice.response || "—";
}

function renderGoal() {
  const snapshot = dashboardState.snapshot;
  if (!snapshot) {
    return;
  }
  const goal = snapshot.goal;
  const nav = snapshot.nav_state || {};
  const meta = snapshot.meta || {};
  const q = `${meta.pending_commands || 0} cmd · ${meta.pending_landmarks || 0} lm`;
  if (!goal) {
    elements["goal-summary"].textContent = `No goal · ${q}`;
  } else {
    const label = goal.label ? `${goal.label} · ` : "";
    elements["goal-summary"].textContent = `${label}${formatNumber(goal.x)}, ${formatNumber(goal.y)} · ${nav.state || "Idle"} · ${q}`;
  }
}

function renderArmVoice() {
  const snapshot = dashboardState.snapshot;
  if (!snapshot) {
    return;
  }
  const armVoice = snapshot.arm_voice || {};
  const status = armVoice.status || "idle";
  const running = Boolean(armVoice.running);
  setPillState(elements["arm-voice-pill"], statusClassFor(status), humanizeStatus(status));
  elements["arm-voice-detail"].textContent = armVoice.detail || "Start, then speak.";
  elements["arm-voice-start-button"].disabled = running || status === "starting";
  elements["arm-voice-stop-button"].disabled = !(running || status === "starting" || status === "stopping");
}

function renderPickup() {
  const snapshot = dashboardState.snapshot;
  if (!snapshot) {
    return;
  }
  const pickup = snapshot.pickup || {};
  const status = pickup.status || "idle";
  const running = Boolean(pickup.running);
  setPillState(elements["pickup-pill"], statusClassFor(status), humanizeStatus(status));
  elements["pickup-detail"].textContent = pickup.detail || "—";
  elements["pickup-refresh-button"].disabled = running;
  elements["pickup-start-button"].disabled = running;
  elements["pickup-stop-button"].disabled = !running;
}

function renderConfig() {
  const snapshot = dashboardState.snapshot;
  const container = elements["config-form"];
  if (!snapshot || !container) {
    return;
  }
  const config = snapshot.config || {};
  const configMeta = snapshot.config_meta || {};
  if (elements["config-note"]) {
    elements["config-note"].textContent = configMeta.save_note || "";
  }
  const hash = JSON.stringify(config);
  if (hash === dashboardState.configRenderedHash) {
    return;
  }
  dashboardState.configRenderedHash = hash;
  const preferredOrder = ["serial", "lidar", "slam", "planner", "safety", "camera", "pickup", "ekf", "phone_imu"];
  const sections = Object.keys(config).sort((a, b) => {
    const ai = preferredOrder.indexOf(a);
    const bi = preferredOrder.indexOf(b);
    if (ai === -1 && bi === -1) {
      return a.localeCompare(b);
    }
    if (ai === -1) {
      return 1;
    }
    if (bi === -1) {
      return -1;
    }
    return ai - bi;
  });
  container.innerHTML = sections.map((name) => renderConfigSection(name, config[name] || {})).join("");
}

function renderConfigSection(sectionName, sectionConfig) {
  const fields = Object.entries(sectionConfig).map(([key, value]) => {
    const label = `
      <div>
        <div class="config-key">${escapeHtml(key)}</div>
        <div class="config-helper">${escapeHtml(sectionName)}</div>
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
            <span>${value ? "On" : "Off"}</span>
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
      <div class="config-section-header">${escapeHtml(capitalize(sectionName))}</div>
      <div class="config-section-body">${fields || emptyState("No fields")}</div>
    </section>
  `;
}

function renderLogs() {
  const container = elements["log-viewer"];
  const logs = dashboardState.visibleLogs || [];
  if (!logs.length) {
    container.innerHTML = emptyState("No logs yet.");
    return;
  }
  const nearBottom = container.scrollTop + container.clientHeight >= container.scrollHeight - 24;
  container.innerHTML = logs.map((entry) => `
    <div class="log-entry">
      <span>${escapeHtml(entry.ts || "")}</span>
      <span class="log-level" style="color:${escapeAttribute(entry.color || "#60a5fa")}">${escapeHtml(entry.level || "INFO")}</span>
      <span>${escapeHtml(entry.source || "")}</span>
      <span>${escapeHtml(entry.message || "")}</span>
    </div>
  `).join("");
  if (nearBottom) {
    container.scrollTop = container.scrollHeight;
  }
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
  ctx.fillStyle = "#0b0d10";
  ctx.fillRect(0, 0, width, height);

  const map = snapshot.occupancy_grid || {};
  drawOccupancyGrid(ctx, canvas, map);
  drawPath(ctx, canvas, map, snapshot.path || []);
  drawLidar(ctx, canvas, map, snapshot.pose || {}, snapshot.lidar_scan || {});
  drawObstaclesMap(ctx, canvas, map, snapshot.pose || {}, snapshot.obstacles || []);
  drawLandmarksMap(ctx, canvas, map, snapshot.pose || {}, snapshot.landmarks || []);
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
      const shade = Math.max(60, Math.min(220, Math.round(200 - value * 140)));
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
  ctx.strokeStyle = "#60a5fa";
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
  ctx.fillStyle = "rgba(61, 214, 140, 0.9)";
  points.forEach((point) => {
    const worldPoint = robotToWorld(point.x, point.y, pose);
    const pixel = worldToCanvas(worldPoint.x, worldPoint.y, map, canvas);
    ctx.beginPath();
    ctx.arc(pixel.x, pixel.y, 2, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.restore();
}

function drawObstaclesMap(ctx, canvas, map, pose, obstacles) {
  if (!obstacles.length) {
    return;
  }
  ctx.save();
  ctx.strokeStyle = "#f87171";
  ctx.fillStyle = "rgba(248, 113, 113, 0.12)";
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

function drawLandmarksMap(ctx, canvas, map, pose, landmarks) {
  if (!landmarks.length) {
    return;
  }
  ctx.save();
  ctx.fillStyle = "#e8eaed";
  ctx.font = "11px Inter, system-ui, sans-serif";
  landmarks.forEach((landmark) => {
    const point = landmark.frame === "robot"
      ? robotToWorld(landmark.x, landmark.y, pose)
      : { x: landmark.x, y: landmark.y };
    const pixel = worldToCanvas(point.x, point.y, map, canvas);
    ctx.beginPath();
    ctx.moveTo(pixel.x, pixel.y - 8);
    ctx.lineTo(pixel.x - 6, pixel.y + 5);
    ctx.lineTo(pixel.x + 6, pixel.y + 5);
    ctx.closePath();
    ctx.fill();
    ctx.fillText(landmark.name || "", pixel.x + 8, pixel.y + 2);
  });
  ctx.restore();
}

function drawGoal(ctx, canvas, map, goal) {
  if (!goal) {
    return;
  }
  const pixel = worldToCanvas(goal.x, goal.y, map, canvas);
  ctx.save();
  ctx.strokeStyle = "#38bdf8";
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
  ctx.fillStyle = "#e8eaed";
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
    if (dashboardState.mapMode === "goal") {
      await postJson("/command", { cmd: "goto", x: world.x, y: world.y, label: "map goal" });
    } else {
      const name = window.prompt("Landmark name");
      if (!name) {
        return;
      }
      await postJson("/landmark", {
        name,
        x: world.x,
        y: world.y,
        frame: "world",
        source: "map",
        descriptor_status: "pending_capture",
      });
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
  ctx.strokeStyle = "#3dd68c";
  ctx.fillStyle = "#3dd68c";
  ctx.lineWidth = 2;
  ctx.font = "11px Inter, system-ui, sans-serif";

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
  const stroke = ready ? "#60a5fa" : "#fbbf24";
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
  const rangeText = target && target.range_m != null ? `${formatNumber(target.range_m)}m` : "n/a";
  ctx.fillText(`${label} ${confidence} · ${rangeText}`, x, Math.max(16, y - 8));
  ctx.restore();
}

function setupJoystick() {
  const pad = elements["joystick-pad"];
  const knob = elements["joystick-knob"];
  const updateJoystick = (clientX, clientY) => {
    const rect = pad.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    const radius = rect.width / 2 - 22;
    let dx = clientX - centerX;
    let dy = clientY - centerY;
    const dist = Math.hypot(dx, dy);
    if (dist > radius) {
      dx = (dx / dist) * radius;
      dy = (dy / dist) * radius;
    }
    knob.style.transform = `translate(${dx}px, ${dy}px)`;
    dashboardState.joystick.vx = Number((-dy / radius) * 0.5);
    dashboardState.joystick.vy = Number((dx / radius) * 0.25);
    elements["joystick-readout"].textContent = `vx ${formatNumber(dashboardState.joystick.vx)} · vy ${formatNumber(dashboardState.joystick.vy)}`;
  };
  const resetJoystick = () => {
    dashboardState.joystick.active = false;
    dashboardState.joystick.pointerId = null;
    dashboardState.joystick.vx = 0;
    dashboardState.joystick.vy = 0;
    knob.style.transform = "translate(0px, 0px)";
    elements["joystick-readout"].textContent = "vx 0 · vy 0";
    stopJoystickPulse();
    postJson("/command", { cmd: "teleop", vx: 0, vy: 0, omega: 0 }).catch(() => {});
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

function renderWaypoints() {
  const snapshot = dashboardState.snapshot;
  const select = elements["waypoint-select"];
  if (!snapshot || !select) {
    return;
  }
  const waypoints = snapshot.waypoints || {};
  const current = select.value;
  select.innerHTML = Object.entries(waypoints).map(([name, coords]) =>
    `<option value="${escapeAttribute(name)}">${escapeHtml(name)} (${formatNumber(coords[0])}, ${formatNumber(coords[1])})</option>`,
  ).join("");
  if (current && waypoints[current]) {
    select.value = current;
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

async function startArmVoiceTest() {
  elements["arm-voice-detail"].textContent = "Starting…";
  try {
    await postJson("/arm/voice", { action: "start", mode: "local_fake" });
  } catch (error) {
    setPillState(elements["arm-voice-pill"], "status-error", "Error");
    elements["arm-voice-detail"].textContent = String(error.message || error);
  }
}

async function stopArmVoiceTest() {
  elements["arm-voice-detail"].textContent = "Stopping…";
  try {
    await postJson("/arm/voice", { action: "stop" });
  } catch (error) {
    elements["arm-voice-detail"].textContent = String(error.message || error);
  }
}

async function refreshPickupDetection() {
  elements["pickup-detail"].textContent = "Refreshing…";
  try {
    await postJson("/pickup", { action: "refresh" });
  } catch (error) {
    elements["pickup-detail"].textContent = String(error.message || error);
  }
}

async function startPickup() {
  elements["pickup-detail"].textContent = "Starting…";
  try {
    await postJson("/pickup", { action: "start" });
  } catch (error) {
    elements["pickup-detail"].textContent = String(error.message || error);
  }
}

async function stopPickup() {
  try {
    await postJson("/pickup", { action: "stop" });
  } catch (error) {
    elements["pickup-detail"].textContent = String(error.message || error);
  }
}

function initCameraStream() {
  const img = elements["camera-stream"];
  img.src = `/camera?stream=${Date.now()}`;
  img.addEventListener("load", () => {
    resizeOverlayCanvas(elements["camera-overlay"]);
    renderCameraOverlay();
  }, { once: false });
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

function updateMapModeLabel() {
  elements["map-mode-label"].textContent = `Click: ${dashboardState.mapMode}`;
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
  const map = {
    connected: "OK",
    running: "Run",
    starting: "Start",
    stopping: "Stop",
    idle: "Idle",
    error: "Err",
    ready: "Ready",
    success: "OK",
    searching: "Scan",
    stopped: "Off",
    warning: "Warn",
    warn: "Warn",
  };
  return map[normalized] || "—";
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
