"""
Lightweight http dashboard for the autonomous wheelchair demo.

Runs in two modes:
  - standalone demo mode (`python laptop/web_ui.py`)
  - embedded mode where the rest of the stack pushes state via setters
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import dataclasses
import io
import json
import logging
import math
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from aiohttp import WSMsgType, web
except ImportError:  # pragma: no cover - handled at runtime
    WSMsgType = None
    web = None

try:
    from config import load_config
except ImportError:  # pragma: no cover - useful if imported as a namespace package
    from laptop.config import load_config

try:
    from foxglove_viz import FakeFoxgloveViz, create_foxglove_viz
except ImportError:  # pragma: no cover - useful if imported as a namespace package
    from laptop.foxglove_viz import FakeFoxgloveViz, create_foxglove_viz

try:
    from pickup_controller import PickupControllerError, create_pickup_controller
except ImportError:  # pragma: no cover - useful if imported as a namespace package
    from laptop.pickup_controller import PickupControllerError, create_pickup_controller

logger = logging.getLogger(__name__)

SNAPSHOT_RATE_HZ = 10.0
MAX_LOG_ENTRIES = 250
MAX_MAP_DIMENSION = 120

DEFAULT_SUBSYSTEMS = {
    "LiDAR": {"icon": "LD", "status": "disconnected", "detail": "Waiting for scan data"},
    "Camera": {"icon": "CA", "status": "disconnected", "detail": "Waiting for frames"},
    "Motors": {"icon": "MT", "status": "disconnected", "detail": "Serial offline"},
    "IMU": {"icon": "IM", "status": "disconnected", "detail": "No packets received"},
    "SLAM": {"icon": "SL", "status": "disconnected", "detail": "Map idle"},
    "Planner": {"icon": "PL", "status": "disconnected", "detail": "No active goal"},
    "Voice": {"icon": "VO", "status": "disconnected", "detail": "Voice loop idle"},
    "Pickup": {"icon": "PK", "status": "disconnected", "detail": "Waiting for pickup controller"},
}

LEVEL_COLORS = {
    "DEBUG": "#7a7a7a",
    "INFO": "#1f6feb",
    "WARN": "#c58b00",
    "WARNING": "#c58b00",
    "ERROR": "#d1242f",
}

# 1x1 white JPEG fallback. Used only if OpenCV/Pillow are unavailable.
PLACEHOLDER_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010101004800480000ffdb00430008060607060508"
    "0707070909080a0c140d0c0b0b0c19120f131d1a1f1e1d1a1c1c20242e2720222c"
    "231c1c2837292c30313434341f27393d38323c2e333432ffdb0043010909090c0b"
    "0c180d0d1832251c25323232323232323232323232323232323232323232323232"
    "323232323232323232323232323232323232323232ffc000110800010001030122"
    "00021101031101ffc4001b00000105010101010101000000000000000001020304"
    "05060708090a0bffc400b5100002010303020403050504040000017d0102030004"
    "1105122131410613516107227114328191a1082342b1c11552d1f0243362728209"
    "0a161718191a25262728292a3435363738393a434445464748494a535455565758"
    "595a636465666768696a737475767778797a838485868788898a92939495969798"
    "999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5"
    "d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9faffc4001b01000301"
    "01010101010101000000000000000102030405060708090a0bffc400b511000201"
    "020404030407050404000102770001020311040521310612415107617113223281"
    "08144291a1b1c109233352f0156272d10a162434e125f11718191a262728292a35"
    "363738393a434445464748494a535455565758595a636465666768696a73747576"
    "7778797a82838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3"
    "b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae2e3e4e5e6e7e8e9"
    "eaf2f3f4f5f6f7f8f9faffda000c03010002110311003f00f7fa28a2803fffd9"
)


def create_web_ui(fake: bool = False) -> "WebUI":
    return WebUI(fake=fake)


class WebUI:
    def __init__(self, fake: bool = False, robot_name: str = "Autonomous Wheelchair"):
        self.fake = fake
        self.robot_name = robot_name
        self._workspace_dir = Path(__file__).resolve().parent.parent
        self._static_dir = Path(__file__).resolve().parent / "static"
        self._waypoints_path = self._workspace_dir / "waypoints" / "demo_space.json"
        self._lock = threading.Lock()

        self._clients: set[Any] = set()
        self._command_queue: List[Dict[str, Any]] = []
        self._landmark_queue: List[Dict[str, Any]] = []
        self._runner = None
        self._site = None
        self._app = None
        self._broadcast_task: Optional[asyncio.Task] = None
        self._demo_task: Optional[asyncio.Task] = None
        self._foxglove_task: Optional[asyncio.Task] = None
        self._foxglove = None
        self._pickup_controller = None
        self._pickup_poll_task: Optional[asyncio.Task] = None
        self._arm_voice_process: Optional[asyncio.subprocess.Process] = None
        self._arm_voice_stdout_task: Optional[asyncio.Task] = None
        self._arm_voice_stderr_task: Optional[asyncio.Task] = None
        self._arm_voice_watch_task: Optional[asyncio.Task] = None
        self._arm_voice_stop_requested = False
        self._started = False

        self._camera_jpeg: bytes = PLACEHOLDER_JPEG
        self._camera_meta = {"available": False, "width": 0, "height": 0, "updated_at": None}

        self._state: Dict[str, Any] = self._build_default_state()
        self._state["config"] = self._load_config_snapshot()
        self._state["waypoints"] = self.load_waypoints()

    async def start(self, host: str = "127.0.0.1", port: int = 8766) -> str:
        if web is None:
            raise RuntimeError("aiohttp is required to run laptop/web_ui.py")
        if self._started:
            return f"http://{host}:{port}"

        self._app = web.Application(client_max_size=10 * 1024 * 1024)
        self._app.router.add_get("/", self._handle_index)
        self._app.router.add_get("/ws", self._handle_ws)
        self._app.router.add_get("/camera", self._handle_camera)
        self._app.router.add_get("/waypoints", self._handle_get_waypoints)
        self._app.router.add_post("/waypoints", self._handle_post_waypoints)
        self._app.router.add_post("/command", self._handle_post_command)
        self._app.router.add_post("/pickup", self._handle_post_pickup)
        self._app.router.add_post("/arm/voice", self._handle_post_arm_voice)
        self._app.router.add_post("/landmark", self._handle_post_landmark)
        self._app.router.add_get("/config", self._handle_get_config)
        self._app.router.add_post("/config", self._handle_post_config)
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_static("/static/", str(self._static_dir), show_index=False)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, host=host, port=port)
        await self._site.start()

        self._started = True
        await self._start_foxglove_bridge()
        await self._start_pickup_integration()
        self._broadcast_task = asyncio.create_task(self._broadcast_loop(), name="web-ui-broadcast")
        if self.fake:
            self._demo_task = asyncio.create_task(self._demo_loop(), name="web-ui-demo")

        self.append_log("Web UI started", level="INFO", source="web_ui")
        logger.info("Web UI listening on http://%s:%d", host, port)
        return f"http://{host}:{port}"

    async def stop(self) -> None:
        self._started = False
        await self._stop_arm_voice_demo(detail="Dashboard shutting down")
        tasks = [
            self._broadcast_task,
            self._demo_task,
            self._foxglove_task,
            self._pickup_poll_task,
            self._arm_voice_stdout_task,
            self._arm_voice_stderr_task,
            self._arm_voice_watch_task,
        ]
        for task in tasks:
            if task is not None:
                task.cancel()
        for task in tasks:
            if task is not None:
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        for ws in list(self._clients):
            with contextlib.suppress(Exception):
                await ws.close()
        self._clients.clear()

        if self._foxglove is not None:
            with contextlib.suppress(Exception):
                await self._foxglove.stop()
        self._foxglove = None

        if self._pickup_controller is not None:
            with contextlib.suppress(Exception):
                await self._pickup_controller.stop()
        self._pickup_controller = None

        if self._runner is not None:
            await self._runner.cleanup()
        self._runner = None
        self._site = None
        self._foxglove_task = None
        self._pickup_poll_task = None

    def set_pose(self, pose: Any = None, **kwargs: Any) -> None:
        payload = self._normalize_pose(pose, **kwargs)
        self._set_state_key("pose", payload)
        if self._has_real_foxglove():
            self._foxglove.set_pose(payload["x"], payload["y"], payload["theta"])

    def set_velocity(self, velocity: Any = None, **kwargs: Any) -> None:
        payload = self._normalize_velocity(velocity, **kwargs)
        self._set_state_key("velocity", payload)
        if self._has_real_foxglove():
            self._foxglove.set_motor_cmd(payload["vx"], payload["vy"], payload["omega"])

    def set_obstacles(self, obstacles: Iterable[Any]) -> None:
        serialized = [self._serialize_obstacle(obstacle) for obstacle in obstacles]
        self._set_state_key("obstacles", serialized)
        if self._has_real_foxglove():
            self._foxglove.set_obstacles(serialized)

    def set_lidar_scan(
        self,
        scan: Any = None,
        *,
        xy_points: Optional[Iterable[Any]] = None,
        polar_points: Optional[Iterable[Any]] = None,
        timestamp: Optional[float] = None,
        frame: str = "robot",
    ) -> None:
        if isinstance(scan, dict):
            payload = {
                "xy_points": self._coerce_xy_points(scan.get("xy_points", [])),
                "polar_points": self._coerce_polar_points(scan.get("polar_points", [])),
                "timestamp_monotonic_s": scan.get("timestamp_monotonic_s", timestamp),
                "frame": scan.get("frame", frame),
            }
        else:
            local_xy = xy_points
            local_polar = polar_points
            local_ts = timestamp
            if scan is not None:
                if hasattr(scan, "points"):
                    local_polar = getattr(scan, "points")
                if hasattr(scan, "as_xy"):
                    try:
                        cfg = load_config().lidar
                        local_xy = scan.as_xy(
                            min_range=cfg.min_range_m,
                            max_range=cfg.max_range_m,
                            angle_offset_deg=cfg.angle_offset_deg,
                        )
                    except Exception:
                        local_xy = local_xy or []
                local_ts = getattr(scan, "timestamp", local_ts)
            payload = {
                "xy_points": self._coerce_xy_points(local_xy or []),
                "polar_points": self._coerce_polar_points(local_polar or []),
                "timestamp_monotonic_s": local_ts,
                "frame": frame,
            }

        self._set_state_key("lidar_scan", payload)
        self.set_subsystem_status("LiDAR", "connected" if payload["xy_points"] else "disconnected",
                                  "Live scan streaming" if payload["xy_points"] else "No scan points")
        if self._has_real_foxglove():
            if payload["xy_points"]:
                scan_xy = np.array([[point["x"], point["y"]] for point in payload["xy_points"]], dtype=np.float32)
            else:
                scan_xy = np.empty((0, 2), dtype=np.float32)
            self._foxglove.set_lidar_scan(scan_xy)

    def set_path(self, path: Iterable[Any]) -> None:
        serialized = [self._serialize_point(point) for point in path]
        self._set_state_key("path", serialized)
        if self._has_real_foxglove():
            self._foxglove.set_path(serialized)

    def set_goal(self, goal: Any = None, **kwargs: Any) -> None:
        if goal is None and not kwargs:
            payload = None
        elif isinstance(goal, dict):
            payload = {
                "x": float(goal.get("x", 0.0)),
                "y": float(goal.get("y", 0.0)),
                "label": goal.get("label"),
                "source": goal.get("source", "api"),
            }
        elif isinstance(goal, (tuple, list)) and len(goal) >= 2:
            payload = {
                "x": float(goal[0]),
                "y": float(goal[1]),
                "label": kwargs.get("label"),
                "source": kwargs.get("source", "api"),
            }
        else:
            payload = {
                "x": float(kwargs.get("x", 0.0)),
                "y": float(kwargs.get("y", 0.0)),
                "label": kwargs.get("label"),
                "source": kwargs.get("source", "api"),
            }
        self._set_state_key("goal", payload)

    def set_nav_state(self, nav_state: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        updates = dict(nav_state or {})
        updates.update(kwargs)
        with self._lock:
            current = copy.deepcopy(self._state["nav_state"])
            current.update(updates)
            self._state["nav_state"] = current
            if "serial_connected" in updates:
                self._state["meta"]["live"] = bool(updates["serial_connected"])
                self._state["meta"]["connection_label"] = "Live" if updates["serial_connected"] else "Offline"
                self._state["meta"]["updated_at"] = self._iso_now()
        if "serial_connected" in updates:
            motors_status = "connected" if updates["serial_connected"] else "disconnected"
            self.set_subsystem_status("Motors", motors_status,
                                      "Serial connected" if updates["serial_connected"] else "Serial offline")
        if self._has_real_foxglove():
            current_state = self.get_snapshot()["nav_state"]
            self._foxglove.set_nav_state(current_state.get("state", "Idle"))
            self._foxglove.set_estop(bool(current_state.get("estop", False)))

    def set_camera_frame(self, frame: Any) -> None:
        if frame is None:
            with self._lock:
                self._camera_jpeg = PLACEHOLDER_JPEG
                self._camera_meta["width"] = 0
                self._camera_meta["height"] = 0
                self._camera_meta["available"] = False
                self._camera_meta["updated_at"] = self._iso_now()
                self._state["camera"] = copy.deepcopy(self._camera_meta)
                self._state["meta"]["updated_at"] = self._iso_now()
            self.set_subsystem_status("Camera", "disconnected", "No frame available")
            return

        encoded = self._encode_frame(frame)
        if encoded is None:
            return
        with self._lock:
            self._camera_jpeg = encoded
            meta = self._extract_frame_meta(frame)
            self._camera_meta.update(meta)
            self._camera_meta["available"] = True
            self._camera_meta["updated_at"] = self._iso_now()
            self._state["camera"] = copy.deepcopy(self._camera_meta)
            self._state["meta"]["updated_at"] = self._iso_now()
        self.set_subsystem_status("Camera", "connected", "MJPEG stream active")
        if self._has_real_foxglove() and hasattr(frame, "shape"):
            self._foxglove.set_camera_frame(frame)

    def set_occupancy_grid(
        self,
        occupancy: Any,
        *,
        resolution_m: Optional[float] = None,
        grid_size: Optional[int] = None,
        origin_cell: Optional[int] = None,
    ) -> None:
        payload = self._serialize_grid(occupancy, resolution_m, grid_size, origin_cell)
        self._set_state_key("occupancy_grid", payload)
        status = "connected" if payload["occupancy"] else "disconnected"
        detail = "Occupancy grid ready" if payload["occupancy"] else "No map data"
        self.set_subsystem_status("SLAM", status, detail)
        if self._has_real_foxglove():
            grid_array = np.asarray(occupancy, dtype=np.float32)
            self._foxglove.set_occupancy_grid(grid_array, resolution_m or load_config().slam.resolution_m)

    def set_landmarks(self, landmarks: Iterable[Any]) -> None:
        serialized = [self._serialize_landmark(landmark) for landmark in landmarks]
        self._set_state_key("landmarks", serialized)

    def set_voice_state(self, transcript: str = "", response: str = "") -> None:
        payload = {
            "transcript": transcript,
            "response": response,
            "updated_at": self._iso_now(),
        }
        self._set_state_key("voice", payload)
        voice_status = "connected" if transcript or response else "disconnected"
        self.set_subsystem_status("Voice", voice_status,
                                  response or "Voice loop idle")

    def set_arm_voice_state(self, **updates: Any) -> None:
        with self._lock:
            current = copy.deepcopy(self._state["arm_voice"])
            current.update(updates)
            current["updated_at"] = self._iso_now()
            self._state["arm_voice"] = current
            self._state["meta"]["updated_at"] = self._iso_now()

    def set_pickup_state(self, pickup_state: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        with self._lock:
            current = copy.deepcopy(self._state["pickup"])
            if isinstance(pickup_state, dict):
                current.update(pickup_state)
            current.update(kwargs)
            current["updated_at"] = self._iso_now()
            self._state["pickup"] = current
            self._state["meta"]["updated_at"] = self._iso_now()

        subsystem_status, connected = self._pickup_subsystem_status(current)
        self.set_subsystem_status(
            "Pickup",
            subsystem_status,
            current.get("detail", "Pickup controller idle."),
            connected=connected,
        )

    def set_imu_state(self, imu: Any = None, **kwargs: Any) -> None:
        payload = {
            "receiving": False,
            "packet_age_s": None,
            "gyro_x": 0.0,
            "gyro_y": 0.0,
            "gyro_z": 0.0,
            "accel_x": 0.0,
            "accel_y": 0.0,
            "accel_z": 0.0,
            "updated_at": self._iso_now(),
        }
        now = time.monotonic()

        if isinstance(imu, dict):
            payload.update({key: imu[key] for key in payload if key in imu})
            if payload["packet_age_s"] is None and "timestamp_s" in imu:
                payload["packet_age_s"] = max(0.0, now - float(imu["timestamp_s"]))
        elif imu is not None:
            timestamp_s = getattr(imu, "timestamp_s", None)
            payload.update(
                {
                    "gyro_x": float(getattr(imu, "gyro_x", 0.0)),
                    "gyro_y": float(getattr(imu, "gyro_y", 0.0)),
                    "gyro_z": float(getattr(imu, "gyro_z", 0.0)),
                    "accel_x": float(getattr(imu, "accel_x", 0.0)),
                    "accel_y": float(getattr(imu, "accel_y", 0.0)),
                    "accel_z": float(getattr(imu, "accel_z", 0.0)),
                }
            )
            if timestamp_s is not None:
                payload["packet_age_s"] = max(0.0, now - float(timestamp_s))

        payload.update({key: value for key, value in kwargs.items() if key in payload})
        if payload["packet_age_s"] is not None:
            payload["packet_age_s"] = float(payload["packet_age_s"])
        payload["receiving"] = bool(payload["receiving"]) or (
            payload["packet_age_s"] is not None and payload["packet_age_s"] < 2.0
        )
        self._set_state_key("imu", payload)
        if payload["receiving"]:
            self.set_subsystem_status("IMU", "connected", f"gyro z {payload['gyro_z']:.3f} rad/s")
        else:
            self.set_subsystem_status("IMU", "disconnected", "No recent packets")

    def set_subsystem_status(
        self,
        name: str,
        status: str,
        detail: Optional[str] = None,
        *,
        connected: Optional[bool] = None,
    ) -> None:
        normalized = status.lower()
        is_connected = connected if connected is not None else normalized == "connected"
        with self._lock:
            subsystem = copy.deepcopy(self._state["subsystems"].get(name, {"icon": name[:2].upper()}))
            subsystem["status"] = normalized
            subsystem["connected"] = is_connected
            if detail is not None:
                subsystem["detail"] = detail
            self._state["subsystems"][name] = subsystem
            self._state["modules"] = self._build_modules(self._state["subsystems"])
            self._state["meta"]["updated_at"] = self._iso_now()

    def append_log(self, message: str, *, level: str = "INFO", source: str = "system") -> None:
        entry = {
            "ts": self._iso_now(),
            "level": level.upper(),
            "source": source,
            "message": message,
            "color": LEVEL_COLORS.get(level.upper(), "#1f6feb"),
        }
        with self._lock:
            logs = list(self._state["logs"])
            logs.append(entry)
            self._state["logs"] = logs[-MAX_LOG_ENTRIES:]
            self._state["meta"]["updated_at"] = self._iso_now()

    def _handle_pickup_update(self, pickup_state: Dict[str, Any], frame: Any = None) -> None:
        self.set_pickup_state(pickup_state)
        if frame is not None:
            self.set_camera_frame(frame)

    def _pickup_subsystem_status(self, pickup_state: Dict[str, Any]) -> Tuple[str, bool]:
        status = str(pickup_state.get("status", "idle")).lower()
        if status == "running":
            return "running", True
        if status in {"ready", "success"}:
            return "connected", True
        if status in {"searching", "warning"}:
            return "warning", True
        if status == "error":
            return "error", False
        return "disconnected", False

    async def _ensure_pickup_controller(self):
        if self._pickup_controller is not None:
            return self._pickup_controller

        cfg = load_config().pickup
        if not cfg.enabled and not cfg.use_fake_vision:
            self.set_pickup_state(
                status="idle",
                running=False,
                detail="Enable pickup in config to start VIAM ball pickup.",
            )
            return None
        if not cfg.use_fake_vision and (not cfg.camera_name or not cfg.vision_service_name):
            self.set_pickup_state(
                status="error",
                running=False,
                detail="Set VIAM camera and vision service names to enable pickup.",
                last_error="Missing VIAM pickup configuration.",
            )
            return None

        controller = create_pickup_controller(update_hook=self._handle_pickup_update)
        try:
            await controller.start()
        except Exception as exc:
            self.set_pickup_state(
                status="error",
                running=False,
                detail=f"Pickup controller unavailable: {exc}",
                last_error=str(exc),
            )
            self.append_log(f"Pickup controller unavailable: {exc}", level="WARN", source="pickup")
            return None

        self._pickup_controller = controller
        self.set_pickup_state(
            status="searching",
            running=False,
            detail="Pickup controller connected. Waiting for ball detection.",
            last_error="",
        )
        return controller

    async def _start_pickup_integration(self) -> None:
        controller = await self._ensure_pickup_controller()
        if controller is None:
            return

        interval_s = max(0.2, float(load_config().pickup.poll_interval_s))
        self._pickup_poll_task = asyncio.create_task(
            self._pickup_poll_loop(interval_s),
            name="web-ui-pickup-poll",
        )

    async def _pickup_poll_loop(self, interval_s: float) -> None:
        while True:
            if not self._started or self._pickup_controller is None:
                return
            try:
                if not self.get_snapshot()["pickup"].get("running"):
                    await self._pickup_controller.refresh_detection(include_image=True)
            except Exception as exc:
                self.set_pickup_state(
                    status="error",
                    running=False,
                    detail=f"Pickup refresh failed: {exc}",
                    last_error=str(exc),
                )
            await asyncio.sleep(interval_s)

    def load_waypoints(self) -> Dict[str, List[float]]:
        try:
            if self._waypoints_path.exists():
                data = json.loads(self._waypoints_path.read_text(encoding="utf-8"))
                return {
                    str(name): [float(coords[0]), float(coords[1])]
                    for name, coords in data.items()
                    if isinstance(coords, (list, tuple)) and len(coords) >= 2
                }
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.warning("Failed to load waypoints: %s", exc)
            self.append_log("Failed to load waypoint file", level="WARN", source="web_ui")
        return {}

    def save_waypoints(self, waypoints: Dict[str, List[float]]) -> None:
        self._waypoints_path.parent.mkdir(parents=True, exist_ok=True)
        self._waypoints_path.write_text(
            json.dumps(waypoints, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def consume_commands(self) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._command_queue)
            self._command_queue.clear()
            self._refresh_queue_meta_locked()
            return items

    def consume_landmark_requests(self) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._landmark_queue)
            self._landmark_queue.clear()
            self._refresh_queue_meta_locked()
            return items

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            snapshot = copy.deepcopy(self._state)
        snapshot["camera"]["stream_url"] = "/camera"
        snapshot["meta"]["snapshot_rate_hz"] = SNAPSHOT_RATE_HZ
        return snapshot

    async def _handle_index(self, request: Any) -> Any:
        return web.FileResponse(self._static_dir / "index.html")

    async def _handle_health(self, request: Any) -> Any:
        return web.json_response({"ok": True, "live": self.get_snapshot()["meta"]["live"]})

    async def _handle_get_waypoints(self, request: Any) -> Any:
        return web.json_response({"type": "waypoints", "data": self.get_snapshot()["waypoints"]})

    async def _handle_post_waypoints(self, request: Any) -> Any:
        payload = await request.json()
        with self._lock:
            current = copy.deepcopy(self._state["waypoints"])
        if "waypoints" in payload and isinstance(payload["waypoints"], dict):
            for name, coords in payload["waypoints"].items():
                if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                    current[str(name)] = [float(coords[0]), float(coords[1])]
        else:
            name = str(payload.get("name", "")).strip()
            x = float(payload.get("x", 0.0))
            y = float(payload.get("y", 0.0))
            if not name:
                return web.json_response({"error": "Waypoint name required"}, status=400)
            current[name] = [x, y]

        self._set_state_key("waypoints", current)
        self.save_waypoints(current)
        self.append_log("Waypoints updated", level="INFO", source="api")
        return web.json_response({"type": "waypoints", "data": current})

    async def _handle_post_command(self, request: Any) -> Any:
        payload = await request.json()
        cmd = str(payload.get("cmd", "")).lower()
        if cmd not in {"goto", "stop", "teleop"}:
            return web.json_response({"error": "Unsupported command"}, status=400)

        command = {"cmd": cmd, "ts": self._iso_now(), "queued_for_controller": True}
        if cmd == "goto":
            if "waypoint" in payload:
                waypoint = str(payload["waypoint"])
                with self._lock:
                    coords = self._state["waypoints"].get(waypoint)
                if coords is None:
                    return web.json_response({"error": "Unknown waypoint"}, status=404)
                command["waypoint"] = waypoint
                command["x"] = float(coords[0])
                command["y"] = float(coords[1])
                self.set_goal({"x": coords[0], "y": coords[1], "label": waypoint, "source": "waypoint"})
            else:
                command["x"] = float(payload.get("x", 0.0))
                command["y"] = float(payload.get("y", 0.0))
                self.set_goal({"x": command["x"], "y": command["y"], "label": payload.get("label"), "source": "map"})
            self.set_nav_state(state="Navigating", mode="autonomous")
            self.set_subsystem_status("Planner", "connected", "Goal accepted")
        elif cmd == "teleop":
            command["vx"] = float(payload.get("vx", 0.0))
            command["vy"] = float(payload.get("vy", 0.0))
            command["omega"] = float(payload.get("omega", 0.0))
            self.set_velocity(command)
            self.set_nav_state(state="Teleop", mode="manual")
        else:
            self.set_velocity({"vx": 0.0, "vy": 0.0, "omega": 0.0})
            self.set_nav_state(state="Stopped", mode="manual")
            self.set_goal(None)
            self.set_subsystem_status("Planner", "disconnected", "No active goal")

        with self._lock:
            self._command_queue.append(command)
            self._refresh_queue_meta_locked()
            command["queue_depth"] = len(self._command_queue)
            self._state["nav_state"]["last_command"] = command
            self._state["meta"]["updated_at"] = self._iso_now()
        self.append_log(f"Command queued for controller: {cmd}", level="INFO", source="api")
        return web.json_response({"type": "command_ack", "data": command})

    async def _handle_post_pickup(self, request: Any) -> Any:
        payload = await request.json()
        action = str(payload.get("action", "refresh")).lower()
        controller = await self._ensure_pickup_controller()
        if controller is None:
            return web.json_response({"error": "Pickup controller unavailable"}, status=503)

        try:
            if action == "refresh":
                state = await controller.refresh_detection(include_image=True)
            elif action == "start":
                state = await controller.pickup_ball()
            elif action == "stop":
                state = await controller.stop_pickup()
            else:
                return web.json_response({"error": "Unsupported pickup action"}, status=400)
        except PickupControllerError as exc:
            self.set_pickup_state(
                status="error",
                running=False,
                detail=str(exc),
                last_error=str(exc),
            )
            return web.json_response({"error": str(exc)}, status=500)
        except Exception as exc:  # pragma: no cover - runtime safety
            message = f"Pickup request failed: {exc}"
            self.set_pickup_state(
                status="error",
                running=False,
                detail=message,
                last_error=str(exc),
            )
            return web.json_response({"error": message}, status=500)

        self.append_log(f"Pickup action handled: {action}", level="INFO", source="pickup")
        return web.json_response({"type": "pickup_ack", "data": state})

    async def _handle_post_arm_voice(self, request: Any) -> Any:
        payload = await request.json()
        action = str(payload.get("action", "")).lower()
        if action == "start":
            mode = str(payload.get("mode", "local_fake")).lower()
            if mode not in {"local_fake", "viam"}:
                return web.json_response({"error": "Unsupported arm voice mode"}, status=400)
            state = await self._start_arm_voice_demo(mode=mode)
            return web.json_response({"type": "arm_voice_ack", "data": state})
        if action == "stop":
            state = await self._stop_arm_voice_demo(detail="Stopping arm voice test...")
            return web.json_response({"type": "arm_voice_ack", "data": state})
        return web.json_response({"error": "Unsupported arm voice action"}, status=400)

    async def _handle_post_landmark(self, request: Any) -> Any:
        payload = await request.json()
        name = str(payload.get("name", "")).strip()
        if not name:
            return web.json_response({"error": "Landmark name required"}, status=400)

        landmark = {
            "name": name,
            "x": float(payload.get("x", 0.0)),
            "y": float(payload.get("y", 0.0)),
            "confidence": float(payload.get("confidence", 1.0)),
            "frame": str(payload.get("frame", "world")),
            "source": payload.get("source", "map"),
            "descriptor_status": str(payload.get("descriptor_status", "pending_capture")),
            "queued_for_capture": True,
            "capture_descriptor": True,
        }

        with self._lock:
            landmarks = [lm for lm in self._state["landmarks"] if lm.get("name") != name]
            landmarks.append(landmark)
            self._state["landmarks"] = landmarks
            self._landmark_queue.append(landmark)
            self._refresh_queue_meta_locked()
            self._state["meta"]["updated_at"] = self._iso_now()
        self.append_log(f"Landmark queued for descriptor capture: {name}", level="INFO", source="api")
        return web.json_response({"type": "landmark", "data": landmark})

    async def _handle_get_config(self, request: Any) -> Any:
        snapshot = self.get_snapshot()
        return web.json_response(
            {
                "type": "config",
                "data": snapshot["config"],
                "meta": snapshot["config_meta"],
            }
        )

    async def _handle_post_config(self, request: Any) -> Any:
        payload = await request.json()
        with self._lock:
            current = copy.deepcopy(self._state["config"])
        merged = self._merge_config(current, payload)
        self._set_state_key("config", merged)
        self.append_log("Dashboard config snapshot updated (hot-reload not wired)", level="INFO", source="api")
        return web.json_response(
            {
                "type": "config",
                "data": merged,
                "meta": self.get_snapshot()["config_meta"],
            }
        )

    async def _handle_ws(self, request: Any) -> Any:
        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)
        self._clients.add(ws)
        await ws.send_json({"type": "snapshot", "data": self.get_snapshot()})
        try:
            async for message in ws:
                if message.type == WSMsgType.ERROR:
                    break
                if message.type == WSMsgType.TEXT and message.data == "ping":
                    await ws.send_json({"type": "pong", "data": {"ts": self._iso_now()}})
        finally:
            self._clients.discard(ws)
        return ws

    async def _handle_camera(self, request: Any) -> Any:
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "multipart/x-mixed-replace; boundary=frame",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
        await response.prepare(request)
        try:
            while True:
                with self._lock:
                    frame = self._camera_jpeg
                chunk = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                    + frame
                    + b"\r\n"
                )
                await response.write(chunk)
                await asyncio.sleep(1.0 / SNAPSHOT_RATE_HZ)
        except asyncio.CancelledError:
            raise
        except (ConnectionResetError, BrokenPipeError):
            pass
        except Exception:
            pass
        finally:
            with contextlib.suppress(Exception):
                await response.write_eof()
        return response

    async def _broadcast_loop(self) -> None:
        while True:
            if self._clients:
                snapshot = {"type": "snapshot", "data": self.get_snapshot()}
                stale: List[Any] = []
                for ws in list(self._clients):
                    try:
                        await ws.send_json(snapshot)
                    except Exception:
                        stale.append(ws)
                for ws in stale:
                    self._clients.discard(ws)
            await asyncio.sleep(1.0 / SNAPSHOT_RATE_HZ)

    async def _demo_loop(self) -> None:
        t = 0.0
        config = load_config()
        self.set_nav_state(state="Demo Ready", mode="autonomous", serial_connected=True, estop=False)
        for name in DEFAULT_SUBSYSTEMS:
            detail = DEFAULT_SUBSYSTEMS[name]["detail"]
            self.set_subsystem_status(name, "connected", detail.replace("Waiting for ", "Streaming "))
        self.append_log("Starting fake dashboard telemetry", level="INFO", source="demo")

        grid = self._build_demo_grid()
        self.set_occupancy_grid(
            grid,
            resolution_m=config.slam.resolution_m * 2.0,
            grid_size=grid.shape[0],
            origin_cell=grid.shape[0] // 2,
        )

        voice_pairs = [
            ("go to the door", "On my way"),
            ("take me to the table", "Routing now"),
            ("stop", "Stopping"),
            ("go to the window", "Heading there"),
        ]

        while True:
            t += 0.1
            pose_x = 1.4 * math.cos(t / 5.0)
            pose_y = 1.1 * math.sin(t / 6.0)
            theta = math.sin(t / 7.0) * 0.8
            self.set_pose((pose_x, pose_y, theta), covariance=[[0.02, 0, 0], [0, 0.02, 0], [0, 0, 0.01]])
            self.set_velocity(
                {
                    "vx": -0.28 * math.sin(t / 5.0),
                    "vy": 0.18 * math.cos(t / 6.0),
                    "omega": 0.12 * math.cos(t / 7.0),
                }
            )

            waypoint_names = list(self.get_snapshot()["waypoints"].keys()) or ["door"]
            target_name = waypoint_names[int(t / 9.0) % len(waypoint_names)]
            target = self.get_snapshot()["waypoints"].get(target_name, [3.0, 0.0])
            self.set_goal({"x": target[0], "y": target[1], "label": target_name, "source": "demo"})
            self.set_path(self._build_demo_path((pose_x, pose_y), (target[0], target[1])))

            obstacles = self._build_demo_obstacles(t)
            self.set_obstacles(obstacles)
            self.set_landmarks(self._build_demo_landmarks())
            self.set_lidar_scan(
                {
                    "xy_points": self._build_demo_scan_points(t, obstacles),
                    "timestamp_monotonic_s": time.monotonic(),
                    "frame": "robot",
                }
            )
            self.set_camera_frame(self._build_demo_camera_frame(t, obstacles))
            self.set_imu_state(
                {
                    "receiving": True,
                    "packet_age_s": 0.08,
                    "gyro_x": 0.0,
                    "gyro_y": 0.0,
                    "gyro_z": 0.18 * math.cos(t / 4.5),
                    "accel_x": 0.02 * math.sin(t / 3.0),
                    "accel_y": 0.01 * math.cos(t / 3.2),
                    "accel_z": 9.81,
                }
            )
            self.set_nav_state(state="Navigating", mode="autonomous", serial_connected=True, estop=False)

            with self._lock:
                arm_voice_running = bool(self._state["arm_voice"]["running"])
            if not arm_voice_running:
                voice_idx = int(t / 8.0) % len(voice_pairs)
                transcript, response = voice_pairs[voice_idx]
                self.set_voice_state(transcript=transcript, response=response)

            if int(t * 10) % 40 == 0:
                self.append_log("Planner replanned around obstacle cluster", level="INFO", source="planner")
            if int(t * 10) % 65 == 0:
                self.append_log("Landmark confidence updated from camera feed", level="DEBUG", source="perception")

            await asyncio.sleep(0.1)

    def _build_default_state(self) -> Dict[str, Any]:
        subsystems = {
            name: {
                "name": name,
                "icon": spec["icon"],
                "status": spec["status"],
                "detail": spec["detail"],
                "connected": False,
            }
            for name, spec in DEFAULT_SUBSYSTEMS.items()
        }
        return {
            "meta": {
                "robot_name": self.robot_name,
                "breadcrumbs": ["Wheelchair", "main"],
                "live": False,
                "connection_label": "Offline",
                "mode": "demo" if self.fake else "embedded",
                "pending_commands": 0,
                "pending_landmarks": 0,
                "updated_at": self._iso_now(),
            },
            "pose": {"x": 0.0, "y": 0.0, "theta": 0.0, "covariance": None, "frame": "world"},
            "velocity": {"vx": 0.0, "vy": 0.0, "omega": 0.0},
            "imu": {
                "receiving": False,
                "packet_age_s": None,
                "gyro_x": 0.0,
                "gyro_y": 0.0,
                "gyro_z": 0.0,
                "accel_x": 0.0,
                "accel_y": 0.0,
                "accel_z": 0.0,
                "updated_at": None,
            },
            "nav_state": {
                "state": "Idle",
                "mode": "manual",
                "estop": False,
                "serial_connected": False,
                "last_command": None,
            },
            "subsystems": subsystems,
            "modules": self._build_modules(subsystems),
            "lidar_scan": {"xy_points": [], "polar_points": [], "timestamp_monotonic_s": None, "frame": "robot"},
            "path": [],
            "goal": None,
            "obstacles": [],
            "landmarks": [],
            "occupancy_grid": {
                "occupancy": [],
                "resolution_m": 0.1,
                "grid_size": 0,
                "origin_cell": 0,
                "shape": [0, 0],
            },
            "voice": {"transcript": "", "response": "", "updated_at": None},
            "arm_voice": {
                "status": "idle",
                "running": False,
                "mode": "local_fake",
                "pid": None,
                "detail": "Press Start to run the arm voice script.",
                "started_at": None,
                "stopped_at": None,
                "last_exit_code": None,
                "command": "",
                "updated_at": None,
            },
            "pickup": {
                "status": "idle",
                "running": False,
                "detail": "Pickup controller idle.",
                "mode": "viam",
                "attempts": 0,
                "last_detection": None,
                "last_target": None,
                "last_result": None,
                "last_error": "",
                "updated_at": None,
            },
            "logs": [],
            "config": {},
            "config_meta": {
                "hot_reload_supported": False,
                "save_note": "Saving updates the dashboard snapshot only. Live module hot-reload is not wired yet.",
            },
            "waypoints": {},
            "camera": copy.deepcopy(self._camera_meta),
            "foxglove": {
                "status": "disconnected",
                "available": False,
                "port": int(load_config().foxglove.port),
                "viewer_url": "https://app.foxglove.dev/~/view",
                "detail": "Foxglove bridge not started",
                "install_command": "pip install foxglove-websocket",
            },
        }

    def _build_modules(self, subsystems: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "name": name,
                "status": subsystem["status"],
                "connected": subsystem.get("connected", False),
            }
            for name, subsystem in subsystems.items()
        ]

    def _load_config_snapshot(self) -> Dict[str, Any]:
        raw = dataclasses.asdict(load_config())
        return {
            "serial": raw.get("serial", {}),
            "lidar": raw.get("lidar", {}),
            "slam": raw.get("slam", {}),
            "planner": raw.get("planner", {}),
            "safety": raw.get("safety", {}),
            "camera": raw.get("camera", {}),
            "pickup": raw.get("pickup", {}),
            "ekf": raw.get("ekf", {}),
            "phone_imu": raw.get("phone_imu", {}),
        }

    def _set_state_key(self, key: str, value: Any) -> None:
        with self._lock:
            self._state[key] = value
            self._state["meta"]["updated_at"] = self._iso_now()

    async def _start_foxglove_bridge(self) -> None:
        cfg = load_config().foxglove
        self._foxglove = create_foxglove_viz(fake=False, host=cfg.host, port=cfg.port)
        if isinstance(self._foxglove, FakeFoxgloveViz):
            self._set_foxglove_state(
                status="disconnected",
                available=False,
                detail="foxglove-websocket is not installed. Install it to enable embedding.",
                port=cfg.port,
            )
            self.append_log("Foxglove bridge unavailable: install foxglove-websocket", level="WARN", source="foxglove")
            return

        try:
            await self._foxglove.start()
            self._foxglove_task = asyncio.create_task(self._foxglove.run(), name="foxglove-bridge")
            self._set_foxglove_state(
                status="connected",
                available=True,
                detail=f"Foxglove WebSocket streaming on port {cfg.port}",
                port=cfg.port,
            )
            self.append_log("Foxglove bridge started", level="INFO", source="foxglove")
        except Exception as exc:
            logger.warning("Foxglove bridge failed to start: %s", exc)
            self._foxglove = None
            self._foxglove_task = None
            self._set_foxglove_state(
                status="error",
                available=False,
                detail=f"Foxglove bridge failed to start: {exc}",
                port=cfg.port,
            )
            self.append_log(f"Foxglove bridge failed: {exc}", level="WARN", source="foxglove")

    def _set_foxglove_state(
        self,
        *,
        status: str,
        available: bool,
        detail: str,
        port: int,
    ) -> None:
        with self._lock:
            self._state["foxglove"] = {
                "status": status,
                "available": available,
                "port": int(port),
                "viewer_url": "https://app.foxglove.dev/~/view",
                "detail": detail,
                "install_command": "pip install foxglove-websocket",
            }
            self._state["meta"]["updated_at"] = self._iso_now()

    def _has_real_foxglove(self) -> bool:
        return self._foxglove is not None and not isinstance(self._foxglove, FakeFoxgloveViz)

    def _refresh_queue_meta_locked(self) -> None:
        self._state["meta"]["pending_commands"] = len(self._command_queue)
        self._state["meta"]["pending_landmarks"] = len(self._landmark_queue)
        self._state["meta"]["updated_at"] = self._iso_now()

    def _get_voice_field(self, field: str) -> str:
        with self._lock:
            return str(self._state["voice"].get(field, "") or "")

    @staticmethod
    def _extract_logged_text(line: str, marker: str) -> str:
        _, _, suffix = line.partition(marker)
        text = suffix.strip()
        if text.startswith("'") and text.endswith("'") and len(text) >= 2:
            return text[1:-1]
        return text

    def _handle_arm_voice_output_line(self, line: str, *, source: str) -> None:
        if not line or "HTTP Request:" in line:
            return

        if line.startswith("INFO:voice:STT:"):
            self.append_log("Arm voice speech-to-text ready", level="INFO", source="arm_voice")
            return

        if line.startswith("INFO:voice:TTS:"):
            self.append_log("Arm voice text-to-speech ready", level="INFO", source="arm_voice")
            return

        if line.startswith("INFO:arm_control:[FAKE] Arm control connected"):
            self.set_arm_voice_state(status="running", running=True, detail="Fake arm connected; listening")
            self.set_subsystem_status("Voice", "connected", "Arm voice test ready")
            self.append_log("Arm voice fake arm connected", level="INFO", source="arm_voice")
            return

        if line.startswith("INFO:voice:Listening..."):
            self.set_arm_voice_state(status="running", running=True, detail="Listening... speak now")
            self.set_subsystem_status("Voice", "connected", "Arm voice test listening")
            return

        if line.startswith("INFO:voice:Recorded "):
            self.set_arm_voice_state(status="running", running=True, detail="Audio captured; transcribing")
            return

        if "ElevenLabs STT: " in line or "Whisper STT: " in line:
            marker = "ElevenLabs STT: " if "ElevenLabs STT: " in line else "Whisper STT: "
            transcript = self._extract_logged_text(line, marker)
            self.set_voice_state(transcript=transcript, response=self._get_voice_field("response"))
            self.set_arm_voice_state(status="running", running=True, detail="Transcript received")
            self.append_log(f"Arm voice heard: {transcript}", level="INFO", source="arm_voice")
            return

        if line.startswith("Heard: "):
            transcript = line.removeprefix("Heard: ").strip()
            self.set_voice_state(transcript=transcript, response=self._get_voice_field("response"))
            self.set_arm_voice_state(status="running", running=True, detail="Command parsed")
            return

        if "TTS: " in line:
            response = self._extract_logged_text(line, "TTS: ")
            self.set_voice_state(transcript=self._get_voice_field("transcript"), response=response)
            self.set_arm_voice_state(status="running", running=True, detail=response or "Responding")
            return

        if line.startswith("Command: ") or line.startswith("Result: "):
            self.append_log(line, level="INFO", source="arm_voice")
            return

        if line.startswith("Arm setup error:"):
            self.set_arm_voice_state(status="error", running=False, detail=line)
            self.set_subsystem_status("Voice", "error", line)
            self.append_log(line, level="ERROR", source="arm_voice")
            return

        if source == "stderr":
            self.append_log(line, level="WARN", source="arm_voice")

    async def _stream_arm_voice_output(self, stream: Any, *, source: str) -> None:
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").strip()
            if text:
                self._handle_arm_voice_output_line(text, source=source)

    async def _watch_arm_voice_process(self, process: asyncio.subprocess.Process) -> None:
        returncode = await process.wait()
        await asyncio.gather(
            *(task for task in (self._arm_voice_stdout_task, self._arm_voice_stderr_task) if task is not None),
            return_exceptions=True,
        )

        stop_requested = self._arm_voice_stop_requested
        self._arm_voice_stop_requested = False
        self._arm_voice_process = None
        self._arm_voice_stdout_task = None
        self._arm_voice_stderr_task = None
        self._arm_voice_watch_task = None

        if stop_requested:
            detail = "Arm voice test stopped"
            status = "idle"
            level = "INFO"
            subsystem_status = "disconnected"
        elif returncode == 0:
            detail = "Arm voice test exited cleanly"
            status = "idle"
            level = "INFO"
            subsystem_status = "disconnected"
        else:
            detail = f"Arm voice test exited with code {returncode}"
            status = "error"
            level = "ERROR"
            subsystem_status = "error"

        self.set_arm_voice_state(
            status=status,
            running=False,
            pid=None,
            detail=detail,
            stopped_at=self._iso_now(),
            last_exit_code=returncode,
        )
        self.set_subsystem_status("Voice", subsystem_status, detail)
        self.append_log(detail, level=level, source="arm_voice")

    async def _start_arm_voice_demo(self, *, mode: str) -> Dict[str, Any]:
        process = self._arm_voice_process
        if process is not None and process.returncode is None:
            self.set_arm_voice_state(detail="Arm voice test already running")
            return self.get_snapshot()["arm_voice"]

        command = [
            sys.executable,
            str(self._workspace_dir / "laptop" / "arm_voice_demo.py"),
            "--no-confirm",
            "--confirmation-timeout-seconds",
            "3",
            "--max-joint-step",
            "10",
        ]
        if mode == "local_fake":
            command.append("--local-fake")

        self._arm_voice_stop_requested = False
        self.set_arm_voice_state(
            status="starting",
            running=True,
            mode=mode,
            pid=None,
            detail="Launching arm voice test...",
            started_at=self._iso_now(),
            stopped_at=None,
            last_exit_code=None,
            command=" ".join(command),
        )
        self.set_subsystem_status("Voice", "connected", "Arm voice test starting")
        self.append_log(f"Starting arm voice test ({mode})", level="INFO", source="arm_voice")

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(self._workspace_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as exc:
            detail = f"Failed to launch arm voice test: {exc}"
            self.set_arm_voice_state(status="error", running=False, pid=None, detail=detail)
            self.set_subsystem_status("Voice", "error", detail)
            self.append_log(detail, level="ERROR", source="arm_voice")
            return self.get_snapshot()["arm_voice"]

        self._arm_voice_process = process
        self.set_arm_voice_state(
            status="starting",
            running=True,
            pid=process.pid,
            detail="Arm voice process started; waiting for microphone",
        )
        self._arm_voice_stdout_task = asyncio.create_task(
            self._stream_arm_voice_output(process.stdout, source="stdout"),
            name="web-ui-arm-voice-stdout",
        )
        self._arm_voice_stderr_task = asyncio.create_task(
            self._stream_arm_voice_output(process.stderr, source="stderr"),
            name="web-ui-arm-voice-stderr",
        )
        self._arm_voice_watch_task = asyncio.create_task(
            self._watch_arm_voice_process(process),
            name="web-ui-arm-voice-watch",
        )
        return self.get_snapshot()["arm_voice"]

    async def _stop_arm_voice_demo(self, *, detail: str = "Stopping arm voice test...") -> Dict[str, Any]:
        process = self._arm_voice_process
        if process is None or process.returncode is not None:
            self.set_arm_voice_state(
                status="idle",
                running=False,
                pid=None,
                detail="Arm voice test is not running",
            )
            return self.get_snapshot()["arm_voice"]

        self._arm_voice_stop_requested = True
        self.set_arm_voice_state(status="stopping", running=True, detail=detail)
        self.append_log("Stopping arm voice test", level="INFO", source="arm_voice")

        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            process.kill()
            await asyncio.wait_for(process.wait(), timeout=2.0)
        return self.get_snapshot()["arm_voice"]

    def _serialize_grid(
        self,
        occupancy: Any,
        resolution_m: Optional[float],
        grid_size: Optional[int],
        origin_cell: Optional[int],
    ) -> Dict[str, Any]:
        array = np.asarray(occupancy, dtype=np.float32)
        if array.ndim != 2 or array.size == 0:
            return {"occupancy": [], "resolution_m": resolution_m or 0.1, "grid_size": 0, "origin_cell": 0, "shape": [0, 0]}

        step = max(1, int(math.ceil(max(array.shape) / MAX_MAP_DIMENSION)))
        reduced = array[::step, ::step]
        reduced_resolution = float(resolution_m or 0.1) * step
        reduced_origin = int((origin_cell if origin_cell is not None else array.shape[0] // 2) / step)
        return {
            "occupancy": np.round(reduced, 2).tolist(),
            "resolution_m": reduced_resolution,
            "grid_size": int(grid_size or array.shape[0]),
            "origin_cell": reduced_origin,
            "shape": [int(reduced.shape[0]), int(reduced.shape[1])],
            "downsample_step": step,
        }

    def _normalize_pose(self, pose: Any = None, **kwargs: Any) -> Dict[str, Any]:
        payload = {"x": 0.0, "y": 0.0, "theta": 0.0, "covariance": None, "frame": "world"}
        if isinstance(pose, dict):
            payload.update({k: pose[k] for k in payload if k in pose})
        elif isinstance(pose, (tuple, list)) and len(pose) >= 3:
            payload["x"], payload["y"], payload["theta"] = pose[:3]
        elif pose is not None and hasattr(pose, "__iter__"):
            values = list(pose)
            if len(values) >= 3:
                payload["x"], payload["y"], payload["theta"] = values[:3]
        payload.update({k: v for k, v in kwargs.items() if k in payload})
        payload["x"] = float(payload["x"])
        payload["y"] = float(payload["y"])
        payload["theta"] = float(payload["theta"])
        if payload["covariance"] is not None:
            payload["covariance"] = np.asarray(payload["covariance"], dtype=np.float32).tolist()
        return payload

    def _normalize_velocity(self, velocity: Any = None, **kwargs: Any) -> Dict[str, Any]:
        payload = {"vx": 0.0, "vy": 0.0, "omega": 0.0}
        if isinstance(velocity, dict):
            for key in payload:
                if key in velocity:
                    payload[key] = velocity[key]
        elif isinstance(velocity, (tuple, list)) and len(velocity) >= 3:
            payload["vx"], payload["vy"], payload["omega"] = velocity[:3]
        payload.update({k: v for k, v in kwargs.items() if k in payload})
        return {key: float(value) for key, value in payload.items()}

    def _serialize_obstacle(self, obstacle: Any) -> Dict[str, float]:
        if isinstance(obstacle, dict):
            x = obstacle.get("x", 0.0)
            y = obstacle.get("y", 0.0)
            radius = obstacle.get("radius", 0.2)
            confidence = obstacle.get("confidence", 1.0)
        else:
            x = getattr(obstacle, "x", 0.0)
            y = getattr(obstacle, "y", 0.0)
            radius = getattr(obstacle, "radius", 0.2)
            confidence = getattr(obstacle, "confidence", 1.0)
        distance = math.hypot(float(x), float(y))
        angle_deg = math.degrees(math.atan2(float(y), float(x))) if distance else 0.0
        return {
            "x": float(x),
            "y": float(y),
            "radius": float(radius),
            "confidence": float(confidence),
            "distance_m": distance,
            "angle_deg": angle_deg,
        }

    def _serialize_landmark(self, landmark: Any) -> Dict[str, Any]:
        if isinstance(landmark, dict):
            name = landmark.get("name", "landmark")
            x = landmark.get("x", 0.0)
            y = landmark.get("y", 0.0)
            confidence = landmark.get("confidence", 1.0)
            frame = landmark.get("frame", "world")
            source = landmark.get("source", "map")
            descriptor_status = landmark.get("descriptor_status")
        else:
            name = getattr(landmark, "name", "landmark")
            x = getattr(landmark, "x", 0.0)
            y = getattr(landmark, "y", 0.0)
            confidence = getattr(landmark, "confidence", 1.0)
            frame = "robot"
            source = "perception"
            descriptor_status = "registered" if getattr(landmark, "descriptor", None) is not None else "unknown"
        if descriptor_status is None:
            descriptor_status = "pending_capture" if source == "map" else "registered"
        return {
            "name": str(name),
            "x": float(x),
            "y": float(y),
            "confidence": float(confidence),
            "frame": str(frame),
            "source": str(source),
            "descriptor_status": str(descriptor_status),
        }

    def _serialize_point(self, point: Any) -> Dict[str, float]:
        if isinstance(point, dict):
            return {"x": float(point.get("x", 0.0)), "y": float(point.get("y", 0.0))}
        if isinstance(point, (tuple, list)) and len(point) >= 2:
            return {"x": float(point[0]), "y": float(point[1])}
        return {"x": float(getattr(point, "x", 0.0)), "y": float(getattr(point, "y", 0.0))}

    def _coerce_xy_points(self, points: Iterable[Any]) -> List[Dict[str, float]]:
        if isinstance(points, np.ndarray):
            iterable = points.tolist()
        else:
            iterable = list(points)
        result = []
        for point in iterable:
            if isinstance(point, dict):
                x = point.get("x", 0.0)
                y = point.get("y", 0.0)
            else:
                x, y = point[0], point[1]
            result.append({"x": float(x), "y": float(y)})
        return result

    def _coerce_polar_points(self, points: Iterable[Any]) -> List[Dict[str, float]]:
        if isinstance(points, np.ndarray):
            iterable = points.tolist()
        else:
            iterable = list(points)
        result = []
        for point in iterable:
            if isinstance(point, dict):
                angle_deg = point.get("angle_deg", 0.0)
                distance_m = point.get("distance_m", 0.0)
                intensity = point.get("intensity", 0.0)
            else:
                angle_deg = point[0]
                distance_m = point[1]
                intensity = point[2] if len(point) > 2 else 0.0
            result.append(
                {
                    "angle_deg": float(angle_deg),
                    "distance_m": float(distance_m),
                    "intensity": float(intensity),
                }
            )
        return result

    def _extract_frame_meta(self, frame: Any) -> Dict[str, Any]:
        if hasattr(frame, "shape") and len(frame.shape) >= 2:
            return {"height": int(frame.shape[0]), "width": int(frame.shape[1])}
        return {"height": 0, "width": 0}

    def _encode_frame(self, frame: Any) -> Optional[bytes]:
        if frame is None:
            return self._camera_jpeg
        if isinstance(frame, (bytes, bytearray)):
            return bytes(frame)

        try:
            import cv2

            ok, buffer = cv2.imencode(".jpg", frame)
            if ok:
                return buffer.tobytes()
        except Exception:
            pass

        try:
            from PIL import Image

            image_array = np.asarray(frame)
            if image_array.ndim == 3 and image_array.shape[2] == 3:
                image_array = image_array[:, :, ::-1]
            image = Image.fromarray(image_array.astype(np.uint8))
            output = io.BytesIO()
            image.save(output, format="JPEG")
            return output.getvalue()
        except Exception:
            return PLACEHOLDER_JPEG

    def _merge_config(self, current: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
        merged = copy.deepcopy(current)
        for key, value in incoming.items():
            if key not in merged:
                continue
            if isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_config(merged[key], value)
            else:
                merged[key] = self._cast_config_value(merged[key], value)
        return merged

    def _cast_config_value(self, current: Any, value: Any) -> Any:
        if isinstance(current, bool):
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
        if isinstance(current, int) and not isinstance(current, bool):
            return int(value)
        if isinstance(current, float):
            return float(value)
        return value

    def _build_demo_grid(self, size: int = 96) -> np.ndarray:
        grid = np.full((size, size), 0.35, dtype=np.float32)
        grid[5:-5, 5] = 0.9
        grid[5:-5, -6] = 0.9
        grid[5, 5:-5] = 0.9
        grid[-6, 5:-5] = 0.9
        grid[20:75, 28] = 0.85
        grid[18, 18:48] = 0.75
        grid[60:65, 60:78] = 0.85
        grid[40:45, 50:55] = 0.95
        grid[25:29, 68:73] = 0.95
        return grid

    def _build_demo_path(self, pose: Tuple[float, float], goal: Tuple[float, float]) -> List[Dict[str, float]]:
        points = []
        for idx in range(15):
            alpha = idx / 14.0
            curve = math.sin(alpha * math.pi) * 0.2
            points.append(
                {
                    "x": pose[0] * (1.0 - alpha) + goal[0] * alpha,
                    "y": pose[1] * (1.0 - alpha) + goal[1] * alpha + curve,
                }
            )
        return points

    def _build_demo_obstacles(self, t: float) -> List[Dict[str, float]]:
        return [
            {"x": 1.3 + 0.2 * math.sin(t), "y": 0.5, "radius": 0.24, "confidence": 0.92},
            {"x": 2.1, "y": -0.7 + 0.15 * math.cos(t * 0.7), "radius": 0.19, "confidence": 0.88},
            {"x": 0.9, "y": 1.0 + 0.12 * math.sin(t * 0.5), "radius": 0.22, "confidence": 0.81},
        ]

    def _build_demo_landmarks(self) -> List[Dict[str, float]]:
        return [
            {"name": "door", "x": 3.0, "y": 0.0, "confidence": 0.97, "frame": "world", "source": "perception", "descriptor_status": "registered"},
            {"name": "table", "x": 1.5, "y": 2.0, "confidence": 0.91, "frame": "world", "source": "perception", "descriptor_status": "registered"},
            {"name": "window", "x": 0.0, "y": 4.0, "confidence": 0.89, "frame": "world", "source": "perception", "descriptor_status": "registered"},
        ]

    def _build_demo_scan_points(self, t: float, obstacles: Sequence[Dict[str, Any]]) -> List[Dict[str, float]]:
        points = []
        for angle_deg in range(-90, 91, 4):
            angle_rad = math.radians(angle_deg)
            distance = 3.0 + 0.2 * math.sin(t + angle_rad * 2.0)
            points.append({"x": distance * math.cos(angle_rad), "y": distance * math.sin(angle_rad)})
        for obstacle in obstacles:
            for offset in np.linspace(-0.15, 0.15, 9):
                points.append({"x": obstacle["x"] + offset, "y": obstacle["y"] + offset * 0.3})
        return points

    def _build_demo_camera_frame(self, t: float, obstacles: Sequence[Dict[str, Any]]) -> np.ndarray:
        width, height = 640, 360
        frame = np.full((height, width, 3), 245, dtype=np.uint8)
        frame[:, :, 1] = 248
        frame[:, :, 2] = 250
        frame[260:, :, :] = np.array([230, 232, 236], dtype=np.uint8)
        frame[120:125, :, :] = np.array([210, 214, 220], dtype=np.uint8)
        for idx in range(8):
            x0 = 50 + idx * 70
            frame[0:260, x0:x0 + 2, :] = np.array([220, 223, 228], dtype=np.uint8)

        for obstacle in obstacles:
            distance = max(0.5, math.hypot(obstacle["x"], obstacle["y"]))
            angle = math.atan2(obstacle["y"], obstacle["x"])
            center_x = int((0.5 - angle / math.radians(60.0)) * width)
            center_y = int(height * 0.72)
            box_w = int(max(24, 120 / distance))
            box_h = int(max(36, 150 / distance))
            x1 = max(0, min(width - 1, center_x - box_w // 2))
            x2 = max(0, min(width - 1, center_x + box_w // 2))
            y1 = max(0, min(height - 1, center_y - box_h))
            y2 = max(0, min(height - 1, center_y))
            frame[y1:y2, x1:x1 + 3, :] = np.array([40, 167, 69], dtype=np.uint8)
            frame[y1:y2, x2 - 3:x2, :] = np.array([40, 167, 69], dtype=np.uint8)
            frame[y1:y1 + 3, x1:x2, :] = np.array([40, 167, 69], dtype=np.uint8)
            frame[y2 - 3:y2, x1:x2, :] = np.array([40, 167, 69], dtype=np.uint8)

        pulse = int((math.sin(t * 2.0) + 1.0) * 18)
        frame[22:42, 22:42, :] = np.array([40, 167 + pulse, 69], dtype=np.uint8)
        return frame

    @staticmethod
    def _iso_now() -> str:
        return datetime.now().isoformat(timespec="seconds")


async def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    ui = create_web_ui(fake=True)
    url = await ui.start(host="127.0.0.1", port=8766)
    print(f"Dashboard running at {url}")
    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        await ui.stop()


if __name__ == "__main__":
    asyncio.run(_main())
