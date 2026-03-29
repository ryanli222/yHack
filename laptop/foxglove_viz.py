"""
Foxglove WebSocket server — publishes robot data for Foxglove Studio.

Runs alongside the web UI. Foxglove Studio connects via ws://localhost:8765
and subscribes to channels (pose, lidar, grid, path, camera).

Can be opened from the web dashboard via a Foxglove deep link:
  https://app.foxglove.dev/~/view?ds=foxglove-websocket&ds.url=ws://localhost:8765

Provides:
  - FoxgloveViz: real publisher using foxglove-websocket
  - FakeFoxgloveViz: no-op stub
  - create_foxglove_viz(): factory

Install: pip install foxglove-websocket
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import struct
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from config import load_config
except ImportError:  # pragma: no cover - useful if imported as a namespace package
    from laptop.config import load_config

logger = logging.getLogger(__name__)


async def _maybe_await(result):
    if inspect.isawaitable(result):
        return await result
    return result


class FoxgloveViz:
    """Publishes robot data over the Foxglove WebSocket protocol."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self._host = host
        self._port = port
        self._server = None
        self._running = False

        # Cached state
        self._pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._obstacles: List[Tuple[float, float]] = []
        self._path: List[Tuple[float, float]] = []
        self._lidar_xy: Optional[np.ndarray] = None
        self._grid: Optional[np.ndarray] = None
        self._grid_resolution: float = 0.05
        self._motor_cmd: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._nav_state: str = "idle"
        self._estop: bool = False
        self._camera_frame: Optional[np.ndarray] = None

    async def start(self) -> None:
        try:
            from foxglove_websocket import FoxgloveServer
        except ImportError:
            try:
                from foxglove_websocket.server import FoxgloveServer
            except ImportError:
                logger.error(
                    "foxglove-websocket not installed. Install with: "
                    "pip install foxglove-websocket"
                )
                raise

        try:
            from foxglove_websocket.types import ChannelWithoutId
        except ImportError:
            from foxglove_websocket.server import ChannelWithoutId

        self._server = FoxgloveServer(
            self._host, self._port, "Wheelchair Foxglove Bridge"
        )

        # Define channels
        self._ch_pose = await self._server.add_channel(
            ChannelWithoutId(
                topic="/pose",
                encoding="json",
                schemaName="foxglove.LocationFix",
                schema=json.dumps({
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "theta": {"type": "number"},
                    },
                }),
            )
        )

        self._ch_scan = await self._server.add_channel(
            ChannelWithoutId(
                topic="/lidar",
                encoding="json",
                schemaName="foxglove.PointCloud",
                schema=json.dumps({
                    "type": "object",
                    "properties": {
                        "points": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "x": {"type": "number"},
                                    "y": {"type": "number"},
                                },
                            },
                        },
                    },
                }),
            )
        )

        self._ch_path = await self._server.add_channel(
            ChannelWithoutId(
                topic="/path",
                encoding="json",
                schemaName="foxglove.LinePrimitive",
                schema=json.dumps({
                    "type": "object",
                    "properties": {
                        "points": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "x": {"type": "number"},
                                    "y": {"type": "number"},
                                },
                            },
                        },
                    },
                }),
            )
        )

        self._ch_obstacles = await self._server.add_channel(
            ChannelWithoutId(
                topic="/obstacles",
                encoding="json",
                schemaName="foxglove.CircleAnnotation",
                schema=json.dumps({
                    "type": "object",
                    "properties": {
                        "obstacles": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "x": {"type": "number"},
                                    "y": {"type": "number"},
                                },
                            },
                        },
                    },
                }),
            )
        )

        self._ch_status = await self._server.add_channel(
            ChannelWithoutId(
                topic="/status",
                encoding="json",
                schemaName="foxglove.Log",
                schema=json.dumps({
                    "type": "object",
                    "properties": {
                        "nav_state": {"type": "string"},
                        "estop": {"type": "boolean"},
                        "vx": {"type": "number"},
                        "vy": {"type": "number"},
                        "omega": {"type": "number"},
                    },
                }),
            )
        )

        await _maybe_await(self._server.start())
        self._running = True
        logger.info("Foxglove server running on ws://%s:%d", self._host, self._port)

    async def run(self) -> None:
        """Publish state at 10 Hz."""
        while self._running:
            now_ns = int(time.time() * 1e9)

            try:
                # Pose
                await self._server.send_message(
                    self._ch_pose, now_ns,
                    json.dumps({
                        "x": self._pose[0],
                        "y": self._pose[1],
                        "theta": self._pose[2],
                    }).encode(),
                )

                # LiDAR scan
                if self._lidar_xy is not None and len(self._lidar_xy) > 0:
                    pts = [{"x": float(p[0]), "y": float(p[1])}
                           for p in self._lidar_xy[:500]]  # cap for perf
                    await self._server.send_message(
                        self._ch_scan, now_ns,
                        json.dumps({"points": pts}).encode(),
                    )

                # Path
                if self._path:
                    pts = [{"x": p[0], "y": p[1]} for p in self._path]
                    await self._server.send_message(
                        self._ch_path, now_ns,
                        json.dumps({"points": pts}).encode(),
                    )

                # Obstacles
                if self._obstacles:
                    obs = [{"x": o[0], "y": o[1]} for o in self._obstacles]
                    await self._server.send_message(
                        self._ch_obstacles, now_ns,
                        json.dumps({"obstacles": obs}).encode(),
                    )

                # Status
                await self._server.send_message(
                    self._ch_status, now_ns,
                    json.dumps({
                        "nav_state": self._nav_state,
                        "estop": self._estop,
                        "vx": self._motor_cmd[0],
                        "vy": self._motor_cmd[1],
                        "omega": self._motor_cmd[2],
                    }).encode(),
                )

            except Exception as exc:
                logger.debug("Foxglove publish error: %s", exc)

            await asyncio.sleep(0.1)  # 10 Hz

    # ------------------------------------------------------------------
    # Setters (called from main.py nav loop)
    # ------------------------------------------------------------------

    def set_pose(self, x: float, y: float, theta: float) -> None:
        self._pose = (x, y, theta)

    def set_obstacles(self, obstacles: Sequence) -> None:
        self._obstacles = [(o[0], o[1]) if isinstance(o, (tuple, list))
                           else (o.x, o.y) if hasattr(o, 'x')
                           else (o["x"], o["y"]) for o in obstacles]

    def set_path(self, path: Sequence) -> None:
        self._path = [(p[0], p[1]) if isinstance(p, (tuple, list))
                      else (p.x, p.y) if hasattr(p, 'x')
                      else (p["x"], p["y"]) for p in path]

    def set_lidar_scan(self, scan_xy) -> None:
        if isinstance(scan_xy, np.ndarray):
            self._lidar_xy = scan_xy
        elif hasattr(scan_xy, 'as_xy'):
            self._lidar_xy = scan_xy.as_xy()
        else:
            self._lidar_xy = np.array(scan_xy, dtype=np.float32) if scan_xy else None

    def set_motor_cmd(self, vx: float, vy: float, omega: float) -> None:
        self._motor_cmd = (vx, vy, omega)

    def set_nav_state(self, state: str) -> None:
        self._nav_state = state

    def set_estop(self, active: bool) -> None:
        self._estop = active

    def set_occupancy_grid(self, grid: np.ndarray, resolution: float = 0.05) -> None:
        self._grid = grid
        self._grid_resolution = resolution

    def set_camera_frame(self, frame: np.ndarray) -> None:
        self._camera_frame = frame

    async def stop(self) -> None:
        self._running = False
        if self._server:
            await _maybe_await(self._server.close())
        logger.info("Foxglove server stopped")


class FakeFoxgloveViz:
    """No-op stub when foxglove-websocket is not installed."""

    def set_pose(self, *a, **kw): pass
    def set_obstacles(self, *a, **kw): pass
    def set_path(self, *a, **kw): pass
    def set_lidar_scan(self, *a, **kw): pass
    def set_motor_cmd(self, *a, **kw): pass
    def set_nav_state(self, *a, **kw): pass
    def set_estop(self, *a, **kw): pass
    def set_occupancy_grid(self, *a, **kw): pass
    def set_camera_frame(self, *a, **kw): pass

    async def start(self): logger.info("[FAKE] Foxglove viz started")
    async def run(self):
        while True:
            await asyncio.sleep(3600)
    async def stop(self): pass


def create_foxglove_viz(fake: bool = False, host: str = "0.0.0.0", port: int = 8765):
    if fake:
        return FakeFoxgloveViz()
    try:
        import foxglove_websocket  # noqa: F401
    except ImportError:
        logger.warning("foxglove-websocket not installed - using FakeFoxgloveViz")
        return FakeFoxgloveViz()
    cfg = load_config().foxglove
    return FoxgloveViz(host=host or cfg.host, port=port or cfg.port)
