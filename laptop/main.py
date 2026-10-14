"""
Main entry point — orchestrates the autonomous wheelchair system.

Full pipeline: voice command → LLM → planner → SLAM → motors → spoken response.

Modes:
  python main.py                   # full system
  python main.py --teleop          # keyboard teleop (no planner/voice)
  python main.py --lidar-only      # LiDAR + SLAM + viz, no motors
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_this_dir = Path(__file__).resolve().parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

from config import load_config
from lidar import create_lidar
from slam import SLAM
from planner import Planner
from pose_estimator import PoseEstimator
from serial_bridge import create_serial_bridge
from spatial_memory import SpatialMemory
from foxglove_viz import create_foxglove_viz
from perception import create_perception
from phone_imu import create_phone_imu
from voice import create_voice_interface
from llm_interface import create_llm_interface
from waypoint_resolver import WaypointResolver

logger = logging.getLogger("main")


class WheelchairSystem:
    """Top-level orchestrator tying all subsystems together."""

    def __init__(self, teleop: bool = False, lidar_only: bool = False,
                 lidar_port: Optional[str] = None,
                 serial_port: Optional[str] = None):
        self._teleop = teleop
        self._lidar_only = lidar_only

        cfg = load_config()

        if lidar_port:
            cfg.lidar.port = lidar_port
        if serial_port:
            cfg.serial.port = serial_port

        # --- Core subsystems ---
        self.lidar = create_lidar(port=cfg.lidar.port)
        self.slam = SLAM()
        self.pose = PoseEstimator()
        self.planner = Planner()
        self.spatial = SpatialMemory(
            self.slam.occupancy_grid.grid_size,
            self.slam.occupancy_grid.resolution,
        )

        # Serial bridge to Arduino (skip if lidar-only)
        self._use_motors = not lidar_only
        if self._use_motors:
            self.serial = create_serial_bridge()
        else:
            self.serial = create_serial_bridge(fake=True)

        # Foxglove viz
        self.viz = create_foxglove_viz()

        # --- Perception (camera) ---
        self.perception = create_perception()

        # --- Phone IMU (optional, UDP) ---
        self.imu = create_phone_imu()

        # --- Voice + LLM (skip in teleop/lidar-only) ---
        self._use_voice = not teleop and not lidar_only
        self.voice = None
        self.llm = None

        # State
        self._running = False
        self._estop = False
        self._cmd_vx = 0.0
        self._cmd_vy = 0.0
        self._cmd_omega = 0.0
        self._nav_state = "idle"
        self._tick_count = 0
        self._last_tick = time.monotonic()
        self._scan_ts = 0.0            # arrival time of the scan behind the current cmd
        self._last_latency_log = 0.0

    async def start(self) -> None:
        logger.info("Starting wheelchair system (teleop=%s, lidar_only=%s)",
                     self._teleop, self._lidar_only)

        await self.lidar.start()

        if self._use_motors:
            try:
                await self.serial.start()
            except Exception as e:
                logger.warning("Serial bridge failed: %s — running without motors", e)
                self._use_motors = False

        try:
            await self.viz.start()
        except Exception as e:
            logger.warning("Foxglove viz failed: %s", e)
            self.viz = create_foxglove_viz(fake=True)
            await self.viz.start()

        # Perception (camera) — non-blocking, don't fail if no camera
        try:
            await self.perception.start()
        except Exception as e:
            logger.warning("Perception failed to start: %s — running without camera", e)
            self.perception = create_perception(fake=True)
            await self.perception.start()

        # Phone IMU — non-blocking, it's optional
        try:
            await self.imu.start()
            logger.info("Phone IMU listener started")
        except Exception as e:
            logger.warning("Phone IMU failed: %s — running without IMU", e)

        # Voice + LLM
        if self._use_voice:
            try:
                self.voice = create_voice_interface()
                await self.voice.start()
                # Build waypoints from spatial memory landmarks + hardcoded waypoints
                waypoints = self._build_waypoints()
                # Cached waypoints / YOLO-labelled landmarks first, LLM on a miss
                self.llm = WaypointResolver(
                    waypoints, create_llm_interface(waypoints=waypoints),
                    detections=self.perception.get_yolo_detections,
                    lookup=self.spatial.lookup_landmark,
                    resolution_m=self.slam.occupancy_grid.resolution,
                )
                logger.info("Voice + LLM ready (%d known locations)", len(waypoints))
            except Exception as e:
                logger.warning("Voice/LLM failed: %s — voice commands unavailable", e)
                self.voice = None
                self.llm = None

        # Load saved spatial memory
        try:
            self.spatial.load()
            logger.info("Loaded saved spatial memory")
        except Exception:
            logger.info("No saved spatial memory, starting fresh")

        self._running = True
        logger.info("All subsystems started")

    def _build_waypoints(self) -> dict:
        """Merge hardcoded demo waypoints with learned landmarks."""
        waypoints = {}
        # Hardcoded fallback
        demo_path = Path(__file__).resolve().parent.parent / "waypoints" / "demo_space.json"
        if demo_path.exists():
            try:
                with open(demo_path) as f:
                    for name, coords in json.load(f).items():
                        waypoints[name] = coords
            except Exception:
                pass
        # Learned landmarks from spatial memory
        for name in self.spatial.get_all_landmark_names():
            coords = self.spatial.lookup_landmark(name)
            if coords:
                waypoints[name] = list(coords)
        return waypoints

    async def run(self) -> None:
        tasks = [
            asyncio.create_task(self.lidar.run(), name="lidar"),
            asyncio.create_task(self.viz.run(), name="foxglove"),
            asyncio.create_task(self.perception.run(), name="perception"),
            asyncio.create_task(self._nav_loop(), name="nav_loop"),
        ]

        if self._teleop:
            tasks.append(asyncio.create_task(self._teleop_loop(), name="teleop"))

        if self._use_motors:
            tasks.append(asyncio.create_task(self._serial_loop(), name="serial"))

        if self._use_voice and self.voice is not None:
            tasks.append(asyncio.create_task(self._voice_loop(), name="voice"))

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        self._running = False
        logger.info("Shutting down...")

        if self._use_motors:
            try:
                await self.serial.send_velocity(0, 0, 0)
            except Exception:
                pass

        await self.lidar.stop()
        if self._use_motors:
            await self.serial.stop()
        await self.viz.stop()
        await self.perception.stop()

        if self.voice:
            try:
                await self.voice.stop()
            except Exception:
                pass

        # Save spatial memory
        try:
            self.spatial.save()
            logger.info("Spatial memory saved")
        except Exception as e:
            logger.warning("Failed to save spatial memory: %s", e)

        logger.info("Shutdown complete")

    # ------------------------------------------------------------------
    # Voice command loop
    # ------------------------------------------------------------------

    async def _voice_loop(self) -> None:
        """Listen → LLM parse → set goal → speak response. Runs continuously."""
        await self.voice.speak("Ready. Where would you like to go?")

        while self._running:
            try:
                transcript = await self.voice.listen()
                if not transcript:
                    continue

                logger.info("Heard: %s", transcript)

                # Parse with LLM
                command, speech = await self.llm.parse(transcript)
                logger.info("Command: %s | Response: %s", command, speech)

                # Speak response while we act
                asyncio.create_task(self.voice.speak(speech))

                cmd = command.get("cmd", "unknown")

                if cmd == "goto":
                    x, y = float(command["x"]), float(command["y"])
                    self.planner.set_goal(x, y)
                    self._nav_state = "navigating"
                    logger.info("Goal set: (%.2f, %.2f)", x, y)
                    # Wait for arrival, then announce
                    asyncio.create_task(self._wait_and_announce_arrival())

                elif cmd == "move":
                    direction = command.get("direction", "forward")
                    distance = float(command.get("distance", 1.0))
                    pose = self.pose.get_pose()
                    dx, dy = self._direction_to_delta(direction, distance, pose[2])
                    self.planner.set_goal(pose[0] + dx, pose[1] + dy)
                    self._nav_state = "navigating"
                    asyncio.create_task(self._wait_and_announce_arrival())

                elif cmd == "stop":
                    self.planner.cancel()
                    self._cmd_vx = 0.0
                    self._cmd_vy = 0.0
                    self._cmd_omega = 0.0
                    self._nav_state = "idle"

            except Exception as e:
                logger.error("Voice loop error: %s", e)
                await asyncio.sleep(1.0)

    async def _wait_and_announce_arrival(self) -> None:
        """Background task: wait for planner to reach goal, then speak."""
        while self._running and self._nav_state == "navigating":
            if self.planner.reached_goal():
                self._nav_state = "arrived"
                if self.voice:
                    await self.voice.speak("I've arrived.")
                return
            await asyncio.sleep(0.5)

    @staticmethod
    def _direction_to_delta(direction: str, distance: float,
                            theta: float) -> Tuple[float, float]:
        """Convert a relative direction + distance to world-frame (dx, dy)."""
        if direction == "forward":
            return (distance * math.cos(theta), distance * math.sin(theta))
        elif direction == "back":
            return (-distance * math.cos(theta), -distance * math.sin(theta))
        elif direction == "left":
            return (distance * math.cos(theta + math.pi / 2),
                    distance * math.sin(theta + math.pi / 2))
        elif direction == "right":
            return (distance * math.cos(theta - math.pi / 2),
                    distance * math.sin(theta - math.pi / 2))
        return (0.0, 0.0)

    # ------------------------------------------------------------------
    # Core navigation loop (10-20 Hz)
    # ------------------------------------------------------------------

    async def _nav_loop(self) -> None:
        cfg = load_config()
        period = 1.0 / cfg.planner.control_rate_hz

        while self._running:
            t0 = time.monotonic()
            dt = t0 - self._last_tick
            self._last_tick = t0
            self._tick_count += 1

            # --- 1. Get LiDAR scan ---
            scan = self.lidar.get_scan()
            scan_xy = self.lidar.get_points_xy()

            if scan_xy is None or scan_xy.shape[0] == 0:
                await asyncio.sleep(0.02)
                continue
            self._scan_ts = scan.timestamp if scan is not None else 0.0

            # --- 2. Get current pose ---
            pose = self.pose.get_pose()

            # --- 3. Phone IMU update (if receiving) ---
            if self.imu.is_receiving():
                gyro_z = self.imu.get_gyro_z()
                self.pose.update_imu(gyro_z, dt)
                pose = self.pose.get_pose()

            # --- 4. SLAM: update occupancy grid + scan match ---
            corrected = self.slam.process_scan(scan_xy, pose)
            if corrected is not None:
                self.pose.update_scan_match(*corrected)
                pose = self.pose.get_pose()

            # --- 5. Feed camera landmarks into spatial memory ---
            # Landmarks are in robot-local frame — transform to world
            cos_t = math.cos(pose[2])
            sin_t = math.sin(pose[2])
            for landmark in self.perception.get_landmarks():
                if landmark.name:
                    wx = pose[0] + cos_t * landmark.x - sin_t * landmark.y
                    wy = pose[1] + sin_t * landmark.x + cos_t * landmark.y
                    self.spatial.update_landmark(
                        landmark.name, wx, wy, landmark.descriptor,
                    )

            # --- 6. Planning (if not teleop) ---
            if not self._teleop and not self._estop:
                if self.planner._goal is not None:
                    self._nav_state = "navigating"
                    vx, vy, omega = self.planner.update(
                        pose, self.slam.occupancy_grid, [])

                    if self.planner.reached_goal():
                        self._nav_state = "arrived"
                        vx, vy, omega = 0.0, 0.0, 0.0
                else:
                    if self._nav_state != "arrived":
                        self._nav_state = "idle"
                    vx, vy, omega = 0.0, 0.0, 0.0

                self._cmd_vx = vx
                self._cmd_vy = vy
                self._cmd_omega = omega

            # --- 7. Safety: LiDAR hard stop ---
            if scan_xy is not None and scan_xy.shape[0] > 0:
                dists = np.linalg.norm(scan_xy, axis=1)
                min_dist = float(dists.min())
                if min_dist < cfg.safety.emergency_stop_distance_m:
                    if self._nav_state == "navigating":
                        logger.warning("LiDAR hard stop! Obstacle at %.2f m", min_dist)
                    self._cmd_vx = 0.0
                    self._cmd_vy = 0.0
                    self._cmd_omega = 0.0

            # --- 8. E-stop check ---
            if self._estop:
                self._cmd_vx = 0.0
                self._cmd_vy = 0.0
                self._cmd_omega = 0.0
                self._nav_state = "estop"

            # --- 9. Update viz ---
            self.viz.set_pose(*pose)
            self.viz.set_lidar_scan(scan_xy)
            self.viz.set_motor_cmd(self._cmd_vx, self._cmd_vy, self._cmd_omega)
            self.viz.set_nav_state(self._nav_state)
            self.viz.set_estop(self._estop)
            self.viz.set_occupancy_grid(
                self.slam.get_occupancy_array(),
                self.slam.occupancy_grid.resolution,
            )

            if self.planner.global_path:
                self.viz.set_path(self.planner.global_path)

            frame = self.perception.get_frame()
            if frame is not None:
                self.viz.set_camera_frame(frame)

            # Log periodically
            if self._tick_count % 100 == 0:
                logger.info(
                    "tick=%d pose=(%.2f, %.2f, %.0f) state=%s vel=(%.2f, %.2f, %.2f) pts=%d",
                    self._tick_count, pose[0], pose[1], math.degrees(pose[2]),
                    self._nav_state,
                    self._cmd_vx, self._cmd_vy, self._cmd_omega,
                    scan_xy.shape[0],
                )

            elapsed = time.monotonic() - t0
            await asyncio.sleep(max(0, period - elapsed))

    # ------------------------------------------------------------------
    # Serial loop: send commands, read feedback
    # ------------------------------------------------------------------

    async def _serial_loop(self) -> None:
        while self._running:
            try:
                await self.serial.send_velocity(
                    self._cmd_vx, self._cmd_vy, self._cmd_omega)

                # Scan arrival -> serial send, logged every 2 s (measured, not assumed)
                now = time.monotonic()
                if self._scan_ts and now - self._last_latency_log > 2.0:
                    self._last_latency_log = now
                    logger.info("scan-to-motor latency: %.0f ms", (now - self._scan_ts) * 1e3)

                resp = await self.serial.read_response()
                if resp:
                    self._estop = self.serial.get_estop()
                    ticks = self.serial.get_encoder_ticks()
                    self.pose.predict(tuple(ticks), 0.05)
            except Exception as e:
                logger.error("Serial error: %s", e)

            await asyncio.sleep(0.05)

    # ------------------------------------------------------------------
    # Teleop: keyboard control
    # ------------------------------------------------------------------

    async def _teleop_loop(self) -> None:
        print("\n=== TELEOP MODE ===")
        print("  W/S = forward/back   A/D = strafe   Q/E = rotate")
        print("  SPACE = stop   X = quit   G = enter goal coords\n")

        loop = asyncio.get_event_loop()
        cfg = load_config().velocity

        while self._running:
            try:
                line = await asyncio.wait_for(
                    loop.run_in_executor(None, sys.stdin.readline),
                    timeout=0.1,
                )
                line = line.strip().lower()
                if not line:
                    continue

                if line == "w":
                    self._cmd_vx, self._cmd_vy, self._cmd_omega = cfg.max_vx * 0.5, 0.0, 0.0
                elif line == "s":
                    self._cmd_vx, self._cmd_vy, self._cmd_omega = -cfg.max_vx * 0.5, 0.0, 0.0
                elif line == "a":
                    self._cmd_vx, self._cmd_vy, self._cmd_omega = 0.0, cfg.max_vy * 0.5, 0.0
                elif line == "d":
                    self._cmd_vx, self._cmd_vy, self._cmd_omega = 0.0, -cfg.max_vy * 0.5, 0.0
                elif line == "q":
                    self._cmd_vx, self._cmd_vy, self._cmd_omega = 0.0, 0.0, cfg.max_omega * 0.5
                elif line == "e":
                    self._cmd_vx, self._cmd_vy, self._cmd_omega = 0.0, 0.0, -cfg.max_omega * 0.5
                elif line in (" ", "space"):
                    self._cmd_vx = self._cmd_vy = self._cmd_omega = 0.0
                elif line == "x":
                    self._running = False
                elif line == "g":
                    print("Enter goal x y: ", end="", flush=True)
                    coords = await loop.run_in_executor(None, sys.stdin.readline)
                    try:
                        x, y = map(float, coords.strip().split())
                        self.planner.set_goal(x, y)
                        self._teleop = False
                        print(f"Goal set: ({x}, {y}) — switching to autonomous")
                    except ValueError:
                        print("Invalid coords")
                else:
                    self._cmd_vx = self._cmd_vy = self._cmd_omega = 0.0

            except asyncio.TimeoutError:
                continue

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_goal(self, x: float, y: float) -> None:
        self.planner.set_goal(x, y)
        self._nav_state = "navigating"

    def cancel_goal(self) -> None:
        self.planner.cancel()
        self._nav_state = "idle"


def main():
    parser = argparse.ArgumentParser(description="Autonomous wheelchair system")
    parser.add_argument("--teleop", action="store_true",
                        help="Keyboard teleop mode (no voice/planner)")
    parser.add_argument("--lidar-only", action="store_true",
                        help="LiDAR + SLAM only, no motors")
    parser.add_argument("--lidar-port", type=str, default=None,
                        help="Override LiDAR serial port (e.g. COM3)")
    parser.add_argument("--serial-port", type=str, default=None,
                        help="Override Arduino serial port (e.g. COM4)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)-18s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    system = WheelchairSystem(
        teleop=args.teleop,
        lidar_only=args.lidar_only,
        lidar_port=args.lidar_port,
        serial_port=args.serial_port,
    )

    async def run():
        await system.start()
        try:
            await system.run()
        finally:
            await system.stop()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


if __name__ == "__main__":
    main()
