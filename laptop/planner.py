"""
Path planner for the autonomous wheelchair.

Three stages:
  1. A* global planner on the occupancy grid  (replans every ~2 s)
  2. DWA + potential-field local planner       (runs every update tick)
  3. Jerk-limited velocity smoother            (passenger comfort)

Usage:
    planner = Planner()
    planner.set_goal(3.0, 1.5)
    vx, vy, omega = planner.update(pose, occupancy_grid, obstacles)
"""

from __future__ import annotations

import heapq
import logging
import math
import time
from typing import List, Optional, Tuple

import numpy as np

from config import load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# A* global planner
# ---------------------------------------------------------------------------

def _astar(grid_binary: np.ndarray, start: Tuple[int, int],
           goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """A* on a boolean occupancy grid.  True = occupied / impassable.

    Returns list of (gx, gy) grid cells from start to goal, or None.
    Uses 8-connected neighbours with diagonal cost sqrt(2).
    """
    rows, cols = grid_binary.shape

    # Inflate obstacles by ~2 cells so the robot does not clip corners.
    # scipy is already a numpy dependency in practice; fall back to manual
    # dilation if unavailable.
    try:
        from scipy.ndimage import binary_dilation
        kernel = np.ones((5, 5), dtype=bool)
        inflated = binary_dilation(grid_binary, structure=kernel)
    except ImportError:
        # Poor-man 1-cell inflation via shift-or
        inflated = grid_binary.copy()
        inflated[1:, :] |= grid_binary[:-1, :]
        inflated[:-1, :] |= grid_binary[1:, :]
        inflated[:, 1:] |= grid_binary[:, :-1]
        inflated[:, :-1] |= grid_binary[:, 1:]

    sx, sy = start
    gx, gy = goal

    if not (0 <= sx < cols and 0 <= sy < rows):
        return None
    if not (0 <= gx < cols and 0 <= gy < rows):
        return None

    # If start is inside inflated zone, clear it so search can begin
    if inflated[sy, sx]:
        inflated[sy, sx] = False
    # If goal is inside inflated zone but not an actual wall, allow it
    if inflated[gy, gx] and not grid_binary[gy, gx]:
        inflated[gy, gx] = False
    # If goal is a hard wall, no path possible
    if grid_binary[gy, gx]:
        return None

    SQRT2 = 1.4142135623730951
    # 8 neighbours: (dx, dy, cost)
    NEIGHBOURS = (
        (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2),
    )

    # Heuristic: octile distance (admissible, consistent)
    def h(x: int, y: int) -> float:
        dx = abs(x - gx)
        dy = abs(y - gy)
        return max(dx, dy) + (SQRT2 - 1.0) * min(dx, dy)

    # Priority queue entries: (f, g, x, y)
    open_set: list = []
    heapq.heappush(open_set, (h(sx, sy), 0.0, sx, sy))
    came_from: dict = {}
    g_score = np.full((rows, cols), np.inf, dtype=np.float64)
    g_score[sy, sx] = 0.0
    closed = np.zeros((rows, cols), dtype=bool)

    while open_set:
        _f, g, cx, cy = heapq.heappop(open_set)
        if closed[cy, cx]:
            continue
        closed[cy, cx] = True

        if cx == gx and cy == gy:
            # Reconstruct path
            path: list = [(cx, cy)]
            while (cx, cy) in came_from:
                cx, cy = came_from[(cx, cy)]
                path.append((cx, cy))
            path.reverse()
            return path

        for dx, dy, cost in NEIGHBOURS:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < cols and 0 <= ny < rows):
                continue
            if inflated[ny, nx]:
                continue
            if closed[ny, nx]:
                continue
            ng = g + cost
            if ng < g_score[ny, nx]:
                g_score[ny, nx] = ng
                came_from[(nx, ny)] = (cx, cy)
                heapq.heappush(open_set, (ng + h(nx, ny), ng, nx, ny))

    return None  # no path found


def _simplify_path(path: List[Tuple[int, int]],
                   max_points: int = 50) -> List[Tuple[int, int]]:
    """Down-sample a grid-cell path to at most *max_points* via uniform spacing."""
    if len(path) <= max_points:
        return path
    indices = np.round(np.linspace(0, len(path) - 1, max_points)).astype(int)
    return [path[int(i)] for i in indices]


# ---------------------------------------------------------------------------
# Helper conversions
# ---------------------------------------------------------------------------

def _world_to_grid(wx: float, wy: float, resolution: float,
                   origin: int) -> Tuple[int, int]:
    gx = int(round(wx / resolution)) + origin
    gy = int(round(wy / resolution)) + origin
    return gx, gy


def _grid_to_world(gx: int, gy: int, resolution: float,
                   origin: int) -> Tuple[float, float]:
    wx = (gx - origin) * resolution
    wy = (gy - origin) * resolution
    return wx, wy


# ---------------------------------------------------------------------------
# Velocity smoother (jerk-limited)
# ---------------------------------------------------------------------------

class VelocitySmoother:
    """Jerk-limited smoother for passenger comfort.

    Each axis (vx, vy, omega) is independently smoothed so that:
      - acceleration never exceeds max_accel
      - jerk (change of acceleration per second) never exceeds max_jerk
    """

    def __init__(self):
        cfg = load_config().velocity
        self._max_accel = cfg.max_accel      # m/s^2
        self._max_jerk = cfg.max_jerk         # m/s^3
        self._max_vx = cfg.max_vx
        self._max_vy = cfg.max_vy
        self._max_omega = cfg.max_omega

        self._vel = np.zeros(3, dtype=np.float64)    # current output (vx, vy, w)
        self._accel = np.zeros(3, dtype=np.float64)   # current accel
        self._last_t: float = 0.0

    def smooth(self, target_vx: float, target_vy: float,
               target_omega: float) -> Tuple[float, float, float]:
        now = time.monotonic()
        dt = now - self._last_t if self._last_t > 0 else 0.0
        self._last_t = now
        if dt <= 0 or dt > 0.5:
            dt = 0.05  # sane default on first call or after a long gap

        target = np.array([target_vx, target_vy, target_omega])
        limits = np.array([self._max_vx, self._max_vy, self._max_omega])

        # Clamp target to velocity limits
        target = np.clip(target, -limits, limits)

        # Desired acceleration to reach target this tick
        desired_accel = (target - self._vel) / dt

        # Limit jerk: how much accel can change this tick
        max_delta_accel = self._max_jerk * dt
        delta_accel = desired_accel - self._accel
        delta_mag = np.abs(delta_accel)
        scale = np.where(delta_mag > max_delta_accel,
                         max_delta_accel / np.maximum(delta_mag, 1e-9),
                         1.0)
        self._accel = self._accel + delta_accel * scale

        # Limit acceleration magnitude
        self._accel = np.clip(self._accel, -self._max_accel, self._max_accel)

        # Integrate
        self._vel = self._vel + self._accel * dt

        # Clamp final velocity
        self._vel = np.clip(self._vel, -limits, limits)

        return float(self._vel[0]), float(self._vel[1]), float(self._vel[2])

    def reset(self):
        self._vel[:] = 0
        self._accel[:] = 0
        self._last_t = 0.0


# ---------------------------------------------------------------------------
# DWA + Potential Field local planner
# ---------------------------------------------------------------------------

class _LocalPlanner:
    """Dynamic Window Approach blended with a potential field."""

    def __init__(self):
        cfg = load_config()
        pc = cfg.planner
        vc = cfg.velocity

        self._dwa_horizon = pc.dwa_time_horizon_s
        self._num_samples = pc.dwa_num_samples
        self._pf_range = pc.potential_field_range_m
        self._dwa_w = pc.dwa_weight
        self._pf_w = pc.potential_field_weight
        self._max_vx = vc.max_vx
        self._max_vy = vc.max_vy
        self._max_omega = vc.max_omega
        self._max_accel = vc.max_accel

    # ---- potential field ---------------------------------------------------

    def _potential_field(self, pose: Tuple[float, float, float],
                         waypoint: Tuple[float, float],
                         obstacles: List) -> Tuple[float, float, float]:
        """Attractive + repulsive force in robot frame -> (vx, vy, omega)."""
        px, py, theta = pose
        wpx, wpy = waypoint

        # -- attractive force toward waypoint (world frame) --
        dx = wpx - px
        dy = wpy - py
        dist = math.hypot(dx, dy)
        if dist < 0.01:
            attr_fx, attr_fy = 0.0, 0.0
        else:
            strength = min(dist, 1.0)
            attr_fx = strength * dx / dist
            attr_fy = strength * dy / dist

        # -- repulsive force from obstacles (world frame) --
        rep_fx, rep_fy = 0.0, 0.0
        for obs in obstacles:
            if len(obs) < 2:
                continue
            ox, oy = float(obs[0]), float(obs[1])
            odx = px - ox
            ody = py - oy
            odist = math.hypot(odx, ody)
            if odist < 0.01:
                odist = 0.01
            if odist < self._pf_range:
                # Inverse-square repulsion, capped
                magnitude = (1.0 / odist - 1.0 / self._pf_range) / (odist * odist)
                magnitude = min(magnitude, 5.0)
                rep_fx += magnitude * odx / odist
                rep_fy += magnitude * ody / odist

        fx = attr_fx + rep_fx
        fy = attr_fy + rep_fy

        # -- world-frame force -> robot-frame velocity command --
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        vx = cos_t * fx + sin_t * fy
        vy = -sin_t * fx + cos_t * fy

        # -- proportional heading controller toward waypoint --
        angle_to_wp = math.atan2(dy, dx)
        angle_err = angle_to_wp - theta
        angle_err = (angle_err + math.pi) % (2 * math.pi) - math.pi
        omega = 2.0 * angle_err

        return vx, vy, omega

    # ---- DWA ---------------------------------------------------------------

    def _dwa(self, pose: Tuple[float, float, float],
             waypoint: Tuple[float, float],
             grid_binary: np.ndarray,
             resolution: float, origin: int) -> Tuple[float, float, float]:
        """Sample velocity space, simulate forward, score, pick best."""
        px, py, theta = pose
        wpx, wpy = waypoint
        dt = self._dwa_horizon
        n = self._num_samples

        rng = np.random.default_rng()
        vxs = rng.uniform(-self._max_vx, self._max_vx, n).astype(np.float64)
        vys = rng.uniform(-self._max_vy, self._max_vy, n).astype(np.float64)
        omegas = rng.uniform(-self._max_omega, self._max_omega, n).astype(np.float64)

        # Inject useful candidates:
        # [0] straight toward goal in robot frame
        angle_to_wp = math.atan2(wpy - py, wpx - px)
        rel_angle = angle_to_wp - theta
        goal_vx = self._max_vx * math.cos(rel_angle)
        goal_vy = self._max_vx * math.sin(rel_angle)
        vxs[0], vys[0], omegas[0] = goal_vx, goal_vy, 0.0
        # [1] stop
        vxs[1], vys[1], omegas[1] = 0.0, 0.0, 0.0
        # [2] pure forward
        vxs[2], vys[2], omegas[2] = self._max_vx * 0.5, 0.0, 0.0
        # [3] turn toward goal only
        angle_err = (rel_angle + math.pi) % (2 * math.pi) - math.pi
        vxs[3], vys[3], omegas[3] = 0.1, 0.0, np.clip(2.0 * angle_err,
                                                         -self._max_omega,
                                                         self._max_omega)

        # Simulate single dt step: robot-frame velocity -> world displacement
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        world_dx = cos_t * vxs - sin_t * vys
        world_dy = sin_t * vxs + cos_t * vys

        end_x = px + world_dx * dt
        end_y = py + world_dy * dt
        end_theta = theta + omegas * dt

        # Also check midpoint for collision
        mid_x = px + world_dx * dt * 0.5
        mid_y = py + world_dy * dt * 0.5

        # Convert to grid
        gx_end = np.rint(end_x / resolution).astype(int) + origin
        gy_end = np.rint(end_y / resolution).astype(int) + origin
        gx_mid = np.rint(mid_x / resolution).astype(int) + origin
        gy_mid = np.rint(mid_y / resolution).astype(int) + origin

        rows, cols = grid_binary.shape

        # Vectorized scoring
        scores = np.full(n, -1e9)

        for i in range(n):
            ex, ey = gx_end[i], gy_end[i]
            mx, my = gx_mid[i], gy_mid[i]

            # Bounds check
            if not (0 <= ex < cols and 0 <= ey < rows):
                continue
            if not (0 <= mx < cols and 0 <= my < rows):
                continue

            # Collision check (end + midpoint)
            if grid_binary[ey, ex] or grid_binary[my, mx]:
                continue

            # Distance to goal (lower is better)
            dist_goal = math.hypot(end_x[i] - wpx, end_y[i] - wpy)

            # Heading alignment with goal
            ang_to_goal = math.atan2(wpy - end_y[i], wpx - end_x[i])
            heading_err = abs((ang_to_goal - end_theta[i] + math.pi)
                              % (2 * math.pi) - math.pi)

            # Forward speed bonus (prefer moving forward)
            speed = math.hypot(vxs[i], vys[i])

            scores[i] = -1.5 * dist_goal - 0.5 * heading_err + 0.3 * speed

        best = int(np.argmax(scores))
        if scores[best] < -1e8:
            # Every sample collides — command stop
            return 0.0, 0.0, 0.0

        return float(vxs[best]), float(vys[best]), float(omegas[best])

    # ---- combined ----------------------------------------------------------

    def compute(self, pose: Tuple[float, float, float],
                waypoint: Tuple[float, float],
                grid_binary: np.ndarray,
                resolution: float, origin: int,
                obstacles: List) -> Tuple[float, float, float]:
        """Blend DWA and potential-field outputs."""
        dwa_vx, dwa_vy, dwa_w = self._dwa(
            pose, waypoint, grid_binary, resolution, origin)
        pf_vx, pf_vy, pf_w = self._potential_field(pose, waypoint, obstacles)

        wd = self._dwa_w
        wp = self._pf_w
        vx = wd * dwa_vx + wp * pf_vx
        vy = wd * dwa_vy + wp * pf_vy
        omega = wd * dwa_w + wp * pf_w

        return vx, vy, omega


# ---------------------------------------------------------------------------
# Main Planner class
# ---------------------------------------------------------------------------

class Planner:
    """Top-level planner: A* global path  +  DWA/PF local  +  velocity smoother.

    Usage::

        planner = Planner()
        planner.set_goal(3.0, 1.5)

        # in your 10-20 Hz control loop:
        vx, vy, omega = planner.update(pose, occupancy_grid, obstacles)
    """

    def __init__(self):
        cfg = load_config()
        self._pcfg = cfg.planner
        self._vcfg = cfg.velocity

        self._goal: Optional[Tuple[float, float]] = None
        self._global_path: Optional[List[Tuple[float, float]]] = None
        self._path_index: int = 0
        self._last_replan: float = 0.0
        self._reached: bool = False

        self._local_planner = _LocalPlanner()
        self._smoother = VelocitySmoother()

    # ---- public API --------------------------------------------------------

    def set_goal(self, x: float, y: float):
        """Set a new navigation goal in world coordinates (metres)."""
        self._goal = (x, y)
        self._global_path = None
        self._path_index = 0
        self._reached = False
        self._last_replan = 0.0
        self._smoother.reset()
        logger.info("Planner: new goal set (%.2f, %.2f)", x, y)

    def reached_goal(self) -> bool:
        """True once the robot is within goal_tolerance_m of the goal."""
        return self._reached

    def update(self, pose: Tuple[float, float, float],
               occupancy_grid,
               obstacles: List) -> Tuple[float, float, float]:
        """Run one planning cycle.  Returns smoothed (vx, vy, omega).

        Args:
            pose: (x, y, theta) in world metres / radians.
            occupancy_grid: OccupancyGrid instance from slam.py
                            (needs .get_binary(), .grid_size, .resolution).
            obstacles: list of (x, y) or (x, y, r) obstacle positions
                       in world frame (metres).

        Returns:
            (vx, vy, omega) — velocity command for the mecanum drive.
        """
        # No goal => coast to zero
        if self._goal is None:
            return self._smoother.smooth(0.0, 0.0, 0.0)

        px, py, theta = pose
        gx, gy = self._goal

        # -- check arrival --
        dist_to_goal = math.hypot(gx - px, gy - py)
        if dist_to_goal < self._pcfg.goal_tolerance_m:
            if not self._reached:
                logger.info("Planner: reached goal (%.2f, %.2f)", gx, gy)
            self._reached = True
            return self._smoother.smooth(0.0, 0.0, 0.0)

        # -- grid info --
        resolution = occupancy_grid.resolution
        grid_size = occupancy_grid.grid_size
        origin = grid_size // 2
        grid_binary = occupancy_grid.get_binary()

        # -- replan global path periodically --
        now = time.monotonic()
        need_replan = (self._global_path is None
                       or now - self._last_replan > self._pcfg.replan_interval_s)
        if need_replan:
            self._replan_global(px, py, gx, gy,
                                grid_binary, resolution, origin)
            self._last_replan = now

        # -- pick local waypoint to track --
        waypoint = self._get_lookahead_waypoint(px, py)

        # -- local planner: DWA + potential field --
        vx, vy, omega = self._local_planner.compute(
            pose, waypoint, grid_binary, resolution, origin, obstacles)

        # Clamp raw output to configured limits
        vx = np.clip(vx, -self._vcfg.max_vx, self._vcfg.max_vx)
        vy = np.clip(vy, -self._vcfg.max_vy, self._vcfg.max_vy)
        omega = np.clip(omega, -self._vcfg.max_omega, self._vcfg.max_omega)

        # -- velocity smoother for passenger comfort --
        return self._smoother.smooth(float(vx), float(vy), float(omega))

    def cancel(self):
        """Cancel the current navigation goal."""
        self._goal = None
        self._global_path = None
        self._path_index = 0
        self._reached = False
        self._smoother.reset()
        logger.info("Planner: navigation cancelled")

    @property
    def global_path(self) -> Optional[List[Tuple[float, float]]]:
        """Current A* global path in world coordinates (for visualization)."""
        return self._global_path

    # ---- internals ---------------------------------------------------------

    def _replan_global(self, px: float, py: float,
                       gx: float, gy: float,
                       grid_binary: np.ndarray,
                       resolution: float, origin: int):
        """Run A* and cache the resulting world-coordinate path."""
        start_cell = _world_to_grid(px, py, resolution, origin)
        goal_cell = _world_to_grid(gx, gy, resolution, origin)

        t0 = time.monotonic()
        cell_path = _astar(grid_binary, start_cell, goal_cell)
        elapsed = time.monotonic() - t0

        if cell_path is None:
            logger.warning("Planner: A* found no path (%.3f s)", elapsed)
            self._global_path = None
            self._path_index = 0
            return

        cell_path = _simplify_path(cell_path)
        self._global_path = [
            _grid_to_world(cx, cy, resolution, origin)
            for cx, cy in cell_path
        ]
        self._path_index = 0
        logger.debug("Planner: A* path %d waypoints (%.3f s)",
                      len(self._global_path), elapsed)

    def _get_lookahead_waypoint(self, px: float,
                                py: float) -> Tuple[float, float]:
        """Advance index past waypoints we've passed, return next target."""
        if not self._global_path:
            # No global path available — head directly toward goal
            return self._goal  # type: ignore[return-value]

        lookahead = 0.30  # metres — how far ahead on the path we aim

        # Skip waypoints that are behind us / too close
        while self._path_index < len(self._global_path) - 1:
            wpx, wpy = self._global_path[self._path_index]
            if math.hypot(wpx - px, wpy - py) < lookahead:
                self._path_index += 1
            else:
                break

        return self._global_path[self._path_index]
