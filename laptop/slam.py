"""
SLAM: occupancy grid mapping + ICP scan matching for pose correction.

Two responsibilities:
  1. Maintain a Bayesian occupancy grid from LiDAR scans (log-odds raycasting)
  2. Match new scans against the grid to correct pose drift (ICP)

The occupancy grid is the static map layer fed into spatial_memory.py.
The scan-match pose correction feeds into pose_estimator.py.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import numpy as np

from config import load_config

logger = logging.getLogger(__name__)


class OccupancyGrid:
    """Log-odds Bayesian occupancy grid with LiDAR raycasting."""

    def __init__(self, size_m: float = 20.0, resolution_m: float = 0.05):
        self._res = resolution_m
        self._size_m = size_m
        self._grid_size = int(size_m / resolution_m)
        self._origin = self._grid_size // 2  # center of grid = world (0,0)

        # Log-odds grid — 0 = unknown, >0 = occupied, <0 = free
        self._log_odds = np.zeros(
            (self._grid_size, self._grid_size), dtype=np.float32
        )

        # Tuning
        self._l_occ = 0.85    # log-odds increment for occupied
        self._l_free = -0.40  # log-odds increment for free
        self._l_max = 5.0
        self._l_min = -5.0

    def update_from_scan(self, robot_x: float, robot_y: float,
                         robot_theta: float, scan_xy: np.ndarray) -> None:
        """Raytrace from robot pose through each scan point (vectorized).

        scan_xy: Nx2 array of (x, y) in robot-local frame.

        Uses interpolated line sampling along each ray for free-space
        clearing, and batch indexing for occupied endpoints. Produces the
        same Bayesian log-odds update as the original Bresenham version
        but runs ~50x faster by staying in numpy.
        """
        if scan_xy.shape[0] == 0:
            return

        cos_t = math.cos(robot_theta)
        sin_t = math.sin(robot_theta)

        # Robot grid coords
        rx = robot_x / self._res + self._origin
        ry = robot_y / self._res + self._origin

        # Transform all scan points: robot-local -> world -> grid  (vectorized)
        lx = scan_xy[:, 0]
        ly = scan_xy[:, 1]
        ex = (robot_x + cos_t * lx - sin_t * ly) / self._res + self._origin
        ey = (robot_y + sin_t * lx + cos_t * ly) / self._res + self._origin

        gs = self._grid_size

        # --- 1. Mark endpoints as occupied (batch) ---
        exi = np.rint(ex).astype(np.intp)
        eyi = np.rint(ey).astype(np.intp)
        valid_end = (exi >= 0) & (exi < gs) & (eyi >= 0) & (eyi < gs)
        ve_x = exi[valid_end]
        ve_y = eyi[valid_end]
        # Use np.add.at for correct duplicate-index handling
        np.add.at(self._log_odds, (ve_y, ve_x), self._l_occ)

        # --- 2. Free-space raycasting via interpolated sampling ---
        # For each ray from (rx, ry) to (ex[i], ey[i]), sample points
        # along the ray and mark those cells as free.
        dx = ex - rx
        dy = ey - ry
        ray_len_cells = np.sqrt(dx * dx + dy * dy)  # length in grid cells

        # Number of samples per ray — one per cell traversed (capped for sanity)
        n_samples_per_ray = np.clip(ray_len_cells.astype(np.intp), 1, 500)
        max_samples = int(n_samples_per_ray.max())
        n_rays = scan_xy.shape[0]

        # Build a (n_rays, max_samples) parameter matrix, t in [0, 1)
        # Each row has linspace(0, 1-eps, n_samples_per_ray[i]), padded with -1
        t = np.full((n_rays, max_samples), -1.0, dtype=np.float32)
        cols = np.arange(max_samples, dtype=np.float32)
        for ns in np.unique(n_samples_per_ray):
            mask = n_samples_per_ray == ns
            # Exclude the endpoint itself (t=1); we already marked it occupied
            t[mask, :ns] = cols[:ns] / ns

        # Interpolate grid coordinates for all sample points
        # shape: (n_rays, max_samples)
        sample_x = rx + t * dx[:, np.newaxis]
        sample_y = ry + t * dy[:, np.newaxis]

        # Flatten, filter valid (t >= 0 means it's a real sample)
        valid_mask = t >= 0
        sx_flat = np.rint(sample_x[valid_mask]).astype(np.intp)
        sy_flat = np.rint(sample_y[valid_mask]).astype(np.intp)

        # Bounds check
        in_bounds = (sx_flat >= 0) & (sx_flat < gs) & (sy_flat >= 0) & (sy_flat < gs)
        sx_flat = sx_flat[in_bounds]
        sy_flat = sy_flat[in_bounds]

        # Apply free-space update
        np.add.at(self._log_odds, (sy_flat, sx_flat), self._l_free)

        # --- 3. Clamp log-odds to [l_min, l_max] ---
        np.clip(self._log_odds, self._l_min, self._l_max, out=self._log_odds)

    def get_occupancy(self) -> np.ndarray:
        """Return probability grid [0, 1]. 0.5 = unknown."""
        return 1.0 / (1.0 + np.exp(-self._log_odds))

    def get_binary(self, threshold: float = 0.65) -> np.ndarray:
        """Return boolean grid: True = occupied."""
        return self.get_occupancy() > threshold

    @property
    def grid_size(self) -> int:
        return self._grid_size

    @property
    def resolution(self) -> float:
        return self._res


class ScanMatcher:
    """Simple point-to-grid ICP for correcting pose against the occupancy grid.

    Aligns a new LiDAR scan to the existing occupancy grid to produce
    a (dx, dy, dtheta) correction. This is the loop closure mechanism.
    """

    def __init__(self, grid: OccupancyGrid):
        self._grid = grid
        cfg = load_config().slam
        self._max_iter = cfg.icp_max_iterations
        self._tolerance = cfg.icp_tolerance

    def match(self, scan_xy: np.ndarray, pose: Tuple[float, float, float]
              ) -> Optional[Tuple[float, float, float]]:
        """Run ICP to find the pose that best aligns scan_xy to the grid.

        Args:
            scan_xy: Nx2 robot-local scan points
            pose: initial guess (x, y, theta)

        Returns:
            Corrected (x, y, theta) or None if match failed.
        """
        if scan_xy.shape[0] < 10:
            return None

        occ = self._grid.get_occupancy()
        res = self._grid.resolution
        origin = self._grid.grid_size // 2
        gs = self._grid.grid_size

        x, y, theta = pose

        for iteration in range(self._max_iter):
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)

            # Transform scan to world frame with current pose estimate
            wx = x + cos_t * scan_xy[:, 0] - sin_t * scan_xy[:, 1]
            wy = y + sin_t * scan_xy[:, 0] + cos_t * scan_xy[:, 1]

            # Convert to grid coords
            gx = (wx / res + origin).astype(int)
            gy = (wy / res + origin).astype(int)

            # Filter valid
            valid = (gx >= 1) & (gx < gs - 1) & (gy >= 1) & (gy < gs - 1)
            if valid.sum() < 10:
                return None

            gx_v = gx[valid]
            gy_v = gy[valid]
            wx_v = wx[valid]
            wy_v = wy[valid]

            # Compute gradient of occupancy at each point
            grad_x = (occ[gy_v, gx_v + 1] - occ[gy_v, gx_v - 1]) / (2.0 * res)
            grad_y = (occ[gy_v + 1, gx_v] - occ[gy_v - 1, gx_v]) / (2.0 * res)
            values = occ[gy_v, gx_v]

            # We want scan points to land on high-occupancy cells.
            # Residual = (1 - occupancy) for occupied-seeking alignment.
            residual = 1.0 - values

            # Build Jacobian: how does (dx, dy, dtheta) change each point's grid value?
            # d(wx)/d(x) = 1, d(wx)/d(theta) = -sin*lx - cos*ly
            lx = scan_xy[:, 0][valid]
            ly = scan_xy[:, 1][valid]

            J = np.zeros((valid.sum(), 3))
            J[:, 0] = grad_x          # d(occ)/dx
            J[:, 1] = grad_y          # d(occ)/dy
            J[:, 2] = grad_x * (-sin_t * lx - cos_t * ly) + \
                       grad_y * ( cos_t * lx - sin_t * ly)  # d(occ)/dtheta

            # Gauss-Newton step
            JtJ = J.T @ J
            Jtr = J.T @ residual

            try:
                delta = np.linalg.solve(JtJ + np.eye(3) * 1e-6, Jtr)
            except np.linalg.LinAlgError:
                return None

            x += delta[0]
            y += delta[1]
            theta += delta[2]

            if np.linalg.norm(delta) < self._tolerance:
                break

        return (x, y, theta)


class SLAM:
    """Top-level SLAM: ties occupancy grid + scan matcher together."""

    def __init__(self):
        cfg = load_config().slam
        self._grid = OccupancyGrid(cfg.map_size_m, cfg.resolution_m)
        self._matcher = ScanMatcher(self._grid)
        self._scan_count = 0

    def process_scan(self, scan_xy: np.ndarray,
                     pose: Tuple[float, float, float]
                     ) -> Optional[Tuple[float, float, float]]:
        """Integrate a scan: update grid + optionally correct pose.

        Args:
            scan_xy: Nx2 robot-local scan points
            pose: current (x, y, theta) estimate

        Returns:
            Corrected pose if scan-match succeeded, else None.
        """
        # Update occupancy grid
        self._grid.update_from_scan(pose[0], pose[1], pose[2], scan_xy)
        self._scan_count += 1

        # Attempt scan-match correction every 5th scan (save CPU)
        corrected = None
        if self._scan_count > 10 and self._scan_count % 5 == 0:
            corrected = self._matcher.match(scan_xy, pose)
            if corrected is not None:
                dx = corrected[0] - pose[0]
                dy = corrected[1] - pose[1]
                dt = corrected[2] - pose[2]
                if abs(dx) > 1.0 or abs(dy) > 1.0 or abs(dt) > 0.5:
                    logger.warning("Scan match diverged (%.2f, %.2f, %.2f) — ignoring",
                                   dx, dy, dt)
                    corrected = None

        return corrected

    @property
    def occupancy_grid(self) -> OccupancyGrid:
        return self._grid

    def get_occupancy_array(self) -> np.ndarray:
        return self._grid.get_occupancy()

    def save(self, path: str) -> None:
        np.savez_compressed(path, log_odds=self._grid._log_odds)
        logger.info("SLAM map saved to %s", path)

    def load(self, path: str) -> None:
        data = np.load(path)
        self._grid._log_odds = data["log_odds"]
        logger.info("SLAM map loaded from %s", path)
