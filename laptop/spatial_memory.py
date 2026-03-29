"""
4-layer persistent spatial memory for the autonomous wheelchair.

Layer 1: Static map — Bayesian-merged occupancy grid across trips.
Layer 2: Landmark database — ORB descriptors + world coordinates.
Layer 3: Dynamic obstacle priors — P(blocked) per cell, EMA across trips.
Layer 4: Learned cost map — traversal success/failure history.

All layers share the same grid dimensions and resolution as the SLAM
occupancy grid (default 400x400 at 5 cm).  Persisted as .npz files
so everything survives between runs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import SpatialMemoryConfig, load_config

logger = logging.getLogger(__name__)


class SpatialMemory:
    """Persistent spatial memory with 4 layers + cost compositor."""

    def __init__(self, grid_size: int, resolution: float,
                 cfg: Optional[SpatialMemoryConfig] = None):
        self._grid_size = grid_size
        self._resolution = resolution
        self._cfg = cfg or load_config().spatial_memory
        self._origin = grid_size // 2  # grid center = world (0, 0)

        # Layer 1: Static map — accumulated log-odds from multiple trips.
        # Bayesian merge: just add log-odds from each trip's SLAM grid.
        self._static_log_odds = np.zeros((grid_size, grid_size), dtype=np.float32)
        self._static_update_count = 0

        # Layer 2: Landmark database — {name: (x, y, descriptor)}
        self._landmarks: Dict[str, Tuple[float, float, np.ndarray]] = {}

        # Layer 3: Dynamic obstacle prior — P(blocked) per cell [0, 1].
        # Updated via exponential moving average each trip.
        self._dynamic_prior = np.full((grid_size, grid_size), 0.5, dtype=np.float32)

        # Layer 4: Learned cost — starts at 1.0 (neutral).
        # Successful traversals lower it, failures raise it.
        self._learned_cost = np.ones((grid_size, grid_size), dtype=np.float32)

        # Paths
        self._map_dir = Path(self._cfg.map_dir)
        self._landmark_dir = Path(self._cfg.landmark_dir)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int(round(x / self._resolution)) + self._origin
        gy = int(round(y / self._resolution)) + self._origin
        return gx, gy

    def _in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self._grid_size and 0 <= gy < self._grid_size

    # ------------------------------------------------------------------
    # Layer 1: Static map
    # ------------------------------------------------------------------

    def update_static_map(self, occupancy_log_odds: np.ndarray) -> None:
        """Merge a SLAM occupancy grid (log-odds) into the persistent static map.

        Called once at end-of-trip.  The Bayesian merge is simply additive
        in log-odds space, clamped to keep values sane.
        """
        if occupancy_log_odds.shape != (self._grid_size, self._grid_size):
            logger.warning("Static map shape mismatch: expected %s, got %s",
                           (self._grid_size, self._grid_size),
                           occupancy_log_odds.shape)
            return
        self._static_log_odds += occupancy_log_odds
        np.clip(self._static_log_odds, -20.0, 20.0, out=self._static_log_odds)
        self._static_update_count += 1
        logger.info("Static map updated (trip %d)", self._static_update_count)

    def get_static_occupancy(self) -> np.ndarray:
        """Return probability grid [0, 1] from merged log-odds."""
        return 1.0 / (1.0 + np.exp(-self._static_log_odds))

    # ------------------------------------------------------------------
    # Layer 2: Landmark database
    # ------------------------------------------------------------------

    def update_landmark(self, name: str, x: float, y: float,
                        descriptor: np.ndarray) -> None:
        """Add or update a named landmark with its world coordinates and ORB descriptor."""
        self._landmarks[name] = (x, y, descriptor)
        logger.info("Landmark '%s' stored at (%.2f, %.2f)", name, x, y)

    def lookup_landmark(self, name: str) -> Optional[Tuple[float, float]]:
        """Return (x, y) for a named landmark, or None if unknown."""
        entry = self._landmarks.get(name)
        if entry is not None:
            return (entry[0], entry[1])
        # Case-insensitive fuzzy fallback: check if name is a substring
        name_lower = name.lower()
        for key, (lx, ly, _desc) in self._landmarks.items():
            if name_lower in key.lower() or key.lower() in name_lower:
                return (lx, ly)
        return None

    def get_all_landmark_names(self) -> List[str]:
        """Return all known landmark names (for LLM prompt grounding)."""
        return list(self._landmarks.keys())

    def get_landmark_descriptor(self, name: str) -> Optional[np.ndarray]:
        """Return the ORB descriptor for a landmark, if it exists."""
        entry = self._landmarks.get(name)
        return entry[2] if entry is not None else None

    # ------------------------------------------------------------------
    # Layer 3: Dynamic obstacle priors
    # ------------------------------------------------------------------

    def update_dynamic_prior(self, occupancy: np.ndarray) -> None:
        """Update P(blocked) per cell using exponential moving average.

        occupancy: probability grid [0, 1] from the current trip's SLAM.
        Called once at end-of-trip.
        """
        if occupancy.shape != (self._grid_size, self._grid_size):
            logger.warning("Dynamic prior shape mismatch: expected %s, got %s",
                           (self._grid_size, self._grid_size), occupancy.shape)
            return
        alpha = self._cfg.dynamic_prior_alpha
        self._dynamic_prior = (1.0 - alpha) * self._dynamic_prior + alpha * occupancy
        logger.info("Dynamic obstacle priors updated (alpha=%.2f)", alpha)

    # ------------------------------------------------------------------
    # Layer 4: Learned cost map
    # ------------------------------------------------------------------

    def record_traversal(self, x: float, y: float, success: bool) -> None:
        """Record a traversal at world (x, y). Adjust cost up or down."""
        gx, gy = self._world_to_grid(x, y)
        if not self._in_bounds(gx, gy):
            return
        if success:
            self._learned_cost[gy, gx] *= self._cfg.cost_success_factor
        else:
            self._learned_cost[gy, gx] *= self._cfg.cost_failure_factor
        # Clamp
        self._learned_cost[gy, gx] = np.clip(
            self._learned_cost[gy, gx], self._cfg.cost_min, self._cfg.cost_max
        )

    def record_path(self, path_xy: np.ndarray, success: bool) -> None:
        """Record traversal result for an entire path (Nx2 world coords)."""
        for x, y in path_xy:
            self.record_traversal(float(x), float(y), success)

    # ------------------------------------------------------------------
    # Cost compositor
    # ------------------------------------------------------------------

    def get_costmap(self) -> np.ndarray:
        """Blend all 4 layers into a single navigation costmap.

        Returns a float32 grid where higher values = more expensive to traverse.
        Free space ~0.1, walls ~10+, unknown ~1.0.
        """
        w = self._cfg.costmap_weights

        # Static: convert log-odds to occupancy probability, scale to cost
        static_occ = self.get_static_occupancy()  # [0, 1]
        static_cost = static_occ * 10.0  # walls -> 10, free -> 0

        # Dynamic: same idea
        dynamic_cost = self._dynamic_prior * 10.0

        # Learned: already a cost multiplier
        learned_cost = self._learned_cost

        # Weighted blend
        costmap = (
            w["static"] * static_cost
            + w["dynamic"] * dynamic_cost
            + w["learned"] * learned_cost
        )

        # Normalize so the weights don't inflate absolute values
        total_weight = w["static"] + w["dynamic"] + w["learned"]
        if total_weight > 0:
            costmap /= total_weight

        # Clamp final values
        np.clip(costmap, self._cfg.cost_min, self._cfg.cost_max, out=costmap)
        return costmap

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Save all layers to disk as .npz and .json files."""
        self._map_dir.mkdir(parents=True, exist_ok=True)
        self._landmark_dir.mkdir(parents=True, exist_ok=True)

        # Grids
        grid_path = self._map_dir / "spatial_memory.npz"
        np.savez_compressed(
            str(grid_path),
            static_log_odds=self._static_log_odds,
            dynamic_prior=self._dynamic_prior,
            learned_cost=self._learned_cost,
            meta=np.array([self._grid_size, self._resolution,
                           self._static_update_count]),
        )
        logger.info("Spatial memory grids saved to %s", grid_path)

        # Landmarks: save coords as JSON, descriptors as npz
        landmark_meta = {}
        landmark_descs = {}
        for name, (lx, ly, desc) in self._landmarks.items():
            landmark_meta[name] = {"x": lx, "y": ly}
            landmark_descs[name] = desc

        meta_path = self._landmark_dir / "landmarks.json"
        with open(str(meta_path), "w") as f:
            json.dump(landmark_meta, f, indent=2)

        if landmark_descs:
            desc_path = self._landmark_dir / "landmark_descriptors.npz"
            np.savez_compressed(str(desc_path), **landmark_descs)

        logger.info("Landmarks saved (%d entries)", len(self._landmarks))

    def load(self) -> None:
        """Load all layers from disk. Missing files are silently skipped."""
        # Grids
        grid_path = self._map_dir / "spatial_memory.npz"
        if grid_path.exists():
            data = np.load(str(grid_path))
            meta = data["meta"]
            saved_size = int(meta[0])
            if saved_size != self._grid_size:
                logger.warning("Grid size mismatch on load: saved %d vs current %d — skipping",
                               saved_size, self._grid_size)
            else:
                self._static_log_odds = data["static_log_odds"]
                self._dynamic_prior = data["dynamic_prior"]
                self._learned_cost = data["learned_cost"]
                self._static_update_count = int(meta[2])
                logger.info("Spatial memory grids loaded from %s (trip %d)",
                            grid_path, self._static_update_count)
        else:
            logger.info("No saved spatial memory found at %s — starting fresh", grid_path)

        # Landmarks
        meta_path = self._landmark_dir / "landmarks.json"
        desc_path = self._landmark_dir / "landmark_descriptors.npz"
        if meta_path.exists():
            with open(str(meta_path)) as f:
                landmark_meta = json.load(f)

            descs = {}
            if desc_path.exists():
                npz = np.load(str(desc_path))
                descs = {k: npz[k] for k in npz.files}

            self._landmarks = {}
            for name, coords in landmark_meta.items():
                desc = descs.get(name, np.zeros(32, dtype=np.uint8))
                self._landmarks[name] = (coords["x"], coords["y"], desc)

            logger.info("Landmarks loaded (%d entries)", len(self._landmarks))

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def end_of_trip(self, slam_log_odds: np.ndarray,
                    slam_occupancy: np.ndarray) -> None:
        """One-call end-of-trip update: merge SLAM data into persistent memory.

        Args:
            slam_log_odds: raw log-odds grid from SLAM (for static map merge).
            slam_occupancy: probability [0,1] grid (for dynamic prior update).
        """
        self.update_static_map(slam_log_odds)
        self.update_dynamic_prior(slam_occupancy)
        self.save()
        logger.info("End-of-trip spatial memory update complete")

    @property
    def grid_size(self) -> int:
        return self._grid_size

    @property
    def resolution(self) -> float:
        return self._resolution

    @property
    def landmark_count(self) -> int:
        return len(self._landmarks)
