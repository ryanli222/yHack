"""
Camera perception: ORB feature extraction, landmark detection, and
mono-depth obstacle estimation.

Stereo calibration is cooked so we use mono heuristics: objects lower in
the frame are closer. Combined with LiDAR this is good enough for the demo.

Provides:
  - Perception: real OpenCV camera pipeline
  - FakePerception: synthetic obstacles for testing
  - create_perception(): factory
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from config import CameraConfig, load_config

logger = logging.getLogger(__name__)


# ======================================================================
# Data types
# ======================================================================

@dataclass
class Obstacle:
    x: float          # robot-local, meters forward
    y: float          # robot-local, meters left
    radius: float     # estimated size
    confidence: float  # 0-1


@dataclass
class Landmark:
    name: str
    x: float               # robot-local at time of detection
    y: float
    descriptor: np.ndarray  # ORB descriptor (32 bytes)
    confidence: float


# ======================================================================
# Real perception
# ======================================================================

class Perception:
    """Mono camera perception: ORB landmarks + edge-based obstacle detection."""

    def __init__(self, camera_index: Optional[int] = None):
        cfg = load_config().camera
        self._cam_idx = camera_index if camera_index is not None else cfg.left_index
        self._width = cfg.width
        self._height = cfg.height
        self._fps = cfg.fps
        self._orb_features = cfg.orb_features
        self._match_threshold = cfg.landmark_match_threshold
        self._cluster_radius = cfg.obstacle_cluster_radius_m

        self._cap = None
        self._orb = None
        self._bf_matcher = None

        self._latest_frame: Optional[np.ndarray] = None
        self._obstacles: List[Obstacle] = []
        self._landmarks: List[Landmark] = []
        self._known_landmarks: List[Tuple[str, np.ndarray]] = []  # (name, descriptor)

        self._running = False
        self._lock = threading.Lock()

    async def start(self) -> None:
        import cv2
        self._cap = cv2.VideoCapture(self._cam_idx)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

        if not self._cap.isOpened():
            logger.error("Failed to open camera %d", self._cam_idx)
            raise RuntimeError(f"Camera {self._cam_idx} not available")

        self._orb = cv2.ORB_create(nfeatures=self._orb_features)
        self._bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._running = True
        logger.info("Perception started (camera %d, %dx%d @ %d fps)",
                     self._cam_idx, self._width, self._height, self._fps)

    async def run(self) -> None:
        """Main loop: grab frames, detect obstacles and landmarks."""
        import cv2
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                await asyncio.sleep(0.05)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            obstacles = self._detect_obstacles(gray)
            landmarks = self._detect_landmarks(gray)

            with self._lock:
                self._latest_frame = frame
                self._obstacles = obstacles
                self._landmarks = landmarks

            await asyncio.sleep(1.0 / self._fps)

    def _detect_obstacles(self, gray: np.ndarray) -> List[Obstacle]:
        """Edge-based mono obstacle detection with depth-from-vertical heuristic.

        Objects lower in the frame are closer. We find strong edges in the
        bottom 2/3 of the image, cluster them, and estimate distance from
        their vertical position.
        """
        import cv2
        h, w = gray.shape

        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Only look at bottom 2/3 (foreground)
        roi = edges[h // 3:, :]
        roi_h = roi.shape[0]

        # Find contours
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        obstacles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # filter noise
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Depth heuristic: bottom of ROI = 0.5m, top of ROI = 3.0m
            frac = 1.0 - (cy / max(roi_h, 1))
            depth = 0.5 + frac * 2.5

            # Horizontal angle: center of image = 0, edges = +/- 30 deg (approx FOV/2)
            angle = (cx / w - 0.5) * math.radians(60)

            obs_x = depth * math.cos(angle)
            obs_y = depth * math.sin(angle)
            confidence = min(area / 5000.0, 1.0)

            obstacles.append(Obstacle(x=obs_x, y=obs_y, radius=0.2, confidence=confidence))

        # Cluster nearby obstacles
        return self._cluster_obstacles(obstacles)

    def _cluster_obstacles(self, obstacles: List[Obstacle]) -> List[Obstacle]:
        """Merge obstacles within cluster_radius of each other."""
        if not obstacles:
            return []

        merged = []
        used = [False] * len(obstacles)

        for i, obs in enumerate(obstacles):
            if used[i]:
                continue
            cluster = [obs]
            used[i] = True
            for j in range(i + 1, len(obstacles)):
                if used[j]:
                    continue
                dist = math.hypot(obs.x - obstacles[j].x, obs.y - obstacles[j].y)
                if dist < self._cluster_radius:
                    cluster.append(obstacles[j])
                    used[j] = True

            # Average the cluster
            avg_x = sum(o.x for o in cluster) / len(cluster)
            avg_y = sum(o.y for o in cluster) / len(cluster)
            max_conf = max(o.confidence for o in cluster)
            merged.append(Obstacle(x=avg_x, y=avg_y, radius=0.3, confidence=max_conf))

        return merged

    def _detect_landmarks(self, gray: np.ndarray) -> List[Landmark]:
        """ORB feature matching against known landmark descriptors."""
        keypoints, descriptors = self._orb.detectAndCompute(gray, None)
        if descriptors is None or len(self._known_landmarks) == 0:
            return []

        found = []
        for name, known_desc in self._known_landmarks:
            matches = self._bf_matcher.match(descriptors, known_desc.reshape(1, -1)
                                             if known_desc.ndim == 1 else known_desc)
            good = [m for m in matches if m.distance < self._match_threshold]
            if len(good) >= 3:
                # Estimate position from matched keypoint centroid
                pts = [keypoints[m.queryIdx].pt for m in good]
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                h = gray.shape[0]
                w = gray.shape[1]

                frac = 1.0 - (cy / max(h, 1))
                depth = 0.5 + frac * 3.0
                angle = (cx / w - 0.5) * math.radians(60)

                found.append(Landmark(
                    name=name,
                    x=depth * math.cos(angle),
                    y=depth * math.sin(angle),
                    descriptor=known_desc,
                    confidence=len(good) / max(len(matches), 1),
                ))

        return found

    def register_landmark(self, name: str, descriptor: np.ndarray) -> None:
        """Add a known landmark descriptor for future matching."""
        self._known_landmarks.append((name, descriptor))
        logger.info("Registered landmark '%s' (%d descriptors)", name, descriptor.shape[0])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_obstacles(self) -> List[Obstacle]:
        with self._lock:
            return list(self._obstacles)

    def get_landmarks(self) -> List[Landmark]:
        with self._lock:
            return list(self._landmarks)

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest_frame

    def capture_landmark_descriptor(self, name: str) -> Optional[np.ndarray]:
        """Capture current frame's ORB descriptors and register as a landmark."""
        import cv2
        frame = self.get_frame()
        if frame is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, descriptors = self._orb.detectAndCompute(gray, None)
        if descriptors is not None and len(descriptors) > 0:
            self.register_landmark(name, descriptors)
            return descriptors
        return None

    async def stop(self) -> None:
        self._running = False
        if self._cap is not None:
            self._cap.release()
        logger.info("Perception stopped")


# ======================================================================
# Fake perception
# ======================================================================

class FakePerception:
    """Synthetic obstacles that drift around for testing."""

    def __init__(self):
        self._running = False
        self._obstacles: List[Obstacle] = []
        self._t = 0.0

    async def start(self) -> None:
        self._running = True
        logger.info("[FAKE] Perception started")

    async def run(self) -> None:
        import random
        while self._running:
            self._t += 0.066
            self._obstacles = [
                Obstacle(
                    x=2.0 + 0.5 * math.sin(self._t),
                    y=0.5 * math.cos(self._t * 0.7),
                    radius=0.2,
                    confidence=0.9 if random.random() > 0.1 else 0.0,
                ),
            ]
            await asyncio.sleep(0.066)  # ~15 fps

    def get_obstacles(self) -> List[Obstacle]:
        return list(self._obstacles)

    def get_landmarks(self) -> List[Landmark]:
        return []

    def get_frame(self) -> Optional[np.ndarray]:
        return None

    def register_landmark(self, name: str, descriptor: np.ndarray) -> None:
        pass

    def capture_landmark_descriptor(self, name: str) -> Optional[np.ndarray]:
        return None

    async def stop(self) -> None:
        self._running = False


# ======================================================================
# Factory
# ======================================================================

def create_perception(fake: bool = False, camera_index: Optional[int] = None):
    if fake:
        return FakePerception()
    try:
        import cv2  # noqa: F401
    except ImportError:
        logger.warning("OpenCV not installed — using FakePerception")
        return FakePerception()
    return Perception(camera_index=camera_index)
