"""
Camera perception: YOLOv8 object detection, ORB landmark matching, and
mono-depth obstacle estimation.

YOLOv8 detects 80 COCO object classes (person, chair, bottle, etc.).
ORB matches previously-seen landmarks for relocalization.
Mono heuristics estimate depth — objects lower in frame are closer.

Provides:
  - Perception: real OpenCV camera pipeline (YOLO + ORB)
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

try:
    from yolo_vision import YoloDetection, YoloV8OpenCvDetector
except ImportError:
    YoloDetection = None
    YoloV8OpenCvDetector = None

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
        self._yolo_detections: list = []

        self._running = False
        self._lock = threading.Lock()
        self._yolo: Optional[YoloV8OpenCvDetector] = None

    async def start(self) -> None:
        import cv2
        self._cap = cv2.VideoCapture(self._cam_idx)

        # Only set resolution if the camera doesn't already match.
        # NVIDIA Broadcast and other virtual cameras can produce black
        # letterbox bars when forced to a non-native resolution.
        native_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        native_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if native_w != self._width or native_h != self._height:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

        # Read actual resolution after any driver negotiation
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not self._cap.isOpened():
            logger.error("Failed to open camera %d", self._cam_idx)
            raise RuntimeError(f"Camera {self._cam_idx} not available")

        self._orb = cv2.ORB_create(nfeatures=self._orb_features)
        self._bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # YOLO object detector (80 COCO classes)
        if YoloV8OpenCvDetector is not None:
            try:
                self._yolo = YoloV8OpenCvDetector()
                logger.info("YOLOv8 detector loaded")
            except Exception as exc:
                logger.warning("YOLOv8 init failed, running without: %s", exc)

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

            # YOLO detection on colour frame
            yolo_dets = []
            if self._yolo is not None:
                try:
                    yolo_dets = self._yolo.detect(frame)
                except Exception as exc:
                    logger.debug("YOLO detect error: %s", exc)

            with self._lock:
                self._latest_frame = frame
                self._obstacles = obstacles
                self._landmarks = landmarks
                self._yolo_detections = yolo_dets

            await asyncio.sleep(1.0 / self._fps)

    # Safety margin (pixels) to trim inward from detected content edges.
    # Eliminates Canny artefacts at the black-bar / content boundary.
    _LETTERBOX_MARGIN = 20

    @staticmethod
    def _find_content_bounds(gray: np.ndarray) -> Tuple[int, int, int, int]:
        """Return (x_start, x_end, y_start, y_end) of actual camera content.

        NVIDIA Broadcast pads the frame with solid-black letterbox bars.
        We detect them by looking for contiguous runs of columns/rows from
        each edge where the *maximum* pixel value is very low (< 10).
        Using per-column max instead of mean avoids false positives when
        real content happens to be dark on average.

        After finding the content box we add ``_LETTERBOX_MARGIN`` pixels
        of inward padding so any residual brightness gradient at the
        boundary is excluded from downstream edge detection.

        When no bars are detected the full frame dimensions are returned
        (minus the safety margin).
        """
        h, w = gray.shape
        margin = Perception._LETTERBOX_MARGIN

        col_max = gray.max(axis=0)   # shape (w,)
        row_max = gray.max(axis=1)   # shape (h,)

        # Scan from the left/right to find first non-black column
        x_start = 0
        while x_start < w and col_max[x_start] < 10:
            x_start += 1
        x_end = w - 1
        while x_end > x_start and col_max[x_end] < 10:
            x_end -= 1

        # Same for top/bottom rows
        y_start = 0
        while y_start < h and row_max[y_start] < 10:
            y_start += 1
        y_end = h - 1
        while y_end > y_start and row_max[y_end] < 10:
            y_end -= 1

        # If bars were detected, add an extra inward margin to eliminate
        # the brightness ramp at the transition edge.
        if x_start > 0:
            x_start = min(x_start + margin, x_end)
        if x_end < w - 1:
            x_end = max(x_end - margin, x_start)
        if y_start > 0:
            y_start = min(y_start + margin, y_end)
        if y_end < h - 1:
            y_end = max(y_end - margin, y_start)

        return x_start, x_end, y_start, y_end

    def _detect_obstacles(self, gray: np.ndarray) -> List[Obstacle]:
        """Edge-based mono obstacle detection with depth-from-vertical heuristic.

        Objects lower in the frame are closer. We find strong edges in the
        bottom 2/3 of the image, cluster them, and estimate distance from
        their vertical position.

        Black letterbox bars (NVIDIA Broadcast) are removed before edge
        detection so their sharp brightness transitions are never reported
        as obstacles.
        """
        import cv2
        h, w = gray.shape

        # --- letterbox removal -----------------------------------------
        x_start, x_end, y_start, y_end = self._find_content_bounds(gray)

        content_w = x_end - x_start + 1
        content_h = y_end - y_start + 1
        if content_w < 50 or content_h < 50:
            return []  # frame is mostly black, skip

        # Crop to verified content region
        cropped = gray[y_start:y_end + 1, x_start:x_end + 1]

        # Zero-out a thin border inside the crop to kill any residual
        # brightness ramp that survived the margin.  GaussianBlur also
        # softens any remaining hard transitions so Canny ignores them.
        border = 8
        cropped[:border, :] = 0
        cropped[-border:, :] = 0
        cropped[:, :border] = 0
        cropped[:, -border:] = 0
        cropped = cv2.GaussianBlur(cropped, (5, 5), 0)

        # Canny edge detection on content region
        edges = cv2.Canny(cropped, 50, 150)

        # Ensure nothing leaks from the zeroed border
        edges[:border, :] = 0
        edges[-border:, :] = 0
        edges[:, :border] = 0
        edges[:, -border:] = 0

        # Only look at bottom 2/3 (foreground)
        ch, cw = cropped.shape
        roi = edges[ch // 3:, :]
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

            # Horizontal angle: center of cropped content = 0, edges = +/- 30 deg (approx FOV/2)
            angle = (cx / max(cw, 1) - 0.5) * math.radians(60)

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

    def get_yolo_detections(self) -> list:
        with self._lock:
            return list(self._yolo_detections)

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

    def get_yolo_detections(self) -> list:
        return []

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
