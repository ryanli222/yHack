"""
VIAM-backed vision wrapper for pickup detections.

This module keeps the pickup detector separate from the navigation perception
stack. It can talk to a real VIAM vision service or a local fake stub used by
tests and dry runs.
"""

from __future__ import annotations

import io
import logging
from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

import numpy as np

from config import load_config

logger = logging.getLogger(__name__)


class ViamVisionError(RuntimeError):
    """Raised when the VIAM vision wrapper cannot complete a request."""


@dataclass
class BallDetection:
    label: str = ""
    confidence: float = 0.0
    x_min: float = 0.0
    y_min: float = 0.0
    x_max: float = 0.0
    y_max: float = 0.0
    image_width: int = 0
    image_height: int = 0
    source: str = "viam"
    service_name: str = ""
    camera_name: str = ""

    @property
    def width_px(self) -> float:
        return max(0.0, float(self.x_max) - float(self.x_min))

    @property
    def height_px(self) -> float:
        return max(0.0, float(self.y_max) - float(self.y_min))

    @property
    def area_px(self) -> float:
        return self.width_px * self.height_px

    @property
    def center_x(self) -> float:
        return float(self.x_min) + self.width_px / 2.0

    @property
    def center_y(self) -> float:
        return float(self.y_min) + self.height_px / 2.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "width_px": self.width_px,
                "height_px": self.height_px,
                "area_px": self.area_px,
                "center_x": self.center_x,
                "center_y": self.center_y,
            }
        )
        return payload


@dataclass
class VisionObservation:
    detection: Optional[BallDetection]
    detections: list[BallDetection]
    frame_bgr: Optional[np.ndarray] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection": None if self.detection is None else self.detection.to_dict(),
            "detections": [detection.to_dict() for detection in self.detections],
            "has_frame": self.frame_bgr is not None,
        }


def select_detection(
    detections: Sequence[BallDetection],
    *,
    target_label: str = "",
    minimum_confidence: float = 0.0,
) -> Optional[BallDetection]:
    """Pick the best matching detection for the current pickup target."""

    normalized_target = target_label.strip().lower()
    candidates = []
    for detection in detections:
        if detection.confidence < float(minimum_confidence):
            continue
        if normalized_target:
            label = detection.label.strip().lower()
            if normalized_target not in label and label not in normalized_target:
                continue
        candidates.append(detection)

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda detection: (float(detection.confidence), float(detection.area_px)),
    )


class ViamVision:
    """Wrapper around a VIAM vision service configured on the same machine as the arm."""

    def __init__(self):
        cfg = load_config()
        self._arm_cfg = cfg.arm
        self._pickup_cfg = cfg.pickup
        self._address = self._arm_cfg.address
        self._api_key = self._arm_cfg.api_key
        self._api_key_id = self._arm_cfg.api_key_id
        self._camera_name = self._pickup_cfg.camera_name
        self._vision_service_name = self._pickup_cfg.vision_service_name
        self._target_label = self._pickup_cfg.target_label
        self._minimum_confidence = self._pickup_cfg.minimum_confidence

        self._robot = None
        self._vision = None

    async def start(self) -> None:
        await self.connect()

    async def connect(self) -> None:
        if self._vision is not None and self._robot is not None:
            return
        self._validate_config()

        try:
            from viam.robot.client import RobotClient
            from viam.services.vision import VisionClient
        except ImportError as exc:
            raise ViamVisionError(
                "viam-sdk is not installed; install it before using VIAM pickup vision"
            ) from exc

        robot = None
        try:
            opts = RobotClient.Options.with_api_key(
                api_key=self._api_key,
                api_key_id=self._api_key_id,
            )
            robot = await RobotClient.at_address(self._address, opts)
            vision = VisionClient.from_robot(robot=robot, name=self._vision_service_name)
            self._robot = robot
            self._vision = vision
            logger.info(
                "Connected to VIAM vision service '%s' on camera '%s'",
                self._vision_service_name,
                self._camera_name,
            )
        except Exception as exc:
            if robot is not None:
                await robot.close()
            raise ViamVisionError(f"Failed to connect to VIAM vision service: {exc}") from exc

    async def stop(self) -> None:
        if self._robot is not None:
            try:
                await self._robot.close()
            except Exception:
                logger.debug("Ignoring VIAM robot close error", exc_info=True)
        self._vision = None
        self._robot = None

    async def get_ball_observation(self, *, include_image: bool = False) -> VisionObservation:
        await self.connect()

        frame_bgr = None
        raw_detections: Sequence[Any] = []
        try:
            if include_image and hasattr(self._vision, "capture_all_from_camera"):
                result = await self._vision.capture_all_from_camera(
                    self._camera_name,
                    return_image=True,
                    return_detections=True,
                )
                frame_bgr = self._image_to_bgr(getattr(result, "image", None))
                raw_detections = list(getattr(result, "detections", None) or [])
            else:
                raw_detections = list(
                    await self._vision.get_detections_from_camera(self._camera_name)
                )
        except Exception as exc:
            raise ViamVisionError(f"Failed to query detections from VIAM: {exc}") from exc

        image_height = int(frame_bgr.shape[0]) if frame_bgr is not None else 0
        image_width = int(frame_bgr.shape[1]) if frame_bgr is not None else 0
        detections = [
            self._normalize_detection(detection, image_width=image_width, image_height=image_height)
            for detection in raw_detections
        ]
        selected = select_detection(
            detections,
            target_label=self._target_label,
            minimum_confidence=self._minimum_confidence,
        )
        return VisionObservation(detection=selected, detections=detections, frame_bgr=frame_bgr)

    def _validate_config(self) -> None:
        if not self._address:
            raise ViamVisionError("VIAM machine address is not configured")
        if not self._api_key or not self._api_key_id:
            raise ViamVisionError("VIAM API credentials are not configured")
        if not self._vision_service_name:
            raise ViamVisionError("VIAM vision service name is not configured")
        if not self._camera_name:
            raise ViamVisionError("VIAM camera name is not configured")

    def _normalize_detection(
        self,
        detection: Any,
        *,
        image_width: int,
        image_height: int,
    ) -> BallDetection:
        return BallDetection(
            label=str(getattr(detection, "class_name", "") or getattr(detection, "label", "")),
            confidence=float(getattr(detection, "confidence", 0.0)),
            x_min=float(getattr(detection, "x_min", 0.0)),
            y_min=float(getattr(detection, "y_min", 0.0)),
            x_max=float(getattr(detection, "x_max", 0.0)),
            y_max=float(getattr(detection, "y_max", 0.0)),
            image_width=int(image_width),
            image_height=int(image_height),
            source="viam",
            service_name=self._vision_service_name,
            camera_name=self._camera_name,
        )

    def _image_to_bgr(self, image: Any) -> Optional[np.ndarray]:
        if image is None:
            return None
        if isinstance(image, np.ndarray):
            array = image.astype(np.uint8, copy=False)
            return self._ensure_bgr(array)

        try:
            from viam.media.utils.pil import viam_to_pil_image

            pil_image = viam_to_pil_image(image)
            array = np.asarray(pil_image)
            return self._ensure_bgr(array)
        except Exception:
            logger.debug("Could not use viam_to_pil_image for capture", exc_info=True)

        data = getattr(image, "data", None)
        if data:
            try:
                from PIL import Image

                with Image.open(io.BytesIO(data)) as pil_image:
                    array = np.asarray(pil_image)
                return self._ensure_bgr(array)
            except Exception:
                logger.debug("Could not decode VIAM image payload", exc_info=True)

        return None

    def _ensure_bgr(self, array: np.ndarray) -> np.ndarray:
        if array.ndim == 2:
            return np.repeat(array[:, :, None], 3, axis=2)
        if array.ndim == 3 and array.shape[2] == 4:
            array = array[:, :, :3]
        if array.ndim == 3 and array.shape[2] == 3:
            return array[:, :, ::-1].copy()
        return np.asarray(array, dtype=np.uint8)


class FakeViamVision:
    """Fake detector used by tests and dry runs without VIAM hardware."""

    def __init__(
        self,
        detections: Optional[Sequence[BallDetection]] = None,
        frame_bgr: Optional[np.ndarray] = None,
    ):
        cfg = load_config().pickup
        self._target_label = cfg.target_label
        self._minimum_confidence = cfg.minimum_confidence
        self._detections = list(detections or [])
        self._frame_bgr = (
            np.asarray(frame_bgr, dtype=np.uint8).copy()
            if frame_bgr is not None
            else self._build_frame(self._detections)
        )
        self._started = False

    async def start(self) -> None:
        self._started = True

    async def connect(self) -> None:
        self._started = True

    async def stop(self) -> None:
        self._started = False

    def set_detections(self, detections: Sequence[BallDetection], frame_bgr: Optional[np.ndarray] = None) -> None:
        self._detections = list(detections)
        if frame_bgr is not None:
            self._frame_bgr = np.asarray(frame_bgr, dtype=np.uint8).copy()
        else:
            self._frame_bgr = self._build_frame(self._detections)

    async def get_ball_observation(self, *, include_image: bool = False) -> VisionObservation:
        if not self._started:
            await self.start()
        frame = self._frame_bgr.copy() if include_image and self._frame_bgr is not None else None
        detections = list(self._detections)
        if frame is not None:
            height, width = frame.shape[:2]
            detections = [
                BallDetection(
                    label=detection.label,
                    confidence=detection.confidence,
                    x_min=detection.x_min,
                    y_min=detection.y_min,
                    x_max=detection.x_max,
                    y_max=detection.y_max,
                    image_width=width,
                    image_height=height,
                    source=detection.source,
                    service_name=detection.service_name,
                    camera_name=detection.camera_name,
                )
                for detection in detections
            ]
        selected = select_detection(
            detections,
            target_label=self._target_label,
            minimum_confidence=self._minimum_confidence,
        )
        return VisionObservation(detection=selected, detections=detections, frame_bgr=frame)

    def _build_frame(self, detections: Sequence[BallDetection]) -> np.ndarray:
        frame = np.full((480, 640, 3), 24, dtype=np.uint8)
        frame[:, :, 0] = 30
        frame[:, :, 1] = 36
        frame[:, :, 2] = 42
        for detection in detections:
            x0 = max(0, int(detection.x_min))
            y0 = max(0, int(detection.y_min))
            x1 = min(frame.shape[1], int(detection.x_max))
            y1 = min(frame.shape[0], int(detection.y_max))
            if x1 <= x0 or y1 <= y0:
                continue
            frame[y0:y1, x0:x1, 1] = 160
            frame[y0:y1, x0:x1, 2] = 215
        return frame


def create_viam_vision(fake: Optional[bool] = None):
    if fake is None:
        fake = load_config().pickup.use_fake_vision
    if fake:
        return FakeViamVision()
    return ViamVision()
