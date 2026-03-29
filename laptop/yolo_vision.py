"""
OpenCV DNN YOLOv8 detector for dashboard target selection.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence
from urllib import request as urlrequest

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

YOLOV8N_ONNX_URL = "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8n.onnx?download=true"

COCO_CLASS_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)


@dataclass
class YoloDetection:
    label: str
    confidence: float
    x: int
    y: int
    width_px: int
    height_px: int
    center_x: float
    center_y: float
    width_ratio: float
    area_ratio: float

    def to_dict(self) -> dict:
        return asdict(self)


def select_target_detection(
    detections: Sequence[YoloDetection],
    target_label: str,
    *,
    distance_preference: Optional[str] = None,
) -> Optional[YoloDetection]:
    label = target_label.strip().lower()
    if not label:
        return None

    candidates = []
    for detection in detections:
        det_label = detection.label.strip().lower()
        if label not in det_label and det_label not in label:
            continue
        candidates.append(detection)

    if not candidates:
        return None

    preference = (distance_preference or "").strip().lower()
    if preference == "closest":
        return max(candidates, key=lambda det: (float(det.area_ratio), float(det.confidence)))
    if preference == "farthest":
        return min(candidates, key=lambda det: (float(det.area_ratio), -float(det.confidence)))

    return max(candidates, key=lambda det: (float(det.confidence), float(det.area_ratio)))


class YoloV8OpenCvDetector:
    def __init__(
        self,
        *,
        confidence_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        input_size: int = 640,
        model_path: Optional[Path] = None,
    ) -> None:
        self._confidence_threshold = float(confidence_threshold)
        self._nms_threshold = float(nms_threshold)
        self._input_size = int(input_size)

        path = model_path or (
            Path(__file__).resolve().parent.parent / "data" / "models" / "yolo" / "yolov8n.onnx"
        )
        self._ensure_file(path, YOLOV8N_ONNX_URL)
        self._net = cv2.dnn.readNetFromONNX(str(path))
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect(self, frame_bgr: np.ndarray) -> list[YoloDetection]:
        frame_height, frame_width = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame_bgr,
            scalefactor=1.0 / 255.0,
            size=(self._input_size, self._input_size),
            mean=(0.0, 0.0, 0.0),
            swapRB=True,
            crop=False,
        )
        self._net.setInput(blob)
        outputs = self._net.forward()
        predictions = self._reshape_predictions(outputs)

        boxes: list[list[int]] = []
        confidences: list[float] = []
        class_ids: list[int] = []
        x_factor = float(frame_width) / float(self._input_size)
        y_factor = float(frame_height) / float(self._input_size)

        for prediction in predictions:
            if prediction.shape[0] < 6:
                continue
            class_scores = prediction[4:]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            if confidence < self._confidence_threshold:
                continue

            cx, cy, width, height = [float(value) for value in prediction[:4]]
            left = int((cx - width / 2.0) * x_factor)
            top = int((cy - height / 2.0) * y_factor)
            width_px = int(width * x_factor)
            height_px = int(height * y_factor)
            if width_px <= 0 or height_px <= 0:
                continue

            left = max(0, min(frame_width - 1, left))
            top = max(0, min(frame_height - 1, top))
            width_px = min(frame_width - left, width_px)
            height_px = min(frame_height - top, height_px)
            if width_px <= 0 or height_px <= 0:
                continue

            boxes.append([left, top, width_px, height_px])
            confidences.append(confidence)
            class_ids.append(class_id)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self._confidence_threshold,
            self._nms_threshold,
        )
        if indices is None or len(indices) == 0:
            return []

        flat_indices = np.asarray(indices).reshape(-1).tolist()
        frame_area = float(max(frame_width * frame_height, 1))
        detections: list[YoloDetection] = []
        for idx in flat_indices:
            x, y, width_px, height_px = boxes[int(idx)]
            class_id = class_ids[int(idx)]
            label = COCO_CLASS_NAMES[class_id] if 0 <= class_id < len(COCO_CLASS_NAMES) else f"class_{class_id}"
            detections.append(
                YoloDetection(
                    label=label,
                    confidence=float(confidences[int(idx)]),
                    x=int(x),
                    y=int(y),
                    width_px=int(width_px),
                    height_px=int(height_px),
                    center_x=float(x + width_px / 2.0),
                    center_y=float(y + height_px / 2.0),
                    width_ratio=float(width_px) / float(max(frame_width, 1)),
                    area_ratio=float(width_px * height_px) / frame_area,
                )
            )
        return detections

    def _reshape_predictions(self, outputs: np.ndarray) -> np.ndarray:
        array = np.asarray(outputs)
        array = np.squeeze(array)
        if array.ndim == 1:
            return array.reshape(1, -1)
        if array.ndim == 2:
            if array.shape[0] in (84, 85) and array.shape[1] > array.shape[0]:
                return array.T
            return array
        if array.ndim == 3:
            if array.shape[1] in (84, 85):
                return array[0].T
            if array.shape[2] in (84, 85):
                return array[0]
        raise RuntimeError(f"Unexpected YOLO output shape: {array.shape}")

    def _ensure_file(self, path: Path, url: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 0:
            return

        tmp_path = path.with_suffix(path.suffix + ".part")
        LOGGER.info("Downloading %s...", path.name)
        try:
            with urlrequest.urlopen(url, timeout=60) as response, open(tmp_path, "wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
        except Exception as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(f"Failed to download {path.name} from {url}: {exc}") from exc

        tmp_path.replace(path)
