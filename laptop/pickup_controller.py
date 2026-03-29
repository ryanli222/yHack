"""
Ball pickup controller built on top of the VIAM detector and arm wrapper.

The first version intentionally assumes a constrained pickup zone and a fixed
grasp motion. It uses a mono-camera estimate from bounding-box size, then turns
that estimate into a small pregrasp -> grasp -> lift state machine.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import math
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from arm_control import ArmPose, create_arm_control
from config import PickupConfig, load_config
from viam_vision import BallDetection, ViamVisionError, create_viam_vision

logger = logging.getLogger(__name__)


class PickupControllerError(RuntimeError):
    """Raised when the pickup controller cannot safely complete a request."""


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def pose_from_meters(x_m: float, y_m: float, z_m: float) -> dict[str, float]:
    """
    Convert meter-space targets into the pose structure expected by ArmControl.

    VIAM poses use millimeters, so we keep the vision math in meters until the
    final command generation step.
    """

    pose = ArmPose(
        x=float(x_m) * 1000.0,
        y=float(y_m) * 1000.0,
        z=float(z_m) * 1000.0,
    )
    return pose.to_dict()


def estimate_pickup_target(
    detection: Optional[BallDetection],
    config: PickupConfig,
) -> Optional[dict[str, Any]]:
    """Project a 2D detection into a constrained mono-camera pickup target."""

    if detection is None or detection.width_px <= 0.0:
        return None

    half_image_width = max(float(detection.image_width) / 2.0, 1.0)
    center_offset_px = float(detection.center_x) - half_image_width
    normalized_offset = center_offset_px / half_image_width
    half_fov_rad = math.radians(float(config.camera_horizontal_fov_deg) / 2.0)
    bearing_rad = normalized_offset * half_fov_rad

    estimated_range_m = (
        float(config.calibration_distance_m) * float(config.calibration_bbox_width_px)
    ) / max(float(detection.width_px), 1.0)
    lateral_m = estimated_range_m * math.tan(bearing_rad)
    centered = abs(center_offset_px) <= float(config.center_tolerance_px)
    in_pickup_zone = (
        float(config.pickup_zone_min_range_m) <= estimated_range_m <= float(config.pickup_zone_max_range_m)
        and abs(lateral_m) <= float(config.pickup_zone_max_lateral_m)
    )

    target_forward_m = max(0.05, estimated_range_m + float(config.arm_forward_offset_m))
    target_lateral_m = lateral_m + float(config.arm_lateral_offset_m)
    base_height_m = float(config.arm_vertical_offset_m)
    pregrasp_forward_m = max(0.05, target_forward_m - float(config.approach_standoff_m))

    pregrasp_pose = pose_from_meters(
        pregrasp_forward_m,
        target_lateral_m,
        base_height_m + float(config.pregrasp_height_m),
    )
    grasp_pose = pose_from_meters(
        target_forward_m,
        target_lateral_m,
        base_height_m + float(config.grasp_height_m),
    )
    lift_pose = pose_from_meters(
        pregrasp_forward_m,
        target_lateral_m,
        base_height_m + float(config.lift_height_m),
    )

    return {
        "label": detection.label,
        "confidence": float(detection.confidence),
        "known_ball_diameter_m": float(config.known_ball_diameter_m),
        "range_m": float(estimated_range_m),
        "lateral_m": float(lateral_m),
        "bearing_deg": math.degrees(bearing_rad),
        "center_offset_px": float(center_offset_px),
        "bbox_width_px": float(detection.width_px),
        "centered": bool(centered),
        "in_pickup_zone": bool(in_pickup_zone),
        "arm_target_m": {
            "x": float(target_forward_m),
            "y": float(target_lateral_m),
        },
        "poses": {
            "pregrasp": pregrasp_pose,
            "grasp": grasp_pose,
            "lift": lift_pose,
        },
    }


class PickupController:
    """High-level pickup coordinator for a detected ball."""

    def __init__(
        self,
        *,
        arm: Any = None,
        vision: Any = None,
        config: Optional[PickupConfig] = None,
        update_hook: Optional[Callable[[dict[str, Any], Any], None]] = None,
    ):
        self._config = config or load_config().pickup
        self._arm = arm or create_arm_control()
        self._vision = vision or create_viam_vision()
        self._update_hook = update_hook

        self._op_lock = asyncio.Lock()
        self._arm_started = False
        self._vision_started = False
        self._stop_requested = False
        self._state: dict[str, Any] = {
            "status": "idle",
            "running": False,
            "detail": "Pickup controller idle.",
            "mode": "fake" if self._config.use_fake_vision else "viam",
            "attempts": 0,
            "last_detection": None,
            "last_target": None,
            "last_result": None,
            "last_error": "",
            "updated_at": _iso_now(),
        }

    async def start(self) -> None:
        await self._ensure_vision_started()

    async def stop(self) -> None:
        self._stop_requested = True
        try:
            if self._arm_started:
                await self._arm.stop()
        finally:
            self._arm_started = False
        try:
            if self._vision_started:
                await self._vision.stop()
        finally:
            self._vision_started = False
        self._update_state(
            status="idle",
            running=False,
            detail="Pickup controller stopped.",
        )

    def get_state(self) -> dict[str, Any]:
        return copy.deepcopy(self._state)

    async def refresh_detection(self, *, include_image: bool = True) -> dict[str, Any]:
        async with self._op_lock:
            observation = await self._observe(include_image=include_image)
            detection = observation.detection
            target = estimate_pickup_target(detection, self._config)
            status, detail = self._status_for_detection(detection, target)
            self._update_state(
                status=status,
                running=False,
                detail=detail,
                last_detection=None if detection is None else detection.to_dict(),
                last_target=target,
                last_result={
                    "ok": detection is not None and target is not None and bool(target["in_pickup_zone"]) and bool(target["centered"]),
                    "action": "refresh_detection",
                },
                last_error="",
                frame=observation.frame_bgr,
            )
            return self.get_state()

    async def pickup_ball(self) -> dict[str, Any]:
        async with self._op_lock:
            self._stop_requested = False
            await self._ensure_vision_started()
            await self._ensure_arm_started()

            total_attempts = max(1, int(self._config.max_retries) + 1)
            self._update_state(
                status="running",
                running=True,
                attempts=0,
                detail="Searching for the ball.",
                last_result=None,
                last_error="",
            )

            last_error = "No detection available."
            for attempt in range(1, total_attempts + 1):
                self._check_stop_requested()
                observation = await self._observe(include_image=True)
                detection = observation.detection
                target = estimate_pickup_target(detection, self._config)
                self._update_state(
                    status="running",
                    running=True,
                    attempts=attempt,
                    detail=f"Evaluating pickup target ({attempt}/{total_attempts}).",
                    last_detection=None if detection is None else detection.to_dict(),
                    last_target=target,
                    last_error="",
                    frame=observation.frame_bgr,
                )

                if detection is None:
                    last_error = "No ball detected."
                    if attempt < total_attempts:
                        await asyncio.sleep(0.2)
                        continue
                    break

                if target is None:
                    last_error = "Ball detection was missing usable geometry."
                    break

                if not target["in_pickup_zone"]:
                    detail = "Ball detected outside the pickup zone. Reposition the arm or base."
                    self._update_state(
                        status="warning",
                        running=False,
                        detail=detail,
                        last_result={"ok": False, "reason": "out_of_zone", "attempt": attempt},
                    )
                    return self.get_state()

                if not target["centered"]:
                    detail = "Ball detected but not centered enough for a safe pickup."
                    self._update_state(
                        status="warning",
                        running=False,
                        detail=detail,
                        last_result={"ok": False, "reason": "off_center", "attempt": attempt},
                    )
                    return self.get_state()

                result = await self._execute_pickup_attempt(target, attempt=attempt)
                if result["ok"]:
                    return self.get_state()
                last_error = str(result.get("error") or "Pickup attempt failed.")
                if attempt < total_attempts:
                    await asyncio.sleep(0.2)

            self._update_state(
                status="error",
                running=False,
                detail=last_error,
                last_result={"ok": False, "reason": "pickup_failed", "attempts": total_attempts},
                last_error=last_error,
            )
            return self.get_state()

    async def stop_pickup(self) -> dict[str, Any]:
        self._stop_requested = True
        try:
            if self._arm_started:
                await self._arm.emergency_stop()
        except Exception:
            logger.debug("Ignoring arm stop error during pickup stop", exc_info=True)
        self._update_state(
            status="stopped",
            running=False,
            detail="Pickup stop requested.",
            last_result={"ok": False, "stopped": True},
        )
        return self.get_state()

    async def _ensure_vision_started(self) -> None:
        if self._vision_started:
            return
        try:
            await self._vision.start()
        except ViamVisionError:
            raise
        except Exception as exc:
            raise PickupControllerError(f"Failed to start pickup vision: {exc}") from exc
        self._vision_started = True

    async def _ensure_arm_started(self) -> None:
        if self._arm_started:
            return
        try:
            await self._arm.start()
        except Exception as exc:
            raise PickupControllerError(f"Failed to start arm control: {exc}") from exc
        self._arm_started = True

    async def _observe(self, *, include_image: bool) -> Any:
        try:
            return await self._vision.get_ball_observation(include_image=include_image)
        except ViamVisionError:
            raise
        except Exception as exc:
            raise PickupControllerError(f"Failed to refresh pickup detections: {exc}") from exc

    async def _execute_pickup_attempt(self, target: dict[str, Any], *, attempt: int) -> dict[str, Any]:
        try:
            self._check_stop_requested()
            self._update_state(
                status="running",
                running=True,
                detail=f"Opening gripper (attempt {attempt}).",
            )
            await self._arm.open_gripper()

            self._check_stop_requested()
            self._update_state(
                status="running",
                running=True,
                detail="Moving to pregrasp pose.",
            )
            await self._arm.move_to_position(target["poses"]["pregrasp"])

            self._check_stop_requested()
            self._update_state(
                status="running",
                running=True,
                detail="Descending to grasp pose.",
            )
            await self._arm.move_to_position(target["poses"]["grasp"])

            self._check_stop_requested()
            self._update_state(
                status="running",
                running=True,
                detail="Closing gripper.",
            )
            holding = bool(await self._arm.close_gripper())
            if not holding:
                await self._arm.open_gripper()
                self._update_state(
                    status="running",
                    running=True,
                    detail="Grasp not confirmed. Preparing retry.",
                    last_result={"ok": False, "reason": "not_holding", "attempt": attempt},
                )
                return {"ok": False, "error": "Gripper did not confirm a grasp."}

            self._check_stop_requested()
            self._update_state(
                status="running",
                running=True,
                detail="Lifting ball.",
            )
            await self._arm.move_to_position(target["poses"]["lift"])

            home_note = ""
            if self._config.home_after_pickup:
                self._check_stop_requested()
                self._update_state(
                    status="running",
                    running=True,
                    detail="Returning arm home.",
                )
                try:
                    await self._arm.home()
                except Exception as exc:
                    home_note = f" Pickup succeeded, but homing failed: {exc}"
                    logger.warning(home_note)

            self._update_state(
                status="success",
                running=False,
                detail=f"Ball pickup complete.{home_note}",
                last_result={"ok": True, "holding": True, "attempt": attempt},
                last_error="",
            )
            return {"ok": True}
        except PickupControllerError as exc:
            if "stop requested" in str(exc).lower():
                self._update_state(
                    status="stopped",
                    running=False,
                    detail="Pickup stopped.",
                    last_result={"ok": False, "stopped": True},
                )
                return {"ok": False, "error": str(exc)}
            raise
        except Exception as exc:
            try:
                await self._arm.emergency_stop()
            except Exception:
                logger.debug("Ignoring emergency stop failure after pickup error", exc_info=True)
            self._update_state(
                status="running",
                running=True,
                detail=f"Pickup attempt failed: {exc}",
                last_result={"ok": False, "reason": "arm_error", "attempt": attempt},
                last_error=str(exc),
            )
            return {"ok": False, "error": str(exc)}

    def _status_for_detection(
        self,
        detection: Optional[BallDetection],
        target: Optional[dict[str, Any]],
    ) -> tuple[str, str]:
        if detection is None:
            return "searching", "No ball detected."
        if target is None:
            return "warning", "Detection found, but range estimate was unavailable."
        if not target["in_pickup_zone"]:
            return "warning", "Ball detected outside the pickup zone."
        if not target["centered"]:
            return "warning", "Ball detected, but it is not centered enough to grab."
        return "ready", "Ball is aligned and inside the pickup zone."

    def _check_stop_requested(self) -> None:
        if self._stop_requested:
            raise PickupControllerError("Pickup stop requested")

    def _update_state(self, frame: Any = None, **updates: Any) -> None:
        self._state.update(updates)
        self._state["updated_at"] = _iso_now()
        if self._update_hook is not None:
            self._update_hook(copy.deepcopy(self._state), frame)


def create_pickup_controller(
    *,
    arm: Any = None,
    vision: Any = None,
    config: Optional[PickupConfig] = None,
    update_hook: Optional[Callable[[dict[str, Any], Any], None]] = None,
):
    return PickupController(
        arm=arm,
        vision=vision,
        config=config,
        update_hook=update_hook,
    )
