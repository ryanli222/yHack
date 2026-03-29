"""
Calibrated arm-kinematics helpers for SO-101-style arms.

This module adds a small, tunable layer between logical IK joint angles and the
joint coordinates exposed by VIAM. It is intentionally lightweight:
- configurable base-frame origin offsets
- configurable link lengths
- configurable per-joint multipliers and offsets

The solver uses a base yaw + 3R planar approximation, which matches the
structure used by several SO-101 examples. It is meant as a practical fallback
layer, not a full rigid-body model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from arm_control import ArmControlError, ArmPose
from config import Config, load_config


def _wrap_degrees(value: float) -> float:
    return ((float(value) + 180.0) % 360.0) - 180.0


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _normalize_vector(vector: Sequence[float]) -> list[float]:
    magnitude = math.sqrt(sum(float(component) * float(component) for component in vector))
    if magnitude < 1e-6:
        raise ArmControlError("Cannot normalize near-zero vector")
    return [float(component) / magnitude for component in vector]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _padded(values: Sequence[float], size: int, *, default: float) -> list[float]:
    padded = [float(value) for value in values[:size]]
    if len(padded) < size:
        padded.extend([float(default)] * (size - len(padded)))
    return padded


@dataclass
class ArmKinematicsCalibration:
    enabled: bool
    origin_offset_mm: tuple[float, float, float]
    link_lengths_mm: tuple[float, float, float]
    joint_multipliers: tuple[float, float, float, float]
    joint_offsets_deg: tuple[float, float, float, float]
    default_tool_pitch_deg: float

    @classmethod
    def from_config(cls, cfg: Config | None = None) -> "ArmKinematicsCalibration":
        arm_cfg = (cfg or load_config()).arm
        origin_offset = tuple(_padded(arm_cfg.ik_origin_offset_mm, 3, default=0.0))
        link_lengths = tuple(_padded(arm_cfg.ik_link_lengths_mm, 3, default=0.0))
        joint_multipliers = tuple(
            scale if abs(scale) >= 1e-6 else 1.0
            for scale in _padded(arm_cfg.ik_joint_multipliers, 4, default=1.0)
        )
        joint_offsets = tuple(_padded(arm_cfg.ik_joint_offsets_deg, 4, default=0.0))
        return cls(
            enabled=bool(arm_cfg.calibrated_ik_enabled),
            origin_offset_mm=origin_offset,
            link_lengths_mm=link_lengths,
            joint_multipliers=joint_multipliers,
            joint_offsets_deg=joint_offsets,
            default_tool_pitch_deg=float(arm_cfg.ik_default_tool_pitch_deg),
        )

    def logical_from_viam(self, joints: Sequence[float]) -> list[float]:
        viam_joints = [float(value) for value in joints]
        logical = list(viam_joints)
        for idx in range(min(4, len(viam_joints))):
            logical[idx] = (float(viam_joints[idx]) - self.joint_offsets_deg[idx]) / self.joint_multipliers[idx]
        return logical

    def viam_from_logical(self, joints: Sequence[float]) -> list[float]:
        logical = [float(value) for value in joints]
        viam = list(logical)
        for idx in range(min(4, len(logical))):
            viam[idx] = logical[idx] * self.joint_multipliers[idx] + self.joint_offsets_deg[idx]
        return viam

    def _tool_pitch_radians(self, pose: ArmPose, base_yaw_rad: float) -> float:
        try:
            forward = _normalize_vector([pose.o_x, pose.o_y, pose.o_z])
        except ArmControlError:
            return math.radians(self.default_tool_pitch_deg)

        radial_unit = [math.cos(base_yaw_rad), math.sin(base_yaw_rad), 0.0]
        radial_component = _dot(forward, radial_unit)
        vertical_component = float(forward[2])
        if abs(radial_component) < 1e-6 and abs(vertical_component) < 1e-6:
            return math.radians(self.default_tool_pitch_deg)
        return math.atan2(vertical_component, radial_component)

    def _solve_planar_candidates(
        self,
        radius_mm: float,
        z_mm: float,
        tool_pitch_rad: float,
    ) -> list[tuple[float, float, float]]:
        l1, l2, l3 = self.link_lengths_mm
        if min(l1, l2, l3) <= 0.0:
            raise ArmControlError("IK link lengths must all be positive")

        wrist_radius = float(radius_mm) - l3 * math.cos(tool_pitch_rad)
        wrist_height = float(z_mm) - l3 * math.sin(tool_pitch_rad)
        radius_squared = wrist_radius * wrist_radius + wrist_height * wrist_height

        cosine_elbow = (radius_squared - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
        if cosine_elbow < -1.0001 or cosine_elbow > 1.0001:
            raise ArmControlError("Calibrated IK target is outside the planar workspace")
        cosine_elbow = _clamp(cosine_elbow, -1.0, 1.0)

        sine_magnitude = math.sqrt(max(0.0, 1.0 - cosine_elbow * cosine_elbow))
        candidates: list[tuple[float, float, float]] = []
        for elbow_sign in (1.0, -1.0):
            sine_elbow = elbow_sign * sine_magnitude
            elbow_rad = math.atan2(sine_elbow, cosine_elbow)
            k1 = l1 + l2 * cosine_elbow
            k2 = l2 * sine_elbow
            shoulder_rad = math.atan2(wrist_height, wrist_radius) - math.atan2(k2, k1)
            wrist_rad = tool_pitch_rad - shoulder_rad - elbow_rad
            candidates.append(
                (
                    _wrap_degrees(math.degrees(shoulder_rad)),
                    _wrap_degrees(math.degrees(elbow_rad)),
                    _wrap_degrees(math.degrees(wrist_rad)),
                )
            )
        return candidates

    def solve_pose_to_joint_targets(
        self,
        target_pose: ArmPose | Mapping[str, Any],
        current_pose: ArmPose | Mapping[str, Any],
        current_joint_positions: Sequence[float],
    ) -> tuple[list[float], dict[str, float]]:
        if not self.enabled:
            raise ArmControlError("Calibrated IK fallback is disabled")

        pose = ArmPose.from_value(target_pose)
        x = float(pose.x) - self.origin_offset_mm[0]
        y = float(pose.y) - self.origin_offset_mm[1]
        z = float(pose.z) - self.origin_offset_mm[2]
        radius_mm = math.hypot(x, y)
        base_yaw_rad = math.atan2(y, x) if radius_mm > 1e-6 else 0.0
        tool_pitch_rad = self._tool_pitch_radians(pose, base_yaw_rad)
        base_yaw_deg = _wrap_degrees(math.degrees(base_yaw_rad))

        candidates = self._solve_planar_candidates(radius_mm, z, tool_pitch_rad)

        current_viam = [float(value) for value in current_joint_positions]
        current_logical = self.logical_from_viam(current_viam)
        current_logical = _padded(current_logical, 4, default=0.0)

        best_solution: list[float] | None = None
        best_score: float | None = None
        for shoulder_deg, elbow_deg, wrist_deg in candidates:
            candidate = [base_yaw_deg, shoulder_deg, elbow_deg, wrist_deg]
            score = sum(abs(_wrap_degrees(candidate[idx] - current_logical[idx])) for idx in range(4))
            if best_score is None or score < best_score:
                best_solution = candidate
                best_score = score

        if best_solution is None:
            raise ArmControlError("No calibrated IK solution candidates were produced")

        viam_target = list(current_viam)
        if len(viam_target) < 4:
            viam_target.extend([0.0] * (4 - len(viam_target)))
        mapped = self.viam_from_logical(best_solution)
        for idx in range(4):
            viam_target[idx] = float(mapped[idx])

        return viam_target, {
            "base_yaw_degrees": best_solution[0],
            "shoulder_degrees": best_solution[1],
            "elbow_degrees": best_solution[2],
            "wrist_degrees": best_solution[3],
            "radius_mm": radius_mm,
            "z_mm": z,
            "tool_pitch_degrees": _wrap_degrees(math.degrees(tool_pitch_rad)),
            "origin_offset_x_mm": self.origin_offset_mm[0],
            "origin_offset_y_mm": self.origin_offset_mm[1],
            "origin_offset_z_mm": self.origin_offset_mm[2],
        }


def calibrated_joint_targets_for_pose(
    target_pose: ArmPose | Mapping[str, Any],
    current_pose: ArmPose | Mapping[str, Any],
    current_joint_positions: Sequence[float],
    *,
    calibration: ArmKinematicsCalibration | None = None,
) -> tuple[list[float], dict[str, float]]:
    calibration = calibration or ArmKinematicsCalibration.from_config()
    return calibration.solve_pose_to_joint_targets(target_pose, current_pose, current_joint_positions)
