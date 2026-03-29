"""
Transcript-to-arm command parsing and dispatch.

This keeps the first voice-to-arm path deterministic and easy to test.
It mirrors the shape of llm_interface.py by returning a command dict plus
short spoken feedback, but uses regex parsing for the initial milestone.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Optional, Sequence, Tuple

from arm_control import ArmControlError, ArmPose

_MOVE_VERB_PATTERN = r"(?:move|set|put|place|bring|take|go|position|turn|rotate)"
_TOOL_PATTERN = r"(?:(?:the|my)\s+)?(?:arm|gripper|tool|hand|claw|end[\s-]?effector)"
_OPTIONAL_TOOL_PATTERN = rf"(?:{_TOOL_PATTERN}\s+)?"
_NUMBER_PATTERN = r"(-?\d+(?:\.\d+)?)"
_POSE_DIRECTION_PATTERN = r"(up|down|left|right|forward|forwards|back|backward|backwards)"
_POSE_UNIT_PATTERN = r"(mm|millimeters?|cm|centimeters?|m|meters?|units?)"
_DEGREE_UNIT_PATTERN = r"(deg|degree|degrees)"
_ROTATION_DIRECTION_PATTERN = r"(left|right|up|down)"

_UNIT_TO_MM = {
    "mm": 1.0,
    "millimeter": 1.0,
    "millimeters": 1.0,
    "unit": 1.0,
    "units": 1.0,
    "cm": 10.0,
    "centimeter": 10.0,
    "centimeters": 10.0,
    "m": 1000.0,
    "meter": 1000.0,
    "meters": 1000.0,
}

_POSE_DIRECTION_TO_DELTA = {
    "up": ("z", 1.0),
    "down": ("z", -1.0),
    "forward": ("x", 1.0),
    "forwards": ("x", 1.0),
    "back": ("x", -1.0),
    "backward": ("x", -1.0),
    "backwards": ("x", -1.0),
}


def _extract_numbers(text: str) -> list[float]:
    return [float(match) for match in re.findall(r"-?\d+(?:\.\d+)?", text)]


def _distance_to_mm(distance_text: str, unit_text: Optional[str]) -> float:
    try:
        distance = abs(float(distance_text))
    except (TypeError, ValueError) as exc:
        raise ArmControlError(f"Invalid relative move distance: {distance_text!r}") from exc
    unit_key = (unit_text or "mm").strip().lower()
    return distance * _UNIT_TO_MM.get(unit_key, 1.0)


def _pose_relative_delta(direction: str, distance_mm: float) -> dict[str, float]:
    normalized_direction = direction.strip().lower()
    axis, sign = _POSE_DIRECTION_TO_DELTA[normalized_direction]
    delta = {"x": 0.0, "y": 0.0, "z": 0.0}
    delta[axis] = float(distance_mm) * sign
    return delta


def _normalize_vector(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(component * component for component in vector))
    if magnitude < 1e-6:
        raise ArmControlError("Cannot normalize near-zero vector")
    return [component / magnitude for component in vector]


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _rotate_about_axis(vector: list[float], axis: list[float], angle_deg: float) -> list[float]:
    angle_rad = math.radians(float(angle_deg))
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    axis_unit = _normalize_vector(axis)
    cross_term = _cross(axis_unit, vector)
    dot_term = _dot(axis_unit, vector)
    return [
        vector[idx] * cos_a + cross_term[idx] * sin_a + axis_unit[idx] * dot_term * (1.0 - cos_a)
        for idx in range(3)
    ]


def _tool_forward_delta(current_pose: ArmPose | Mapping[str, Any], distance_mm: float) -> dict[str, float]:
    pose_value = ArmPose.from_value(current_pose)
    direction = _normalize_vector([float(pose_value.o_x), float(pose_value.o_y), float(pose_value.o_z)])
    scale = float(distance_mm)
    return {
        "x": direction[0] * scale,
        "y": direction[1] * scale,
        "z": direction[2] * scale,
    }


def _tool_lateral_delta(current_pose: ArmPose | Mapping[str, Any], distance_mm: float) -> dict[str, float]:
    pose_value = ArmPose.from_value(current_pose)
    forward = _normalize_vector([float(pose_value.o_x), float(pose_value.o_y), float(pose_value.o_z)])
    reference_up = [0.0, 0.0, 1.0]
    if abs(_dot(forward, reference_up)) >= 0.999:
        reference_up = [1.0, 0.0, 0.0]
    left = _normalize_vector(_cross(reference_up, forward))
    left = _normalize_vector(_rotate_about_axis(left, forward, float(pose_value.theta)))
    scale = float(distance_mm)
    return {
        "x": left[0] * scale,
        "y": left[1] * scale,
        "z": left[2] * scale,
    }


def target_pose_from_fields(
    current_pose: ArmPose | Mapping[str, Any],
    pose_fields: Mapping[str, Any],
) -> ArmPose:
    pose_value = ArmPose.from_value(current_pose)
    if not isinstance(pose_fields, Mapping):
        raise ArmControlError("Pose fields must be a mapping")
    return ArmPose(
        x=float(pose_fields.get("x", pose_value.x)),
        y=float(pose_fields.get("y", pose_value.y)),
        z=float(pose_fields.get("z", pose_value.z)),
        o_x=float(pose_fields.get("o_x", pose_value.o_x)),
        o_y=float(pose_fields.get("o_y", pose_value.o_y)),
        o_z=float(pose_fields.get("o_z", pose_value.o_z)),
        theta=float(pose_fields.get("theta", pose_value.theta)),
    )


def target_pose_from_relative(
    current_pose: ArmPose | Mapping[str, Any],
    command: Mapping[str, Any],
) -> ArmPose:
    pose_value = ArmPose.from_value(current_pose)
    direction = str(command.get("direction", "")).strip().lower()
    distance_mm = float(command.get("distance_mm", 0.0))
    if direction in {"forward", "forwards"}:
        delta = _tool_forward_delta(pose_value, distance_mm)
    elif direction in {"back", "backward", "backwards"}:
        delta = _tool_forward_delta(pose_value, -distance_mm)
    elif direction == "left":
        delta = _tool_lateral_delta(pose_value, distance_mm)
    elif direction == "right":
        delta = _tool_lateral_delta(pose_value, -distance_mm)
    elif direction in {"up", "down"}:
        delta = _pose_relative_delta(direction, distance_mm)
    else:
        delta = command.get("delta", {})
        if not isinstance(delta, Mapping):
            raise ArmControlError("move_relative command requires a delta mapping")
    return target_pose_from_fields(
        pose_value,
        {
            "x": float(pose_value.x) + float(delta.get("x", 0.0)),
            "y": float(pose_value.y) + float(delta.get("y", 0.0)),
            "z": float(pose_value.z) + float(delta.get("z", 0.0)),
        },
    )


class RegexArmParser:
    """Small deterministic parser for arm/gripper voice commands."""

    async def parse(self, text: str) -> Tuple[dict[str, Any], str]:
        text_lower = text.lower().strip()
        if not text_lower:
            return {"cmd": "unknown", "message": "Empty transcript"}, "I didn't hear a command."

        if re.search(r"\b(exit|quit|goodbye)\b", text_lower):
            return {"cmd": "exit"}, "Exiting arm control."

        if re.search(r"\b(stop|halt|freeze|cancel)\b", text_lower):
            return {"cmd": "stop"}, "Stopping the arm."

        if re.search(r"\b(open)\b.*\b(gripper|hand|claw)\b", text_lower) or re.search(
            r"\b(gripper|hand|claw)\b.*\b(open)\b", text_lower
        ):
            return {"cmd": "open_gripper"}, "Opening the gripper."

        if re.search(r"\b(close|grab)\b.*\b(gripper|hand|claw)\b", text_lower) or re.search(
            r"\b(gripper|hand|claw)\b.*\b(close|grab)\b", text_lower
        ):
            return {"cmd": "close_gripper"}, "Closing the gripper."

        if re.search(r"\b(home|reset|park)\b", text_lower) and re.search(r"\barm\b", text_lower):
            return {"cmd": "home"}, "Moving the arm home."

        match = re.search(
            rf"(?:turn|rotate)\s+{_OPTIONAL_TOOL_PATTERN}{_ROTATION_DIRECTION_PATTERN}\s+(?:by\s+)?{_NUMBER_PATTERN}\s*(?:{_DEGREE_UNIT_PATTERN})?\b",
            text_lower,
        )
        if not match:
            match = re.search(
                rf"(?:turn|rotate)\s+{_OPTIONAL_TOOL_PATTERN}(?:by\s+)?{_NUMBER_PATTERN}\s*(?:{_DEGREE_UNIT_PATTERN})?\s+{_ROTATION_DIRECTION_PATTERN}\b",
                text_lower,
            )
            if match:
                amount_text = match.group(1)
                direction = match.group(3)
            else:
                amount_text = ""
                direction = ""
        else:
            direction = match.group(1)
            amount_text = match.group(2)
        if not match:
            match = re.search(
                rf"(?:move|go|nudge|shift|slide|bring)\s+{_OPTIONAL_TOOL_PATTERN}{_ROTATION_DIRECTION_PATTERN}\s+(?:by\s+)?"
                rf"{_NUMBER_PATTERN}\s*(?:{_DEGREE_UNIT_PATTERN})\b",
                text_lower,
            )
            if not match:
                match = re.search(
                    rf"(?:move|go|nudge|shift|slide|bring)\s+{_OPTIONAL_TOOL_PATTERN}(?:by\s+)?"
                    rf"{_NUMBER_PATTERN}\s*(?:{_DEGREE_UNIT_PATTERN})\s+{_ROTATION_DIRECTION_PATTERN}\b",
                    text_lower,
                )
                if match:
                    amount_text = match.group(1)
                    direction = match.group(3)
                else:
                    amount_text = ""
                    direction = ""
            else:
                direction = match.group(1)
                amount_text = match.group(2)
        if match:
            delta_degrees = abs(float(amount_text))
            if direction in {"right", "down"}:
                delta_degrees *= -1.0
            joint_index = 0 if direction in {"left", "right"} else 1
            return (
                {
                    "cmd": "move_joint_relative",
                    "direction": direction,
                    "joint_index": joint_index,
                    "delta_degrees": delta_degrees,
                },
                f"Rotating {direction} {abs(delta_degrees):.1f} degrees.",
            )

        match = re.search(
            rf"(?:move|go|nudge|shift|slide|bring)\s+{_OPTIONAL_TOOL_PATTERN}{_POSE_DIRECTION_PATTERN}\s+(?:by\s+)?"
            rf"{_NUMBER_PATTERN}\s*(?:{_POSE_UNIT_PATTERN})?\b",
            text_lower,
        )
        if not match:
            match = re.search(
                rf"(?:move|go|nudge|shift|slide|bring)\s+{_OPTIONAL_TOOL_PATTERN}(?:by\s+)?"
                rf"{_NUMBER_PATTERN}\s*(?:{_POSE_UNIT_PATTERN})?\s+{_POSE_DIRECTION_PATTERN}\b",
                text_lower,
            )
            if match:
                distance_text = match.group(1)
                unit_text = match.group(2)
                direction = match.group(3)
            else:
                distance_text = ""
                unit_text = None
                direction = ""
        else:
            direction = match.group(1)
            distance_text = match.group(2)
            unit_text = match.group(3)
        if match:
            distance_mm = _distance_to_mm(distance_text, unit_text)
            command = {
                "cmd": "move_relative",
                "direction": direction,
                "distance_mm": distance_mm,
            }
            if direction in {"up", "down"}:
                command["delta"] = _pose_relative_delta(direction, distance_mm)
            return (
                command,
                f"Moving the gripper {direction} {distance_mm:.1f} millimeters.",
            )

        match = re.search(
            rf"{_MOVE_VERB_PATTERN}\s+{_OPTIONAL_TOOL_PATTERN}([xyz])\s*(?:to|=)\s*{_NUMBER_PATTERN}\b",
            text_lower,
        )
        if match:
            axis = match.group(1)
            value = float(match.group(2))
            return (
                {"cmd": "move_axis_to", "axis": axis, "value": value},
                f"Moving gripper {axis.upper()} to {value:.1f}.",
            )

        match = re.search(
            r"(?:move|set)\s+(?:joint\s+)?(\d+)\s+(?:to\s+)?(-?\d+(?:\.\d+)?)",
            text_lower,
        )
        if match:
            joint_index = max(0, int(match.group(1)) - 1)
            value = float(match.group(2))
            return (
                {"cmd": "move_joint", "joint_index": joint_index, "value": value},
                f"Moving joint {joint_index + 1}.",
            )

        match = re.search(
            r"(?:set|move)\s+(?:the\s+)?joints?\s+(?:to\s+)?(.+)",
            text_lower,
        )
        if match:
            joints = _extract_numbers(match.group(1))
            if joints:
                return (
                    {"cmd": "move_joints", "joints": joints},
                    f"Moving {len(joints)} joints.",
                )

        match = re.search(
            rf"{_MOVE_VERB_PATTERN}\s+{_OPTIONAL_TOOL_PATTERN}(?:to|at|pose|position)\s+"
            rf"x\s*=?\s*{_NUMBER_PATTERN}\s*[, ]+\s*"
            rf"y\s*=?\s*{_NUMBER_PATTERN}\s*[, ]+\s*"
            rf"z\s*=?\s*{_NUMBER_PATTERN}",
            text_lower,
        )
        if match:
            pose = {
                "x": float(match.group(1)),
                "y": float(match.group(2)),
                "z": float(match.group(3)),
            }
            return {"cmd": "move_to_position", "pose": pose}, "Moving the gripper."

        match = re.search(
            rf"{_MOVE_VERB_PATTERN}\s+{_OPTIONAL_TOOL_PATTERN}(?:to|at|pose|position)\s+"
            rf"{_NUMBER_PATTERN}\s*[, ]+\s*{_NUMBER_PATTERN}\s*[, ]+\s*{_NUMBER_PATTERN}\b",
            text_lower,
        )
        if match:
            pose = {
                "x": float(match.group(1)),
                "y": float(match.group(2)),
                "z": float(match.group(3)),
            }
            return {"cmd": "move_to_position", "pose": pose}, "Moving the gripper."

        return (
            {"cmd": "unknown", "message": f"Could not understand: {text}"},
            "I didn't understand that arm command.",
        )


def create_arm_parser(provider: Optional[str] = None) -> RegexArmParser:
    del provider
    return RegexArmParser()


async def dispatch_arm_command(arm: Any, command: Mapping[str, Any]) -> dict[str, Any]:
    cmd = str(command.get("cmd", "unknown"))

    if cmd == "open_gripper":
        await arm.open_gripper()
        return {"ok": True, "handled": cmd}

    if cmd == "close_gripper":
        holding = await arm.close_gripper()
        return {"ok": True, "handled": cmd, "holding": bool(holding)}

    if cmd == "home":
        await arm.home()
        return {"ok": True, "handled": cmd}

    if cmd == "stop":
        await arm.emergency_stop()
        return {"ok": True, "handled": cmd}

    if cmd == "move_joint":
        state = await arm.get_state()
        joint_index = int(command["joint_index"])
        value = float(command["value"])
        current = list(state.joint_positions)
        if not current:
            current = [0.0] * max(joint_index + 1, 5)
        elif joint_index >= len(current):
            current.extend([0.0] * (joint_index + 1 - len(current)))
        current[joint_index] = value
        await arm.move_joints(current)
        return {"ok": True, "handled": cmd, "joint_index": joint_index, "value": value}

    if cmd == "move_joint_relative":
        state = await arm.get_state()
        joint_index = int(command["joint_index"])
        delta_degrees = float(command["delta_degrees"])
        current = list(state.joint_positions)
        if not current:
            current = [0.0] * max(joint_index + 1, 5)
        elif joint_index >= len(current):
            current.extend([0.0] * (joint_index + 1 - len(current)))
        current[joint_index] = float(current[joint_index]) + delta_degrees
        await arm.move_joints(current)
        return {
            "ok": True,
            "handled": cmd,
            "joint_index": joint_index,
            "delta_degrees": delta_degrees,
            "value": current[joint_index],
        }

    if cmd == "move_joints":
        joints = [float(value) for value in command.get("joints", [])]
        if not joints:
            raise ArmControlError("move_joints command requires joint values")
        await arm.move_joints(joints)
        return {"ok": True, "handled": cmd, "joint_count": len(joints)}

    if cmd == "move_to_position":
        state = await arm.get_state()
        pose = target_pose_from_fields(state.pose, command.get("pose", {}))
        await arm.move_to_position(pose)
        return {"ok": True, "handled": cmd, "pose": pose.to_dict()}

    if cmd == "move_relative":
        state = await arm.get_state()
        target_pose = target_pose_from_relative(state.pose, command)
        await arm.move_to_position(target_pose)
        return {
            "ok": True,
            "handled": cmd,
            "direction": str(command.get("direction", "")),
            "distance_mm": float(command.get("distance_mm", 0.0)),
            "pose": target_pose.to_dict(),
        }

    if cmd == "move_axis_to":
        state = await arm.get_state()
        axis = str(command.get("axis", "")).lower()
        if axis not in {"x", "y", "z"}:
            raise ArmControlError(f"Unsupported pose axis: {axis}")
        target_pose = target_pose_from_fields(state.pose, {axis: float(command.get("value", 0.0))})
        await arm.move_to_position(target_pose)
        return {
            "ok": True,
            "handled": cmd,
            "axis": axis,
            "value": float(command.get("value", 0.0)),
            "pose": target_pose.to_dict(),
        }

    if cmd in {"unknown", "exit"}:
        return {"ok": False, "handled": cmd, "message": str(command.get("message", ""))}

    raise ArmControlError(f"Unsupported arm command: {cmd}")


async def handle_transcript(
    arm: Any,
    transcript: str,
    parser: Optional[RegexArmParser] = None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    parser = parser or create_arm_parser()
    command, speech = await parser.parse(transcript)
    result = await dispatch_arm_command(arm, command)
    return command, speech, result
