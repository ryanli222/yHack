"""
Transcript-to-arm command parsing and dispatch.

This keeps the first voice-to-arm path deterministic and easy to test.
It mirrors the shape of llm_interface.py by returning a command dict plus
short spoken feedback, but uses regex parsing for the initial milestone.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Sequence, Tuple

from arm_control import ArmControlError, ArmPose


def _extract_numbers(text: str) -> list[float]:
    return [float(match) for match in re.findall(r"-?\d+(?:\.\d+)?", text)]


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
            r"(?:move|set)\s+(?:the\s+)?(?:arm\s+)?(?:to|pose)\s+"
            r"x\s*=?\s*(-?\d+(?:\.\d+)?)\s*[, ]+\s*"
            r"y\s*=?\s*(-?\d+(?:\.\d+)?)\s*[, ]+\s*"
            r"z\s*=?\s*(-?\d+(?:\.\d+)?)",
            text_lower,
        )
        if match:
            pose = {
                "x": float(match.group(1)),
                "y": float(match.group(2)),
                "z": float(match.group(3)),
                "o_x": 0.0,
                "o_y": 0.0,
                "o_z": 1.0,
                "theta": 0.0,
            }
            return {"cmd": "move_to_position", "pose": pose}, "Moving the arm."

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

    if cmd == "move_joints":
        joints = [float(value) for value in command.get("joints", [])]
        if not joints:
            raise ArmControlError("move_joints command requires joint values")
        await arm.move_joints(joints)
        return {"ok": True, "handled": cmd, "joint_count": len(joints)}

    if cmd == "move_to_position":
        pose = ArmPose.from_value(command.get("pose", {}))
        await arm.move_to_position(pose)
        return {"ok": True, "handled": cmd, "pose": pose.to_dict()}

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
