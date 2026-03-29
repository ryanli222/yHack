"""
Standalone voice-to-arm demo.

By default this connects to VIAM resources named by ArmConfig, which lets you
drive the fake arm/gripper configured in the VIAM UI. Use --local-fake to
exercise the local FakeArmControl instead.

Examples:
    python laptop/arm_voice_demo.py
    python laptop/arm_voice_demo.py --local-fake
    python laptop/arm_voice_demo.py --typed-input --local-fake
    # then type: move left 5
    # or: move y to -10
    # or: rotate left 5 degrees
    # or: rotate up 5 degrees
    python laptop/arm_voice_demo.py --fake-voice --transcript "open the gripper" --once
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

try:
    import msvcrt  # Windows non-blocking key check

    def _kbhit_q() -> bool:
        """Return True if 'q' was pressed (non-blocking, Windows)."""
        if msvcrt.kbhit():
            key = msvcrt.getch().decode("utf-8", errors="ignore").lower()
            return key == "q"
        return False
except ImportError:
    def _kbhit_q() -> bool:
        """Non-Windows stub — no non-blocking key check available."""
        return False

# Ensure laptop/ is importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from arm_control import ArmControlError, ArmPose, create_arm_control
from arm_voice_flow import (
    create_arm_parser,
    dispatch_arm_command,
    target_pose_from_fields,
    target_pose_from_relative,
)
from config import load_config
from voice import create_voice_interface

logger = logging.getLogger(__name__)
DEFAULT_CONFIRM_DELAY_S = 3.0
DEFAULT_CHECKPOINT_INTERVAL_S = 5.0
DEFAULT_CONFIRM_TIMEOUT_S = 3.0
DEFAULT_MAX_CARTESIAN_STEP_MM = 20.0
DEFAULT_MAX_JOINT_STEP = 10.0


def _console_safe(value: object) -> str:
    text = str(value)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="backslashreplace").decode(encoding)


def _requires_confirmation(command: Mapping[str, Any]) -> bool:
    return str(command.get("cmd", "")).lower() not in {"exit", "stop", "unknown"}


def _is_segmented_motion(command: Mapping[str, Any]) -> bool:
    return str(command.get("cmd", "")).lower() in {
        "move_joint",
        "move_joint_relative",
        "move_joints",
        "move_to_position",
        "move_axis_to",
        "move_relative",
        "home",
    }


def _describe_command(command: Mapping[str, Any]) -> str:
    cmd = str(command.get("cmd", "unknown")).lower()
    if cmd == "open_gripper":
        return "open the gripper"
    if cmd == "close_gripper":
        return "close the gripper"
    if cmd == "home":
        return "move the arm home"
    if cmd == "move_joint":
        return f"move joint {int(command.get('joint_index', 0)) + 1} to {float(command.get('value', 0.0)):.1f}"
    if cmd == "move_joint_relative":
        direction = str(command.get("direction", "")).lower()
        if direction in {"left", "right", "up", "down"}:
            return f"rotate {direction} by {abs(float(command.get('delta_degrees', 0.0))):.1f} degrees"
        return (
            f"move joint {int(command.get('joint_index', 0)) + 1} "
            f"by {float(command.get('delta_degrees', 0.0)):.1f} degrees"
        )
    if cmd == "move_joints":
        return f"move joints to {list(command.get('joints', []))}"
    if cmd == "move_to_position":
        pose = dict(command.get("pose", {}))
        return (
            f"move gripper to x={float(pose.get('x', 0.0)):.1f}, "
            f"y={float(pose.get('y', 0.0)):.1f}, z={float(pose.get('z', 0.0)):.1f}"
        )
    if cmd == "move_axis_to":
        return f"move gripper {str(command.get('axis', '?')).lower()} to {float(command.get('value', 0.0)):.1f}"
    if cmd == "move_relative":
        return (
            f"move gripper {str(command.get('direction', 'relative')).lower()} "
            f"by {abs(float(command.get('distance_mm', 0.0))):.1f} mm"
        )
    if cmd == "stop":
        return "emergency stop"
    if cmd == "exit":
        return "exit"
    return ascii(dict(command))


class SafetyAbortError(RuntimeError):
    """Raised when an operator safety checkpoint is not confirmed in time."""


async def _pause_with_estop(duration_s: float) -> bool:
    """Sleep for *duration_s*, polling for 'q' keypress. Returns True if e-stop triggered."""
    elapsed = 0.0
    step = 0.05
    while elapsed < duration_s:
        if _kbhit_q():
            return True
        await asyncio.sleep(step)
        elapsed += step
    return False


def _approval_from_text(text: str) -> bool:
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    if tokens & {"stop", "halt", "abort", "cancel", "no", "shutdown", "quit"}:
        return False
    if tokens & {"yes", "y", "continue", "go", "okay", "ok", "correct"}:
        return True
    return False


def _build_joint_segments(current: list[float], target: list[float], *, max_joint_step: float) -> list[list[float]]:
    if not target:
        return []
    current_values = list(current)
    if len(current_values) < len(target):
        current_values.extend([0.0] * (len(target) - len(current_values)))

    max_delta = max(abs(float(goal) - float(start)) for start, goal in zip(current_values, target))
    steps = max(1, int(math.ceil(max_delta / max(max_joint_step, 1e-6))))
    segments = []
    for step_idx in range(1, steps + 1):
        alpha = step_idx / steps
        segments.append(
            [
                float(start) + (float(goal) - float(start)) * alpha
                for start, goal in zip(current_values, target)
            ]
        )
    return segments


def _build_pose_segments(current: ArmPose, target: ArmPose, *, max_cartesian_step_mm: float) -> list[ArmPose]:
    deltas = [
        abs(float(target.x) - float(current.x)),
        abs(float(target.y) - float(current.y)),
        abs(float(target.z) - float(current.z)),
    ]
    max_delta = max(deltas) if deltas else 0.0
    steps = max(1, int(math.ceil(max_delta / max(max_cartesian_step_mm, 1e-6))))
    segments = []
    for step_idx in range(1, steps + 1):
        alpha = step_idx / steps
        segments.append(
            ArmPose(
                x=float(current.x) + (float(target.x) - float(current.x)) * alpha,
                y=float(current.y) + (float(target.y) - float(current.y)) * alpha,
                z=float(current.z) + (float(target.z) - float(current.z)) * alpha,
                o_x=float(current.o_x) + (float(target.o_x) - float(current.o_x)) * alpha,
                o_y=float(current.o_y) + (float(target.o_y) - float(current.o_y)) * alpha,
                o_z=float(current.o_z) + (float(target.o_z) - float(current.o_z)) * alpha,
                theta=float(current.theta) + (float(target.theta) - float(current.theta)) * alpha,
            )
        )
    return segments


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _wrap_degrees(value: float) -> float:
    return ((float(value) + 180.0) % 360.0) - 180.0


async def _wait_for_confirmation(command: Mapping[str, Any], *, confirm_prompt: bool, delay_s: float) -> bool:
    description = _describe_command(command)
    if confirm_prompt:
        try:
            answer = await asyncio.to_thread(
                input,
                _console_safe(f"Execute '{description}'? [y/N]: "),
            )
        except EOFError:
            print("No CLI input available for confirmation. Cancelling command.")
            return False
        if answer.strip().lower() not in {"y", "yes"}:
            print("Cancelled.")
            return False
    if delay_s > 0:
        print(_console_safe(f"Executing '{description}' in {delay_s:.1f} seconds. Press Ctrl+C to abort."))
        await asyncio.sleep(delay_s)
    return True


async def _checkpoint_confirmation(
    voice: Any,
    command: Mapping[str, Any],
    *,
    segment_idx: int,
    total_segments: int,
    confirm_timeout_s: float,
    use_cli_prompt: bool,
) -> bool:
    description = _describe_command(command)
    message = (
        f"Safety checkpoint for {description}. "
        f"Segment {segment_idx} of {total_segments}. "
        f"Confirm within {confirm_timeout_s:.1f} seconds or I will stop."
    )
    spoken_prompt = f"Safety check. Say yes within {confirm_timeout_s:.0f} seconds to continue."
    print(_console_safe(message))

    if use_cli_prompt:
        try:
            answer = await asyncio.wait_for(
                asyncio.to_thread(
                    input,
                    _console_safe("Continue arm motion? [y/N]: "),
                ),
                timeout=max(0.1, confirm_timeout_s),
            )
        except (EOFError, asyncio.TimeoutError):
            print("No confirmation received. Stopping the arm test.")
            return False
        return answer.strip().lower() in {"y", "yes"}

    if hasattr(voice, "confirm"):
        heard = await voice.confirm(spoken_prompt, confirm_timeout_s)
        print(_console_safe(f"Confirmation heard: {heard or '<none>'}"))
        return _approval_from_text(heard)

    await asyncio.sleep(max(0.1, confirm_timeout_s))
    return False


async def _read_command_text(
    voice: Any,
    *,
    typed_input: bool,
    input_reader: Callable[[str], str] = input,
) -> str:
    if not typed_input:
        return await voice.listen()

    try:
        return (await asyncio.to_thread(input_reader, _console_safe("arm> "))).strip()
    except EOFError:
        print("Input closed. Exiting arm test.")
        return "exit"


async def _deliver_response(voice: Any, text: str, *, typed_input: bool) -> None:
    if not text:
        return
    if typed_input:
        print(_console_safe(f"Response: {text}"))
        return
    await voice.speak(text)


async def _dispatch_with_estop(arm: Any, command: Mapping[str, Any]) -> dict[str, Any]:
    """Dispatch an arm command while polling for 'q' e-stop in parallel."""
    task = asyncio.create_task(dispatch_arm_command(arm, command))
    try:
        while not task.done():
            if _kbhit_q():
                task.cancel()
                print("E-stop! Pressing q detected — stopping arm.")
                await arm.emergency_stop()
                raise SafetyAbortError("Emergency stop requested (q pressed).")
            await asyncio.sleep(0.05)
        return task.result()
    except asyncio.CancelledError:
        await arm.emergency_stop()
        raise SafetyAbortError("Emergency stop requested (q pressed).")


async def execute_guarded_arm_command(
    arm: Any,
    voice: Any,
    command: Mapping[str, Any],
    *,
    checkpoint_interval_s: float,
    confirm_timeout_s: float,
    max_cartesian_step_mm: float,
    max_joint_step: float,
    use_cli_prompt: bool,
) -> dict[str, Any]:
    """Execute an arm command with 'q' e-stop monitoring. VIAM handles smooth motion."""
    return await _dispatch_with_estop(arm, command)


async def run_demo(
    *,
    local_fake: bool = False,
    fake_voice: bool = False,
    typed_input: bool = False,
    transcript: str = "",
    once: bool = False,
    confirm_prompt: bool = False,
    delay_s: float = 0.0,
    checkpoint_interval_s: float = DEFAULT_CHECKPOINT_INTERVAL_S,
    confirm_timeout_s: float = DEFAULT_CONFIRM_TIMEOUT_S,
    max_cartesian_step_mm: float = DEFAULT_MAX_CARTESIAN_STEP_MM,
    max_joint_step: float = DEFAULT_MAX_JOINT_STEP,
) -> None:
    arm = create_arm_control(fake=local_fake)
    voice = create_voice_interface(fake=(fake_voice or typed_input))
    parser = create_arm_parser()

    await voice.start()
    try:
        await arm.start()
    except ArmControlError:
        await voice.stop()
        raise

    if fake_voice and not typed_input and transcript and hasattr(voice, "inject_transcript"):
        voice.inject_transcript(transcript)

    if typed_input:
        print("Typed arm demo ready.")
        print("Type commands like: 'open the gripper', 'close the gripper', 'move left 5',")
        print("'move up 10', 'move y to -10', 'move gripper to x 255 y -8 z 221', or 'move gripper to 255, -8, 221'.")
        print("For degree rotation, use explicit phrases like 'rotate left 5 degrees', 'turn right 10 degrees', or 'rotate up 5 degrees'.")
        print("Forward/back/left/right move in the current gripper frame. Up/down are vertical pose nudges.")
        print("Distances use millimeters unless you add a unit.")
        print("Press 'q' during motion to emergency stop. Type 'exit' to quit.")
    else:
        print("Voice arm demo ready.")
        print("Say commands like: 'open the gripper', 'close the gripper', 'move left 5',")
        print("'move up 10', 'move y to -10', or 'move gripper to 255, -8, 221'.")
        print("For degree rotation, say explicit phrases like 'rotate left 5 degrees', 'turn right 10 degrees', or 'rotate up 5 degrees'.")
        print("Forward/back/left/right move in the current gripper frame. Up/down are vertical pose nudges.")
        print("Distances use millimeters unless you add a unit.")
        print("Say 'exit' to stop.")
    if confirm_prompt or delay_s > 0:
        print(
            _console_safe(
                f"CLI safety gate enabled: confirm={'on' if confirm_prompt else 'off'}, "
                f"delay={delay_s:.1f}s."
            )
        )
    print(
        _console_safe(
            "Segmented motion safety enabled: "
            "confirmation after every segment, "
            f"timeout {confirm_timeout_s:.1f}s, "
            f"max cartesian step {max_cartesian_step_mm:.1f}mm, "
            f"max joint step {max_joint_step:.1f}."
        )
    )
    use_cli_checkpoints = typed_input or confirm_prompt

    try:
        while True:
            heard = await _read_command_text(voice, typed_input=typed_input)
            if not heard:
                if typed_input:
                    print("No command entered.")
                else:
                    await _deliver_response(voice, "I didn't hear a command.", typed_input=typed_input)
                if once:
                    break
                continue

            print(_console_safe(f"Heard: {heard}"))
            command, speech = await parser.parse(heard)
            print(_console_safe(f"Command: {ascii(command)}"))

            if _requires_confirmation(command):
                approved = await _wait_for_confirmation(
                    command,
                    confirm_prompt=confirm_prompt,
                    delay_s=delay_s,
                )
                if not approved:
                    result = {"ok": False, "handled": str(command.get("cmd", "unknown")), "cancelled": True}
                    speech = "Cancelled."
                    print(_console_safe(f"Result: {ascii(result)}"))
                    await _deliver_response(voice, speech, typed_input=typed_input)
                    if once:
                        break
                    continue

            try:
                result = await execute_guarded_arm_command(
                    arm,
                    voice,
                    command,
                    checkpoint_interval_s=checkpoint_interval_s,
                    confirm_timeout_s=confirm_timeout_s,
                    max_cartesian_step_mm=max_cartesian_step_mm,
                    max_joint_step=max_joint_step,
                    use_cli_prompt=use_cli_checkpoints,
                )
            except SafetyAbortError as exc:
                logger.warning("Arm motion stopped by safety checkpoint: %s", exc)
                await _deliver_response(voice, "Safety checkpoint missed. Shutting down arm test.", typed_input=typed_input)
                break
            except ArmControlError as exc:
                logger.error("Arm command failed: %s", exc)
                await _deliver_response(voice, "Arm command failed.", typed_input=typed_input)
                if once:
                    break
                continue

            print(_console_safe(f"Result: {ascii(result)}"))
            await _deliver_response(voice, speech, typed_input=typed_input)

            if command.get("cmd") == "exit":
                break
            if once:
                break
    finally:
        await arm.stop()
        await voice.stop()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone voice or typed arm demo")
    parser.add_argument(
        "--local-fake",
        action="store_true",
        help="Use the local FakeArmControl instead of connecting to VIAM resources",
    )
    parser.add_argument(
        "--typed-input",
        action="store_true",
        help="Read commands from the terminal instead of the microphone",
    )
    parser.add_argument(
        "--fake-voice",
        action="store_true",
        help="Use the FakeVoiceInterface instead of the real microphone/speakers",
    )
    parser.add_argument(
        "--transcript",
        default="",
        help="Inject a transcript when using --fake-voice",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process a single transcript, then exit",
    )
    parser.add_argument(
        "--confirm",
        dest="confirm",
        action="store_true",
        help="Require CLI confirmation before executing arm motion commands",
    )
    parser.add_argument(
        "--no-confirm",
        dest="confirm",
        action="store_false",
        help="Skip CLI confirmation before executing arm motion commands",
    )
    parser.set_defaults(confirm=None)
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=None,
        help="Wait this many seconds before executing confirmed arm motion commands",
    )
    parser.add_argument(
        "--checkpoint-seconds",
        type=float,
        default=DEFAULT_CHECKPOINT_INTERVAL_S,
        help="Retained for compatibility; segmented moves now ask for confirmation after every segment",
    )
    parser.add_argument(
        "--confirmation-timeout-seconds",
        type=float,
        default=DEFAULT_CONFIRM_TIMEOUT_S,
        help="Wait this many seconds for an operator confirmation before stopping the arm test",
    )
    parser.add_argument(
        "--max-cartesian-step-mm",
        type=float,
        default=DEFAULT_MAX_CARTESIAN_STEP_MM,
        help="Maximum Cartesian segment size in millimeters for guarded move_to_position commands",
    )
    parser.add_argument(
        "--max-joint-step",
        type=float,
        default=DEFAULT_MAX_JOINT_STEP,
        help="Maximum per-joint segment size for guarded joint motion commands",
    )
    return parser


async def _main() -> int:
    args = _build_arg_parser().parse_args()
    if args.delay_seconds is not None and args.delay_seconds < 0:
        print("--delay-seconds must be non-negative.")
        return 2
    if args.checkpoint_seconds < 0:
        print("--checkpoint-seconds must be non-negative.")
        return 2
    if args.confirmation_timeout_seconds <= 0:
        print("--confirmation-timeout-seconds must be positive.")
        return 2
    if args.max_cartesian_step_mm <= 0 or args.max_joint_step <= 0:
        print("--max-cartesian-step-mm and --max-joint-step must be positive.")
        return 2
    default_confirm = sys.stdin.isatty() and not args.local_fake and not args.fake_voice and not args.typed_input
    confirm_prompt = default_confirm if args.confirm is None else bool(args.confirm)
    delay_s = args.delay_seconds
    if delay_s is None:
        delay_s = DEFAULT_CONFIRM_DELAY_S if confirm_prompt else 0.0
    try:
        await run_demo(
            local_fake=args.local_fake,
            fake_voice=args.fake_voice,
            typed_input=args.typed_input,
            transcript=args.transcript,
            once=args.once,
            confirm_prompt=confirm_prompt,
            delay_s=delay_s,
            checkpoint_interval_s=args.checkpoint_seconds,
            confirm_timeout_s=args.confirmation_timeout_seconds,
            max_cartesian_step_mm=args.max_cartesian_step_mm,
            max_joint_step=args.max_joint_step,
        )
        return 0
    except ArmControlError as exc:
        print(f"Arm setup error: {exc}")
        print("Expected env vars for VIAM mode: VIAM_MACHINE_ADDRESS, VIAM_API_KEY, VIAM_API_KEY_ID,")
        print("VIAM_ARM_NAME, and VIAM_GRIPPER_NAME.")
        return 1
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(asyncio.run(_main()))
