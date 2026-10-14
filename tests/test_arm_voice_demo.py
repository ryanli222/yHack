"""
Focused safety checks for arm_voice_demo.py.

Run from HACKDAY root:
    python tests/test_arm_voice_demo.py
"""

import asyncio
import sys
from pathlib import Path

# Ensure laptop/ modules are importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "laptop"))

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} — {detail}")


print("\n=== arm_voice_demo.py ===")
from arm_control import ArmControlError, FakeArmControl
from arm_voice_flow import create_arm_parser, target_pose_from_relative
from arm_voice_demo import SafetyAbortError, _read_command_text, execute_guarded_arm_command
from voice import create_voice_interface


class CartesianIkFailingArm(FakeArmControl):
    def __init__(self, joint_count: int = 5):
        super().__init__(joint_count=joint_count)
        self.pose_attempts = 0

    async def move_to_position(self, pose, timeout_s=None):
        del pose, timeout_s
        self.pose_attempts += 1
        raise ArmControlError(
            "Failed to move arm to pose: (<Status.UNKNOWN: 2>, "
            "'zero IK solutions produced, goal positions appears to be physically unreachable', None)"
        )


class NonIkFailingArm(FakeArmControl):
    async def move_to_position(self, pose, timeout_s=None):
        del pose, timeout_s
        raise ArmControlError("Failed to move arm to pose: network timeout")


async def test_segmented_pose_motion():
    arm = FakeArmControl()
    voice = create_voice_interface(fake=True)
    await arm.start()
    await voice.start()

    # Four cartesian segments, with confirmation required between segments.
    for _ in range(3):
        voice.inject_transcript("yes continue")

    result = await execute_guarded_arm_command(
        arm,
        voice,
        {"cmd": "move_to_position", "pose": {"x": 100, "y": 0, "z": 200}},
        checkpoint_interval_s=0.0,
        confirm_timeout_s=0.5,
        max_cartesian_step_mm=50.0,
        max_joint_step=5.0,
        use_cli_prompt=False,
    )

    state = await arm.get_state()
    check("segmented pose motion succeeds", result["ok"], f"got {result}")
    check("pose motion uses four segments", result["segments"] == 4, f"got {result}")
    check("pose reaches x target", abs(state.pose.x - 100.0) < 1e-6, f"got {state.pose}")
    check("pose reaches z target", abs(state.pose.z - 200.0) < 1e-6, f"got {state.pose}")

    await arm.stop()
    await voice.stop()


asyncio.run(test_segmented_pose_motion())


async def test_safety_abort_without_confirmation():
    arm = FakeArmControl()
    voice = create_voice_interface(fake=True)
    await arm.start()
    await voice.start()

    try:
        await execute_guarded_arm_command(
            arm,
            voice,
            {"cmd": "move_joint", "joint_index": 0, "value": 25},
            checkpoint_interval_s=0.0,
            confirm_timeout_s=0.1,
            max_cartesian_step_mm=20.0,
            max_joint_step=10.0,
            use_cli_prompt=False,
        )
    except SafetyAbortError:
        aborted = True
    else:
        aborted = False

    state = await arm.get_state()
    check("motion aborts without confirmation", aborted, "command did not abort")
    check(
        "abort stops after first guarded segment",
        0.0 < state.joint_positions[0] <= 10.0,
        f"got {state.joint_positions}",
    )

    await arm.stop()
    await voice.stop()


asyncio.run(test_safety_abort_without_confirmation())


async def test_direct_gripper_command_still_works():
    arm = FakeArmControl()
    voice = create_voice_interface(fake=True)
    await arm.start()
    await voice.start()

    result = await execute_guarded_arm_command(
        arm,
        voice,
        {"cmd": "close_gripper"},
        checkpoint_interval_s=5.0,
        confirm_timeout_s=0.5,
        max_cartesian_step_mm=20.0,
        max_joint_step=10.0,
        use_cli_prompt=False,
    )

    state = await arm.get_state()
    check("non-motion command bypasses segmentation", result["handled"] == "close_gripper", f"got {result}")
    check("direct gripper command still closes", state.holding and not state.gripper_open, f"got {state}")

    await arm.stop()
    await voice.stop()


asyncio.run(test_direct_gripper_command_still_works())


async def test_relative_command_parsing():
    parser = create_arm_parser()
    command, speech = await parser.parse("move up 2 cm")
    check("relative parser command type", command["cmd"] == "move_relative", f"got {command}")
    check("relative parser preserves direction", command["direction"] == "up", f"got {command}")
    check("relative parser converts cm to mm", abs(command["distance_mm"] - 20.0) < 1e-6, f"got {command}")
    check("relative parser builds z delta", abs(command["delta"]["z"] - 20.0) < 1e-6, f"got {command}")
    check("relative parser speech mentions direction", "up" in speech.lower(), f"got {speech!r}")


asyncio.run(test_relative_command_parsing())


async def test_forward_command_uses_distance_only():
    parser = create_arm_parser()
    command, speech = await parser.parse("move the gripper forward 10")
    check("forward parser command type", command["cmd"] == "move_relative", f"got {command}")
    check("forward parser keeps direction", command["direction"] == "forward", f"got {command}")
    check("forward parser keeps units in mm", abs(command["distance_mm"] - 10.0) < 1e-6, f"got {command}")
    check("forward parser does not precompute world delta", "delta" not in command, f"got {command}")
    check("forward parser speech mentions gripper", "gripper" in speech.lower(), f"got {speech!r}")


asyncio.run(test_forward_command_uses_distance_only())


async def test_natural_language_gripper_relative_parsing():
    parser = create_arm_parser()
    command, speech = await parser.parse("move the gripper left 10")
    check("left parser command type", command["cmd"] == "move_relative", f"got {command}")
    check("left parser direction", command["direction"] == "left", f"got {command}")
    check("left parser default units are millimeters", abs(command["distance_mm"] - 10.0) < 1e-6, f"got {command}")
    check("left parser leaves runtime to resolve delta", "delta" not in command, f"got {command}")
    check("left parser speech mentions gripper", "gripper" in speech.lower(), f"got {speech!r}")


asyncio.run(test_natural_language_gripper_relative_parsing())


async def test_explicit_degree_rotation_parsing():
    parser = create_arm_parser()
    command, speech = await parser.parse("rotate left 10 degrees")
    check("rotate parser command type", command["cmd"] == "move_joint_relative", f"got {command}")
    check("rotate parser direction", command["direction"] == "left", f"got {command}")
    check("rotate parser uses base joint", command["joint_index"] == 0, f"got {command}")
    check("rotate parser delta degrees", abs(command["delta_degrees"] - 10.0) < 1e-6, f"got {command}")
    check("rotate parser speech mentions rotation", "rotating" in speech.lower() or "rotate" in speech.lower(), f"got {speech!r}")


asyncio.run(test_explicit_degree_rotation_parsing())


async def test_degree_qualified_move_parsing():
    parser = create_arm_parser()
    command, speech = await parser.parse("move right 5 degrees")
    check("degree move parser command type", command["cmd"] == "move_joint_relative", f"got {command}")
    check("degree move parser direction", command["direction"] == "right", f"got {command}")
    check("degree move parser delta degrees", abs(command["delta_degrees"] + 5.0) < 1e-6, f"got {command}")
    check("degree move parser speech mentions rotation", "rotating" in speech.lower() or "rotate" in speech.lower(), f"got {speech!r}")


asyncio.run(test_degree_qualified_move_parsing())


async def test_explicit_up_rotation_parsing():
    parser = create_arm_parser()
    command, speech = await parser.parse("rotate up 6 degrees")
    check("up rotate parser command type", command["cmd"] == "move_joint_relative", f"got {command}")
    check("up rotate parser direction", command["direction"] == "up", f"got {command}")
    check("up rotate parser uses shoulder joint", command["joint_index"] == 1, f"got {command}")
    check("up rotate parser delta degrees", abs(command["delta_degrees"] - 6.0) < 1e-6, f"got {command}")
    check("up rotate parser speech mentions rotation", "rotating" in speech.lower() or "rotate" in speech.lower(), f"got {speech!r}")


asyncio.run(test_explicit_up_rotation_parsing())


async def test_degree_qualified_down_move_parsing():
    parser = create_arm_parser()
    command, speech = await parser.parse("move down 4 degrees")
    check("down degree parser command type", command["cmd"] == "move_joint_relative", f"got {command}")
    check("down degree parser direction", command["direction"] == "down", f"got {command}")
    check("down degree parser uses shoulder joint", command["joint_index"] == 1, f"got {command}")
    check("down degree parser delta degrees", abs(command["delta_degrees"] + 4.0) < 1e-6, f"got {command}")
    check("down degree parser speech mentions rotation", "rotating" in speech.lower() or "rotate" in speech.lower(), f"got {speech!r}")


asyncio.run(test_degree_qualified_down_move_parsing())


async def test_axis_command_parsing():
    parser = create_arm_parser()
    command, speech = await parser.parse("move y to -10")
    check("axis parser command type", command["cmd"] == "move_axis_to", f"got {command}")
    check("axis parser chooses y", command["axis"] == "y", f"got {command}")
    check("axis parser keeps target value", abs(command["value"] + 10.0) < 1e-6, f"got {command}")
    check("axis parser speech mentions axis", "y" in speech.lower(), f"got {speech!r}")


asyncio.run(test_axis_command_parsing())


async def test_triplet_position_parsing():
    parser = create_arm_parser()
    command, speech = await parser.parse("move gripper to 274, -10, 222")
    check("triplet parser command type", command["cmd"] == "move_to_position", f"got {command}")
    check("triplet parser x value", abs(command["pose"]["x"] - 274.0) < 1e-6, f"got {command}")
    check("triplet parser y value", abs(command["pose"]["y"] + 10.0) < 1e-6, f"got {command}")
    check("triplet parser z value", abs(command["pose"]["z"] - 222.0) < 1e-6, f"got {command}")
    check("triplet parser speech mentions gripper", "gripper" in speech.lower(), f"got {speech!r}")


asyncio.run(test_triplet_position_parsing())


async def test_forward_relative_follows_tool_orientation():
    current_pose = {
        "x": 254.7,
        "y": -7.57,
        "z": 221.07,
        "o_x": 0.67,
        "o_y": -0.02,
        "o_z": -0.74,
        "theta": 5.41,
    }
    target_pose = target_pose_from_relative(
        current_pose,
        {"cmd": "move_relative", "direction": "forward", "distance_mm": 10.0},
    )
    check("forward local motion changes x", abs(target_pose.x - 261.410535) < 1e-3, f"got {target_pose}")
    check("forward local motion changes y", abs(target_pose.y + 7.770314) < 1e-3, f"got {target_pose}")
    check("forward local motion changes z", abs(target_pose.z - 213.658503) < 1e-3, f"got {target_pose}")
    check("forward local motion preserves theta", abs(target_pose.theta - 5.41) < 1e-6, f"got {target_pose}")


asyncio.run(test_forward_relative_follows_tool_orientation())


async def test_explicit_degree_rotation_segmented_motion():
    arm = FakeArmControl()
    voice = create_voice_interface(fake=True)
    await arm.start()
    await voice.start()

    voice.inject_transcript("yes continue")

    result = await execute_guarded_arm_command(
        arm,
        voice,
        {
            "cmd": "move_joint_relative",
            "direction": "left",
            "joint_index": 0,
            "delta_degrees": 15.0,
        },
        checkpoint_interval_s=0.0,
        confirm_timeout_s=0.5,
        max_cartesian_step_mm=20.0,
        max_joint_step=10.0,
        use_cli_prompt=False,
    )

    state = await arm.get_state()
    check("explicit degree rotation succeeds", result["ok"], f"got {result}")
    check("explicit degree rotation uses two segments", result["segments"] == 2, f"got {result}")
    check("explicit degree rotation updates base joint", abs(state.joint_positions[0] - 15.0) < 1e-6, f"got {state.joint_positions}")

    await arm.stop()
    await voice.stop()


asyncio.run(test_explicit_degree_rotation_segmented_motion())


async def test_explicit_up_rotation_segmented_motion():
    arm = FakeArmControl()
    voice = create_voice_interface(fake=True)
    await arm.start()
    await voice.start()

    voice.inject_transcript("yes continue")

    result = await execute_guarded_arm_command(
        arm,
        voice,
        {
            "cmd": "move_joint_relative",
            "direction": "up",
            "joint_index": 1,
            "delta_degrees": 12.0,
        },
        checkpoint_interval_s=0.0,
        confirm_timeout_s=0.5,
        max_cartesian_step_mm=20.0,
        max_joint_step=10.0,
        use_cli_prompt=False,
    )

    state = await arm.get_state()
    check("explicit up rotation succeeds", result["ok"], f"got {result}")
    check("explicit up rotation uses two segments", result["segments"] == 2, f"got {result}")
    check("explicit up rotation updates shoulder joint", abs(state.joint_positions[1] - 12.0) < 1e-6, f"got {state.joint_positions}")

    await arm.stop()
    await voice.stop()


asyncio.run(test_explicit_up_rotation_segmented_motion())


async def test_left_relative_pose_motion():
    arm = FakeArmControl()
    voice = create_voice_interface(fake=True)
    await arm.start()
    await voice.start()

    await arm.move_to_position(
        {
            "x": 254.7,
            "y": -7.57,
            "z": 221.07,
            "o_x": 0.67,
            "o_y": -0.02,
            "o_z": -0.74,
            "theta": 5.41,
        }
    )

    result = await execute_guarded_arm_command(
        arm,
        voice,
        {
            "cmd": "move_relative",
            "direction": "left",
            "distance_mm": 15.0,
        },
        checkpoint_interval_s=0.0,
        confirm_timeout_s=0.5,
        max_cartesian_step_mm=20.0,
        max_joint_step=10.0,
        use_cli_prompt=False,
    )

    state = await arm.get_state()
    check("left gripper motion succeeds", result["ok"], f"got {result}")
    check("left gripper motion uses one segment", result["segments"] == 1, f"got {result}")
    check("left gripper motion updates x along tool", abs(state.pose.x - 253.206742) < 1e-3, f"got {state.pose}")
    check("left gripper motion updates y along tool", abs(state.pose.y + 22.465260) < 1e-3, f"got {state.pose}")
    check("left gripper motion updates z along tool", abs(state.pose.z - 220.120570) < 1e-3, f"got {state.pose}")
    check("left gripper motion leaves joints alone", abs(state.joint_positions[0]) < 1e-6, f"got {state.joint_positions}")
    check("left gripper motion preserves orientation", abs(state.pose.o_x - 0.67) < 1e-6 and abs(state.pose.theta - 5.41) < 1e-6, f"got {state.pose}")

    await arm.stop()
    await voice.stop()


asyncio.run(test_left_relative_pose_motion())


async def test_segmented_relative_pose_motion():
    arm = FakeArmControl()
    voice = create_voice_interface(fake=True)
    await arm.start()
    await voice.start()

    voice.inject_transcript("yes continue")

    result = await execute_guarded_arm_command(
        arm,
        voice,
        {
            "cmd": "move_relative",
            "direction": "up",
            "distance_mm": 30.0,
            "delta": {"x": 0.0, "y": 0.0, "z": 30.0},
        },
        checkpoint_interval_s=0.0,
        confirm_timeout_s=0.5,
        max_cartesian_step_mm=20.0,
        max_joint_step=10.0,
        use_cli_prompt=False,
    )

    state = await arm.get_state()
    check("relative segmented motion succeeds", result["ok"], f"got {result}")
    check("relative segmented motion uses two segments", result["segments"] == 2, f"got {result}")
    check("relative motion updates z pose", abs(state.pose.z - 30.0) < 1e-6, f"got {state.pose}")
    check("relative motion keeps x pose", abs(state.pose.x) < 1e-6, f"got {state.pose}")

    await arm.stop()
    await voice.stop()


asyncio.run(test_segmented_relative_pose_motion())


async def test_forward_relative_pose_motion_uses_tool_vector():
    arm = FakeArmControl()
    voice = create_voice_interface(fake=True)
    await arm.start()
    await voice.start()

    await arm.move_to_position(
        {
            "x": 254.7,
            "y": -7.57,
            "z": 221.07,
            "o_x": 0.67,
            "o_y": -0.02,
            "o_z": -0.74,
            "theta": 5.41,
        }
    )

    result = await execute_guarded_arm_command(
        arm,
        voice,
        {"cmd": "move_relative", "direction": "forward", "distance_mm": 10.0},
        checkpoint_interval_s=0.0,
        confirm_timeout_s=0.5,
        max_cartesian_step_mm=20.0,
        max_joint_step=10.0,
        use_cli_prompt=False,
    )

    state = await arm.get_state()
    check("forward motion succeeds", result["ok"], f"got {result}")
    check("forward motion updates x along tool", abs(state.pose.x - 261.410535) < 1e-3, f"got {state.pose}")
    check("forward motion updates y along tool", abs(state.pose.y + 7.770314) < 1e-3, f"got {state.pose}")
    check("forward motion updates z along tool", abs(state.pose.z - 213.658503) < 1e-3, f"got {state.pose}")
    check("forward motion preserves orientation", abs(state.pose.o_x - 0.67) < 1e-6 and abs(state.pose.theta - 5.41) < 1e-6, f"got {state.pose}")

    await arm.stop()
    await voice.stop()


asyncio.run(test_forward_relative_pose_motion_uses_tool_vector())


async def test_axis_move_preserves_orientation():
    arm = FakeArmControl()
    voice = create_voice_interface(fake=True)
    await arm.start()
    await voice.start()

    await arm.move_to_position(
        {
            "x": 254.7,
            "y": -7.57,
            "z": 221.07,
            "o_x": 0.67,
            "o_y": -0.02,
            "o_z": -0.74,
            "theta": 5.41,
        }
    )

    result = await execute_guarded_arm_command(
        arm,
        voice,
        {"cmd": "move_axis_to", "axis": "y", "value": -10.0},
        checkpoint_interval_s=0.0,
        confirm_timeout_s=0.5,
        max_cartesian_step_mm=20.0,
        max_joint_step=10.0,
        use_cli_prompt=False,
    )

    state = await arm.get_state()
    check("axis move succeeds", result["ok"], f"got {result}")
    check("axis move updates only y", abs(state.pose.y + 10.0) < 1e-6, f"got {state.pose}")
    check("axis move preserves x", abs(state.pose.x - 254.7) < 1e-6, f"got {state.pose}")
    check("axis move preserves z", abs(state.pose.z - 221.07) < 1e-6, f"got {state.pose}")
    check("axis move preserves orientation", abs(state.pose.o_x - 0.67) < 1e-6 and abs(state.pose.theta - 5.41) < 1e-6, f"got {state.pose}")

    await arm.stop()
    await voice.stop()


asyncio.run(test_axis_move_preserves_orientation())


async def test_relative_move_uses_joint_fallback_after_ik_failure():
    arm = CartesianIkFailingArm()
    voice = create_voice_interface(fake=True)
    await arm.start()
    await voice.start()

    await arm.move_joints([0.0, 35.0, -25.0, -10.0, 0.0])
    arm._state.pose.x = 254.7
    arm._state.pose.y = -7.57
    arm._state.pose.z = 221.07
    arm._state.pose.o_x = 0.67
    arm._state.pose.o_y = -0.02
    arm._state.pose.o_z = -0.74
    arm._state.pose.theta = 5.41

    result = await execute_guarded_arm_command(
        arm,
        voice,
        {"cmd": "move_relative", "direction": "left", "distance_mm": 15.0},
        checkpoint_interval_s=0.0,
        confirm_timeout_s=0.5,
        max_cartesian_step_mm=20.0,
        max_joint_step=10.0,
        use_cli_prompt=False,
    )

    state = await arm.get_state()
    fallback = result.get("fallback", {})
    check("relative move retries after IK failure", arm.pose_attempts == 1, f"got {arm.pose_attempts}")
    check("relative move reports fallback usage", fallback.get("used") is True, f"got {result}")
    check(
        "relative move fallback changes joints",
        any(abs(value) > 1e-6 for value in state.joint_positions[:4]),
        f"got {state.joint_positions}",
    )
    check(
        "relative move fallback keeps original handler",
        result.get("handled") == "move_relative",
        f"got {result}",
    )

    await arm.stop()
    await voice.stop()


asyncio.run(test_relative_move_uses_joint_fallback_after_ik_failure())


async def test_relative_move_prefers_calibrated_ik_when_solvable():
    arm = CartesianIkFailingArm()
    voice = create_voice_interface(fake=True)
    await arm.start()
    await voice.start()

    await arm.move_joints([0.0, 20.0, -35.0, 15.0, 0.0])
    arm._state.pose.x = 170.0
    arm._state.pose.y = 0.0
    arm._state.pose.z = 20.0
    arm._state.pose.o_x = 0.0
    arm._state.pose.o_y = 0.0
    arm._state.pose.o_z = -1.0
    arm._state.pose.theta = 0.0
    voice.inject_transcript("yes continue")

    result = await execute_guarded_arm_command(
        arm,
        voice,
        {"cmd": "move_relative", "direction": "up", "distance_mm": 10.0},
        checkpoint_interval_s=0.0,
        confirm_timeout_s=0.5,
        max_cartesian_step_mm=20.0,
        max_joint_step=90.0,
        use_cli_prompt=False,
    )

    fallback = result.get("fallback", {})
    check("calibrated IK fallback is preferred when solvable", fallback.get("strategy") == "calibrated-planar-ik", f"got {result}")
    check("calibrated IK fallback records base yaw", "base_yaw_degrees" in fallback, f"got {result}")

    await arm.stop()
    await voice.stop()


asyncio.run(test_relative_move_prefers_calibrated_ik_when_solvable())


async def test_non_ik_cartesian_failure_still_raises():
    arm = NonIkFailingArm()
    voice = create_voice_interface(fake=True)
    await arm.start()
    await voice.start()

    try:
        await execute_guarded_arm_command(
            arm,
            voice,
            {"cmd": "move_relative", "direction": "left", "distance_mm": 15.0},
            checkpoint_interval_s=0.0,
            confirm_timeout_s=0.5,
            max_cartesian_step_mm=20.0,
            max_joint_step=10.0,
            use_cli_prompt=False,
        )
    except ArmControlError as exc:
        failed = "network timeout" in str(exc).lower()
    else:
        failed = False

    check("non-IK failure does not trigger fallback", failed, "unexpectedly swallowed non-IK move error")

    await arm.stop()
    await voice.stop()


asyncio.run(test_non_ik_cartesian_failure_still_raises())


async def test_typed_command_input():
    voice = create_voice_interface(fake=True)
    await voice.start()

    heard = await _read_command_text(
        voice,
        typed_input=True,
        input_reader=lambda prompt: "move joint 2 to 45",
    )
    check("typed command input returns entered text", heard == "move joint 2 to 45", f"got {heard!r}")

    await voice.stop()


asyncio.run(test_typed_command_input())


async def test_typed_command_eof_maps_to_exit():
    voice = create_voice_interface(fake=True)
    await voice.start()

    def _raise_eof(_prompt: str) -> str:
        raise EOFError

    heard = await _read_command_text(
        voice,
        typed_input=True,
        input_reader=_raise_eof,
    )
    check("typed command EOF exits cleanly", heard == "exit", f"got {heard!r}")

    await voice.stop()


asyncio.run(test_typed_command_eof_maps_to_exit())

print(f"\n{'='*50}")
print(f"  {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
print(f"{'='*50}")
sys.exit(1 if FAIL > 0 else 0)
