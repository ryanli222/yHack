"""
Focused transcript-to-arm checks for arm_voice_flow.py.

Run from HACKDAY root:
    python tests/test_arm_voice_flow.py
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


print("\n=== arm_voice_flow.py ===")
from arm_control import FakeArmControl
from arm_voice_flow import RegexArmParser, dispatch_arm_command, handle_transcript


async def test_voice_mapping():
    parser = RegexArmParser()
    arm = FakeArmControl()
    await arm.start()

    cmd, speech = await parser.parse("open the gripper")
    check("parse open_gripper", cmd["cmd"] == "open_gripper", f"got {cmd}")
    check("speech for open_gripper", "opening" in speech.lower(), f"got {speech}")

    await dispatch_arm_command(arm, {"cmd": "close_gripper"})
    state = await arm.get_state()
    check("dispatch close_gripper updates state", not state.gripper_open and state.holding, f"got {state}")

    await dispatch_arm_command(arm, cmd)
    state = await arm.get_state()
    check("dispatch open_gripper updates state", state.gripper_open and not state.holding, f"got {state}")

    cmd, _ = await parser.parse("move the arm home")
    check("parse home", cmd["cmd"] == "home", f"got {cmd}")
    await arm.move_joints([10, 20, 30, 40, 50])
    await dispatch_arm_command(arm, cmd)
    state = await arm.get_state()
    check("home resets joints through dispatcher", state.joint_positions == [0.0] * 5, f"got {state.joint_positions}")

    cmd, _ = await parser.parse("move joint 2 to 45 degrees")
    check("parse move_joint", cmd["cmd"] == "move_joint", f"got {cmd}")
    check("joint index is zero-based", cmd["joint_index"] == 1, f"got {cmd}")
    await dispatch_arm_command(arm, cmd)
    state = await arm.get_state()
    check("move_joint updates single joint", state.joint_positions[1] == 45.0, f"got {state.joint_positions}")

    cmd, _ = await parser.parse("set joints to 1 2 3 4 5")
    check("parse move_joints", cmd["cmd"] == "move_joints", f"got {cmd}")
    await dispatch_arm_command(arm, cmd)
    state = await arm.get_state()
    check("move_joints updates all joints", state.joint_positions == [1.0, 2.0, 3.0, 4.0, 5.0], f"got {state.joint_positions}")

    cmd, _ = await parser.parse("move arm to x 100 y 5 z 200")
    check("parse move_to_position", cmd["cmd"] == "move_to_position", f"got {cmd}")
    await dispatch_arm_command(arm, cmd)
    state = await arm.get_state()
    check("move_to_position updates pose x", state.pose.x == 100.0, f"got {state.pose}")
    check("move_to_position updates pose z", state.pose.z == 200.0, f"got {state.pose}")

    cmd, speech = await parser.parse("sing me a song")
    check("unknown command is safe", cmd["cmd"] == "unknown", f"got {cmd}")
    check("unknown speech returned", "didn't understand" in speech.lower(), f"got {speech}")

    await arm.close_gripper()
    before = await arm.get_state()
    handled_cmd, handled_speech, result = await handle_transcript(arm, "open the gripper", parser=parser)
    after = await arm.get_state()
    check("handle_transcript returns parsed command", handled_cmd["cmd"] == "open_gripper", f"got {handled_cmd}")
    check("handle_transcript returns speech", "opening" in handled_speech.lower(), f"got {handled_speech}")
    check("handle_transcript returns result", result["handled"] == "open_gripper", f"got {result}")
    check("handle_transcript changes state", before.gripper_open != after.gripper_open, f"before={before} after={after}")

    await arm.stop()


asyncio.run(test_voice_mapping())

print(f"\n{'='*50}")
print(f"  {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
print(f"{'='*50}")
sys.exit(1 if FAIL > 0 else 0)
