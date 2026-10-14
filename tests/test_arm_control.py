"""
Focused checks for arm_control.py.

Run from HACKDAY root:
    python tests/test_arm_control.py
"""

import asyncio
import builtins
import sys
from pathlib import Path
from unittest.mock import patch

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


print("\n=== arm_control.py ===")
from config import load_config
from arm_control import ArmControl, ArmControlError, ArmPose, FakeArmControl, create_arm_control

cfg = load_config()
check("config has arm block", hasattr(cfg, "arm"))
check("default arm name is set", bool(cfg.arm.arm_name), f"got {cfg.arm.arm_name!r}")
check("default gripper name is set", bool(cfg.arm.gripper_name), f"got {cfg.arm.gripper_name!r}")

check("factory returns fake arm", isinstance(create_arm_control(fake=True), FakeArmControl))


async def test_fake_arm():
    arm = FakeArmControl()
    initial = await arm.get_state()
    check("fake arm starts disconnected", not initial.connected)
    check("fake arm has default joints", len(initial.joint_positions) == 5, f"got {initial.joint_positions}")

    await arm.start()
    started = await arm.get_state()
    check("fake arm connects on start", started.connected)

    joints = await arm.move_joints([10, 20, 30, 40, 50])
    moved = await arm.get_state()
    check("move_joints returns values", joints == [10.0, 20.0, 30.0, 40.0, 50.0], f"got {joints}")
    check("joint state updated", moved.joint_positions == [10.0, 20.0, 30.0, 40.0, 50.0], f"got {moved.joint_positions}")

    pose = await arm.move_to_position({"x": 100, "y": 5, "z": 200, "o_z": 1.0, "theta": 0.0})
    moved_pose = await arm.get_state()
    check("move_to_position returns ArmPose", isinstance(pose, ArmPose))
    check("pose x updated", moved_pose.pose.x == 100.0, f"got {moved_pose.pose}")
    check("pose z updated", moved_pose.pose.z == 200.0, f"got {moved_pose.pose}")

    await arm.close_gripper()
    closed = await arm.get_state()
    check("close_gripper closes gripper", not closed.gripper_open)
    check("close_gripper sets holding", closed.holding)

    await arm.open_gripper()
    opened = await arm.get_state()
    check("open_gripper opens gripper", opened.gripper_open)
    check("open_gripper clears holding", not opened.holding)

    await arm.home()
    homed = await arm.get_state()
    check("home resets joints", homed.joint_positions == [0.0] * 5, f"got {homed.joint_positions}")
    check("home resets pose", homed.pose == ArmPose(), f"got {homed.pose}")

    await arm.stop()
    stopped = await arm.get_state()
    check("fake arm disconnects on stop", not stopped.connected)


asyncio.run(test_fake_arm())


def test_import_and_missing_sdk():
    module = __import__("arm_control")
    check("arm_control imports without viam-sdk", module is not None)

    arm = ArmControl()
    arm._address = "fake-machine"
    arm._api_key = "fake-key"
    arm._api_key_id = "fake-key-id"
    arm._arm_name = "arm"
    arm._gripper_name = "gripper"

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("viam"):
            raise ImportError("No module named 'viam'")
        return original_import(name, globals, locals, fromlist, level)

    async def _run():
        with patch("builtins.__import__", side_effect=fake_import):
            try:
                await arm.connect()
            except ArmControlError as exc:
                message = str(exc).lower()
                check("connect raises clean ArmControlError without sdk", "viam-sdk" in message, f"got {exc}")
                check("last_error stored on state", "viam-sdk" in arm._state.last_error.lower(), f"got {arm._state.last_error}")
            else:
                check("connect raises clean ArmControlError without sdk", False, "did not raise")

    asyncio.run(_run())


test_import_and_missing_sdk()

print(f"\n{'='*50}")
print(f"  {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
print(f"{'='*50}")
sys.exit(1 if FAIL > 0 else 0)
