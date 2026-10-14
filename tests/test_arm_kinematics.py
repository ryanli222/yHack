"""
Focused checks for calibrated arm kinematics.

Run from HACKDAY root:
    python tests/test_arm_kinematics.py
"""

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


print("\n=== arm_kinematics.py ===")
from arm_control import ArmControlError
from arm_kinematics import ArmKinematicsCalibration, calibrated_joint_targets_for_pose


def test_calibrated_solver_returns_joint_targets():
    calibration = ArmKinematicsCalibration(
        enabled=True,
        origin_offset_mm=(0.0, 0.0, 0.0),
        link_lengths_mm=(120.0, 123.0, 80.0),
        joint_multipliers=(1.0, 1.0, 1.0, 1.0),
        joint_offsets_deg=(0.0, 0.0, 0.0, 0.0),
        default_tool_pitch_deg=-90.0,
    )
    target_joints, details = calibrated_joint_targets_for_pose(
        {
            "x": 170.0,
            "y": 0.0,
            "z": 20.0,
            "o_x": 0.0,
            "o_y": 0.0,
            "o_z": -1.0,
            "theta": 0.0,
        },
        {
            "x": 165.0,
            "y": 0.0,
            "z": 20.0,
            "o_x": 0.0,
            "o_y": 0.0,
            "o_z": -1.0,
            "theta": 0.0,
        },
        [0.0, 20.0, -35.0, 15.0, 0.0],
        calibration=calibration,
    )
    check("solver returns five joint targets", len(target_joints) == 5, f"got {target_joints}")
    check("solver base yaw stays near zero", abs(details["base_yaw_degrees"]) < 1e-6, f"got {details}")
    check(
        "solver produces finite targets",
        all(abs(value) < 360.0 for value in target_joints[:4]),
        f"got {target_joints}",
    )


test_calibrated_solver_returns_joint_targets()


def test_calibrated_solver_respects_joint_mapping():
    calibration = ArmKinematicsCalibration(
        enabled=True,
        origin_offset_mm=(0.0, 0.0, 0.0),
        link_lengths_mm=(120.0, 123.0, 80.0),
        joint_multipliers=(1.0, -1.5, 2.0, 0.5),
        joint_offsets_deg=(10.0, 20.0, 30.0, 40.0),
        default_tool_pitch_deg=-90.0,
    )
    target_joints, details = calibrated_joint_targets_for_pose(
        {
            "x": 170.0,
            "y": 0.0,
            "z": 20.0,
            "o_x": 0.0,
            "o_y": 0.0,
            "o_z": -1.0,
            "theta": 0.0,
        },
        {
            "x": 165.0,
            "y": 0.0,
            "z": 20.0,
            "o_x": 0.0,
            "o_y": 0.0,
            "o_z": -1.0,
            "theta": 0.0,
        },
        [10.0, 20.0, 30.0, 40.0, 0.0],
        calibration=calibration,
    )
    logical = calibration.logical_from_viam(target_joints)
    check("joint mapping preserves solved base", abs(logical[0] - details["base_yaw_degrees"]) < 1e-6, f"got {logical}, {details}")
    check(
        "joint mapping preserves solved shoulder/elbow/wrist",
        abs(logical[1] - details["shoulder_degrees"]) < 1e-6
        and abs(logical[2] - details["elbow_degrees"]) < 1e-6
        and abs(logical[3] - details["wrist_degrees"]) < 1e-6,
        f"got {logical}, {details}",
    )


test_calibrated_solver_respects_joint_mapping()


def test_calibrated_solver_rejects_unreachable_pose():
    calibration = ArmKinematicsCalibration(
        enabled=True,
        origin_offset_mm=(0.0, 0.0, 0.0),
        link_lengths_mm=(120.0, 123.0, 80.0),
        joint_multipliers=(1.0, 1.0, 1.0, 1.0),
        joint_offsets_deg=(0.0, 0.0, 0.0, 0.0),
        default_tool_pitch_deg=-90.0,
    )
    try:
        calibrated_joint_targets_for_pose(
            {
                "x": 450.0,
                "y": 0.0,
                "z": 250.0,
                "o_x": 0.0,
                "o_y": 0.0,
                "o_z": -1.0,
                "theta": 0.0,
            },
            {
                "x": 165.0,
                "y": 0.0,
                "z": 20.0,
                "o_x": 0.0,
                "o_y": 0.0,
                "o_z": -1.0,
                "theta": 0.0,
            },
            [0.0, 20.0, -35.0, 15.0, 0.0],
            calibration=calibration,
        )
    except ArmControlError:
        unreachable = True
    else:
        unreachable = False
    check("solver rejects unreachable pose", unreachable, "pose unexpectedly solved")


test_calibrated_solver_rejects_unreachable_pose()


print(f"\n{'='*50}")
print(f"  {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
print(f"{'='*50}")
sys.exit(1 if FAIL > 0 else 0)
