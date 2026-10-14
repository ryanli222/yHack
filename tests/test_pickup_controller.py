"""
Focused checks for viam_vision.py and pickup_controller.py.

Run from HACKDAY root:
    python tests/test_pickup_controller.py
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


print("\n=== viam_vision.py + pickup_controller.py ===")
from arm_control import FakeArmControl
from config import PickupConfig, load_config
from pickup_controller import PickupController, estimate_pickup_target
from viam_vision import (
    BallDetection,
    FakeViamVision,
    ViamVision,
    ViamVisionError,
    select_detection,
)

cfg = load_config()
check("config has pickup block", hasattr(cfg, "pickup"))

ball_detection = BallDetection(
    label="ball",
    confidence=0.92,
    x_min=260.0,
    y_min=180.0,
    x_max=380.0,
    y_max=300.0,
    image_width=640,
    image_height=480,
)
other_detection = BallDetection(
    label="chair",
    confidence=0.99,
    x_min=10.0,
    y_min=20.0,
    x_max=80.0,
    y_max=100.0,
    image_width=640,
    image_height=480,
)

selected = select_detection(
    [other_detection, ball_detection],
    target_label="ball",
    minimum_confidence=0.50,
)
check("select_detection respects target label", selected == ball_detection, f"got {selected}")

pickup_cfg = PickupConfig(
    enabled=True,
    use_fake_vision=True,
    target_label="ball",
    minimum_confidence=0.50,
    calibration_distance_m=0.45,
    calibration_bbox_width_px=120.0,
    camera_horizontal_fov_deg=62.0,
    center_tolerance_px=40.0,
    pickup_zone_min_range_m=0.20,
    pickup_zone_max_range_m=0.65,
    pickup_zone_max_lateral_m=0.12,
    arm_forward_offset_m=-0.02,
    arm_lateral_offset_m=0.0,
    arm_vertical_offset_m=0.0,
    approach_standoff_m=0.04,
    pregrasp_height_m=0.14,
    grasp_height_m=0.05,
    lift_height_m=0.18,
    max_retries=1,
    home_after_pickup=False,
)

target = estimate_pickup_target(ball_detection, pickup_cfg)
check("estimate returns target", target is not None)
check("estimated range uses calibration", abs(target["range_m"] - 0.45) < 0.02, f"got {target['range_m']:.3f}")
check("centered target flagged ready", target["centered"], f"got {target}")
check("target inside pickup zone", target["in_pickup_zone"], f"got {target}")
check("pregrasp is behind grasp", target["poses"]["pregrasp"]["x"] < target["poses"]["grasp"]["x"])

off_center_detection = BallDetection(
    label="ball",
    confidence=0.91,
    x_min=420.0,
    y_min=170.0,
    x_max=540.0,
    y_max=290.0,
    image_width=640,
    image_height=480,
)
off_center_target = estimate_pickup_target(off_center_detection, pickup_cfg)
check("off-center target computed", off_center_target is not None)
check("off-center detection not centered", not off_center_target["centered"], f"got {off_center_target}")


async def test_fake_pickup_success():
    arm = FakeArmControl()
    vision = FakeViamVision([ball_detection])
    controller = PickupController(arm=arm, vision=vision, config=pickup_cfg)

    await controller.start()
    state = await controller.refresh_detection(include_image=True)
    check("refresh_detection reports ready", state["status"] == "ready", f"got {state}")

    state = await controller.pickup_ball()
    arm_state = await arm.get_state()
    check("pickup_ball succeeds with fake arm", state["status"] == "success", f"got {state}")
    check("pickup closes fake gripper", arm_state.holding, f"got {arm_state}")
    check("pickup tracks attempts", state["attempts"] == 1, f"got {state['attempts']}")

    await controller.stop()


asyncio.run(test_fake_pickup_success())


async def test_off_center_warning():
    arm = FakeArmControl()
    vision = FakeViamVision([off_center_detection])
    controller = PickupController(arm=arm, vision=vision, config=pickup_cfg)

    state = await controller.pickup_ball()
    check("off-center pickup returns warning", state["status"] == "warning", f"got {state}")
    check("off-center pickup does not report success", not state["last_result"]["ok"], f"got {state}")

    await controller.stop()


asyncio.run(test_off_center_warning())


def test_missing_viam_sdk():
    vision = ViamVision()
    vision._address = "fake-machine"
    vision._api_key = "fake-key"
    vision._api_key_id = "fake-key-id"
    vision._camera_name = "pickup-cam"
    vision._vision_service_name = "ball-detector"

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("viam"):
            raise ImportError("No module named 'viam'")
        return original_import(name, globals, locals, fromlist, level)

    async def _run():
        with patch("builtins.__import__", side_effect=fake_import):
            try:
                await vision.connect()
            except ViamVisionError as exc:
                message = str(exc).lower()
                check("missing SDK raises clean ViamVisionError", "viam-sdk" in message, f"got {exc}")
            else:
                check("missing SDK raises clean ViamVisionError", False, "did not raise")

    asyncio.run(_run())


test_missing_viam_sdk()

print(f"\n{'='*50}")
print(f"  {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
print(f"{'='*50}")
sys.exit(1 if FAIL > 0 else 0)
