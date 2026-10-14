"""
Smoke test for the perception stack — runs everything with fake/synthetic data.
No hardware required. Verifies imports, data flow, and basic math.

Run from HACKDAY root:
    python tests/test_perception_stack.py
"""

import asyncio
import sys
import math
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


# ==================================================================
# 1. Config
# ==================================================================
print("\n=== config.py ===")
from config import load_config, Config

cfg = load_config()
check("load_config returns Config", isinstance(cfg, Config))
check("LiDAR baud = 230400", cfg.lidar.baud == 230400)
check("encoder CPR set", cfg.serial.encoder_ticks_per_rev > 0,
      f"got {cfg.serial.encoder_ticks_per_rev}")
check("EKF process noise > 0", cfg.ekf.process_noise_xy > 0)
check("mecanum geometry > 0", cfg.mecanum.wheel_radius_m > 0)


# ==================================================================
# 2. LiDAR (Fake)
# ==================================================================
print("\n=== lidar.py (FakeLidar) ===")
from lidar import FakeLidar, LidarScan

async def test_lidar():
    lidar = FakeLidar()
    await lidar.start()

    # No scan yet
    check("no scan before run", lidar.get_scan() is None)

    # Run one cycle
    lidar._generate_scan()
    scan = lidar.get_scan()
    check("scan generated", scan is not None)
    check("scan is LidarScan", isinstance(scan, LidarScan))
    check("scan has points", len(scan.points) > 0, f"got {len(scan.points)}")

    # as_numpy
    arr = scan.as_numpy(min_range=0.02, max_range=12.0)
    check("as_numpy shape", arr.ndim == 2 and arr.shape[1] == 3,
          f"got {arr.shape}")

    # as_xy
    xy = scan.as_xy(min_range=0.02, max_range=12.0)
    check("as_xy shape", xy.ndim == 2 and xy.shape[1] == 2,
          f"got {xy.shape}")
    check("as_xy has points", xy.shape[0] > 0)

    await lidar.stop()

asyncio.run(test_lidar())


# ==================================================================
# 3. Perception (Fake)
# ==================================================================
print("\n=== perception.py (FakePerception) ===")
from perception import FakePerception

async def test_perception():
    perc = FakePerception()
    await perc.start()

    # Run a few cycles
    for _ in range(5):
        perc._t += 0.066
        perc._obstacles = [
            __import__("perception").Obstacle(x=2.0, y=0.5, radius=0.2, confidence=0.9)
        ]

    obs = perc.get_obstacles()
    check("get_obstacles returns list", isinstance(obs, list))
    check("obstacle count > 0", len(obs) > 0)
    check("obstacle has x,y", hasattr(obs[0], "x") and hasattr(obs[0], "y"))

    lm = perc.get_landmarks()
    check("get_landmarks returns list", isinstance(lm, list))

    await perc.stop()

asyncio.run(test_perception())


# ==================================================================
# 4. Phone IMU (Fake)
# ==================================================================
print("\n=== phone_imu.py (FakePhoneIMU) ===")
from phone_imu import FakePhoneIMU, IMUData

async def test_imu():
    imu = FakePhoneIMU(gyro_z=0.1)
    await imu.start()

    check("is_receiving", imu.is_receiving())

    gz = imu.get_gyro_z()
    check("gyro_z ~ 0.1", abs(gz - 0.1) < 0.05, f"got {gz}")

    data = imu.get_latest()
    check("returns IMUData", isinstance(data, IMUData))
    check("accel_z ~ 9.81", abs(data.accel_z - 9.81) < 0.1)

    await imu.stop()

asyncio.run(test_imu())


# ==================================================================
# 5. Pose Estimator (EKF)
# ==================================================================
print("\n=== pose_estimator.py (EKF) ===")
from pose_estimator import PoseEstimator

def test_ekf():
    ekf = PoseEstimator()

    # Start at origin
    x, y, th = ekf.get_pose()
    check("initial pose at origin", abs(x) < 0.01 and abs(y) < 0.01 and abs(th) < 0.01)

    # Drive forward: all 4 wheels positive ticks = forward (vx > 0)
    # With encoder_ticks_per_rev=1440, wheel_radius=0.05m:
    #   100 ticks in 0.05s → (100/1440)*2pi/0.05 = ~8.73 rad/s per wheel
    #   vx = (r/4)*(w1+w2+w3+w4) = (0.05/4)*4*8.73 = 0.436 m/s
    #   dx = 0.436 * 0.05 = 0.0218 m per step
    dt = 0.05
    for _ in range(20):
        ekf.predict((100, 100, 100, 100), dt)

    x, y, th = ekf.get_pose()
    check("forward motion: x > 0", x > 0.1, f"x={x:.4f}")
    check("forward motion: y ~ 0", abs(y) < 0.05, f"y={y:.4f}")
    check("forward motion: theta ~ 0", abs(th) < 0.1, f"th={th:.4f}")

    # Reset and test rotation: FL/RL negative, FR/RR positive = rotate CCW
    ekf.reset()
    for _ in range(20):
        ekf.predict((-50, 50, -50, 50), dt)

    x, y, th = ekf.get_pose()
    check("rotation: theta != 0", abs(th) > 0.05, f"th={th:.4f}")

    # Reset and test strafe: FL/RR negative, FR/RL positive = strafe left (vy > 0)
    ekf.reset()
    for _ in range(20):
        ekf.predict((-50, 50, 50, -50), dt)

    x, y, th = ekf.get_pose()
    check("strafe: y != 0", abs(y) > 0.05, f"y={y:.4f}")

    # Test scan match update
    ekf.reset()
    ekf.predict((100, 100, 100, 100), dt)
    ekf.update_scan_match(1.0, 0.5, 0.1)
    x, y, th = ekf.get_pose()
    check("scan match correction applied", x > 0.3, f"x={x:.4f}")

    # Covariance should be finite
    P = ekf.get_covariance()
    check("covariance finite", all(math.isfinite(P[i, i]) for i in range(3)))

test_ekf()


# ==================================================================
# 6. SLAM (Occupancy Grid + Scan Matcher)
# ==================================================================
print("\n=== slam.py (OccupancyGrid + ScanMatcher) ===")
from slam import SLAM
import numpy as np

def test_slam():
    slam = SLAM()

    # Create a synthetic scan: walls forming a box at ~3m
    angles = np.linspace(0, 2 * math.pi, 360, endpoint=False)
    points = []
    for a in angles:
        cos_a = abs(math.cos(a))
        sin_a = abs(math.sin(a))
        dist = 3.0 / max(cos_a, sin_a, 0.01)
        dist = min(dist, 12.0)
        points.append([dist * math.cos(a), dist * math.sin(a)])
    scan_xy = np.array(points, dtype=np.float32)

    check("scan_xy shape", scan_xy.shape == (360, 2))

    # Process a few scans from origin
    for _ in range(3):
        slam.process_scan(scan_xy, (0.0, 0.0, 0.0))

    occ = slam.get_occupancy_array()
    check("occupancy grid shape", occ.shape[0] > 0 and occ.shape[1] > 0,
          f"got {occ.shape}")

    # Cells near walls should be occupied (>0.5), center should be free (<0.5)
    gs = slam.occupancy_grid.grid_size
    center = gs // 2
    center_val = occ[center, center]
    check("center is free", center_val < 0.4, f"got {center_val:.3f}")

    # Cell at ~3m (60 cells at 0.05m/cell) from center should be occupied
    wall_idx = center + 60
    if wall_idx < gs:
        wall_val = occ[center, wall_idx]
        check("wall cell is occupied", wall_val > 0.5, f"got {wall_val:.3f}")

    # Test scan matcher (needs >10 scans, runs every 5th)
    result = None
    for i in range(20):
        r = slam.process_scan(scan_xy, (0.0, 0.0, 0.0))
        if r is not None:
            result = r

    # Scan match at origin should return close to origin
    if result is not None:
        rx, ry, rt = result
        check("scan match near origin", abs(rx) < 0.5 and abs(ry) < 0.5,
              f"got ({rx:.2f}, {ry:.2f})")
    else:
        check("scan match returned result", False, "returned None")

test_slam()


# ==================================================================
# 7. Integration: LiDAR → SLAM → Pose Estimator
# ==================================================================
print("\n=== Integration: LiDAR -> SLAM -> EKF ===")

async def test_integration():
    lidar = FakeLidar()
    await lidar.start()
    lidar._generate_scan()

    scan = lidar.get_scan()
    scan_xy = scan.as_xy(min_range=0.02, max_range=12.0)
    check("integration: got scan_xy", scan_xy.shape[0] > 0)

    slam = SLAM()
    ekf = PoseEstimator()

    # Simulate 20 ticks of driving forward
    dt = 0.05
    for i in range(20):
        # Encoder predict
        ekf.predict((80, 80, 80, 80), dt)

        # LiDAR scan
        lidar._generate_scan()
        scan = lidar.get_scan()
        scan_xy = scan.as_xy(min_range=0.02, max_range=12.0)

        # SLAM process
        pose = ekf.get_pose()
        corrected = slam.process_scan(scan_xy, pose)
        if corrected is not None:
            ekf.update_scan_match(*corrected)

    final = ekf.get_pose()
    check("integration: pose moved", abs(final[0]) > 0.1 or abs(final[1]) > 0.1,
          f"pose={final}")

    occ = slam.get_occupancy_array()
    check("integration: grid updated", occ.max() > 0.5)

    await lidar.stop()

asyncio.run(test_integration())


# ==================================================================
# Summary
# ==================================================================
print(f"\n{'='*50}")
print(f"  {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
print(f"{'='*50}")
sys.exit(1 if FAIL > 0 else 0)
