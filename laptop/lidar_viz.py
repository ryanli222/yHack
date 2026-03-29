"""
Quick LiDAR visualizer — shows live scan in an OpenCV window.
Use this to find which angle is the front of your sensor.

Usage:
    python laptop/lidar_viz.py
    python laptop/lidar_viz.py --port COM5
    python laptop/lidar_viz.py --fake          # synthetic scan, no hardware
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import cv2
except ImportError:
    print("OpenCV required: pip install opencv-python")
    sys.exit(1)

from lidar import create_lidar

SIZE = 600          # window pixels
MAX_DIST_M = 5.0    # metres shown at edge of window
CENTER = SIZE // 2


def _polar_to_xy(angle_deg: float, dist_m: float) -> tuple[int, int]:
    """Convert a scan point to pixel coords. 0° = top (forward)."""
    angle_rad = math.radians(angle_deg)
    scale = (SIZE // 2 - 20) / MAX_DIST_M
    px = int(CENTER + dist_m * scale * math.sin(angle_rad))
    py = int(CENTER - dist_m * scale * math.cos(angle_rad))
    return px, py


def _draw_frame(scan: np.ndarray) -> np.ndarray:
    canvas = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)

    # Grid rings
    scale = (SIZE // 2 - 20) / MAX_DIST_M
    for r_m in [1, 2, 3, 4, 5]:
        r_px = int(r_m * scale)
        cv2.circle(canvas, (CENTER, CENTER), r_px, (40, 40, 40), 1)
        cv2.putText(canvas, f"{r_m}m", (CENTER + r_px + 3, CENTER),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 60, 60), 1)

    # Crosshairs
    cv2.line(canvas, (CENTER, 20), (CENTER, SIZE - 20), (40, 40, 40), 1)
    cv2.line(canvas, (20, CENTER), (SIZE - 20, CENTER), (40, 40, 40), 1)

    # Forward arrow (0°)
    cv2.arrowedLine(canvas, (CENTER, CENTER), (CENTER, CENTER - 80),
                    (0, 200, 0), 2, tipLength=0.3)
    cv2.putText(canvas, "0deg (forward?)", (CENTER + 8, CENTER - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)

    # Scan points
    if scan is not None and len(scan):
        angles = scan[:, 0]
        dists  = scan[:, 1]
        valid  = (dists > 0.05) & (dists < MAX_DIST_M)
        for angle, dist in zip(angles[valid], dists[valid]):
            px, py = _polar_to_xy(angle, dist)
            cv2.circle(canvas, (px, py), 2, (0, 255, 255), -1)

    # Robot dot
    cv2.circle(canvas, (CENTER, CENTER), 6, (255, 100, 100), -1)

    cv2.putText(canvas, "q = quit", (8, SIZE - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    return canvas


async def _run(port: str, fake: bool) -> None:
    lidar = create_lidar(fake=fake, port=port)
    await lidar.start()
    asyncio.create_task(lidar.run())
    cv2.namedWindow("LiDAR viz", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("LiDAR viz", SIZE, SIZE)

    last_scan = None
    try:
        while True:
            scan = lidar.get_scan()
            if scan is not None:
                arr = scan.as_numpy()
                if len(arr):
                    last_scan = arr
            frame = _draw_frame(last_scan)
            cv2.imshow("LiDAR viz", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            await asyncio.sleep(0.05)
    finally:
        await lidar.stop()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Live LiDAR visualizer")
    parser.add_argument("--port", default=None, help="Serial port (default: from config)")
    parser.add_argument("--fake", action="store_true", help="Use synthetic scan")
    args = parser.parse_args()

    if args.port is None:
        from config import load_config
        args.port = load_config().lidar.port

    asyncio.run(_run(args.port, args.fake))


if __name__ == "__main__":
    main()
