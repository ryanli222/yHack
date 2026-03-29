"""
LiDAR driver for LDROBOT STL-19P (same binary protocol as LD19).

47-byte packets over serial at 230400 baud, 12 measurement points each.
Accumulates packets into full 360-degree scans and exposes them as numpy
arrays of (angle_deg, distance_m, intensity).

Provides:
  - STL19Lidar: real serial driver
  - FakeLidar:  synthetic scans for testing
  - create_lidar(): factory
"""

from __future__ import annotations

import asyncio
import logging
import math
import struct
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from config import LidarConfig, load_config

logger = logging.getLogger(__name__)

# ======================================================================
# Protocol constants (LDROBOT LD19 / STL-19P)
# ======================================================================

HEADER = 0x54
VERLEN = 0x2C              # version 1, 12 points per packet
POINTS_PER_PACKET = 12
PACKET_SIZE = 47            # 1+1+2+2 + 12*3 + 2+2+1 = 47 bytes

# CRC8 lookup table — polynomial 0x4D
_CRC_TABLE = [
    0x00, 0x4D, 0x9A, 0xD7, 0x79, 0x34, 0xE3, 0xAE,
    0xF2, 0xBF, 0x68, 0x25, 0x8B, 0xC6, 0x11, 0x5C,
    0xA9, 0xE4, 0x33, 0x7E, 0xD0, 0x9D, 0x4A, 0x07,
    0x5B, 0x16, 0xC1, 0x8C, 0x22, 0x6F, 0xB8, 0xF5,
    0x1F, 0x52, 0x85, 0xC8, 0x66, 0x2B, 0xFC, 0xB1,
    0xED, 0xA0, 0x77, 0x3A, 0x94, 0xD9, 0x0E, 0x43,
    0xB6, 0xFB, 0x2C, 0x61, 0xCF, 0x82, 0x55, 0x18,
    0x44, 0x09, 0xDE, 0x93, 0x3D, 0x70, 0xA7, 0xEA,
    0x3E, 0x73, 0xA4, 0xE9, 0x47, 0x0A, 0xDD, 0x90,
    0xCC, 0x81, 0x56, 0x1B, 0xB5, 0xF8, 0x2F, 0x62,
    0x97, 0xDA, 0x0D, 0x40, 0xEE, 0xA3, 0x74, 0x39,
    0x65, 0x28, 0xFF, 0xB2, 0x1C, 0x51, 0x86, 0xCB,
    0x21, 0x6C, 0xBB, 0xF6, 0x58, 0x15, 0xC2, 0x8F,
    0xD3, 0x9E, 0x49, 0x04, 0xAA, 0xE7, 0x30, 0x7D,
    0x88, 0xC5, 0x12, 0x5F, 0xF1, 0xBC, 0x6B, 0x26,
    0x7A, 0x37, 0xE0, 0xAD, 0x03, 0x4E, 0x99, 0xD4,
    0x7C, 0x31, 0xE6, 0xAB, 0x05, 0x48, 0x9F, 0xD2,
    0x8E, 0xC3, 0x14, 0x59, 0xF7, 0xBA, 0x6D, 0x20,
    0xD5, 0x98, 0x4F, 0x02, 0xAC, 0xE1, 0x36, 0x7B,
    0x27, 0x6A, 0xBD, 0xF0, 0x5E, 0x13, 0xC4, 0x89,
    0x63, 0x2E, 0xF9, 0xB4, 0x1A, 0x57, 0x80, 0xCD,
    0x91, 0xDC, 0x0B, 0x46, 0xE8, 0xA5, 0x72, 0x3F,
    0xCA, 0x87, 0x50, 0x1D, 0xB3, 0xFE, 0x29, 0x64,
    0x38, 0x75, 0xA2, 0xEF, 0x41, 0x0C, 0xDB, 0x96,
    0x42, 0x0F, 0xD8, 0x95, 0x3B, 0x76, 0xA1, 0xEC,
    0xB0, 0xFD, 0x2A, 0x67, 0xC9, 0x84, 0x53, 0x1E,
    0xEB, 0xA6, 0x71, 0x3C, 0x92, 0xDF, 0x08, 0x45,
    0x19, 0x54, 0x83, 0xCE, 0x60, 0x2D, 0xFA, 0xB7,
    0x5D, 0x10, 0xC7, 0x8A, 0x24, 0x69, 0xBE, 0xF3,
    0xAF, 0xE2, 0x35, 0x78, 0xD6, 0x9B, 0x4C, 0x01,
]


def _crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc = _CRC_TABLE[(crc ^ b) & 0xFF]
    return crc


# ======================================================================
# Scan data type
# ======================================================================

@dataclass
class LidarScan:
    """One full 360-degree scan: list of (angle_deg, distance_m, intensity)."""
    points: List[Tuple[float, float, int]] = field(default_factory=list)
    timestamp: float = 0.0

    def as_numpy(self, min_range: float = 0.0, max_range: float = 100.0) -> np.ndarray:
        """Return Nx3 array (angle_deg, distance_m, intensity), range-filtered."""
        if not self.points:
            return np.empty((0, 3), dtype=np.float32)
        arr = np.array(self.points, dtype=np.float32)
        mask = (arr[:, 1] >= min_range) & (arr[:, 1] <= max_range)
        return arr[mask]

    def as_xy(self, min_range: float = 0.0, max_range: float = 100.0,
              angle_offset_deg: float = 0.0) -> np.ndarray:
        """Return Nx2 array of (x, y) in meters, robot-local frame."""
        arr = self.as_numpy(min_range, max_range)
        if arr.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float32)
        angles_rad = np.deg2rad(arr[:, 0] + angle_offset_deg)
        dists = arr[:, 1]
        x = dists * np.cos(angles_rad)
        y = dists * np.sin(angles_rad)
        return np.column_stack([x, y]).astype(np.float32)


# ======================================================================
# Real LiDAR driver
# ======================================================================

class STL19Lidar:
    """Reads the LDROBOT STL-19P (LD19-compatible) binary serial protocol."""

    def __init__(self, port: Optional[str] = None, baud: Optional[int] = None):
        cfg = load_config().lidar
        self._port = port or cfg.port
        self._baud = baud or cfg.baud
        self._angle_offset = cfg.angle_offset_deg

        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._running = False

        self._current_scan: List[Tuple[float, float, int]] = []
        self._last_angle: float = 0.0
        self._latest_scan: Optional[LidarScan] = None
        self._scan_count: int = 0

    async def start(self) -> None:
        import serial_asyncio
        self._reader, self._writer = await serial_asyncio.open_serial_connection(
            url=self._port, baudrate=self._baud,
        )
        self._running = True
        logger.info("STL-19P connected on %s @ %d baud", self._port, self._baud)

    async def run(self) -> None:
        """Main read loop: parse packets, assemble scans."""
        buf = bytearray()
        while self._running:
            try:
                chunk = await asyncio.wait_for(
                    self._reader.read(512), timeout=1.0,
                )
                if not chunk:
                    continue
                buf.extend(chunk)

                while len(buf) >= PACKET_SIZE:
                    # Scan for header
                    idx = buf.find(bytes([HEADER, VERLEN]))
                    if idx < 0:
                        buf = buf[-1:]   # keep last byte in case it's 0x54
                        break
                    if idx > 0:
                        buf = buf[idx:]  # discard bytes before header
                    if len(buf) < PACKET_SIZE:
                        break

                    packet = bytes(buf[:PACKET_SIZE])
                    buf = buf[PACKET_SIZE:]

                    # CRC check
                    if _crc8(packet[:-1]) != packet[-1]:
                        logger.debug("CRC fail, skipping packet")
                        continue

                    self._parse_packet(packet)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("LiDAR read error: %s", exc)
                await asyncio.sleep(0.1)

    def _parse_packet(self, packet: bytes) -> None:
        """Extract 12 measurement points from a 47-byte packet."""
        # Bytes 2-3: radar speed (deg/s), little-endian uint16
        # Bytes 4-5: start angle * 100, little-endian uint16
        start_angle = struct.unpack_from("<H", packet, 4)[0] / 100.0

        # Bytes 6-41: 12 points, each 3 bytes (distance_mm LE uint16 + intensity uint8)
        points = []
        for i in range(POINTS_PER_PACKET):
            offset = 6 + i * 3
            dist_mm = struct.unpack_from("<H", packet, offset)[0]
            intensity = packet[offset + 2]
            points.append((dist_mm, intensity))

        # Bytes 42-43: end angle * 100
        end_angle = struct.unpack_from("<H", packet, 42)[0] / 100.0

        # Interpolate angles for each of the 12 points
        angle_span = end_angle - start_angle
        if angle_span < 0:
            angle_span += 360.0

        for i, (dist_mm, intensity) in enumerate(points):
            angle = start_angle + angle_span * i / (POINTS_PER_PACKET - 1)
            angle = angle % 360.0
            dist_m = dist_mm / 1000.0

            self._current_scan.append((angle, dist_m, intensity))

        # Detect scan completion: angle wrapped around
        if end_angle < self._last_angle and self._last_angle > 300.0 and end_angle < 60.0:
            self._latest_scan = LidarScan(
                points=list(self._current_scan),
                timestamp=time.monotonic(),
            )
            self._current_scan.clear()
            self._scan_count += 1
            if self._scan_count % 50 == 0:
                logger.debug("Scan #%d: %d points", self._scan_count,
                             len(self._latest_scan.points))

        self._last_angle = end_angle

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_scan(self) -> Optional[LidarScan]:
        return self._latest_scan

    def get_points_xy(self) -> Optional[np.ndarray]:
        if self._latest_scan is None:
            return None
        cfg = load_config().lidar
        return self._latest_scan.as_xy(
            min_range=cfg.min_range_m,
            max_range=cfg.max_range_m,
            angle_offset_deg=self._angle_offset,
        )

    async def stop(self) -> None:
        self._running = False
        if self._writer is not None:
            self._writer.close()
        logger.info("STL-19P stopped (%d scans captured)", self._scan_count)


# ======================================================================
# Fake LiDAR for testing
# ======================================================================

class FakeLidar:
    """Generates synthetic scans with a few obstacles for testing."""

    def __init__(self):
        self._running = False
        self._latest_scan: Optional[LidarScan] = None
        self._t: float = 0.0

    async def start(self) -> None:
        self._running = True
        logger.info("[FAKE] LiDAR started")

    async def run(self) -> None:
        while self._running:
            self._generate_scan()
            await asyncio.sleep(0.1)  # 10 Hz

    def _generate_scan(self) -> None:
        self._t += 0.1
        points = []
        for i in range(360):
            angle = float(i)
            # Walls at 3m on all sides
            rad = math.radians(angle)
            cos_a = abs(math.cos(rad))
            sin_a = abs(math.sin(rad))
            wall_dist = 3.0 / max(cos_a, sin_a, 0.01)
            dist = min(wall_dist, 12.0)

            # Rotating obstacle
            obs_angle = (self._t * 30.0) % 360.0
            if abs(angle - obs_angle) < 10.0 or abs(angle - obs_angle + 360) < 10.0:
                dist = min(dist, 1.5)

            points.append((angle, dist, 200))

        self._latest_scan = LidarScan(points=points, timestamp=time.monotonic())

    def get_scan(self) -> Optional[LidarScan]:
        return self._latest_scan

    def get_points_xy(self) -> Optional[np.ndarray]:
        if self._latest_scan is None:
            return None
        cfg = load_config().lidar
        return self._latest_scan.as_xy(
            min_range=cfg.min_range_m, max_range=cfg.max_range_m,
        )

    async def stop(self) -> None:
        self._running = False
        logger.info("[FAKE] LiDAR stopped")


# ======================================================================
# Factory
# ======================================================================

def create_lidar(fake: bool = False, port: Optional[str] = None):
    if fake:
        return FakeLidar()
    try:
        import serial_asyncio  # noqa: F401
    except ImportError:
        logger.warning("pyserial-asyncio not installed — using FakeLidar")
        return FakeLidar()
    return STL19Lidar(port=port)
