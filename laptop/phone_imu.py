"""
Phone IMU listener for HyperIMU app (Android).

Streams gyro/accel over UDP as CSV. Phone must be mounted flat (screen up)
so gyro Z-axis = yaw rate.

HyperIMU CSV format:
  timestamp_ms, gx_dps, gy_dps, gz_dps, ax_ms2, ay_ms2, az_ms2
  (gyro in deg/s, accel in m/s^2)

Provides:
  - PhoneIMU: real UDP listener
  - FakePhoneIMU: synthetic data
  - create_phone_imu(): factory
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

from config import load_config

logger = logging.getLogger(__name__)


@dataclass
class IMUData:
    timestamp_s: float
    gyro_x: float     # rad/s
    gyro_y: float     # rad/s
    gyro_z: float     # rad/s — yaw rate, the one we care about
    accel_x: float    # m/s^2
    accel_y: float
    accel_z: float


class PhoneIMU:
    """Listens for HyperIMU UDP packets."""

    def __init__(self, port: Optional[int] = None):
        cfg = load_config().phone_imu
        self._port = port or cfg.udp_port
        self._latest: Optional[IMUData] = None
        self._transport = None
        self._packet_count = 0

    def get_latest(self) -> Optional[IMUData]:
        return self._latest

    def get_gyro_z(self) -> float:
        if self._latest is None:
            return 0.0
        return self._latest.gyro_z

    def is_receiving(self) -> bool:
        if self._latest is None:
            return False
        return (time.monotonic() - self._latest.timestamp_s) < 2.0

    async def start(self) -> None:
        loop = asyncio.get_event_loop()
        self._transport, _ = await loop.create_datagram_endpoint(
            lambda: _IMUProtocol(self._on_packet),
            local_addr=("0.0.0.0", self._port),
        )
        logger.info("PhoneIMU listening on UDP :%d", self._port)

    async def stop(self) -> None:
        if self._transport:
            self._transport.close()
        logger.info("PhoneIMU stopped (%d packets)", self._packet_count)

    def _on_packet(self, data: bytes, addr: tuple) -> None:
        try:
            text = data.decode("utf-8", errors="replace").strip()
            imu = self._parse(text)
            if imu:
                self._latest = imu
                self._packet_count += 1
        except Exception as exc:
            logger.debug("IMU parse error: %s", exc)

    @staticmethod
    def _parse(text: str) -> Optional[IMUData]:
        parts = text.split(",")
        if len(parts) < 7:
            return None
        try:
            gx = float(parts[1])
            gy = float(parts[2])
            gz = float(parts[3])
            ax = float(parts[4])
            ay = float(parts[5])
            az = float(parts[6])
        except ValueError:
            return None

        d2r = math.pi / 180.0
        return IMUData(
            timestamp_s=time.monotonic(),
            gyro_x=gx * d2r, gyro_y=gy * d2r, gyro_z=gz * d2r,
            accel_x=ax, accel_y=ay, accel_z=az,
        )


class _IMUProtocol(asyncio.DatagramProtocol):
    def __init__(self, callback):
        self._cb = callback

    def datagram_received(self, data, addr):
        self._cb(data, addr)


class FakePhoneIMU:
    """Returns zero gyro + 9.81 accel_z for testing."""

    def __init__(self, gyro_z: float = 0.0):
        self._gz = gyro_z

    def get_latest(self) -> IMUData:
        return IMUData(time.monotonic(), 0, 0, self._gz, 0, 0, 9.81)

    def get_gyro_z(self) -> float:
        return self._gz

    def is_receiving(self) -> bool:
        return True

    async def start(self) -> None:
        logger.info("[FAKE] PhoneIMU started")

    async def stop(self) -> None:
        pass


def create_phone_imu(fake: bool = False, port: Optional[int] = None):
    if fake:
        return FakePhoneIMU()
    return PhoneIMU(port=port)
