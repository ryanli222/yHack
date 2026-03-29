"""
Serial bridge: laptop <-> Arduino Mega over USB.

Sends: {"vx": f, "vy": f, "w": f}\n
Receives: {"estop": 0|1, "enc": [dFL, dFR, dRL, dRR]}\n

Baud 115200, line-delimited JSON. Watchdog: no cmd for 500ms -> motors stop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import List, Optional, Tuple

from config import load_config

logger = logging.getLogger(__name__)


class SerialBridge:
    """Async serial connection to Arduino Mega."""

    def __init__(self):
        cfg = load_config().serial
        self._port = cfg.port
        self._baud = cfg.baud
        self._watchdog_ms = cfg.watchdog_timeout_ms
        self._ser = None
        self._reader = None
        self._writer = None
        self._last_send = 0.0
        self._estop = False
        self._encoder_ticks: List[int] = [0, 0, 0, 0]

    async def start(self) -> None:
        import serial_asyncio
        self._reader, self._writer = await serial_asyncio.open_serial_connection(
            url=self._port, baudrate=self._baud,
        )
        logger.info("Serial connected: %s @ %d", self._port, self._baud)

    async def send_velocity(self, vx: float, vy: float, omega: float) -> None:
        msg = json.dumps({"vx": round(vx, 3), "vy": round(vy, 3), "w": round(omega, 3)})
        self._writer.write((msg + "\n").encode())
        await self._writer.drain()
        self._last_send = time.monotonic()

    async def read_response(self) -> Optional[dict]:
        try:
            line = await asyncio.wait_for(self._reader.readline(), timeout=0.1)
            data = json.loads(line.decode().strip())
            self._estop = bool(data.get("estop", 0))
            self._encoder_ticks = data.get("enc", [0, 0, 0, 0])
            return data
        except (asyncio.TimeoutError, json.JSONDecodeError, UnicodeDecodeError):
            return None

    def get_estop(self) -> bool:
        return self._estop

    def get_encoder_ticks(self) -> List[int]:
        return list(self._encoder_ticks)

    async def stop(self) -> None:
        if self._writer:
            await self.send_velocity(0, 0, 0)
            self._writer.close()
        logger.info("Serial closed")


class FakeSerialBridge:
    """No-hardware stub. Simulates encoder ticks from velocity."""

    def __init__(self):
        self._estop = False
        self._ticks = [0, 0, 0, 0]
        self._vx = 0.0
        self._vy = 0.0
        self._omega = 0.0

    async def start(self) -> None:
        logger.info("[FAKE] Serial bridge started")

    async def send_velocity(self, vx: float, vy: float, omega: float) -> None:
        self._vx, self._vy, self._omega = vx, vy, omega
        # Simulate encoder ticks from mecanum IK (arbitrary scale)
        scale = 100.0
        self._ticks = [
            int((vx - vy - omega) * scale),
            int((vx + vy + omega) * scale),
            int((vx + vy - omega) * scale),
            int((vx - vy + omega) * scale),
        ]

    async def read_response(self) -> Optional[dict]:
        return {"estop": int(self._estop), "enc": self._ticks}

    def get_estop(self) -> bool:
        return self._estop

    def get_encoder_ticks(self) -> List[int]:
        return list(self._ticks)

    def set_estop(self, val: bool) -> None:
        self._estop = val

    async def stop(self) -> None:
        logger.info("[FAKE] Serial bridge stopped")


def create_serial_bridge(fake: bool = False):
    if fake:
        return FakeSerialBridge()
    return SerialBridge()
