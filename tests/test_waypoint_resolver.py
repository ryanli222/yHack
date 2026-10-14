"""Waypoint fast path: cached hit skips the LLM, miss falls back to it."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "laptop"))

from waypoint_resolver import WaypointResolver  # noqa: E402

WAYPOINTS = {"door": [3.0, 0.0], "front door": [5.0, 1.0]}


class SpyLLM:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    async def parse(self, text):
        self.calls.append(text)
        if self.fail:
            raise ConnectionError("api down")
        return {"cmd": "move", "direction": "forward", "distance": 2.0}, "Moving."


def test_cached_waypoint_hit_skips_llm():
    llm = SpyLLM()
    cmd, _ = asyncio.run(WaypointResolver(WAYPOINTS, llm).parse("take me to the front door"))
    assert cmd == {"cmd": "goto", "x": 5.0, "y": 1.0}   # longest name wins
    assert llm.calls == []


def test_yolo_label_anchored_to_landmark_cell():
    llm = SpyLLM()
    resolver = WaypointResolver(
        WAYPOINTS, llm,
        detections=lambda: [SimpleNamespace(label="chair")],
        lookup=lambda name: (1.23, 0.47) if name == "chair" else None,
        resolution_m=0.05,
    )
    cmd, _ = asyncio.run(resolver.parse("go to the chair"))
    assert cmd["cmd"] == "goto"
    assert (round(cmd["x"], 2), round(cmd["y"], 2)) == (1.25, 0.45)   # snapped to 5 cm grid
    assert llm.calls == []


def test_miss_falls_back_to_llm_then_regex():
    llm = SpyLLM()
    cmd, _ = asyncio.run(WaypointResolver(WAYPOINTS, llm).parse("move forward 2 meters"))
    assert cmd["cmd"] == "move" and llm.calls == ["move forward 2 meters"]

    down = SpyLLM(fail=True)
    cmd, _ = asyncio.run(WaypointResolver(WAYPOINTS, down).parse("go forward 2 meters"))
    assert cmd == {"cmd": "move", "direction": "forward", "distance": 2.0}   # regex fallback
