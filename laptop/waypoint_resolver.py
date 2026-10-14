"""
Cached-waypoint resolver: spoken text -> goto target without an LLM round trip.

Resolution order (first hit wins):
  1. Known waypoint names (waypoints/*.json + learned landmarks)       -> goto
  2. YOLOv8 labels currently in view that exist in the landmark map    -> goto,
     snapped to the map grid (5 cm by default)
  3. LLM (Anthropic / OpenAI)                                          -> whatever it says
  4. Regex parser if the LLM call raises (API down, no network)

Per-command resolution latency is logged so the fast path can be measured,
not assumed.

Usage:
  llm = WaypointResolver(waypoints, create_llm_interface(waypoints=waypoints),
                         detections=perception.get_yolo_detections,
                         lookup=spatial.lookup_landmark)
  cmd, speech = await llm.parse("take me to the door")
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from llm_interface import RegexLLM

logger = logging.getLogger(__name__)

_STOP_WORDS = re.compile(r"\b(stop|halt|wait|freeze|pause)\b")


class WaypointResolver:
    """Drop-in replacement for an LLM interface: same ``parse(text)`` contract."""

    def __init__(
        self,
        waypoints: Dict[str, Sequence[float]],
        llm: Any,
        *,
        detections: Optional[Callable[[], Iterable[Any]]] = None,
        lookup: Optional[Callable[[str], Optional[Tuple[float, float]]]] = None,
        resolution_m: float = 0.05,
    ):
        self._waypoints = {k.lower(): (float(v[0]), float(v[1])) for k, v in waypoints.items()}
        self._llm = llm
        self._regex = RegexLLM(dict(waypoints))
        self._detections = detections or (lambda: [])
        self._lookup = lookup or (lambda _name: None)
        self._res = resolution_m

    # ------------------------------------------------------------------

    def resolve(self, text: str) -> Optional[Tuple[str, float, float]]:
        """Return (name, x, y) if the text names a cached target, else None."""
        text = text.lower()
        if _STOP_WORDS.search(text):
            return None  # "stop at the door" must not become a goto

        # ponytail: whole-word substring match, longest name first so
        # "front door" beats "door". No fuzzy/phonetic matching; the LLM
        # handles anything this misses.
        for name in sorted(self._waypoints, key=len, reverse=True):
            if re.search(rf"\b{re.escape(name)}\b", text):
                x, y = self._waypoints[name]
                return name, x, y

        # YOLO label in view, anchored to a landmark-map cell
        for det in self._detections():
            label = str(getattr(det, "label", det)).lower()
            if label and re.search(rf"\b{re.escape(label)}\b", text):
                xy = self._lookup(label)
                if xy:
                    x, y = self._snap(xy)
                    return label, x, y
        return None

    def _snap(self, xy: Tuple[float, float]) -> Tuple[float, float]:
        r = self._res
        return (round(xy[0] / r) * r, round(xy[1] / r) * r)

    # ------------------------------------------------------------------

    async def parse(self, text: str) -> Tuple[Dict[str, Any], str]:
        t0 = time.perf_counter()
        hit = self.resolve(text)
        if hit:
            name, x, y = hit
            logger.info("Fast path: '%s' -> (%.2f, %.2f) in %.1f ms",
                        name, x, y, (time.perf_counter() - t0) * 1e3)
            return {"cmd": "goto", "x": x, "y": y}, f"On my way to {name}."

        try:
            result = await self._llm.parse(text)
            path = "LLM"
        except Exception as exc:
            logger.warning("LLM parse failed (%s); using regex fallback", exc)
            result = await self._regex.parse(text)
            path = "regex"
        logger.info("%s path resolved in %.0f ms", path, (time.perf_counter() - t0) * 1e3)
        return result
