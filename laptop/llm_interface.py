"""
LLM command parser: natural language -> navigation JSON.

Supports Anthropic Claude (primary) and OpenAI GPT (fallback).
Falls back to regex-based parsing if no API key is available.

Output command format:
  {"cmd": "goto",    "x": float, "y": float}
  {"cmd": "move",    "direction": str, "distance": float}
  {"cmd": "stop"}
  {"cmd": "unknown", "message": str}

Each parse also returns a short spoken response for TTS.

Usage:
  llm = create_llm_interface(waypoints={"door": [3.0, 0.0]})
  cmd, speech = await llm.parse("take me to the door")
  # cmd = {"cmd": "goto", "x": 3.0, "y": 0.0}
  # speech = "On my way to the door"
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import load_config

logger = logging.getLogger(__name__)

# Load .env if present
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
        logger.info("Loaded .env from %s", _env_path)
except ImportError:
    pass


SYSTEM_PROMPT_TEMPLATE = """\
You are the navigation controller for an autonomous wheelchair.

Available locations (name -> x, y coordinates):
{locations}

Parse the user's spoken command into a JSON action. Respond with ONLY a JSON \
object on the first line, then a short spoken response (under 10 words) on the \
second line.

Commands you can output:
{{"cmd": "goto", "x": <float>, "y": <float>}}
{{"cmd": "move", "direction": "forward|back|left|right", "distance": <meters>}}
{{"cmd": "stop"}}
{{"cmd": "unknown", "message": "<reason>"}}

Examples:
User: take me to the door
{{"cmd": "goto", "x": 3.0, "y": 0.0}}
Taking you to the door.

User: go forward 2 meters
{{"cmd": "move", "direction": "forward", "distance": 2.0}}
Moving forward 2 meters.

User: stop
{{"cmd": "stop"}}
Stopping now.

User: go to the cafeteria
{{"cmd": "unknown", "message": "I don't know where the cafeteria is"}}
I don't know where that is. Can you guide me?
"""


def _format_locations(waypoints: Dict[str, List[float]]) -> str:
    if not waypoints:
        return "  (no locations loaded)"
    lines = []
    for name, coords in waypoints.items():
        lines.append(f"  {name}: ({coords[0]:.1f}, {coords[1]:.1f})")
    return "\n".join(lines)


def load_waypoints(path: Optional[str] = None) -> Dict[str, List[float]]:
    """Load waypoints from JSON file."""
    if path is None:
        path = str(Path(__file__).resolve().parent.parent / "waypoints" / "demo_space.json")
    try:
        with open(path) as f:
            wp = json.load(f)
        logger.info("Loaded %d waypoints from %s", len(wp), path)
        return wp
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.warning("Could not load waypoints from %s: %s", path, exc)
        return {}


# ======================================================================
# Anthropic Claude
# ======================================================================

class AnthropicLLM:
    """Parse commands using Anthropic Claude API."""

    def __init__(self, waypoints: Dict[str, List[float]], model: Optional[str] = None):
        self._waypoints = waypoints
        cfg = load_config().llm
        self._model = model or cfg.model
        self._api_key = cfg.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client = None

    async def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def parse(self, text: str) -> Tuple[Dict[str, Any], str]:
        client = await self._get_client()

        system = SYSTEM_PROMPT_TEMPLATE.format(
            locations=_format_locations(self._waypoints),
        )

        message = await client.messages.create(
            model=self._model,
            max_tokens=200,
            system=system,
            messages=[{"role": "user", "content": text}],
        )

        raw = message.content[0].text.strip()
        return _parse_llm_response(raw, self._waypoints)


# ======================================================================
# OpenAI GPT
# ======================================================================

class OpenAILLM:
    """Parse commands using OpenAI GPT API."""

    def __init__(self, waypoints: Dict[str, List[float]], model: Optional[str] = None):
        self._waypoints = waypoints
        cfg = load_config().llm
        self._model = model or cfg.openai_model
        self._api_key = cfg.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client = None

    async def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def parse(self, text: str) -> Tuple[Dict[str, Any], str]:
        client = await self._get_client()

        system = SYSTEM_PROMPT_TEMPLATE.format(
            locations=_format_locations(self._waypoints),
        )

        response = await client.chat.completions.create(
            model=self._model,
            max_tokens=200,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
        )

        raw = response.choices[0].message.content.strip()
        return _parse_llm_response(raw, self._waypoints)


# ======================================================================
# Regex fallback (no API needed)
# ======================================================================

class RegexLLM:
    """Regex-based command parser — works offline, no API key needed."""

    def __init__(self, waypoints: Dict[str, List[float]]):
        self._waypoints = waypoints

    async def parse(self, text: str) -> Tuple[Dict[str, Any], str]:
        text_lower = text.lower().strip()

        # Stop
        if re.search(r'\b(stop|halt|wait|freeze|pause)\b', text_lower):
            return {"cmd": "stop"}, "Stopping now."

        # Goto: "go to <place>" / "take me to <place>" / "navigate to <place>"
        m = re.search(r'(?:go\s+to|take\s+me\s+to|navigate\s+to|head\s+to)\s+(?:the\s+)?(.+)', text_lower)
        if m:
            place = m.group(1).strip().rstrip('.')
            coords = self._fuzzy_lookup(place)
            if coords:
                return (
                    {"cmd": "goto", "x": coords[0], "y": coords[1]},
                    f"On my way to {place}.",
                )
            return (
                {"cmd": "unknown", "message": f"I don't know where '{place}' is"},
                f"I don't know where {place} is.",
            )

        # Move: "move forward 2 meters" / "go left 1.5m"
        m = re.search(
            r'(?:move|go)\s+(forward|back(?:ward)?|left|right)\s+'
            r'(\d+(?:\.\d+)?)\s*(?:m(?:eters?)?)?',
            text_lower,
        )
        if m:
            direction = m.group(1)
            if direction.startswith("back"):
                direction = "back"
            distance = float(m.group(2))
            return (
                {"cmd": "move", "direction": direction, "distance": distance},
                f"Moving {direction} {distance} meters.",
            )

        # Simple direction without distance
        m = re.search(r'(?:move|go)\s+(forward|back(?:ward)?|left|right)', text_lower)
        if m:
            direction = m.group(1)
            if direction.startswith("back"):
                direction = "back"
            return (
                {"cmd": "move", "direction": direction, "distance": 1.0},
                f"Moving {direction} 1 meter.",
            )

        # Try direct place name match
        for name, coords in self._waypoints.items():
            if name.lower() in text_lower:
                return (
                    {"cmd": "goto", "x": coords[0], "y": coords[1]},
                    f"Heading to {name}.",
                )

        return (
            {"cmd": "unknown", "message": f"Could not understand: {text}"},
            "I didn't understand that.",
        )

    def _fuzzy_lookup(self, query: str) -> Optional[List[float]]:
        """Find the best matching waypoint name."""
        query_lower = query.lower()
        # Exact match
        for name, coords in self._waypoints.items():
            if name.lower() == query_lower:
                return coords
        # Substring match
        for name, coords in self._waypoints.items():
            if name.lower() in query_lower or query_lower in name.lower():
                return coords
        return None


# ======================================================================
# Response parser
# ======================================================================

def _parse_llm_response(raw: str, waypoints: Dict[str, List[float]]
                         ) -> Tuple[Dict[str, Any], str]:
    """Extract JSON command + spoken response from LLM output."""
    lines = raw.strip().split("\n")

    cmd = None
    speech = "OK."

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                cmd = json.loads(line)
            except json.JSONDecodeError:
                # Try to extract JSON from the line
                m = re.search(r'\{.*\}', line)
                if m:
                    try:
                        cmd = json.loads(m.group())
                    except json.JSONDecodeError:
                        pass
        elif cmd is not None and not line.startswith("{"):
            speech = line
            break

    if cmd is None:
        # Last resort: try to parse entire output as JSON
        try:
            cmd = json.loads(raw)
        except json.JSONDecodeError:
            cmd = {"cmd": "unknown", "message": f"Could not parse LLM output"}
            speech = "I didn't understand that."

    # Validate command structure
    if "cmd" not in cmd:
        cmd = {"cmd": "unknown", "message": "Missing cmd field"}

    return cmd, speech


# ======================================================================
# Factory
# ======================================================================

def create_llm_interface(
    waypoints: Optional[Dict[str, List[float]]] = None,
    provider: Optional[str] = None,
) -> Any:
    """Create the best available LLM interface.

    Priority: Anthropic > OpenAI > Regex fallback.
    Set provider="anthropic", "openai", or "regex" to force one.
    """
    if waypoints is None:
        waypoints = load_waypoints()

    cfg = load_config().llm
    anthropic_key = cfg.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    openai_key = cfg.openai_api_key or os.environ.get("OPENAI_API_KEY", "")

    if provider == "anthropic" or (provider is None and anthropic_key):
        try:
            import anthropic  # noqa: F401
            logger.info("Using Anthropic Claude (%s)", cfg.model)
            return AnthropicLLM(waypoints)
        except ImportError:
            logger.warning("anthropic package not installed")

    if provider == "openai" or (provider is None and openai_key):
        try:
            import openai  # noqa: F401
            logger.info("Using OpenAI GPT (%s)", cfg.openai_model)
            return OpenAILLM(waypoints)
        except ImportError:
            logger.warning("openai package not installed")

    logger.info("Using regex-based command parser (no API key)")
    return RegexLLM(waypoints)


# ======================================================================
# Quick test
# ======================================================================

async def _test():
    import sys
    logging.basicConfig(level=logging.INFO)

    llm = create_llm_interface()
    print(f"LLM backend: {type(llm).__name__}")

    tests = [
        "take me to the door",
        "go to the table",
        "move forward 2 meters",
        "stop",
        "go to the cafeteria",
        "navigate to window",
    ]
    for t in tests:
        cmd, speech = await llm.parse(t)
        print(f"  '{t}' -> {cmd} | \"{speech}\"")


if __name__ == "__main__":
    import asyncio
    asyncio.run(_test())
