"""Parsing utilities for tool call traces."""
from __future__ import annotations

import io
import re
from typing import Iterable

from .registry import call_tool

_CALL_RE = re.compile(r"^CALL\s+(?P<name>[a-zA-Z0-9_\-]+):\s*\"(?P<arg>.*)\"\s*$")


def maybe_inject_tool_result(text: str) -> str:
    """Inject RESULT lines following tool CALL instructions."""

    output_lines = []
    for line in io.StringIO(text):
        stripped = line.rstrip("\n")
        output_lines.append(stripped)
        match = _CALL_RE.match(stripped)
        if match:
            name = match.group("name").lower()
            arg = match.group("arg")
            result = call_tool(name, arg)
            output_lines.append(f"RESULT: {result}")
    return "\n".join(output_lines)
