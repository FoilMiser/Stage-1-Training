"""Scratchpad helper used for tool reasoning."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Scratchpad:
    max_lines: int = 8
    lines: List[str] = field(default_factory=list)

    def add(self, label: str, content: str) -> None:
        line = f"{label.strip()}: {content.strip()}"
        if len(self.lines) >= self.max_lines:
            self.lines.pop(0)
        self.lines.append(line)

    def render(self) -> str:
        return "\n".join(self.lines)


def new_scratchpad(max_lines: int = 8) -> Scratchpad:
    return Scratchpad(max_lines=max_lines)
