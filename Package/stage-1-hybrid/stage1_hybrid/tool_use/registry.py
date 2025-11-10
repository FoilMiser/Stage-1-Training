"""Tool registry used by the training pipeline."""
from __future__ import annotations

from typing import Dict, Protocol

from . import calculator, scratchpad


class ToolFn(Protocol):
    def __call__(self, argument: str) -> str:
        ...


class ScratchpadTool:
    def __init__(self) -> None:
        self.pad = scratchpad.new_scratchpad()

    def __call__(self, argument: str) -> str:
        label, _, content = argument.partition(":")
        if not content:
            content = label
            label = "note"
        self.pad.add(label.strip(), content.strip())
        return self.pad.render()


def _calculator_tool(argument: str) -> str:
    return calculator.evaluate(argument)


def build_registry() -> Dict[str, ToolFn]:
    return {
        "calculator": _calculator_tool,
        "scratchpad": ScratchpadTool(),
    }


_TOOLS = build_registry()


def call_tool(name: str, argument: str) -> str:
    tool = _TOOLS.get(name)
    if tool is None:
        return "ERROR:UNKNOWN_TOOL"
    return tool(argument)
