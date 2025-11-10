"""Safe scientific calculator used for tool traces."""
from __future__ import annotations

import ast
import math
from typing import Any, Dict

_ALLOWED_NAMES: Dict[str, Any] = {
    "pi": math.pi,
    "e": math.e,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sqrt": math.sqrt,
    "floor": math.floor,
    "ceil": math.ceil,
}

_ALLOWED_NODES = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Num,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.Call,
    ast.Name,
    ast.Constant,
}


class UnsafeExpressionError(ValueError):
    """Raised when an expression is considered unsafe."""


def _check_node(node: ast.AST) -> None:
    if type(node) not in _ALLOWED_NODES:
        raise UnsafeExpressionError(f"Node {type(node).__name__} is not allowed")
    for child in ast.iter_child_nodes(node):
        _check_node(child)
    if isinstance(node, ast.Name) and node.id not in _ALLOWED_NAMES:
        raise UnsafeExpressionError(f"Unknown name {node.id}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise UnsafeExpressionError("Only direct function calls supported")
        if node.func.id not in _ALLOWED_NAMES:
            raise UnsafeExpressionError(f"Function {node.func.id} not permitted")


def evaluate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result."""

    if any(bad in expression for bad in ("__", "import", "open", "os.")):
        return "ERROR:FORBIDDEN"
    try:
        tree = ast.parse(expression, mode="eval")
        _check_node(tree)
        value = eval(compile(tree, filename="<calculator>", mode="eval"), {"__builtins__": {}}, _ALLOWED_NAMES)
    except UnsafeExpressionError as exc:
        return f"ERROR:{exc}"[:128]
    except Exception as exc:  # pragma: no cover - numeric errors
        return f"ERROR:{type(exc).__name__}"[:128]
    if isinstance(value, complex):
        return "ERROR:COMPLEX"
    return f"{float(value):.12g}"
