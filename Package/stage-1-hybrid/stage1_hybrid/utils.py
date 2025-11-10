"""Utility helpers for logging, configuration, and distributed setup."""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import torch

LOGGER_NAME = "liquid_stage1"


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure the root logger used by the package.

    This helper keeps Vertex console logs concise by emitting JSON lines for
    metrics while still allowing human readable informational messages.
    """

    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def json_log(logger: logging.Logger, payload: Dict[str, Any]) -> None:
    """Emit a JSON encoded payload to the logs."""

    logger.info(json.dumps(payload, sort_keys=True))


@dataclass
class TrainingDevice:
    """Information about the target accelerator and torch device."""

    device: torch.device
    name: str
    total_memory: Optional[int]
    capability: Optional[str]


def detect_training_device() -> TrainingDevice:
    """Inspect the CUDA runtime and gather device metadata."""

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        total_mem = getattr(props, "total_memory", None)
        capability = f"{props.major}.{props.minor}" if props else None
        name = props.name if props else "unknown"
        device = torch.device("cuda", idx)
        return TrainingDevice(device=device, name=name, total_memory=total_mem, capability=capability)
    cpu_device = torch.device("cpu")
    return TrainingDevice(device=cpu_device, name="cpu", total_memory=None, capability=None)


def set_seed(seed: int) -> None:
    """Set random seeds for Python and PyTorch."""

    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:  # pragma: no cover - optional dependency
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Backoff:
    """Simple exponential backoff helper used for GCS retries."""

    def __init__(self, base: float = 1.0, factor: float = 2.0, max_sleep: float = 30.0) -> None:
        self.base = base
        self.factor = factor
        self.max_sleep = max_sleep
        self.attempt = 0

    def sleep(self) -> None:
        delay = min(self.base * (self.factor ** self.attempt), self.max_sleep)
        time.sleep(delay)
        self.attempt += 1


@dataclass
class AnnealingSchedule:
    """Linear schedule between two values over a percentage of training steps."""

    start: float
    end: float
    pct: float

    def value(self, step: int, total_steps: int) -> float:
        if total_steps <= 0 or self.pct <= 0:
            return self.end
        cutoff = int(total_steps * self.pct)
        if cutoff <= 0 or step >= cutoff:
            return self.end
        progress = step / float(cutoff)
        return self.start + (self.end - self.start) * progress


@dataclass
class CosineLRSchedule:
    """Cosine learning rate scheduler with warmup."""

    base_lr: float
    warmup_steps: int
    total_steps: int

    def value(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / max(1, self.warmup_steps)
        progress = min(1.0, (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps))
        return self.base_lr * 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi))).item()


@dataclass
class RunMetadata:
    """Metadata saved alongside checkpoints."""

    step: int
    val_ppl: float
    losses: Dict[str, float]
    frozen_blocks: Iterable[str]

    def to_json(self) -> str:
        return json.dumps(
            {
                "step": self.step,
                "val_ppl": self.val_ppl,
                "losses": self.losses,
                "frozen_blocks": list(self.frozen_blocks),
            },
            sort_keys=True,
        )


def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""

    os.makedirs(path, exist_ok=True)
