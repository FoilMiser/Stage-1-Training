"""Teacher model utilities for knowledge distillation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import configure_logging

logger = configure_logging()


@dataclass
class TeacherConfig:
    model_id: str
    device: str = "auto"
    dtype: torch.dtype = torch.bfloat16


class TeacherWrapper:
    """Wraps a HuggingFace causal LM for KD."""

    def __init__(self, config: TeacherConfig) -> None:
        logger.info("Loading teacher model %s", config.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=config.dtype,
            device_map=config.device,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.device = next(self.model.parameters()).device
        logger.info(
            "Teacher ready | vocab_size=%d | device=%s | dtype=%s",
            len(self.tokenizer),
            self.device,
            config.dtype,
        )

    @torch.inference_mode()
    def logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        outputs = self.model(input_ids=input_ids)
        return outputs.logits


def save_logits(sample_ids: Iterable[str], logits: torch.Tensor, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for sid, logit in zip(sample_ids, logits):
        out_path = Path(output_dir) / f"{sid}.pt"
        torch.save(logit.cpu(), out_path)


def maybe_load_logits(sample_ids: Iterable[str], input_dir: Optional[str]) -> Optional[torch.Tensor]:
    if not input_dir:
        return None
    loaded: List[torch.Tensor] = []
    for sid in sample_ids:
        path = Path(input_dir) / f"{sid}.pt"
        if not path.exists():
            return None
        loaded.append(torch.load(path, map_location="cpu"))
    return torch.stack(loaded, dim=0)


def precompute_teacher_logits(
    teacher: TeacherWrapper,
    dataset: Iterable[Dict[str, torch.Tensor]],
    output_dir: str,
    batch_size: int = 1,
) -> None:
    """Iterate through the dataset and persist teacher logits."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for step, batch in enumerate(dataset):
        input_ids = batch["input_ids"].to(teacher.model.device)
        logits = teacher.logits(input_ids)
        sample_ids = batch.get("sample_ids")
        if sample_ids is None:
            sample_ids = [f"sample_{step}_{i}" for i in range(input_ids.size(0))]
        save_logits(sample_ids, logits, output_dir)
        logger.info("Saved teacher logits for step %d", step)
