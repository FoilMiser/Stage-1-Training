"""Evaluation helpers for validation perplexity and tool metrics."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .losses import ce_loss
from .tool_use import traces
from .utils import configure_logging

logger = configure_logging()


def run_validation(model: torch.nn.Module, dataloader: DataLoader, max_batches: int = 16) -> Dict[str, Optional[float]]:
    model.eval()
    losses: List[float] = []
    tool_preds: List[str] = []
    tool_targets: List[str] = []
    tokenizer = getattr(getattr(dataloader, "dataset", None), "tokenizer", None)
    processed_batches = 0
    max_tool_samples = 16
    with torch.inference_mode():
        for batch in dataloader:
            if processed_batches >= max_batches:
                break
            processed_batches += 1
            input_ids = batch["input_ids"].to(model.lm_head.weight.device)
            student_inputs = input_ids[:, :-1]
            logits = model(student_inputs)
            loss = ce_loss(logits, input_ids[:, 1:])
            losses.append(loss.item())
            if tokenizer is None:
                continue
            types = batch.get("type")
            if types is None:
                continue
            sample_types = types
            if isinstance(sample_types, torch.Tensor):
                sample_types = [t.item() for t in sample_types]
            tool_count = 0
            for idx, sample_type in enumerate(sample_types):
                if sample_type != "math_tool" or tool_count >= max_tool_samples:
                    continue
                text = _decode_sample(batch, idx, tokenizer)
                if not text:
                    continue
                actual, expected = _extract_tool_results(text)
                if actual and expected and len(actual) == len(expected):
                    tool_preds.extend(actual)
                    tool_targets.extend(expected)
                    tool_count += 1
    model.train()
    metrics: Dict[str, Optional[float]] = {"perplexity": float("inf"), "tool_em": None}
    if losses:
        metrics["perplexity"] = math.exp(sum(losses) / len(losses))
    if tool_preds and tool_targets:
        metrics["tool_em"] = compute_tool_accuracy(tool_preds, tool_targets)
    return metrics


def _decode_sample(batch: Dict[str, torch.Tensor], idx: int, tokenizer) -> str:
    input_ids = batch["input_ids"][idx]
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        mask = attention_mask[idx]
        if isinstance(mask, torch.Tensor):
            length = int(mask.sum().item())
            if length > 0:
                input_ids = input_ids[:length]
    try:
        return tokenizer.decode(input_ids.tolist(), skip_special_tokens=True)
    except Exception:
        return ""


def _extract_tool_results(text: str) -> Tuple[List[str], List[str]]:
    actual: List[str] = []
    for line in text.splitlines():
        if line.strip().startswith("RESULT:"):
            actual.append(line.split("RESULT:", 1)[1].strip())
    stripped = "\n".join(line for line in text.splitlines() if not line.strip().startswith("RESULT:"))
    reinjected = traces.maybe_inject_tool_result(stripped)
    expected: List[str] = []
    for line in reinjected.splitlines():
        if line.strip().startswith("RESULT:"):
            expected.append(line.split("RESULT:", 1)[1].strip())
    return actual, expected


def compute_tool_accuracy(preds: Iterable[str], targets: Iterable[str]) -> float:
    total = 0
    correct = 0
    for pred, target in zip(preds, targets):
        total += 1
        if pred.strip() == target.strip():
            correct += 1
    return correct / total if total else 0.0
