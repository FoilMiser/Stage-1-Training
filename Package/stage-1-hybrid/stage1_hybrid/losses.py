"""Loss functions used for Stage-1 KD."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def ce_loss(student_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(student_logits.reshape(-1, student_logits.size(-1)), labels.reshape(-1))


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    student_logits = student_logits.to(torch.float32)
    teacher_logits = teacher_logits.to(torch.float32)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    return loss * (temperature ** 2)


def logit_l2(student_logits: torch.Tensor, reference_logits: Optional[torch.Tensor]) -> torch.Tensor:
    if reference_logits is None:
        return torch.tensor(0.0, device=student_logits.device)
    return F.mse_loss(student_logits, reference_logits)
