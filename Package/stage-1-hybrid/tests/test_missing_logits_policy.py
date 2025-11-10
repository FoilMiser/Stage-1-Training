import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage1_hybrid import data
from stage1_hybrid.train import Trainer
from stage1_hybrid.utils import AnnealingSchedule


class _ToyModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 8, seq_len: int = 6) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, 16)
        self.ln = torch.nn.LayerNorm(16)
        self.lm_head = torch.nn.Linear(16, vocab_size)
        self.seq_len = seq_len

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(input_ids)
        hidden = self.ln(hidden)
        return self.lm_head(hidden)


class _MissingDataset(Dataset):
    def __init__(self, strict: bool) -> None:
        seq = 6
        self.strict = strict
        self.samples = [
            {
                "input_ids": torch.arange(seq, dtype=torch.long),
                "attention_mask": torch.ones(seq, dtype=torch.long),
                "sample_type": "lm",
                "sample_id": "lm-1",
                "teacher_logits": None,
            },
            {
                "input_ids": torch.arange(seq, dtype=torch.long) + 2,
                "attention_mask": torch.ones(seq, dtype=torch.long),
                "sample_type": "math_tool",
                "sample_id": "math-1",
                "teacher_logits": None,
                "hybrid_skip_kd": strict,
            },
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


class _Teacher:
    def __init__(self, vocab_size: int = 8) -> None:
        self.calls: list[torch.Tensor] = []
        self.vocab_size = vocab_size

    def logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.calls.append(input_ids.detach().cpu())
        batch, seq = input_ids.shape
        torch.manual_seed(0)
        return torch.randn(batch, seq, self.vocab_size)


def _run_training(tmp_path: Path, strict: bool) -> tuple[float, int]:
    dataset = _MissingDataset(strict=strict)
    loader = DataLoader(dataset, batch_size=2, collate_fn=data.collate_batch)
    model = _ToyModel()
    teacher = _Teacher()
    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=None,
        device=torch.device("cpu"),
        output_dir=str(tmp_path / ("strict" if strict else "lenient")),
        output_gcs_uri=None,
        run_id="missing",
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        warmup_steps=0,
        max_steps=1,
        kd_temperature=2.0,
        kd_alpha_schedule=AnnealingSchedule(0.7, 0.4, 0.3),
        ce_beta_schedule=AnnealingSchedule(0.3, 0.6, 0.3),
        logit_l2_gamma_schedule=AnnealingSchedule(0.0, 0.0, 1.0),
        logit_reference=None,
        precision="fp32",
        lm_teacher=teacher,
        teacher_mode="hybrid",
        teacher_logits_dir=None,
        math_logits_dir="/tmp/math",
        code_logits_dir="/tmp/code",
        hybrid=True,
        hybrid_strict=strict,
        eval_every=0,
        save_every=0,
        metrics_interval=1,
    )
    trainer.train()
    kd = trainer._last_metrics["kd_loss"]
    return kd, len(teacher.calls)


def test_missing_logits_lenient_allows_lm_kd(tmp_path):
    kd_loss, call_count = _run_training(tmp_path, strict=False)
    assert math.isfinite(kd_loss) and kd_loss > 0
    assert call_count == 1


def test_missing_logits_strict_skips_batch_kd(tmp_path):
    kd_loss, call_count = _run_training(tmp_path, strict=True)
    assert kd_loss == 0.0
    assert call_count == 0
