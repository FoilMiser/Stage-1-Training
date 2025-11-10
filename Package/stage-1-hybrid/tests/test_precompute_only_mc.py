import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage1_hybrid import data
from stage1_hybrid.precompute_hybrid_logits import run as run_precompute


class _Tokenizer:
    def __init__(self, seq_len: int) -> None:
        self.vocab_size = 8
        self.seq_len = seq_len
        self.pad_token = 0
        self.eos_token = 0

    def __call__(self, text: str, *, truncation: bool, max_length: int, padding: str, return_tensors: str):
        tokens = torch.zeros(max_length, dtype=torch.long)
        words = text.split()
        for idx, _ in enumerate(words[:max_length]):
            tokens[idx] = idx + 1
        attention = torch.zeros(max_length, dtype=torch.long)
        attention[: len(words[:max_length])] = 1
        return {
            "input_ids": tokens.unsqueeze(0),
            "attention_mask": attention.unsqueeze(0),
        }


def test_precompute_generates_only_math_and_code(tmp_path, monkeypatch):
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    def _write_sample(path: Path, sample_id: str, sample_type: str) -> None:
        path.write_text(
            json.dumps({"text": f"{sample_type} sample", "sample_id": sample_id, "type": sample_type}) + "\n",
            encoding="utf-8",
        )

    lm_path = data_dir / "lm.jsonl"
    math_path = data_dir / "math.jsonl"
    code_path = data_dir / "code.jsonl"
    _write_sample(lm_path, "lm-1", "lm")
    _write_sample(math_path, "math-1", "math_tool")
    _write_sample(code_path, "code-1", "code")

    manifest_path = manifest_dir / "manifest.jsonl"
    manifest_lines = [
        {"path": str(lm_path), "type": "lm"},
        {"path": str(math_path), "type": "math_tool"},
        {"path": str(code_path), "type": "code"},
    ]
    manifest_path.write_text("\n".join(json.dumps(line) for line in manifest_lines), encoding="utf-8")

    # Monkeypatch toolkit + tokenizer helpers to avoid external dependencies.
    monkeypatch.setattr("stage1_hybrid.precompute_hybrid_logits.ensure_toolkit", lambda *args, **kwargs: str(tmp_path))
    monkeypatch.setattr("stage1_hybrid.precompute_hybrid_logits.load_datasets_yaml", lambda *args, **kwargs: {})
    monkeypatch.setattr("stage1_hybrid.precompute_hybrid_logits.prepare_if_needed", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        "stage1_hybrid.precompute_hybrid_logits.data.build_tokenizer",
        lambda _model_id: _Tokenizer(seq_len=16),
    )

    class _Teacher:
        def __init__(self, _config) -> None:
            self.device = torch.device("cpu")

        def logits(self, input_ids: torch.Tensor) -> torch.Tensor:
            batch, seq = input_ids.shape
            torch.manual_seed(0)
            return torch.randn(batch, seq, 8)

    monkeypatch.setattr("stage1_hybrid.precompute_hybrid_logits.TeacherWrapper", _Teacher)

    math_dir = tmp_path / "math_out"
    code_dir = tmp_path / "code_out"
    args = [
        "--mc-teacher-id",
        "meta-llama/Llama-3.1-8B",
        "--dataset-manifest",
        str(manifest_path),
        "--math-logits-dir",
        str(math_dir),
        "--code-logits-dir",
        str(code_dir),
        "--seq-len",
        "16",
        "--batch-size",
        "1",
        "--num-workers",
        "0",
        "--overwrite",
        "true",
        "--prep-toolkit-zip-uri",
        "gs://stub",
    ]
    run_precompute(args)

    math_logits = list(math_dir.glob("*.pt"))
    code_logits = list(code_dir.glob("*.pt"))
    assert len(math_logits) == 1
    assert len(code_logits) == 1
    assert not any(tmp_path.glob("*.pt"))  # outputs stay in target dirs
