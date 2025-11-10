"""Dataset loading utilities for Stage‑1 hybrid training.

Key changes in this rewrite
- Explicit, reusable tokenizer constructor (`build_tokenizer`) with safe pad handling.
- Early, consistent sample type canonicalization ("math_tool" | "code" | "lm").
- Robust manifest resolution with GCS/local support and shard expansion.
- Teacher logits loader that normalizes sequence length and *vocabulary size to the tokenizer*.
- Hybrid routing for math/code logits with strict mode behavior surfaced in-batch.
- Clear logging of tokenizer vocab and per-epoch KD coverage stats.

This module intentionally does **not** decide which tokenizer to use; pass the
chosen HF model id from the CLI (e.g., `--dataset-tokenizer-id`).
"""
from __future__ import annotations

import glob
import gzip
import hashlib
import io
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

try:  # optional dependency for parquet manifests
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from . import tool_use
from .gcs_io import GCSIOError, gcs_to_local, list_gcs
from .prep import DatasetSpec, shards_ready
from .utils import configure_logging

logger = configure_logging()

# -------------------------------
# Local caches
# -------------------------------
_CACHE_DIR = Path("/tmp/manifest_cache"); _CACHE_DIR.mkdir(parents=True, exist_ok=True)
_LOGIT_CACHE_DIR = Path("/tmp/teacher_logits"); _LOGIT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------
# Manifest support
# -------------------------------
@dataclass
class ManifestEntry:
    """Represents one line in a manifest after resolution."""
    path: str
    resolved_path: str
    type: str
    weight: float = 1.0
    dataset_id: Optional[str] = None


def read_manifest(
    manifest_path: str,
    datasets_cfg: Optional[Dict[str, DatasetSpec]] = None,
) -> Tuple[List[ManifestEntry], List[Dict[str, object]]]:
    """Load a JSONL manifest (local or GCS) and resolve dataset roots.

    Returns a tuple `(entries, snapshot)` where `snapshot` is a plain-JSON
    summary useful for run metadata.
    """
    logger.info("Loading manifest from %s", manifest_path)
    if manifest_path.startswith("gs://"):
        local_path = gcs_to_local(manifest_path, "/tmp/manifest.jsonl")
    else:
        local_path = manifest_path

    entries: List[ManifestEntry] = []
    snapshot: List[Dict[str, object]] = []
    with open(local_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            raw_path = obj["path"]
            dtype = obj.get("type", "lm")
            weight = float(obj.get("weight", 1.0))
            dataset_id = _match_dataset(raw_path, datasets_cfg)
            resolved_path = _resolve_dataset_path(raw_path, dataset_id, datasets_cfg)
            entry = ManifestEntry(
                path=raw_path,
                resolved_path=resolved_path,
                type=dtype,
                weight=weight,
                dataset_id=dataset_id,
            )
            entries.append(entry)
            snapshot.append({
                "dataset_id": dataset_id,
                "type": dtype,
                "weight": weight,
                "path": raw_path,
                "resolved_path": resolved_path,
            })
    return entries, snapshot


# -------- Robust dataset root matching (string or list) --------

def _normalize_path(p: str) -> str:
    """Normalize local and GCS paths for prefix comparisons.
    - Converts backslashes to forward slashes
    - Collapses duplicate slashes (except in 'gs://')
    - Strips trailing slashes
    """
    s = str(p).replace("\\", "/").rstrip("/")
    s = re.sub(r"(?<!:)//+", "/", s)  # collapse dup slashes but keep scheme
    return s


def _iter_spec_candidates(spec) -> Iterator[str]:
    """Yield candidate root paths from a dataset spec.
    Supports dict or attribute style; values can be string or list/tuple/set.
    """
    def get_val(obj, key):
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    fields = ["manifest", "inp", "in", "out", "root", "roots"]
    for key in fields:
        v = get_val(spec, key)
        if not v:
            continue
        if isinstance(v, (list, tuple, set)):
            for x in v:
                if x:
                    yield str(x)
        else:
            yield str(v)


def _match_dataset(path: str, datasets_cfg: Optional[Dict[str, DatasetSpec]]) -> Optional[str]:
    if not datasets_cfg:
        return None
    normalized = _normalize_path(path)
    best_job = None
    best_len = -1
    for job, spec in datasets_cfg.items():
        for candidate in _iter_spec_candidates(spec):
            c = _normalize_path(candidate)
            # exact or directory-prefix match
            if normalized == c or normalized.startswith(c + "/"):
                clen = len(c)
                if clen > best_len:
                    best_job = job
                    best_len = clen
    return best_job


# ----------------------- type canonization -----------------------

_MATH_HINTS = re.compile(r"(?:\b|/)(gsm8k|asdiv|svamp|mawps|aqua|mathqa|math|arithmetic)(?:\b|/)", re.IGNORECASE)
_CODE_HINTS = re.compile(r"(?:\b|/)(code|codesearchnet|humaneval|mbpp|leetcode|github|programming|coding)(?:\b|/)", re.IGNORECASE)


def _canonicalize_type(raw_type: Optional[str], path: str, dataset_id: Optional[str]) -> str:
    """Normalize many possible labels to a small set used by the pipeline."""
    s = (raw_type or "").strip().lower().replace("-", "_")
    if s in {"math_tool", "math", "mathreasoning", "math_reasoning"}:
        return "math_tool"
    if s in {"code", "code_tool", "coding", "programming"}:
        return "code"
    hint = f"{path}::{dataset_id or ''}".lower()
    if _CODE_HINTS.search(hint):
        return "code"
    if _MATH_HINTS.search(hint):
        return "math_tool"
    return "lm"


# ----------------------- path resolution -----------------------

def _resolve_dataset_path(
    raw_path: str,
    dataset_id: Optional[str],
    datasets_cfg: Optional[Dict[str, DatasetSpec]],
) -> str:
    if not datasets_cfg or not dataset_id:
        return raw_path
    spec = datasets_cfg.get(dataset_id)
    if not spec:
        return raw_path

    # If shards are ready for this dataset, use its shard glob.
    try:
        if shards_ready(getattr(spec, "out", None)):
            return spec.shard_glob()
    except Exception:
        pass

    extension = Path(raw_path).suffix
    if extension in {".jsonl", ".json", ".json.gz", ".parquet"}:
        return raw_path
    return raw_path


def expand_gcs_pattern(pattern: str) -> List[str]:
    if pattern.startswith("gs://"):
        try:
            return list_gcs(pattern)
        except GCSIOError:
            logger.warning("Failed to expand GCS pattern %s", pattern)
            return [pattern]
    matched = glob.glob(pattern)
    return matched if matched else [pattern]


def _resolve_local(path: str) -> str:
    if path.startswith("gs://"):
        digest = hashlib.md5(path.encode("utf-8")).hexdigest()
        local = _CACHE_DIR / f"{digest}_{Path(path).name}"
        if not local.exists():
            gcs_to_local(path, str(local))
        return str(local)
    return path


# ----------------------- record streaming -----------------------

def _stream_json_lines(local_path: str) -> Iterator[Dict[str, object]]:
    with open(local_path, "rb") as fh:
        reader = io.BufferedReader(fh)
        for raw in reader:
            line = raw.decode("utf-8")
            if not line.strip():
                continue
            yield json.loads(line)


def _stream_json_gz(local_path: str) -> Iterator[Dict[str, object]]:
    with gzip.open(local_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            yield json.loads(line)


def _stream_parquet(local_path: str) -> Iterator[Dict[str, object]]:  # pragma: no cover
    if pq is None:
        raise RuntimeError("pyarrow is required to read parquet manifests")
    table = pq.ParquetFile(local_path)
    for batch in table.iter_batches():
        for row in batch.to_pylist():
            yield row


def load_records(path: str) -> Iterator[Dict[str, object]]:
    local = _resolve_local(path)
    suffix = Path(local).suffix
    if suffix == ".gz":
        yield from _stream_json_gz(local)
    elif suffix == ".parquet":
        yield from _stream_parquet(local)
    else:
        yield from _stream_json_lines(local)


# ----------------------- dataset -----------------------
class ManifestDataset(Dataset):
    """Torch dataset built from a manifest file.

    Exposes the following fields per item:
      - input_ids: LongTensor [seq]
      - attention_mask: LongTensor [seq]
      - sample_type: str in {"lm","math_tool","code"}
      - sample_id: str (stable id used for teacher logit filenames)
      - teacher_logits (optional): FloatTensor [seq, vocab]
      - teacher_status: str in {"ok","missing","invalid","cached","disabled"}
      - hybrid_skip_kd (optional): bool when strict mode disables KD for batch
    """

    def __init__(
        self,
        manifest_entries: Sequence[ManifestEntry],
        tokenizer: PreTrainedTokenizerBase,
        seq_len: int,
        tool_use_ratio: float = 0.0,
        teacher_mode: str = "precompute",
        teacher_logits_dir: Optional[str] = None,
        *,
        hybrid: bool = False,
        math_logits_dir: Optional[str] = None,
        code_logits_dir: Optional[str] = None,
        hybrid_strict: bool = False,
        split: str = "train",
    ) -> None:
        self.entries = list(manifest_entries)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.tool_use_ratio = tool_use_ratio
        self.samples: List[Dict[str, str]] = []
        self.dataset_counts: Dict[str, int] = {}
        self.teacher_mode = teacher_mode
        self.teacher_logits_dir = teacher_logits_dir.rstrip("/") if teacher_logits_dir else None
        self.hybrid = hybrid
        self.math_logits_dir = math_logits_dir.rstrip("/") if math_logits_dir else None
        self.code_logits_dir = code_logits_dir.rstrip("/") if code_logits_dir else None
        self.hybrid_strict = hybrid_strict
        self.split = split

        # caches + stats
        self._teacher_cache: Dict[tuple[str, str], torch.Tensor] = {}
        self._auto_sample_counts: Dict[str, int] = defaultdict(int)
        self._bad_shape_warned: set[str] = set()
        self._missing_warned: Dict[str, set[str]] = defaultdict(set)
        self._epoch_total = 0
        self._epoch_missing = 0
        self.sample_type_counts: Dict[str, int] = {}
        self._hybrid_missing_warned_types: set[str] = set()

        logger.info("Tokenizer vocab_size=%s | pad_token=%r", getattr(self.tokenizer, "vocab_size", None), getattr(self.tokenizer, "pad_token", None))
        self._build_index()

    # ----- lifecycle helpers -----
    def begin_epoch(self) -> None:
        self._epoch_total = 0
        self._epoch_missing = 0

    def epoch_teacher_stats(self) -> tuple[int, int]:
        return self._epoch_total, self._epoch_missing

    # ----- core dataset API -----
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        tokens = self.tokenizer(
            sample["text"],
            truncation=True,
            max_length=self.seq_len,
            padding="max_length",
            return_tensors="pt",
        )

        # normalize sample_type (defensive)
        st = sample["sample_type"]
        if st == "math":
            st = "math_tool"
        elif st in {"code_tool", "coding", "programming"}:
            st = "code"

        item: Dict[str, object] = {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "sample_type": st,
            "sample_id": sample["sample_id"],
            "type": st,  # legacy alias
        }

        directory: Optional[str] = None
        source_tag = "disabled"

        # unified precompute mode (single output dir)
        if self.teacher_mode == "precompute" and self.teacher_logits_dir:
            directory = self.teacher_logits_dir
            source_tag = "precompute"
        # hybrid routing (two output dirs) — relies on normalized type
        elif self.hybrid and st in {"math_tool", "code"}:
            directory = self.math_logits_dir if st == "math_tool" else self.code_logits_dir
            source_tag = st

        if directory:
            logits, status = self._load_teacher_logits(sample["sample_id"], directory, source_tag)
            self._epoch_total += 1
            if status in {"missing", "invalid"}:
                self._epoch_missing += 1
            if logits is not None:
                item["teacher_logits"] = logits
            if status == "missing" and self.hybrid and self.hybrid_strict and source_tag in {"math_tool", "code"}:
                warn_key = f"{self.split}:{source_tag}"
                if warn_key not in self._hybrid_missing_warned_types:
                    self._hybrid_missing_warned_types.add(warn_key)
                    logger.warning(
                        "Hybrid strict mode: missing %s logits detected on split=%s; disabling KD for affected batches",
                        source_tag,
                        self.split,
                    )
                item["hybrid_skip_kd"] = True
            item["teacher_status"] = status

        return item

    # ----- index construction -----
    def _build_index(self) -> None:
        counts: Dict[str, int] = defaultdict(int)
        type_counts: Dict[str, int] = defaultdict(int)
        for entry in self.entries:
            resolved_paths = expand_gcs_pattern(entry.resolved_path)
            dataset_key = entry.dataset_id or entry.resolved_path
            for path in resolved_paths:
                for row_idx, obj in enumerate(load_records(path)):
                    text = obj.get("text", "")
                    if not isinstance(text, str) or not text:
                        continue

                    # canonicalize sample_type early
                    declared = obj.get("type", entry.type)
                    sample_type = _canonicalize_type(declared, path, entry.dataset_id)

                    # optional tool-use injection for math
                    if sample_type == "math_tool":
                        text = tool_use.traces.maybe_inject_tool_result(text)

                    sample_id = obj.get("sample_id")
                    if not sample_id:
                        digest = hashlib.sha1(f"{path}:{row_idx}:{text[:200]}".encode("utf-8")).hexdigest()
                        sample_id = f"auto_{digest}"
                        self._auto_sample_counts[dataset_key] += 1

                    self.samples.append({
                        "text": text,
                        "sample_type": sample_type,
                        "sample_id": sample_id,
                    })
                    counts[dataset_key] += 1
                    type_counts[sample_type] += 1

        self.dataset_counts = dict(counts)
        self.auto_sample_counts = dict(self._auto_sample_counts)
        self.sample_type_counts = dict(type_counts)
        logger.info("Loaded %d samples from manifest", len(self.samples))
        logger.info("Sample-type distribution: %s", dict(self.sample_type_counts))
        for dataset_key, missing in self._auto_sample_counts.items():
            if missing:
                logger.warning(
                    "Dataset %s missing sample_id for %d records; auto-generated using SHA1",
                    dataset_key,
                    missing,
                )

    # ----- teacher logits IO -----
    def _materialize_teacher_path(self, directory: str, sample_id: str, ext: str) -> Optional[str]:
        base = f"{directory}/{sample_id}{ext}"
        if base.startswith("gs://"):
            cache_name = hashlib.md5(base.encode("utf-8")).hexdigest()
            local_path = _LOGIT_CACHE_DIR / f"{cache_name}{ext}"
            if local_path.exists():
                return str(local_path)
            try:
                return gcs_to_local(base, str(local_path))
            except GCSIOError:
                return None
        path = Path(base)
        if path.exists():
            return str(path)
        return None

    def _normalize_teacher_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Pad/crop logits to [seq_len, tokenizer_vocab].

        This is *crucial* when mixing teachers with student/dataset tokenizers.
        """
        logits = torch.as_tensor(logits).float()
        if logits.dim() != 2:
            logits = logits.view(-1, logits.shape[-1])
        seq_len, vocab_size = logits.shape

        # sequence length normalization
        if seq_len < self.seq_len:
            logits = F.pad(logits, (0, 0, 0, self.seq_len - seq_len))
        elif seq_len > self.seq_len:
            logits = logits[: self.seq_len]

        # vocab normalization to the tokenizer
        target_vocab = int(getattr(self.tokenizer, "vocab_size", vocab_size) or vocab_size)
        if vocab_size < target_vocab:
            logits = F.pad(logits, (0, target_vocab - vocab_size))
        elif vocab_size > target_vocab:
            logits = logits[:, :target_vocab]
        return logits

    def _load_teacher_logits(self, sample_id: str, directory: str, tag: str) -> tuple[Optional[torch.Tensor], str]:
        cache_key = (directory, sample_id)
        if cache_key in self._teacher_cache:
            return self._teacher_cache[cache_key], "cached"
        if not directory:
            return None, "disabled"

        for ext in (".pt", ".npy"):
            path = self._materialize_teacher_path(directory, sample_id, ext)
            if not path:
                continue
            try:
                tensor = torch.load(path, map_location="cpu") if ext == ".pt" else torch.from_numpy(np.load(path, allow_pickle=False))
            except Exception:
                continue

            logits = torch.as_tensor(tensor)
            if logits.dim() != 2:
                key = f"shape:{Path(path).suffix}:{tag}"
                if key not in self._bad_shape_warned:
                    self._bad_shape_warned.add(key)
                    logger.warning("Teacher logits at %s have invalid shape %s; expected [T, V]; skipping", path, tuple(logits.shape))
                return None, "invalid"

            logits = self._normalize_teacher_logits(logits)
            self._teacher_cache[cache_key] = logits
            return logits, "ok"

        if sample_id not in self._missing_warned[directory]:
            self._missing_warned[directory].add(sample_id)
            logger.warning("Teacher logits missing for sample %s in %s", sample_id, directory)
        return None, "missing"


# ----------------------- collate -----------------------

def collate_batch(samples: Sequence[Dict[str, object]]) -> Dict[str, object]:
    input_ids = torch.stack([sample["input_ids"] for sample in samples])
    attention_mask = torch.stack([sample["attention_mask"] for sample in samples])
    sample_ids = [str(sample["sample_id"]) for sample in samples]
    sample_types = [str(sample.get("sample_type", "lm")) for sample in samples]
    teacher_logits = [sample.get("teacher_logits") for sample in samples]
    teacher_status = [sample.get("teacher_status", "disabled") for sample in samples]
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "sample_ids": sample_ids,
        "sample_types": sample_types,
        "teacher_logits": teacher_logits,
        "teacher_status": teacher_status,
    }
    if any(sample.get("hybrid_skip_kd") for sample in samples):
        batch["hybrid_skip_kd"] = True
    return batch


# ----------------------- tokenizer -----------------------

def build_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    """Create a fast tokenizer and ensure we have a pad token.

    For Llama tokenizers (often no pad by default), align pad to EOS so we can
    use `padding="max_length"` without producing -100 labels downstream unless
    the trainer overrides.
    """
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    return tok
