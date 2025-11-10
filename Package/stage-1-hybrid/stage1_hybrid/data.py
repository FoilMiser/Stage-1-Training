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
import hashlib
import io
import json
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from . import tool_use
from .gcs_io import GCSIOError, gcs_to_local, iter_gcs, open_uri
from .interleave import interleave
from .prep import DatasetSpec, shards_ready
from .stream_io import stream_jsonl, stream_jsonl_gz, stream_parquet
from .utils import configure_logging

logger = configure_logging()

# -------------------------------
# Local caches
# -------------------------------
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
    approx_records: Optional[int] = None


def _maybe_int(value) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def read_manifest(
    manifest_path: str,
    datasets_cfg: Optional[Dict[str, DatasetSpec]] = None,
) -> Tuple[List[ManifestEntry], List[Dict[str, object]]]:
    """Load a JSONL manifest (local or GCS) and resolve dataset roots.

    Returns a tuple `(entries, snapshot)` where `snapshot` is a plain-JSON
    summary useful for run metadata.
    """
    logger.info("Loading manifest from %s", manifest_path)
    entries: List[ManifestEntry] = []
    snapshot: List[Dict[str, object]] = []
    if manifest_path.startswith("gs://"):
        with open_uri(manifest_path, "rb") as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8")
            for line in text:
                if not line.strip():
                    continue
                obj = json.loads(line)
                raw_path = obj["path"]
                dtype = obj.get("type", "lm")
                weight = float(obj.get("weight", 1.0))
                dataset_id = _match_dataset(raw_path, datasets_cfg)
                resolved_path = _resolve_dataset_path(raw_path, dataset_id, datasets_cfg)
                approx = obj.get("count") or obj.get("num_records") or obj.get("records")
                entry = ManifestEntry(
                    path=raw_path,
                    resolved_path=resolved_path,
                    type=dtype,
                    weight=weight,
                    dataset_id=dataset_id,
                    approx_records=_maybe_int(approx),
                )
                entries.append(entry)
                snapshot.append({
                    "dataset_id": dataset_id,
                    "type": dtype,
                    "weight": weight,
                    "path": raw_path,
                    "resolved_path": resolved_path,
                })
    else:
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                raw_path = obj["path"]
                dtype = obj.get("type", "lm")
                weight = float(obj.get("weight", 1.0))
                dataset_id = _match_dataset(raw_path, datasets_cfg)
                resolved_path = _resolve_dataset_path(raw_path, dataset_id, datasets_cfg)
                approx = obj.get("count") or obj.get("num_records") or obj.get("records")
                entry = ManifestEntry(
                    path=raw_path,
                    resolved_path=resolved_path,
                    type=dtype,
                    weight=weight,
                    dataset_id=dataset_id,
                    approx_records=_maybe_int(approx),
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


def expand_gcs_pattern(pattern: str) -> Iterator[str]:
    if pattern.startswith("gs://"):
        try:
            yielded = False
            for uri in iter_gcs(pattern):
                yielded = True
                yield uri
            if not yielded:
                yield pattern
        except GCSIOError:
            logger.warning("Failed to expand GCS pattern %s", pattern)
            yield pattern
        return

    matched = False
    for path in glob.iglob(pattern):
        matched = True
        yield path
    if not matched:
        yield pattern


class _TensorCache:
    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max(0, int(max_bytes))
        self._bytes = 0
        self._store: OrderedDict[tuple[str, str], torch.Tensor] = OrderedDict()

    def get(self, key: tuple[str, str]) -> Optional[torch.Tensor]:
        tensor = self._store.pop(key, None)
        if tensor is not None:
            self._store[key] = tensor
        return tensor

    def put(self, key: tuple[str, str], tensor: torch.Tensor) -> None:
        if self.max_bytes <= 0:
            return
        tensor = tensor.detach().cpu()
        size = tensor.element_size() * tensor.nelement()
        if size > self.max_bytes:
            return
        existing = self._store.pop(key, None)
        if existing is not None:
            self._bytes -= existing.element_size() * existing.nelement()
        while self._bytes + size > self.max_bytes and self._store:
            _, old = self._store.popitem(last=False)
            self._bytes -= old.element_size() * old.nelement()
        self._store[key] = tensor
        self._bytes += size


class ManifestDataset(IterableDataset):
    """Torch dataset built from a manifest file using streaming readers."""

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
        streaming: bool = True,
        parquet_batch_rows: int = 1024,
        data_max_open_files: int = 2,
        data_max_cache_bytes: int = 0,
        arrow_num_threads: int = 1,
        per_source_buffer: int = 1024,
    ) -> None:
        self.entries = list(manifest_entries)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.tool_use_ratio = tool_use_ratio
        self.teacher_mode = teacher_mode
        self.teacher_logits_dir = teacher_logits_dir.rstrip('/') if teacher_logits_dir else None
        self.hybrid = hybrid
        self.math_logits_dir = math_logits_dir.rstrip('/') if math_logits_dir else None
        self.code_logits_dir = code_logits_dir.rstrip('/') if code_logits_dir else None
        self.hybrid_strict = hybrid_strict
        self.split = split
        self.streaming = streaming
        self.parquet_batch_rows = max(1, int(parquet_batch_rows))
        self.data_max_open_files = max(1, int(data_max_open_files))
        self.arrow_num_threads = max(1, int(arrow_num_threads))
        self.per_source_buffer = max(1, int(per_source_buffer))

        self.dataset_counts: Dict[str, int] = defaultdict(int)
        self.sample_type_counts: Dict[str, int] = defaultdict(int)
        self.auto_sample_counts: Dict[str, int] = {}

        self._teacher_cache = _TensorCache(int(data_max_cache_bytes))
        self._auto_sample_counts: Dict[str, int] = defaultdict(int)
        self._bad_shape_warned: set[str] = set()
        self._missing_warned: Dict[str, set[str]] = defaultdict(set)
        self._hybrid_missing_warned_types: set[str] = set()
        self._epoch_total = 0
        self._epoch_missing = 0

        self._length_hint = sum(entry.approx_records or 0 for entry in self.entries)
        logger.info(
            "Tokenizer vocab_size=%s | pad_token=%r",
            getattr(self.tokenizer, "vocab_size", None),
            getattr(self.tokenizer, "pad_token", None),
        )

    def begin_epoch(self) -> None:
        self._epoch_total = 0
        self._epoch_missing = 0

    def epoch_teacher_stats(self) -> tuple[int, int]:
        return self._epoch_total, self._epoch_missing

    def __len__(self) -> int:
        return int(self._length_hint) if self._length_hint else 0

    def _iter_entry_paths(self, entry: ManifestEntry) -> Iterator[str]:
        seen = False
        for path in expand_gcs_pattern(entry.resolved_path):
            seen = True
            yield path
        if not seen:
            yield entry.resolved_path

    def _stream_records(self, path: str) -> Iterator[Dict[str, object]]:
        suffix = Path(path).suffix.lower()
        try:
            if suffix == '.gz':
                yield from stream_jsonl_gz(path)
            elif suffix == '.parquet':
                yield from stream_parquet(
                    path,
                    batch_rows=self.parquet_batch_rows,
                    arrow_num_threads=self.arrow_num_threads,
                )
            else:
                yield from stream_jsonl(path)
        except Exception as exc:
            logger.warning("Failed to stream shard %s: %s", path, exc)

    def _iter_entry(self, entry: ManifestEntry) -> Iterator[tuple[ManifestEntry, Dict[str, object]]]:
        for path in self._iter_entry_paths(entry):
            for record in self._stream_records(path):
                record.setdefault('path', path)
                yield entry, record

    def _generate_sample(
        self,
        entry: ManifestEntry,
        payload: Dict[str, object],
    ) -> Optional[Dict[str, object]]:
        text = payload.get('text', '')
        if not isinstance(text, str) or not text:
            return None
        declared = payload.get('type', entry.type)
        sample_type = _canonicalize_type(declared, payload.get('path', entry.resolved_path), entry.dataset_id)
        if sample_type == 'math_tool':
            text = tool_use.traces.maybe_inject_tool_result(text)

        sample_id = payload.get('sample_id')
        dataset_key = entry.dataset_id or entry.resolved_path
        if not sample_id:
            digest = hashlib.sha1(
                f"{payload.get('path', entry.resolved_path)}:{payload.get('sha', '')}:{text[:200]}".encode('utf-8')
            ).hexdigest()
            sample_id = f'auto_{digest}'
            self._auto_sample_counts[dataset_key] += 1

        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.seq_len,
            padding='max_length',
            return_tensors='pt',
        )

        del text

        st = sample_type
        if st == 'math':
            st = 'math_tool'
        elif st in {'code_tool', 'coding', 'programming'}:
            st = 'code'

        item: Dict[str, object] = {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'sample_type': st,
            'sample_id': sample_id,
            'type': st,
        }

        self.dataset_counts[dataset_key] += 1
        self.sample_type_counts[st] += 1

        directory: Optional[str] = None
        source_tag = 'disabled'
        if self.teacher_mode == 'precompute' and self.teacher_logits_dir:
            directory = self.teacher_logits_dir
            source_tag = 'precompute'
        elif self.hybrid and st in {'math_tool', 'code'}:
            directory = self.math_logits_dir if st == 'math_tool' else self.code_logits_dir
            source_tag = st

        if directory:
            logits, status = self._load_teacher_logits(sample_id, directory, source_tag)
            self._epoch_total += 1
            if status in {'missing', 'invalid'}:
                self._epoch_missing += 1
            if logits is not None:
                item['teacher_logits'] = logits
            if status == 'missing' and self.hybrid and self.hybrid_strict and source_tag in {'math_tool', 'code'}:
                warn_key = f"{self.split}:{source_tag}"
                if warn_key not in self._hybrid_missing_warned_types:
                    self._hybrid_missing_warned_types.add(warn_key)
                    logger.warning(
                        "Hybrid strict mode: missing %s logits detected on split=%s; disabling KD for affected batches",
                        source_tag,
                        self.split,
                    )
                item['hybrid_skip_kd'] = True
            item['teacher_status'] = status

        return item

    def __iter__(self) -> Iterator[Dict[str, object]]:
        worker = get_worker_info()
        if worker is None:
            entries = self.entries
        else:
            entries = self.entries[worker.id :: worker.num_workers]

        sources = [self._iter_entry(entry) for entry in entries]
        for entry, payload in interleave(sources, per_source_buffer=self.per_source_buffer):
            sample = self._generate_sample(entry, payload)
            if sample is not None:
                yield sample

        if not self.auto_sample_counts:
            self.auto_sample_counts = dict(self._auto_sample_counts)

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
        logits = torch.as_tensor(logits).float()
        if logits.dim() != 2:
            logits = logits.view(-1, logits.shape[-1])
        seq_len, vocab_size = logits.shape
        if seq_len < self.seq_len:
            logits = F.pad(logits, (0, 0, 0, self.seq_len - seq_len))
        elif seq_len > self.seq_len:
            logits = logits[: self.seq_len]
        target_vocab = int(getattr(self.tokenizer, "vocab_size", vocab_size) or vocab_size)
        if vocab_size < target_vocab:
            logits = F.pad(logits, (0, target_vocab - vocab_size))
        elif vocab_size > target_vocab:
            logits = logits[:, :target_vocab]
        return logits

    def _load_teacher_logits(self, sample_id: str, directory: str, tag: str) -> tuple[Optional[torch.Tensor], str]:
        cache_key = (directory, sample_id)
        cached = self._teacher_cache.get(cache_key)
        if cached is not None:
            return cached, "cached"
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
                    logger.warning(
                        "Teacher logits at %s have invalid shape %s; expected [T, V]; skipping",
                        path,
                        tuple(logits.shape),
                    )
                return None, "invalid"

            logits = self._normalize_teacher_logits(logits)
            self._teacher_cache.put(cache_key, logits)
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
