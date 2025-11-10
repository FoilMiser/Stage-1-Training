# train.py
"""Training loop implementation for Stage-1 KD (HuggingFace-friendly)."""
from __future__ import annotations

import json
import math
import os
import shutil
import signal
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from . import losses
from .eval import run_validation
from .gcs_io import ensure_local_dir, local_to_gcs
from .runtime_setup import enable_flash_attn_if_available
from .utils import (
    AnnealingSchedule,
    CosineLRSchedule,
    RunMetadata,
    configure_logging,
    json_log,
)

logger = configure_logging()


class MetricsStream:
    """Append-only metrics writer that mirrors to GCS."""

    def __init__(self, path: Path, run_uri: Optional[str]) -> None:
        self.path = path
        self.run_uri = run_uri.rstrip("/") if run_uri else None
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, payload: Dict[str, object]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self.flush()

    def flush(self) -> None:
        if self.run_uri and self.path.exists():
            local_to_gcs(str(self.path), f"{self.run_uri}/metrics.jsonl")


class SignalHandler:
    """Simple RAII helper that installs signal handlers."""

    def __init__(self, trainer: "Trainer") -> None:
        self.trainer = trainer
        self._orig_handlers: Dict[int, object] = {}

    def __enter__(self) -> "SignalHandler":
        for sig in (signal.SIGTERM, signal.SIGINT):
            self._orig_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self.trainer.handle_interrupt)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for sig, handler in self._orig_handlers.items():
            if handler is not None:
                signal.signal(sig, handler)


@dataclass
class StepResult:
    success: bool
    metrics: Optional[Dict[str, float]]
    tokens: int
    examples: int
    grad_norm: float


class Trainer:
    """Main training orchestrator (now compatible with HF AutoModelForCausalLM)."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        device: torch.device,
        output_dir: str,
        output_gcs_uri: Optional[str],
        run_id: str,
        lr: float,
        betas: tuple[float, float],
        weight_decay: float,
        warmup_steps: int,
        max_steps: int,
        kd_temperature: float,
        kd_alpha_schedule: AnnealingSchedule,
        ce_beta_schedule: AnnealingSchedule,
        logit_l2_gamma_schedule: AnnealingSchedule,
        logit_reference: Optional[torch.Tensor] = None,
        precision: str = "bfloat16",
        lm_teacher: Optional[object] = None,
        teacher_mode: str = "online",
        teacher_logits_dir: Optional[str] = None,
        math_logits_dir: Optional[str] = None,
        code_logits_dir: Optional[str] = None,
        hybrid: bool = False,
        hybrid_strict: bool = False,
        eval_every: int = 1000,
        save_every: int = 2000,
        grad_accum_steps: int = 1,
        metrics_interval: int = 100,
        limit_batches: int = 0,
        early_stop_ppl: float = 0.0,
        dry_run: bool = False,
        # --- NEW HF-friendly knobs ---
        hf_tokenizer: Optional[object] = None,
        save_hf_snapshots: bool = False,
        hf_max_shard_size: str = "1GB",
    ) -> None:
        # Core wiring
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = ensure_local_dir(output_dir)
        self.output_gcs_uri = output_gcs_uri.rstrip("/") if output_gcs_uri else None
        self.run_id = run_id
        self.optimizer = AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.scheduler = CosineLRSchedule(base_lr=lr, warmup_steps=warmup_steps, total_steps=max_steps)
        self.max_steps = max_steps
        self.kd_temperature = kd_temperature
        self.kd_alpha_schedule = kd_alpha_schedule
        self.ce_beta_schedule = ce_beta_schedule
        self.logit_l2_gamma_schedule = logit_l2_gamma_schedule
        self.logit_reference = logit_reference
        enable_flash_attn_if_available(log=False)

        # AMP dtype selection
        if precision == "bfloat16" and torch.cuda.is_available():
            self.amp_dtype = torch.bfloat16
        elif precision == "fp16":
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.float32

        # Use new torch.amp.GradScaler API (silences deprecation warning)
        self.scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16" and torch.cuda.is_available()))

        # KD / hybrid fields
        self.lm_teacher = lm_teacher
        self.teacher_mode = teacher_mode
        self.teacher_logits_dir = teacher_logits_dir
        self.math_logits_dir = math_logits_dir.rstrip("/") if math_logits_dir else None
        self.code_logits_dir = code_logits_dir.rstrip("/") if code_logits_dir else None
        self.hybrid = hybrid
        self.hybrid_strict = hybrid_strict

        # HF saving / tokenizer references
        self.hf_tokenizer = hf_tokenizer
        self.save_hf_snapshots = save_hf_snapshots or bool(int(os.environ.get("STAGE1_SAVE_HF", "0")))
        self.hf_max_shard_size = hf_max_shard_size

        # Housekeeping
        self.eval_every = eval_every
        self.save_every = save_every
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.current_grad_accum_steps = self.grad_accum_steps
        self.metrics_interval = max(1, metrics_interval)
        self.limit_batches = limit_batches
        self.early_stop_ppl = max(0.0, early_stop_ppl)
        self.dry_run = dry_run
        self._run_uri = f"{self.output_gcs_uri}/{self.run_id}" if self.output_gcs_uri else None
        self._metrics_stream = MetricsStream(Path(self.output_dir) / "metrics.jsonl", self._run_uri)
        self._val_metrics_path = Path(self.output_dir) / "val_metrics.jsonl"
        self._provenance_synced = False
        self._global_step = 0
        self._best_ppl = float("inf")
        self._last_metrics: Optional[Dict[str, float]] = None
        self._consecutive_non_finite = 0
        self._oom_retry_active = False
        self._last_batch_shapes: Dict[str, Iterable[int]] = {}
        self._batch_iterator = self._infinite_batches()
        self._retry_batches: deque[Dict[str, torch.Tensor]] = deque()
        self._bad_teacher_shape_warned = False
        self._vocab_mismatch_warned = False

        # --- NEW: early, explicit vocab sanity gate (env-driven) ---
        target_vocab_env = os.environ.get("STAGE1_TOKENIZER_VOCAB_SIZE", "").strip()
        if target_vocab_env:
            try:
                tgt = int(target_vocab_env.replace("_", ""))
                student_vocab = self._student_vocab_size()
                if student_vocab != tgt:
                    raise RuntimeError(
                        f"Vocab mismatch: tokenizer={tgt} vs student={student_vocab}. "
                        "Ensure student was resized in surgery or switch dataset tokenizer."
                    )
            except ValueError:
                logger.warning("Invalid STAGE1_TOKENIZER_VOCAB_SIZE=%r; skipping strict vocab check", target_vocab_env)

    # ---------------------- helpers ----------------------
    def _student_vocab_size(self) -> int:
        # Prefer lm_head if present; else fall back to embedding
        if hasattr(self.model, "lm_head") and hasattr(self.model.lm_head, "out_features"):
            return int(self.model.lm_head.out_features)
        if hasattr(self.model, "get_input_embeddings"):
            try:
                return int(self.model.get_input_embeddings().num_embeddings)
            except Exception:
                pass
        if hasattr(self.model, "embed") and hasattr(self.model.embed, "num_embeddings"):
            return int(self.model.embed.num_embeddings)
        return -1

    def _forward_student_logits(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Robustly obtain logits from either:
          * a custom student that returns logits directly when called, or
          * a HuggingFace AutoModelForCausalLM that returns a ModelOutput with .logits
        """
        try:
            out = self.model(input_ids)  # legacy path (returns logits tensor)
        except TypeError:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)  # HF-style
        if isinstance(out, torch.Tensor):
            logits = out
        elif hasattr(out, "logits"):
            logits = out.logits
        elif isinstance(out, (list, tuple)) and len(out) > 0:
            logits = out[0]
        else:
            raise RuntimeError("Student model forward did not return logits")
        return logits

    @property
    def global_step(self) -> int:
        return self._global_step

    def _infinite_batches(self) -> Iterator[Dict[str, torch.Tensor]]:
        dataset = getattr(self.train_loader, "dataset", None)
        if hasattr(dataset, "begin_epoch"):
            dataset.begin_epoch()
        data_iter = iter(self.train_loader)
        batches_in_epoch = 0
        while True:
            try:
                batch = next(data_iter)
            except StopIteration:
                self._on_epoch_end(dataset)
                data_iter = iter(self.train_loader)
                batches_in_epoch = 0
                continue
            batches_in_epoch += 1
            if self.limit_batches and batches_in_epoch > self.limit_batches:
                self._on_epoch_end(dataset)
                data_iter = iter(self.train_loader)
                batches_in_epoch = 0
                continue
            yield batch

    def _on_epoch_end(self, dataset: Optional[object]) -> None:
        if dataset and hasattr(dataset, "epoch_teacher_stats"):
            total, missing = dataset.epoch_teacher_stats()
            if total:
                ratio = missing / float(total)
                if ratio > 0.1:
                    logger.warning(
                        "KD logits missing for %.2f%% of samples this epoch", ratio * 100.0
                    )
        if dataset and hasattr(dataset, "begin_epoch"):
            dataset.begin_epoch()

    def handle_interrupt(self, signum, frame) -> None:  # pragma: no cover - signal handler
        logger.warning("Received signal %s; saving last checkpoint before exit", signum)
        metrics = self._last_metrics or {}
        self._save_checkpoint("last.pt", self.global_step, self._best_ppl, metrics)
        self._metrics_stream.flush()
        raise SystemExit(0)

    # ---------- HF snapshot helpers ----------
    def _mirror_dir_to_gcs(self, local_dir: Path, gcs_prefix: str) -> None:
        if not self._run_uri:
            return
        local_dir = Path(local_dir)
        for root, _, files in os.walk(local_dir):
            for fn in files:
                full = Path(root) / fn
                rel = str(full.relative_to(local_dir)).replace("\\", "/")
                local_to_gcs(str(full), f"{gcs_prefix}/{rel}")

    def _save_hf_snapshot(self, tag: str) -> None:
        """Save HF-style snapshot alongside .pt checkpoints."""
        if not self.save_hf_snapshots:
            return
        out_dir = Path(self.output_dir) / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Safe serialization + sharding for large models
            self.model.save_pretrained(
                out_dir,
                safe_serialization=True,
                max_shard_size=self.hf_max_shard_size,
            )
            if self.hf_tokenizer is not None:
                # Ensure pad token is persisted if it was set to eos
                if getattr(self.hf_tokenizer, "pad_token", None) is None and getattr(self.hf_tokenizer, "eos_token", None):
                    self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
                self.hf_tokenizer.save_pretrained(out_dir)
        except Exception as e:
            logger.warning("Failed to save HF snapshot %s: %s", tag, e)
        if self._run_uri:
            self._mirror_dir_to_gcs(out_dir, f"{self._run_uri}/{tag}")

    def _save_checkpoint(self, name: str, step: int, val_ppl: float, losses_dict: Dict[str, float]) -> None:
        self._ensure_provenance_artifacts()
        path = Path(self.output_dir) / name
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": step,
        }
        torch.save(state, path)
        if self._run_uri:
            local_to_gcs(str(path), f"{self._run_uri}/{name}")
        # Save a parallel HF snapshot named after this checkpoint (e.g., hf_best/, hf_last/)
        tag = f"hf_{Path(name).stem}"
        self._save_hf_snapshot(tag)

        metadata = RunMetadata(step=step, val_ppl=val_ppl, losses=losses_dict, frozen_blocks=("block_0", "block_1"))
        run_meta_path = Path(self.output_dir) / "run_meta.json"
        with run_meta_path.open("w", encoding="utf-8") as f:
            f.write(metadata.to_json())
        if self._run_uri:
            local_to_gcs(str(run_meta_path), f"{self._run_uri}/run_meta.json")
            self._upload_frozen_mask()
        logger.info("Saved checkpoint %s", path)

    def _upload_frozen_mask(self) -> None:
        if not self._run_uri:
            return
        mask_path = Path(self.output_dir) / "frozen_mask.json"
        if mask_path.exists():
            local_to_gcs(str(mask_path), f"{self._run_uri}/frozen_mask.json")

    def _ensure_provenance_artifacts(self) -> None:
        if self._provenance_synced:
            return
        source_dir = Path(os.environ.get("STAGE1_DATA_PROVENANCE_DIR", self.output_dir))
        target_dir = Path(self.output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in ("data_readiness.json", "datasets_yaml_snapshot.yaml", "manifest_snapshot.jsonl"):
            src = source_dir / name
            dst = target_dir / name
            if src.exists() and not dst.exists():
                shutil.copy(src, dst)
        self._provenance_synced = True

    def _write_validation_metrics(self, payload: Dict[str, object]) -> None:
        self._val_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self._val_metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        if self._run_uri:
            local_to_gcs(str(self._val_metrics_path), f"{self._run_uri}/val_metrics.jsonl")

    def _normalize_teacher_tensor(
        self, tensor: torch.Tensor, target_seq_len: int, target_vocab: int
    ) -> Optional[torch.Tensor]:
        logits = torch.as_tensor(tensor, device=self.device).float()
        if logits.dim() == 3:
            if logits.size(0) == 1:
                logits = logits.squeeze(0)
            else:
                logits = logits.reshape(-1, logits.size(-1))
        if logits.dim() != 2:
            if not self._bad_teacher_shape_warned:
                logger.warning(
                    "Teacher logits have invalid shape %s; expected [T, V]; dropping from KD",
                    tuple(logits.shape),
                )
                self._bad_teacher_shape_warned = True
            return None
        current_len = logits.size(0)
        if current_len != target_seq_len:
            if current_len > target_seq_len:
                delta = current_len - target_seq_len
                start = delta // 2
                logits = logits[start : start + target_seq_len]
            else:
                diff = target_seq_len - current_len
                pad_before = diff // 2
                pad_after = diff - pad_before
                logits = F.pad(logits, (0, 0, pad_before, pad_after))
        vocab = logits.size(1)
        if vocab != target_vocab:
            if not self._vocab_mismatch_warned:
                logger.warning(
                    "Teacher vocab mismatch detected (student=%d, teacher=%d); skipping KD logits",
                    target_vocab,
                    vocab,
                )
                self._vocab_mismatch_warned = True
            return None
        return logits

    def _collect_teacher_logits(
        self,
        batch: Dict[str, object],
        input_ids: torch.Tensor,
        student_logits: torch.Tensor,
    ) -> list[Optional[torch.Tensor]]:
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        vocab_size = student_logits.size(-1)
        if self.hybrid and batch.get("hybrid_skip_kd"):
            return [None] * batch_size
        teacher_list: list[Optional[torch.Tensor]] = [None] * batch_size
        precomputed = batch.get("teacher_logits")
        if isinstance(precomputed, torch.Tensor):
            precomputed = [precomputed[i] for i in range(min(precomputed.size(0), batch_size))]
        if isinstance(precomputed, list):
            for idx, tensor in enumerate(precomputed[:batch_size]):
                if tensor is None:
                    continue
                normalized = self._normalize_teacher_tensor(tensor, seq_len, vocab_size)
                if normalized is not None:
                    teacher_list[idx] = normalized
        sample_types = batch.get("sample_types") or ["lm"] * batch_size
        if self.lm_teacher is not None:
            lm_indices = [
                idx
                for idx, sample_type in enumerate(sample_types[:batch_size])
                if sample_type == "lm" and teacher_list[idx] is None
            ]
            if lm_indices:
                with torch.inference_mode():
                    teacher_inputs = input_ids[lm_indices]
                    lm_logits = self.lm_teacher.logits(teacher_inputs)
                for offset, batch_idx in enumerate(lm_indices):
                    normalized = self._normalize_teacher_tensor(lm_logits[offset], seq_len, vocab_size)
                    if normalized is not None:
                        teacher_list[batch_idx] = normalized
        return teacher_list

    def _dump_crash(self, reason: str) -> None:
        path = Path(self.output_dir) / "crash_dump.pt"
        payload = {
            "reason": reason,
            "step": self.global_step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "batch_shapes": self._last_batch_shapes,
        }
        torch.save(payload, path)
        if self._run_uri:
            local_to_gcs(str(path), f"{self._run_uri}/crash_dump.pt")

    def _handle_non_finite(self, what: str) -> StepResult:
        self._consecutive_non_finite += 1
        if self._consecutive_non_finite == 1:
            logger.warning("Detected non-finite %s; skipping update", what)
            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler.is_enabled():
                self.scaler.update()
            return StepResult(False, None, 0, 0, 0.0)
        self._dump_crash(f"non_finite_{what}")
        raise RuntimeError(f"Encountered consecutive non-finite {what}; aborting")

    def _handle_oom(self, err: RuntimeError) -> StepResult:
        if "out of memory" not in str(err).lower() or not torch.cuda.is_available():
            raise err
        torch.cuda.empty_cache()
        if not self._oom_retry_active:
            new_accum = max(1, self.current_grad_accum_steps // 2)
            if new_accum == self.current_grad_accum_steps and new_accum == 1:
                self._dump_crash("cuda_oom")
                raise RuntimeError("CUDA OOM even at grad_accum_steps=1; aborting") from err
            logger.warning(
                "CUDA OOM detected; reducing grad_accum_steps from %d to %d and retrying once",
                self.current_grad_accum_steps,
                new_accum,
            )
            self.current_grad_accum_steps = new_accum
            self._oom_retry_active = True
            self.optimizer.zero_grad(set_to_none=True)
            return StepResult(False, None, 0, 0, 0.0)
        self._dump_crash("cuda_oom")
        raise RuntimeError("Repeated CUDA OOM during the same step; aborting") from err

    def _step(self) -> StepResult:
        self.optimizer.zero_grad(set_to_none=True)
        use_autocast = self.device.type != "cpu"
        total_tokens = 0
        total_examples = 0
        loss_ce = 0.0
        loss_kd = 0.0
        loss_l2 = 0.0
        loss_total = 0.0
        step_start = time.perf_counter()
        current_batches: deque[Dict[str, object]] = deque()
        try:
            for _ in range(self.current_grad_accum_steps):
                if self._retry_batches:
                    batch = self._retry_batches.popleft()
                else:
                    batch = next(self._batch_iterator)
                current_batches.append(batch)
                self._last_batch_shapes = {
                    key: tuple(value.shape)
                    for key, value in batch.items()
                    if hasattr(value, "shape")
                }

                input_ids = batch["input_ids"].to(self.device)  # shape [B, T]
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                # Labels are the next-token targets; student predicts on tokens[:-1]
                labels = input_ids[:, 1:].contiguous()
                student_inputs = input_ids[:, :-1]
                student_mask = attention_mask[:, :-1] if attention_mask is not None else None

                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=use_autocast):
                    logits = self._forward_student_logits(student_inputs, student_mask)
                    teacher_logits_list = self._collect_teacher_logits(batch, input_ids, logits)

                    ce = losses.ce_loss(logits, labels)
                    kd = logits.new_zeros(())
                    available_indices = [idx for idx, v in enumerate(teacher_logits_list) if v is not None]
                    if available_indices:
                        gathered = torch.stack([teacher_logits_list[idx] for idx in available_indices], dim=0)
                        kd = losses.kd_loss(
                            logits[available_indices],
                            gathered[:, :-1, :],  # align with student_inputs length
                            self.kd_temperature,
                        )
                    l2 = losses.logit_l2(logits, self.logit_reference)

                    alpha = self.kd_alpha_schedule.value(self.global_step, self.max_steps)
                    beta = self.ce_beta_schedule.value(self.global_step, self.max_steps)
                    gamma = self.logit_l2_gamma_schedule.value(self.global_step, self.max_steps)
                    total_loss = alpha * kd + beta * ce + gamma * l2

                if not torch.isfinite(total_loss):
                    return self._handle_non_finite("loss")

                loss_ce += float(ce.detach().cpu())
                loss_kd += float(kd.detach().cpu())
                loss_l2 += float(l2.detach().cpu())
                loss_total += float(total_loss.detach().cpu())

                step_loss = total_loss / self.current_grad_accum_steps
                try:
                    if self.scaler.is_enabled():
                        self.scaler.scale(step_loss).backward()
                    else:
                        step_loss.backward()
                except RuntimeError as err:
                    result = self._handle_oom(err)
                    if not result.success:
                        self._retry_batches = current_batches
                        return result

                tokens = int(student_inputs.shape[0] * student_inputs.shape[1])
                if student_mask is not None:
                    tokens = int(student_mask.sum().item())
                total_tokens += tokens
                total_examples += int(student_inputs.shape[0])

        except StopIteration:
            raise

        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        if not math.isfinite(float(grad_norm)):
            return self._handle_non_finite("grad")

        overflow = False
        if self.scaler.is_enabled():
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            scale_after = self.scaler.get_scale()
            if scale_after < scale_before:
                overflow = True
                logger.warning(
                    "GradScaler overflow detected (scale %.3e -> %.3e); skipping optimizer step",
                    scale_before,
                    scale_after,
                )

        if not self.scaler.is_enabled() or not overflow:
            if not self.scaler.is_enabled():
                self.optimizer.step()
            lr = self.scheduler.value(self.global_step)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            self._consecutive_non_finite = 0
            self._oom_retry_active = False
            self._retry_batches.clear()
            self.optimizer.zero_grad(set_to_none=True)
            elapsed = max(1e-6, time.perf_counter() - step_start)
            metrics = {
                "global_step": self.global_step + 1,
                "train_loss": loss_total / self.current_grad_accum_steps,
                "ce_loss": loss_ce / self.current_grad_accum_steps,
                "kd_loss": loss_kd / self.current_grad_accum_steps,
                "logit_l2": loss_l2 / self.current_grad_accum_steps,
                "lr": lr,
                "tokens_per_sec": total_tokens / elapsed,
                "examples_per_sec": total_examples / elapsed,
                "grad_norm": float(grad_norm),
                "gpu_mem_alloc_MB": self._gpu_mem_mb(),
            }
            self._last_metrics = metrics
            return StepResult(True, metrics, total_tokens, total_examples, float(grad_norm))

        self.optimizer.zero_grad(set_to_none=True)
        self._retry_batches = current_batches
        return StepResult(False, None, total_tokens, total_examples, float(grad_norm))

    def _gpu_mem_mb(self) -> float:
        if torch.cuda.is_available() and self.device.type == "cuda":
            return torch.cuda.memory_allocated(self.device) / 1_000_000.0
        return 0.0

    def _maybe_eval(self) -> float:
        if self.val_loader is None:
            return float("inf")
        metrics = run_validation(self.model, self.val_loader)
        payload = {"step": self.global_step, **metrics}
        logger.info("Validation metrics at step %d: %s", self.global_step, json.dumps(metrics))
        json_log(logger, {"global_step": self.global_step, **metrics})
        self._write_validation_metrics(payload)
        val_ppl = float(metrics.get("perplexity", float("inf")))
        metrics_record = dict(self._last_metrics or {})
        metrics_record["val_ppl"] = val_ppl
        metrics_record["global_step"] = self.global_step
        self._metrics_stream.append(metrics_record)
        return val_ppl

    def train(self) -> None:
        dry_batches_seen = 0
        dry_steps_done = 0
        with SignalHandler(self):
            while self.global_step < self.max_steps:
                result = self._step()
                if not result.success:
                    continue
                self._global_step += 1

                if result.metrics:
                    should_log = self.global_step == 1 or (self.global_step % self.metrics_interval == 0)
                    if should_log:
                        json_log(logger, result.metrics)
                        self._metrics_stream.append(result.metrics)

                dry_steps_done += 1
                dry_batches_seen += self.current_grad_accum_steps

                if self.eval_every and self.val_loader is not None and self.global_step % self.eval_every == 0:
                    val_ppl = self._maybe_eval()
                    if val_ppl < self._best_ppl:
                        self._best_ppl = val_ppl
                        self._save_checkpoint("best.pt", self.global_step, val_ppl, self._last_metrics or {})
                    if self.early_stop_ppl and val_ppl <= self.early_stop_ppl:
                        logger.info("Stopping early due to val perplexity %.3f", val_ppl)
                        break

                if self.save_every and self.global_step % self.save_every == 0:
                    self._save_checkpoint("last.pt", self.global_step, self._best_ppl, self._last_metrics or {})

                if self.dry_run and dry_steps_done >= 1:
                    while dry_batches_seen < max(2, self.current_grad_accum_steps):
                        try:
                            next(self._batch_iterator)
                            dry_batches_seen += 1
                        except StopIteration:
                            break
                    if self.val_loader is not None:
                        self._maybe_eval()
                    self._save_checkpoint("last.pt", self.global_step, self._best_ppl, self._last_metrics or {})
                    logger.info("Dry run complete; exiting after sanity checks")
                    break

        # Final HF snapshot for convenience
        self._save_hf_snapshot("hf_final")
        self._metrics_stream.flush()
