#!/usr/bin/env python3
"""
CLI for Stage-1 Hybrid KD — HF-style student folder + split manifests.

- No vocab resizing here (assumed done externally in model surgery).
- Optional strict vocab check against the dataset tokenizer size.
- Works with hybrid KD (LM+tool), online teacher, or precomputed logits.
- Accepts --lm-manifest / --mc-manifest; falls back to --dataset-manifest.

Usage (Vertex runner calls this):
  python -m stage1_hybrid.cli --student-model-path /opt/models/student ...
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple, List

from torch.utils.data import DataLoader

# ---- Project modules (unchanged import paths expected in your repo) ----
from . import data
from .model_init import initialize_student, apply_grad_checkpointing
from .prep import (
    ensure_toolkit,
    load_datasets_yaml,
    prepare_if_needed,
    normalize_gcs_uri,
    upload_pipeline_logs,
)
from .runtime_setup import login_hf, enable_flash_attn_if_available, install_flash_attn_from_gcs
from .teacher import TeacherConfig, TeacherWrapper
from .train import Trainer
from .utils import (
    configure_logging,
    set_seed,
    AnnealingSchedule,
    detect_training_device,
    json_log,
)

logger = configure_logging()

# ----------------------------
# Defaults
# ----------------------------
DEFAULTS = dict(
    # IO / logging
    output_gcs_uri="gs://liquid-llm-bucket-2/stage1/Checkpoints/vertex-runs",
    logs_gcs_uri="gs://liquid-llm-bucket-2/logs/stage1_console/",
    tb_gcs_uri="gs://liquid-llm-bucket-2/logs/tb/",

    # Data / manifests
    dataset_manifest="gs://liquid-llm-bucket-2/datasets/stage1/manifests/stage1_lm.filesliced.jsonl",
    lm_manifest="",            # optional: used instead of dataset_manifest
    mc_manifest="",            # optional: extra manifest for math/code
    val_manifest="",

    # Teachers / KD mode
    teacher_mode="precompute",                # online | precompute
    lm_teacher_id="meta-llama/Llama-3.2-3B",
    mc_teacher_id="meta-llama/Llama-3.1-8B",
    teacher_logits_dir="gs://liquid-llm-bucket-2/teacher/llama-3.2-3b/logits",
    math_logits_dir="gs://liquid-llm-bucket-2/teacher/llama-3.1-8b/logits/math/",
    code_logits_dir="gs://liquid-llm-bucket-2/teacher/llama-3.1-8b/logits/code/",
    hybrid=True,
    hybrid_strict=False,
    teacher_cache_dir="",      # optional cache directory for teacher weights

    # Tokenization / sequence
    dataset_tokenizer_id="",                  # default to lm_teacher_id if empty
    seq_len=1024,
    tool_use_ratio=0.08,

    # Optimization
    precision="bfloat16",                     # compute dtype: bfloat16 | fp16 | fp32
    lr=2.5e-4,
    weight_decay=0.1,
    betas="0.9,0.95",
    warmup_steps=3000,
    max_steps=120000,
    eval_every=1000,
    save_every=2000,
    grad_checkpoint=True,
    grad_accum_steps=1,
    batch_size=1,
    batch_per_device=1,                       # reserved for DDP setups
    num_workers=4,
    prefetch_factor=2,
    metrics_interval=100,
    early_stop_ppl=0.0,

    # KD schedules
    kd_temperature=2.0,
    kd_alpha_start=0.7,
    kd_alpha_end=0.4,
    kd_anneal_pct=0.3,
    keep_old_logit_l2=0.1,
    keep_old_logit_l2_fade_step=30000,

    # FlashAttention wheel (optional)
    fa_wheel_gcs="gs://liquid-llm-bucket-2/FlashAttention/flash_attn-2.8.3+cu12torch2.4cxx11abiTRUE-cp310-cp310-linux_x86_64.whl",

    # Prep toolkit
    prep_toolkit_zip_uri="gs://liquid-llm-bucket-2/sandbox/preprocess-toolkit/preprocess-toolkit-stage1-1-0.zip",
    prep_extract_dir="/opt/preprocess-toolkit",
    prep_install_requirements=True,
    prep_timeout_s=0,
    prepare_data="auto",                      # auto | skip | force
    prep_continue_on_error=True,              # NEW: do not crash job if a pipeline fails
    prep_tail_logs=200,                       # NEW: how many lines to tail from failed logs

    # ---- Student model ----
    student_model_path="",                    # local HF dir (unzipped student.zip) or Hub ID
    student_trust_remote_code=False,
    student_dtype="bfloat16",                 # bfloat16 | fp16 | fp32
    torch_dtype="",                           # alias from runner; if set, overrides student_dtype
    assert_vocab_match=True,                  # verify student vocab == dataset tokenizer size
    save_hf_snapshots=True,
    hf_max_shard_size="1GB",

    # Deepspeed (optional; logged only)
    deepspeed="",                             # path to DS JSON (optional; not wired unless Trainer supports it)

    # Misc
    seed=1337,
    dry_run=False,
)

# ----------------------------
# Arg parsing helpers
# ----------------------------
def _str_to_bool(v) -> bool:
    if isinstance(v, bool):
        return True if v else False
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {v}")

def _parse_betas(betas: str) -> Tuple[float, float]:
    parts = [float(x) for x in betas.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("betas must be 'beta1,beta2'")
    return parts[0], parts[1]


def _env_or_default(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value is not None else default


def _env_bool(name: str, default: str) -> bool:
    try:
        return _str_to_bool(_env_or_default(name, default))
    except argparse.ArgumentTypeError:
        raise argparse.ArgumentTypeError(f"Invalid boolean for {name}")


def _parse_size_to_int(raw: str) -> int:
    if isinstance(raw, (int, float)):
        return int(raw)
    s = str(raw).strip().lower()
    if not s:
        return 0
    import re as _re

    match = _re.fullmatch(r"(\d+(?:\.\d+)?)([kmgt]i?b?)?", s)
    if not match:
        raise argparse.ArgumentTypeError(f"Invalid size: {raw}")
    value = float(match.group(1))
    unit = match.group(2) or ""
    unit = unit.lower()
    multipliers = {
        "": 1,
        "k": 1024,
        "kb": 1024,
        "ki": 1024,
        "kib": 1024,
        "m": 1024 ** 2,
        "mb": 1024 ** 2,
        "mi": 1024 ** 2,
        "mib": 1024 ** 2,
        "g": 1024 ** 3,
        "gb": 1024 ** 3,
        "gi": 1024 ** 3,
        "gib": 1024 ** 3,
        "t": 1024 ** 4,
        "tb": 1024 ** 4,
        "ti": 1024 ** 4,
        "tib": 1024 ** 4,
    }
    factor = multipliers.get(unit)
    if factor is None:
        raise argparse.ArgumentTypeError(f"Unsupported size unit in {raw}")
    return int(value * factor)

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("Liquid-LLM Stage-1 CLI (HF student)")

    # Flags we will add with explicit typing/choices (skip these in the generic loop)
    explicitly_typed = {
        "teacher_mode",
        "precision",
        "student_dtype",
        "hybrid",
        "hybrid_strict",
        "student_trust_remote_code",
        "assert_vocab_match",
        "save_hf_snapshots",
        "batch_size",
        "num_workers",
        "prefetch_factor",
        "grad_accum_steps",
        "metrics_interval",
        "eval_every",
        "save_every",
        "warmup_steps",
        "max_steps",
        "seq_len",
        "keep_old_logit_l2_fade_step",
        "lr",
        "weight_decay",
        "kd_temperature",
        "kd_alpha_start",
        "kd_alpha_end",
        "kd_anneal_pct",
        "tool_use_ratio",
        "grad_checkpoint",
        "dry_run",
        "seed",
        "prep_continue_on_error",
        "prep_tail_logs",
    }

    # First add all defaults generically, except those we type below
    for k, v in DEFAULTS.items():
        if k in explicitly_typed:
            continue
        p.add_argument(f"--{k}".replace("_", "-"), default=v)

    # Now add the typed/validated ones (no duplicates)
    p.add_argument("--teacher-mode", choices=["online", "precompute"], default=DEFAULTS["teacher_mode"])
    p.add_argument("--precision", choices=["bfloat16", "fp16", "fp32"], default=DEFAULTS["precision"])
    p.add_argument("--student-dtype", choices=["bfloat16", "fp16", "fp32"], default=DEFAULTS["student_dtype"])

    p.add_argument("--hybrid", type=_str_to_bool, default=DEFAULTS["hybrid"])
    p.add_argument("--hybrid-strict", type=_str_to_bool, default=DEFAULTS["hybrid_strict"])
    p.add_argument("--student-trust-remote-code", type=_str_to_bool, default=DEFAULTS["student_trust_remote_code"])
    p.add_argument("--assert-vocab-match", type=_str_to_bool, default=DEFAULTS["assert_vocab_match"])
    p.add_argument("--save-hf-snapshots", type=_str_to_bool, default=DEFAULTS["save_hf_snapshots"])

    p.add_argument("--batch-size", type=int, default=int(DEFAULTS["batch_size"]))
    p.add_argument("--num-workers", type=int, default=int(DEFAULTS["num_workers"]))
    p.add_argument("--prefetch-factor", type=int, default=int(DEFAULTS["prefetch_factor"]))
    p.add_argument("--grad-accum-steps", type=int, default=int(DEFAULTS["grad_accum_steps"]))
    p.add_argument("--metrics-interval", type=int, default=int(DEFAULTS["metrics_interval"]))
    p.add_argument("--eval-every", type=int, default=int(DEFAULTS["eval_every"]))
    p.add_argument("--save-every", type=int, default=int(DEFAULTS["save_every"]))
    p.add_argument("--warmup-steps", type=int, default=int(DEFAULTS["warmup_steps"]))
    p.add_argument("--max-steps", type=int, default=int(DEFAULTS["max_steps"]))
    p.add_argument("--seq-len", type=int, default=int(DEFAULTS["seq_len"]))
    p.add_argument("--keep-old-logit-l2-fade-step", type=int, default=int(DEFAULTS["keep_old_logit_l2_fade_step"]))
    p.add_argument("--lr", type=float, default=float(DEFAULTS["lr"]))
    p.add_argument("--weight-decay", type=float, default=float(DEFAULTS["weight_decay"]))
    p.add_argument("--kd-temperature", type=float, default=float(DEFAULTS["kd_temperature"]))
    p.add_argument("--kd-alpha-start", type=float, default=float(DEFAULTS["kd_alpha_start"]))
    p.add_argument("--kd-alpha-end", type=float, default=float(DEFAULTS["kd_alpha_end"]))
    p.add_argument("--kd-anneal-pct", type=float, default=float(DEFAULTS["kd_anneal_pct"]))
    p.add_argument("--tool-use-ratio", type=float, default=float(DEFAULTS["tool_use_ratio"]))
    p.add_argument("--grad-checkpoint", type=_str_to_bool, default=DEFAULTS["grad_checkpoint"])
    p.add_argument("--dry-run", type=_str_to_bool, default=DEFAULTS["dry_run"])
    p.add_argument("--seed", type=int, default=int(DEFAULTS["seed"]))

    # New explicit prep controls
    p.add_argument("--prep-continue-on-error", type=_str_to_bool, default=DEFAULTS["prep_continue_on_error"])
    p.add_argument("--prep-tail-logs", type=int, default=int(DEFAULTS["prep_tail_logs"]))

    # Streaming + dataloader controls
    p.add_argument("--streaming", type=_str_to_bool, default=_env_bool("STREAMING", "true"))
    p.add_argument("--dl-num-workers", type=int, default=int(_env_or_default("DL_NUM_WORKERS", "2")))
    p.add_argument("--dl-prefetch-factor", type=int, default=int(_env_or_default("DL_PREFETCH_FACTOR", "1")))
    p.add_argument("--parquet-batch-rows", type=int, default=int(_env_or_default("PARQUET_BATCH_ROWS", "1024")))
    p.add_argument("--data-max-open-files", type=int, default=int(_env_or_default("DATA_MAX_OPEN_FILES", "2")))
    p.add_argument(
        "--data-max-cache-bytes",
        default=_env_or_default("DATA_MAX_CACHE_BYTES", "3GiB"),
    )
    p.add_argument("--arrow-num-threads", type=int, default=int(_env_or_default("ARROW_NUM_THREADS", "1")))

    return p.parse_args(argv)

# ----------------------------
# Utilities (snapshots / banners)
# ----------------------------
def _detect_git_commit() -> Optional[str]:
    try:
        root = Path(__file__).resolve().parents
        for cand in root:
            if (cand / ".git").exists():
                res = subprocess.run(["git", "-C", str(cand), "rev-parse", "HEAD"], capture_output=True, text=True)
                if res.returncode == 0:
                    return res.stdout.strip()
                break
    except Exception:
        return None
    return None

def _write_args_snapshot(run_dir: str, args: argparse.Namespace, run_uri: Optional[str]) -> None:
    snap = vars(args).copy()
    snap["env"] = {
        "HF_TOKEN_PRESENT": "true" if (os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")) else "false",
        "GOOGLE_CLOUD_PROJECT": os.environ.get("GOOGLE_CLOUD_PROJECT"),
        "git_commit": _detect_git_commit() or "",
    }
    p = Path(run_dir) / "args_snapshot.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(snap, indent=2, sort_keys=True), encoding="utf-8")
    if run_uri:
        from .gcs_io import local_to_gcs
        local_to_gcs(str(p), f"{run_uri}/args_snapshot.json")

def _tail_and_upload_prep_logs(logs_dir: Path, tail_lines: int, logs_gcs_uri: str, run_id: str) -> None:
    """Best-effort: echo last N lines from each toolkit log and upload all logs to GCS."""
    try:
        if logs_dir.exists():
            for lf in sorted(logs_dir.glob("*.log")):
                try:
                    lines = lf.read_text(encoding="utf-8", errors="ignore").splitlines()
                    tail = "\n".join(lines[-max(1, tail_lines):])
                    logger.error("---- %s (last %d lines) ----\n%s\n-------------------------------", lf.name, tail_lines, tail)
                except Exception:
                    logger.exception("Failed to tail %s", lf)
            upload_pipeline_logs(logs_dir, logs_gcs_uri, run_id)
    except Exception:
        logger.exception("Failed while uploading/tailing prep logs")

def _augment_data_readiness(
    prep_summary: dict,
    dataset: data.ManifestDataset,
    manifest_snapshot: List[dict],
    toolkit_zip: str,
    tokenizer_id: str,
) -> dict:
    summary = dict(prep_summary or {})
    summary["toolkit_zip_uri"] = toolkit_zip
    summary["manifest_size"] = len(manifest_snapshot)
    summary["total_records"] = len(dataset)
    summary["tokenizer_id"] = tokenizer_id
    summary["tokenizer_vocab_size"] = int(getattr(dataset.tokenizer, "vocab_size", 0) or 0)
    datasets_summary: Dict[str, dict] = summary.setdefault("datasets", {})
    key_to_path: Dict[str, str] = {}
    for entry in dataset.entries:
        key = entry.dataset_id or entry.resolved_path
        key_to_path.setdefault(key, entry.resolved_path)
    for key, count in dataset.dataset_counts.items():
        ds_entry = datasets_summary.setdefault(key, {})
        ds_entry.setdefault("output", key_to_path.get(key))
        ds_entry["record_count"] = count
    return summary

def _write_data_artifacts(run_dir: str, data_readiness: dict, datasets_cfg: Dict[str, data.DatasetSpec], manifest_snapshot: List[dict]) -> None:
    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "data_readiness.json").write_text(json.dumps(data_readiness, indent=2, sort_keys=True), encoding="utf-8")
    yaml_snapshot = {
        job: {"in": spec.inp, "out": spec.out, "type": spec.dtype, "manifest": spec.manifest}
        for job, spec in datasets_cfg.items()
    }
    import yaml as _yaml
    (d / "datasets_yaml_snapshot.yaml").write_text(_yaml.safe_dump(yaml_snapshot, sort_keys=True), encoding="utf-8")
    with (d / "manifest_snapshot.jsonl").open("w", encoding="utf-8") as fh:
        for m in manifest_snapshot:
            fh.write(json.dumps(m, sort_keys=True) + "\n")

def _startup_banner(
    seed: int,
    device_info,
    precision: str,
    sdpa_enabled: bool,
    fa_installed: bool,
    toolkit_zip: str,
    prepare_mode: str,
    data_summary: dict,
    teacher_mode: str,
    dataset_tokenizer_id: str,
    hybrid: bool,
    lm_teacher_id: str,
    mc_teacher_id: str,
    math_logits_dir: Optional[str],
    code_logits_dir: Optional[str],
    kd_temperature: float,
    alpha_start: float,
    alpha_end: float,
    anneal_pct: float,
    grad_accum_steps: int,
    metrics_interval: int,
    dry_run: bool,
    student_model_path: str,
    lm_manifest: str,
    mc_manifest: str,
    deepspeed: str,
    teacher_cache_dir: str,
) -> None:
    logger.info("========== STAGE-1 STARTUP ==========")
    logger.info(
        "seed=%d | device=%s (%s) | precision=%s | sdpa=%s | flash_attn=%s",
        seed, device_info.name, device_info.device.type, precision,
        "on" if sdpa_enabled else "off",
        "installed" if fa_installed else "missing",
    )
    logger.info("prepare_data=%s | toolkit=%s", prepare_mode, toolkit_zip)
    for key, info in sorted((data_summary.get("datasets") or {}).items()):
        status = str(info.get("status", "ready")).upper()
        out_dir = info.get("output")
        count = info.get("record_count", "?")
        logger.info("  dataset=%s | status=%s | output=%s | count=%s", key, status, out_dir, count)
    logger.info("teacher_mode=%s | hybrid=%s | dataset_tokenizer_id=%s", teacher_mode, hybrid, dataset_tokenizer_id)
    logger.info("lm_teacher=%s | mc_teacher=%s | math_logits_dir=%s | code_logits_dir=%s",
                lm_teacher_id, mc_teacher_id, math_logits_dir, code_logits_dir)
    logger.info("kd_temp=%.2f | alpha_start=%.2f | alpha_end=%.2f | anneal_pct=%.2f",
                kd_temperature, alpha_start, alpha_end, anneal_pct)
    logger.info("student_model_path=%s", student_model_path)
    logger.info("lm_manifest=%s | mc_manifest=%s | deepspeed=%s | teacher_cache_dir=%s",
                lm_manifest or "<unset>", mc_manifest or "<unset>", deepspeed or "<unset>", teacher_cache_dir or "<unset>")
    logger.info("=====================================")

# ----------------------------
# Main
# ----------------------------
def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    set_seed(int(args.seed))
    login_hf()

    # Prefer HF_HOME over deprecated TRANSFORMERS_CACHE env (Transformers v5 notice)
    if os.environ.get("TRANSFORMERS_CACHE") and not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = os.environ["TRANSFORMERS_CACHE"]

    # Adopt TORCH_DTYPE env as alias for student dtype if provided
    env_torch_dtype = os.environ.get("TORCH_DTYPE", "").strip()
    if args.torch_dtype:
        args.student_dtype = str(args.torch_dtype)
    elif env_torch_dtype:
        args.student_dtype = env_torch_dtype

    # Teacher cache dir hint
    if args.teacher_cache_dir:
        os.environ.setdefault("TRANSFORMERS_CACHE", args.teacher_cache_dir)
        os.environ.setdefault("HF_HOME", args.teacher_cache_dir)

    # FlashAttention wheel (optional)
    fa_installed = install_flash_attn_from_gcs(args.fa_wheel_gcs)
    sdpa_enabled = enable_flash_attn_if_available(log=False)

    device_info = detect_training_device()
    run_id = os.environ.get("AIP_TRAINING_JOB_ID", "local")
    local_root = os.path.join("/tmp", "vertex_run", run_id)
    os.makedirs(local_root, exist_ok=True)
    os.environ["STAGE1_DATA_PROVENANCE_DIR"] = local_root

    # --------------- Prep: toolkit + datasets ----------------
    toolkit_zip = normalize_gcs_uri(args.prep_toolkit_zip_uri)
    toolkit_dir = ensure_toolkit(toolkit_zip, args.prep_extract_dir, _str_to_bool(args.prep_install_requirements))

    # Load dataset recipes
    datasets_cfg = load_datasets_yaml(toolkit_dir)

    # Run (or not) the preprocess pipelines with robust error handling
    prep_summary: dict = {}
    if str(args.prepare_data).lower() != "skip":
        try:
            prep_summary = prepare_if_needed(args.prepare_data, toolkit_dir, datasets_cfg, timeout_s=int(args.prep_timeout_s))
        except Exception as e:
            logger.error("Preprocess toolkit failed: %s", e)
            logger.error("Traceback:\n%s", "".join(traceback.format_exc()))
            # Tail logs and upload to GCS
            _tail_and_upload_prep_logs(Path("/tmp/preprocess_toolkit_logs"), int(args.prep_tail_logs), str(args.logs_gcs_uri), run_id)
            if _str_to_bool(args.prep_continue_on_error):
                logger.warning("Continuing despite preprocess failure (prep_continue_on_error=true).")
                prep_summary = {"status": "error", "error": str(e)}
            else:
                raise
    else:
        prep_summary = {"status": "skipped"}

    dataset_tokenizer_id = args.dataset_tokenizer_id or args.lm_teacher_id
    dataset_tokenizer = data.build_tokenizer(dataset_tokenizer_id)

    # Build manifest set (lm+mc if provided, else dataset_manifest)
    manifests: List[str] = []
    if args.lm_manifest:
        manifests.append(args.lm_manifest)
    if args.mc_manifest:
        manifests.append(args.mc_manifest)
    if not manifests:
        manifests.append(args.dataset_manifest)

    all_entries: List[dict] = []
    manifest_snapshot: List[dict] = []
    for m in manifests:
        entries, snap = data.read_manifest(m, datasets_cfg)
        all_entries.extend(entries)
        manifest_snapshot.extend(snap)

    if not all_entries:
        raise ValueError("No training entries found. Check --lm-manifest / --mc-manifest / --dataset-manifest inputs.")

    # Determine effective teacher mode
    teacher_mode_arg = args.teacher_mode
    hybrid_enabled = _str_to_bool(args.hybrid)
    effective_teacher_mode = "hybrid" if hybrid_enabled else teacher_mode_arg

    teacher_logits_dir = args.teacher_logits_dir if (teacher_mode_arg == "precompute" and not hybrid_enabled) else None
    math_logits_dir = args.math_logits_dir if hybrid_enabled else None
    code_logits_dir = args.code_logits_dir if hybrid_enabled else None

    data_cache_bytes = _parse_size_to_int(args.data_max_cache_bytes)

    train_dataset = data.ManifestDataset(
        all_entries,
        dataset_tokenizer,
        seq_len=int(args.seq_len),
        tool_use_ratio=float(args.tool_use_ratio),
        teacher_mode=effective_teacher_mode,
        teacher_logits_dir=teacher_logits_dir,
        hybrid=hybrid_enabled,
        math_logits_dir=math_logits_dir,
        code_logits_dir=code_logits_dir,
        hybrid_strict=_str_to_bool(args.hybrid_strict),
        split="train",
        streaming=_str_to_bool(args.streaming),
        parquet_batch_rows=int(args.parquet_batch_rows),
        data_max_open_files=int(args.data_max_open_files),
        data_max_cache_bytes=data_cache_bytes,
        arrow_num_threads=int(args.arrow_num_threads),
    )

    data_readiness = _augment_data_readiness(prep_summary, train_dataset, manifest_snapshot, toolkit_zip, dataset_tokenizer_id)
    # For transparency in artifacts:
    data_readiness["lm_manifest"] = args.lm_manifest or ""
    data_readiness["mc_manifest"] = args.mc_manifest or ""
    data_readiness["dataset_manifest_fallback"] = args.dataset_manifest if not (args.lm_manifest or args.mc_manifest) else ""

    # Save artifacts
    _write_data_artifacts(local_root, data_readiness, datasets_cfg, manifest_snapshot)

    # DataLoaders
    dl_workers = max(0, int(args.dl_num_workers))
    dl_prefetch = max(1, int(args.dl_prefetch_factor))
    loader_kwargs = {
        "batch_size": int(args.batch_size),
        "shuffle": False,
        "num_workers": dl_workers,
        "pin_memory": False,
        "persistent_workers": dl_workers > 0,
        "collate_fn": data.collate_batch,
    }
    if dl_workers > 0:
        loader_kwargs["prefetch_factor"] = dl_prefetch
    train_loader = DataLoader(train_dataset, **loader_kwargs)

    val_loader: Optional[DataLoader] = None
    if args.val_manifest:
        val_entries, _ = data.read_manifest(args.val_manifest, datasets_cfg)
        val_dataset = data.ManifestDataset(
            val_entries,
            dataset_tokenizer,
            seq_len=int(args.seq_len),
            tool_use_ratio=0.0,
            teacher_mode="precompute" if not hybrid_enabled else effective_teacher_mode,
            teacher_logits_dir=teacher_logits_dir,
            hybrid=hybrid_enabled,
            math_logits_dir=math_logits_dir,
            code_logits_dir=code_logits_dir,
            hybrid_strict=_str_to_bool(args.hybrid_strict),
            split="val",
            streaming=_str_to_bool(args.streaming),
            parquet_batch_rows=int(args.parquet_batch_rows),
            data_max_open_files=int(args.data_max_open_files),
            data_max_cache_bytes=data_cache_bytes,
            arrow_num_threads=int(args.arrow_num_threads),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=min(dl_workers, 2),
            pin_memory=False,
            persistent_workers=dl_workers > 0,
            collate_fn=data.collate_batch,
        )

    # --------------- Student model (from --student-model-path) ---------------
    student_model_path = args.student_model_path or os.environ.get("STUDENT_MODEL_PATH", "")
    if not student_model_path:
        raise ValueError("Missing --student-model-path (unzipped HF directory or Hub ID).")

    # Initialize student without resizing; assert vocab if requested
    expected_vocab = len(dataset_tokenizer) if _str_to_bool(args.assert_vocab_match) else None
    model, student_tok = initialize_student(
        student_model_path=student_model_path,
        run_local_dir=local_root,
        seq_len=int(args.seq_len),
        output_gcs_uri=args.output_gcs_uri.rstrip("/"),
        run_id=run_id,
        expected_vocab_size=expected_vocab,
        strict_vocab_check=_str_to_bool(args.assert_vocab_match),
        torch_dtype=str(args.student_dtype),
        trust_remote_code=_str_to_bool(args.student_trust_remote_code),
        force_pt_override=False,  # prefer safetensors unless you want to force .pt
    )

    # Log vocab sizes
    logger.info(
        "Vocab sizes — dataset_tokenizer=%d | student_tokenizer=%d",
        int(getattr(dataset_tokenizer, "vocab_size", len(dataset_tokenizer)) or len(dataset_tokenizer)),
        int(getattr(student_tok, "vocab_size", len(student_tok)) or len(student_tok)),
    )

    # Grad checkpointing
    model = apply_grad_checkpointing(model, _str_to_bool(args.grad_checkpoint))

    # KD schedules
    beta1, beta2 = _parse_betas(args.betas)
    kd_alpha_sched = AnnealingSchedule(float(args.kd_alpha_start), float(args.kd_alpha_end), float(args.kd_anneal_pct))
    ce_beta_sched = AnnealingSchedule(1 - float(args.kd_alpha_start), 1 - float(args.kd_alpha_end), float(args.kd_anneal_pct))
    logit_l2_sched = AnnealingSchedule(float(args.keep_old_logit_l2), 0.0,
                                       float(args.keep_old_logit_l2_fade_step) / max(1, int(args.max_steps)))

    # Teachers (LM teacher constructed when needed)
    lm_teacher: Optional[TeacherWrapper] = None
    if hybrid_enabled:
        if train_dataset.sample_type_counts.get("lm", 0) > 0:
            lm_teacher = TeacherWrapper(TeacherConfig(model_id=args.lm_teacher_id))
    elif args.teacher_mode == "online":
        lm_teacher = TeacherWrapper(TeacherConfig(model_id=args.lm_teacher_id))

    # Banner + args snapshot
    _startup_banner(
        seed=int(args.seed),
        device_info=device_info,
        precision=str(args.precision),
        sdpa_enabled=sdpa_enabled,
        fa_installed=fa_installed,
        toolkit_zip=toolkit_zip,
        prepare_mode=str(args.prepare_data),
        data_summary=data_readiness,
        teacher_mode=effective_teacher_mode,
        dataset_tokenizer_id=dataset_tokenizer_id,
        hybrid=hybrid_enabled,
        lm_teacher_id=str(args.lm_teacher_id),
        mc_teacher_id=str(args.mc_teacher_id),
        math_logits_dir=math_logits_dir,
        code_logits_dir=code_logits_dir,
        kd_temperature=float(args.kd_temperature),
        alpha_start=float(args.kd_alpha_start),
        alpha_end=float(args.kd_alpha_end),
        anneal_pct=float(args.kd_anneal_pct),
        grad_accum_steps=int(args.grad_accum_steps),
        metrics_interval=int(args.metrics_interval),
        dry_run=_str_to_bool(args.dry_run),
        student_model_path=student_model_path,
        lm_manifest=str(args.lm_manifest),
        mc_manifest=str(args.mc_manifest),
        deepspeed=str(args.deepspeed),
        teacher_cache_dir=str(args.teacher_cache_dir),
    )

    out_root = args.output_gcs_uri.rstrip("/")
    run_uri = f"{out_root}/{run_id}" if out_root else None
    _write_args_snapshot(local_root, args, run_uri)

    # --------------- Trainer ---------------
    if args.deepspeed:
        logger.info("Deepspeed config provided: %s (informational; wire into Trainer if supported).", args.deepspeed)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device_info.device,
        output_dir=local_root,
        output_gcs_uri=out_root,
        run_id=run_id,
        lr=float(args.lr),
        betas=(beta1, beta2),
        weight_decay=float(args.weight_decay),
        warmup_steps=int(args.warmup_steps),
        max_steps=int(args.max_steps),
        kd_temperature=float(args.kd_temperature),
        kd_alpha_schedule=kd_alpha_sched,
        ce_beta_schedule=ce_beta_sched,
        logit_l2_gamma_schedule=logit_l2_sched,
        logit_reference=None,
        precision=str(args.precision),
        lm_teacher=lm_teacher,
        teacher_mode=effective_teacher_mode,
        teacher_logits_dir=teacher_logits_dir,
        math_logits_dir=math_logits_dir,
        code_logits_dir=code_logits_dir,
        hybrid=hybrid_enabled,
        hybrid_strict=_str_to_bool(args.hybrid_strict),
        eval_every=int(args.eval_every),
        save_every=int(args.save_every),
        grad_accum_steps=int(args.grad_accum_steps),
        metrics_interval=int(args.metrics_interval),
        limit_batches=0,
        early_stop_ppl=float(args.early_stop_ppl),
        dry_run=_str_to_bool(args.dry_run),
        # HF snapshot support:
        hf_tokenizer=student_tok,
        save_hf_snapshots=_str_to_bool(args.save_hf_snapshots),
        hf_max_shard_size=str(args.hf_max_shard_size),
    )

    json_log(
        logger,
        {
            "teacher_mode": effective_teacher_mode,
            "dataset_size": len(train_dataset),
            "seq_len": int(args.seq_len),
            "output_gcs": f"{out_root}/{run_id}",
            "grad_checkpoint": _str_to_bool(args.grad_checkpoint),
            "grad_accum_steps": int(args.grad_accum_steps),
            "metrics_interval": int(args.metrics_interval),
            "dry_run": _str_to_bool(args.dry_run),
            "hybrid": hybrid_enabled,
            "dataset_tokenizer_id": dataset_tokenizer_id,
            "dataset_tokenizer_vocab": int(getattr(dataset_tokenizer, "vocab_size", 0) or 0),
        },
    )

    trainer.train()

if __name__ == "__main__":
    main()
