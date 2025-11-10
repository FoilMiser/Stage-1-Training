"""Data preparation bootstrap utilities for Stage-1 training."""
from __future__ import annotations

import os
import shutil
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml

from .gcs_io import GCSIOError, dir_to_gcs, gcs_to_local, list_gcs
from .utils import Backoff, configure_logging

logger = configure_logging()


_DEFAULT_TOOLKIT_ZIP = "gs://liquid-llm-bucket-2/sandbox/preprocess-toolkit/preprocess-toolkit-stage1-1-0.zip"
_PIPELINE_LOG_DIR = Path("/tmp/preprocess_toolkit_logs")
_PIPELINE_LOG_DIR.mkdir(parents=True, exist_ok=True)
_STREAM_LOG_EVERY = 50_000


@dataclass
class DatasetSpec:
    """Configuration for a dataset provided by the preprocess toolkit."""

    job: str
    inp: str
    out: str
    dtype: str
    manifest: str | None

    def shard_glob(self) -> str:
        return f"{self.out.rstrip('/')}/*.jsonl"


def normalize_gcs_uri(uri: str | None) -> str:
    """Ensure URIs include the gs:// prefix."""

    if not uri:
        return _DEFAULT_TOOLKIT_ZIP
    if uri.startswith("gs://"):
        return uri
    return f"gs://{uri.lstrip('/')}"


def ensure_toolkit(zip_uri: str, extract_dir: str, install_requirements: bool) -> str:
    """Download and unpack the preprocess toolkit if necessary."""

    normalized = normalize_gcs_uri(zip_uri)
    local_zip = "/tmp/preprocess.zip"
    logger.info("Ensuring preprocess toolkit from %s", normalized)
    gcs_to_local(normalized, local_zip)
    extract_path = Path(extract_dir)
    if extract_path.exists():
        shutil.rmtree(extract_path)
    extract_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(local_zip, "r") as zf:
        zf.extractall(extract_path)
    logger.info("Toolkit extracted to %s", extract_path)
    if install_requirements:
        requirements = extract_path / "requirements.txt"
        if requirements.exists():
            _install_requirements(requirements)
        else:
            logger.warning("Toolkit requirements.txt missing at %s", requirements)
    return str(extract_path)


def _install_requirements(path: Path) -> None:
    backoff = Backoff()
    for attempt in range(3):
        logger.info("Installing toolkit requirements (attempt %d)", attempt + 1)
        result = subprocess.run(
            ["python", "-m", "pip", "install", "-r", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info("Toolkit requirements installed successfully")
            return
        logger.warning("Toolkit requirements install failed: %s", result.stderr)
        backoff.sleep()
    raise RuntimeError("Failed to install preprocess toolkit requirements")


def load_datasets_yaml(extract_dir: str) -> Dict[str, DatasetSpec]:
    config_path = Path(extract_dir) / "preprocess_toolkit" / "config" / "datasets.yaml"
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    datasets: Dict[str, DatasetSpec] = {}
    for job, payload in data.items():
        datasets[job] = DatasetSpec(
            job=job,
            inp=payload.get("in", ""),
            out=payload.get("out", ""),
            dtype=payload.get("type", "lm"),
            manifest=payload.get("manifest"),
        )
    return datasets


def shards_ready(out_dir: str) -> bool:
    if not out_dir:
        return False
    pattern = f"{out_dir.rstrip('/')}/*.jsonl"
    if out_dir.startswith("gs://"):
        try:
            matches = list_gcs(pattern)
        except GCSIOError:
            logger.debug("Shard readiness check failed for %s", out_dir)
            return False
        return any(match.endswith(".jsonl") for match in matches)
    return any(Path(out_dir).glob("*.jsonl"))


def upload_pipeline_logs(
    logs_dir: str | Path,
    logs_gcs_uri: str,
    run_id: str,
    *,
    content_type_by_ext: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Upload toolkit logs to a stable GCS prefix.

    Parameters
    ----------
    logs_dir:
        Local directory containing ``*.log`` files from the preprocess toolkit.
    logs_gcs_uri:
        Root GCS directory for log uploads. When empty, the upload is skipped.
    run_id:
        Unique identifier for the current run. Used to build a subdirectory so
        concurrent jobs do not clobber each other.
    content_type_by_ext:
        Optional mapping from file extension to desired content-type metadata.

    Returns
    -------
    Optional[str]
        The destination prefix if the upload succeeded, otherwise ``None``.
    """

    if not logs_gcs_uri:
        logger.debug("Skipping log upload: no destination provided")
        return None

    path = Path(logs_dir)
    if not path.exists():
        logger.warning("Log directory %s missing; nothing to upload", path)
        return None

    dst_prefix = logs_gcs_uri.rstrip("/") + f"/prep_logs/{run_id}"
    mapping = content_type_by_ext or {".log": "text/plain"}

    try:
        dir_to_gcs(str(path), dst_prefix, content_type_by_ext=mapping)
        logger.info("Uploaded preprocess logs to %s", dst_prefix)
        return dst_prefix
    except Exception as exc:  # pragma: no cover - runtime environment
        logger.warning("Failed to upload preprocess logs to %s: %s", dst_prefix, exc)
        return None


def run_pipeline(job: str, inp: str, out: str, extract_dir: str, timeout_s: int) -> Path:
    log_path = _PIPELINE_LOG_DIR / f"{job}.log"
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{extract_dir}:{pythonpath}" if pythonpath else extract_dir
    cmd = [
        "python",
        "-m",
        "preprocess_toolkit.driver",
        "--job",
        job,
        "--in",
        inp,
        "--out",
        out,
        "--max-records",
        "20000",
        "--manifest",
    ]
    logger.info("Running preprocess pipeline %s", job)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=extract_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        line_count = 0
        start = time.time()
        try:
            while True:
                line = process.stdout.readline() if process.stdout else ""
                if not line and process.poll() is not None:
                    break
                if line:
                    log_file.write(line)
                    line_count += 1
                    if line_count % _STREAM_LOG_EVERY == 0:
                        logger.info("Pipeline %s processed ~%d log lines", job, line_count)
                if timeout_s and (time.time() - start) > timeout_s:
                    process.terminate()
                    raise TimeoutError(f"Pipeline {job} timed out after {timeout_s}s; log: {log_path}")
        finally:
            if process.stdout:
                process.stdout.close()
        ret = process.wait()
    if ret != 0:
        raise RuntimeError(f"Pipeline {job} failed with exit code {ret}; see {log_path}")
    logger.info("Pipeline %s completed; log=%s", job, log_path)
    return log_path


def _count_jsonl_files(out_dir: str) -> int:
    pattern = f"{out_dir.rstrip('/')}/*.jsonl"
    if out_dir.startswith("gs://"):
        try:
            matches = list_gcs(pattern)
        except GCSIOError:
            return 0
        return len(matches)
    return len(list(Path(out_dir).glob("*.jsonl")))


def prepare_if_needed(
    mode: str,
    extract_dir: str,
    datasets_config: Dict[str, DatasetSpec],
    *,
    timeout_s: int = 0,
) -> Dict[str, object]:
    mode = mode.lower()
    if mode not in {"auto", "force", "skip"}:
        raise ValueError(f"Unknown prepare mode: {mode}")
    summary: Dict[str, object] = {
        "mode": mode,
        "datasets": {},
        "prepared_at": time.time(),
    }
    for job, spec in datasets_config.items():
        before_ready = shards_ready(spec.out)
        before_files = _count_jsonl_files(spec.out)
        status = "READY" if before_ready else "MISSING"
        should_run = mode == "force" or (mode == "auto" and not before_ready)
        if mode == "skip":
            status = "SKIPPED"
        elif should_run:
            status = "BUILDING"
        dataset_summary = {
            "input": spec.inp,
            "output": spec.out,
            "manifest": spec.manifest,
            "type": spec.dtype,
            "before_ready": before_ready,
            "before_files": before_files,
            "after_ready": before_ready,
            "after_files": before_files,
            "ran_pipeline": False,
            "status": status,
        }
        if should_run and mode != "skip":
            try:
                run_pipeline(job, spec.inp, spec.out, extract_dir, timeout_s)
                dataset_summary["ran_pipeline"] = True
            except Exception as exc:
                dataset_summary["status"] = "FAILED"
                dataset_summary["error"] = str(exc)
                summary["datasets"][job] = dataset_summary
                logger.error("Pipeline %s failed: %s", job, exc)
                raise
        after_ready = shards_ready(spec.out)
        after_files = _count_jsonl_files(spec.out)
        if dataset_summary["status"] != "FAILED":
            dataset_summary["after_ready"] = after_ready
            dataset_summary["after_files"] = after_files
            if dataset_summary["status"] != "SKIPPED":
                dataset_summary["status"] = "READY" if after_ready else dataset_summary["status"]
        summary["datasets"][job] = dataset_summary
    return summary


__all__ = [
    "DatasetSpec",
    "ensure_toolkit",
    "load_datasets_yaml",
    "normalize_gcs_uri",
    "prepare_if_needed",
    "run_pipeline",
    "shards_ready",
    "upload_pipeline_logs",
]
