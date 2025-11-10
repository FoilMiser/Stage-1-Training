"""I/O helpers for interacting with Google Cloud Storage."""

from __future__ import annotations

import logging
import os
import pathlib
import subprocess
import time
from typing import Iterable, List, Optional

try:
    import gcsfs  # type: ignore
except ImportError:  # pragma: no cover - optional dependency in tests
    gcsfs = None

LOGGER = logging.getLogger(__name__)


def _split_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS uri: {uri}")
    parts = uri[5:].split("/", 1)
    bucket = parts[0]
    path = parts[1] if len(parts) == 2 else ""
    return bucket, path


def _run_gcloud(args: Iterable[str], retries: int = 3, delay: float = 1.0) -> subprocess.CompletedProcess:
    cmd = ["gcloud", "storage", *args]
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            LOGGER.debug("Running gcloud command: %s", " ".join(cmd))
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return proc
        except FileNotFoundError as exc:  # pragma: no cover - environment specific
            raise
        except subprocess.CalledProcessError as exc:
            last_err = exc
            LOGGER.warning("gcloud command failed (attempt %s/%s): %s", attempt, retries, exc)
            time.sleep(delay * attempt)
    if last_err:
        raise last_err
    raise RuntimeError("Unknown gcloud execution failure")


def _ensure_gcsfs():
    if gcsfs is None:
        raise RuntimeError("gcsfs is required when gcloud is unavailable")
    try:
        retry_module = getattr(gcsfs, "retry", None)
        if retry_module and hasattr(retry_module, "FilesystemInconsistency"):
            return gcsfs.GCSFileSystem(retry_handler=retry_module.FilesystemInconsistency())
    except Exception:  # pragma: no cover - fallback when retry is unsupported
        pass
    return gcsfs.GCSFileSystem()


def gcs_to_local(uri: str, local_path: str) -> str:
    """Download a GCS object to ``local_path``.

    Uses ``gcloud storage cp`` when available. Falls back to ``gcsfs`` when
    ``gcloud`` is missing or fails.
    """

    directory = os.path.dirname(local_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    try:
        _run_gcloud(["cp", uri, local_path])
        return local_path
    except (FileNotFoundError, subprocess.CalledProcessError):
        LOGGER.info("Falling back to gcsfs for download: %s", uri)
    fs = _ensure_gcsfs()
    bucket, path = _split_gs_uri(uri)
    with fs.open(f"{bucket}/{path}", "rb") as src, open(local_path, "wb") as dst:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            dst.write(chunk)
    return local_path


def local_to_gcs(local_path: str, uri: str) -> str:
    """Upload ``local_path`` to the destination ``uri`` on GCS."""

    if uri.endswith("/"):
        uri = uri + pathlib.Path(local_path).name
    try:
        _run_gcloud(["cp", local_path, uri])
        return uri
    except (FileNotFoundError, subprocess.CalledProcessError):
        LOGGER.info("Falling back to gcsfs for upload: %s", uri)
    fs = _ensure_gcsfs()
    bucket, path = _split_gs_uri(uri)
    with open(local_path, "rb") as src, fs.open(f"{bucket}/{path}", "wb") as dst:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            dst.write(chunk)
    return uri


def list_gcs(glob_uri: str) -> List[str]:
    """List objects matching ``glob_uri`` on GCS."""

    try:
        proc = _run_gcloud(["ls", glob_uri])
        return [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.CalledProcessError):
        LOGGER.info("Falling back to gcsfs for list: %s", glob_uri)
    fs = _ensure_gcsfs()
    bucket, path = _split_gs_uri(glob_uri)
    matches = fs.glob(f"{bucket}/{path}")
    return [f"gs://{match}" for match in matches]
