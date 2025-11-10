"""Helpers for interacting with Google Cloud Storage in Vertex jobs."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Any, Tuple, Optional

import io

try:  # Primary, robust client
    from google.cloud import storage
    from google.api_core.exceptions import PreconditionFailed, NotFound
except Exception:  # pragma: no cover - handled at runtime
    storage = None
    PreconditionFailed = NotFound = Exception  # type: ignore

try:  # Optional fallback
    import gcsfs
except Exception:  # pragma: no cover - handled at runtime
    gcsfs = None

from .utils import Backoff, configure_logging

logger = configure_logging()


class GCSIOError(RuntimeError):
    """Raised when a GCS operation fails after retries."""


_DEF_RETRIES = 3


def _with_retries(func, *, retries: int = _DEF_RETRIES) -> Any:
    backoff = Backoff()
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return func()
        except Exception as exc:  # pragma: no cover - runtime only
            last_error = exc
            logger.warning("GCS operation failed (attempt %s/%s): %s",
                           attempt + 1, retries + 1, exc)
            if attempt >= retries:
                break
            backoff.sleep()
    if last_error is None:
        raise GCSIOError("GCS operation failed")
    raise GCSIOError(str(last_error))


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _parse_gcs_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("gs://"):
        raise GCSIOError(f"Invalid GCS URI (must start with gs://): {uri}")
    without_scheme = uri[5:]
    if "/" not in without_scheme:
        # bucket only
        return without_scheme, ""
    bucket, key = without_scheme.split("/", 1)
    return bucket, key


def _have_storage_client() -> bool:
    return storage is not None


def _storage_client():
    if not _have_storage_client():
        raise GCSIOError(
            "google-cloud-storage is required but not available in this image. "
            "Install it or provide gcsfs as a fallback."
        )
    return storage.Client()


def _gcsfs_filesystem():
    if gcsfs is None:
        raise GCSIOError(
            "gcsfs is required but not installed and google-cloud-storage is unavailable."
        )
    return gcsfs.GCSFileSystem()


def gcs_to_local(gcs_uri: str, local_path: str, retries: int = _DEF_RETRIES) -> str:
    """Copy a single object from GCS to a local path."""
    _ensure_parent(local_path)

    if _have_storage_client():
        def _download() -> str:
            client = _storage_client()
            bucket_name, key = _parse_gcs_uri(gcs_uri)
            if not key:
                raise GCSIOError(f"gcs_to_local expects an object, not a bucket: {gcs_uri}")
            blob = client.bucket(bucket_name).blob(key)
            if not blob.exists():
                raise GCSIOError(f"GCS object not found: {gcs_uri}")
            # CRC32C by default; adds integrity
            blob.download_to_filename(local_path, checksum="crc32c")
            logger.info("Copied %s -> %s", gcs_uri, local_path)
            return local_path

        return _with_retries(_download, retries=retries)

    # Fallback to gcsfs
    def _download_fs() -> str:
        fs = _gcsfs_filesystem()
        with fs.open(gcs_uri, "rb") as src, open(local_path, "wb") as dst:
            dst.write(src.read())
        logger.info("Copied (gcsfs) %s -> %s", gcs_uri, local_path)
        return local_path

    return _with_retries(_download_fs, retries=retries)


def local_to_gcs(local_path: str, gcs_uri: str, retries: int = _DEF_RETRIES) -> None:
    """
    Upload a local file or directory to GCS.
    Safe by default: uses if_generation_match=0 (no-clobber). If the object already
    exists, the upload is skipped gracefully (treated as success).
    """
    src_path = Path(local_path)

    if _have_storage_client():
        def _upload_file(file_src: Path, object_uri: str) -> None:
            bucket_name, key = _parse_gcs_uri(object_uri)
            if not key:
                raise GCSIOError(f"local_to_gcs target must be an object path: {object_uri}")
            client = _storage_client()
            blob = client.bucket(bucket_name).blob(key)

            # Attempt no-clobber upload; treat 'already exists' as success.
            try:
                blob.upload_from_filename(
                    filename=str(file_src),
                    if_generation_match=0,   # only create if object does not exist
                    checksum="crc32c",       # integrity
                )
                logger.info("Uploaded %s -> %s", file_src, object_uri)
            except PreconditionFailed:
                logger.debug("Skipped existing object (no-clobber): %s", object_uri)
            except Exception as exc:
                raise GCSIOError(f"Upload failed for {object_uri}: {exc}")

        def _upload_dir(dir_src: Path, base_uri: str) -> None:
            base_uri = base_uri.rstrip("/")
            for child in sorted(dir_src.rglob("*")):
                if child.is_file():
                    rel = child.relative_to(dir_src).as_posix()
                    _upload_file(child, f"{base_uri}/{rel}")

        def _upload() -> None:
            if src_path.is_dir():
                _upload_dir(src_path, gcs_uri)
            else:
                _upload_file(src_path, gcs_uri)

        return _with_retries(_upload, retries=retries)

    # Fallback to gcsfs if storage client is unavailable.
    def _upload_fs_file(file_src: Path, object_uri: str) -> None:
        fs = _gcsfs_filesystem()
        # Implement no-clobber by checking existence before write.
        try:
            if fs.exists(object_uri):
                logger.debug("Skipped existing object (no-clobber, gcsfs): %s", object_uri)
                return
        except Exception:
            # If exists check fails, proceed to attempt write; race is acceptable.
            pass
        with open(file_src, "rb") as fh, fs.open(object_uri, "wb") as dst:
            dst.write(fh.read())
        logger.info("Uploaded (gcsfs) %s -> %s", file_src, object_uri)

    def _upload_fs() -> None:
        if src_path.is_dir():
            base = gcs_uri.rstrip("/")
            for child in sorted(src_path.rglob("*")):
                if child.is_file():
                    rel = child.relative_to(src_path).as_posix()
                    _upload_fs_file(child, f"{base}/{rel}")
        else:
            _upload_fs_file(src_path, gcs_uri)

    return _with_retries(_upload_fs, retries=retries)


def list_gcs(uri: str, retries: int = _DEF_RETRIES) -> List[str]:
    """
    List objects that match the provided GCS URI.
    Supported patterns:
      - Exact object: gs://bucket/path/to.obj
      - Prefix (directory-like): gs://bucket/prefix/
      - Trailing wildcard: gs://bucket/prefix/*
    Returns gs:// URIs for matches (possibly empty).
    """
    if _have_storage_client():
        def _list() -> List[str]:
            client = _storage_client()
            bucket_name, key = _parse_gcs_uri(uri)
            out: List[str] = []

            # Exact object (no wildcard, no trailing slash)
            is_prefix = key.endswith("/") or key.endswith("*")
            if not key or not is_prefix:
                blob = client.bucket(bucket_name).blob(key)
                if blob.exists():
                    return [uri]
                return []

            # Prefix listing
            if key.endswith("*"):
                prefix = key[:-1]
            else:
                prefix = key
            # Strip accidental leading slashes in key
            prefix = prefix.lstrip("/")

            for blob in client.list_blobs(bucket_name, prefix=prefix):
                out.append(f"gs://{bucket_name}/{blob.name}")
            return out

        return _with_retries(_list, retries=retries)

    # Fallback to gcsfs globbing
    def _list_fs() -> List[str]:
        fs = _gcsfs_filesystem()
        # gcsfs expects patterns without scheme in some cases; use glob on full URI.
        matches = fs.glob(uri)
        # Normalize to gs://...
        results: List[str] = []
        for m in matches:
            results.append(m if m.startswith("gs://") else f"gs://{m}")
        return results

    return _with_retries(_list_fs, retries=retries)


def maybe_sync_dir(local_dir: str, gcs_dir: str) -> None:
    """Upload a directory tree to a GCS prefix if provided (no-clobber per-object)."""
    if not gcs_dir:
        logger.debug("Skipping sync for %s; no destination provided", local_dir)
        return
    root = Path(local_dir)
    if not root.exists():
        logger.warning("Local directory %s missing; nothing to sync", local_dir)
        return
    local_to_gcs(str(root), gcs_dir)


def ensure_local_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path
