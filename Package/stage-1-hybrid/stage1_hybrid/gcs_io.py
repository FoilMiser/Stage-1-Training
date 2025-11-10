"""Helpers for interacting with Google Cloud Storage in Vertex jobs."""
from __future__ import annotations

import io
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

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
_STREAM_CHUNK_BYTES = 8 * 1024 * 1024
_GCSFS_SINGLETON: Optional["gcsfs.GCSFileSystem"] = None


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


def _gcsfs_filesystem() -> "gcsfs.GCSFileSystem":
    global _GCSFS_SINGLETON
    if gcsfs is None:
        raise GCSIOError(
            "gcsfs is required but not installed and google-cloud-storage is unavailable."
        )
    if _GCSFS_SINGLETON is None:
        _GCSFS_SINGLETON = gcsfs.GCSFileSystem(
            token="cloud",
            default_cache_type="none",
            cache_timeout=0,
            consistency_period=0.0,
        )
    return _GCSFS_SINGLETON


def get_gcsfs() -> "gcsfs.GCSFileSystem":
    """Return a shared gcsfs filesystem instance configured for streaming."""

    return _gcsfs_filesystem()


def open_uri(uri: str, mode: str = "rb", *, block_size: int = _STREAM_CHUNK_BYTES):
    """Open a URI for streaming without materialising it on disk."""

    if uri.startswith("gs://"):
        fs = _gcsfs_filesystem()
        return fs.open(uri, mode, block_size=block_size, cache_type="none")
    return open(uri, mode)  # noqa: SIM115


def gcs_to_local(gcs_uri: str, local_path: str, retries: int = _DEF_RETRIES) -> str:
    """Copy a single object from GCS to a local path."""
    _ensure_parent(local_path)

    def _download_stream(copy_func: Callable[[], None]) -> str:
        copy_func()
        logger.info("Copied %s -> %s", gcs_uri, local_path)
        return local_path

    if _have_storage_client():
        def _download() -> str:
            client = _storage_client()
            bucket_name, key = _parse_gcs_uri(gcs_uri)
            if not key:
                raise GCSIOError(f"gcs_to_local expects an object, not a bucket: {gcs_uri}")
            blob = client.bucket(bucket_name).blob(key)
            if not blob.exists():
                raise GCSIOError(f"GCS object not found: {gcs_uri}")

            def _copy() -> None:
                with open(local_path, "wb") as dst:
                    blob.download_to_file(dst, checksum="crc32c")

            return _download_stream(_copy)

        return _with_retries(_download, retries=retries)

    def _download_fs() -> str:
        def _copy() -> None:
            with open_uri(gcs_uri, "rb") as src, open(local_path, "wb") as dst:
                shutil.copyfileobj(src, dst, length=_STREAM_CHUNK_BYTES)

        return _download_stream(_copy)

    return _with_retries(_download_fs, retries=retries)


def _iter_local_files(src_path: Path) -> Iterator[Tuple[Path, str]]:
    if src_path.is_dir():
        for child in sorted(src_path.rglob("*")):
            if child.is_file():
                yield child, child.relative_to(src_path).as_posix()
    else:
        yield src_path, src_path.name


def _upload_file(
    file_src: Path,
    object_uri: str,
    *,
    retries: int,
    content_type: Optional[str] = None,
) -> None:
    def _upload_with_storage() -> None:
        client = _storage_client()
        bucket_name, key = _parse_gcs_uri(object_uri)
        if not key:
            raise GCSIOError(f"local_to_gcs target must be an object path: {object_uri}")
        blob = client.bucket(bucket_name).blob(key)

        try:
            blob.upload_from_filename(
                filename=str(file_src),
                if_generation_match=0,
                checksum="crc32c",
                content_type=content_type,
            )
            logger.info("Uploaded %s -> %s", file_src, object_uri)
        except PreconditionFailed:
            logger.debug("Skipped existing object (no-clobber): %s", object_uri)

    def _upload_with_fs() -> None:
        fs = _gcsfs_filesystem()
        try:
            if fs.exists(object_uri):
                logger.debug("Skipped existing object (no-clobber, gcsfs): %s", object_uri)
                return
        except Exception:
            pass
        with open(file_src, "rb") as fh, fs.open(object_uri, "wb") as dst:
            shutil.copyfileobj(fh, dst, length=_STREAM_CHUNK_BYTES)
        logger.info("Uploaded (gcsfs) %s -> %s", file_src, object_uri)

    uploader = _upload_with_storage if _have_storage_client() else _upload_with_fs
    _with_retries(uploader, retries=retries)


def local_to_gcs(local_path: str, gcs_uri: str, retries: int = _DEF_RETRIES) -> None:
    """
    Upload a local file or directory to GCS.
    Safe by default: uses if_generation_match=0 (no-clobber). If the object already
    exists, the upload is skipped gracefully (treated as success).
    """
    src_path = Path(local_path)
    base_uri = gcs_uri.rstrip("/")
    original_uri = gcs_uri

    if src_path.is_dir():
        for file_src, rel in _iter_local_files(src_path):
            target = f"{base_uri}/{rel}"
            _upload_file(file_src, target, retries=retries)
    else:
        target = f"{base_uri}/{src_path.name}" if original_uri.endswith("/") else original_uri
        _upload_file(src_path, target, retries=retries)


def iter_gcs(uri: str, retries: int = _DEF_RETRIES) -> Iterator[str]:
    """
    List objects that match the provided GCS URI.
    Supported patterns:
      - Exact object: gs://bucket/path/to.obj
      - Prefix (directory-like): gs://bucket/prefix/
      - Trailing wildcard: gs://bucket/prefix/*
    Yields gs:// URIs for matches (possibly empty).
    """
    if _have_storage_client():
        def _list() -> List[str]:
            client = _storage_client()
            bucket_name, key = _parse_gcs_uri(uri)
            results: List[str] = []

            is_prefix = key.endswith("/") or key.endswith("*")
            if not key or not is_prefix:
                blob = client.bucket(bucket_name).blob(key)
                if blob.exists():
                    results.append(uri)
                return results

            prefix = key[:-1] if key.endswith("*") else key
            prefix = prefix.lstrip("/")

            for blob in client.list_blobs(bucket_name, prefix=prefix):
                results.append(f"gs://{bucket_name}/{blob.name}")
            return results

        results = _with_retries(_list, retries=retries)
        for item in results:
            yield item
        return

    def _list_fs() -> List[str]:
        fs = _gcsfs_filesystem()
        matches = fs.glob(uri)
        return [m if m.startswith("gs://") else f"gs://{m}" for m in matches]

    results = _with_retries(_list_fs, retries=retries)
    for item in results:
        yield item


def list_gcs(uri: str, retries: int = _DEF_RETRIES) -> List[str]:
    return list(iter_gcs(uri, retries=retries))


def dir_to_gcs(
    src_dir: str,
    dst_prefix: str,
    *,
    content_type_by_ext: Optional[Dict[str, str]] = None,
    retries: int = _DEF_RETRIES,
) -> None:
    """Upload a directory tree to a GCS prefix with optional content types."""

    base = Path(src_dir)
    if not base.is_dir():
        raise GCSIOError(f"dir_to_gcs expects a directory, got: {src_dir}")
    normalized = dst_prefix.rstrip("/")
    mapping = {k.lower(): v for k, v in (content_type_by_ext or {}).items()}

    for file_src, rel in _iter_local_files(base):
        ext = file_src.suffix.lower()
        content_type = mapping.get(ext)
        target = f"{normalized}/{rel}"
        _upload_file(file_src, target, retries=retries, content_type=content_type)


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
