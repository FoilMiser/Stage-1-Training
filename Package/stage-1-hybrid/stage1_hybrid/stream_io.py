"""Streaming readers for dataset shards stored in GCS."""
from __future__ import annotations

import gzip
import io
import json
from typing import Dict, Iterator

try:
    import pyarrow.dataset as ds
except Exception:  # pragma: no cover - runtime only
    ds = None

from .gcs_io import get_gcsfs, open_uri


def _iter_json_lines(handle: io.TextIOBase) -> Iterator[Dict[str, object]]:
    for line in handle:
        if not line.strip():
            continue
        yield json.loads(line)


def stream_jsonl(uri: str, *, encoding: str = "utf-8") -> Iterator[Dict[str, object]]:
    """Stream a JSONL file without buffering the entire shard."""

    with open_uri(uri, "rb", block_size=8 * 1024 * 1024) as handle:
        text = io.TextIOWrapper(handle, encoding=encoding)
        yield from _iter_json_lines(text)


def stream_jsonl_gz(uri: str, *, encoding: str = "utf-8") -> Iterator[Dict[str, object]]:
    """Stream a gzipped JSONL shard directly from GCS."""

    with open_uri(uri, "rb") as raw:
        with gzip.GzipFile(fileobj=raw) as gz:
            text = io.TextIOWrapper(gz, encoding=encoding)
            yield from _iter_json_lines(text)


def stream_parquet(
    uri: str,
    *,
    batch_rows: int = 1024,
    arrow_num_threads: int = 1,
) -> Iterator[Dict[str, object]]:
    """Stream Parquet rows from local disk or GCS."""

    if ds is None:  # pragma: no cover - pyarrow missing in minimal envs
        raise RuntimeError("pyarrow is required to stream parquet datasets")

    filesystem = get_gcsfs() if uri.startswith("gs://") else None
    dataset = ds.dataset(uri, format="parquet", filesystem=filesystem)
    scanner = dataset.scanner(batch_size=max(1, batch_rows), use_threads=arrow_num_threads > 1)
    for batch in scanner.to_batches():
        names = batch.schema.names
        columns = [batch.column(i) for i in range(len(names))]
        for row_idx in range(batch.num_rows):
            record: Dict[str, object] = {}
            for col_idx, name in enumerate(names):
                value = columns[col_idx][row_idx]
                record[name] = value.as_py() if hasattr(value, "as_py") else value
            yield record


__all__ = [
    "stream_jsonl",
    "stream_jsonl_gz",
    "stream_parquet",
]
