"""Pipeline for WikiText datasets."""

from __future__ import annotations

import io
import os
import zipfile
from typing import Iterable, List, Tuple

from .. import normalize, shard


def _iter_documents(zf: zipfile.ZipFile, name: str) -> Iterable[str]:
    with zf.open(name) as fh:
        text_io = io.TextIOWrapper(fh, encoding="utf-8", errors="ignore")
        buffer: List[str] = []
        for line in text_io:
            stripped = line.rstrip("\n")
            if stripped.strip() == "":
                if buffer:
                    yield "\n".join(buffer)
                    buffer = []
            else:
                buffer.append(stripped)
        if buffer:
            yield "\n".join(buffer)


def run(
    local_inputs: List[Tuple[str, str]],
    output_dir: str,
    *,
    dataset_type: str,
    max_records: int,
    work_dir: str,
) -> dict:
    del work_dir  # Unused but kept for signature compatibility.
    os.makedirs(output_dir, exist_ok=True)
    record_index = 0
    with shard.ShardWriter(output_dir, dataset_type, max_records=max_records) as writer:
        for source_uri, path in local_inputs:
            with zipfile.ZipFile(path) as zf:
                for name in zf.namelist():
                    if name.endswith("/"):
                        continue
                    for doc in _iter_documents(zf, name):
                        text = normalize.normalize_text(doc)
                        if not text:
                            record_index += 1
                            continue
                        sample_id = shard.compute_sample_id(source_uri, record_index, text)
                        writer.write(sample_id, text, dataset_type)
                        record_index += 1
    return {"records": writer.total_records, "shards": writer.shard_paths}
