"""Pipeline for Dolma JSONL samples."""

from __future__ import annotations

import gzip
import json
import os
from typing import List, Tuple

from .. import normalize, shard


def _iter_records(path: str):
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def run(
    local_inputs: List[Tuple[str, str]],
    output_dir: str,
    *,
    dataset_type: str,
    max_records: int,
    work_dir: str,
) -> dict:
    del work_dir
    os.makedirs(output_dir, exist_ok=True)
    record_index = 0
    with shard.ShardWriter(output_dir, dataset_type, max_records=max_records) as writer:
        for source_uri, path in local_inputs:
            for record in _iter_records(path):
                text = normalize.normalize_text(record.get("text"))
                if text:
                    sample_id = shard.compute_sample_id(source_uri, record_index, text)
                    writer.write(sample_id, text, dataset_type)
                record_index += 1
    return {"records": writer.total_records, "shards": writer.shard_paths}
