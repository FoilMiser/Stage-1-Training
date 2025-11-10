"""Pipeline for SVAMP math dataset."""

from __future__ import annotations

import json
import os
from typing import List, Tuple

from .. import normalize, shard


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
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for item in data:
                body = item.get("Body") or ""
                question = item.get("Question") or item.get("question") or ""
                answer = item.get("Answer") or item.get("answer") or ""
                text = f"{body}\n\nQ: {question}\nA:"
                text = normalize.normalize_text(text, min_length=5)
                if text:
                    sample_id = shard.compute_sample_id(source_uri, record_index, text)
                    writer.write(sample_id, text, dataset_type)
                record_index += 1
    return {"records": writer.total_records, "shards": writer.shard_paths}
