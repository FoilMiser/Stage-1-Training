"""Pipeline for GSM8K math dataset."""

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
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        record_index += 1
                        continue
                    question = obj.get("question") or obj.get("question_text")
                    answer = obj.get("answer") or obj.get("answer_text")
                    pieces = [question or ""]
                    if answer:
                        pieces.append("")
                        pieces.append(f"Answer: {answer}")
                    text = normalize.normalize_text("\n".join(pieces), min_length=5)
                    if text:
                        sample_id = shard.compute_sample_id(source_uri, record_index, text)
                        writer.write(sample_id, text, dataset_type)
                    record_index += 1
    return {"records": writer.total_records, "shards": writer.shard_paths}
