"""Pipeline for ASDiv XML dataset."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
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
            tree = ET.parse(path)
            root = tree.getroot()
            for problem in root.iter():
                if problem.tag.lower() not in {"problem", "item"}:
                    continue
                body = (
                    problem.findtext("Body")
                    or problem.findtext("body")
                    or problem.findtext("Description")
                    or ""
                )
                question = (
                    problem.findtext("Question")
                    or problem.findtext("question")
                    or problem.findtext("Problem")
                    or ""
                )
                answer = (
                    problem.findtext("Answer")
                    or problem.findtext("answer")
                    or ""
                )
                text = f"{body}\n\nQ: {question}\nA:"
                text = normalize.normalize_text(text, min_length=5)
                if text:
                    sample_id = shard.compute_sample_id(source_uri, record_index, text)
                    writer.write(sample_id, text, dataset_type)
                record_index += 1
    return {"records": writer.total_records, "shards": writer.shard_paths}
