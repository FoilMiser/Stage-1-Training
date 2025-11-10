"""Pipeline for Wikipedia dumps using WikiExtractor."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

from .. import normalize, shard


def _extract_dump(xml_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "wikiextractor.WikiExtractor",
        xml_path,
        "--processes",
        "1",
        "-q",
        "-o",
        output_dir,
    ]
    subprocess.run(cmd, check=True)


def run(
    local_inputs: List[Tuple[str, str]],
    output_dir: str,
    *,
    dataset_type: str,
    max_records: int,
    work_dir: str,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    record_index = 0
    with shard.ShardWriter(output_dir, dataset_type, max_records=max_records) as writer:
        for source_uri, path in local_inputs:
            extract_dir = os.path.join(work_dir, f"extract_{record_index}")
            _extract_dump(path, extract_dir)
            for text_file in Path(extract_dir).rglob("*.txt"):
                content = text_file.read_text(encoding="utf-8", errors="ignore")
                text = normalize.normalize_text(content)
                if not text:
                    record_index += 1
                    continue
                sample_id = shard.compute_sample_id(source_uri, record_index, text)
                writer.write(sample_id, text, dataset_type)
                record_index += 1
    return {"records": writer.total_records, "shards": writer.shard_paths}
