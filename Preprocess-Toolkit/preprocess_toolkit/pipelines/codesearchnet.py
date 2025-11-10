"""Pipeline for CodeSearchNet archives."""

from __future__ import annotations

import io
import json
import os
import zipfile
from typing import List, Tuple

from .. import normalize, shard

CODE_EXTENSIONS = {".py", ".java", ".js", ".go", ".php", ".rb"}
JSON_FIELDS = ["code", "func_code", "original_string", "content"]


def _handle_jsonl(
    zf: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    source_uri: str,
    writer: shard.ShardWriter,
    start_index: int,
    dataset_type: str,
) -> int:
    record_index = start_index
    with zf.open(info) as fh:
        text_io = io.TextIOWrapper(fh, encoding="utf-8", errors="ignore")
        for line in text_io:
            line = line.strip()
            if not line:
                continue
            record_index += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = None
            for field in JSON_FIELDS:
                if field in obj and obj[field]:
                    value = obj[field]
                    break
            if value is None:
                continue
            text = normalize.normalize_text(value, min_length=5, collapse_whitespace=False)
            if not text:
                continue
            sample_id = shard.compute_sample_id(source_uri, record_index, text)
            writer.write(sample_id, text, dataset_type)
    return record_index


def _handle_code_file(
    zf: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    source_uri: str,
    writer: shard.ShardWriter,
    start_index: int,
    dataset_type: str,
) -> int:
    record_index = start_index + 1
    with zf.open(info) as fh:
        content = fh.read().decode("utf-8", "ignore")
    text = normalize.normalize_text(content, min_length=5, collapse_whitespace=False)
    if text:
        sample_id = shard.compute_sample_id(source_uri, record_index, text)
        writer.write(sample_id, text, dataset_type)
    return record_index


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
            with zipfile.ZipFile(path) as zf:
                jsonl_files = [info for info in zf.infolist() if info.filename.lower().endswith(".jsonl")]
                if jsonl_files:
                    for info in jsonl_files:
                        record_index = _handle_jsonl(
                            zf,
                            info,
                            source_uri=source_uri,
                            writer=writer,
                            start_index=record_index,
                            dataset_type=dataset_type,
                        )
                else:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        _, ext = os.path.splitext(info.filename)
                        if ext.lower() not in CODE_EXTENSIONS:
                            continue
                        record_index = _handle_code_file(
                            zf,
                            info,
                            source_uri=source_uri,
                            writer=writer,
                            start_index=record_index,
                            dataset_type=dataset_type,
                        )
    return {"records": writer.total_records, "shards": writer.shard_paths}
