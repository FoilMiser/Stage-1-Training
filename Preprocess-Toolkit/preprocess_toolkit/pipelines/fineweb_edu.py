"""Pipeline for FineWeb-Edu parquet samples."""

from __future__ import annotations

import os
from typing import List, Tuple

import pyarrow.parquet as pq

from .. import normalize, shard


def _detect_text_column(pf: pq.ParquetFile) -> str:
    schema = pf.schema
    for candidate in ("text", "content"):
        if schema.get_field_index(candidate) != -1:
            return candidate
    raise ValueError("No text/content column found in parquet file")


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
            pf = pq.ParquetFile(path)
            column = _detect_text_column(pf)
            for batch in pf.iter_batches(batch_size=512):
                col_index = batch.schema.get_field_index(column)
                col = batch.column(col_index)
                texts = col.to_pylist()
                for value in texts:
                    text = normalize.normalize_text(value)
                    if text:
                        sample_id = shard.compute_sample_id(source_uri, record_index, text)
                        writer.write(sample_id, text, dataset_type)
                    record_index += 1
    return {"records": writer.total_records, "shards": writer.shard_paths}
