"""Shard writing utilities."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import IO, Optional


def compute_sample_id(source_uri: str, index: int, normalized_text: str) -> str:
    """Compute a deterministic sample identifier."""

    sha = hashlib.sha1()
    snippet = normalized_text[:200]
    sha.update(source_uri.encode("utf-8"))
    sha.update(str(index).encode("utf-8"))
    sha.update(snippet.encode("utf-8"))
    return sha.hexdigest()


@dataclass
class ShardStats:
    records: int = 0
    shards: int = 0


class ShardWriter:
    """Incrementally writes normalized JSONL shards with deduplication."""

    def __init__(
        self,
        output_dir: str,
        sample_type: str,
        max_records: int = 20000,
        encoding: str = "utf-8",
    ) -> None:
        self.output_dir = output_dir
        self.sample_type = sample_type
        self.max_records = max_records
        self.encoding = encoding
        self._shard_index = 0
        self._records_in_shard = 0
        self._total_records = 0
        self._fh: Optional[IO[str]] = None
        self._dedup_hashes: set[str] = set()
        os.makedirs(self.output_dir, exist_ok=True)

    def _open_new_shard(self) -> None:
        if self._fh:
            self._fh.close()
        filename = f"part-{self._shard_index:05d}.jsonl"
        path = os.path.join(self.output_dir, filename)
        self._fh = open(path, "w", encoding=self.encoding)
        self._records_in_shard = 0
        self._dedup_hashes = set()
        self._shard_index += 1

    @property
    def shard_paths(self) -> list[str]:
        return [
            os.path.join(self.output_dir, name)
            for name in sorted(os.listdir(self.output_dir))
            if name.endswith(".jsonl")
        ]

    def write(self, sample_id: str, text: str, sample_type: Optional[str] = None) -> bool:
        """Write a normalized sample to the current shard.

        Returns ``True`` if the sample was written, ``False`` if it was deduplicated.
        """

        if not text:
            return False
        hashed_text = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if hashed_text in self._dedup_hashes:
            return False
        if self._fh is None or self._records_in_shard >= self.max_records:
            self._open_new_shard()
        payload = {
            "sample_id": sample_id,
            "type": sample_type or self.sample_type,
            "text": text,
        }
        assert self._fh is not None
        self._fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._records_in_shard += 1
        self._total_records += 1
        self._dedup_hashes.add(hashed_text)
        return True

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def total_records(self) -> int:
        return self._total_records

    @property
    def num_shards(self) -> int:
        return max(self._shard_index if self._records_in_shard else self._shard_index - 1, 0)
