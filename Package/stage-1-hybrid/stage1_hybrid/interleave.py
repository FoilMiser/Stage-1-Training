"""Small, fair round-robin interleaver with bounded per-source buffers."""
from __future__ import annotations

from collections import deque
from typing import Iterable, Iterator, List, TypeVar

T = TypeVar("T")


def interleave(iterables: Iterable[Iterator[T]], per_source_buffer: int = 1024) -> Iterator[T]:
    """Yield items from multiple iterables in a fair, low-memory fashion."""

    if per_source_buffer <= 0:
        raise ValueError("per_source_buffer must be > 0")

    iterators: List[Iterator[T]] = [iter(it) for it in iterables]
    buffers = [deque() for _ in iterators]
    active = len(iterators)

    while active:
        progress = False
        for idx, iterator in enumerate(iterators):
            if iterator is None:
                continue
            buffer = buffers[idx]
            if not buffer:
                try:
                    while len(buffer) < per_source_buffer:
                        buffer.append(next(iterator))
                        if buffer:
                            break
                except StopIteration:
                    iterators[idx] = None
                    active -= 1
                    continue
            if buffer:
                progress = True
                yield buffer.popleft()
        if not progress:
            break


__all__ = ["interleave"]
