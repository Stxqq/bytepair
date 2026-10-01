"""count_pairs and merge_pair, shared by the reference trainer and the tests."""

from __future__ import annotations

from collections.abc import Sequence

Pair = tuple[int, int]


def count_pairs(
    ids: Sequence[int], counts: dict[Pair, int] | None = None
) -> dict[Pair, int]:
    """Count adjacent pairs in ``ids``, adding into ``counts`` if given."""
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge_pair(ids: Sequence[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair``, left to right."""
    a, b = pair
    merged = []
    i, n = 0, len(ids)
    while i < n:
        if ids[i] == a and i + 1 < n and ids[i + 1] == b:
            merged.append(new_id)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    return merged
