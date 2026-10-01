"""The on-disk format.

A ``.model`` file is plain text that ``load`` reads back exactly:

    bytepair 1
    pattern "<split regex as a JSON string>"      (pattern none for Basic)
    bytes identity                                (or the 256 byte token ids)
    special 1
    100257 "<|endoftext|>"
    merges 2
    116 104
    32 256

Merge ``i`` (counting from 0) creates token ``256 + i``. The ``.vocab`` file
written next to it is for people only and is never read back.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .pairs import Pair

MAGIC = "bytepair 1"


@dataclass
class ModelSpec:
    pattern: str | None
    byte_ids: list[int]
    special_tokens: dict[str, int]
    merges: list[Pair]


def dump_model(spec: ModelSpec) -> str:
    pattern = "none" if spec.pattern is None else json.dumps(spec.pattern)
    if spec.byte_ids == list(range(256)):
        byte_ids = "identity"
    else:
        byte_ids = " ".join(map(str, spec.byte_ids))
    lines = [MAGIC, f"pattern {pattern}", f"bytes {byte_ids}"]
    lines.append(f"special {len(spec.special_tokens)}")
    lines += [f"{i} {json.dumps(name)}" for name, i in spec.special_tokens.items()]
    lines.append(f"merges {len(spec.merges)}")
    lines += [f"{a} {b}" for a, b in spec.merges]
    return "\n".join(lines) + "\n"


def parse_model(text: str) -> ModelSpec:
    lines = iter(text.splitlines())
    if next(lines, None) != MAGIC:
        raise ValueError(f"not a bytepair model (expected {MAGIC!r} header)")

    pattern = _field(lines, "pattern")
    byte_ids = _field(lines, "bytes")
    special_count = int(_field(lines, "special"))
    special_tokens = {}
    for _ in range(special_count):
        token_id, name = _next_line(lines).split(" ", 1)
        special_tokens[json.loads(name)] = int(token_id)
    merge_count = int(_field(lines, "merges"))
    merges = []
    for i in range(merge_count):
        a, b = (int(v) for v in _next_line(lines).split())
        if not (a < 256 + i and b < 256 + i):
            raise ValueError(f"merge {i} uses a token that does not exist yet")
        merges.append((a, b))

    return ModelSpec(
        pattern=None if pattern == "none" else json.loads(pattern),
        byte_ids=(list(range(256)) if byte_ids == "identity" else _ints(byte_ids, 256)),
        special_tokens=special_tokens,
        merges=merges,
    )


def _next_line(lines) -> str:
    line = next(lines, None)
    if line is None:
        raise ValueError("truncated model file")
    return line


def _field(lines, key: str) -> str:
    line = _next_line(lines)
    name, _, value = line.partition(" ")
    if name != key:
        raise ValueError(f"expected a {key!r} line, found {line!r}")
    return value


def _ints(text: str, count: int) -> list[int]:
    values = [int(v) for v in text.split()]
    if len(values) != count:
        raise ValueError(f"expected {count} byte ids, found {len(values)}")
    return values
