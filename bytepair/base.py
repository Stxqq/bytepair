"""What every tokenizer in the package shares: the vocabulary built from an
ordered merge table, the encode/decode loop and the chunk cache."""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Iterable

from .pairs import Pair, merge_pair
from .train import OnMerge, train_bpe

_NO_MERGE = float("inf")


class Tokenizer:
    """Byte-level BPE over 256 byte tokens plus an ordered table of merges.

    Merged token ids double as merge ranks: a lower id was learned earlier and
    is always applied first.
    """

    cache_size = 1 << 16
    _cacheable_chunk = 256

    def __init__(self) -> None:
        self.merges: dict[Pair, int] = {}
        # Raw byte -> token id. The identity for anything trained here; GPT-4
        # numbers its byte tokens in a different order.
        self.byte_ids: list[int] = list(range(256))
        self._refresh()

    def split(self, text: str) -> list[str]:
        """Pre-split text into chunks that merges never cross."""
        return [text] if text else []

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(
        self, text: str, vocab_size: int, on_merge: OnMerge | None = None
    ) -> None:
        """Learn ``vocab_size - 256`` merges from ``text``, replacing any old ones."""
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be at least 256, got {vocab_size}")
        chunks = Counter(self.split(text))
        merges = train_bpe(
            ((chunk.encode("utf-8"), freq) for chunk, freq in chunks.items()),
            vocab_size - 256,
            on_merge,
        )
        self.byte_ids = list(range(256))
        self.merges = {pair: 256 + i for i, pair in enumerate(merges)}
        self._refresh()

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in self.split(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        try:
            return b"".join(self.vocab[i] for i in ids)
        except KeyError as err:
            raise ValueError(f"unknown token id {err.args[0]}") from None

    def decode(self, ids: Iterable[int], errors: str = "replace") -> str:
        """Decode ids to text. Partial UTF-8 sequences become U+FFFD by default."""
        return self.decode_bytes(ids).decode("utf-8", errors=errors)

    def _encode_chunk(self, chunk: str) -> list[int]:
        ids = self._cache.get(chunk)
        if ids is not None:
            return ids
        ids = self._merge_bytes(chunk.encode("utf-8"))
        if len(chunk) <= self._cacheable_chunk:
            if len(self._cache) >= self.cache_size:
                self._cache.clear()
            self._cache[chunk] = ids
        return ids

    def _merge_bytes(self, raw: bytes) -> list[int]:
        byte_ids, merges = self.byte_ids, self.merges
        ids = [byte_ids[b] for b in raw]
        while len(ids) > 1:
            pair = min(zip(ids, ids[1:]), key=lambda p: merges.get(p, _NO_MERGE))
            new_id = merges.get(pair)
            if new_id is None:
                break
            ids = merge_pair(ids, pair, new_id)
        return ids

    def _refresh(self) -> None:
        vocab = {token: bytes([b]) for b, token in enumerate(self.byte_ids)}
        for (a, b), new_id in self.merges.items():
            vocab[new_id] = vocab[a] + vocab[b]
        self.vocab: dict[int, bytes] = vocab
        self._cache: dict[str, list[int]] = {}


def render_token(token: bytes) -> str:
    """Printable form of a token: invalid UTF-8 shows as U+FFFD and control
    characters are escaped, so every token fits on one line."""
    text = token.decode("utf-8", errors="replace")
    return "".join(
        repr(ch)[1:-1] if unicodedata.category(ch)[0] == "C" else ch for ch in text
    )
