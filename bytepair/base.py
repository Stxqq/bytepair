"""What every tokenizer in the package shares: the vocabulary built from an
ordered merge table, the encode/decode loop and the chunk cache."""

from __future__ import annotations

import heapq
import unicodedata
from collections import Counter
from collections.abc import Collection, Iterable
from pathlib import Path
from typing import Literal

import regex

from .modelfile import ModelSpec, dump_model
from .pairs import Pair
from .train import OnMerge, train_bpe

AllowedSpecial = Literal["all", "none", "none_raise"] | Collection[str]


class Tokenizer:
    """Byte-level BPE over 256 byte tokens plus an ordered table of merges.

    Merged token ids double as merge ranks: a lower id was learned earlier and
    is always applied first.
    """

    pattern: str | None = None
    cache_size = 1 << 16
    # Long chunks (whitespace runs, minified code) rarely repeat; caching them
    # would only push out the short ones that do.
    _cacheable_chunk = 256

    def __init__(self) -> None:
        self.merges: dict[Pair, int] = {}
        # Raw byte -> token id. The identity for anything trained here; GPT-4
        # numbers its byte tokens in a different order.
        self.byte_ids: list[int] = list(range(256))
        self.special_tokens: dict[str, int] = {}
        self._refresh()

    def split(self, text: str) -> list[str]:
        """Pre-split text into chunks that merges never cross."""
        return [text] if text else []

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    def register_special_tokens(self, tokens: dict[str, int]) -> None:
        """Add special tokens such as ``{"<|endoftext|>": 100257}``."""
        _check_special_ids(tokens, self.vocab)
        self.special_tokens.update(tokens)
        self._refresh()

    def train(
        self, text: str, vocab_size: int, on_merge: OnMerge | None = None
    ) -> None:
        """Learn ``vocab_size - 256`` merges from ``text``, replacing any old ones."""
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be at least 256, got {vocab_size}")
        # checked up front: finding out after a long training run is no help
        for name, token_id in self.special_tokens.items():
            if token_id < vocab_size:
                raise ValueError(
                    f"special token {name!r} has id {token_id}, which training to "
                    f"vocab_size {vocab_size} would give to a regular token"
                )
        chunks = Counter(self.split(text))
        merges = train_bpe(
            ((chunk.encode("utf-8"), freq) for chunk, freq in chunks.items()),
            vocab_size - 256,
            on_merge,
        )
        self.byte_ids = list(range(256))
        self.merges = {pair: 256 + i for i, pair in enumerate(merges)}
        self._refresh()

    def encode(
        self, text: str, allowed_special: AllowedSpecial = "none_raise"
    ) -> list[int]:
        """Encode text to token ids.

        ``allowed_special`` decides what happens to special tokens that appear
        in the text, with the same rules as tiktoken: ``"all"`` turns each into
        its special id, a set of names allows only those, ``"none"`` encodes
        them as ordinary text and ``"none_raise"`` (the default) refuses to
        encode text that contains any of them.
        """
        allowed = self._allowed_special(allowed_special, text)
        if not allowed:
            return self.encode_ordinary(text)
        names = sorted(allowed, key=len, reverse=True)
        parts = regex.split("(" + "|".join(map(regex.escape, names)) + ")", text)
        ids: list[int] = []
        for i, part in enumerate(parts):
            # splitting on a capture group alternates text, special, text, ...
            if i % 2:
                ids.append(allowed[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text, treating special token strings as plain text."""
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            # Lone surrogates, e.g. from JSON with half an escaped pair. tiktoken
            # turns them into U+FFFD, and so does the browser's TextEncoder.
            text = text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
        ids: list[int] = []
        for chunk in self.split(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        try:
            return b"".join(self._id_to_bytes[i] for i in ids)
        except KeyError as err:
            raise ValueError(f"unknown token id {err.args[0]}") from None

    def decode(self, ids: Iterable[int], errors: str = "replace") -> str:
        """Decode ids to text. Partial UTF-8 sequences become U+FFFD by default."""
        return self.decode_bytes(ids).decode("utf-8", errors=errors)

    def save(self, prefix: str | Path) -> Path:
        """Write ``prefix.model`` (loadable) and ``prefix.vocab`` (readable)."""
        merge_ids = list(self.merges.values())
        if merge_ids != list(range(256, 256 + len(merge_ids))):
            raise ValueError("merged token ids must run contiguously from 256")
        spec = ModelSpec(
            pattern=self.pattern,
            byte_ids=self.byte_ids,
            special_tokens=self.special_tokens,
            merges=list(self.merges),
        )
        model_path = Path(f"{prefix}.model")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text(dump_model(spec), encoding="utf-8")
        Path(f"{prefix}.vocab").write_text(self._vocab_listing(), encoding="utf-8")
        return model_path

    def _vocab_listing(self) -> str:
        parents = {new_id: pair for pair, new_id in self.merges.items()}
        lines = []
        for token_id in sorted(self.vocab):
            token = f"[{render_token(self.vocab[token_id])}]"
            if token_id in parents:
                a, b = parents[token_id]
                left = render_token(self.vocab[a])
                right = render_token(self.vocab[b])
                token = f"[{left}] [{right}] -> {token}"
            lines.append(f"{token_id:>6}  {token}")
        for name, token_id in sorted(self.special_tokens.items(), key=lambda t: t[1]):
            lines.append(f"{token_id:>6}  {name}  special")
        return "\n".join(lines) + "\n"

    def _allowed_special(self, allowed_special, text) -> dict[str, int]:
        if allowed_special == "all":
            return self.special_tokens
        if allowed_special == "none":
            return {}
        if allowed_special == "none_raise":
            allowed = set()
        elif isinstance(allowed_special, str):
            raise ValueError(
                f"allowed_special must be 'all', 'none', 'none_raise' or a set "
                f"of names, got {allowed_special!r}"
            )
        else:
            allowed = set(allowed_special)
            unknown = allowed - self.special_tokens.keys()
            if unknown:
                raise ValueError(f"unknown special tokens: {sorted(unknown)}")
        for name in self.special_tokens.keys() - allowed:
            if name in text:
                raise ValueError(
                    f"text contains the special token {name!r}. Pass "
                    f"allowed_special={{{name!r}}} or 'all' to encode it as a "
                    f"special token, or 'none' to encode it as plain text."
                )
        return {name: self.special_tokens[name] for name in allowed}

    def _encode_chunk(self, chunk: str) -> list[int]:
        ids = self._cache.get(chunk)
        if ids is not None:
            return ids
        ids = self._merge_bytes(chunk.encode("utf-8"))
        if len(chunk) <= self._cacheable_chunk:
            if len(self._cache) >= self.cache_size:
                # cheaper than LRU bookkeeping on every hit; it refills quickly
                self._cache.clear()
            self._cache[chunk] = ids
        return ids

    def _merge_bytes(self, raw: bytes) -> list[int]:
        # Rescanning every pair after each merge is quadratic in the chunk
        # length, which a long run of letters or CJK text makes painful. Instead
        # every adjacent pair sits in a heap as (merge id, position) and the
        # parts form a linked list. The position breaks ties, so the leftmost
        # of two equal pairs still merges first. Even on 4-byte chunks this is
        # about twice as fast as the rescan.
        ids = [self.byte_ids[b] for b in raw]
        merges = self.merges
        n = len(ids)
        nxt = list(range(1, n + 1))
        prev = list(range(-1, n - 1))
        heap = [
            (new_id, i)
            for i in range(n - 1)
            if (new_id := merges.get((ids[i], ids[i + 1]))) is not None
        ]
        heapq.heapify(heap)
        while heap:
            new_id, i = heapq.heappop(heap)
            j = nxt[i]
            # stale: i was absorbed by its left neighbour, or a side changed
            if ids[i] < 0 or j >= n or merges.get((ids[i], ids[j])) != new_id:
                continue
            ids[i], ids[j] = new_id, -1
            nxt[i] = nxt[j]
            if nxt[i] < n:
                prev[nxt[i]] = i
                right = merges.get((new_id, ids[nxt[i]]))
                if right is not None:
                    heapq.heappush(heap, (right, i))
            if prev[i] >= 0:
                left = merges.get((ids[prev[i]], new_id))
                if left is not None:
                    heapq.heappush(heap, (left, prev[i]))
        return [token for token in ids if token >= 0]

    def _refresh(self) -> None:
        vocab = {token: bytes([b]) for b, token in enumerate(self.byte_ids)}
        for (a, b), new_id in self.merges.items():
            vocab[new_id] = vocab[a] + vocab[b]
        _check_special_ids(self.special_tokens, vocab)
        self.vocab: dict[int, bytes] = vocab
        self._id_to_bytes = vocab | {
            token_id: name.encode("utf-8")
            for name, token_id in self.special_tokens.items()
        }
        self._cache: dict[str, list[int]] = {}


def _check_special_ids(special_tokens: dict[str, int], vocab: dict[int, bytes]):
    for name, token_id in special_tokens.items():
        if token_id in vocab:
            raise ValueError(
                f"special token {name!r} reuses id {token_id}, "
                f"which is already the regular token {vocab[token_id]!r}"
            )


def render_token(token: bytes) -> str:
    """Printable form of a token. Bytes that are not valid UTF-8 on their own
    and control characters are escaped, so every token fits on one line."""
    text = token.decode("utf-8", errors="backslashreplace")
    return "".join(
        repr(ch)[1:-1] if unicodedata.category(ch)[0] == "C" else ch for ch in text
    )
