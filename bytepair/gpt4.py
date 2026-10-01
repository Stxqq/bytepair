"""GPT-4's cl100k_base, rebuilt as an ordinary merge table.

tiktoken stores the vocabulary as ``token bytes -> rank`` and never says which
two tokens were merged to make each one. The pair can be recovered: running
BPE on a token's bytes, using only merges ranked below the token itself, stops
at exactly the two halves that were joined to make it.

cl100k also numbers its 256 single-byte tokens in its own order, which is what
``byte_ids`` is for.
"""

from __future__ import annotations

import base64
import hashlib
import os
import urllib.request
from pathlib import Path

from .pairs import Pair
from .patterns import GPT4_SPLIT_PATTERN
from .tokenizers import RegexTokenizer

CL100K_URL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
CL100K_SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
CL100K_SPECIAL_TOKENS = {
    "<|endoftext|>": 100257,
    "<|fim_prefix|>": 100258,
    "<|fim_middle|>": 100259,
    "<|fim_suffix|>": 100260,
    "<|endofprompt|>": 100276,
}


class GPT4Tokenizer(RegexTokenizer):
    """The cl100k_base tokenizer. Produces the same ids as tiktoken."""

    def __init__(self, ranks: dict[bytes, int] | None = None) -> None:
        super().__init__(GPT4_SPLIT_PATTERN)
        ranks = load_cl100k_ranks() if ranks is None else ranks
        self.byte_ids = [ranks[bytes([b])] for b in range(256)]
        self.merges = merges_from_ranks(ranks)
        self.special_tokens = dict(CL100K_SPECIAL_TOKENS)
        self._refresh()

    def train(self, *args, **kwargs):
        raise NotImplementedError("GPT4Tokenizer is pretrained; use RegexTokenizer")


def merges_from_ranks(ranks: dict[bytes, int]) -> dict[Pair, int]:
    merges = {}
    for token, rank in sorted(ranks.items(), key=lambda item: item[1]):
        if len(token) > 1:
            left, right = _split_in_two(token, ranks, rank)
            merges[ranks[left], ranks[right]] = rank
    return merges


def _split_in_two(token: bytes, ranks: dict[bytes, int], max_rank: int):
    parts = [token[i : i + 1] for i in range(len(token))]
    while len(parts) > 2:
        best_rank, best_at = max_rank, -1
        for i in range(len(parts) - 1):
            rank = ranks.get(parts[i] + parts[i + 1], max_rank)
            if rank < best_rank:
                best_rank, best_at = rank, i
        if best_at < 0:
            break
        parts[best_at : best_at + 2] = [parts[best_at] + parts[best_at + 1]]
    if len(parts) != 2:
        raise ValueError(f"token {token!r} is not the merge of two ranked tokens")
    return parts[0], parts[1]


def load_cl100k_ranks() -> dict[bytes, int]:
    """cl100k_base ranks from tiktoken if installed, else from a verified download."""
    try:
        import tiktoken
    except ImportError:
        return read_tiktoken_file(_fetch_cl100k())
    # tiktoken has no public accessor for the ranks; this attribute has been
    # stable since its first release.
    return tiktoken.get_encoding("cl100k_base")._mergeable_ranks


def read_tiktoken_file(path: Path) -> dict[bytes, int]:
    ranks = {}
    for line in path.read_bytes().splitlines():
        if line:
            token, rank = line.split()
            ranks[base64.b64decode(token)] = int(rank)
    return ranks


def cache_dir() -> Path:
    if "BYTEPAIR_CACHE" in os.environ:
        return Path(os.environ["BYTEPAIR_CACHE"])
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "bytepair"


def _fetch_cl100k() -> Path:
    path = cache_dir() / "cl100k_base.tiktoken"
    if path.exists() and _sha256(path.read_bytes()) == CL100K_SHA256:
        return path
    with urllib.request.urlopen(CL100K_URL, timeout=60) as response:
        payload = response.read()
    digest = _sha256(payload)
    if digest != CL100K_SHA256:
        raise OSError(
            f"cl100k_base download has sha256 {digest}, expected {CL100K_SHA256}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    partial.write_bytes(payload)
    partial.replace(path)
    return path


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
