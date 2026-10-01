from __future__ import annotations

from pathlib import Path

import regex

from .base import Tokenizer
from .modelfile import parse_model
from .patterns import NAMED_PATTERNS


class BasicTokenizer(Tokenizer):
    """BPE over the raw byte stream: merges may cross word boundaries."""


class RegexTokenizer(Tokenizer):
    """BPE inside the chunks of a split pattern, like GPT-2 and GPT-4.

    ``pattern`` is ``"gpt4"``, ``"gpt2"`` or any regex for the ``regex`` module.
    """

    def __init__(self, pattern: str = "gpt4") -> None:
        self.pattern = NAMED_PATTERNS.get(pattern, pattern)
        self._splitter = regex.compile(self.pattern)
        super().__init__()

    def split(self, text: str) -> list[str]:
        return self._splitter.findall(text)


def load(path: str | Path) -> Tokenizer:
    """Load a tokenizer written by ``Tokenizer.save``."""
    spec = parse_model(Path(path).read_text(encoding="utf-8"))
    tok = BasicTokenizer() if spec.pattern is None else RegexTokenizer(spec.pattern)
    tok.byte_ids = spec.byte_ids
    tok.merges = {pair: 256 + i for i, pair in enumerate(spec.merges)}
    tok.special_tokens = spec.special_tokens
    tok._refresh()
    return tok
