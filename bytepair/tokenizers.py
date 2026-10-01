from __future__ import annotations

import regex

from .base import Tokenizer
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
