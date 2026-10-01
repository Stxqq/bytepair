"""Pre-tokenization split patterns.

Both are the possessive forms tiktoken ships for r50k_base (GPT-2) and
cl100k_base (GPT-4). They split exactly like the originals, just faster.
"""

GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|"""
    r"""\s++$|\s+(?!\S)|\s"""
)

GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+|"""
    r""" ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
)

NAMED_PATTERNS = {"gpt2": GPT2_SPLIT_PATTERN, "gpt4": GPT4_SPLIT_PATTERN}
