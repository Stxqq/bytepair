import random
import sys
from pathlib import Path

import pytest
from samples import EDGE_CASES

from bytepair import load
from bytepair.gpt4 import GPT4Tokenizer, load_cl100k_ranks

tiktoken = pytest.importorskip("tiktoken")


@pytest.fixture(scope="module")
def reference():
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception as err:  # offline and nothing cached
        pytest.skip(f"cl100k_base unavailable: {err}")


@pytest.fixture(scope="module")
def gpt4(reference):
    return GPT4Tokenizer()


def random_text(rng: random.Random) -> str:
    blocks = [
        (0x20, 0x7E),
        (0x00, 0x1F),
        (0xA0, 0x24F),
        (0x370, 0x3FF),
        (0x400, 0x4FF),
        (0x590, 0x6FF),
        (0x900, 0x97F),
        (0x3040, 0x30FF),
        (0x4E00, 0x4FFF),
        (0xAC00, 0xAD00),
        (0x1F300, 0x1FAFF),
    ]
    pieces = []
    for _ in range(rng.randint(1, 40)):
        if rng.random() < 0.3:
            pieces.append(
                rng.choice([" ", "  ", "\n", "\n\n", "\t", "'s", "'LL", "123456"])
            )
        else:
            lo, hi = rng.choice(blocks)
            pieces.append(
                "".join(chr(rng.randint(lo, hi)) for _ in range(rng.randint(1, 8)))
            )
    return "".join(pieces)


def corpus(alice: str) -> list[str]:
    source = Path(__file__).resolve().parent.parent / "bytepair"
    rng = random.Random(0)
    return [
        *EDGE_CASES,
        alice[:20_000],
        *(path.read_text() for path in sorted(source.glob("*.py"))),
        *(random_text(rng) for _ in range(400)),
    ]


def test_matches_tiktoken(gpt4, reference, alice):
    for text in corpus(alice):
        expected = reference.encode_ordinary(text)
        assert gpt4.encode_ordinary(text) == expected, text[:80]
        assert gpt4.decode(expected) == reference.decode(expected)


def test_matches_tiktoken_with_special_tokens(gpt4, reference):
    text = "<|endoftext|>hello<|fim_prefix|>def f():<|fim_suffix|> x<|endofprompt|>"
    ids = gpt4.encode(text, allowed_special="all")
    assert ids == reference.encode(text, allowed_special="all")
    assert gpt4.decode(ids) == text
    with pytest.raises(ValueError):
        gpt4.encode(text)


def test_known_ids(gpt4):
    assert gpt4.encode("hello world") == [15339, 1917]
    assert gpt4.vocab_size == 100_256 + 5


def test_cannot_be_trained(gpt4):
    with pytest.raises(NotImplementedError):
        gpt4.train("text", 300)


def test_save_load_keeps_byte_order(gpt4, tmp_path, alice):
    loaded = load(gpt4.save(tmp_path / "cl100k"))
    text = alice[:5_000]
    assert loaded.encode_ordinary(text) == gpt4.encode_ordinary(text)


def test_download_path_without_tiktoken(monkeypatch, tmp_path, reference):
    monkeypatch.setenv("BYTEPAIR_CACHE", str(tmp_path))
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    try:
        ranks = load_cl100k_ranks()
    except OSError as err:
        pytest.skip(f"no network: {err}")
    assert (tmp_path / "cl100k_base.tiktoken").exists()
    assert ranks == reference._mergeable_ranks
