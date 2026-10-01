import pytest
from samples import EDGE_CASES

from bytepair import BasicTokenizer, RegexTokenizer
from bytepair.base import render_token

SENTENCES = (
    "the cat sat on the mat. the dog sat on the log. "
    "it's the cat's mat, isn't it? 2024 2025 2026."
)
TRAINING_TEXT = "\n".join(EDGE_CASES) * 3 + SENTENCES * 20


def make(kind):
    return BasicTokenizer() if kind == "basic" else RegexTokenizer(kind)


@pytest.mark.parametrize("kind", ["basic", "gpt2", "gpt4"])
def test_untrained_tokenizer_is_plain_utf8(kind):
    tok = make(kind)
    text = "héllo 👋"
    assert tok.encode(text) == list(text.encode("utf-8"))
    assert tok.decode(tok.encode(text)) == text


@pytest.mark.parametrize("kind", ["basic", "gpt2", "gpt4"])
@pytest.mark.parametrize("text", EDGE_CASES)
def test_round_trip(kind, text, trained):
    tok = trained[kind]
    assert tok.decode(tok.encode(text)) == text


@pytest.fixture(scope="module")
def trained():
    tokenizers = {}
    for kind in ["basic", "gpt2", "gpt4"]:
        tok = make(kind)
        tok.train(TRAINING_TEXT, 256 + 120)
        tokenizers[kind] = tok
    return tokenizers


def test_wikipedia_example():
    tok = BasicTokenizer()
    tok.train("aaabdaaabac", 256 + 3)
    assert tok.encode("aaabdaaabac") == [258, 100, 258, 97, 99]
    assert tok.vocab[258] == b"aaab"
    assert tok.decode([258, 100, 258, 97, 99]) == "aaabdaaabac"


def test_regex_merges_never_cross_chunks():
    tok = RegexTokenizer("gpt4")
    tok.train("ab ab ab ab ab ab", 300)
    merged = [tok.vocab[i] for i in tok.merges.values()]
    assert merged
    for token in merged:
        assert len(tok.split(token.decode("utf-8"))) == 1


def test_compresses_what_it_was_trained_on(trained):
    text = TRAINING_TEXT
    for tok in trained.values():
        assert len(tok.encode(text)) < 0.6 * len(text.encode("utf-8"))


def test_training_is_deterministic(alice):
    sample = alice[:40_000]
    first, second = RegexTokenizer(), RegexTokenizer()
    first.train(sample, 600)
    second.train(sample, 600)
    assert first.merges == second.merges
    assert list(first.merges) == list(second.merges)


def test_vocab_size_counts_bytes_and_merges(trained):
    assert trained["gpt4"].vocab_size == 256 + 120


def test_rejects_vocab_smaller_than_bytes():
    with pytest.raises(ValueError):
        BasicTokenizer().train("abc", 100)


def test_decode_unknown_id():
    with pytest.raises(ValueError, match="unknown token id 999"):
        BasicTokenizer().decode([104, 999])


def test_decode_partial_utf8_is_replaced():
    tok = BasicTokenizer()
    euro = list("€".encode())
    assert tok.decode(euro[:2]) == "�"
    assert tok.decode_bytes(euro[:2]) == b"\xe2\x82"


def test_cache_does_not_change_results(trained):
    tok = trained["gpt4"]
    text = "the cat sat on the mat " * 10
    first = tok.encode(text)
    assert tok.encode(text) == first
    tok._cache.clear()
    assert tok.encode(text) == first


def test_render_token():
    assert render_token(b"hello") == "hello"
    assert render_token(b"\n\t") == "\\n\\t"
    assert render_token(b"\xe2\x82") == "�"
