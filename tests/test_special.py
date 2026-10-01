import re

import pytest

from bytepair import RegexTokenizer

EOT, FIM = "<|endoftext|>", "<|fim_prefix|>"


@pytest.fixture
def tok(alice):
    tok = RegexTokenizer()
    tok.train(alice[:20_000], 300)
    tok.register_special_tokens({EOT: 1000, FIM: 1001})
    return tok


def test_all_encodes_specials_as_single_ids(tok):
    ids = tok.encode(f"hello{EOT}world{FIM}", allowed_special="all")
    assert ids.count(1000) == 1 and ids.count(1001) == 1
    assert tok.decode(ids) == f"hello{EOT}world{FIM}"


def test_default_refuses_text_with_specials(tok):
    with pytest.raises(ValueError, match="special token"):
        tok.encode(f"hello{EOT}")
    assert tok.encode("hello") == tok.encode_ordinary("hello")


def test_none_treats_specials_as_text(tok):
    ids = tok.encode(f"a{EOT}b", allowed_special="none")
    assert 1000 not in ids
    assert ids == tok.encode_ordinary(f"a{EOT}b")
    assert tok.decode(ids) == f"a{EOT}b"


def test_set_allows_only_named_specials(tok):
    ids = tok.encode(f"x{EOT}y", allowed_special={EOT})
    assert 1000 in ids
    with pytest.raises(ValueError, match="fim_prefix"):
        tok.encode(f"x{EOT}y{FIM}", allowed_special={EOT})


def test_bad_allowed_special_values(tok):
    with pytest.raises(ValueError, match="unknown special"):
        tok.encode("x", allowed_special={"<|nope|>"})
    with pytest.raises(ValueError, match="must be"):
        tok.encode("x", allowed_special=EOT)


def test_adjacent_and_repeated_specials(tok):
    text = f"{EOT}{EOT}hello{FIM}{EOT}"
    ids = tok.encode(text, allowed_special="all")
    assert ids[:2] == [1000, 1000] and ids[-2:] == [1001, 1000]
    assert tok.decode(ids) == text


def test_special_id_may_not_shadow_a_regular_token(tok):
    with pytest.raises(ValueError, match="reuses id 256"):
        tok.register_special_tokens({"<|bad|>": 256})


def test_vocab_size_includes_specials(tok):
    assert tok.vocab_size == 300 + 2


def test_failed_registration_leaves_tokenizer_unchanged(tok):
    before = tok.encode(f"hi{EOT}", allowed_special="all")
    with pytest.raises(ValueError):
        tok.register_special_tokens({"<|ok|>": 2000, "<|bad|>": 299})
    assert set(tok.special_tokens) == {EOT, FIM}
    assert tok.encode(f"hi{EOT}", allowed_special="all") == before


def test_train_refuses_specials_inside_the_new_vocab(tok, alice):
    merges, text = dict(tok.merges), alice[:2_000]
    ids = tok.encode(text)
    with pytest.raises(ValueError, match=re.escape(f"{EOT!r} has id 1000")):
        tok.train(text, 1200)
    assert tok.merges == merges
    assert tok.decode(tok.encode(text)) == text
    assert tok.encode(text) == ids
