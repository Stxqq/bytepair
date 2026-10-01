import pytest
from samples import EDGE_CASES

from bytepair import BasicTokenizer, RegexTokenizer, load


@pytest.mark.parametrize("make", [BasicTokenizer, RegexTokenizer])
def test_save_load_round_trip(make, alice, tmp_path):
    tok = make()
    tok.train(alice[:30_000], 400)
    tok.register_special_tokens({"<|endoftext|>": 400, 'odd "name"\t': 401})
    model = tok.save(tmp_path / "alice")
    assert model.name == "alice.model"
    assert (tmp_path / "alice.vocab").exists()

    loaded = load(model)
    assert type(loaded) is type(tok)
    assert loaded.pattern == tok.pattern
    assert loaded.merges == tok.merges
    assert list(loaded.merges) == list(tok.merges)
    assert loaded.special_tokens == tok.special_tokens
    assert loaded.vocab == tok.vocab
    for text in [alice[50_000:52_000], *EDGE_CASES]:
        assert loaded.encode(text) == tok.encode(text)


def test_custom_pattern_survives(tmp_path):
    tok = RegexTokenizer(r"\w+|\s+|[^\w\s]+")
    tok.train("one two three, four five six. " * 30, 280)
    loaded = load(tok.save(tmp_path / "words"))
    assert loaded.pattern == r"\w+|\s+|[^\w\s]+"
    assert loaded.split("a, b") == ["a", ",", " ", "b"]


def test_permuted_byte_ids_survive(tmp_path):
    tok = BasicTokenizer()
    tok.byte_ids = list(reversed(range(256)))
    tok._refresh()
    loaded = load(tok.save(tmp_path / "reversed"))
    assert loaded.byte_ids == tok.byte_ids
    assert loaded.encode("abc") == [255 - 97, 255 - 98, 255 - 99]


def test_vocab_file_shows_merge_parents(tmp_path):
    tok = BasicTokenizer()
    tok.train("aaabdaaabac", 259)
    tok.save(tmp_path / "wiki")
    lines = (tmp_path / "wiki.vocab").read_text(encoding="utf-8").splitlines()
    assert lines[10] == "    10  [\\n]"
    assert lines[256:] == [
        "   256  [a] [a] -> [aa]",
        "   257  [a] [b] -> [ab]",
        "   258  [aa] [ab] -> [aaab]",
    ]


def test_rejects_other_files(tmp_path):
    path = tmp_path / "x.model"
    path.write_text("hello\n")
    with pytest.raises(ValueError, match="not a bytepair model"):
        load(path)


@pytest.mark.parametrize(
    "body, problem",
    [
        ("pattern none\nbytes identity\nspecial 0\nmerges 2\n97 98\n", "truncated"),
        ("pattern none\nbytes identity\nspecial 1\n", "truncated"),
        ("pattern none\nbytes identity\nspecial 0\nmerges 1\n97 256\n", "merge 0"),
    ],
)
def test_rejects_broken_files(tmp_path, body, problem):
    path = tmp_path / "broken.model"
    path.write_text("bytepair 1\n" + body, encoding="utf-8")
    with pytest.raises(ValueError, match=problem):
        load(path)
