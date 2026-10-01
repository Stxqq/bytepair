import io

import pytest

from bytepair import load
from bytepair.cli import main


@pytest.fixture
def model(tmp_path, capsys, alice):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(alice[:30_000])
    assert main(["train", str(corpus), "-o", str(tmp_path / "fox"), "-v", "300"]) == 0
    assert "wrote" in capsys.readouterr().out
    return tmp_path / "fox.model"


def test_train_writes_a_loadable_model(model):
    assert load(model).vocab_size == 300


def test_encode_then_decode(model, capsys):
    main(["encode", "-m", str(model), "the White Rabbit"])
    ids = capsys.readouterr().out.split()
    assert len(ids) < len("the White Rabbit")
    main(["decode", "-m", str(model), *ids])
    assert capsys.readouterr().out == "the White Rabbit"


def test_model_suffix_is_optional(model, capsys):
    main(["encode", "-m", str(model.with_suffix("")), "dog"])
    assert capsys.readouterr().out.strip()


def test_reads_stdin(model, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("brown dog"))
    main(["encode", "-m", str(model)])
    ids = capsys.readouterr().out
    monkeypatch.setattr("sys.stdin", io.StringIO(ids))
    main(["decode", "-m", str(model)])
    assert capsys.readouterr().out == "brown dog"


def test_inspect_plain(model, capsys):
    main(["inspect", "-m", str(model), "--color", "never", "the dog\n"])
    tokens, ids, summary = capsys.readouterr().out.splitlines()
    assert tokens.replace("|", "") == "the dog\\n"
    assert len(ids.split()) == tokens.count("|") + 1
    assert summary.startswith("8 chars, 8 bytes,")


def test_inspect_color_uses_pastel_backgrounds(model, capsys):
    main(["inspect", "-m", str(model), "--color", "always", "the dog"])
    assert "\x1b[48;2;253;230;138m" in capsys.readouterr().out


def test_errors_exit_nonzero(tmp_path, capsys):
    assert main(["encode", "-m", str(tmp_path / "missing.model"), "x"]) == 1
    assert capsys.readouterr().err.startswith("bytepair: ")
