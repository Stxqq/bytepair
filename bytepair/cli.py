"""Command line interface: ``bytepair train | encode | decode | inspect``."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from .base import Tokenizer, render_token
from .tokenizers import BasicTokenizer, RegexTokenizer, load

PRETRAINED = ("cl100k_base", "gpt4")

# The portfolio's note colors as (fill, ink).
PASTELS = [
    ((0xFD, 0xE6, 0x8A), (0x78, 0x35, 0x0F)),
    ((0xBA, 0xE6, 0xFD), (0x0C, 0x4A, 0x6E)),
    ((0xFB, 0xCF, 0xE8), (0x83, 0x18, 0x43)),
    ((0xBB, 0xF7, 0xD0), (0x14, 0x53, 0x2D)),
    ((0xDD, 0xD6, 0xFE), (0x4C, 0x1D, 0x95)),
    ((0xFE, 0xD7, 0xAA), (0x7C, 0x2D, 0x12)),
]
GRAY = (0x9A, 0x9A, 0xA2)
RESET = "\x1b[0m"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.run(args)
    except (ValueError, OSError) as err:
        print(f"bytepair: {err}", file=sys.stderr)
        return 1
    except BrokenPipeError:
        return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bytepair", description="Byte-level BPE tokenizer."
    )
    commands = parser.add_subparsers(required=True, metavar="command")

    train = commands.add_parser("train", help="learn merges from a text file")
    train.add_argument("corpus", type=Path, help="UTF-8 text file")
    train.add_argument("-o", "--out", required=True, help="output prefix")
    train.add_argument("-v", "--vocab-size", type=int, default=1024)
    train.add_argument(
        "-s",
        "--split",
        default="gpt4",
        help="pre-split pattern: gpt4, gpt2, none, or a regex (default gpt4)",
    )
    train.set_defaults(run=_train)

    _add_text_command(commands, "encode", _encode, "text to token ids")
    inspect = _add_text_command(
        commands, "inspect", _inspect, "show how text splits into tokens"
    )
    inspect.add_argument("--color", choices=["auto", "always", "never"], default="auto")

    decode = commands.add_parser("decode", help="token ids to text")
    _add_model_option(decode)
    decode.add_argument("ids", nargs="*", type=int, help="ids (default: stdin)")
    decode.set_defaults(run=_decode)
    return parser


def _add_text_command(commands, name, run, helptext):
    command = commands.add_parser(name, help=helptext)
    _add_model_option(command)
    command.add_argument("text", nargs="?", help="text (default: stdin)")
    command.add_argument(
        "--allow-special",
        action="store_true",
        help="encode special tokens like <|endoftext|> as their ids "
        "instead of as plain text",
    )
    command.set_defaults(run=run)
    return command


def _add_model_option(command):
    command.add_argument(
        "-m",
        "--model",
        default="cl100k_base",
        help="a .model file, or cl100k_base (default)",
    )


def _load_model(name: str) -> Tokenizer:
    if name in PRETRAINED:
        from .gpt4 import GPT4Tokenizer

        return GPT4Tokenizer()
    path = Path(name)
    if not path.exists() and path.with_suffix(".model").exists():
        path = path.with_suffix(".model")
    return load(path)


def _train(args) -> int:
    text = args.corpus.read_text(encoding="utf-8")
    if args.split == "none":
        tok: Tokenizer = BasicTokenizer()
    else:
        tok = RegexTokenizer(args.split)
    num_merges = args.vocab_size - 256
    width = len(str(num_merges))
    live = sys.stderr.isatty()
    learned = {i: bytes([i]) for i in range(256)}

    def report(new_id, pair, count):
        learned[new_id] = learned[pair[0]] + learned[pair[1]]
        done = new_id - 255
        if live and (done % 16 == 0 or done == num_merges):
            a, b = (render_token(learned[i]) for i in pair)
            line = f"merge {done:>{width}}/{num_merges}  [{a}] [{b}]  x{count}"
            print(f"\r\x1b[2K{line}", end="", file=sys.stderr, flush=True)
        elif done % 500 == 0:
            print(f"merge {done}/{num_merges}", file=sys.stderr)

    start = time.perf_counter()
    tok.train(text, args.vocab_size, on_merge=report)
    elapsed = time.perf_counter() - start
    if live:
        print(file=sys.stderr)

    model = tok.save(args.out)
    size = len(text.encode("utf-8"))
    tokens = len(tok.encode_ordinary(text))
    print(
        f"{len(tok.merges)} merges in {elapsed:.2f}s, "
        f"{size:,} bytes -> {tokens:,} tokens ({size / tokens:.2f} bytes/token)\n"
        f"wrote {model} and {model.with_suffix('.vocab')}"
    )
    return 0


def _read_text(args) -> str:
    return args.text if args.text is not None else sys.stdin.read()


def _encode_text(tok: Tokenizer, args) -> list[int]:
    allowed = "all" if args.allow_special else "none"
    return tok.encode(_read_text(args), allowed_special=allowed)


def _encode(args) -> int:
    tok = _load_model(args.model)
    print(" ".join(map(str, _encode_text(tok, args))))
    return 0


def _decode(args) -> int:
    tok = _load_model(args.model)
    ids = args.ids or [int(i) for i in sys.stdin.read().split()]
    sys.stdout.write(tok.decode(ids))
    if sys.stdout.isatty():
        sys.stdout.write("\n")
    return 0


def _inspect(args) -> int:
    tok = _load_model(args.model)
    ids = _encode_text(tok, args)
    colored = args.color == "always" or (
        args.color == "auto" and sys.stdout.isatty() and "NO_COLOR" not in os.environ
    )
    tokens = [render_token(tok.decode_bytes([i])) for i in ids]
    raw = tok.decode_bytes(ids)

    if colored:
        pieces = []
        for n, (token, token_id) in enumerate(zip(tokens, ids)):
            fill, ink = PASTELS[n % len(PASTELS)]
            pieces.append(f"{_bg(fill)}{_fg(ink)}{token}{RESET}")
            if tok.decode_bytes([token_id]).endswith(b"\n"):
                pieces.append("\n")
        print("".join(pieces))
        print()
        inks = [PASTELS[n % len(PASTELS)][1] for n in range(len(ids))]
        print(" ".join(f"{_fg(ink)}{i}{RESET}" for ink, i in zip(inks, ids)))
        print(f"\n{_fg(GRAY)}{_summary(raw, ids)}{RESET}")
    else:
        print("|".join(tokens))
        print(" ".join(map(str, ids)))
        print(_summary(raw, ids))
    return 0


def _summary(raw: bytes, ids: list[int]) -> str:
    chars = len(raw.decode("utf-8", errors="replace"))
    per_token = len(raw) / len(ids) if ids else 0.0
    return (
        f"{chars} chars, {len(raw)} bytes, {len(ids)} tokens, "
        f"{per_token:.2f} bytes/token"
    )


def _fg(rgb) -> str:
    return "\x1b[38;2;{};{};{}m".format(*rgb)


def _bg(rgb) -> str:
    return "\x1b[48;2;{};{};{}m".format(*rgb)


if __name__ == "__main__":
    sys.exit(main())
