"""Training and encoding benchmarks on a few MB of Project Gutenberg text.

Every timing except the naive trainer's is the best of five runs.

    python benchmarks/run.py            # writes benchmarks/results.json

The books are downloaded once into benchmarks/.cache.
"""

from __future__ import annotations

import gc
import hashlib
import json
import platform
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

from bytepair import GPT4Tokenizer, RegexTokenizer
from bytepair.train import train_bpe, train_bpe_naive

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"

# what the six books hashed to when the README numbers were taken; Gutenberg
# revises its files now and then
CORPUS_SHA256 = "48a4b88020c0ac521f4254b1f7260d2a9d47d419e94fbfc492dc47fefb97fd31"

BOOKS = {
    1342: "Pride and Prejudice",
    2701: "Moby Dick",
    1661: "The Adventures of Sherlock Holmes",
    84: "Frankenstein",
    98: "A Tale of Two Cities",
    345: "Dracula",
}


def gutenberg_text(book_id: int) -> str:
    path = CACHE / f"pg{book_id}.txt"
    if not path.exists():
        CACHE.mkdir(exist_ok=True)
        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        with urllib.request.urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    # keep only the book, not the license header and footer
    start = text.index("\n", text.index("*** START OF")) + 1
    return text[start : text.index("*** END OF")].strip() + "\n"


def timed(fn, repeat: int = 1, setup=None):
    """Best of ``repeat`` runs, with the garbage collector off while timing."""
    best = float("inf")
    for _ in range(repeat):
        if setup is not None:
            setup()
        gc.collect()
        gc.disable()
        try:
            start = time.perf_counter()
            value = fn()
            best = min(best, time.perf_counter() - start)
        finally:
            gc.enable()
    return value, best


def cpu_name() -> str:
    try:
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.processor() or platform.machine()


def bench_training(text: str) -> list[dict]:
    chunks = RegexTokenizer("gpt4").split(text)
    encoded = [chunk.encode("utf-8") for chunk in chunks]
    print(f"{len(chunks):,} chunks, {len(set(chunks)):,} unique", file=sys.stderr)

    rows = []
    naive_merges = 256
    reference, naive_s = timed(lambda: train_bpe_naive(encoded, naive_merges))
    rows.append(_row("naive", naive_merges, naive_s))
    print(f"naive        {naive_merges:>6} merges  {naive_s:8.2f}s", file=sys.stderr)

    for num_merges in [256, 1024, 4096, 16_384, 32_768]:
        # both trainers start from the same pre-split chunks, so the
        # deduplication is part of what gets timed
        merges, seconds = timed(
            lambda n=num_merges: train_bpe(Counter(encoded).items(), n), repeat=5
        )
        if num_merges == naive_merges:
            assert merges == reference, "incremental and naive trainers disagree"
        rows.append(_row("incremental", num_merges, seconds))
        print(f"incremental  {num_merges:>6} merges  {seconds:8.2f}s", file=sys.stderr)

    tok = RegexTokenizer("gpt4")
    _, seconds = timed(lambda: tok.train(text, 256 + 32_768), repeat=5)
    rows.append(_row("RegexTokenizer.train, split included", 32_768, seconds))
    print(f"end to end   {32_768:>6} merges  {seconds:8.2f}s", file=sys.stderr)
    return rows


def _row(trainer: str, num_merges: int, seconds: float) -> dict:
    return {"trainer": trainer, "vocab_size": 256 + num_merges, "seconds": seconds}


def bench_encoding(text: str) -> dict:
    size_mb = len(text.encode("utf-8")) / 1e6
    gpt4, load_s = timed(GPT4Tokenizer)
    ids, cold_s = timed(
        lambda: gpt4.encode_ordinary(text), repeat=5, setup=gpt4._cache.clear
    )
    _, warm_s = timed(lambda: gpt4.encode_ordinary(text), repeat=5)
    result = {
        "tokens": len(ids),
        "bytes_per_token": round(len(text.encode("utf-8")) / len(ids), 3),
        "gpt4_tokenizer_load_seconds": round(load_s, 3),
        "bytepair_mb_per_s_cold_cache": round(size_mb / cold_s, 2),
        "bytepair_mb_per_s_warm_cache": round(size_mb / warm_s, 2),
    }
    try:
        import tiktoken
    except ImportError:
        return result
    reference = tiktoken.get_encoding("cl100k_base")
    expected, tiktoken_s = timed(lambda: reference.encode_ordinary(text), repeat=5)
    assert expected == ids, "GPT4Tokenizer disagrees with tiktoken"
    result["tiktoken_mb_per_s"] = round(size_mb / tiktoken_s, 2)
    result["identical_to_tiktoken"] = True
    return result


def main() -> None:
    text = "".join(gutenberg_text(book_id) for book_id in BOOKS)
    raw = text.encode("utf-8")
    print(f"corpus: {len(raw) / 1e6:.2f} MB", file=sys.stderr)
    sha256 = hashlib.sha256(raw).hexdigest()
    if sha256 != CORPUS_SHA256:
        print(
            f"warning: corpus sha256 is {sha256}, not the one the README "
            "numbers were measured on; a book has probably been revised",
            file=sys.stderr,
        )

    training = bench_training(text)
    naive = next(r for r in training if r["trainer"] == "naive")
    fast = next(
        r
        for r in training
        if r["trainer"] == "incremental" and r["vocab_size"] == naive["vocab_size"]
    )
    results = {
        "machine": {
            "cpu": cpu_name(),
            "python": platform.python_version(),
            "os": f"{platform.system()} {platform.release()}",
        },
        "corpus": {
            "books": [f"{title} (Gutenberg #{i})" for i, title in BOOKS.items()],
            "bytes": len(raw),
            "sha256": sha256,
            "split": "gpt4",
        },
        "training": [{**r, "seconds": round(r["seconds"], 3)} for r in training],
        "speedup_at_equal_vocab": round(naive["seconds"] / fast["seconds"], 1),
        "encoding": bench_encoding(text),
    }
    out = HERE / "results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
