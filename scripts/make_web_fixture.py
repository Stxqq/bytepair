"""Record what the playground's JavaScript must reproduce.

Writes scripts/fixtures/web.json: tiktoken's cl100k_base ids for a corpus of
awkward strings, and the first merges RegexTokenizer("gpt4") learns from a
slice of Alice. scripts/check_web.mjs compares the browser code against it.
"""

import json
import random
from pathlib import Path

import tiktoken

from bytepair import RegexTokenizer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "scripts" / "fixtures" / "web.json"

# Places where a JavaScript port of the split pattern tends to drift from
# tiktoken: whitespace that only one of the \s definitions knows, case folding
# in the contractions, digit runs and line endings.
CASES = [
    "hello world",
    "Hello, World! How's it going? I'M fine, they'VE said, we'Ll see.",
    "'ſ 'S 's 'ſt",
    "trailing spaces   ",
    "   leading spaces",
    "a\u0085b \u0085 next line",
    "\u0085\u0085.a \u0085﻿'s",
    "﻿byte order mark ﻿",
    "nbsp here and thin　ideographic",
    "line\r\nendings\r\rand\n\n\nblank lines\n",
    "tabs\tand\t\tmore\t",
    "1234567890 3.14159 1e-9 0xFF -42 1,000,000 ٣٤٥ ⅷ",
    "naïve café résumé coöperate é",
    "Привет, мир! Γειά σου Κόσμε",
    "こんにちは世界、東京タワー。",
    "مرحبا بالعالم שלום עולם",
    "👋🌍 👨‍👩‍👧‍👦 👍🏽 🏳️‍🌈 🇩🇪",
    "def f(x):\n    return {'a': [1, 2, 3]}  # comment\n",
    '<html><body class="x">&amp;&lt;</body></html>',
    "\x00\x01\x1c\x1d\x1e\x1f\x7f control bytes",
    "<|endoftext|>hello<|fim_prefix|>def f():<|fim_suffix|> x<|endofprompt|>",
    " " * 40 + "x",
    "!!!???...\n\n---\n",
]


def random_text(rng: random.Random) -> str:
    blocks = [
        (0x20, 0x7E),
        (0x00, 0x1F),
        (0x80, 0x24F),
        (0x370, 0x4FF),
        (0x590, 0x6FF),
        (0x900, 0x97F),
        (0x2000, 0x206F),
        (0x3000, 0x30FF),
        (0x4E00, 0x4FFF),
        (0xFE00, 0xFEFF),
        (0x1F300, 0x1FAFF),
    ]
    pieces = []
    for _ in range(rng.randint(1, 40)):
        if rng.random() < 0.3:
            pieces.append(
                rng.choice([" ", "  ", "\n", "\r\n", "\t", "'s", "'LL", "123456"])
            )
        else:
            lo, hi = rng.choice(blocks)
            pieces.append(
                "".join(chr(rng.randint(lo, hi)) for _ in range(rng.randint(1, 8)))
            )
    return "".join(pieces)


def main() -> None:
    enc = tiktoken.get_encoding("cl100k_base")
    alice = (ROOT / "examples" / "alice.txt").read_text(encoding="utf-8")
    rng = random.Random(7)
    texts = [
        *CASES,
        alice[:30_000],
        (ROOT / "README.md").read_text(encoding="utf-8"),
        *(random_text(rng) for _ in range(300)),
    ]
    encode = [{"text": t, "ids": enc.encode(t, allowed_special="all")} for t in texts]

    train_text = alice[:40_000]
    tokenizer = RegexTokenizer("gpt4")
    tokenizer.train(train_text, 256 + 300)
    train = {"text": train_text, "merges": [list(pair) for pair in tokenizer.merges]}

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(
        json.dumps({"encode": encode, "train": train}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"{OUT.relative_to(ROOT)}: {len(encode)} texts, {len(tokenizer.merges)} merges"
    )


if __name__ == "__main__":
    main()
