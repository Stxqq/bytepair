"""Train a 1024-token vocabulary on Alice in Wonderland and compare it with
GPT-4's 100k-token cl100k_base on a passage from the book.

    python examples/train_alice.py
"""

from pathlib import Path

from bytepair import GPT4Tokenizer, RegexTokenizer
from bytepair.base import render_token

HERE = Path(__file__).resolve().parent
PASSAGE = "Would you tell me, please, which way I ought to go from here?"

text = (HERE / "alice.txt").read_text(encoding="utf-8")

tok = RegexTokenizer("gpt4")
tok.train(text, vocab_size=1024)
tok.save(HERE.parent / "models" / "alice")

print("first merges:")
for pair, new_id in list(tok.merges.items())[:8]:
    a, b = (render_token(tok.vocab[i]) for i in pair)
    print(f"  {new_id}  [{a}] + [{b}] -> [{render_token(tok.vocab[new_id])}]")

print("\nlongest tokens:")
for token in sorted(tok.vocab.values(), key=len, reverse=True)[:6]:
    print(f"  [{render_token(token)}]")

gpt4 = GPT4Tokenizer()
print(f"\n{PASSAGE!r}")
for name, model in [("alice-1024", tok), ("cl100k_base", gpt4)]:
    ids = model.encode(PASSAGE)
    pieces = "|".join(render_token(model.decode_bytes([i])) for i in ids)
    print(f"  {name:<12} {len(ids):>3} tokens  {pieces}")

book_bytes = len(text.encode("utf-8"))
for name, model in [("alice-1024", tok), ("cl100k_base", gpt4)]:
    ratio = book_bytes / len(model.encode(text))
    print(f"  {name:<12} whole book: {ratio:.2f} bytes/token")
