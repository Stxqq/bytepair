# bytepair

A byte-level BPE tokenizer written from scratch in Python, small enough to read in an evening and exact enough to reproduce GPT-4's `cl100k_base` token for token.

<p align="center">
  <a href="https://github.com/Stxqq/bytepair/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Stxqq/bytepair/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://stxqq.github.io/bytepair/"><img alt="Live demo" src="https://img.shields.io/badge/demo-live-111113?style=flat&labelColor=111113"></a>
  <a href="LICENSE"><img alt="License MIT" src="https://img.shields.io/badge/license-MIT-2563eb?style=flat&labelColor=111113"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%E2%80%933.13-BAE6FD?style=flat&labelColor=111113">
  <img alt="tiktoken compatible" src="https://img.shields.io/badge/cl100k__base-tiktoken--identical-BBF7D0?style=flat&labelColor=111113">
</p>

## What it is

Language models never see text. They see integers, and the tokenizer decides
which integers. `bytepair` is a complete implementation of the scheme GPT-2 and
GPT-4 use:

- `BasicTokenizer` runs BPE over the raw byte stream.
- `RegexTokenizer` first splits text with the GPT-2 or GPT-4 pattern, so merges
  never cross word, number or punctuation boundaries.
- `GPT4Tokenizer` rebuilds the merge table of OpenAI's `cl100k_base` from its
  published ranks and produces the same ids as `tiktoken`.

Training keeps pair counts up to date across merges instead of recounting the
corpus, which makes it about 500x faster than the textbook loop on a few MB of
text. The package is under 900 lines including the CLI, with one dependency
(`regex`).

## How it works

```
 "Hello world!"
       |  split (GPT-4 pattern)
       v
 ["Hello", " world", "!"]                 each chunk is cached after its first encode
       |  utf-8
       v
 [72 101 108 108 111] [32 119 ...] [33]   256 byte tokens
       |  merge loop: repeatedly apply the lowest-ranked pair present
       v
 [9906] [1917] [0]                        ids, identical to tiktoken
```

**Training** counts every adjacent pair, merges the most frequent one into a new
token and repeats. The naive version recounts the whole corpus after each merge.
`bytepair` deduplicates chunks first (4.6 MB of novels is 1,019,184 chunks but
only 42,339 distinct ones), weights them by frequency and keeps an index from
pair to the chunks containing it. A merge then only touches those chunks and
patches the counts locally: around `x a b y`, the pairs `(x,a) (a,b) (b,y)` go
away and `(x,ab) (ab,y)` appear. A lazy max-heap picks the next pair. Ties go to
the smallest pair, so training is deterministic and the fast trainer produces
exactly the merges of the naive one (the tests check this on random corpora).

**GPT-4 compatibility.** `tiktoken` ships `cl100k_base` as `token bytes -> rank`
without saying which two tokens each one was merged from. Running BPE on a
token's own bytes, allowing only merges ranked below it, stops at exactly the
two halves that formed it, which recovers all 100,000 merges in about half a
second. cl100k also numbers its 256 byte tokens in its own order, so each
tokenizer carries a `byte_ids` table (the identity for anything trained here).

## Quickstart

```bash
git clone https://github.com/Stxqq/bytepair && cd bytepair
python -m venv .venv && source .venv/bin/activate
pip install -e ".[tiktoken]"       # tiktoken is optional, see below
```

Without `tiktoken`, the cl100k ranks are downloaded once from
`openaipublic.blob.core.windows.net`, checked against their SHA-256 and cached in
`~/.cache/bytepair` (override with `BYTEPAIR_CACHE`).

## Usage

```python
from bytepair import GPT4Tokenizer, RegexTokenizer, load

tok = RegexTokenizer("gpt4")
tok.train(open("examples/alice.txt").read(), vocab_size=1024)
tok.save("models/alice")  # alice.model + alice.vocab

ids = tok.encode("Would you tell me, please, which way I ought to go from here?")
tok.decode(ids)

gpt4 = GPT4Tokenizer()
gpt4.encode("hello world")  # [15339, 1917]
gpt4.encode("<|endoftext|>", allowed_special="all")  # [100257]
gpt4.encode("<|endoftext|>")  # raises, like tiktoken
gpt4.encode("<|endoftext|>", allowed_special="none")  # encoded as plain text

alice = load("models/alice.model")
```

`allowed_special` follows tiktoken: `"all"`, a set of names, `"none"` (treat
them as text) or `"none_raise"` (the default, refuse text containing them).

The command line tool:

```console
$ bytepair train examples/alice.txt -o models/alice -v 1024
768 merges in 0.04s, 151,095 bytes -> 50,088 tokens (3.02 bytes/token)
wrote models/alice.model and models/alice.vocab

$ bytepair encode "hello world"
15339 1917

$ bytepair decode 15339 1917
hello world

$ bytepair inspect --color never "The tokenizer is a separate stage of the LLM pipeline 🙂"
The| tokenizer| is| a| separate| stage| of| the| L|LM| pipeline| 🙂
791 47058 374 264 8821 6566 315 279 445 11237 15660 28584
55 chars, 58 bytes, 12 tokens, 4.83 bytes/token
```

In a terminal, `inspect` draws each token on a pastel background with its id
underneath in the matching color. `-m path/to/model` switches from cl100k_base
to a model you trained. The `.vocab` file next to each model shows how every
token was built:

```
   256  [ ] [t] -> [ t]
   257  [h] [e] -> [he]
   258  [\xe2] [\x80] -> [\xe2\x80]
   259  [ ] [a] -> [ a]
```

Token 258 is the first two bytes of `’`, the curly apostrophe Alice uses
everywhere. It is not valid UTF-8 on its own, which is why decoding arbitrary
id sequences uses `errors="replace"`.

`python examples/train_alice.py` trains on the book and compares the result with
GPT-4's vocabulary:

```
'Would you tell me, please, which way I ought to go from here?'
  alice-1024    19 tokens  W|ould| you| tell| me|,| pleas|e|,| which| way| I| |ought| to| go| from| here|?
  cl100k_base   16 tokens  Would| you| tell| me|,| please|,| which| way| I| ought| to| go| from| here|?
  alice-1024   whole book: 3.02 bytes/token
  cl100k_base  whole book: 4.09 bytes/token
```

## Results

Measured with `python benchmarks/run.py` on an Apple M4 Pro, Python 3.14, using
six Project Gutenberg novels (4.60 MB, GPT-4 split). Raw numbers are in
[`benchmarks/results.json`](benchmarks/results.json).

| trainer | vocab size | time |
|---|---:|---:|
| naive (recount every merge) | 512 | 175.60 s |
| incremental | 512 | 0.34 s |
| incremental | 1,280 | 0.65 s |
| incremental | 4,352 | 0.99 s |
| incremental | 16,640 | 1.13 s |
| incremental | 33,024 | 1.43 s |

At the same vocabulary the incremental trainer is 513x faster and learns
identical merges. Training a 33k vocabulary end to end, regex split included,
takes 1.47 s.

Encoding the same corpus with `GPT4Tokenizer` gives 1,114,325 tokens
(4.13 bytes/token), identical to `tiktoken`. Pure Python runs at 5.1 MB/s with a
cold chunk cache and 10.2 MB/s warm, against 16.6 MB/s for tiktoken's Rust core
in the same run.

## Project layout

```
bytepair/
  pairs.py        count_pairs and merge_pair, the two BPE primitives
  train.py        incremental trainer and the naive reference
  base.py         Tokenizer: vocab, encode/decode, special tokens, save
  tokenizers.py   BasicTokenizer, RegexTokenizer, load
  patterns.py     GPT-2 and GPT-4 split patterns
  gpt4.py         cl100k_base: rank loading, merge recovery, byte order
  modelfile.py    the .model text format
  cli.py          bytepair train | encode | decode | inspect
benchmarks/       run.py and results.json
examples/         alice.txt (public domain) and train_alice.py
tests/            pytest suite, including a tiktoken comparison
```

## Tests

```bash
pip install -e ".[dev,tiktoken]"
pytest -q
```

The suite covers the Wikipedia `aaabdaaabac` example, round trips over emoji,
combining marks, right-to-left scripts, control bytes and code, training
determinism, incremental versus naive merges on random corpora, save/load,
special-token rules and a token-for-token comparison with tiktoken on 400
random multilingual strings plus the package's own source. Tests that need
tiktoken or the network skip cleanly without them.

## References

- Andrej Karpathy, [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE).
  The structure of this project (basic, regex and GPT-4 tokenizers, recovering
  merges from tiktoken ranks) follows the ideas in that lecture and in
  [minbpe](https://github.com/karpathy/minbpe); the code here is a separate
  implementation.
- Sennrich, Haddow and Birch, [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) (2016).
- Radford et al., [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (2019), for byte-level BPE.
- OpenAI, [tiktoken](https://github.com/openai/tiktoken), for the cl100k_base ranks and split pattern.
- [Byte pair encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding) on Wikipedia, for the worked example used in the tests.
- *Alice's Adventures in Wonderland* by Lewis Carroll and the benchmark novels come from [Project Gutenberg](https://www.gutenberg.org/).

## License

MIT © 2026 Stefan Carapic
