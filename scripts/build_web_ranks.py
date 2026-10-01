"""Pack the cl100k_base ranks for the playground in docs/.

The format is as small as it can be while staying trivial to parse: tokens in
rank order, each written as one length byte followed by its bytes, gzipped.
Ranks are implicit, so this only works because cl100k's run 0..100255 without
gaps, which is checked here.
"""

import gzip
from pathlib import Path

from bytepair.gpt4 import load_cl100k_ranks

OUT = Path(__file__).resolve().parent.parent / "docs" / "data" / "cl100k_base.bin.gz"


def pack(ranks: dict[bytes, int]) -> bytes:
    tokens = sorted(ranks, key=ranks.get)
    if [ranks[t] for t in tokens] != list(range(len(tokens))):
        raise ValueError("ranks are not contiguous from 0")
    if max(map(len, tokens)) > 255:
        raise ValueError("a token is too long for a one-byte length prefix")
    return b"".join(bytes([len(t)]) + t for t in tokens)


def main() -> None:
    packed = pack(load_cl100k_ranks())
    # mtime=0 keeps the file byte-identical between runs
    OUT.write_bytes(gzip.compress(packed, compresslevel=9, mtime=0))
    print(f"{OUT.name}: {len(packed):,} bytes, {OUT.stat().st_size:,} gzipped")


if __name__ == "__main__":
    main()
