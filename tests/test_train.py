import random
from collections import Counter

import pytest

from bytepair.pairs import count_pairs, merge_pair
from bytepair.train import train_bpe, train_bpe_naive


def weighted(chunks):
    return Counter(chunks).items()


def test_merge_pair_is_left_to_right():
    assert merge_pair([1, 1, 1, 1, 1], (1, 1), 9) == [9, 9, 1]
    assert merge_pair([1, 2, 3, 1, 2], (1, 2), 7) == [7, 3, 7]
    assert merge_pair([1], (1, 2), 7) == [1]


def test_count_pairs_accumulates():
    counts = count_pairs([1, 2, 1, 2])
    assert counts == {(1, 2): 2, (2, 1): 1}
    count_pairs([1, 2], counts)
    assert counts[(1, 2)] == 3


def test_wikipedia_example():
    # https://en.wikipedia.org/wiki/Byte_pair_encoding
    # aaabdaaabac -> ZabdZabac -> ZYdZYac -> XdXac
    a, b = ord("a"), ord("b")
    merges = train_bpe(weighted([b"aaabdaaabac"]), 3)
    assert merges == [(a, a), (a, b), (256, 257)]
    assert merges == train_bpe_naive([b"aaabdaaabac"], 3)


def test_stops_when_nothing_is_left_to_merge():
    assert train_bpe(weighted([b"ab", b"ab"]), 10) == [(97, 98)]
    assert train_bpe_naive([b"ab", b"ab"], 10) == [(97, 98)]
    assert train_bpe(weighted([b"a", b""]), 10) == []


def test_runs_of_one_byte():
    for n in range(2, 12):
        chunk = b"z" * n
        assert train_bpe(weighted([chunk]), 8) == train_bpe_naive([chunk], 8)


@pytest.mark.parametrize("seed", range(25))
def test_incremental_matches_naive_on_random_corpora(seed):
    rng = random.Random(seed)
    alphabet = rng.choice([b"ab", b"abc", b"abcd ", bytes(range(97, 110))])
    chunks = [
        bytes(rng.choice(alphabet) for _ in range(rng.randint(0, 30)))
        for _ in range(rng.randint(1, 60))
    ]
    num_merges = rng.randint(1, 80)
    assert train_bpe(weighted(chunks), num_merges) == train_bpe_naive(
        chunks, num_merges
    )


def test_on_merge_reports_ids_and_counts():
    seen = []
    train_bpe(weighted([b"aaabdaaabac"]), 3, on_merge=lambda *e: seen.append(e))
    assert seen == [(256, (97, 97), 4), (257, (97, 98), 2), (258, (256, 257), 2)]
