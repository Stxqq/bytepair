"""BPE training.

``train_bpe`` is the one the tokenizers use. It deduplicates the pre-split
chunks, weights them by frequency and keeps the pair counts up to date across
merges, so each merge only touches the chunks that actually contain the pair.

``train_bpe_naive`` recounts every pair in the corpus on every merge. It is
kept as a readable reference and as the oracle for the tests.

Both break ties between equally frequent pairs by picking the smallest pair,
so they produce identical merges and training is deterministic.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from collections.abc import Callable, Iterable

from .pairs import Pair, count_pairs, merge_pair

OnMerge = Callable[[int, Pair, int], None]


def train_bpe(
    chunks: Iterable[tuple[bytes, int]],
    num_merges: int,
    on_merge: OnMerge | None = None,
) -> list[Pair]:
    """Learn up to ``num_merges`` merges from ``(chunk, frequency)`` pairs.

    Merge ``i`` creates token id ``256 + i``. Training stops early when no
    pair is left to merge.
    """
    stats = _PairStats()
    words: list[list[int]] = []
    weights: list[int] = []
    for chunk, freq in chunks:
        if len(chunk) >= 2:
            stats.add_word(len(words), chunk, freq)
            words.append(list(chunk))
            weights.append(freq)

    # Lazy max-heap: an entry is valid only while its count matches the stats.
    # On equal counts heapq yields the smaller pair, the same tie-break as
    # train_bpe_naive, which is what makes training deterministic.
    heap = [(-count, pair) for pair, count in stats.counts.items()]
    heapq.heapify(heap)

    merges: list[Pair] = []
    while len(merges) < num_merges:
        pair = _pop_most_frequent(heap, stats.counts)
        if pair is None:
            break
        new_id = 256 + len(merges)
        if on_merge is not None:
            on_merge(new_id, pair, stats.counts[pair])
        merges.append(pair)

        for index in stats.where.pop(pair):
            words[index] = stats.merge_word(
                index, words[index], pair, new_id, weights[index]
            )
        for changed, count in stats.flush():
            heapq.heappush(heap, (-count, changed))

    return merges


def _pop_most_frequent(heap, counts) -> Pair | None:
    while heap:
        neg_count, pair = heapq.heappop(heap)
        if counts.get(pair) == -neg_count:
            return pair
    return None


class _PairStats:
    def __init__(self):
        self.counts: dict[Pair, int] = defaultdict(int)
        # pair -> words that contain it. Entries go stale once a word loses the
        # pair; merging a stale word is a no-op, so they are never pruned.
        self.where: dict[Pair, set[int]] = defaultdict(set)
        self.touched: set[Pair] = set()

    def add_word(self, index, ids, weight):
        for pair in zip(ids, ids[1:]):
            self.counts[pair] += weight
            self.where[pair].add(index)

    def merge_word(self, index, ids, pair, new_id, weight) -> list[int]:
        """Merge ``pair`` inside one word and patch the counts locally.

        Around each merge site ``x a b y`` the pairs (x, a), (a, b), (b, y)
        vanish and (x, new), (new, y) appear. When two sites are adjacent, the
        (new, y) added by the first is removed again as the (x, a) of the
        second, which is exactly right.
        """
        a, b = pair
        merged: list[int] = []
        i, n = 0, len(ids)
        while i < n:
            if ids[i] == a and i + 1 < n and ids[i + 1] == b:
                self._shift(pair, -weight)
                if merged:
                    left = merged[-1]
                    self._shift((left, a), -weight)
                    self._shift((left, new_id), weight, index)
                if i + 2 < n:
                    right = ids[i + 2]
                    self._shift((b, right), -weight)
                    self._shift((new_id, right), weight, index)
                merged.append(new_id)
                i += 2
            else:
                merged.append(ids[i])
                i += 1
        return merged

    def _shift(self, pair, delta, index=None):
        self.counts[pair] += delta
        self.touched.add(pair)
        if index is not None:
            self.where[pair].add(index)

    def flush(self):
        """Yield (pair, count) for every pair touched since the last flush.

        Pairs that dropped to zero are removed from ``counts`` here, so stale
        heap entries for them never validate.
        """
        for pair in self.touched:
            count = self.counts[pair]
            if count:
                yield pair, count
            else:
                del self.counts[pair]
        self.touched.clear()


def train_bpe_naive(chunks: Iterable[bytes], num_merges: int) -> list[Pair]:
    """Reference trainer: recount the whole corpus before every merge."""
    words = [list(chunk) for chunk in chunks]
    merges: list[Pair] = []
    while len(merges) < num_merges:
        counts: dict[Pair, int] = {}
        for ids in words:
            count_pairs(ids, counts)
        if not counts:
            break
        pair = min(counts, key=lambda p: (-counts[p], p))
        new_id = 256 + len(merges)
        words = [merge_pair(ids, pair, new_id) for ids in words]
        merges.append(pair)
    return merges
