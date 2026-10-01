// The incremental BPE trainer from bytepair/train.py, ported for the Train
// tab. Same algorithm, same tie-break (the smallest pair wins), so for the same
// text it learns the same merges as RegexTokenizer("gpt4").train.
import { splitGpt4 } from "./split.js";

// Pairs are packed into one number. Ids stay far below 2^16 here and a*2^16+b
// orders pairs the way Python orders (a, b) tuples.
const pack = (a, b) => a * 65536 + b;
const unpack = (key) => [Math.floor(key / 65536), key % 65536];

const utf8 = new TextEncoder();

/** Count the distinct pre-split chunks of text as [bytes, frequency] pairs. */
export function chunkCounts(text) {
  const counts = new Map();
  for (const chunk of splitGpt4(text)) counts.set(chunk, (counts.get(chunk) ?? 0) + 1);
  return [...counts].map(([chunk, freq]) => [utf8.encode(chunk), freq]);
}

/**
 * Learn up to numMerges merges. onMerge(newId, [a, b], count) is called for
 * each one as it is chosen. Returns the merges as [a, b] pairs.
 */
export function trainBpe(chunks, numMerges, onMerge) {
  const counts = new Map();
  const where = new Map();
  const words = [];
  const weights = [];
  const touched = new Set();

  const shift = (key, delta, word) => {
    const count = (counts.get(key) ?? 0) + delta;
    if (count) counts.set(key, count);
    else counts.delete(key);
    touched.add(key);
    if (word !== undefined) {
      let set = where.get(key);
      if (!set) where.set(key, (set = new Set()));
      set.add(word);
    }
  };

  for (const [bytes, freq] of chunks) {
    if (bytes.length < 2) continue;
    const word = words.length;
    words.push(Array.from(bytes));
    weights.push(freq);
    for (let i = 0; i < bytes.length - 1; i++) shift(pack(bytes[i], bytes[i + 1]), freq, word);
  }
  touched.clear();

  const heap = new PairHeap();
  for (const [key, count] of counts) heap.push(count, key);

  const merges = [];
  while (merges.length < numMerges) {
    const key = heap.popValid(counts);
    if (key === undefined) break;
    const newId = 256 + merges.length;
    const [a, b] = unpack(key);
    onMerge?.(newId, [a, b], counts.get(key));
    merges.push([a, b]);

    for (const word of where.get(key)) {
      const ids = words[word];
      const weight = weights[word];
      const merged = [];
      for (let i = 0; i < ids.length; ) {
        if (ids[i] === a && ids[i + 1] === b) {
          // around x a b y: (x,a) (a,b) (b,y) go away, (x,ab) (ab,y) appear
          shift(key, -weight);
          if (merged.length) {
            const left = merged[merged.length - 1];
            shift(pack(left, a), -weight);
            shift(pack(left, newId), weight, word);
          }
          if (i + 2 < ids.length) {
            shift(pack(b, ids[i + 2]), -weight);
            shift(pack(newId, ids[i + 2]), weight, word);
          }
          merged.push(newId);
          i += 2;
        } else {
          merged.push(ids[i]);
          i += 1;
        }
      }
      words[word] = merged;
    }
    where.delete(key);
    for (const changed of touched) {
      const count = counts.get(changed);
      if (count) heap.push(count, changed);
    }
    touched.clear();
  }
  return merges;
}

/** Encode text with merges learned by trainBpe, one array of ids per chunk. */
export function encodeWithMerges(text, merges) {
  const ranks = new Map(merges.map(([a, b], i) => [pack(a, b), 256 + i]));
  return splitGpt4(text).map((chunk) => {
    let ids = Array.from(utf8.encode(chunk));
    while (ids.length > 1) {
      let best = Infinity;
      for (let i = 0; i < ids.length - 1; i++) {
        const rank = ranks.get(pack(ids[i], ids[i + 1]));
        if (rank !== undefined && rank < best) best = rank;
      }
      if (best === Infinity) break;
      const [a, b] = merges[best - 256];
      ids = mergePair(ids, a, b, best);
    }
    return ids;
  });
}

/** Replace every non-overlapping a, b in ids with id, left to right. */
export function mergePair(ids, a, b, id) {
  const merged = [];
  for (let i = 0; i < ids.length; ) {
    if (ids[i] === a && ids[i + 1] === b) {
      merged.push(id);
      i += 2;
    } else {
      merged.push(ids[i++]);
    }
  }
  return merged;
}

// Max-heap on (count, then smallest pair). Entries go stale when a count
// changes; popValid skips those, like the lazy heap in train.py.
class PairHeap {
  constructor() {
    this.counts = [];
    this.keys = [];
  }

  above(i, j) {
    const ci = this.counts[i];
    const cj = this.counts[j];
    return ci > cj || (ci === cj && this.keys[i] < this.keys[j]);
  }

  swap(i, j) {
    [this.counts[i], this.counts[j]] = [this.counts[j], this.counts[i]];
    [this.keys[i], this.keys[j]] = [this.keys[j], this.keys[i]];
  }

  push(count, key) {
    this.counts.push(count);
    this.keys.push(key);
    for (let i = this.keys.length - 1; i > 0; ) {
      const parent = (i - 1) >> 1;
      if (!this.above(i, parent)) break;
      this.swap(i, parent);
      i = parent;
    }
  }

  pop() {
    const top = [this.counts[0], this.keys[0]];
    const last = this.keys.length - 1;
    this.swap(0, last);
    this.counts.pop();
    this.keys.pop();
    for (let i = 0; ; ) {
      const l = 2 * i + 1;
      const r = l + 1;
      let best = i;
      if (l < last && this.above(l, best)) best = l;
      if (r < last && this.above(r, best)) best = r;
      if (best === i) break;
      this.swap(i, best);
      i = best;
    }
    return top;
  }

  popValid(counts) {
    while (this.keys.length) {
      const [count, key] = this.pop();
      if (counts.get(key) === count) return key;
    }
    return undefined;
  }
}
