// cl100k_base in the browser: the same split, the same merges, the same ids
// as tiktoken. scripts/check_web.mjs holds it to that.
import { splitGpt4 } from "./split.js";

export const SPECIAL_TOKENS = {
  "<|endoftext|>": 100257,
  "<|fim_prefix|>": 100258,
  "<|fim_middle|>": 100259,
  "<|fim_suffix|>": 100260,
  "<|endofprompt|>": 100276,
};

const SPECIAL_SPLIT = new RegExp(
  "(" + Object.keys(SPECIAL_TOKENS).map((name) => name.replace(/\|/g, "\\|")).join("|") + ")",
);

const utf8 = new TextEncoder();

// Bytes are kept as "binary strings", one char per byte. They make cheap Map
// keys and slice without copying.
const toBinary = (bytes) => {
  let out = "";
  for (let i = 0; i < bytes.length; i += 4096) {
    out += String.fromCharCode.apply(null, bytes.subarray(i, i + 4096));
  }
  return out;
};

/** Unpack docs/data/cl100k_base.bin.gz (already gunzipped) into tokens by rank. */
export function unpackRanks(packed) {
  const tokens = [];
  for (let at = 0; at < packed.length; ) {
    const length = packed[at];
    tokens.push(toBinary(packed.subarray(at + 1, at + 1 + length)));
    at += 1 + length;
  }
  return tokens;
}

/** Fetch and unpack the ranks file, then build an encoder from it. */
export async function loadCl100k(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  let bytes = new Uint8Array(await response.arrayBuffer());
  // Some servers send .gz files with Content-Encoding and the browser has
  // already inflated them; only gunzip when the magic bytes are still there.
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
    const inflated = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
    bytes = new Uint8Array(await new Response(inflated).arrayBuffer());
  }
  return new Cl100k(unpackRanks(bytes));
}

export class Cl100k {
  constructor(tokens) {
    this.tokens = tokens;
    this.ranks = new Map(tokens.map((token, rank) => [token, rank]));
    this.cache = new Map();
    this.trees = new Map();
  }

  get size() {
    return this.tokens.length;
  }

  /** Token ids for text. Special token strings become their special ids. */
  encode(text) {
    return this.tokenize(text).map((token) => token.id);
  }

  /** Tokens as { id, bytes } (bytes as a binary string), with special: true for special tokens. */
  tokenize(text) {
    const tokens = [];
    text.split(SPECIAL_SPLIT).forEach((part, i) => {
      if (i % 2) {
        tokens.push({ id: SPECIAL_TOKENS[part], bytes: part, special: true });
        return;
      }
      for (const chunk of splitGpt4(part)) {
        for (const id of this.encodeChunk(chunk)) tokens.push({ id, bytes: this.tokens[id] });
      }
    });
    return tokens;
  }

  // Same cache policy as base.py: long chunks rarely repeat, and a full clear
  // is cheaper than LRU bookkeeping.
  encodeChunk(chunk) {
    let ids = this.cache.get(chunk);
    if (ids) return ids;
    ids = this.mergeBytes(toBinary(utf8.encode(chunk)));
    if (chunk.length <= 256) {
      if (this.cache.size >= 65_536) this.cache.clear();
      this.cache.set(chunk, ids);
    }
    return ids;
  }

  // tiktoken's byte_pair_merge: keep the boundaries between parts and join
  // the adjacent pair whose concatenation has the lowest rank, leftmost first.
  // onStep(bounds, rank) lets the explainer watch the loop.
  mergeBytes(piece, maxRank = Infinity, onStep) {
    const whole = this.ranks.get(piece);
    if (whole !== undefined && whole < maxRank) return [whole];
    if (piece.length > 8 && maxRank === Infinity && !onStep) return this.mergeLong(piece);
    const bounds = Array.from({ length: piece.length + 1 }, (_, i) => i);
    while (bounds.length > 2) {
      let best = maxRank;
      let at = -1;
      for (let i = 0; i < bounds.length - 2; i++) {
        const rank = this.ranks.get(piece.slice(bounds[i], bounds[i + 2]));
        if (rank !== undefined && rank < best) {
          best = rank;
          at = i;
        }
      }
      if (at < 0) break;
      bounds.splice(at + 1, 1);
      onStep?.(bounds, best);
    }
    const ids = [];
    for (let i = 0; i < bounds.length - 1; i++) {
      ids.push(this.ranks.get(piece.slice(bounds[i], bounds[i + 1])));
    }
    return ids;
  }

  // The rescan above is quadratic, which freezes the tab on a long run of
  // letters. Here every pair waits in a heap as rank * 2^32 + position and the
  // parts form a linked list; the position keeps ties leftmost first.
  mergeLong(piece) {
    const n = piece.length;
    const next = new Int32Array(n);
    const prev = new Int32Array(n);
    const dead = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      next[i] = i + 1;
      prev[i] = i - 1;
    }
    // part i is piece[i, next[i]); its pair with the part after it ends at next[next[i]]
    const pairRank = (i) => (next[i] < n ? this.ranks.get(piece.slice(i, next[next[i]])) : undefined);
    const heap = new MinHeap();
    const offer = (i) => {
      const rank = pairRank(i);
      if (rank !== undefined) heap.push(rank * SLOT + i);
    };
    for (let i = 0; i < n - 1; i++) offer(i);
    while (heap.size) {
      const top = heap.pop();
      const i = top % SLOT;
      if (dead[i] || pairRank(i) !== Math.floor(top / SLOT)) continue;
      dead[next[i]] = 1;
      next[i] = next[next[i]];
      if (next[i] < n) prev[next[i]] = i;
      offer(i);
      if (prev[i] >= 0) offer(prev[i]);
    }
    const ids = [];
    for (let i = 0; i < n; i = next[i]) ids.push(this.ranks.get(piece.slice(i, next[i])));
    return ids;
  }

  /**
   * The binary tree of merges that produced a token, rebuilt like the Python
   * package does it: BPE on the token's own bytes, allowing only merges ranked
   * below it, stops at exactly the two halves that were joined.
   */
  mergeTree(id) {
    let tree = this.trees.get(id);
    if (tree) return tree;
    const bytes = this.tokens[id];
    if (bytes.length === 1) {
      tree = { id, bytes };
    } else {
      const [left, right] = this.mergeBytes(bytes, id);
      tree = { id, bytes, left: this.mergeTree(left), right: this.mergeTree(right) };
    }
    this.trees.set(id, tree);
    return tree;
  }
}

const SLOT = 2 ** 32;

class MinHeap {
  constructor() {
    this.items = [];
  }

  get size() {
    return this.items.length;
  }

  push(value) {
    const items = this.items;
    let i = items.push(value) - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (items[parent] <= value) break;
      items[i] = items[parent];
      i = parent;
    }
    items[i] = value;
  }

  pop() {
    const items = this.items;
    const top = items[0];
    const last = items.pop();
    if (items.length) {
      let i = 0;
      for (;;) {
        let child = 2 * i + 1;
        if (child >= items.length) break;
        if (child + 1 < items.length && items[child + 1] < items[child]) child++;
        if (items[child] >= last) break;
        items[i] = items[child];
        i = child;
      }
      items[i] = last;
    }
    return top;
  }
}
