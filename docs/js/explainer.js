// "How it works": one tile per stage of encoding, each a spec sheet drawn from
// the real encoder, so every number on it is what the code actually does.
import { drawToken, mini } from "./glyphs.js";
import { splitGpt4 } from "./split.js";
import { chunkCounts, mergePair, trainBpe } from "./trainer.js";

const SENTENCE = "I'm tokenizing naïve text in 2026.";
const fmt = new Intl.NumberFormat("en-US");
const utf8 = new TextEncoder();
const binary = (text) => String.fromCharCode(...utf8.encode(text));

export function mountExplainer(section, encoderReady) {
  const feed = section.querySelector("#how-feed");
  let built = false;

  const layout = () => feed.querySelectorAll(".anno-frame").forEach(placeNotes);

  encoderReady.then((encoder) => {
    feed.append(
      splitTile(),
      bytesTile(),
      mergesTile(encoder),
      idsTile(encoder),
      trainingTile(),
    );
    built = true;
    if (!section.hidden) layout();
  });
  addEventListener("resize", () => {
    if (built && !section.hidden) layout();
  });

  return {
    shown() {
      if (built) requestAnimationFrame(layout);
    },
  };
}

function splitTile() {
  const chunks = splitGpt4(SENTENCE);
  const line = el("div", "split-line");
  const pieces = chunks.map((chunk) => {
    const piece = el("span", "piece");
    drawToken(piece, binary(chunk));
    line.append(piece);
    return piece;
  });
  const find = (test) => pieces[chunks.findIndex(test)];
  return tile({
    title: "Split",
    subtitle: "Step 1 · regex pre-split",
    object: line,
    dims: { top: `${SENTENCE.length} chars`, bottom: `${chunks.length} chunks` },
    notes: [
      { side: "left", target: find((c) => c === "'m"), name: "Contraction", text: "'s 'm 'll 've split off" },
      { side: "right", target: find((c) => c.startsWith(" tok")), name: "Word", text: "The space joins the word after it" },
      { side: "left", target: find((c) => /^\d+$/.test(c)), name: "Digits", text: "At most three per chunk" },
      { side: "right", target: find((c) => c === "."), name: "Symbols", text: "Merges never cross a chunk" },
    ],
  });
}

function bytesTile() {
  const chunk = " naïve";
  const grid = el("div", "bytes-rows");
  const rows = [...chunk].map((ch) => {
    const bytes = utf8.encode(ch);
    const row = el("div", `byte-row${bytes.length > 1 ? " multi c4" : ""}`);
    const glyph = el("span", "glyph");
    drawToken(glyph, binary(ch));
    const cells = el("span", "cells");
    for (const b of bytes) cells.append(el("span", "byte-cell", b.toString(16).padStart(2, "0")));
    row.append(glyph, cells);
    grid.append(row);
    return row;
  });
  const total = utf8.encode(chunk).length;
  return tile({
    title: "UTF-8 bytes",
    subtitle: "Step 2 · every chunk becomes bytes",
    object: grid,
    dims: { top: `${[...chunk].length} chars`, side: `${total} B` },
    notes: [
      { side: "left", target: rows[0], name: "Space", text: "U+0020 is the byte 20" },
      { side: "right", target: rows[3], name: "Two bytes", text: "ï is c3 af in UTF-8" },
      { side: "left", target: rows[1], name: "Letters", text: "a to z take one byte each" },
      { side: "right", target: rows[5], name: "Base", text: "256 byte tokens: no input is unknown" },
    ],
  });
}

function mergesTile(encoder) {
  const chunk = binary(" tokenizing");
  const states = [{ bounds: Array.from({ length: chunk.length + 1 }, (_, i) => i), rank: null }];
  encoder.mergeBytes(chunk, Infinity, (bounds, rank) => states.push({ bounds: [...bounds], rank }));

  const steps = el("div", "merge-steps");
  const rows = states.map(({ bounds, rank }, s) => {
    const row = el("div", "step");
    const parts = el("span", "parts");
    const previous = states[s - 1]?.bounds;
    for (let i = 0; i < bounds.length - 1; i++) {
      const piece = chunk.slice(bounds[i], bounds[i + 1]);
      const id = encoder.ranks.get(piece);
      const joined = previous && isNew(previous, bounds[i], bounds[i + 1]);
      parts.append(mini(piece, `${piece.length > 1 ? `c${id % 6}` : "byte"}${joined ? " joined" : ""}`));
    }
    row.append(parts, el("span", "r", rank === null ? "bytes" : `#${fmt.format(rank)}`));
    steps.append(row);
    return row;
  });
  const merges = states.length - 1;
  return tile({
    title: "Merges",
    subtitle: "Step 3 · lowest rank first, until nothing merges",
    object: steps,
    dims: { top: `${chunk.length} bytes`, side: `${merges} merges` },
    notes: [
      { side: "left", target: rows[0], name: "Bytes", text: "Start from single bytes" },
      { side: "right", target: rows[1], name: "Lowest rank", text: "The pair learned first goes first" },
      { side: "left", target: rows[Math.floor(merges / 2)], name: "One pair", text: "Each step joins two neighbours" },
      { side: "right", target: rows[merges], name: "Done", text: "No ranked pair left, so stop" },
    ],
  });
}

// true when [start, end) was not already one part in the previous state
function isNew(previous, start, end) {
  const i = previous.indexOf(start);
  return i < 0 || previous[i + 1] !== end;
}

function idsTile(encoder) {
  const tokens = encoder.tokenize(SENTENCE);
  const line = el("div", "ids-line");
  const stacks = tokens.map((token, i) => {
    const stack = el("span", `stack c${i % 6}`);
    stack.append(mini(token.bytes, `c${i % 6}`), el("small", "", String(token.id)));
    line.append(stack);
    return stack;
  });
  const naive = tokens.findIndex((t) => t.bytes.startsWith(" na"));
  return tile({
    title: "Ids",
    subtitle: "Step 4 · what the model actually sees",
    object: line,
    dims: { top: `${utf8.encode(SENTENCE).length} bytes`, bottom: `${tokens.length} tokens` },
    notes: [
      { side: "left", target: stacks[0], name: "Token", text: `"I" is id ${tokens[0].id}` },
      { side: "right", target: stacks[2], name: "Common", text: "Frequent words stay whole" },
      { side: "left", target: stacks[naive], name: "Rare", text: "Rare words fall apart" },
      { side: "right", target: stacks[stacks.length - 1], name: "Exact", text: "Same ids as tiktoken, checked in CI" },
    ],
  });
}

function trainingTile() {
  const text = "aaabdaaabac";
  const vocab = Array.from({ length: 256 }, (_, b) => String.fromCharCode(b));
  const learned = [];
  trainBpe(chunkCounts(text), 3, (id, pair, count) => {
    vocab[id] = vocab[pair[0]] + vocab[pair[1]];
    learned.push({ id, pair, count });
  });

  let ids = Array.from(utf8.encode(text));
  const states = [{ ids }];
  for (const { id, pair: [a, b] } of learned) states.push({ ids: (ids = mergePair(ids, a, b, id)) });

  const box = el("div", "train-rows");
  const rows = states.map((state) => {
    const row = el("div", "train-row");
    for (const id of state.ids) row.append(mini(vocab[id], id < 256 ? "byte" : `c${id % 6}`));
    box.append(row);
    return row;
  });
  const end = states[states.length - 1].ids.length;
  return tile({
    title: "Training",
    subtitle: "Count pairs, merge the top one, repeat",
    object: box,
    dims: { top: `${text.length} bytes`, side: `${learned.length} merges` },
    notes: [
      { side: "left", target: rows[1], name: "Count", text: `aa is the top pair, seen ${learned[0].count} times` },
      { side: "right", target: rows[2], name: "Tie", text: "Equal counts: the smaller pair wins" },
      { side: "left", target: rows[3], name: "Repeat", text: "Only pairs around a merge change" },
      { side: "right", target: rows[3], name: "Result", text: `${text.length} tokens become ${end}` },
    ],
    caption: "In Python, keeping counts up to date makes 256 merges on 4.6 MB take 0.34 s instead of 175.6 s.",
  });
}

function tile({ title, subtitle, object, dims, notes, caption }) {
  const item = el("article", "feed-item");
  const stage = el("div", "feed-tile");
  stage.tabIndex = 0;
  stage.setAttribute("aria-label", `${title}: ${notes.map((n) => `${n.name}, ${n.text}`).join(". ")}`);

  const frame = el("div", "anno-frame");
  const body = el("div", "anno-object");
  body.append(object);

  const keylines = el("div", "kl");
  const corner = el("div", "kl-corner");
  corner.append(el("span", "", "R16"));
  keylines.append(el("div", "kl-grid"), el("div", "kl-box"), corner);
  for (const [where, label] of Object.entries(dims)) {
    const dim = el("div", `kl-dim kl-dim-${where}`);
    dim.append(el("i"), el("b", "", label), el("i"));
    keylines.append(dim);
  }

  const list = el("div", "anno-items");
  notes.forEach((note, i) => {
    const item = el("div", `anno-item ai-${note.side}`);
    const label = el("div", "anno-label");
    label.append(el("em", "", String(i + 1).padStart(2, "0")), el("b", "", note.name), el("span", "", note.text));
    const parts = [el("i", "anno-line"), el("i", "anno-dot")];
    if (note.side === "left") item.append(label, ...parts);
    else item.append(...parts.reverse(), label);
    item.target = note.target;
    list.append(item);
  });

  frame.append(body, keylines, list);
  stage.append(frame, el("div", "anno-hint", "Hover for the spec sheet"));
  const text = el("div", "caption");
  text.append(el("h2", "", title), el("p", "", subtitle));
  if (caption) text.append(el("p", "", caption));
  item.append(stage, text);
  return item;
}

// Labels hang off the object's edge at the height of what they point to,
// nudged apart when two on the same side would overlap.
function placeNotes(frame) {
  const top = frame.getBoundingClientRect().top;
  for (const side of ["left", "right"]) {
    const items = [...frame.querySelectorAll(`.ai-${side}`)]
      .map((item) => {
        const r = item.target.getBoundingClientRect();
        return { item, y: r.top - top + r.height / 2, half: item.offsetHeight / 2 };
      })
      .sort((a, b) => a.y - b.y);
    let floor = -Infinity;
    for (const entry of items) {
      entry.y = Math.max(entry.y, floor + entry.half);
      floor = entry.y + entry.half + 8;
      entry.item.style.top = `${entry.y}px`;
    }
  }
}

function el(tag, className = "", text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
