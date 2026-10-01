// The Train tab: BPE on pasted text in a worker, replayed merge by merge.
import { drawToken, mini } from "./glyphs.js";
import { Counter, reducedMotion, SpringLayout, springTo } from "./motion.js";
import { encodeWithMerges } from "./trainer.js";

const ROW_HEIGHT = 38;
const fmt = new Intl.NumberFormat("en-US");
const utf8 = new TextEncoder();

// The first merges are worth watching one at a time; after that the replay
// speeds up so a 2,048-merge run still finishes in about ten seconds.
const delayBefore = (i) => Math.max(320 * 0.9 ** i, i < 120 ? 14 : (14 * 120) / i);

export function mountTrainer(section) {
  const corpus = section.querySelector("#train-input");
  const size = section.querySelector("#train-size");
  const reset = section.querySelector("#train-reset");
  const slider = section.querySelector("#train-merges");
  const sliderOut = section.querySelector("#train-merges-out");
  const go = section.querySelector("#train-go");
  const skip = section.querySelector("#train-skip");
  const list = section.querySelector("#vocab-list");
  const listEmpty = section.querySelector("#vocab-empty");
  const learnedCount = section.querySelector("#vocab-count");
  const caption = section.querySelector("#train-caption");
  const idleCaption = caption.textContent;
  const sampleInput = section.querySelector("#sample-input");
  const sampleOut = section.querySelector("#sample-tokens");
  const stat = (name, decimals) => new Counter(section.querySelector(`[data-stat="${name}"]`), decimals);
  const stats = { vocab: stat("vocab"), tokens: stat("sample-tokens"), ratio: stat("sample-ratio", 2) };

  let alice = null;
  let worker = null;
  let vocab = [];
  let merges = [];
  let counts = [];
  let shown = 0;
  let finished = false;
  let fastForward = false;
  let clock = 0;
  let last = 0;
  let frame = 0;
  let sampleChips = new Map();
  const reflow = new SpringLayout();

  function resetVocab() {
    vocab = Array.from({ length: 256 }, (_, b) => String.fromCharCode(b));
    merges = [];
    counts = [];
    shown = 0;
    list.replaceChildren();
    listEmpty.hidden = false;
    learnedCount.textContent = "0";
  }

  function describeCorpus() {
    const bytes = utf8.encode(corpus.value).length;
    size.textContent = `${fmt.format(bytes)} bytes`;
    go.disabled = bytes === 0;
  }

  function train() {
    worker?.terminate();
    cancelAnimationFrame(frame);
    resetVocab();
    drawSample(false);
    finished = false;
    fastForward = reducedMotion.matches;
    clock = 0;
    last = performance.now();
    skip.hidden = false;
    caption.textContent = "Training…";

    worker = new Worker(new URL("./train-worker.js", import.meta.url), { type: "module" });
    worker.addEventListener("message", ({ data }) => {
      if (data.type === "merges") {
        for (const [a, b, count] of data.batch) {
          merges.push([a, b]);
          counts.push(count);
          vocab.push(vocab[a] + vocab[b]);
        }
      } else {
        finished = true;
        caption.textContent =
          `${fmt.format(merges.length)} merges from ${fmt.format(data.chunks)} distinct chunks ` +
          `in ${fmt.format(Math.max(1, Math.round(data.ms)))} ms`;
      }
    });
    worker.addEventListener("error", (event) => {
      caption.textContent = `Training failed: ${event.message}`;
      skip.hidden = true;
    });
    worker.postMessage({ text: corpus.value, merges: +slider.value });
    frame = requestAnimationFrame(play);
  }

  function play(now) {
    // clamped so a background tab does not dump hundreds of merges at once
    clock += Math.min(now - last, 100);
    last = now;
    const from = shown;
    while (shown < merges.length && (fastForward || clock >= delayBefore(shown))) {
      if (!fastForward) clock -= delayBefore(shown);
      shown++;
    }
    if (shown > from) reveal(from, shown);
    if (finished && shown === merges.length) {
      skip.hidden = true;
      frame = 0;
      return;
    }
    if (shown === merges.length) clock = 0;
    frame = requestAnimationFrame(play);
  }

  function reveal(from, to) {
    listEmpty.hidden = true;
    learnedCount.textContent = fmt.format(to);
    const added = [];
    for (let i = from; i < to; i++) added.unshift(mergeRow(i));
    const instant = to - from > 12 || reducedMotion.matches;
    const settled = instant ? [] : [...list.children].slice(0, 12);
    list.prepend(...added);
    if (!instant) {
      const shift = added.length * ROW_HEIGHT;
      for (const row of settled) {
        row.getAnimations().forEach((a) => a.cancel());
        springTo(row, [{ transform: `translateY(${-shift}px)` }, { transform: "none" }]);
      }
      added.forEach((row, i) => {
        springTo(
          row,
          [
            { opacity: 0, transform: "translateY(-12px) scale(.94)" },
            { opacity: 1, transform: "none" },
          ],
          { delay: i * 40, fill: "backwards" },
        );
      });
    }
    drawSample(!instant);
    stats.vocab.set(256 + to);
  }

  function mergeRow(i) {
    const [a, b] = merges[i];
    const id = 256 + i;
    const row = document.createElement("li");
    row.className = "merge";
    const rank = document.createElement("span");
    rank.className = "rank";
    rank.textContent = id;
    const parts = document.createElement("span");
    parts.className = "parts";
    const plus = document.createElement("span");
    plus.className = "plus";
    plus.textContent = "+";
    const arrow = document.createElement("span");
    arrow.className = "arrow";
    arrow.textContent = "→";
    parts.append(part(a), plus, part(b), arrow, mini(vocab[id], `out c${id % 6}`));
    const count = document.createElement("span");
    count.className = "count";
    count.textContent = `×${fmt.format(counts[i])}`;
    row.append(rank, parts, count);
    return row;
  }

  const part = (id) => mini(vocab[id], id < 256 ? "byte" : `c${id % 6}`);

  // Chips are keyed by the byte offset they start at. A merge keeps the left
  // chip, grows it over its neighbour and lets everything after it slide in.
  function drawSample(animate) {
    const text = sampleInput.value;
    const pieces = [];
    let at = 0;
    for (const ids of encodeWithMerges(text, merges.slice(0, shown))) {
      for (const id of ids) {
        pieces.push({ at, id });
        at += vocab[id].length;
      }
    }

    const grown = [];
    const next = new Map();
    const chips = pieces.map(({ at, id }) => {
      let chip = sampleChips.get(at);
      if (!chip) {
        chip = document.createElement("span");
      } else if (+chip.dataset.id === id) {
        next.set(at, chip);
        return chip;
      } else {
        chip.replaceChildren();
        grown.push(chip);
      }
      chip.className = `tok c${id % 6}`;
      chip.dataset.id = id;
      chip.title = `${id}`;
      drawToken(chip, vocab[id]);
      next.set(at, chip);
      return chip;
    });

    const mutate = () => sampleOut.replaceChildren(...chips);
    if (animate) reflow.update([...sampleChips.values()], mutate);
    else mutate();
    sampleChips = next;
    if (animate) {
      for (const chip of grown) springTo(chip, [{ scale: ".82" }, { scale: "1" }]);
    }

    const chars = [...text].length;
    stats.tokens.set(pieces.length);
    stats.ratio.set(pieces.length ? chars / pieces.length : 0);
  }

  async function loadAlice() {
    if (alice === null) {
      const book = await fetch(new URL("../data/alice.txt", import.meta.url)).then((r) => r.text());
      // skip the title page and contents, start at the first chapter
      alice = book.slice(Math.max(0, book.search(/^CHAPTER I\.$/m)));
    }
    corpus.value = alice;
    const opening = alice.indexOf("Alice was beginning");
    if (opening >= 0) sampleInput.value = alice.slice(opening, alice.indexOf("\n\n", opening)).replace(/\s+/g, " ");
    describeCorpus();
    resetVocab();
    drawSample(false);
  }

  slider.addEventListener("input", () => {
    sliderOut.textContent = slider.value;
  });
  corpus.addEventListener("input", describeCorpus);
  sampleInput.addEventListener("input", () => {
    sampleChips = new Map();
    drawSample(false);
  });
  go.addEventListener("click", train);
  skip.addEventListener("click", () => {
    fastForward = true;
  });
  reset.addEventListener("click", () => {
    worker?.terminate();
    cancelAnimationFrame(frame);
    skip.hidden = true;
    caption.textContent = idleCaption;
    loadAlice();
  });

  resetVocab();
  return {
    shown() {
      if (alice === null) loadAlice();
    },
  };
}
