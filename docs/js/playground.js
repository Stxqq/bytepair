// The GPT-4 tab: type, see cl100k_base tokens, hover one for its anatomy.
import { TokenCallout } from "./callout.js";
import { drawToken, tokenText } from "./glyphs.js";
import { Counter } from "./motion.js";

// Drawing tens of thousands of chips makes typing stutter, and nobody reads
// past the first few screens anyway. The stats still cover all of the text.
const MAX_CHIPS = 2500;

const SAMPLES = {
  prose:
    "The tokenizer is a separate stage of the language model pipeline. It has its own training set, and it learns its vocabulary with byte pair encoding.",
  code: 'def merge(ids, pair, idx):\n    out, i = [], 0\n    while i < len(ids):\n        if ids[i:i + 2] == list(pair):\n            out.append(idx)\n            i += 2\n        else:\n            out.append(ids[i])\n            i += 1\n    return out\n',
  numbers: "127 + 677 = 804\n1275 + 6773 = 8048\n3.14159265358979 · 2026-10-01 · 0xFF · 1,000,000",
  world: "Hello, world.\nHallo, Welt.\nПривет, мир.\nこんにちは、世界。\n안녕하세요, 세계.\nمرحبا بالعالم.",
  emoji: "Ship it 🚀 👍🏽 👨‍👩‍👧 🇩🇪 and <|endoftext|>",
};

const utf8 = new TextEncoder();

export function mountPlayground(section, encoderReady) {
  const input = section.querySelector("#play-input");
  const output = section.querySelector("#play-tokens");
  const wrap = section.querySelector(".tokens-wrap");
  const status = section.querySelector("#play-status");
  const stat = (name, decimals) => new Counter(section.querySelector(`[data-stat="${name}"]`), decimals);
  const stats = { tokens: stat("tokens"), chars: stat("chars"), bytes: stat("bytes"), ratio: stat("ratio", 2) };

  let encoder = null;
  let callout = null;
  let tokens = [];
  let chips = [];
  let active = -1;
  let showIds = false;
  let pending = 0;

  input.value = SAMPLES.prose;

  const fit = () => {
    input.style.height = "auto";
    input.style.height = `${input.scrollHeight}px`;
  };

  function render() {
    if (!encoder) return;
    const text = input.value;
    tokens = encoder.tokenize(text);
    const shown = tokens.slice(0, MAX_CHIPS);
    const fragment = document.createDocumentFragment();
    chips = shown.map((token, i) => {
      const chip = document.createElement("span");
      chip.className = token.special ? "tok special" : `tok c${i % 6}`;
      chip.dataset.i = i;
      chip.dataset.bytes = token.special ? utf8.encode(token.bytes).length : token.bytes.length;
      let newlines = 0;
      if (token.special) chip.textContent = showIds ? token.id : token.bytes;
      else if (showIds) {
        chip.textContent = token.id;
        newlines = (tokenText(token.bytes) ?? "").split("\n").length - 1;
      } else newlines = drawToken(chip, token.bytes);
      fragment.append(chip);
      for (let n = 0; n < newlines; n++) fragment.append(document.createElement("br"));
      return chip;
    });
    if (!tokens.length) {
      const empty = document.createElement("p");
      empty.className = "empty";
      empty.textContent = "Nothing to tokenize yet.";
      fragment.append(empty);
    } else if (tokens.length > shown.length) {
      const more = document.createElement("span");
      more.className = "mono-note more";
      more.textContent = `and ${(tokens.length - shown.length).toLocaleString("en-US")} more tokens`;
      fragment.append(more);
    }
    output.replaceChildren(fragment);
    output.classList.toggle("ids", showIds);
    setActive(-1);

    const chars = [...text].length;
    stats.tokens.set(tokens.length);
    stats.chars.set(chars);
    stats.bytes.set(utf8.encode(text).length);
    if (tokens.length) stats.ratio.set(chars / tokens.length);
    else stats.ratio.clear("—");
  }

  // Short text re-renders on the next frame. Past a few thousand characters a
  // pass takes long enough that typing would queue them up, so wait for a pause.
  const schedule = () => {
    fit();
    clearTimeout(pending);
    pending = setTimeout(render, input.value.length > 4000 ? 120 : 0);
  };

  function setActive(i) {
    chips[active]?.classList.remove("hot");
    active = i;
    const chip = chips[i];
    output.classList.toggle("has-hot", !!chip);
    if (!chip) {
      callout?.hide();
      return;
    }
    chip.classList.add("hot");
    const token = tokens[i];
    callout.show(chip, token);
    status.textContent = `Token ${i + 1} of ${tokens.length}, id ${token.id}`;
  }

  input.addEventListener("input", schedule);
  addEventListener("resize", () => {
    fit();
    if (active >= 0) callout.place(chips[active]);
  });

  section.querySelectorAll("[data-sample]").forEach((button) => {
    button.addEventListener("click", () => {
      input.value = SAMPLES[button.dataset.sample];
      schedule();
    });
  });

  section.querySelectorAll("[data-show]").forEach((button) => {
    button.addEventListener("click", () => {
      showIds = button.dataset.show === "ids";
      section.querySelectorAll("[data-show]").forEach((b) => {
        b.classList.toggle("on", b === button);
        b.setAttribute("aria-pressed", b === button);
      });
      render();
    });
  });

  output.addEventListener("pointerover", (event) => {
    const chip = event.target.closest(".tok");
    if (chip && +chip.dataset.i !== active) setActive(+chip.dataset.i);
  });
  output.addEventListener("pointerleave", () => {
    if (document.activeElement !== output) setActive(-1);
  });
  output.addEventListener("focus", () => {
    if (active < 0 && chips.length) setActive(0);
  });
  output.addEventListener("blur", () => setActive(-1));
  output.addEventListener("keydown", (event) => {
    const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[event.key];
    if (step && chips.length) {
      event.preventDefault();
      setActive(Math.max(0, Math.min(chips.length - 1, active + step)));
      chips[active].scrollIntoView({ block: "nearest" });
    } else if (event.key === "Escape") {
      setActive(-1);
    }
  });

  fit();
  encoderReady.then((loaded) => {
    encoder = loaded;
    callout = new TokenCallout(wrap, encoder);
    render();
  }, (error) => {
    output.querySelector(".loading").textContent = `Could not load cl100k_base: ${error.message}`;
  });

  return { shown: fit };
}
