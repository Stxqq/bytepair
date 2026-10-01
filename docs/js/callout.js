// The keyline callout: a dimension line over the hovered token, a dot on its
// edge, a line growing out of the dot and a small spec sheet at its end.
import { drawToken, hexBytes, mini } from "./glyphs.js";
import { springTo } from "./motion.js";

const LINE = 20;
const TREE_WIDTH = 268;
const TREE_ROWS = 6;
const ROW = 22;

const fmt = new Intl.NumberFormat("en-US");

export class TokenCallout {
  constructor(wrap, encoder) {
    this.wrap = wrap;
    this.encoder = encoder;
    this.root = wrap.querySelector(".callout");
    this.dim = this.root.querySelector(".co-dim");
    this.dot = this.root.querySelector(".co-dot");
    this.line = this.root.querySelector(".co-line");
    this.card = this.root.querySelector(".co-card");
    this.editor = wrap.closest(".play").querySelector(".editor");
    this.visible = false;
  }

  show(chip, token) {
    this.card.replaceChildren(...this.rows(token, chip.className.match(/\bc\d\b/)?.[0]));
    this.place(chip);
    if (!this.visible) {
      this.visible = true;
      this.root.classList.add("on");
      springTo(this.line, [{ scale: "1 0" }, { scale: "1 1" }]);
    }
  }

  hide() {
    this.visible = false;
    this.root.classList.remove("on");
  }

  place(chip) {
    const box = this.wrap.getBoundingClientRect();
    const r = chip.getBoundingClientRect();
    const x = r.left - box.left;
    const y = r.top - box.top;
    const cx = x + r.width / 2;
    const cardW = this.card.offsetWidth;
    const cardH = this.card.offsetHeight;

    // Flip above only when the card clears the editor; covering what someone
    // is typing is worse than hanging over the stats for a moment.
    const roomBelow = innerHeight - r.bottom - LINE;
    const editorBottom = this.editor.getBoundingClientRect().bottom;
    const fitsAbove = r.top - LINE - cardH > Math.max(110, editorBottom + 12);
    const above = roomBelow < cardH + 16 && fitsAbove;
    const edge = above ? y : y + r.height;
    const cardX = Math.max(0, Math.min(cx - cardW / 2, box.width - cardW));
    const cardY = above ? edge - LINE - cardH : edge + LINE;

    // jumping into view from nowhere should not glide in from the last spot
    this.root.classList.toggle("snap", !this.visible);
    this.dim.style.width = `${r.width}px`;
    this.dim.style.transform = `translate(${x}px, ${above ? y + r.height + 1 : y - 11}px)`;
    this.dot.style.transform = `translate(${cx}px, ${edge}px)`;
    this.line.style.transform = `translate(${cx}px, ${above ? edge - LINE : edge}px)`;
    this.line.style.transformOrigin = above ? "bottom" : "top";
    this.card.style.transform = `translate(${cardX}px, ${cardY}px)`;
    this.dim.querySelector("b").textContent = `${chip.dataset.bytes} B`;
  }

  rows(token, color) {
    const { id } = token;
    if (token.special) {
      return [
        row("01", "Id", big(id)),
        row("02", "Text", mono(token.bytes)),
        row("03", "Kind", text("Special token. Added by hand, never learned by merging.")),
      ];
    }
    const merged = token.bytes.length > 1;
    const rows = [
      row("01", "Id", big(id)),
      row("02", "Bytes", mono(hexBytes(token.bytes))),
      row(
        "03",
        "Rank",
        text(merged ? `Merge ${fmt.format(id - 255)} of ${fmt.format(this.encoder.size - 256)}` : "One of the 256 byte tokens"),
      ),
    ];
    if (merged) {
      const tree = this.encoder.mergeTree(id);
      const joined = document.createElement("span");
      joined.append(mini(tree.left.bytes, "byte"), " + ", mini(tree.right.bytes, "byte"));
      const treeRow = row("04", "Built", joined);
      treeRow.append(drawTree(tree, color));
      rows.push(treeRow);
    }
    return rows;
  }
}

function drawTree(tree, color) {
  const leaves = tree.bytes.length;
  const cell = Math.min(30, TREE_WIDTH / leaves);
  const el = document.createElement("div");
  el.className = `tree ${color ?? ""}`;
  let depth = 0;
  const visit = (node, start, level) => {
    depth = Math.max(depth, level);
    const span = node.bytes.length;
    const box = document.createElement("div");
    box.className = "tree-node";
    box.style.left = `${start * cell}px`;
    box.style.top = `${level * ROW}px`;
    box.style.width = `${span * cell - 2}px`;
    if (level === 0) box.classList.add("root");
    if (!node.left) {
      box.classList.add("leaf");
      if (cell >= 15) box.textContent = hexBytes(node.bytes);
    } else if (level === TREE_ROWS - 1) {
      box.classList.add("cut");
    } else if (span * cell > node.bytes.length * 7 + 6) {
      drawToken(box, node.bytes);
    }
    el.append(box);
    if (node.left && level < TREE_ROWS - 1) {
      visit(node.left, start, level + 1);
      visit(node.right, start + node.left.bytes.length, level + 1);
    }
  };
  visit(tree, 0, 0);
  el.style.width = `${leaves * cell}px`;
  el.style.height = `${(depth + 1) * ROW - 4}px`;
  return el;
}

function row(index, label, value) {
  const el = document.createElement("div");
  el.className = "co-row";
  const em = document.createElement("em");
  em.textContent = index;
  const b = document.createElement("b");
  b.textContent = label;
  el.append(em, b, value);
  return el;
}

function text(value) {
  const el = document.createElement("span");
  el.textContent = value;
  return el;
}

function mono(value) {
  const el = text(value);
  el.className = "mono";
  return el;
}

function big(id) {
  const el = text(String(id));
  el.className = "big";
  return el;
}
