// How a token's bytes are drawn: visible whitespace, and hex for anything that
// is not text on its own (half of an emoji, a control byte).
const strict = new TextDecoder("utf-8", { fatal: true });
const utf8 = new TextEncoder();

const WHITESPACE = { " ": "·", "\n": "↵", "\r": "␍", "\t": "→", "\u00a0": "⍽" };

const hex = (code) => code.toString(16).padStart(2, "0");

export function binaryToBytes(binary) {
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

export function hexBytes(binary) {
  return Array.from(binary, (ch) => hex(ch.charCodeAt(0))).join(" ");
}

/** The token as text, or null when its bytes are not valid UTF-8 alone. */
export function tokenText(binary) {
  try {
    return strict.decode(binaryToBytes(binary));
  } catch {
    return null;
  }
}

// Walk the bytes one UTF-8 sequence at a time. Complete characters come out
// as strings, bytes that only make sense next to another token as numbers.
function units(binary) {
  const bytes = binaryToBytes(binary);
  const out = [];
  for (let i = 0; i < bytes.length; ) {
    const lead = bytes[i];
    const length = lead < 0x80 ? 1 : lead >> 5 === 6 ? 2 : lead >> 4 === 14 ? 3 : lead >> 3 === 30 ? 4 : 0;
    let ch = null;
    if (length && i + length <= bytes.length) {
      try {
        ch = strict.decode(bytes.subarray(i, i + length));
      } catch {
        ch = null;
      }
    }
    if (ch === null) {
      out.push(lead);
      i += 1;
    } else {
      out.push(ch);
      i += length;
    }
  }
  return out;
}

/**
 * Fill el with the token's glyphs. Returns how many newlines it holds, so the
 * caller can break the line after the chip the way the text does.
 */
export function drawToken(el, binary) {
  let newlines = 0;
  let run = "";
  let stray = [];
  const flush = () => {
    if (run) el.append(run);
    if (stray.length) el.append(span("hex", stray.map(hex).join(" ")));
    run = "";
    stray = [];
  };
  for (const unit of units(binary)) {
    if (typeof unit === "number") {
      if (run) flush();
      stray.push(unit);
      continue;
    }
    const code = unit.codePointAt(0);
    if (WHITESPACE[unit]) {
      flush();
      el.append(span("ws", WHITESPACE[unit]));
      if (unit === "\n") newlines++;
    } else if (code < 0x20 || (code >= 0x7f && code <= 0xa0) || code === 0xfeff) {
      flush();
      el.append(span("hex", Array.from(utf8.encode(unit), hex).join(" ")));
    } else {
      if (stray.length) flush();
      run += unit;
    }
  }
  flush();
  return newlines;
}

function span(className, text) {
  const el = document.createElement("span");
  el.className = className;
  el.textContent = text;
  return el;
}

/** A small inline token, as used in the merge list and the explainer tiles. */
export function mini(binary, className = "") {
  const el = document.createElement("span");
  el.className = `mini ${className}`.trim();
  drawToken(el, binary);
  return el;
}
