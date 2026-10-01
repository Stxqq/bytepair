// GPT-4's pre-split pattern (cl100k_base), rewritten for JavaScript.
//
// JavaScript has no possessive quantifiers, but in this pattern every
// possessive part ends its alternative, so the greedy form matches the same
// text. Two things do differ and are spelled out by hand:
//  - tiktoken's \s is Unicode White_Space. JavaScript's \s also matches U+FEFF
//    and misses U+0085, so whitespace is an explicit class here.
//  - (?i:...) folds "s" to the long s "ſ" as well, so it is in the class.
const WS = "\\t-\\r \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";

export const GPT4_SPLIT = new RegExp(
  [
    "'(?:[sdmtſSDMT]|[lL][lL]|[vV][eE]|[rR][eE])",
    "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+",
    "\\p{N}{1,3}",
    ` ?[^${WS}\\p{L}\\p{N}]+[\\r\\n]*`,
    `[${WS}]+$`,
    `[${WS}]*[\\r\\n]`,
    `[${WS}]+(?![^${WS}])`,
    `[${WS}]`,
  ].join("|"),
  "gu",
);

/** Split text into the chunks that merges never cross. */
export function splitGpt4(text) {
  return text.match(GPT4_SPLIT) ?? [];
}
