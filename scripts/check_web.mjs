// Holds the playground's JavaScript to the Python side: cl100k_base ids must
// match tiktoken exactly, and the trainer must learn the same merges as
// RegexTokenizer("gpt4"). Run from anywhere: node scripts/check_web.mjs
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";

import { Cl100k, unpackRanks } from "../docs/js/cl100k.js";
import { chunkCounts, trainBpe } from "../docs/js/trainer.js";

const read = (path) => readFileSync(new URL(path, import.meta.url));
const fixture = JSON.parse(read("fixtures/web.json"));
const encoder = new Cl100k(unpackRanks(gunzipSync(read("../docs/data/cl100k_base.bin.gz"))));

let failures = 0;
const fail = (message) => {
  failures++;
  console.error(message);
};

const started = performance.now();
let tokens = 0;
for (const { text, ids } of fixture.encode) {
  const got = encoder.encode(text);
  tokens += got.length;
  const at = got.findIndex((id, i) => id !== ids[i]);
  if (at >= 0 || got.length !== ids.length) {
    const where = at >= 0 ? at : Math.min(got.length, ids.length);
    fail(
      `encode ${JSON.stringify(text.slice(0, 60))}: first difference at token ${where}\n` +
        `  expected ${ids.slice(where, where + 8).join(" ")}\n` +
        `  got      ${got.slice(where, where + 8).join(" ")}`,
    );
  }
}
const ms = performance.now() - started;
console.log(`encode: ${fixture.encode.length} texts, ${tokens} tokens in ${ms.toFixed(0)} ms`);

const { text, merges } = fixture.train;
const learned = trainBpe(chunkCounts(text), merges.length);
const at = learned.findIndex(([a, b], i) => a !== merges[i]?.[0] || b !== merges[i]?.[1]);
if (at >= 0 || learned.length !== merges.length) {
  fail(`train: merges differ from Python at merge ${at >= 0 ? at : learned.length}`);
}
console.log(`train: ${learned.length} merges compared`);

for (const id of [1917, 100255, 9906]) {
  const tree = encoder.mergeTree(id);
  const leaves = [];
  const walk = (node) => (node.left ? (walk(node.left), walk(node.right)) : leaves.push(node.bytes));
  walk(tree);
  if (leaves.join("") !== encoder.tokens[id]) fail(`mergeTree(${id}) does not spell the token`);
}

if (failures) {
  console.error(`${failures} check(s) failed`);
  process.exit(1);
}
console.log("ok");
