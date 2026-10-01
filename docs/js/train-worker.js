import { chunkCounts, trainBpe } from "./trainer.js";

// Merges go out in small batches as they are learned, so the page can start
// animating long before a big vocabulary is finished.
addEventListener("message", ({ data: { text, merges } }) => {
  const started = performance.now();
  const chunks = chunkCounts(text);
  let batch = [];
  trainBpe(chunks, merges, (_, [a, b], count) => {
    batch.push([a, b, count]);
    if (batch.length === 32) {
      postMessage({ type: "merges", batch });
      batch = [];
    }
  });
  postMessage({ type: "merges", batch });
  postMessage({ type: "done", ms: performance.now() - started, chunks: chunks.length });
});
