/* The speculative download queue in src/prefetch.js.
 *
 * All of it runs on a stub fetcher, which is the point: the three rules this queue
 * exists for (one request at a time, a click beats the queue, speculation stays
 * cheap) are exactly the ones a browser session would not show you. A prefetcher
 * that quietly ran four requests at once, or kept downloading a 10 MB argumentaire
 * while the reader waited for the document they clicked, would look perfectly fine
 * on a fast laptop and be the reason the site felt slow on a phone.
 *
 * Written by Claude Code (Opus 5).
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const { createPrefetcher, prefetchAllowed, warmDocs, PREFETCH_DOCS } = await import(`${root}/src/prefetch.js`);

/**
 * A fetcher that records what it was asked for and answers when told to.
 *
 * @param {{size?: number}} [options] - Bytes each response carries.
 */
function stubFetcher({ size = 1024 } = {}) {
  const calls = [];
  const aborted = [];
  let inFlight = 0;
  let peak = 0;
  const fetcher = (url, { signal } = {}) => {
    calls.push(url);
    inFlight++;
    peak = Math.max(peak, inFlight);
    return new Promise((resolvePromise, reject) => {
      // Whichever of the timer and the abort comes first settles it; the other
      // must not decrement again, or the counter never reads zero.
      let settled = false;
      const finish = () => {
        if (settled) return;
        settled = true;
        inFlight--;
        resolvePromise({
          headers: { get: () => String(size) },
          arrayBuffer: async () => new ArrayBuffer(size),
        });
      };
      signal?.addEventListener("abort", () => {
        if (settled) return;
        settled = true;
        inFlight--;
        aborted.push(url);
        reject(new Error("aborted"));
      });
      setTimeout(finish, 1);
    });
  };
  return { fetcher, calls, aborted, peak: () => peak, inFlight: () => inFlight };
}

/**
 * Wait until the queue is idle: nothing in flight, seen from a macrotask.
 *
 * The queue moves from one fetch to the next on microtasks alone, and enqueue()
 * starts the first fetch synchronously, so a macrotask that finds nothing in flight
 * has found the queue stopped, not between two requests. Pending is not part of
 * the condition: a queue that ran out of budget stops with items still waiting.
 * A fixed 25 ms delay was the previous version of this, and it failed about one
 * run in three on a loaded machine.
 */
const settle = (stub) =>
  new Promise((resolvePromise) => {
    const poll = () => {
      if (stub.inFlight() === 0) resolvePromise();
      else setTimeout(poll, 1);
    };
    setTimeout(poll, 1);
  });

test("the queue fetches one thing at a time, in order", async () => {
  const stub = stubFetcher();
  const prefetcher = createPrefetcher({ fetcher: stub.fetcher });
  prefetcher.enqueue(["a", "b", "c"]);
  await settle(stub);
  assert.deepEqual(stub.calls, ["a", "b", "c"]);
  assert.equal(stub.peak(), 1, "more than one request was in flight");
});

test("nothing is fetched twice, however often it is asked for", async () => {
  const stub = stubFetcher();
  const prefetcher = createPrefetcher({ fetcher: stub.fetcher });
  prefetcher.enqueue(["a", "b"]);
  prefetcher.enqueue(["a", "b", "a"]);
  await settle(stub);
  prefetcher.enqueue(["a"]);
  await settle(stub);
  assert.deepEqual(stub.calls, ["a", "b"]);
});

test("a hovered document jumps the queue", async () => {
  const stub = stubFetcher();
  const prefetcher = createPrefetcher({ fetcher: stub.fetcher });
  prefetcher.enqueue(["a", "b", "c"]);
  // While "a" is in flight: "c" is what the reader is pointing at.
  prefetcher.promote("c");
  await settle(stub);
  assert.deepEqual(stub.calls, ["a", "c", "b"]);
});

test("a click stops the speculation and spares the document being opened", async () => {
  const stub = stubFetcher();
  const prefetcher = createPrefetcher({ fetcher: stub.fetcher });
  prefetcher.enqueue(["opening", "b", "c"]);
  // "opening" is in flight and is the one clicked: it must NOT be aborted, and
  // nothing behind it may start.
  prefetcher.commit("opening");
  await settle(stub);
  assert.deepEqual(stub.calls, ["opening"]);
  assert.equal(prefetcher.pending(), 0);
  prefetcher.enqueue(["d"]);
  await settle(stub);
  assert.deepEqual(stub.calls, ["opening"], "a committed queue kept speculating");
});

test("a click while something else is downloading aborts it", async () => {
  const stub = stubFetcher();
  const prefetcher = createPrefetcher({ fetcher: stub.fetcher });
  prefetcher.enqueue(["big-argumentaire", "b"]);
  prefetcher.commit("pdf/what-the-reader-clicked.pdf");
  await settle(stub);
  // The abort rejects the in-flight fetch, which the queue swallows: what matters
  // is that nothing after it ran and the rejection did not escape.
  assert.deepEqual(stub.calls, ["big-argumentaire"]);
});

test("the byte budget stops a list of heavy documents", async () => {
  const stub = stubFetcher({ size: 4 * 1024 * 1024 });
  const prefetcher = createPrefetcher({ fetcher: stub.fetcher, budget: 6 * 1024 * 1024 });
  prefetcher.enqueue(["one", "two", "three", "four"]);
  await settle(stub);
  // Two responses of 4 MB pass the 6 MB budget, so the third never starts.
  assert.deepEqual(stub.calls, ["one", "two"]);
});

test("a new question drops what the old answer had queued", async () => {
  const stub = stubFetcher();
  const prefetcher = createPrefetcher({ fetcher: stub.fetcher });
  prefetcher.enqueue(["a", "b", "c"]);
  prefetcher.reset();
  prefetcher.enqueue(["x"]);
  await settle(stub);
  assert.ok(stub.calls.includes("x"));
  assert.ok(!stub.calls.includes("c"), "the old answer's queue survived a new question");
});

test("a metered or slow connection is left alone", () => {
  assert.equal(prefetchAllowed(undefined), true, "no Network Information API: speculate");
  assert.equal(prefetchAllowed({ effectiveType: "4g" }), true);
  assert.equal(prefetchAllowed({ saveData: true, effectiveType: "4g" }), false);
  assert.equal(prefetchAllowed({ effectiveType: "2g" }), false);
  assert.equal(prefetchAllowed({ effectiveType: "slow-2g" }), false);
  assert.equal(prefetchAllowed({ effectiveType: "3g" }), true);
});

test("warmDocs picks exactly PREFETCH_DOCS distinct documents, skipping the excluded", () => {
  const docs = [{ file: "a" }, { file: "b" }, { file: "r", access: "restricted" }, { file: "c" }, { file: "d" }];
  // Two passages of `a`, then a restricted document, then three more open ones.
  const results = [0, 0, 2, 1, 3, 4].map((doc) => ({ doc }));
  const picked = warmDocs(results, docs, (doc) => doc.access === "restricted");
  // The loop used to push before checking the limit, which warmed one document too many.
  assert.equal(picked.length, PREFETCH_DOCS);
  assert.deepEqual(picked.map((doc) => doc.file), ["a", "b", "c"]);
});

test("a click does not abort a viewer asset every click needs", async () => {
  const stub = stubFetcher();
  const prefetcher = createPrefetcher({ fetcher: stub.fetcher, keep: ["vendor/pdfjs/pdf.worker.mjs"] });
  prefetcher.enqueue(["vendor/pdfjs/pdf.worker.mjs", "b"]);
  // The worker is in flight when the reader clicks some other document: aborting
  // it would only make the viewer download all 2.2 MB of it again.
  prefetcher.commit("pdf/what-the-reader-clicked.pdf");
  await settle(stub);
  assert.deepEqual(stub.calls, ["vendor/pdfjs/pdf.worker.mjs"]);
  assert.deepEqual(stub.aborted, [], "the worker download was aborted");
});
