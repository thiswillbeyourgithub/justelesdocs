/* The search service (server/service.mjs) and its queue (server/queue.mjs).
 *
 * The service runs here over a real socket against the tiny on-disk index and a
 * STUB ENCODER that speaks the shared wire format (base64 int8, `dim` echoed
 * back), so the request path is the shipped one end to end: parse, embed over
 * HTTP, rank, fold, attach text. What the stub cannot do is rank well, which is
 * what scripts/check_search.mjs is for.
 *
 * Written by Claude Code (Fable 5.1).
 */

import { test, after } from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { DIMS, axis, writeTinyIndex } from "./lib/tiny_index.mjs";

// Taken before the index loader replaces globalThis.fetch with the filesystem shim.
const realFetch = globalThis.fetch;
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const { createQueue } = await import(`${root}/server/queue.mjs`);
const { createSearchService, createHandler, checkWidth, cleanFilters, filterFields, boundTextCache, describeError, readConfig }
  = await import(`${root}/server/service.mjs`);
const { loadIndexFromDisk } = await import(`${root}/server/load.mjs`);

/** Listen on a free port and resolve with the base URL. */
const listen = (server) => new Promise((done) => server.listen(0, "127.0.0.1", () => done(`http://127.0.0.1:${server.address().port}`)));

/** An encoder that answers axis 2 for a question mentioning "two", axis 0 otherwise. */
function stubEncoder() {
  return createServer((req, res) => {
    let body = "";
    req.on("data", (c) => { body += c; });
    req.on("end", () => {
      res.setHeader("Content-Type", "application/json");
      if (req.url === "/api/sem/health") { res.end(JSON.stringify({ ok: true })); return; }
      const { q, dim } = JSON.parse(body);
      if (q.length < 5) { res.statusCode = 400; res.end(JSON.stringify({ error: "q must be 5-512 chars" })); return; }
      const vector = axis(/two/.test(q) ? 2 : 0);
      res.end(JSON.stringify({ q: Buffer.from(vector.buffer).toString("base64"), dim }));
    });
  });
}

const encoder = stubEncoder();
const embedUrl = await listen(encoder);
const indexDir = await writeTinyIndex();
const { search, index } = await loadIndexFromDisk({ indexDir, embedUrl });
const service = createSearchService({ search, index, floor: 0.5, depth: 2, textCache: 8 });
const web = createServer(createHandler(service, { index }));
const base = await listen(web);
after(() => { web.close(); encoder.close(); });

const post = (body, raw = false) => realFetch(`${base}/api/search`, {
  method: "POST", headers: { "Content-Type": "application/json" }, body: raw ? body : JSON.stringify(body),
});

test("a question is embedded, ranked, folded and answered with its text", async () => {
  const res = await post({ q: "passage two please" });
  assert.equal(res.status, 200);
  const answer = await res.json();
  assert.equal(answer.dims, DIMS);
  assert.equal(answer.floor, 0.5);
  assert.equal(answer.results[0].chunk, 2);
  assert.equal(answer.results[0].text, "passage 3");
  assert.equal(answer.results[0].page, 2);
  // Below the floor: the two chunks orthogonal to the query.
  assert.equal(answer.results.length, 1);
  assert.deepEqual(Object.keys(answer.results[0]).sort(),
    ["alternates", "chunk", "cosine", "doc", "figure", "lexical", "more", "page", "pages", "score", "text"]);
});

test("the lexical rescore is applied when asked for", async () => {
  const answer = await (await post({ q: "passage two please", bm25: true })).json();
  assert.equal(answer.results[0].chunk, 2);
  assert.ok(answer.results[0].score >= answer.results[0].cosine);
});

test("the reference penalty is on unless the request turns it off", async () => {
  // The tiny index ships no reference row, so one is put on it here: chunk 2, the
  // one the stub encoder answers with, scored a full bibliography. The service is
  // what decides the default, and the default is on.
  index.refs = Uint8Array.from([0, 0, 255]);
  index.refScale = 1;
  try {
    const plain = await (await post({ q: "passage two please", demote_references: false })).json();
    const demoted = await (await post({ q: "passage two please" })).json();
    const asked = await (await post({ q: "passage two please", demote_references: true })).json();
    assert.equal(plain.results[0].chunk, 2);
    assert.equal(demoted.results[0].chunk, 2);
    assert.ok(demoted.results[0].cosine < plain.results[0].cosine,
      `${demoted.results[0].cosine} should be under ${plain.results[0].cosine}`);
    assert.equal(asked.results[0].cosine, demoted.results[0].cosine);
  } finally {
    index.refs = null;
    index.refScale = 0;
  }
});

test("the recency bonus is on unless the request turns it off", async () => {
  // One document in the tiny index, so the bonus is a pure offset here and the
  // test is about the FLAG reaching rank(), not about the arithmetic (which
  // tests/search.test.mjs checks on documents of different years). Exaggerated to
  // 0.5 of a cosine for the same reason: the shipped 0.05 is a tie-breaker.
  index.recencyBonus = 0.5;
  try {
    const plain = await (await post({ q: "passage two please", favour_recent: false })).json();
    const biased = await (await post({ q: "passage two please" })).json();
    const asked = await (await post({ q: "passage two please", favour_recent: true })).json();
    assert.ok(biased.results[0].cosine > plain.results[0].cosine,
      `${biased.results[0].cosine} should be over ${plain.results[0].cosine}`);
    assert.equal(asked.results[0].cosine, biased.results[0].cosine, "omitted means on");
    assert.ok(Math.abs((biased.results[0].cosine - plain.results[0].cosine) - 0.5) < 1e-6);
  } finally {
    delete index.recencyBonus;
  }
});

test("the figures filter reaches rank(), and a mode it does not know is refused", async () => {
  // The stub encoder answers with chunk 2; calling it the document's one figure
  // makes "exclude" lose it and "only" keep nothing else.
  index.docs[0].figure_count = 1;
  try {
    const chunks = async (figures) => (await (await post({ q: "passage two please", figures })).json())
      .results.map((r) => r.chunk);
    assert.deepEqual(await chunks(undefined), [2], "omitted means include");
    assert.deepEqual(await chunks("include"), [2]);
    assert.ok(!(await chunks("exclude")).includes(2));
    assert.deepEqual(await chunks("only"), [2]);
    for (const figures of ["all", "", 1, false]) {
      const res = await post({ q: "passage two please", figures });
      assert.equal(res.status, 400, JSON.stringify(figures));
    }
  } finally {
    delete index.docs[0].figure_count;
  }
});

test("filters narrow the corpus server-side, and an unknown field is refused", async () => {
  assert.equal((await (await post({ q: "passage two please", filters: { file: ["b.pdf"] } })).json()).results.length, 0);
  assert.equal((await (await post({ q: "passage two please", filters: { file: "a.pdf", year: { from: 2019 } } })).json()).results.length, 1);
  assert.equal((await post({ q: "passage two please", filters: { bogus: ["x"] } })).status, 400);
});

test("the page's refusals keep their names over HTTP", async () => {
  const short = await post({ q: "abc" });
  assert.equal(short.status, 400);
  assert.equal((await short.json()).error, "short");
  const empty = await post({ q: "-(only a negative)" });
  assert.equal((await empty.json()).error, "empty");
  assert.equal((await post("{not json", true)).status, 400);
  assert.equal((await post({ q: 42 })).status, 400);
  assert.equal((await realFetch(`${base}/api/search`, { method: "PUT" })).status, 405);
  assert.equal((await realFetch(`${base}/api/nope`)).status, 404);
});

test("health says what is loaded and how much the process weighs", async () => {
  const health = await (await realFetch(`${base}/api/search/health`)).json();
  assert.equal(health.ok, true);
  assert.equal(health.dims, DIMS);
  assert.equal(health.n_chunks, 3);
  assert.ok(health.rss_mb > 0);
});

test("an encoder that does not answer makes the search unavailable, not broken", async () => {
  const dead = createServer(() => {});
  const deadUrl = await listen(dead);
  dead.close();
  const { search: s2, index: i2 } = await loadIndexFromDisk({ indexDir, embedUrl: deadUrl });
  const lonely = createSearchService({ search: s2, index: i2 });
  s2.clearQueryCache();
  await assert.rejects(lonely.answer({ q: "passage two please" }), (e) => describeError(e)[0] === 502);
  // Back to the live stub for anything that runs after this.
  await loadIndexFromDisk({ indexDir, embedUrl });
});

test("the queue refuses past its depth and skips a job whose client left", async () => {
  const queue = createQueue({ depth: 1 });
  let release;
  const gate = new Promise((r) => { release = r; });
  const ran = [];
  const first = queue.run(async () => { ran.push("first"); await gate; return 1; });
  // The queue yields to the event loop before each job, so `first` is still in
  // the line until the loop has turned once.
  await new Promise((turn) => setImmediate(turn));
  const controller = new AbortController();
  const second = queue.run(() => { ran.push("second"); return 2; }, { signal: controller.signal });
  await assert.rejects(queue.run(() => { ran.push("third"); }), /full/);
  assert.equal(queue.waiting(), 1);
  controller.abort();
  release();
  assert.equal(await first, 1);
  await assert.rejects(second, /gone/);
  assert.deepEqual(ran, ["first"]);
  assert.equal(queue.busy(), false);
});

test("a burst that arrives within one turn of the event loop is bounded, not served one by one", async () => {
  // Each request reaches the queue through a few awaits (the body, the encoder
  // cache), never in the same synchronous batch as the others. With the job run
  // the moment it is queued, every request found the line empty and was ranked
  // before the next was read: fifty at once meant fifty ranks and no refusal.
  const { createQueue } = await import(`${root}/server/queue.mjs`);
  const queue = createQueue({ depth: 24 });
  const settled = [];
  for (let i = 0; i < 35; i++) {
    settled.push(queue.run(() => i).then(() => "ran", (e) => e.message));
    await Promise.resolve();
  }
  const outcome = await Promise.all(settled);
  assert.equal(outcome.filter((x) => x === "ran").length, 24);
  assert.equal(outcome.filter((x) => x === "full").length, 11);
});

test("a queued search whose client hung up is never ranked", async () => {
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(service.answer({ q: "passage two please" }, { signal: controller.signal }), /gone/);
});

test("the width check refuses to start on a mismatch and accepts an unset variable", () => {
  checkWidth({ dims: 8 }, undefined);
  checkWidth({ dims: 8 }, "");
  checkWidth({ dims: 8 }, "8");
  assert.throws(() => checkWidth({ dims: 8 }, "16"), /SEARCH_DIM asks for 16/);
  assert.throws(() => checkWidth({ dims: 8 }, "many"), /not a width/);
});

test("filters are cleaned to the shapes the ranker understands", () => {
  const allowed = new Set(["topic", "issuer", "year", "guideline", "file"]);
  const clean = (raw) => cleanFilters(raw, allowed);
  assert.deepEqual(clean(undefined), {});
  assert.deepEqual(clean({ topic: [], issuer: "", year: {} }), {});
  assert.deepEqual(clean({ topic: ["a", "b"], year: { from: "2010", to: 2020 } }), { topic: ["a", "b"], year: { from: 2010, to: 2020 } });
  assert.throws(() => clean([]), /bad_request/);
  assert.throws(() => clean({ topic: [1] }), /bad_request/);
  assert.throws(() => clean({ year: { from: "soon" } }), /bad_request/);
  assert.throws(() => clean({ topic: "x".repeat(300) }), /bad_request/);
  // The level travels as a plain single-valued string. Dropped from the allow-list
  // it would not be a degraded search but a rejected one: the page sends it on every
  // query it makes at anything but "all", so the whole site would answer bad_request.
  assert.deepEqual(clean({ guideline: "family" }), { guideline: "family" });
  assert.throws(() => clean({ level: "family" }), /bad_request/);
});

test("the fields a request may filter on are the index's facets, its tier and file", async () => {
  const search = await import(`${root}/src/search.js`);
  const index = { meta: { facet_fields: ["issuer", "year"], tier_field: "level" } };
  assert.deepEqual([...filterFields(search, index)].sort(), ["file", "issuer", "level", "year"]);
  // An index from before the corpus declared its facets: no facets, the tier where
  // it used to be, so an old index answers without filters rather than wrongly.
  assert.deepEqual([...filterFields(search, { meta: {} })].sort(), ["file", "guideline"]);
});

test("the text cache keeps the newest documents and drops the oldest", () => {
  const fake = { textCache: new Map([[1, "a"], [2, "b"], [3, "c"]]) };
  boundTextCache(fake, 2);
  assert.deepEqual([...fake.textCache.keys()], [2, 3]);
});

test("errors map to the statuses the page and Caddy act on", () => {
  assert.equal(describeError(new Error("full"))[0], 503);
  assert.equal(describeError(new Error("gone"))[0], 499);
  // A question that subtracts itself is the reader's mistake, and the page turns
  // this exact 400 into its "cancels the whole question" sentence. It was 499 once,
  // the same code as a client that hung up, and the handler answers 499 with
  // nothing at all: the page sat on "Searching..." forever.
  assert.deepEqual(describeError(new Error("cancelled")), [400, { error: "cancelled" }]);
  assert.equal(describeError(Object.assign(new Error("The operation was aborted"), { name: "TimeoutError" }))[0], 502);
  assert.equal(describeError(new Error("embed 500"))[0], 502);
  assert.deepEqual(describeError(Object.assign(new Error("terms"), { got: 12, max: 10 })), [400, { error: "terms", got: 12, max: 10 }]);
});

/* The serving cross-encoder ("the resort") was removed on 2026-10-01 after it
 * exhausted the VPS's memory. These pin the removal: a page or a script still
 * built against it must meet a plain 404, and nothing in an answer may suggest
 * the feature is there. */
test("the removed resort endpoint is a 404 and no payload mentions it", async () => {
  const res = await realFetch(`${base}/api/search/rerank`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ q: "passage two", refs: [{ doc: 0, chunk: 2 }] }),
  });
  assert.equal(res.status, 404);
  assert.deepEqual(await res.json(), { error: "not_found" });
  const health = await (await realFetch(`${base}/api/search/health`)).json();
  assert.ok(!("rerank" in health) && !("rerank_window" in health));
  const answer = await (await post({ q: "passage two please" })).json();
  assert.ok(!("rerank" in answer) && !("rerank_window" in answer));
});

test("a malformed setting stops the service at start instead of unbounding it", () => {
  // Before 2026-10-03 SEARCH_QUEUE_DEPTH=2O (a letter O) became NaN, and a NaN
  // depth refuses nothing: the burst bound was silently gone.
  assert.deepEqual(
    { ...readConfig({}), indexDir: undefined, embedUrl: undefined },
    { indexDir: undefined, embedUrl: undefined, port: 8650, floor: 0, depth: 24, textCache: 64 });
  const ok = readConfig({ SEARCH_QUEUE_DEPTH: "8", SEARCH_TEXT_CACHE: "0", SEARCH_FLOOR: "0.28", SEARCH_PORT: " " });
  assert.deepEqual([ok.depth, ok.textCache, ok.floor, ok.port], [8, 0, 0.28, 8650]);
  assert.throws(() => readConfig({ SEARCH_QUEUE_DEPTH: "2O" }), /SEARCH_QUEUE_DEPTH=2O/);
  assert.throws(() => readConfig({ SEARCH_QUEUE_DEPTH: "0" }), /SEARCH_QUEUE_DEPTH/);
  assert.throws(() => readConfig({ SEARCH_TEXT_CACHE: "-1" }), /SEARCH_TEXT_CACHE/);
  assert.throws(() => readConfig({ SEARCH_PORT: "70000" }), /SEARCH_PORT/);
  assert.throws(() => readConfig({ SEARCH_FLOOR: "0,28" }), /SEARCH_FLOOR/);
  assert.throws(() => readConfig({ SEARCH_FLOOR: "1" }), /SEARCH_FLOOR/);
});

test("searches waiting on the encoder count against the bound, not only those in the queue", async () => {
  // The queue only bounded ranking, so any number of requests could sit in the
  // encoder round trip at once. An encoder that never answers holds them there.
  let held = 0;
  const stall = createServer(() => { held += 1; });
  const stallUrl = await listen(stall);
  const { search: s3, index: i3 } = await loadIndexFromDisk({ indexDir, embedUrl: stallUrl });
  s3.clearQueryCache();
  const bounded = createSearchService({ search: s3, index: i3, depth: 1 });
  const pending = [0, 1].map((i) => bounded.answer({ q: `stalled question ${i}` }).catch((e) => e.message));
  await assert.rejects(bounded.answer({ q: "one too many" }), /full/);
  assert.equal(bounded.inFlight(), 2);
  stall.closeAllConnections();
  stall.close();
  await Promise.all(pending);
  assert.equal(bounded.inFlight(), 0);
  await loadIndexFromDisk({ indexDir, embedUrl });
});
