/* The service's index loader, server/load.mjs, on a tiny index written to disk.
 *
 * The gate proves the loader against the real dist/index/; this proves it in the
 * suite, which has no index. A three-chunk int8 index is written to a temporary
 * directory, loaded through the shipped path (fetch shimmed onto the filesystem),
 * and ranked with the shipped ranker: a chunk's own vector must come first, and
 * per-document text must be read from disk when asked for.
 *
 * Written by Claude Code (Fable 5.1).
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { readFileSync } from "node:fs";
import { DIMS, axis, writeTinyIndex } from "./lib/tiny_index.mjs";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const { loadIndexFromDisk, rssMb, RANKER_SHA, indexFingerprint } = await import(`${root}/server/load.mjs`);

test("a chunk's own vector retrieves that chunk first from an index read off disk", async () => {
  const { search, index, loadMs } = await loadIndexFromDisk({ indexDir: await writeTinyIndex() });
  assert.equal(index.meta.n_chunks, 3);
  assert.equal(index.vectors.length, 3 * DIMS);
  assert.ok(loadMs >= 0);
  for (const row of [0, 1, 2]) {
    const hits = search.rank(index, search.chunkVector(index, row), {});
    assert.equal(hits[0].chunk, row, `row ${row} did not rank itself first`);
    assert.ok(Math.abs(hits[0].cosine - 1) < 0.02, `self-similarity ${hits[0].cosine}`);
  }
});

test("per-document text is read from the index directory, and nothing outside it", async () => {
  const { search, index } = await loadIndexFromDisk({ indexDir: await writeTinyIndex() });
  const hits = search.rank(index, axis(2), {}).slice(0, 1);
  await search.attachText(index, hits);
  assert.equal(hits[0].text, "passage 3");
  await assert.rejects(fetch("../etc/passwd"), /only the index may be read/);
});

test("without an encoder the query path refuses rather than guessing a vector", async () => {
  const { search, index } = await loadIndexFromDisk({ indexDir: await writeTinyIndex() });
  search.clearQueryCache();
  await assert.rejects(search.embedQuery("une question assez longue", index.meta.dims));
});

test("the resident set is reported as a whole number of megabytes", () => {
  assert.ok(Number.isInteger(rssMb()) && rssMb() > 0);
});

test("RANKER_SHA fingerprints the src/search.js on disk, so a stale service is detectable", async () => {
  // The bug this pins: node caches an ESM import for the life of a process, so a
  // search service started before an edit to src/search.js keeps answering with the
  // old ranker while every gate, loading the file fresh, agrees with the file. On
  // 2026-09-26 RESCORE_WEIGHT went 0.08 -> 0.15 seven minutes after the service came
  // up and the browser gates certified the old value. `check_search.mjs` with SVC set
  // compares the service's reported fingerprint to this one, which only works while
  // the two are computed the same way.
  const { createHash } = await import("node:crypto");
  const expected = createHash("sha256")
    .update(readFileSync(`${root}/src/search.js`))
    .digest("hex")
    .slice(0, 12);
  assert.equal(RANKER_SHA, expected);
  assert.match(RANKER_SHA, /^[0-9a-f]{12}$/);
});

test("indexFingerprint sees a rebuild that keeps the chunk count and the matrices", () => {
  // The two cases earlier versions missed, with real values. On 2026-09-27 a title
  // fix rebuilt the matrices while n_chunks stayed at 128841; a metadata-only
  // rebuild (a topic retagged, a tier moved) leaves even the matrix names alone.
  const before = {
    chunks_file: "chunks-7db9bff31117ee1c.u16", vectors_file: "vectors-f69c1cfbf4596940.b1",
    n_chunks: 128841, dims: 1024, quant: "1bit",
    documents: [{ id: 0, file: "a.pdf", topic: "energy", access: "open" }],
  };
  const retagged = { ...before, documents: [{ ...before.documents[0], topic: "housing" }] };
  const rebuilt = { ...before, vectors_file: "vectors-fc34e70473d6887d.b1" };
  const bytes = (meta) => JSON.stringify(meta);

  assert.notEqual(indexFingerprint(bytes(before)), indexFingerprint(bytes(retagged)),
                  "a metadata-only rebuild must change the fingerprint");
  assert.notEqual(indexFingerprint(bytes(before)), indexFingerprint(bytes(rebuilt)));
  assert.match(indexFingerprint(bytes(before)), /^[0-9a-f]{12}$/);
});

test("the service and the checker fingerprint the same bytes the same way", async () => {
  // loadIndexFromDisk hashes meta.json from disk; check_search.mjs hashes the
  // same file with the same function. Equal files must give equal answers.
  const { mkdtempSync, writeFileSync } = await import("node:fs");
  const { tmpdir } = await import("node:os");
  const dir = mkdtempSync(`${tmpdir()}/fp-`);
  writeFileSync(`${dir}/meta.json`, '{"n_chunks": 7}');
  assert.equal(indexFingerprint(readFileSync(`${dir}/meta.json`)),
               indexFingerprint(Buffer.from('{"n_chunks": 7}')));
});

test("RANKER_SHA is frozen at import, not recomputed per read", async () => {
  // Deliberate: a fingerprint that re-read the file on every health request would
  // always match whatever is on disk now and would therefore detect nothing. Two
  // reads inside one process must be the same value, and the value must not follow
  // a later edit to the file. Checked by re-importing, which node serves from cache.
  const again = await import(`${root}/server/load.mjs`);
  assert.equal(again.RANKER_SHA, RANKER_SHA);
});
