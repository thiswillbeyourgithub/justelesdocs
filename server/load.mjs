/* Load the shipped index off the filesystem and run src/search.js on it under node.
 *
 * This is the one place the service and the scripts get an index from. It installs
 * the fetch shim (server/lib/browser_shim.mjs), so `loadIndex`, `embedQuery` and
 * `attachText` in src/search.js read `index/` from a directory and reach the
 * encoder over HTTP, and it returns the module itself beside the index so a caller
 * ranks with the SAME `rank` the gate measures. There is no second ranker here and
 * there must not be one: DESIGN.md's numbers are about src/search.js.
 *
 * Run directly it is the proof that the path works outside the gate, and the
 * source of the two numbers the handoff asked for: how long the load takes and
 * what the process weighs once the index is resident.
 *
 *   node server/load.mjs [dist/index]
 *   EMBED_URL=http://127.0.0.1:8461 node server/load.mjs      # plus one real query
 *   (QUESTION=... picks that query; the default is a neutral one)
 *
 * Written by Claude Code (Fable 5.1).
 */

import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { installFetchShim, installDocumentStub } from "./lib/browser_shim.mjs";
import { corpusPath } from "./lib/corpus.mjs";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");

/**
 * Fingerprint of the ranker this process is running, taken ONCE when this module
 * is first imported.
 *
 * It exists because node caches an ESM import for the life of the process and
 * nothing here reloads: a service started before an edit to src/search.js keeps
 * answering with the old ranker, holds its port so the replacement dies on
 * `Address already in use` in a log nobody reads, and passes every gate, because
 * the gates load src/search.js themselves in a fresh process and therefore agree
 * with the file rather than with the service. That has cost four sessions. The
 * most recent: RESCORE_WEIGHT went from 0.08 to 0.15 seven minutes after the
 * service came up, three of five sample queries rank differently between those
 * two values, and the browser gates then certified behaviour nobody had built.
 *
 * Frozen at import time ON PURPOSE. Hashing the file per request would read
 * whatever is on disk now, always match, and detect nothing.
 */
export const RANKER_SHA = createHash("sha256")
  .update(readFileSync(`${root}/src/search.js`))
  .digest("hex")
  .slice(0, 12);

/**
 * Load an index directory and the ranking module that reads it.
 *
 * Installs `globalThis.fetch` for the life of the process: a second call with a
 * different directory replaces the first, so a process serves one index.
 *
 * @param {object} options
 * @param {string} options.indexDir - Absolute path of the index directory.
 * @param {string} [options.embedUrl] - Base URL of the encoder; empty means none,
 *   and `embedQuery` then throws instead of asking.
 * @param {number} [options.apiTimeoutMs] - Give up on the encoder after this long.
 * @returns {Promise<{search: object, index: object, indexSha: string, loadMs: number,
 *   reachable: (base: string) => Promise<boolean>}>}
 */
export async function loadIndexFromDisk({ indexDir, embedUrl = "", apiTimeoutMs = 0 }) {
  const { reachable } = installFetchShim({ indexDir, embedUrl, apiTimeoutMs });
  installDocumentStub();
  const search = await import(`${root}/src/search.js`);
  // Read BEFORE the load: should meta.json be rewritten in between, the service
  // reports the older fingerprint while holding the newer index, and the staleness
  // check then raises a false alarm rather than missing a real one.
  const indexSha = indexFingerprint(readFileSync(`${indexDir}/meta.json`));
  const started = performance.now();
  const index = await search.loadIndex("index", { vectors: true });
  return { search, index, indexSha, loadMs: performance.now() - started, reachable };
}

/**
 * Fingerprint of an index: a hash of its raw `meta.json` bytes.
 *
 * Two earlier versions were each too weak, and each said "current" about a stale
 * service. `n_chunks` alone survived a rebuild that changed what the vectors hold
 * (2026-09-27: a title fix rebuilt the matrices while n_chunks stayed at 128841).
 * The content-addressed matrix NAMES that replaced it survived a rebuild that
 * changed only metadata: retag a topic, move a document to another tier or fix a
 * year, and the matrices are byte-identical while the filters a service applies
 * are not. `meta.json` holds both (it names every matrix by its hash AND carries
 * every document's metadata and the blend weights), so its bytes cover every
 * rebuild the ranker can observe, and hashing a few hundred kB costs nothing.
 *
 * It takes the bytes rather than a path so the service can fingerprint what it
 * LOADED while the checker fingerprints what is on disk now.
 *
 * @param {Buffer|string} metaBytes - The contents of an index's `meta.json`.
 * @returns {string} Twelve hex characters.
 */
export function indexFingerprint(metaBytes) {
  return createHash("sha256").update(metaBytes).digest("hex").slice(0, 12);
}

/** Resident set size in MB, the number A1 of the handoff asked to be logged. */
export function rssMb() {
  return Math.round(process.memoryUsage().rss / 1048576);
}

async function main() {
  const indexDir = resolve(process.argv[2] || corpusPath("dist/index"));
  const embedUrl = (process.env.EMBED_URL || "").replace(/\/$/, "");
  console.log(`rss before load: ${rssMb()} MB`);
  const { search, index, loadMs, reachable } = await loadIndexFromDisk({ indexDir, embedUrl });
  const { meta } = index;
  console.log(`loaded ${meta.n_documents} documents, ${meta.n_chunks} chunks at ${meta.dims} dims `
    + `(${meta.quant}) in ${loadMs.toFixed(0)} ms; rss ${rssMb()} MB`);

  // A passage's own vector must retrieve that passage: the same self-retrieval
  // the gate runs, on a handful of rows, so a broken load shows here and not in
  // a service that ranks nonsense politely.
  const pure = { ...index, pageWeight: 0, prevPageWeight: 0, sectionWeight: 0 };
  let failures = 0;
  const started = performance.now();
  const probes = 5;
  for (let i = 0; i < probes; i++) {
    const row = Math.floor((i + 0.5) * meta.n_chunks / probes);
    const hits = search.rank(pure, search.chunkVector(index, row), {});
    if (!hits.some((h) => h.chunk === row && h.cosine === hits[0].cosine)) failures++;
  }
  const perRank = (performance.now() - started) / probes;
  console.log(`self-retrieval: ${probes - failures}/${probes} probes rank themselves first, `
    + `${perRank.toFixed(0)} ms per rank`);

  if (embedUrl && await reachable(embedUrl)) {
    const question = process.env.QUESTION || "qualité de l'air en ville et santé respiratoire";
    const t0 = performance.now();
    const vector = await search.embedQuery(question, meta.dims);
    const t1 = performance.now();
    const hits = search.rank(index, vector, {}).slice(0, search.RESCORE_DEPTH);
    const t2 = performance.now();
    await search.attachText(index, hits);
    search.rescore(hits, question);
    const t3 = performance.now();
    console.log(`query "${question}": embed ${(t1 - t0).toFixed(0)} ms, rank ${(t2 - t1).toFixed(0)} ms, `
      + `text+rescore ${(t3 - t2).toFixed(0)} ms; rss ${rssMb()} MB`);
    for (const h of hits.slice(0, 3)) {
      console.log(`  ${h.score.toFixed(3)}  ${index.docs[h.doc].title.slice(0, 70)}  p.${h.page}`);
    }
  } else if (embedUrl) {
    console.log(`no encoder answering at ${embedUrl}: real query skipped`);
  }
  if (failures) {
    console.error(`FAIL: ${failures} probes did not retrieve themselves`);
    process.exit(1);
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
