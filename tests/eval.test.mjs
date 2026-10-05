/* scripts/lib/eval.mjs: the sweeps rank as the site does and count a hit the same way.
 *
 * Written by Claude Code (Opus 5.5).
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { join, resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { axis, writeTinyIndex } from "./lib/tiny_index.mjs";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const { rankLikeTheSite, goldRank, medianYear, SITE_DEFAULTS } = await import(`${root}/scripts/lib/eval.mjs`);
const { createSearchService } = await import(`${root}/server/service.mjs`);
const { loadIndexFromDisk } = await import(`${root}/server/load.mjs`);

test("the sweeps' defaults are the options the service ranks with when a request names none", async () => {
  let seen = null;
  const fake = {
    FIGURE_MODES: ["include", "exclude", "only"], RESCORE_DEPTH: 50,
    parseQuery: (q) => ({ positive: [q], negative: [], lexical: q }),
    embedParsedQuery: async () => new Int8Array(8),
    rank: (_index, _vector, _filters, options) => { seen = options; return []; },
    rescore: () => { throw new Error("the service rescored a request that did not ask"); },
    foldResults: () => [],
    attachText: async (_index, list) => list,
    facetFields: () => [], tierField: () => "guideline",
  };
  const service = createSearchService({ search: fake, index: { meta: { dims: 8 }, textCache: new Map() } });
  await service.answer({ q: "a plain question" });
  const { bm25, ...rankOptions } = SITE_DEFAULTS;
  assert.equal(bm25, false);
  assert.deepEqual(seen, rankOptions);
});

test("a hit counts every page the passage covers, not only the first", async () => {
  // Chunk 0 runs from page 1 onto page 2. Before 2026-10-03 a sweep that did not
  // attach text saw only page 1, scored chunk 1 as the first hit, and so reported
  // rank 2 where the reader gets the answer at rank 1.
  const dir = await writeTinyIndex();
  await writeFile(join(dir, "doc/0.json"), JSON.stringify({
    chunks: [[1, 2], [2], [2]].map((pages, n) => ({ text: `passage ${n}`, pages, boxes: [] })),
  }));
  const { search, index } = await loadIndexFromDisk({ indexDir: dir });
  const q = { gold_file: "a.pdf", gold_page: "2", gold_alt: "" };
  const list = await rankLikeTheSite(search, index, axis(0), { lexical: "passage" }, { depth: 3 });
  assert.equal(list[0].chunk, 0);
  assert.deepEqual(list[0].pages, [1, 2]);
  assert.equal(goldRank(index, list, q), 1);
  // Without the pages there is no honest answer, so there is none.
  const bare = search.rank(index, axis(0), {});
  assert.throws(() => goldRank(index, bare, q), /attachText before goldRank/);
});

test("the median year counts each document once", () => {
  const index = { docs: [{ year: "2010" }, { year: "2020" }, { year: "" }] };
  // Three passages of the 2010 document are still one old document.
  const list = [0, 0, 0, 1, 2].map((doc) => ({ doc }));
  assert.equal(medianYear(index, list), 2020);
  assert.equal(medianYear(index, [{ doc: 2 }]), null);
});
