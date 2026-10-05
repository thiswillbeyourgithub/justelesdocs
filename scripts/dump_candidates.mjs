/* What the reader would actually see, for one index, dumped for an offline judge.
 *
 * The chunking sweep builds one index per strategy and has to compare them. The
 * (file, page) ground truth in data/EVAL_QUERIES.tsv answers "did the labelled page
 * come back", which counts a BETTER passage from another document as a miss: with
 * 535 documents and several renditions of the same guideline, that is a real and
 * one-sided error. So each index's top K is dumped here, verbatim, and graded
 * afterwards by a cross-encoder that never learns which strategy produced what
 * (scripts/judge_rerank.py).
 *
 * The list dumped is ranked by the SHIPPED src/search.js, through the same
 * rankLikeTheSite as every other sweep (lib/eval.mjs), with the blend weights
 * baked into the index. Its options are PINNED, not the site's defaults: BM25 at
 * search.RESCORE_WEIGHT on, bibliography penalty and recency bonus off. That is
 * what every dump already judged under data/grid/ was ranked with, and the
 * sweep skips a strategy whose dump exists, so changing them here would mix two
 * rankers in one table without a word. They are written into the dump's `meta`
 * (`rank_options`) so the table can be checked rather than trusted.
 *
 * Query vectors are cached on disk, not just in the process: the sweep runs this
 * script once per strategy, and an encoder that has to stay up for twelve hours is a
 * twelve-hour dependency. With a warm cache the only thing this needs is the index.
 * The cache key carries the width, because a vector is only reusable at the width it
 * was asked for.
 *
 *   EMBED_URL=http://127.0.0.1:8461 INDEX_DIR=data/grid/t1024-o256-section/index \
 *     OUT=data/grid/t1024-o256-section/candidates.json node scripts/dump_candidates.mjs
 *
 * Environment: INDEX_DIR, OUT, EMBED_URL, QUERIES, TOPK (10), QVEC_CACHE,
 * STRATEGY (a label carried into the dump), LIMIT (first N queries, for a smoke run).
 *
 * Written by Claude Code (Opus 5).
 */
import { resolve, dirname } from "node:path";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { createHash } from "node:crypto";
import { openEvalRun, rankLikeTheSite, goldRank, ROOT as root } from "./lib/eval.mjs";
import { corpusPath } from "../server/lib/corpus.mjs";

/** See the header: pinned for comparability with the dumps already judged. */
const RANK_OPTIONS = Object.freeze({ bm25: true, demoteReferences: false, favourRecent: false, figures: "include" });

const outPath = resolve(process.env.OUT || corpusPath("data/grid/candidates.json"));
const cachePath = resolve(process.env.QVEC_CACHE || corpusPath("data/grid/query-vectors.json"));
const topk = Number(process.env.TOPK || 10);
const limit = Number(process.env.LIMIT || 0);
const strategy = process.env.STRATEGY || resolve(process.env.INDEX_DIR || corpusPath("dist/index")).split("/").slice(-2)[0];

/** The disk cache: {"<dims>\u0000<question>": "<base64 of the int8 vector>"}. */
async function loadCache() {
  try {
    return JSON.parse(await readFile(cachePath, "utf-8"));
  } catch {
    return {};
  }
}

const cache = await loadCache();
let embedded = 0;

// Without an encoder the run is still fine as long as every query is already in
// the cache, which is the normal case from the second strategy onwards.
const run = await openEvalRun({ encoder: "optional" });
const { search, index, indexDir, embedUrl, haveEncoder } = run;
const queries = limit > 0 ? run.queries.slice(0, limit) : run.queries;

/** One query's vector, from the disk cache or from the encoder. */
async function queryVector(question) {
  const key = `${index.meta.dims}\u0000${question}`;
  const hit = cache[key];
  if (hit) return Int8Array.from(Buffer.from(hit, "base64"));
  if (!haveEncoder) {
    console.error(`no encoder at "${embedUrl}" and "${question.slice(0, 40)}..." is not cached`);
    process.exit(2);
  }
  const vector = await search.embedQuery(question, index.meta.dims);
  cache[key] = Buffer.from(vector.buffer, vector.byteOffset, vector.byteLength).toString("base64");
  embedded++;
  return vector;
}

console.log(`${strategy}: ${index.meta.n_chunks} chunks at ${index.meta.dims} dims ` +
            `(${index.meta.quant}), ${queries.length} queries, top ${topk}`);

const dumped = [];
for (const q of queries) {
  const vector = await queryVector(q.query);
  // Rescored before it is cut to K: rankLikeTheSite rescores its first
  // RESCORE_DEPTH candidates, so a passage from rank 40 can still reach the top K.
  const candidates = await rankLikeTheSite(search, index, vector, search.parseQuery(q.query),
                                           { ...RANK_OPTIONS, depth: Math.max(topk, search.RESCORE_DEPTH) });
  const top = candidates.slice(0, topk);
  dumped.push({
    query_id: q.query_id,
    query: q.query,
    type: q.type,
    query_lang: q.query_lang,
    // Kept alongside the judge's verdict on purpose: the two metrics fail
    // differently, and a strategy that wins on one and loses on the other is the
    // interesting case rather than an inconsistency to hide.
    gold_rank: goldRank(index, candidates, q),
    candidates: top.map((c, i) => ({
      rank: i + 1,
      doc: c.doc,
      file: index.docs[c.doc].file,
      chunk: c.chunk,
      pages: c.pages || [c.page],
      score: Number(c.score.toFixed(6)),
      // The JavaScript twin of scripts/lib/judge_cache.py's sha(), pinned to it by
      // tests/test_grid.py. The judge caches its verdicts by (query, this hash), so the same passage
      // retrieved by three strategies is scored once. With 23 strategies over 200
      // queries that is the difference between four GPU hours and twelve.
      text_sha: createHash("sha256").update(c.text || "").digest("hex").slice(0, 16),
      text: c.text || "",
    })),
  });
}

await mkdir(dirname(outPath), { recursive: true });
await writeFile(outPath, JSON.stringify({
  strategy,
  index_dir: indexDir.replace(root + "/", ""),
  meta: {
    n_chunks: index.meta.n_chunks, dims: index.meta.dims, quant: index.meta.quant,
    page_weight: index.meta.page_weight, section_weight: index.meta.section_weight,
    rank_options: RANK_OPTIONS,
  },
  topk,
  queries: dumped,
}, null, 1) + "\n", "utf-8");

if (embedded) {
  await mkdir(dirname(cachePath), { recursive: true });
  await writeFile(cachePath, JSON.stringify(cache), "utf-8");
}
const found = dumped.filter((d) => d.gold_rank > 0 && d.gold_rank <= topk).length;
console.log(`wrote ${outPath.replace(root + "/", "")}: ${dumped.length} queries, ` +
            `${embedded} embedded (${Object.keys(cache).length} cached), ` +
            `labelled page in top ${topk} for ${found}`);
