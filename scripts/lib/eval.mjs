/* What the JavaScript evaluators share: the setup, the query file, the ranking
 * the site runs, the definition of a hit, the metrics and the report.
 *
 * evaluate_rescore.mjs (the BM25 weight), sweep_blend.mjs (the context weights),
 * sweep_recency.mjs (the recency bonus) and dump_candidates.mjs (the chunking
 * sweep's lists) all run the SHIPPED src/search.js over data/EVAL_QUERIES.tsv.
 * They have to agree on what counts as a hit, or their tables could not be read
 * against each other, and on how a list is ranked, or they measure a ranker
 * nobody runs. Each used to set itself up and rank on its own, and three of them
 * ranked with the bibliography penalty and the recency bonus OFF, which the site
 * has shipped ON since both existed. `rankLikeTheSite` is that ranking, once.
 *
 * Written by Claude Code (Fable 5.1).
 */

import { readFile, writeFile } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { installFetchShim } from "../../server/lib/browser_shim.mjs";
import { corpusPath } from "../../server/lib/corpus.mjs";

/** The repository root, so every evaluator resolves its defaults the same way. */
export const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");

/**
 * Set up an evaluation run: environment, encoder, src/search.js, index, queries.
 *
 * Reads INDEX_DIR (default dist/index), EMBED_URL and QUERIES (default
 * data/EVAL_QUERIES.tsv). The shim is installed BEFORE src/search.js is imported,
 * because the module reads `fetch` as the browser would and the shim is what maps
 * "index/..." onto the disk and "api/..." onto the encoder.
 *
 * @param {{encoder?: "required"|"optional"}} [options] - "optional" for a script
 *   that can run off a query-vector cache (dump_candidates.mjs); it then gets
 *   `haveEncoder` and decides for itself. "required" exits 2 without one.
 * @returns {Promise<{search: object, index: object, queries: object[], indexDir: string,
 *   embedUrl: string, queriesPath: string, haveEncoder: boolean}>}
 */
export async function openEvalRun({ encoder = "required" } = {}) {
  const indexDir = resolve(process.env.INDEX_DIR || corpusPath("dist/index"));
  const embedUrl = (process.env.EMBED_URL || "").replace(/\/$/, "");
  const queriesPath = process.env.QUERIES || corpusPath("data/EVAL_QUERIES.tsv");
  if (encoder === "required" && !embedUrl) {
    console.error("EMBED_URL is not set; this needs a running encoder (the real wire format is part of what is measured)");
    process.exit(2);
  }
  const { reachable } = installFetchShim({ indexDir, embedUrl });
  const haveEncoder = embedUrl ? await reachable(embedUrl) : false;
  if (encoder === "required" && !haveEncoder) {
    console.error(`no encoder answers at ${embedUrl}`);
    process.exit(2);
  }
  const search = await import(`${ROOT}/src/search.js`);
  const index = await search.loadIndex("index");
  const queries = await readQueries(queriesPath);
  return { search, index, queries, indexDir, embedUrl, queriesPath, haveEncoder };
}

/**
 * The options the site ranks with when the reader changed nothing.
 *
 * Mirrors the defaults of `createSearchService().answer` (server/service.mjs)
 * and of the page's toggles (src/app.js): bibliographies demoted, recent
 * documents favoured, described figures competing, NO lexical rescore.
 * server/service.mjs is not imported for them because loading it starts nothing
 * but still pulls in the HTTP layer; tests/eval.test.mjs pins the two together.
 */
export const SITE_DEFAULTS = Object.freeze({ bm25: false, demoteReferences: true, favourRecent: true, figures: "include" });

/**
 * Rank one question as the service would, minus folding, and return the list.
 *
 * The steps and their order are `answer`'s: rank with the request's options, copy
 * the first RESCORE_DEPTH candidates, rescore them lexically on the question's
 * POSITIVE words (`parsed.lexical`, never a subtracted one) when asked, then put
 * the rest back behind them. Folding is left out on purpose: it replaces a
 * document with its family's representative, and against a (file, page) label
 * that turns a correct retrieval into a miss.
 *
 * @param {object} search - The src/search.js module.
 * @param {object} index
 * @param {Int8Array} vector - The question's vector.
 * @param {{lexical: string}} parsed - `search.parseQuery(question)`.
 * @param {object} [options] - Any of SITE_DEFAULTS' keys, plus `weight` (the
 *   BM25 weight, default the shipped one) and `depth` (how much of the list to
 *   return, default RESCORE_DEPTH).
 * @returns {Promise<object[]>} Candidates, best first, with text and pages attached.
 */
export async function rankLikeTheSite(search, index, vector, parsed, options = {}) {
  const { bm25, demoteReferences, favourRecent, figures } = { ...SITE_DEFAULTS, ...options };
  const weight = options.weight ?? search.RESCORE_WEIGHT;
  const candidates = search.rank(index, vector, {}, { demoteReferences, favourRecent, figures });
  const head = candidates.slice(0, search.RESCORE_DEPTH).map((c) => ({ ...c }));
  if (bm25) {
    await search.attachText(index, head);
    search.rescore(head, parsed.lexical, weight);
  }
  const list = head.concat(candidates.slice(search.RESCORE_DEPTH)).slice(0, options.depth ?? search.RESCORE_DEPTH);
  // Text attached to everything returned, not only to what BM25 read, because
  // the pages a passage covers come with it and `goldRank` needs them.
  await search.attachText(index, list);
  return list;
}

/**
 * Write rows as a TSV, numbers that are not integers to four decimals.
 *
 * @param {string} path
 * @param {object[]} rows
 * @param {string[]} [columns] - Default: the first row's keys.
 */
export async function writeTsv(path, rows, columns = Object.keys(rows[0] ?? {})) {
  const cell = (v) => (typeof v === "number" && !Number.isInteger(v) ? v.toFixed(4) : String(v));
  const lines = [columns.join("\t"), ...rows.map((r) => columns.map((c) => cell(r[c])).join("\t"))];
  await writeFile(path, lines.join("\n") + "\n", "utf-8");
}

/**
 * Median publication year of the distinct documents in the first `k` candidates.
 *
 * Each document counts once: a list holding three passages of one 2012
 * argumentaire is one old document, not three. Documents with no year are
 * skipped.
 *
 * @returns {number|null} null when none of them has a year.
 */
export function medianYear(index, candidates, k = 10) {
  const years = [];
  const seen = new Set();
  for (const c of candidates.slice(0, k)) {
    if (seen.has(c.doc)) continue;
    seen.add(c.doc);
    const y = Number(String(index.docs[c.doc].year ?? "").trim());
    if (Number.isFinite(y) && y > 0) years.push(y);
  }
  if (!years.length) return null;
  years.sort((a, b) => a - b);
  return years[years.length >> 1];
}

/** The rows of an EVAL_QUERIES.tsv, one object per query keyed by the header. */
export async function readQueries(path) {
  const rows = (await readFile(path, "utf-8")).trim().split("\n");
  const header = rows.shift().split("\t");
  return rows.map((line) => Object.fromEntries(line.split("\t").map((v, i) => [header[i], v])));
}

/** Every (file, page) that answers a query: the gold label, then its alternatives.
 *
 * A hit is a candidate from the gold document whose pages cover the gold page, or
 * from any document named in the query's gold_alt cell: a corpus may hold four
 * editions of one reference book, and a question answered from the 14th when the
 * label names the 15th is a correct answer, not a retrieval failure.
 *
 * Deliberately the same six lines as `acceptable` in scripts/lib/evalbake.py: the
 * two evaluators are in different languages on purpose, and the cell syntax is
 * simple enough that a shared third format would cost more than the copy. Change
 * both together.
 */
export function acceptable(q) {
  const pairs = [[q.gold_file, Number(q.gold_page)]];
  for (const entry of (q.gold_alt || "").split(";")) {
    const cut = entry.lastIndexOf("#");
    if (cut > 0 && /^\d+$/.test(entry.slice(cut + 1))) {
      pairs.push([entry.slice(0, cut), Number(entry.slice(cut + 1))]);
    }
  }
  return pairs;
}

/**
 * Rank of the first candidate covering an acceptable page, or 0 for none in the shortlist.
 *
 * A candidate covers every page its passage's text sits on, which only
 * `attachText` knows: `rank` returns the first page alone. This used to fall
 * back to `[c.page]` when `pages` was missing, so a sweep that never attached
 * text scored a passage running from page 11 onto the gold page 12 as a miss,
 * and two sweeps of the same configuration disagreed (MRR 0.3344 against 0.3316
 * on 19 queries). A candidate without `pages` is now an error, not a guess.
 */
export function goldRank(index, candidates, q) {
  const wanted = acceptable(q);
  for (let i = 0; i < candidates.length; i++) {
    const c = candidates[i];
    const file = index.docs[c.doc].file;
    const pages = c.pages;
    if (!Array.isArray(pages)) throw new Error(`candidate ${c.chunk} has no pages: attachText before goldRank`);
    if (wanted.some(([name, page]) => file === name && pages.includes(page))) return i + 1;
  }
  return 0;
}

/** page@1, page@5, page@10 and MRR over a list of gold ranks (0 = not found). */
export function metrics(ranks) {
  const n = ranks.length;
  const at = (k) => ranks.filter((r) => r > 0 && r <= k).length / n;
  const mrr = ranks.reduce((sum, r) => sum + (r ? 1 / r : 0), 0) / n;
  return { n, page_1: at(1), page_5: at(5), page_10: at(10), mrr };
}

/** The standard error a page@k carries on n queries, at the worst case p = 0.5. */
export function standardError(n) {
  return 0.5 / Math.sqrt(n);
}
