/* Does BM25 rescoring of the shortlist help, and at what weight?
 *
 * The same queries and the same (file, page) ground truth as
 * scripts/evaluate.py, but run through the SHIPPED browser path rather than a
 * Python restatement of it: this imports src/search.js, so the tokeniser, the
 * BM25 and the blend measured here are the ones that run in the page. That is the
 * whole reason this is JavaScript and lives next to check_search.mjs instead of
 * being another mode of evaluate.py, where a second tokeniser would have to be
 * written and would then drift from the shipped one.
 *
 * A hit is a candidate from the gold document whose pages cover the gold page, or
 * from any document named in the query's gold_alt cell: a corpus may hold four
 * editions of one reference book, and a question answered from the 14th when the
 * label names the 15th is a correct answer, not a retrieval failure.
 *
 * Otherwise a hit is as evaluate.py defines it,
 * matching evaluate.py's definition: the eval set survives a chunker change, and
 * it matches what the reader gets, which is a page with a highlight on it.
 *
 * Only the top RESCORE_DEPTH candidates are measured, because that is all the
 * rescoring can reach; ranks below it are the dense ranking either way.
 *
 * Needs a running encoder, as evaluate.py does:
 *   cd ../justelesRCP && uv run src/embed-service.py --port 8461 --no-backlog
 *   EMBED_URL=http://127.0.0.1:8461 node scripts/evaluate_rescore.mjs
 *
 * Written by Claude Code.
 */

import { openEvalRun, rankLikeTheSite, goldRank, metrics, standardError, ROOT } from "./lib/eval.mjs";

// The weights to compare. 0 is the dense baseline and must stay in the list: the
// only interesting number here is the DIFFERENCE, and it is a few queries wide.
const WEIGHTS = (process.env.WEIGHTS || "0,0.05,0.1,0.12,0.2,0.3,0.5,1").split(",").map(Number);
const DEPTH = Number(process.env.DEPTH || 50);
// The other tuning parameter of the blend, swept the same way. 1 is the single
// global scale the term had before 2026-09-26, 0 is full per-language normalisation,
// and src/search.js ships the value in between; DESIGN.md has why.
const FLOOR = process.env.LEXICAL_FLOOR === undefined
  ? undefined : Number(process.env.LEXICAL_FLOOR);

const { search, index, queries, queriesPath, indexDir } = await openEvalRun();
console.log(`index: ${index.meta.n_chunks} chunks at ${index.meta.dims} dims (${index.meta.quant})`);
console.log(`queries: ${queries.length} from ${queriesPath.replace(ROOT + "/", "")}, shortlist depth ${DEPTH}`);

// One pass over the queries collects the shortlist with text attached; every
// weight is then scored off that, so the encoder is called once per query and the
// comparison is exactly like-for-like.
const shortlists = [];
let fetchedDocs = 0;
for (const q of queries) {
  const parsed = search.parseQuery(q.query);
  const vector = await search.embedParsedQuery(parsed, index.meta.dims);
  // The dense list as the site ranks it (bibliographies demoted, recent favoured),
  // so the weight measured is the one that would be added to what readers get.
  const candidates = await rankLikeTheSite(search, index, vector, parsed, { depth: DEPTH });
  fetchedDocs += new Set(candidates.map((c) => c.doc)).size;
  shortlists.push({ q, lexical: parsed.lexical, candidates });
}
console.log(`documents whose text a query needs: ${(fetchedDocs / queries.length).toFixed(1)} on average\n`);

// The definition of a hit lives in scripts/lib/eval.mjs, shared with the other sweeps.

const CUTOFFS = [1, 5, 10];
if (FLOOR !== undefined) console.log(`lexical group floor ${FLOOR}, overriding the shipped ${search.LEXICAL_GROUP_FLOOR}`);
console.log("weight  page@1  page@5  page@10    MRR   moved");
for (const weight of WEIGHTS) {
  const ranks = [];
  let moved = 0;
  for (const { q, lexical, candidates } of shortlists) {
    // A copy per weight: rescore sorts in place, and the dense order is the
    // baseline every run has to start from.
    const shortlist = candidates.map((c) => ({ ...c }));
    const before = goldRank(index, shortlist, q);
    search.rescore(shortlist, lexical, weight, FLOOR);
    const after = goldRank(index, shortlist, q);
    if (before !== after) moved++;
    ranks.push(after);
  }
  const m = metrics(ranks);
  console.log(
    `${String(weight).padStart(6)}  ${m.page_1.toFixed(4)}  ${m.page_5.toFixed(4)}   ${m.page_10.toFixed(4)} ` +
    ` ${m.mrr.toFixed(4)}  ${String(moved).padStart(5)}`
  );
}
// Computed rather than written down: the set has grown from 117 to 200 queries and
// a stale figure here would have a reader calling a real difference noise.
const stderr = standardError(queries.length);
console.log(`\nstandard error on ${queries.length} queries is about ${stderr.toFixed(3)}: `
  + `read differences, not decimals.`);

// --- per kind of question ----------------------------------------------------
// The average over all of them hides the thing most likely to break on its own. A
// crosslingual query is a French question whose answer is an English guideline, or
// the reverse: it is the claim this whole index rests on (one multilingual model,
// no translation step anywhere), it is 23 of the queries, and a change that helps
// the 94 others while quietly killing it would look like an improvement here. The
// same goes for `precise`, whose questions were written while looking at the
// passage, and `vague`, which are the ones a reader actually types.
//
// Reported at the shipped weight by default: the sweep above is for choosing the
// weight, this is for seeing which kind of question paid for it. KIND_WEIGHTS asks
// for the same breakdown at several weights, which is the only way to see a weight
// that buys page@1 overall by spending the crosslingual group: BM25 scores shared
// WORDS, and a French question about an English guideline shares none with the
// passage that answers it, so raising the weight can only dilute its cosine.
const SHIPPED_WEIGHT = search.RESCORE_WEIGHT;
const KIND_WEIGHTS = (process.env.KIND_WEIGHTS || String(SHIPPED_WEIGHT)).split(",").map(Number);
const kinds = [...new Set(queries.map((q) => q.type))].filter(Boolean).sort();
// Collected while the table below is printed, and written out as JSON when
// REPORT_JSON asks for it. scripts/bench.py reads that file rather than recomputing
// the metrics or scraping this table: one definition of a hit, the one above.
const byKind = [];
// "all" alone when every query carries the same type, which is what a filtered
// QUERIES file gives: the loop still runs so REPORT_JSON always has a row, only the
// table is skipped because a one-line breakdown of one kind says nothing.
const showTable = kinds.length > 1;
if (showTable) {
  console.log(`\nby kind of question, at weight ${KIND_WEIGHTS.join(", ")}` +
    (KIND_WEIGHTS.length === 1 ? " (the shipped one)" : ` (the shipped one is ${SHIPPED_WEIGHT})`) + ":");
  console.log("kind          weight    n  page@1  page@5  page@10    MRR");
}
for (const weight of KIND_WEIGHTS) {
  for (const kind of ["all", ...(showTable ? kinds : [])]) {
    const group = shortlists.filter(({ q }) => kind === "all" || q.type === kind);
    const ranks = group.map(({ q, lexical, candidates }) => {
      const shortlist = candidates.map((c) => ({ ...c }));
      search.rescore(shortlist, lexical, weight, FLOOR);
      return goldRank(index, shortlist, q);
    });
    const m = metrics(ranks);
    if (showTable) {
      console.log(
        `${kind.padEnd(13)}${String(weight).padStart(6)}${String(m.n).padStart(5)}  ` +
        `${m.page_1.toFixed(4)}  ${m.page_5.toFixed(4)}` +
        `   ${m.page_10.toFixed(4)}  ${m.mrr.toFixed(4)}`
      );
    }
    // REPORT_JSON carries the shipped weight's rows only, so bench.py keeps
    // comparing like with like however many weights this run printed.
    if (weight === SHIPPED_WEIGHT) byKind.push({ kind, ...m });
  }
}
// A group of 23 is noisy on its own: one query is 0.043 of its own page@1.
if (showTable) {
  console.log("a group of n queries carries a standard error near 0.5/sqrt(n): " +
              kinds.map((k) => {
                const n = queries.filter((q) => q.type === k).length;
                return `${k} ${standardError(n).toFixed(3)}`;
              }).join(", "));
}

// --- what a floor would cost -------------------------------------------------
// SEARCH_FLOOR hides anything scoring below it, which is only ever safe to set
// from the distribution of scores that CORRECT hits get. The scale moved when the
// index went binary (a cosine against a +/-1 vector is not a cosine against the
// real one), so the old evidence, "a good hit scores 0.55 to 0.70", says nothing
// about this index and the number has to be re-read off the corpus.
const goldCosines = [];
for (const { q, candidates } of shortlists) {
  const r = goldRank(index, candidates, q);
  if (r) goldCosines.push(candidates[r - 1].cosine);
}
goldCosines.sort((a, b) => a - b);
const pct = (p) => goldCosines[Math.min(goldCosines.length - 1, Math.floor(p * goldCosines.length))];
console.log(`\ncosine of the correct hit, over the ${goldCosines.length} queries that find one in the shortlist:`);
console.log(`  min ${goldCosines[0].toFixed(3)}  p05 ${pct(0.05).toFixed(3)}  p25 ${pct(0.25).toFixed(3)} ` +
            ` median ${pct(0.5).toFixed(3)}  p75 ${pct(0.75).toFixed(3)}  max ${goldCosines[goldCosines.length - 1].toFixed(3)}`);
const tops = shortlists.map(({ candidates }) => candidates[0].cosine).sort((a, b) => a - b);
console.log(`best candidate of each query: min ${tops[0].toFixed(3)}  median ${tops[Math.floor(tops.length / 2)].toFixed(3)}  max ${tops[tops.length - 1].toFixed(3)}`);
// The other half of the question, and the half the eval set cannot answer: what
// does a question this corpus does NOT cover score? Without it there is no gap to
// put the floor in, only a lower bound on the good hits. Two of these are
// deliberately medical-but-elsewhere, because those are the ones that score high
// and the ones a floor cannot honestly separate.
const OFF_CORPUS = [
  "recette de la tarte aux pommes",
  "comment changer la courroie de distribution d'une clio",
  "quel est le meilleur langage de programmation",
  "prix du billet de train Paris Lyon",
  "règles du jeu de pétanque",
  "traitement de l'insuffisance cardiaque chronique",
  "vaccination contre la grippe saisonnière",
];
console.log("\nbest cosine for a question this corpus does not answer:");
const offTops = [];
for (const q of OFF_CORPUS) {
  const parsed = search.parseQuery(q);
  const v = await search.embedParsedQuery(parsed, index.meta.dims);
  const top = (await rankLikeTheSite(search, index, v, parsed, { depth: 1 }))[0].cosine;
  offTops.push(top);
  console.log(`  ${top.toFixed(3)}  ${q}`);
}
console.log("");
console.log("floor  correct hits it would hide  queries left with nothing");
for (const floor of [0, 0.25, 0.3, 0.32, 0.35, 0.4, 0.45, 0.5]) {
  const hidden = goldCosines.filter((c) => c < floor).length;
  const empty = tops.filter((c) => c < floor).length;
  console.log(`${String(floor).padStart(5)}  ${String(hidden).padStart(24)}  ${String(empty).padStart(24)}`);
}

// --- the machine-readable report ---------------------------------------------
// Only written when asked for. scripts/bench.py sets REPORT_JSON and reads the
// numbers back, which is why the metrics above are collected rather than only
// printed: a sweep that parsed this script's table would silently start recording
// nonsense the day a column is widened.
const reportPath = process.env.REPORT_JSON;
if (reportPath) {
  const { writeFile } = await import("node:fs/promises");
  await writeFile(reportPath, JSON.stringify({
    queries: queries.length,
    queries_path: queriesPath,
    index_dir: indexDir,
    n_chunks: index.meta.n_chunks,
    dims: index.meta.dims,
    quant: index.meta.quant,
    shipped_weight: SHIPPED_WEIGHT,
    depth: DEPTH,
    by_kind: byKind,
  }, null, 2) + "\n", "utf-8");
  console.log(`wrote ${reportPath}`);
}
