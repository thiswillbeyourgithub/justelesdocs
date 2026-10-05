/* How much does favouring recent documents cost the retrieval it biases?
 *
 * `src/search.js` adds a per-document bonus to every passage's score, worth
 * RECENCY_BONUS at the newest year in the index and halving every
 * RECENCY_HALFLIFE years (see the constants there for why a bonus and not a sort
 * key). This sweeps both numbers over a grid THROUGH THE SHIPPED RANKER: one
 * index is loaded, each query is embedded once, and for every point the two knobs
 * are set on the index object and rank() runs again, so each row is what a reader
 * would get from a site built with those numbers.
 *
 * READ THE TABLE AS A COST, NOT A BENEFIT. data/EVAL_QUERIES.tsv labels each
 * query with the (file, page) that answers it, and that label was written by
 * reading the passage, not by preferring a recent document. A bias towards recent
 * documents can therefore only lose ground here, and the question this answers is
 * how much: the value worth shipping is the largest one whose loss stays inside a
 * standard error, because the thing it buys (the 2023 edition ahead of the 2012
 * one it replaced) is not in this file and cannot be measured from it.
 *
 * The `year` column is what makes the loss readable: it is the median publication
 * year of the documents in the first ten results, averaged over the queries. That
 * is the effect the reader actually sees, and it moves a lot faster than the
 * metrics fall.
 *
 * Ranked the way the page ranks by default: bibliographies demoted, no lexical
 * rescore (the BM25 box ships unticked), so the numbers here read against the
 * default column of the other evaluators rather than against their best row.
 *
 * Needs a running encoder:
 *   cd ../justelesRCP && uv run src/embed-service.py --port 8461 --no-backlog
 *   EMBED_URL=http://127.0.0.1:8461 node scripts/sweep_recency.mjs
 *
 * GRID selects the points: "coarse" (the default), or an explicit list
 * "bonus,halflife;bonus,halflife". REPORT_TSV appends one row per point.
 *
 * Written by Claude Code (Opus 5).
 */

import { openEvalRun, rankLikeTheSite, goldRank, metrics, standardError, writeTsv, medianYear, ROOT }
  from "./lib/eval.mjs";

const DEPTH = Number(process.env.DEPTH || 50);

const { search, index, queries, queriesPath } = await openEvalRun();

/** The points to try. Bonus 0 is always one of them: it is the baseline every
    other row is read against, and it is also what the reader gets from the
    unticked box, so it has to come out of the same run rather than from memory. */
function grid(spec) {
  const points = [];
  if (!spec || spec === "coarse") {
    for (const halfLife of [4, 8, 16]) {
      for (const bonus of [0.01, 0.02, 0.03, 0.05, 0.08, 0.12]) points.push({ bonus, halfLife });
    }
  } else {
    for (const pair of spec.split(";")) {
      const [bonus, halfLife] = pair.split(",").map(Number);
      points.push({ bonus, halfLife });
    }
  }
  points.unshift({ bonus: 0, halfLife: search.RECENCY_HALFLIFE });
  return points;
}

console.log(`index: ${index.meta.n_chunks} chunks in ${index.docs.length} documents ` +
            `at ${index.meta.dims} dims (${index.meta.quant})`);
console.log(`queries: ${queries.length} from ${queriesPath.replace(ROOT + "/", "")}, ` +
            `shortlist depth ${DEPTH}, ranked as the site ranks (no BM25, bibliographies demoted)`);

const embedded = [];
for (const q of queries) {
  const parsed = search.parseQuery(q.query);
  embedded.push({ q, parsed, vector: await search.embedParsedQuery(parsed, index.meta.dims) });
}

const kinds = [...new Set(queries.map((q) => q.type))].filter(Boolean).sort();
const points = grid(process.env.GRID);
console.log(`\n${points.length} points; standard error near ${standardError(queries.length).toFixed(3)} ` +
            `on all ${queries.length} queries`);
console.log("  bonus  half   page@1   page@5  page@10      MRR   year   (year = median year of the top 10 docs)");

const report = [];
for (const point of points) {
  index.recencyBonus = point.bonus;
  index.recencyHalfLife = point.halfLife;
  const ranks = [];
  const years = [];
  for (const { q, parsed, vector } of embedded) {
    const candidates = await rankLikeTheSite(search, index, vector, parsed, { depth: DEPTH });
    ranks.push(goldRank(index, candidates, q));
    const year = medianYear(index, candidates, 10);
    if (year !== null) years.push(year);
  }
  const m = metrics(ranks);
  const meanYear = years.reduce((a, b) => a + b, 0) / (years.length || 1);
  const perKind = Object.fromEntries(kinds.map((k) =>
    [k, metrics(ranks.filter((_, i) => queries[i].type === k))]));
  report.push({ bonus: point.bonus, half_life: point.halfLife, ...m, median_year: meanYear,
                ...Object.fromEntries(kinds.flatMap((k) => [[`${k}_mrr`, perKind[k].mrr]])) });
  console.log(`${String(point.bonus).padStart(7)}${String(point.halfLife).padStart(6)}` +
              `${m.page_1.toFixed(4).padStart(9)}${m.page_5.toFixed(4).padStart(9)}` +
              `${m.page_10.toFixed(4).padStart(9)}${m.mrr.toFixed(4).padStart(9)}` +
              `${meanYear.toFixed(1).padStart(8)}`);
}

if (process.env.REPORT_TSV) {
  await writeTsv(process.env.REPORT_TSV, report);
  console.log(`\nwrote ${process.env.REPORT_TSV}`);
}
