/* Which context weights should the score blend carry?
 *
 * The score of a passage is the weighted sum of four cosines (src/search.js:
 * its own, its page's, its previous page's, its section's). This sweeps the
 * three context weights over a grid THROUGH THE SHIPPED RANKER: one index is
 * loaded, each query is embedded once, and for every point of the grid the
 * weights are set on the index object and the list is ranked again exactly as
 * the site ranks it (lib/eval.mjs: rankLikeTheSite, the service's defaults), so
 * every row of the table is what a reader would get from a site built with those
 * numbers. BM25=1 measures the reader who ticked the lexical rescore instead.
 *
 * The index must have been built with every matrix present (a nonzero
 * --section-weight, or the section file is not written and the section term
 * cannot be measured). The weights it was built with are irrelevant here: they
 * are overridden per point. The row where every context weight is what
 * dist/index/meta.json ships is marked, so the table always has the shipped
 * configuration in it to read the others against.
 *
 * What was known before this existed (DESIGN.md): page 0.15 alone is worth 0.09
 * page@5 on a one-bit index, and the previous-page term measured ALONE
 * (eval_pagefirst.py) did not pay. This is the first measurement of the section
 * term, and the first of the three together.
 *
 * Needs a running encoder:
 *   cd ../justelesRCP && uv run src/embed-service.py --port 8461 --no-backlog
 *   EMBED_URL=http://127.0.0.1:8461 INDEX_DIR=data/index-sweep node scripts/sweep_blend.mjs
 *
 * GRID selects the points: "coarse" (the 5 x 3 x 3 grid from the handoff, plus
 * the shipped row), or an explicit list "section,page,prev;section,page,prev".
 * REPORT_TSV writes one row per point and kind, for DESIGN.md.
 *
 * Written by Claude Code (Fable 5.1).
 */

import { openEvalRun, rankLikeTheSite, goldRank, metrics, standardError, writeTsv, ROOT } from "./lib/eval.mjs";

const DEPTH = Number(process.env.DEPTH || 50);
const BM25 = process.env.BM25 === "1";
/** The configuration dist/index ships today, always one row of the table. */
const SHIPPED = { section: 0, page: 0.15, prev: 0 };

const { search, index, queries, queriesPath } = await openEvalRun();

function grid(spec) {
  const points = [];
  if (!spec || spec === "coarse") {
    for (const section of [0, 0.05, 0.1, 0.15, 0.2]) {
      for (const page of [0.05, 0.1, 0.15]) {
        for (const prev of [0, 0.025, 0.05]) points.push({ section, page, prev });
      }
    }
  } else {
    for (const triple of spec.split(";")) {
      const [section, page, prev] = triple.split(",").map(Number);
      points.push({ section, page, prev });
    }
  }
  const key = (p) => `${p.section},${p.page},${p.prev}`;
  if (!points.some((p) => key(p) === key(SHIPPED))) points.unshift(SHIPPED);
  return points;
}

console.log(`index: ${index.meta.n_chunks} chunks, ${index.meta.n_pages} pages, ` +
            `${index.meta.n_sections || 0} sections at ${index.meta.dims} dims (${index.meta.quant})`);
console.log(`queries: ${queries.length} from ${queriesPath.replace(ROOT + "/", "")}, ` +
            `shortlist depth ${DEPTH}, ` + (BM25 ? `BM25 at ${search.RESCORE_WEIGHT}` : "no BM25 (the site's default)"));

// Embedded once: the grid changes nothing about the query.
const embedded = [];
for (const q of queries) {
  const parsed = search.parseQuery(q.query);
  embedded.push({ q, parsed, vector: await search.embedParsedQuery(parsed, index.meta.dims) });
}

const kinds = [...new Set(queries.map((q) => q.type))].filter(Boolean).sort();
const points = grid(process.env.GRID);
// Sections are one term of three, so an index without them is only a problem when the
// grid actually asks for section weight. It is refused here rather than at load time
// because the page and previous-page questions are worth asking on their own, and
// building a section matrix costs a second corpus-sized bake to answer neither of them.
// `src/search.js` already treats a missing section matrix as weight zero rather than
// an error, so the ranker needs nothing from this.
if (!index.sections && points.some((p) => p.section > 0)) {
  console.error("this index ships no section matrix, and GRID asks for section weight: " +
                "build it with a nonzero --section-weight, or set every section term to 0");
  process.exit(2);
}
console.log(`\n${points.length} points; standard error near ${standardError(queries.length).toFixed(3)} ` +
            `on all ${queries.length}, ` + kinds.map((k) =>
              `${k} ${standardError(queries.filter((q) => q.type === k).length).toFixed(3)}`).join(", "));
const head = "section  page   prev  " + ["all", ...kinds].map((k) => k.padStart(12)).join("") + "   (page@5)";
console.log(head);
console.log("   MRR column, then page@5 per kind on the line below each");
const report = [];
for (const point of points) {
  index.sectionWeight = point.section;
  index.pageWeight = point.page;
  index.prevPageWeight = point.prev;
  const ranked = [];
  for (const { q, parsed, vector } of embedded) {
    // attachText caches per document, so after the first point the BM25 text is a lookup.
    const candidates = await rankLikeTheSite(search, index, vector, parsed, { bm25: BM25, depth: DEPTH });
    ranked.push({ q, rank: goldRank(index, candidates, q) });
  }
  const perKind = {};
  for (const kind of ["all", ...kinds]) {
    const ranks = ranked.filter(({ q }) => kind === "all" || q.type === kind).map((r) => r.rank);
    perKind[kind] = metrics(ranks);
    report.push({ ...point, kind, ...perKind[kind] });
  }
  const mark = point.section === SHIPPED.section && point.page === SHIPPED.page && point.prev === SHIPPED.prev
    ? " shipped" : "";
  const cell = (k, f) => perKind[k][f].toFixed(4).padStart(12);
  console.log(`${String(point.section).padStart(7)}${String(point.page).padStart(6)}${String(point.prev).padStart(7)}  ` +
              ["all", ...kinds].map((k) => cell(k, "mrr")).join("") + "   MRR" + mark);
  console.log(`${"".padStart(20)}  ` + ["all", ...kinds].map((k) => cell(k, "page_5")).join("") + "   page@5");
}

const reportPath = process.env.REPORT_TSV;
if (reportPath) {
  await writeTsv(reportPath, report, ["section", "page", "prev", "kind", "n", "page_1", "page_5", "page_10", "mrr"]);
  console.log(`\nwrote ${reportPath}`);
}
