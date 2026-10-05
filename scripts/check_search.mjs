#!/usr/bin/env node
/**
 * Hard gate on the shipped search path: run src/search.js against a real
 * dist/index/ and refuse a build whose ranking is wrong.
 *
 * This exists because every other check in the chain stops at the artefacts.
 * verify_chunks.py proves the boxes are drawable, build_index.py proves the
 * vectors and the manifest agree, and neither can notice that the browser reads
 * the same bytes back in the wrong order. The two failures that motivated the
 * script are silent: a byte-order or stride mistake makes chunks.u16 decode into
 * page numbers in the thousands, and a wrong offset in the per-document text
 * files puts a neighbouring passage's words under a correct highlight. Both
 * still render.
 *
 * The trick that makes an end-to-end test cheap is that queries and passages
 * live in the same space: a passage's own int8 row is a legitimate query vector,
 * and it must retrieve itself at rank 1 with a cosine of 1. No embedding
 * service, no browser and no network are involved, so this runs in the build
 * chain rather than in a test suite nobody starts.
 *
 * With EMBED_URL pointing at a running encoder (the sibling's embed-service.py,
 * see scripts/dev_server.py), it also puts real clinical questions through the real
 * wire format and prints what comes back. That part asserts only what can be
 * asserted mechanically, that the width matches and the answers are ordered and
 * in range; whether the hits are the right documents is a judgement a human makes
 * by reading the list, which is why it is printed rather than scored.
 *
 * Usage: node scripts/check_search.mjs [dist/index]
 *        EMBED_URL=http://127.0.0.1:8461 node scripts/check_search.mjs
 *
 * Written by Claude Code.
 */

import { existsSync, readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { installFetchShim, installDocumentStub } from "../server/lib/browser_shim.mjs";
import { medianYear } from "./lib/eval.mjs";
import { corpusPath } from "../server/lib/corpus.mjs";
import { loadScenarios } from "./lib/scenarios.mjs";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const indexDir = resolve(process.argv[2] || corpusPath("dist/index"));
const embedUrl = (process.env.EMBED_URL || "").replace(/\/$/, "");
const serviceUrl = (process.env.SVC || "").replace(/\/$/, "");
// Captured BEFORE installFetchShim below, which allows the index directory and
// the encoder and refuses everything else. The staleness check at the bottom talks
// to a running service, which is neither of those. Distinct from `realFetch` lower
// down: that one is captured AFTER the shim is installed, so it IS the shim, and
// line 597 restores it. This is the genuine one.
const preShimFetch = globalThis.fetch;

// The shim serves index/ off the filesystem and forwards the one encoder request
// to EMBED_URL when it is set, so src/search.js runs here exactly as it does in a
// browser. It lives in scripts/lib because evaluate_rescore.mjs needs it too.
const { reachable } = installFetchShim({ indexDir, embedUrl });

// The corpus's string overlay and languages, as the browser gets them: from the
// staged site-config.js, which i18n.js reads at import, so it is set before the
// import below. Absent (nothing staged), the software's own tables are checked.
const siteConfig = corpusPath("dist/www/site-config.js");
if (existsSync(siteConfig)) {
  const text = readFileSync(siteConfig, "utf8");
  globalThis.__SITE__ = JSON.parse(text.slice(text.indexOf("{"), text.lastIndexOf("}") + 1));
}

const search = await import(`${root}/src/search.js`);
// The same module instance search.js itself uses, so setLang below moves the whole
// shipped path, not a private copy of the string table.
const i18n = await import(`${root}/src/i18n.js`);
installDocumentStub();  // setLang mirrors the choice into <html lang>; see the shim

let failures = 0;
const check = (ok, label, detail = "") => {
  if (ok) {
    console.log(`  ok   ${label}${detail ? ` (${detail})` : ""}`);
  } else {
    failures++;
    console.error(`  FAIL ${label}${detail ? `: ${detail}` : ""}`);
  }
};

const index = await search.loadIndex("index");
const { meta } = index;
console.log(`index: ${meta.n_documents} documents, ${meta.n_chunks} chunks at ${meta.dims} dims (${meta.quant}, ${meta.bytes_per_vector} B/vector), bake ${meta.bake}`);

console.log("layout");
check(index.locations.length === meta.n_chunks * search.LOC_STRIDE,
      `chunks.u16 holds ${search.LOC_STRIDE} values per chunk`,
      `${index.locations.length} for ${meta.n_chunks} chunks`);
// Offsets must tile the row space with no gap and no overlap, because rank()
// reaches vectors through them: a gap makes passages unreachable, an overlap
// attributes one document's passage to another.
let cursor = 0;
let tiled = true;
for (const doc of index.docs) {
  if (doc.chunk_offset !== cursor) tiled = false;
  cursor += doc.chunk_count;
}
check(tiled && cursor === meta.n_chunks, "document offsets tile the vector rows", `end ${cursor}`);
// Decoded as anything but little-endian u16 pairs, these are absurd rather than
// merely wrong, which is what makes the assertion worth making.
// A loop rather than Math.max(...spread): 120k arguments overflow the call stack.
let minPage = Infinity, maxPage = 0, maxDoc = 0;
for (let c = 0; c < meta.n_chunks; c++) {
  const page = index.locations[c * search.LOC_STRIDE + 1];
  if (page < minPage) minPage = page;
  if (page > maxPage) maxPage = page;
  const doc = index.locations[c * search.LOC_STRIDE];
  if (doc > maxDoc) maxDoc = doc;
}
check(maxPage <= 2000 && minPage >= 1, "page numbers are plausible", `${minPage}..${maxPage}`);
check(maxDoc === meta.n_documents - 1, "document ids in chunks.u16 span the corpus");

// Same index, every context term off. rank() reads the weights off the object
// it is given, so this needs no second load and no flag inside the ranker.
const pure = { ...index, pageWeight: 0, prevPageWeight: 0, sectionWeight: 0 };
console.log("retrieval (a passage's own vector must retrieve that passage)");
const sample = [];
for (let i = 0; i < 12; i++) sample.push(Math.floor((i + 0.5) * meta.n_chunks / 12));
let worstSelf = Infinity;
let wrongTop = 0;
let outOfOrder = 0;
for (const row of sample) {
  const query = search.chunkVector(index, row);
  const hits = search.rank(index, query, {});
  // The self-similarity BOUND below is a statement about the passage arithmetic
  // alone, so it is taken with the page term off: blended, a self-match scores
  // 0.85 + 0.15 * (its page's cosine) and would say nothing about strides. The
  // rank-itself-first check stays on the real index, where it matters.
  const pureHits = search.rank(pure, query, {});
  worstSelf = Math.min(worstSelf, pureHits[0].cosine);
  // A tie at the top counts as ranking itself first, because it is one. The corpus
  // holds several renditions of the same guidance, and where two of them repeat a
  // passage word for word the two vectors are identical: nothing in the arithmetic
  // can prefer one, and demanding a winner asks the index to invent an order. What
  // the probe is really about is that no DIFFERENT passage outscores the one asked
  // for, and that is what this says.
  const self = hits.find((hit) => hit.chunk === row);
  if (!self || self.cosine < hits[0].cosine) wrongTop++;
  for (let i = 1; i < hits.length; i++) if (hits[i].cosine > hits[i - 1].cosine) outOfOrder++;
  // Whatever came top, the manifest must agree about which document owns it.
  const owner = index.docs[hits[0].doc];
  if (!(hits[0].chunk >= owner.chunk_offset
        && hits[0].chunk < owner.chunk_offset + owner.chunk_count)) wrongTop++;
}
check(wrongTop === 0, "every sampled passage ranks itself first", `${sample.length} probes`);
// Not exactly 1, and it cannot be, on either index layout. int8: the cosine is
// the raw dot divided by the fixed 127*127 with no per-vector norm correction,
// and rounding components whose typical magnitude is 8 counts moves the
// reconstructed norm by a fraction of a percent. Binary: chunkVector rebuilds the
// query as +/-round(127/sqrt(dims)), and that rounding leaves the scale off by
// about a percent in either direction. The tolerance is wide enough for both and
// far too tight for a stride, offset or bit-order mistake, which lands near zero.
check(worstSelf > 0.98 && worstSelf < 1.02, "self-similarity is 1 within quantisation", `worst ${worstSelf.toFixed(5)}`);
check(outOfOrder === 0, "candidates come back in descending order");

// Plain float cosine of two unpacked vectors. Used by the page-blend checks and
// by the query-algebra section further down.
const unit = (v) => {
  const out = Float64Array.from(v);
  let n = 0;
  for (const x of out) n += x * x;
  n = Math.sqrt(n) || 1;
  return out.map((x) => x / n);
};
const cos = (a, b) => {
  const ua = unit(a), ub = unit(b);
  let d = 0;
  for (let i = 0; i < ua.length; i++) d += ua[i] * ub[i];
  return d;
};

console.log("context blend (the page and section matrices must line up with the passage matrix)");
check(index.pages !== null && meta.n_pages > 0, "the index ships page vectors",
      `${meta.n_pages} pages for ${meta.n_chunks} chunks`);
const weights = {
  page: meta.page_weight || 0, prev: meta.prev_page_weight || 0, section: meta.section_weight || 0,
};
const contextShare = weights.page + weights.prev + weights.section;
check(weights.page > 0 && contextShare < 0.5,
      "the context terms are a minority share of the score",
      `page ${weights.page}, previous page ${weights.prev}, section ${weights.section}`);
check((weights.section > 0) === (index.sections !== null),
      "the section matrix ships exactly when its weight is on",
      `weight ${weights.section}, ${index.sections ? meta.n_sections : "no"} sections`);
const S = search.LOC_STRIDE;
let badRow = 0;
let noPage = 0;
let badSection = 0;
let noSection = 0;
for (let c = 0; c < meta.n_chunks; c++) {
  const row = index.locations[c * S + 2];
  if (row === search.NO_PAGE_ROW) noPage++;
  else if (row >= meta.n_pages) badRow++;
  const section = index.locations[c * S + 3];
  if (section === search.NO_PAGE_ROW) noSection++;
  else if (section >= (meta.n_sections || 0)) badSection++;
}
// An out-of-range row would read past the page matrix and score a passage on
// undefined, which JavaScript turns into NaN rather than an error: NaN loses
// every comparison, so the passage would quietly become unrankable.
check(badRow === 0, "every chunk's page row is inside the page matrix", `${badRow} out of range`);
// A page with boxes but no packable text is rare and legitimate (a full-page
// figure). A lot of them would mean the page bake and the passage bake were
// chunked from different text.
check(noPage < meta.n_chunks * 0.02, "almost every chunk has a page vector",
      `${noPage} of ${meta.n_chunks} without`);
if (index.sections) {
  check(badSection === 0, "every chunk's section row is inside the section matrix",
        `${badSection} out of range`);
  // Same reasoning as the page: a chunk no section claims is a chunk whose
  // boxes the section bake never saw, which build_index.py already caps.
  check(noSection < meta.n_chunks * 0.02, "almost every chunk has a section vector",
        `${noSection} of ${meta.n_chunks} without`);
}
// The alignment check proper. Lengths and ranges all stay happy if the page rows
// are offset by a document, and the site would then score every passage against
// somebody else's page; being nearest to its OWN page is what an offset breaks.
// The section matrix gets the same treatment, with the same sample.
let ownPageBest = 0;
let pageProbes = 0;
let pageTies = 0;
let ownSectionBest = 0;
let sectionProbes = 0;
let worstBlend = 0;
// `own` is every row the passage may call its own: a passage running over a page
// break is filed under its first page, yet most of its words can be on the second
// (a textbook table that starts at the foot of page 682 scores 0.40 against that
// page and 0.57 against 683), so every page it has a box on counts.
const nearestIsOwn = (v, own, rows, vectorOf) => {
  // Strictly greater, not >=. A one-bit row takes discrete cosines, so an exact
  // tie between two pages is ordinary rather than suspicious: it first turned up
  // between two pages of a reference handbook whose entries are
  // the same page with the subject swapped. An offset does not produce ties, it
  // produces an own-page score that loses outright, which this still catches.
  const ownScore = Math.max(...own.map((r) => cos(v, vectorOf(r))));
  let best = true;
  let tied = false;
  for (let k = 0; k < 24; k++) {
    const other = Math.floor((k + 0.5) * rows / 24);
    if (own.includes(other)) continue;
    const score = cos(v, vectorOf(other));
    if (score > ownScore) best = false;
    else if (score === ownScore) tied = true;
  }
  return { best, tied, ownScore };
};
const pageRows = search.pageRowMap(index.locations, meta.n_chunks);
for (const row of sample) {
  const pageRow = index.locations[row * S + 2];
  if (pageRow === search.NO_PAGE_ROW) continue;
  pageProbes++;
  const v = search.chunkVector(index, row);
  const docId = index.locations[row * S];
  const { chunks } = JSON.parse(readFileSync(`${indexDir}/doc/${docId}.json`, "utf8"));
  const pages = chunks[row - meta.documents[docId].chunk_offset].pages;
  const own = [pageRow, ...pages.map((p) => pageRows.get(search.pageKey(docId, p)))
    .filter((r) => r !== undefined && r !== pageRow)];
  const page = nearestIsOwn(v, own, meta.n_pages, (r) => search.pageVector(index, r));
  if (page.best) ownPageBest++;
  if (page.best && page.tied) pageTies++;
  // The blend's other two terms, read the way rank() reads them: a missing row
  // takes the chunk's own cosine.
  const alone = search.rank(pure, v, {}).find((h) => h.chunk === row);
  let prev = alone.cosine;
  const before = index.prevRows[row];
  if (before !== search.NO_PAGE_ROW) prev = cos(v, search.pageVector(index, before));
  let section = alone.cosine;
  const sectionRow = index.locations[row * S + 3];
  if (index.sections && sectionRow !== search.NO_PAGE_ROW) {
    sectionProbes++;
    const s = nearestIsOwn(v, [sectionRow], meta.n_sections, (r) => search.sectionVector(index, r));
    if (s.best) ownSectionBest++;
    section = s.ownScore;
  }
  // And the arithmetic the docstrings promise: a passage's blended score must be
  // (1 - p - q - s) times its own cosine plus the weighted context cosines,
  // within the same 2% the self-similarity check allows for unpacking a one-bit
  // row onto the query scale.
  const blended = search.rank(index, v, {}).find((h) => h.chunk === row);
  const expected = (1 - contextShare) * alone.cosine + weights.page * cos(v, search.pageVector(index, pageRow))
    + weights.prev * prev + weights.section * section;
  worstBlend = Math.max(worstBlend, Math.abs(blended.cosine - expected));
}
check(ownPageBest === pageProbes, "no sampled passage is nearer another page than its own",
      `${ownPageBest}/${pageProbes}, ${pageTies} tied`);
if (index.sections) {
  check(ownSectionBest === sectionProbes, "no sampled passage is nearer another section than its own",
        `${ownSectionBest}/${sectionProbes}`);
}
check(worstBlend < 0.02, "a blended score is the weighted sum the module docstring states",
      `worst error ${worstBlend.toFixed(4)}`);

console.log("text (the lazy per-document fetch must land on the right chunk)");
const probe = sample[4];
const hits = await search.attachText(index, search.foldResults(index, search.rank(index, search.chunkVector(index, probe), {})), "index");
check(hits.length > 0, "folding keeps at least the exact match");
const top = hits[0];
check(typeof top.text === "string" && top.text.length > 0, "the top hit carries its text",
      `${(top.text || "").length} chars`);
check(Array.isArray(top.pages) && top.pages.includes(top.page), "the listed page is one of the chunk's pages",
      `page ${top.page} of ${JSON.stringify(top.pages)}`);
const drawn = Object.entries(top.boxes || {});
check(drawn.length > 0 && drawn.every(([, rects]) => Array.isArray(rects) && rects.length),
      "the top hit carries highlight boxes", `${drawn.length} page(s), ${drawn.reduce((n, [, r]) => n + r.length, 0)} rects`);
check(drawn.every(([page]) => top.pages.includes(Number(page))),
      "every page the boxes mention is a page the chunk claims");
// The page the list shows is the page the viewer opens. Both derive it as "the
// page carrying most of the match", one in build_index.py and one in viewer.js,
// so this is the check that keeps those two copies of the rule agreeing.
const busiest = drawn.sort((a, b) => b[1].length - a[1].length)[0];
check(Number(busiest[0]) === top.page, "the shown page is the one with most of the match",
      `${top.page} vs ${busiest[0]}`);

console.log("filters");
// The fields the index declares (corpus.toml [facets]), so this gate checks what
// the page offers rather than a list of its own.
const available = search.facets(index, search.facetFields(index));
console.log(`  note  ${available.length} facets have more than one value: ${available.map((f) => `${f.field} (${f.values.length})`).join(", ") || "none"}`);
const unfiltered = search.rank(index, search.chunkVector(index, probe), {}).length;

const facet = available.find((f) => f.field === "issuer") || available[0];
const wanted = facet.values[0];
const filtered = search.rank(index, search.chunkVector(index, probe), { [facet.field]: [wanted] });
check(filtered.every((h) => search.cellValues(index, index.docs[h.doc], facet.field).includes(wanted)),
      "a filtered search returns only matching documents", `${facet.field}=${wanted}, ${filtered.length} hits`);
check(filtered.length > 0 && filtered.length <= unfiltered, "filtering narrows rather than widens");
const blanks = index.docs.filter((d) => !(d[facet.field] || "").trim()).length;
check(filtered.every((h) => (index.docs[h.doc][facet.field] || "").trim()),
      "documents with a blank cell are excluded by a chosen filter", `${blanks} blank`);

// doc_type and topic hold lists. The failure this guards against is the facet
// offering the raw cell as one option: "recommandation;synthese" would appear in
// the dropdown as a third kind of document, matching the single row that carries
// it, and every other recommandation would be missing from "Recommandation".
const sep = index.meta.separator;
check(typeof sep === "string" && sep.length === 1,
      "meta.json carries the multi-value separator", JSON.stringify(sep));
const multi = available.filter((f) => f.values.some((v) => v.includes(sep)));
check(multi.length === 0, "no facet option is an unsplit multi-value cell",
      multi.map((f) => f.field).join(", ") || "none");
const listy = index.docs.filter((d) => (d.topic || "").includes(sep)).length;
check(listy > 0, "the corpus actually exercises multi-valued cells",
      `${listy}/${index.docs.length} documents carry several topics`);

// ANY, not ALL: a document tagged tsa;adulte;diagnostic must appear under each of
// those three, or the reader has to reproduce the curator's exact tag set.
const topicFacet = available.find((f) => f.field === "topic");
const topic = topicFacet.values.map((v) => [v, index.docs.filter((d) => search.cellValues(index, d, "topic").includes(v)).length])
                               .sort((a, b) => b[1] - a[1])[0][0];
const expected = index.docs.map((d, id) => [d, id]).filter(([d]) => search.cellValues(index, d, "topic").includes(topic)).map(([, id]) => id);
const allowed = search.allowedDocs(index, { topic });
check(expected.length > 1 && expected.every((id) => allowed.has(id)) && allowed.size === expected.length,
      "a multi-valued cell matches on ANY of its values", `topic=${topic}, ${expected.length} documents`);
check(expected.some((id) => search.cellValues(index, index.docs[id], "topic").length > 1),
      "and at least one of them carries other topics too");

// The year slider. Its filter is an object, not a string or an array, which is
// exactly what the old `v && v.length` activity test dropped on the floor.
const years = index.docs.map((d) => Number(d.year)).filter(Number.isFinite).sort((a, b) => a - b);
const lo = years[Math.floor(years.length / 3)];
const hi = years[Math.floor((2 * years.length) / 3)];
const inRange = search.allowedDocs(index, { year: { from: lo, to: hi } });
check(inRange !== null, "a year range is recognised as an active filter");
check([...inRange].every((id) => {
        const y = Number(index.docs[id].year);
        return Number.isFinite(y) && y >= lo && y <= hi;
      }), "a year range admits only documents inside it", `${lo}-${hi}, ${inRange.size} documents`);
check(inRange.size < index.docs.length && inRange.size > 0, "and it narrows the corpus");
const oneEnded = search.allowedDocs(index, { year: { from: lo } });
check(oneEnded.size >= inRange.size && [...oneEnded].every((id) => Number(index.docs[id].year) >= lo),
      "an open-ended range works from one handle alone", `from ${lo}, ${oneEnded.size} documents`);

// Every value the reader can select must be readable in every language. An
// unlabelled slug falls back to itself, which is a deliberate bug report rather
// than a blank, so this is the check that keeps it from ever being seen.
const unlabelled = [];
const ranges = search.rangeFields(index);
for (const langCode of i18n.languages()) {
  i18n.setLang(langCode);
  for (const { field, values } of available) {
    if (ranges.has(field)) continue;  // numbers, not slugs: nothing to translate
    for (const value of values) {
      const key = `facet_${field}_${value}`;
      if (i18n.t(key) === key) unlabelled.push(`${langCode}:${key}`);
    }
  }
}
i18n.setLang(i18n.languages()[0]);
check(unlabelled.length === 0, "every shipped facet value has a label in every language",
      unlabelled.join(", ") || `${available.reduce((n, f) => n + (ranges.has(f.field) ? 0 : f.values.length), 0)} values x ${i18n.languages().length}`);

console.log("folding (family collapse and the per-document cap are separate rules)");
// Synthetic candidates, because the corpus cannot exercise every branch: only 50
// of 534 documents carry a family today, and a pnds family holds no two
// redundant renditions.
const fake = {
  docs: [
    { id: 0, family: "f", rendition: "synthese", chunk_offset: 0, chunk_count: 9 },
    { id: 1, family: "f", rendition: "argumentaire", chunk_offset: 9, chunk_count: 9 },
    { id: 2, family: "f", rendition: "lap", chunk_offset: 18, chunk_count: 9 },
    { id: 3, family: "f", rendition: "unheard_of", chunk_offset: 27, chunk_count: 9 },
    { id: 4, family: "", rendition: "synthese", chunk_offset: 36, chunk_count: 9 },
  ],
  redundant: new Set(["synthese", "argumentaire"]),
};
const cand = (doc, cosine) => ({ chunk: doc * 9, doc, page: 1, cosine });
let folded = search.foldResults(fake, [cand(0, 0.9), cand(1, 0.8)]);
check(folded.length === 1 && folded[0].doc === 0 && folded[0].alternates.length === 1,
      "two redundant renditions of one family collapse into one row");
folded = search.foldResults(fake, [cand(0, 0.9), cand(2, 0.8), cand(3, 0.7)]);
check(folded.length === 3, "a companion and an unknown rendition are never collapsed",
      `${folded.length} rows`);
folded = search.foldResults(fake, [cand(4, 0.9), cand(0, 0.8)]);
check(folded.length === 2, "a blank family never collapses, whatever the rendition");
const many = [0.9, 0.89, 0.88, 0.87, 0.86].map((c, i) => ({ chunk: i, doc: 4, page: 1, cosine: c }));
folded = search.foldResults(fake, many);
check(folded.length === 3 && folded[0].more === 2, "the per-document cap keeps three and counts the rest",
      `${folded.length} rows, more=${folded[0] && folded[0].more}`);
folded = search.foldResults(fake, [cand(4, 0.9), cand(0, 0.3)], 0.5);
check(folded.length === 1, "the cosine floor drops weak candidates");

console.log("grouping by document (the second view of the same answer)");
// The contract of the grouped view: same documents, same passages, one row a
// document, ranked by each document's CLOSEST passage. Checked on synthetic rows
// because the ordering only becomes interesting when a document's passages
// straddle another document's, which the corpus need not produce.
const mixed = [
  { chunk: 0, doc: 4, page: 1, cosine: 0.90, more: 2, alternates: [] },
  { chunk: 9, doc: 2, page: 1, cosine: 0.85, more: 0, alternates: [{ doc: 1, rendition: "argumentaire" }] },
  { chunk: 1, doc: 4, page: 2, cosine: 0.80, more: 1, alternates: [] },
  { chunk: 10, doc: 2, page: 3, cosine: 0.70, more: 0, alternates: [{ doc: 1, rendition: "argumentaire" }] },
];
const groups = search.groupByDocument(fake, mixed);
check(groups.length === 2, "one row a document", `${groups.length} rows`);
check(groups[0].doc === 4 && groups[1].doc === 2, "documents keep the order of their best passage");
check(groups[0].cosine === 0.90 && groups[1].cosine === 0.85, "a group scores as its closest passage");
check(groups[0].results.length === 2 && groups[1].results.length === 2,
      "every passage survives the regrouping");
check(groups.reduce((n, g) => n + g.results.length, 0) === mixed.length,
      "and none is invented or lost");
check(groups[0].more === 3, "the folded-away passage counts are summed per document",
      `${groups[0].more}`);
check(groups[1].alternates.length === 1, "and the alternate renditions are merged, not repeated");

console.log("browsing (the corpus listing, with no query in play)");
const browsed = search.browseDocs(index, {}, "fr");
check(browsed.length === meta.n_documents, "every document is listed", `${browsed.length}`);
check(new Set(browsed).size === browsed.length, "each exactly once");
const collator = new Intl.Collator("fr", { sensitivity: "base", numeric: true });
const titles = browsed.map((id) => index.docs[id].title || index.docs[id].file);
check(titles.every((tt, i) => i === 0 || collator.compare(titles[i - 1], tt) <= 0),
      "in alphabetical order, collated for French");
if (available.length) {
  const narrowed = search.browseDocs(index, { [facet.field]: [wanted] }, "fr");
  check(narrowed.length > 0 && narrowed.length < browsed.length,
        "and a facet narrows the listing the same way it narrows a search",
        `${narrowed.length} of ${browsed.length}`);
  check(narrowed.every((id) => (index.docs[id][facet.field] || "").trim() === wanted),
        "to documents that actually carry the chosen value");
}

// The query algebra (parsing, and what "|", "+()" and "-()" do to the vector) and
// the BM25 blend used to be checked here, on a stub encoder. Neither needs the real
// index, so both live in tests/search.test.mjs, which runs on every push whether or
// not dist/index/ is built.

// Real questions, real encoder, real wire format. Skipped without EMBED_URL so the
// gate stays runnable in the build chain with no service around.
//
// An EMBED_URL that is SET but not listening is also a skip rather than a failure,
// and deliberately so: .githooks/pre-push points this at the usual local encoder so
// the live queries run whenever it happens to be up, and a push must not be refused
// because a service on the developer's laptop is not running. A connection refused
// says nothing about the code being pushed. Everything above this line ran already
// and is what the gate actually asserts.
if (embedUrl && !(await reachable(embedUrl))) {
  console.log(`queries: skipped (nothing answering at ${embedUrl})`);
} else if (embedUrl) {
  console.log(`queries (live encoder at ${embedUrl})`);
  // The questions are the corpus's (scenarios.json search.queries). Without them,
  // a few document titles stand in: a title is text the corpus certainly answers,
  // which is all a live smoke test of the encoder-to-ranker path needs.
  const questions = loadScenarios().search?.queries
    || index.docs.map((d) => d.title).filter((t) => t && t.length > 12).slice(0, 4);
  for (const question of questions) {
    const query = await search.embedQuery(question, meta.dims);
    check(query.length === meta.dims, `encoder returns ${meta.dims} dims`, `${query.length} for "${question.slice(0, 30)}..."`);
    const found = await search.attachText(index, search.foldResults(index, search.rank(index, query, {})), "index");
    check(found.length > 0 && found.every((r) => r.cosine > -1 && r.cosine <= 1.001), "scores are in range");
    console.log(`  "${question}"`);
    for (const r of found.slice(0, 4)) {
      const doc = index.docs[r.doc];
      console.log(`      ${r.cosine.toFixed(3)}  p${r.page}  ${doc.title.slice(0, 62)}`);
      console.log(`             ${search.snippet(r.text || "", 110).replace(/\s+/g, " ")}`);
    }
  }

  // The service, which is what the deployed page actually asks, answers a known
  // question with the same top hit the ranker above gives when called directly.
  // The service wraps that same ranker in a queue, a parse, a fold at the floor
  // and a public projection of each row, and any of those could drop or reorder
  // the head of the list without the direct path noticing. Floor 0 so the cut
  // cannot hide the comparison; the top hit of the direct path is above any
  // floor a deployment would set anyway.
  //
  // `demoteReferences: true` and `favourRecent: true` because the two sides
  // default the OTHER WAY from each other: rank() defaults both false
  // (src/search.js), the service defaults both true (`!== false`, so the browser
  // opts out rather than in). Without them this compares two different
  // configurations and only passes when neither the penalty nor the bonus happens
  // to touch the head of the list, which is how it passed for a year and then
  // failed the day the encoder changed.
  const { createSearchService } = await import(`${root}/server/service.mjs`);
  const service = createSearchService({ search, index, floor: 0 });
  const question = questions[0];
  const direct = search.foldResults(
    index,
    search.rank(index, await search.embedQuery(question, meta.dims), {},
                { demoteReferences: true, favourRecent: true }),
    0)[0];
  const served = (await service.answer({ q: question, filters: {}, bm25: false })).results[0];
  check(served && direct && served.doc === direct.doc && served.chunk === direct.chunk,
        "the service answers a known query with the same top hit the direct rank gets",
        `service doc ${served?.doc} chunk ${served?.chunk} vs direct doc ${direct?.doc} chunk ${direct?.chunk}`);
  check(typeof served?.text === "string" && served.text.length > 0,
        "and the served row carries its text, so the page needs no second fetch to show it");

  // The recency bonus against the REAL corpus, which is the only place it can be
  // seen doing anything: a tie-breaker worth 0.05 of a cosine changes nothing on a
  // fixture where no two documents are close. Checked as a direction rather than a
  // threshold, because how much it moves depends on which years happen to answer
  // the sample questions, and a gate that asserts an amount would fail on a corpus
  // addition rather than on a bug.
  let moved = 0;
  let older = 0;
  for (const q of questions.slice(0, 5)) {
    const vector = await search.embedQuery(q, meta.dims);
    const plain = search.rank(index, vector, {}, { demoteReferences: true }).slice(0, 10);
    const recent = search.rank(index, vector, {}, { demoteReferences: true, favourRecent: true }).slice(0, 10);
    if (plain.some((hit, i) => hit.chunk !== recent[i]?.chunk)) moved++;
    if (medianYear(index, recent) < medianYear(index, plain)) older++;
  }
  check(moved > 0, "favouring recent documents reorders at least one sample query",
        `${moved} of 5 questions changed their first ten`);
  check(older === 0, "and never leaves a first ten older than the unbiased one",
        `${older} of 5 questions came back older`);
} else {
  console.log("queries: skipped (set EMBED_URL to a running embed-service.py to include them)");
}

// The one failure nothing else in this file can see: a service left running from an
// earlier session. Every check above loads src/search.js and dist/index/ in THIS
// process, so they all describe the files on disk. A service started before the
// last edit keeps answering with what it imported, holds port 8650 so the
// replacement dies on `Address already in use` in a log nobody reads, and agrees
// with these checks about nothing. It has cost four sessions: a 400 on a filter
// field the running build predated, a 404 on a route it did not have, and most
// recently a RESCORE_WEIGHT change that landed seven minutes after the service
// came up, where three of five sample queries rank differently between the old
// value and the new one, so the browser gates certified behaviour nobody had built.
//
// Opt-in via SVC because the service is optional here and a stopped one must not
// fail a push, the same rule EMBED_URL follows above.
if (serviceUrl) {
  const { RANKER_SHA, indexFingerprint } = await import(`${root}/server/load.mjs`);
  let health = null;
  try {
    const res = await preShimFetch(`${serviceUrl}/api/search/health`, { signal: AbortSignal.timeout(5000) });
    health = await res.json();
  } catch (error) {
    check(false, `the service at ${serviceUrl} answers /api/search/health`,
          String(error?.message || error));
  }
  if (health) {
    // The detail has to read correctly in BOTH directions, because `check` prints it
    // on a pass as well: the remedy is appended only when there is something to remedy.
    const stale = health.ranker_sha !== RANKER_SHA;
    check(!stale, `the service at ${serviceUrl} is running the CURRENT src/search.js`,
          `service ranker ${health.ranker_sha || "(absent: older than this check itself)"}, `
          + `this file ${RANKER_SHA}`
          + (stale ? ". Restart it: ss -ltnp | grep 8650, kill that pid, relaunch" : ""));
    // The same trap's other half: the ranker can be current while the index it
    // loaded is the previous build. n_chunks moved 69270 -> 128841 on 2026-09-26,
    // so this is not hypothetical, and it was the first version of this check.
    // n_chunks alone is too weak, though, and said "ok" through the case that
    // motivated it: on 2026-09-27 a title fix rebuilt the index, the vectors,
    // pages and refs matrices all took new content hashes, and the chunk count
    // did not move. The matrix names that replaced it missed a metadata-only
    // rebuild in turn, so indexFingerprint now hashes meta.json's bytes, which
    // cover both (see its docblock). n_chunks stays in the detail because it is
    // the part a reader can interpret without opening meta.json.
    const ourIndex = indexFingerprint(readFileSync(`${indexDir}/meta.json`));
    const wrongIndex = health.index_sha !== ourIndex;
    check(!wrongIndex, `and the index it holds is the one on disk`,
          `service index ${health.index_sha || "(absent: older than this check itself)"} `
          + `(${health.n_chunks} chunks), dist/index ${ourIndex} (${meta.n_chunks} chunks)`
          + (wrongIndex ? ". Restart it, as above" : ""));
  }
} else {
  console.log("service: skipped (set SVC=http://127.0.0.1:8650 to check a RUNNING service for staleness)");
}

console.log(failures ? `\n${failures} check(s) failed` : "\nall checks passed");
process.exit(failures ? 1 : 0);
