/* Whole-corpus semantic search, entirely in the browser.

   This is NOT retrieval-augmented generation. There is no language model in the
   serving path. One request leaves the page per query, to have the query text
   turned into a vector by the shared encoder; everything after that is a dot
   product against an index the browser already holds.

   Why the index is client-side at all: the VPS is small and is shared with
   a sibling site, so it cannot afford a per-request search process. At 1024
   dimensions and one bit each the whole corpus is about 3.5 MB, which is one
   image's worth of download, and then every subsequent query is free and private.

   Ranking arithmetic, BINARY index (the default, meta.quant === "binary"). Each
   passage keeps one bit per dimension: the sign of the component, packed eight to
   a byte, most significant bit first. The implied vector is +/-1 everywhere, so
   every passage shares one norm, sqrt(dims). The QUERY is not quantised down to
   match: it arrives as int8 at the fixed scale of 127 and stays that way, which is
   what makes this worth doing. Scoring is therefore sum over dimensions of
   +/-query[d], and the divisor that turns it back into a cosine is
   127 * sqrt(dims).

   Scoring a passage costs dims/8 lookups rather than dims multiplications, thanks
   to a table built once per query: for each of the dims/8 byte positions, the sum
   of +/-query over the eight dimensions that byte encodes, for all 256 possible
   byte values. 128 KB of table, built in about 33k additions, and then the scan of
   27k passages touches 128 bytes each. The wider index is the faster one to scan.

   Ranking arithmetic, INT8 index (meta.quant === "int8/fixed127"). Passage vectors
   are int8 at the same fixed scale of 127 as the query. Both sides are unit-norm
   before quantisation, so the raw dot product is 127 * 127 * cosine and dividing
   by 16129 recovers a cosine without ever dequantising a passage.

   A passage is also scored by its CONTEXT: the page it sits on, the page before
   that, and the section (heading run) it belongs to. The index ships two much
   smaller matrices, one vector per page and one per section, each chunk row
   names its page's row and its section's row in them, and the final score is

       (1 - p - q - s) * cosine(query, chunk) + p * cosine(query, page)
                       + q * cosine(query, previous page) + s * cosine(query, section)

   with the three weights read from meta.json. The context dot products are
   computed once per query, one per page (20,349 today) and one per section,
   against every passage (65,813), so the blend costs three lookups and four
   multiplications per passage. A weight of zero skips its matrix entirely: an
   index built without sections ships no section file and fetches none.

   The page term is a quantisation repair, not a semantic idea, and
   build_index.py's docstring records the measurement: it is worth 0.09 page@5 on
   the one-bit index and almost nothing at int8, because a page vector averages
   four times as much text and its SIGNS are what a one-bit index keeps. The two
   other terms are measured in DESIGN.md ("Section-level retrieval"). A chunk
   whose page, previous page or section has no vector keeps its own score for
   that term, which is the same as saying the context agrees with it exactly, so
   it is neither promoted nor punished for the gap.

   The two paths differ ONLY in how one passage is scored. Everything downstream,
   the bounded candidate list, the floor, folding, is shared, and meta.json decides
   which path runs, so the two can never disagree about the file they read. */

import { t } from "./i18n.js";
import { okJson } from "./paths.js";

/** Fixed quantisation scale, matching build_index.py's INT8_SCALE. */
const SCALE = 127;
/** Index format this client knows how to read. */
const SUPPORTED_FORMAT = 5;
/**
 * The chunk row value meaning "this chunk's page has no vector", matching
 * build_index.py's NO_PAGE_ROW. A real row can never reach it: that would take
 * 65,535 pages, eight times the corpus. The section slot uses the same value
 * for the same meaning.
 */
export const NO_PAGE_ROW = 0xFFFF;
/**
 * uint16 values per chunk in the locations file: doc, page, page vector row,
 * section vector row. Exported for the gate, which reads the file by this stride.
 */
export const LOC_STRIDE = 4;
/** meta.quant value selecting the one-bit-per-dimension path. */
const BINARY_QUANT = "binary";
/**
 * For each of the 256 byte values with exactly one bit set, which of the eight
 * dimensions that bit is. Most significant bit first, matching numpy.packbits's
 * default in scripts/build_index.py: get this backwards and every score is a dot
 * product against a bit-reversed vector, which looks like noise, not like a bug.
 */
const BIT_DIM = new Uint8Array(256);
for (let j = 0; j < 8; j++) BIT_DIM[1 << (7 - j)] = j;
/** Candidates kept before collapsing. Generous, because collapsing and the
    per-document cap both discard, and a family of four renditions can eat four
    slots for one answer. */
const CANDIDATES = 300;
/**
 * How much of a chunk's reference score is taken off its cosine when the reader
 * asks for citation pages to be pushed down.
 *
 * Sized against the distribution the build writes, measured on this corpus
 * (build_index.py logs it): the median chunk scores 0.098 on the axis and the most
 * reference-like scores 0.608, with p90 at 0.363. At 0.30 a bibliography loses
 * about 0.18 of cosine and a paragraph of prose loses 0.03, which is the gap
 * between them: enough to move a citation list off the first screen, not enough to
 * bury a real passage that happens to cite its sources.
 *
 * A penalty rather than a filter on purpose. The axis is mined from a heuristic and
 * it is wrong sometimes; a wrong penalty costs a few ranks and stays visible, while
 * a wrong filter removes the answer and leaves nothing on screen to explain why.
 */
const REFERENCE_PENALTY = 0.30;
/**
 * How much a document's AGE may move its passages, as a fraction of a cosine.
 *
 * A guideline is not a paper: the 2023 edition is meant to replace the 2012 one,
 * and a reader asking what to prescribe wants the current answer even when the
 * withdrawn one is worded closer to their question. The embedding cannot know
 * that, because both documents say the same thing in the same vocabulary, so the
 * year has to enter the score from outside it.
 *
 * A BONUS on the score, not a sort key and not a filter, for the reason the
 * reference penalty is a penalty: a tie-breaker stays recoverable. Sorting by year
 * would put every 2026 leaflet above the 2011 argumentaire that answers the
 * question, and filtering by year would remove the older document with nothing on
 * screen to say it existed. Here the old document still wins whenever it is
 * enough better, and the reader can see it did.
 *
 * MEASURED by scripts/sweep_recency.mjs, which is a COST measurement and not a
 * benefit one: the eval queries carry a (file, page) ground truth written by
 * reading the passage, with no preference for recent documents in it, so a
 * recency bias can only lose ground there. What the sweep answers is how much.
 *
 * Over the 200 queries of data/EVAL_QUERIES.tsv, at a half-life of 8 years
 * (data/EVAL_RECENCY.tsv holds the whole grid, standard error 0.035):
 *
 *     bonus   page@1   page@5   MRR      median year of the top 10
 *     0       0.1150   0.2850   0.1943   2017.0
 *     0.02    0.1100   0.2850   0.1939   2017.7
 *     0.05    0.1250   0.2850   0.2020   2018.8
 *     0.08    0.1150   0.2900   0.1966   2019.7
 *     0.12    0.1250   0.2550   0.1905   2020.3
 *
 * 0.05 is where the visible effect is worth having (the median document in a
 * first screen of results moves forward by nearly two years) and no metric has
 * gone below the unbiased row. Everything up to it is inside the noise, so the
 * table cannot say it HELPS; what it can say is where it starts to hurt, and that
 * is above 0.05: page@5 falls at 0.12, and it falls first at the short half-lives
 * (0.2300 at 0.12/4y), which is the signature of a bias that has stopped breaking
 * ties and started overruling the embedding. The benefit is not in that file at
 * all and is taken on the editorial argument above.
 */
export const RECENCY_BONUS = 0.05;
/**
 * How many years halve that bonus.
 *
 * Counted from the NEWEST year in the index rather than from today's date, so the
 * ranking of a built index never changes with the wall clock: an index shipped in
 * 2026 would otherwise start demoting its own newest documents in 2030 without
 * anything being rebuilt. It also means the bonus says what it is worth: the
 * newest document in the corpus gets all of RECENCY_BONUS.
 *
 * 8 years is roughly two review cycles for an agency that revisits its guidance
 * on about a five-year horizon, and the first corpus measured spanned 1943 to 2026
 * with a median of 2017, so a median document keeps about 45% of the
 * bonus and anything from the 1990s keeps essentially none. The sweep tried 4, 8
 * and 16 and could not separate them on the metrics, which is expected: the
 * half-life decides the SHAPE of the ordering inside the bonus and the eval file
 * has no opinion about that. 8 is the editorial answer, not the measured one.
 */
export const RECENCY_HALFLIFE = 8;
/** Results shown. */
const SHOWN = 40;
/**
 * Most separate passages shown from one document before the rest are folded into
 * a "N more passages in this document" line.
 *
 * This is the mitigation for the long-document dominance problem: ten documents
 * are half the corpus, and an argumentaire of 2,573 chunks will otherwise fill a
 * result list with itself. DESIGN.md lists three candidate fixes (per-document
 * capping, length-normalised scoring, grouping by document); this is the first,
 * chosen because it is the only one that cannot change which document ranks
 * first, and being wrong about relevance is worse than being repetitive.
 */
const PER_DOC = 3;
/**
 * How many candidates the lexical rescoring looks at, before folding.
 *
 * Before folding, because folding decides which document in a family stands for
 * the family, and that decision should see the rescored order. The depth is what
 * it costs: rescoring needs the chunk TEXT, which arrives per document, so a
 * deeper shortlist means fetching more of `doc/<id>.json`. 50 covers the
 * candidates a reader can realistically reach and usually spans a handful of
 * documents that the result list was about to fetch anyway.
 */
export const RESCORE_DEPTH = 50;
/**
 * How much the lexical score may move a candidate, as a fraction of a cosine.
 *
 * MEASURED, not chosen: see scripts/evaluate_rescore.mjs and DESIGN.md. It was
 * 0.08 while a chunk was 1024 tokens, where the curve was flat from 0.04 to 0.12
 * and turning rescoring off cost 0.008 of MRR. The move to 256 tokens changed
 * both numbers: over the 200 eval queries, turning it off now costs 0.0985 of MRR
 * and the plateau is 0.10 to 0.30, because BM25 is scored over the candidate's own
 * text and a quarter of the terms makes a shared word a quarter as likely to be
 * shared by accident. The metrics disagree about where in that plateau to sit
 * (MRR and page@1 say 0.20, page@5 says 0.12, page@10 says 0.10), so the shipped
 * value is the middle of it. Past 0.30 it falls off, because a passage that merely
 * repeats the query's words out-ranks the one that answers it. The embedding is
 * still the primary signal and BM25 still a tiebreaker over a shortlist it already
 * agrees is relevant. 0 disables rescoring entirely.
 */
export const RESCORE_WEIGHT = 0.15;
/**
 * How far a language group's lexical scale may fall below the shortlist's best.
 *
 * MEASURED, and it fixes a bug in the idea rather than tuning it. BM25 is divided
 * by the best BM25 in the shortlist, and the shortlist for a French question is 77%
 * French documents, which are the ones a French question shares words with. So that
 * best belongs to a same-language candidate in nearly every shortlist, and the
 * English passage that answers the question is scored against a scale it cannot
 * reach: 0.137 against 0.517, measured over the eval set's crosslingual group. The
 * lexical term was not a tiebreaker on that axis, it was a vote for the reader's
 * own language, and RESCORE_WEIGHT was its volume.
 *
 * The scale is therefore per language: a candidate is divided by the best BM25
 * among the candidates written in ITS language, so the term asks "the most
 * on-point passage among those in this language" instead of "the most on-point
 * passage". This floor is what keeps that from overcorrecting, because the best of
 * a weak group would otherwise be handed a full 1.0: the divisor is the group's own
 * best or this fraction of the shortlist's best, whichever is larger.
 *
 * 0.5 is Pareto-dominant over the single global scale on the 200 eval queries: the
 * same page@1 and page@5, page@10 a point better, and the crosslingual group's MRR
 * doubled from 0.0611 to 0.1234 with its page@1 off zero for the first time. 1
 * restores the single global scale exactly; 0 is full per-language normalisation,
 * which buys another 0.049 of crosslingual MRR and costs 0.025 of overall page@1,
 * a trade rather than a gain. DESIGN.md, "BM25 was a same-language preference".
 */
export const LEXICAL_GROUP_FLOOR = 0.5;
/** Standard BM25 saturation and length-normalisation constants. */
const BM25_K1 = 1.2;
const BM25_B = 0.75;
/**
 * Most texts one question may be split into, positives and negatives together.
 *
 * The encoder answers one text per request, so a compound query costs the SHARED
 * encoder one embedding per term. `docker/Caddyfile` rate-limits /api/search per
 * request, not per term, so without this cap one counted request with thirty
 * alternatives in it would cost the encoder thirty. Refusing is better than silently dropping terms: a
 * query that quietly ignored half of what was asked would rank plausibly and be
 * wrong in a way nothing on screen could show.
 */
const MAX_QUERY_TERMS = 10;

/**
 * Most characters one question may run to, across every term it names.
 *
 * A question is a sentence. This cap is about ten times a long one, so nobody
 * writing in good faith will meet it, and it exists for the other case: pasting a
 * chapter into the box, or arriving at `?query=` with one in the URL, sends that
 * text to an encoder shared with justelesRCP, where a long input costs quadratic
 * attention and real RAM on a small VPS. Caddy caps the request body and the
 * service caps its own input, but both of those refuse AFTER the request has been
 * made: refusing here costs nothing and can say why in the reader's language.
 *
 * Counted over the parsed terms rather than the raw string, so the punctuation of
 * the query syntax is not what puts a question over the line.
 */
const MAX_QUERY_CHARS = 512;

/**
 * Fewest characters one term may run to.
 *
 * The shared encoder refuses a shorter one with a 400 (`EMBED_MIN_QUERY_CHARS`,
 * 5 by default in justelesRCP's service), and it is right to: five characters
 * are not a question, and this index answers questions rather than keywords. The
 * same floor here turns that refusal into a sentence in the reader's language,
 * before a request is made, which is the whole difference between "error 400"
 * and being told that "test" is too short to mean anything to the model.
 *
 * The encoder's floor is a deployment variable and this one is a copy, so the
 * 400 path below stays and reads the real number back out of the refusal.
 */
const MIN_QUERY_CHARS = 5;

/**
 * Load the index.
 * @param {string} base URL prefix, normally "index".
 * @param {object} [options]
 * @param {boolean} [options.vectors=true] fetch the matrices. False loads meta.json
 *   alone, which is everything a page that lists documents needs and none of what
 *   only ranking needs.
 * @returns {Promise<object>} meta, the passage matrix, the page matrix and the
 *   per-chunk locations.
 */
export async function loadIndex(base = "index", { vectors: wantVectors = true } = {}) {
  // meta.json first, alone: it names the vector file. From format 2 that name is
  // the builder's to choose (layout, and later a content hash), so hardcoding it
  // here is exactly the coupling this avoids. The cost is one serial round trip
  // for a 13 KB gzipped file before the big fetch starts.
  const meta = await fetch(`${base}/meta.json`).then(okJson);
  if (meta.format_version !== SUPPORTED_FORMAT) {
    // Refuse rather than guess. A layout change that this client misreads would
    // rank confidently and wrongly, which is far worse than an error message.
    throw new Error(`index format ${meta.format_version}, expected ${SUPPORTED_FORMAT}`);
  }
  const binary = meta.quant === BINARY_QUANT;
  const redundant = new Set(meta.renditions?.redundant || []);
  // browse.html lists documents and cannot rank, so the three big files are 20 MB
  // it would pay for and never score. meta.json carries the facets, the titles and
  // the rendition sets, which is the whole of what listing and filtering read.
  if (!wantVectors) {
    return { meta, vectors: null, locations: null, pages: null, pageWeight: 0,
             sections: null, sectionWeight: 0, prevPageWeight: 0, prevRows: null,
             refs: null, refScale: 0,
             redundant, binary, stride: 0, docs: meta.documents, textCache: new Map() };
  }
  const name = meta.vectors_file;
  const locName = meta.chunks_file;
  // The page half is optional: an index built with --page-weight 0 names no file
  // and this stays a pure passage search, with no second fetch and no blend. The
  // previous-page term reads the same file, so it needs the pages but no file of
  // its own, and either weight is reason enough to fetch them.
  const pagesName = meta.pages_file || "";
  const pageWeight = pagesName ? (meta.page_weight || 0) : 0;
  const prevPageWeight = pagesName ? (meta.prev_page_weight || 0) : 0;
  const sectionsName = meta.sections_file || "";
  const sectionWeight = sectionsName ? (meta.section_weight || 0) : 0;
  // Format 5. One byte per chunk, so it is fetched whenever it exists rather than
  // only when the penalty is on: the reader ticks the box per query, and refetching
  // 45 KB on a tick would make a checkbox cost a round trip.
  const refsName = meta.refs_file || "";
  if (!name || !locName) throw new Error("meta.json does not name its data files");
  // In parallel: the big eager files. Every name carries a content hash, which
  // is what makes the caching safe: docker/Caddyfile serves them immutable for a
  // year, so a repeat visit re-reads them from disk with no request at all, and a
  // rebuilt index cannot be served out of cache because its names changed. The
  // pairing matters as much as the freshness, since meta.json carries the chunk
  // offsets that index INTO these files.
  const [vectorsBuf, locBuf, pagesBuf, sectionsBuf, refsBuf] = await Promise.all([
    fetch(`${base}/${name}`).then(okBuf),
    fetch(`${base}/${locName}`).then(okBuf),
    pageWeight > 0 || prevPageWeight > 0 ? fetch(`${base}/${pagesName}`).then(okBuf) : null,
    sectionWeight > 0 ? fetch(`${base}/${sectionsName}`).then(okBuf) : null,
    refsName ? fetch(`${base}/${refsName}`).then(okBuf) : null,
  ]);
  // Unsigned for the binary path (the byte IS the table index, 0..255) and signed
  // for int8 (the byte IS a component). Same buffer, and reading it through the
  // wrong view is the one mistake here that would silently rank nonsense.
  const vectors = binary ? new Uint8Array(vectorsBuf) : new Int8Array(vectorsBuf);
  const stride = binary ? meta.dims >> 3 : meta.dims;
  if (binary && meta.dims % 8) throw new Error(`binary index at ${meta.dims} dims is not byte-aligned`);
  if (vectors.length !== meta.n_chunks * stride) {
    throw new Error(`${name} holds ${vectors.length} bytes, expected ${meta.n_chunks * stride}`);
  }
  // Little-endian explicitly at the writer; Uint16Array follows the platform, and
  // every platform this runs on is little-endian. Checked rather than assumed:
  // a big-endian reader would see page numbers in the thousands.
  const locations = new Uint16Array(locBuf);
  if (locations.length !== meta.n_chunks * LOC_STRIDE) {
    throw new Error(`${locName} holds ${locations.length} values, expected ${meta.n_chunks * LOC_STRIDE}`);
  }
  // Same view rules as the passage matrix, same reason: reading packed bits as
  // signed bytes ranks nonsense instead of failing.
  const pages = pagesBuf ? (binary ? new Uint8Array(pagesBuf) : new Int8Array(pagesBuf)) : null;
  if (pages && pages.length !== meta.n_pages * stride) {
    throw new Error(`${pagesName} holds ${pages.length} bytes, expected ${meta.n_pages * stride}`);
  }
  const sections = sectionsBuf ? (binary ? new Uint8Array(sectionsBuf) : new Int8Array(sectionsBuf)) : null;
  if (sections && sections.length !== meta.n_sections * stride) {
    throw new Error(`${sectionsName} holds ${sections.length} bytes, expected ${meta.n_sections * stride}`);
  }
  // Unsigned, and one byte per chunk rather than one per dimension: this is a
  // score, not a vector. Reading it as signed would turn every strong reference
  // score into a negative one and the penalty into a reward.
  const refs = refsBuf ? new Uint8Array(refsBuf) : null;
  if (refs && refs.length !== meta.n_chunks) {
    throw new Error(`${refsName} holds ${refs.length} scores, expected ${meta.n_chunks}`);
  }
  // Built whenever pages ship, not only when the weight is on: it is a 120k-entry
  // array, and having it lets a measurement turn the term on without a rebuild.
  const prevRows = pages ? previousPageRows(locations, meta.n_chunks) : null;
  return { meta, vectors, locations, pages, pageWeight, sections, sectionWeight,
           prevPageWeight, prevRows, refs, refScale: meta.ref_scale || 0,
           redundant, binary, stride,
           docs: meta.documents, textCache: new Map() };
}

/**
 * Every page's page-vector row, keyed by `pageKey(document, page)`.
 *
 * No file carries this: each chunk's location names its own page and that
 * page's row, so one pass over the locations collects them. A page with no
 * chunk of its own (a full-page figure, an empty page) is absent.
 *
 * @param {Uint16Array} locations the locations file, `LOC_STRIDE` per chunk.
 * @param {number} nChunks
 * @returns {Map<number, number>}
 */
export function pageRowMap(locations, nChunks) {
  const rowOf = new Map();
  for (let c = 0; c < nChunks; c++) {
    const row = locations[c * LOC_STRIDE + 2];
    if (row === NO_PAGE_ROW) continue;
    rowOf.set(pageKey(locations[c * LOC_STRIDE], locations[c * LOC_STRIDE + 1]), row);
  }
  return rowOf;
}

/** The `pageRowMap` key of a 1-based page of a document. */
export function pageKey(doc, page) {
  // 65,536 pages per document is the same bound the uint16 page number has.
  return doc * 65536 + page;
}

/**
 * For every chunk, the page-vector row of the page BEFORE its own.
 *
 * `pageRowMap` collects (document, page) to row, and one pass looks each
 * chunk's predecessor up. Two guards, both deliberate. Page 1 has no
 * predecessor. And a page with no chunks of its own (a full-page figure, an
 * empty page) is absent from the map, in which case the chunk gets no
 * previous-page term rather than a row borrowed from whatever page came before
 * that, which could be another chapter.
 *
 * @param {Uint16Array} locations the locations file, `LOC_STRIDE` per chunk.
 * @param {number} nChunks
 * @returns {Uint16Array} one row per chunk, `NO_PAGE_ROW` where there is none.
 */
export function previousPageRows(locations, nChunks) {
  const rowOf = pageRowMap(locations, nChunks);
  const out = new Uint16Array(nChunks).fill(NO_PAGE_ROW);
  for (let c = 0; c < nChunks; c++) {
    const page = locations[c * LOC_STRIDE + 1];
    if (page <= 1) continue;
    const row = rowOf.get(pageKey(locations[c * LOC_STRIDE], page - 1));
    if (row !== undefined) out[c] = row;
  }
  return out;
}

async function okBuf(r) {
  if (!r.ok) throw new Error(`${r.status} ${r.url.split("/").pop()}`);
  return r.arrayBuffer();
}

/* One page-lifetime cache of question text to its vector.
 *
 * The encoder is asked once per distinct text, not once per search. That is not a
 * micro-optimisation: a reader who narrows a filter, flips between the passage and
 * document views, walks back through history, or asks the same question twice runs
 * the SAME query text through `run()` again, and each of those was a round trip to a
 * shared service that Caddy rate-limits to 120 events a minute per IP. A single
 * question with the "|" operator costs several texts, so a handful of filter tweaks
 * could reach that limit and start failing.
 *
 * Deliberately in memory only, and deliberately not sessionStorage. The question is
 * the one string this site promises never to store: it is kept out of analytics on
 * purpose (see site.js), so writing it to disk to save a request would trade the
 * promise for latency. Closing the tab forgets everything, which is the right
 * lifetime for it.
 *
 * Keyed on width as well as text, because the same question at 256 and at 1024 dims
 * is two different vectors and the index width is a deployment variable. Capped so a
 * long session cannot grow it without bound; oldest entry out, which for a search box
 * is close enough to least-recently-used. */
const QUERY_CACHE_MAX = 64;
const queryCache = new Map();

/** Empty the query cache. Exists for the gates, which stub the encoder per text. */
export function clearQueryCache() { queryCache.clear(); }

/**
 * Turn a question into a vector, using the shared encoder behind /api/sem/embed.
 * @param {string} question
 * @param {number} dims the index's width; a mismatch throws rather than ranking.
 * @returns {Promise<Int8Array>}
 */
export async function embedQuery(question, dims) {
  const key = `${dims}\u0000${question}`;
  const hit = queryCache.get(key);
  // A copy on the way out. The caller owns its vector: embedParsedQuery averages and
  // renormalises in place, so handing back the cached array would corrupt the cache
  // the first time a question was used inside a "|" group.
  if (hit) return Int8Array.from(hit);
  const vector = await fetchQueryVector(question, dims);
  queryCache.set(key, vector);
  if (queryCache.size > QUERY_CACHE_MAX) queryCache.delete(queryCache.keys().next().value);
  return Int8Array.from(vector);
}

/** The uncached round trip. Split out so the cache above reads as a cache. */
async function fetchQueryVector(question, dims) {
  const response = await fetch("api/sem/embed", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // `dim` asks the SHARED encoder for the width this index was baked at,
    // instead of accepting whatever width the sibling project configured for
    // its own catalog. Without it the two are coupled: justelesRCP serves 256
    // because that is what its ~11k baked pages are, and this index could only
    // ever be 256 too. The encoder truncates one cached full-width vector per
    // query (MRL), so asking costs it nothing and re-embeds nothing.
    //
    // Sent unconditionally rather than only on a mismatch, so the request says
    // out loud what the page expects. An encoder too old to know the field
    // ignores it and answers at its own width, which the check below then
    // catches as a mismatch rather than letting it rank at the wrong width.
    body: JSON.stringify({ q: question, dim: dims }),
  });
  if (!response.ok) {
    // A 400 that names a character range is the encoder's own length floor,
    // which can be configured above the copy in MIN_QUERY_CHARS. Read the number
    // back out of the refusal so the page says the deployment's floor rather
    // than its own guess. Any other 400 is a malformed request, which is a bug
    // here rather than something the reader can act on, so it keeps its status.
    if (response.status === 400) {
      const detail = await response.json().catch(() => null);
      const range = /(\d+)\s*-\s*\d+\s*chars/.exec(detail?.error || "");
      if (range) {
        const err = new Error("short");
        err.min = Number(range[1]);
        throw err;
      }
    }
    // 502 is the expected shape of "the shared encoder is not attached to this
    // site's network", which is a deployment state, not a bug in the page.
    throw new Error(response.status === 502 ? "unavailable" : `embed ${response.status}`);
  }
  const payload = await response.json();
  if (payload.dim !== dims) {
    const err = new Error("dim");
    err.got = payload.dim;
    err.want = dims;
    throw err;
  }
  // base64 -> bytes -> signed. atob gives one char per byte; the Int8Array view
  // reinterprets the same buffer as signed, which is exactly the round trip of
  // the server's numpy int8 -> base64.
  const raw = atob(payload.q);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return new Int8Array(bytes.buffer);
}

/**
 * Split one question into the texts to embed, positive and negative.
 *
 * The syntax, in full:
 *
 *   `qualité de l'eau potable`             one positive text, the ordinary case
 *   `loyer | location`                     `|` separates alternatives to average
 *   `(loyer | location)`                   the same, parenthesised
 *   `loyer +(location)`                    the same again, as an added group
 *   `eau potable -(zone rurale)`           subtract what the parenthesis is about
 *   `(loyer | location) -(bureau | commerce)` both sides, each with alternatives
 *
 * There is one idea here and three spellings of it: every text the reader gives
 * on the positive side is embedded on its own and the results are averaged, so
 * `A | B`, `(A | B)` and `A +(B)` all produce the same two texts. The spellings
 * exist because readers reach for different ones, and a syntax that accepts only
 * the author's favourite reads as a bug.
 *
 * Only `-(` and `+(` open a group, so a hyphen or a plus inside ordinary prose
 * ("émissions sous-estimées", "bus + tramway") is left alone and a
 * question with no punctuation at all parses to exactly one positive text, which
 * is what every query written before this syntax existed does.
 *
 * Parentheses are structure, not content: they are stripped from the text handed
 * to the encoder, because "(sevrage)" and "sevrage" must embed identically.
 *
 * An unclosed `-(` or `+(` swallows the rest of the string, which is the reading
 * that matches what someone mid-typing means. It is also why parsing never
 * throws: the box is parsed on every submit, including while the reader is still
 * typing a group, and an error message at that moment would be noise.
 *
 * @param {string} text the raw contents of the search box.
 * @returns {{positive: string[], negative: string[], lexical: string}} the texts
 *   to embed on each side, and the positive text alone for the BM25 rescoring:
 *   a word the reader asked to subtract must never boost a passage lexically.
 */
export function parseQuery(text) {
  const groups = { "+": [], "-": [] };
  let rest = "";
  for (let i = 0; i < text.length; i++) {
    const sign = text[i];
    if ((sign === "-" || sign === "+") && text[i + 1] === "(") {
      let depth = 1;
      let j = i + 2;
      for (; j < text.length && depth > 0; j++) {
        if (text[j] === "(") depth++;
        else if (text[j] === ")") depth--;
      }
      // depth > 0 means the group was never closed: take the rest of the string.
      groups[sign].push(text.slice(i + 2, depth > 0 ? text.length : j - 1));
      i = j - 1;
      continue;
    }
    rest += sign;
  }
  const split = (part) => part
    .split("|")
    .map((piece) => piece.replace(/[()]/g, " ").replace(/\s+/g, " ").trim())
    .filter(Boolean);
  // What is left outside the groups first, in the order it was written: it is the
  // question, and an added group is a second way of saying "also this".
  const positives = split(rest).concat(groups["+"].flatMap(split));
  return {
    positive: positives,
    negative: groups["-"].flatMap(split),
    lexical: positives.join(" "),
  };
}

/**
 * Unit-length float mean of several int8 query vectors.
 *
 * Averaging is what `|` means here: two alternatives become the point between
 * them, which retrieves what they have in common more strongly than what only
 * one of them says. That is a real choice and not the only one (scoring each
 * alternative separately and taking the best score per passage is the other),
 * and it is the one the request asked for, in its words, "sum queries
 * embeddings".
 *
 * @param {Int8Array[]} vectors at least one, all the same width.
 * @returns {Float64Array} unit norm, or all zeros if the mean cancels out.
 */
function meanUnit(vectors) {
  const dims = vectors[0].length;
  const out = new Float64Array(dims);
  for (const vector of vectors) {
    for (let d = 0; d < dims; d++) out[d] += vector[d];
  }
  let norm = 0;
  for (let d = 0; d < dims; d++) norm += out[d] * out[d];
  norm = Math.sqrt(norm);
  if (!norm) return out;
  for (let d = 0; d < dims; d++) out[d] /= norm;
  return out;
}

/**
 * Refuse a parsed query the encoder would refuse, before any request is made.
 *
 * Shared by `embedParsedQuery` (the service, the gates) and the page, which calls
 * it on its own before `searchRemote` so that a malformed or too-short question
 * costs no round trip and is told about in the same words either way.
 *
 * @param {{positive: string[], negative: string[]}} parsed from `parseQuery`.
 * @throws {Error} "empty" with nothing to embed, "long" over MAX_QUERY_CHARS,
 *   "terms" over MAX_QUERY_TERMS, "short" under MIN_QUERY_CHARS.
 */
export function checkParsedQuery(parsed) {
  const texts = parsed.positive.concat(parsed.negative);
  if (!parsed.positive.length) throw new Error("empty");
  const length = texts.reduce((total, text) => total + text.length, 0);
  if (length > MAX_QUERY_CHARS) {
    // Checked before the term count, because a single pasted chapter is one term
    // and would otherwise pass every check here on its way to the encoder.
    const err = new Error("long");
    err.got = length;
    err.max = MAX_QUERY_CHARS;
    throw err;
  }
  if (texts.length > MAX_QUERY_TERMS) {
    // Carried on the error the way embedQuery carries the widths, so the page can
    // say how many terms were written and how many are allowed without either
    // number being duplicated into the message here.
    const err = new Error("terms");
    err.got = texts.length;
    err.max = MAX_QUERY_TERMS;
    throw err;
  }
  // Every term is embedded on its own, so one short term is enough to earn the
  // encoder's 400: check the shortest rather than the total. Last of the three,
  // because a query that is too long or names too many terms has a problem with
  // its shape, and being told that one of its twelve terms is also short would
  // send the reader after the wrong thing. Positive terms only: "-(ado)" is a
  // subtraction rather than a question, and there is no reason to hold it to the
  // length of one.
  const shortest = Math.min(...parsed.positive.map((textOfTerm) => textOfTerm.length));
  if (shortest < MIN_QUERY_CHARS) {
    const err = new Error("short");
    err.min = MIN_QUERY_CHARS;
    throw err;
  }
}

/**
 * Embed a parsed query, subtracting the negative side from the positive one.
 *
 * The subtraction is a PROJECTION removal, `q - (q·n)n`, not a plain `q - n`:
 *
 *   It has no weight to tune. Plain subtraction needs one, because at weight 1
 *   a negative term close to the question cancels it and the query lands on the
 *   far side of the negative, ranking passages for being the OPPOSITE of what
 *   was subtracted rather than for answering the question.
 *
 *   It says exactly what the reader asked for: rank by the part of my question
 *   that is not about this. What remains is orthogonal to the negative term, so
 *   a passage about it scores zero from that direction rather than negatively.
 *
 * The residual is renormalised, because `rank` expects a query on the fixed
 * scale 127 and the two scoring paths divide by it to recover a cosine.
 *
 * @param {{positive: string[], negative: string[]}} parsed from `parseQuery`.
 * @param {number} dims the index's width.
 * @returns {Promise<Int8Array>} a query vector `rank` can take.
 * @throws {Error} "empty" with nothing to embed, "long" over MAX_QUERY_CHARS,
 *   "terms" over MAX_QUERY_TERMS, "short" under MIN_QUERY_CHARS, and whatever `embedQuery` throws (an unreachable
 *   encoder, a width mismatch).
 */
export async function embedParsedQuery(parsed, dims) {
  checkParsedQuery(parsed);
  const texts = parsed.positive.concat(parsed.negative);
  // In parallel: they are independent, and the encoder answers a query on the
  // request thread rather than queueing it, so N at once costs about one.
  const vectors = await Promise.all(texts.map((text) => embedQuery(text, dims)));
  const positive = meanUnit(vectors.slice(0, parsed.positive.length));
  if (parsed.negative.length) {
    const negative = meanUnit(vectors.slice(parsed.positive.length));
    let dot = 0;
    for (let d = 0; d < dims; d++) dot += positive[d] * negative[d];
    for (let d = 0; d < dims; d++) positive[d] -= dot * negative[d];
    let norm = 0;
    for (let d = 0; d < dims; d++) norm += positive[d] * positive[d];
    norm = Math.sqrt(norm);
    // Only when the negative is the question restated: everything the reader
    // asked for has been subtracted away and the residual is rounding error,
    // which would rank as noise rather than as nothing.
    if (norm < 1e-3) throw new Error("cancelled");
    for (let d = 0; d < dims; d++) positive[d] /= norm;
  }
  const out = new Int8Array(dims);
  for (let d = 0; d < dims; d++) {
    // Clamped because rounding 127 * 1.0 can land on 128, which wraps to -128
    // in an Int8Array and silently flips that dimension's sign.
    out[d] = Math.max(-SCALE, Math.min(SCALE, Math.round(positive[d] * SCALE)));
  }
  return out;
}

/**
 * Ask the search service for the answer to one question.
 *
 * The page does not rank: the service (server/service.mjs) holds the index and
 * runs `rank`, `rescore` and `foldResults` on it, so the browser downloads no
 * matrix at all. What comes back is the folded list the page used to compute
 * itself, with each result's text attached, plus the floor it was cut at and the
 * width it was ranked at.
 *
 * Refusals keep the names the page already translates: a 400 carries "short"
 * (with `min`), "terms" or "long" (with `got` and `max`) or "empty"; a 500 with
 * "dim" carries `got` and `want`; a 502 is "unavailable"; a 503 is "busy", the
 * queue being full, and worth asking again in a moment. A network failure is
 * "unavailable" too: to the reader the two are the same thing.
 *
 * @param {string} question the raw search box, parsed again server-side.
 * @param {object} filters the page's facet selections, as `rank` takes them.
 * @param {object} [options]
 * @param {boolean} [options.bm25=false] whether to rescore the head of the list lexically.
 * @param {boolean} [options.demoteRefs=true] whether to push bibliographies down the list.
 * @param {boolean} [options.recent=true] whether to favour recent documents.
 * @param {string} [options.figures] one of FIGURE_MODES: described figures competing
 *   with the text (the default), left out, or alone.
 * @param {string} [options.base] the endpoint, relative like every other fetch here.
 * @returns {Promise<{results: object[], floor: number, dims: number}>}
 */
export async function searchRemote(question, filters, { bm25 = false, demoteRefs = true, recent = true,
  figures = FIGURE_MODES[0], base = "api/search" } = {}) {
  let response;
  try {
    response = await fetch(base, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        q: question,
        filters,
        bm25: Boolean(bm25),
        demote_references: demoteRefs !== false,
        favour_recent: recent !== false,
        figures,
      }),
    });
  } catch {
    throw new Error("unavailable");
  }
  if (response.ok) return response.json();
  const detail = await response.json().catch(() => ({}));
  if (response.status === 503) throw new Error("busy");
  if (response.status === 502) throw new Error("unavailable");
  if ((response.status === 400 || response.status === 500) && detail.error) {
    const err = new Error(detail.error);
    for (const key of ["min", "got", "max", "want"]) if (detail[key] != null) err[key] = detail[key];
    throw err;
  }
  throw new Error(`search ${response.status}`);
}

/**
 * The individual values held by one metadata cell.
 *
 * `doc_type` and `topic` are lists: a document can be a recommandation AND its own
 * synthese, and is always about several topics at once. Every other field holds a
 * single value, and splitting it is a no-op because no single value contains the
 * separator. So there is one accessor rather than a per-field branch, and adding a
 * third multi-valued column needs no change here at all.
 *
 * The separator comes from meta.json, which carries it from manifest.py's
 * VALUE_SEPARATOR. Hardcoding ";" here would be a second copy that drifts in the
 * worst direction: the client would stop splitting and each multi-valued cell would
 * become one dropdown entry matching exactly one document.
 *
 * @param {object} index
 * @param {object} doc  a document record from meta.json
 * @param {string} field
 * @returns {string[]} the trimmed, non-empty values; `[]` for a blank cell
 */
export function cellValues(index, doc, field) {
  const raw = doc[field];
  if (!raw) return [];
  const sep = (index.meta && index.meta.separator) || ";";
  return raw.split(sep).map((piece) => piece.trim()).filter(Boolean);
}

/**
 * Is a facet selection actually narrowing anything?
 *
 * Three shapes reach here, and `v && v.length` was true for only two of them: a
 * plain string, an array of strings, and `{from, to}` for a numeric range, which
 * has no length and would have been silently dropped.
 */
export function isActive(wanted) {
  if (!wanted) return false;
  if (Array.isArray(wanted)) return wanted.length > 0;
  if (typeof wanted === "object") return wanted.from != null || wanted.to != null;
  return String(wanted).length > 0;
}

/**
 * Does one document's cell satisfy one selection?
 *
 * Matching is ANY, not ALL: a document tagged `tsa;adulte;diagnostic` appears
 * under each of those three topics. ALL would mean the reader has to reproduce the
 * curator's exact tag set to see anything, which is not a filter, it is a password.
 */
function matchesFilter(values, wanted) {
  if (wanted && typeof wanted === "object" && !Array.isArray(wanted)) {
    // A numeric range, used by the year slider. The first value is the whole cell:
    // year is single-valued, and a document with two publication years would be a
    // manifest bug, not a range to widen.
    const n = Number(values[0]);
    if (!Number.isFinite(n)) return false;
    return (wanted.from == null || n >= wanted.from) && (wanted.to == null || n <= wanted.to);
  }
  if (Array.isArray(wanted)) return values.some((v) => wanted.includes(v));
  return values.includes(wanted);
}

/**
 * The one GRADED column: an ordered vocabulary where choosing a value means "this
 * or narrower", rather than a set of independent tags where choosing one means
 * exactly it. Which manifest column that is belongs to the corpus (corpus.toml,
 * [tiers] column), so it is read from `meta.tier_field`; an index built before that
 * key existed held its tier in `guideline`, hence the fallback.
 *
 * It gets its own name here because it breaks both of the rules the facets above
 * follow, and a reader of `allowedDocs` should be told which two:
 *
 *   1. Selecting a tier also keeps every narrower one, because the tiers are
 *      nested. A facet match is equality; this one is a position in
 *      `meta.guideline_tiers`, which the corpus's configuration owns and
 *      `build_index.py` ships narrowest first.
 *   2. A BLANK cell passes, where a blank cell fails every facet. A corpus being
 *      curated has many documents with no tier yet, so the facet rule would hide
 *      them from a reader who never touched a filter. Blank means "nobody has
 *      judged this document", which is not the same answer as the widest tier,
 *      and only the second one is a reason to hide it.
 *
 * @param {object} index
 * @returns {string}
 */
export function tierField(index) {
  return (index && index.meta && index.meta.tier_field) || "guideline";
}

/**
 * The manifest columns offered as filters, in panel order, from meta.json. Empty
 * for an index built before the corpus declared them, which offers no facets
 * rather than guessing at columns it may not have.
 *
 * @param {object} index
 * @returns {string[]}
 */
export function facetFields(index) {
  return (index && index.meta && index.meta.facet_fields) || [];
}

/**
 * The facet fields shown as a two-handle numeric range rather than a dropdown.
 *
 * @param {object} index
 * @returns {Set<string>}
 */
export function rangeFields(index) {
  return new Set((index && index.meta && index.meta.range_fields) || []);
}

/**
 * Does one document pass the guideline level?
 *
 * @param {object} index
 * @param {object} doc a document record from meta.json
 * @param {string} level a tier from `index.meta.guideline_tiers`
 * @returns {boolean} true when the document is at that tier or narrower, and true
 *   for anything this cannot place: a blank cell, an index built before the tiers
 *   existed, or a level naming a tier that is not in the list. Every one of those
 *   is a reason to show the document and let the reader judge it, never a reason to
 *   remove it from a search with nothing on screen to explain the absence.
 */
export function matchesGuideline(index, doc, level) {
  const tiers = (index.meta && index.meta.guideline_tiers) || null;
  if (!tiers || !tiers.length) return true;
  const cut = tiers.indexOf(level);
  if (cut < 0) return true;
  const value = (doc[tierField(index)] || "").trim();
  if (!value) return true;
  const rank = tiers.indexOf(value);
  return rank >= 0 && rank <= cut;
}

/**
 * Which documents pass the current facet selections.
 * @returns {Set<number>|null} null when nothing is filtered, which lets the
 *   scan skip the membership test entirely.
 */
export function allowedDocs(index, filters) {
  const active = Object.entries(filters).filter(([, v]) => isActive(v));
  if (!active.length) return null;
  const allowed = new Set();
  const tier = tierField(index);
  index.docs.forEach((doc, id) => {
    const ok = active.every(([field, wanted]) => {
      if (field === tier) return matchesGuideline(index, doc, String(wanted));
      const values = cellValues(index, doc, field);
      if (!values.length) return false;  // a blank cell cannot satisfy a chosen filter
      return matchesFilter(values, wanted);
    });
    if (ok) allowed.add(id);
  });
  return allowed;
}

/**
 * Build the per-query lookup table the binary path scores with.
 *
 * `table[(b << 8) + v]` is what a passage byte `v` at position `b` contributes:
 * the sum over that byte's eight dimensions of +query[d] where the bit is set and
 * -query[d] where it is clear. So scoring one passage is `dims / 8` lookups.
 *
 * Built by flipping one bit at a time rather than summing eight terms per entry:
 * setting a bit turns -query[d] into +query[d], a delta of 2*query[d], so each of
 * the 32,768 entries costs one addition off an entry already computed.
 *
 * @param {Int8Array} query int8 query vector at the index's width.
 * @param {number} dims
 * @returns {Float32Array} dims/8 * 256 entries, about 128 KB at 1024 dims.
 */
function binaryScoreTable(query, dims) {
  const nbytes = dims >> 3;
  const table = new Float32Array(nbytes << 8);
  for (let b = 0; b < nbytes; b++) {
    const off = b << 8;
    const dim0 = b << 3;
    // Byte 0: every bit clear, so every one of the eight dimensions contributes
    // -query[d]. Every other entry is reached from here.
    let zero = 0;
    for (let j = 0; j < 8; j++) zero -= query[dim0 + j];
    table[off] = zero;
    for (let v = 1; v < 256; v++) {
      const low = v & -v;                       // lowest set bit
      table[off + v] = table[off + (v ^ low)] + 2 * query[dim0 + BIT_DIM[low]];
    }
  }
  return table;
}

/**
 * The stored vector of one chunk, in the shape `rank` takes as a query.
 *
 * Only the gate (scripts/check_search.mjs) needs this, and it needs it badly: its
 * strongest check is that a passage's own vector retrieves that passage, which is
 * what catches a stride, offset or bit-order mistake, and on a binary index the
 * stored row is packed bits rather than a query vector.
 *
 * Unpacking to +/-round(127/sqrt(dims)) keeps the result on the query's own fixed
 * scale, so the self-match scores a cosine of about 1 like it does on an int8
 * index, instead of one scaled by sqrt(dims).
 *
 * @returns {Int8Array} `meta.dims` components.
 */
export function chunkVector(index, chunk) {
  return storedVector(index, index.vectors, chunk);
}

/**
 * The stored vector of one PAGE, in the shape `rank` takes as a query.
 *
 * The gate uses it to prove the two matrices are aligned: a chunk's own page
 * must be its nearest page, which an off-by-one document in the page rows
 * destroys immediately while leaving every length check happy.
 *
 * @returns {Int8Array} `meta.dims` components, or null on an index with no pages.
 */
export function pageVector(index, page) {
  if (!index.pages) return null;
  return storedVector(index, index.pages, page);
}

/**
 * The stored vector of one SECTION, for the same alignment check on the
 * section matrix.
 * @returns {Int8Array} `meta.dims` components, or null on an index with no sections.
 */
export function sectionVector(index, section) {
  if (!index.sections) return null;
  return storedVector(index, index.sections, section);
}

/** Shared by chunkVector, pageVector and sectionVector: the matrices use one layout. */
function storedVector(index, matrix, row) {
  const { meta, binary, stride } = index;
  const base = row * stride;
  if (!binary) return matrix.slice(base, base + stride);
  const dims = meta.dims;
  const unit = Math.round(SCALE / Math.sqrt(dims)) || 1;
  const out = new Int8Array(dims);
  for (let b = 0; b < stride; b++) {
    const byte = matrix[base + b];
    for (let j = 0; j < 8; j++) out[(b << 3) + j] = (byte & (1 << (7 - j))) ? unit : -unit;
  }
  return out;
}

/**
 * One recency bonus per document, in whatever unit `bonus` is given in.
 *
 * `bonus * 0.5 ** (age / halfLife)`, where age counts from the newest year the
 * index carries. Exponential rather than linear over the corpus range because the
 * range is 83 years long and almost none of it matters: linear would make the gap
 * between 2024 and 2023 (one year of a modern guideline's life) the same as the
 * gap between 1944 and 1943, and would let one 1943 paper redefine the scale for
 * all 533 other documents.
 *
 * A document with no year gets the MEDIAN year's bonus, which is the only value
 * that neither rewards nor punishes a blank cell against the corpus it sits in.
 * Two of the 535 documents are in that case, so this is a correctness detail
 * rather than a lever; what makes it worth naming is that the obvious two
 * alternatives are both wrong in a way nobody would notice: treating a blank as
 * year 0 buries the document forever, and treating it as the newest year makes a
 * missing cell the best thing that can happen to a document.
 *
 * @param {object[]} docs `index.docs`, each with a `year` cell (a string, as the
 *   manifest writes it, possibly empty).
 * @param {{bonus: number, halfLife: number}} params
 * @returns {Float64Array|null} one entry per document, or null when the bonus is
 *   zero or the corpus carries no year at all.
 */
export function recencyBonuses(docs, { bonus, halfLife }) {
  if (!(bonus > 0) || !(halfLife > 0) || !docs || !docs.length) return null;
  const years = new Array(docs.length);
  const known = [];
  for (let i = 0; i < docs.length; i++) {
    const y = Number(String(docs[i].year ?? "").trim());
    years[i] = Number.isFinite(y) && y > 0 ? y : null;
    if (years[i] !== null) known.push(years[i]);
  }
  if (!known.length) return null;
  known.sort((a, b) => a - b);
  const newest = known[known.length - 1];
  const median = known[known.length >> 1];
  const out = new Float64Array(docs.length);
  for (let i = 0; i < docs.length; i++) {
    const year = years[i] === null ? median : years[i];
    // Future-dated documents exist (a 2026 guideline in a 2026 corpus is the
    // newest, but so is a mis-parsed 2031): clamping keeps age at or above zero
    // so a bad year cannot buy more than the bonus.
    const age = Math.max(0, newest - year);
    out[i] = bonus * Math.pow(0.5, age / halfLife);
  }
  return out;
}

/** What the figures filter accepts; the first is what an omitted one means. */
export const FIGURE_MODES = ["include", "exclude", "only"];

/**
 * The rows of one document that the figures filter lets `rank()` score.
 *
 * @param {{chunk_offset:number, chunk_count:number, figure_count?:number}} doc
 * @param {string} figures - One of FIGURE_MODES.
 * @returns {[number, number]} start (inclusive) and end (exclusive) rows.
 */
export function figureRange(doc, figures) {
  const end = doc.chunk_offset + doc.chunk_count;
  const firstFigure = end - (doc.figure_count || 0);
  if (figures === "exclude") return [doc.chunk_offset, firstFigure];
  if (figures === "only") return [firstFigure, end];
  return [doc.chunk_offset, end];
}

/**
 * Score every eligible chunk and keep the best `CANDIDATES`.
 * @returns {Array<{chunk:number, doc:number, page:number, cosine:number}>}
 */
export function rank(index, query, filters, { demoteReferences = false, favourRecent = false,
  figures = "include" } = {}) {
  const { vectors, locations, pages, sections, prevRows, refs, meta, binary, stride } = index;
  // Read per call rather than destructured once at load, so that a measurement
  // can set them on the index object and re-rank without rebuilding anything.
  const pageWeight = index.pageWeight || 0;
  const prevPageWeight = pages && prevRows ? (index.prevPageWeight || 0) : 0;
  const sectionWeight = sections ? (index.sectionWeight || 0) : 0;
  const dims = meta.dims;
  const allowed = allowedDocs(index, filters);
  // The two scoring paths differ in their raw unit, so the divisor that turns a
  // raw score into a cosine differs too, and the candidate floor is kept in raw
  // units to avoid dividing once per chunk. Binary: query is int8 at scale 127
  // against a +/-1 passage of norm sqrt(dims). int8: 127 * 127 * cosine.
  const table = binary ? binaryScoreTable(query, dims) : null;
  const perCosine = binary ? SCALE * Math.sqrt(dims) : SCALE * SCALE;
  // The reference penalty, folded into one multiplier so the inner loop does a
  // single multiply-subtract on a byte rather than three divisions per chunk.
  // `refs[c]` is the cosine to the mined reference axis, stored as 0..255 over
  // `refScale`; the penalty is REFERENCE_PENALTY of that cosine, converted to the
  // raw units the rest of this loop works in so the candidate floor keeps meaning
  // the same thing. Zero when the index ships no axis or the reader left the box
  // unticked, and a zero multiplier is cheaper to keep than a branch per chunk.
  const refPenalty = demoteReferences && refs
    ? REFERENCE_PENALTY * (index.refScale || 0) / 255 * perCosine : 0;
  // The recency bonus, per DOCUMENT rather than per chunk, so it is read once for
  // a 2,573-chunk argumentaire instead of 2,573 times. Converted to raw units here
  // for the same reason the reference penalty is: the inner loop never divides.
  // Both knobs are read off the index when it carries them, so a sweep can set
  // them and re-rank without rebuilding anything (scripts/sweep_recency.mjs).
  const recencyBonus = favourRecent
    ? (index.recencyBonus != null ? index.recencyBonus : RECENCY_BONUS) : 0;
  const halfLife = index.recencyHalfLife != null ? index.recencyHalfLife : RECENCY_HALFLIFE;
  const recency = recencyBonuses(index.docs, { bonus: recencyBonus * perCosine, halfLife });
  // Every page and every section is scored once, here, rather than once per
  // chunk that sits on it: one row a page instead of one lookup a chunk for the
  // same answer. Both matrices are stored in the passage file's format, so the
  // same table scores all three, and the raw scores share the perCosine divisor,
  // which is what lets the blend happen in raw units and keeps the candidate
  // floor a single comparison.
  const pageScores = (pageWeight > 0 || prevPageWeight > 0) && pages
    ? scoreMatrix(pages, meta.n_pages, stride, dims, table, query) : null;
  const sectionScores = sectionWeight > 0 && sections
    ? scoreMatrix(sections, meta.n_sections, stride, dims, table, query) : null;
  const chunkWeight = 1 - pageWeight - prevPageWeight - sectionWeight;

  // Iterating documents rather than chunks lets a filtered-out document be
  // skipped in one test instead of once per chunk, which matters when a filter
  // excludes an argumentaire of 2,573 chunks.
  const best = [];        // ascending by cosine, at most CANDIDATES entries
  let floor = -Infinity;

  for (let docId = 0; docId < index.docs.length; docId++) {
    if (allowed && !allowed.has(docId)) continue;
    const doc = index.docs[docId];
    // A document's described figures are the LAST `figure_count` rows of its
    // range (scripts/build_index.py refuses anything else), so the figures
    // filter is a cut in the range rather than a test per chunk.
    const [start, end] = figureRange(doc, figures);
    // Constant for every chunk of this document, and added inside the loop below
    // rather than to the floor, because the floor is one number shared by all
    // documents and this is not.
    const ageBonus = recency ? recency[docId] : 0;

    for (let c = start; c < end; c++) {
      let dot = 0;
      const base = c * stride;
      // Plain loops on purpose: no allocation, and it is a few milliseconds for
      // 27k passages either way. If this ever becomes visible on slow hardware,
      // the fix is a Web Worker, not a cleverer inner loop.
      if (table) {
        for (let b = 0; b < stride; b++) dot += table[(b << 8) + vectors[base + b]];
      } else {
        for (let d = 0; d < dims; d++) dot += vectors[base + d] * query[d];
      }
      if (pageScores || sectionScores) {
        // A chunk whose page (or section, or previous page) has no vector keeps
        // its own score for that term: that is the blend with the missing term
        // set equal to the chunk term, so a missing row neither promotes nor
        // punishes it. Blending BEFORE the floor test is what makes the floor
        // correct, since a page can lift a passage over it.
        const own = dot;
        let page = own, prev = own, section = own;
        if (pageScores) {
          const row = locations[c * LOC_STRIDE + 2];
          if (row !== NO_PAGE_ROW) page = pageScores[row];
          const before = prevRows ? prevRows[c] : NO_PAGE_ROW;
          if (before !== NO_PAGE_ROW) prev = pageScores[before];
        }
        if (sectionScores) {
          const row = locations[c * LOC_STRIDE + 3];
          if (row !== NO_PAGE_ROW) section = sectionScores[row];
        }
        dot = chunkWeight * own + pageWeight * page + prevPageWeight * prev + sectionWeight * section;
      }
      // After the blend and BEFORE the floor, for the same reason the blend is:
      // the floor is the score of the worst candidate kept, so a chunk has to be
      // compared against it at the score it will actually be ranked with. Demoting
      // after the cut would let a bibliography hold a place it no longer deserves
      // and push a real passage out of the list to do it.
      if (refPenalty) dot -= refPenalty * refs[c];
      // Before the floor test, like the penalty above and for the same reason: the
      // floor is the score of the worst candidate kept, so a chunk has to meet it
      // at the score it will be ranked with. A recent document's passage that the
      // bonus lifts over the cut is exactly the case this feature exists for.
      dot += ageBonus;
      if (dot <= floor) continue;
      // `lang` is carried for `rescore`, which normalises the lexical score within
      // a language and cannot see `index.docs`. It is read off the document in
      // scope here, so it costs a property on the few hundred entries that clear
      // the floor rather than a lookup per chunk.
      const entry = { chunk: c, doc: docId, page: locations[c * LOC_STRIDE + 1],
                      cosine: dot / perCosine, lang: doc.language || "" };
      // Binary insertion into a bounded, sorted array. Cheaper than sorting 27k
      // entries, and the bound is what keeps memory flat.
      let lo = 0, hi = best.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (best[mid].cosine < entry.cosine) lo = mid + 1; else hi = mid;
      }
      best.splice(lo, 0, entry);
      if (best.length > CANDIDATES) {
        best.shift();
        floor = best[0].cosine * perCosine;
      }
    }
  }
  return best.reverse();
}

/** Raw score of every row of one stored matrix against the query. */
function scoreMatrix(matrix, rows, stride, dims, table, query) {
  const out = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    const base = r * stride;
    let dot = 0;
    if (table) {
      for (let b = 0; b < stride; b++) dot += table[(b << 8) + matrix[base + b]];
    } else {
      for (let d = 0; d < dims; d++) dot += matrix[base + d] * query[d];
    }
    out[r] = dot;
  }
  return out;
}

/**
 * Fold candidates into the list a reader sees.
 *
 * Two different groupings act here and they are not the same thing:
 *
 *  - **Family collapse** hides a document behind another. Two documents in one
 *    family whose renditions are both "redundant" (a summary, the full text, its
 *    background report and so on) are the same guidance at different lengths, so the
 *    best-ranking one stands and the others become "also published as". A
 *    rendition that is NOT in that set is never collapsed, even inside a family:
 *    a companion list of procedures enumerates items that appear nowhere in the
 *    guideline it accompanies, and hiding it would make it unreachable. The set comes
 *    from meta.json, which carries it from scripts/manifest.py, so this file
 *    holds no copy of the policy.
 *
 *  - **The per-document cap** hides passages behind other passages of the SAME
 *    document. Unrelated to families; it exists because long documents otherwise
 *    crowd out short ones.
 *
 * An unknown rendition name is treated as never-collapse, because the cost of the
 * two errors is asymmetric: showing one answer twice is untidy, hiding the only
 * document that answers the query is a broken search.
 */
export function foldResults(index, candidates, floorCosine = 0) {
  const results = [];
  const familyWinner = new Map();   // family slug -> result holding it
  const docFirst = new Map();       // document id -> its first result
  const docShown = new Map();       // document id -> how many of its passages shown

  for (const cand of candidates) {
    if (cand.cosine < floorCosine) continue;
    const doc = index.docs[cand.doc];
    const family = (doc.family || "").trim();
    const rendition = (doc.rendition || "").trim();
    const collapsible = family && index.redundant.has(rendition);

    if (collapsible) {
      const winner = familyWinner.get(family);
      if (winner && winner.doc !== cand.doc) {
        // Same answer, different length. Record the alternative once.
        if (!winner.alternates.some((a) => a.doc === cand.doc)) {
          winner.alternates.push({ doc: cand.doc, rendition });
        }
        continue;
      }
    }

    const shown = docShown.get(cand.doc) || 0;
    if (shown >= PER_DOC) {
      docFirst.get(cand.doc).more++;
      continue;
    }

    // `score` is what ORDERED this list: the cosine on its own, or the cosine plus
    // the lexical term when the caller rescored. It is carried explicitly rather
    // than left to `rescore` to have set, because a candidate below RESCORE_DEPTH
    // was never rescored and would otherwise reach the renderer without one, and a
    // row that displays a different number from the one it was sorted by reads as
    // a broken sort. `cosine` stays the pure vector number, which is what the floor
    // is calibrated on and what DESIGN.md's measurements are in.
    const result = { ...cand, score: cand.score ?? cand.cosine, more: 0, alternates: [], text: null };
    results.push(result);
    docShown.set(cand.doc, shown + 1);
    if (!docFirst.has(cand.doc)) docFirst.set(cand.doc, result);
    if (collapsible && !familyWinner.has(family)) familyWinner.set(family, result);
    if (results.length >= SHOWN) break;
  }
  return results;
}

/**
 * Regroup folded results by document, best document first.
 *
 * A presentation of the SAME results the passage list shows, not a second ranking:
 * the toggle in the UI must not change which documents are in the answer, only how
 * they are stacked. So this takes the folded list and buckets it, rather than going
 * back to the candidates.
 *
 * A document's rank is its best passage's score, the same number that ordered the
 * passage list, which is what "sorted by the distance to the closest chunk" means. DESIGN.md lists grouping by document as one
 * of three candidate fixes for long-document dominance and rejects it as a fix,
 * because it would change which document ranks first; offering it as a VIEW is the
 * other half of that decision, and costs nothing, because the per-document cap has
 * already thinned each document to at most PER_DOC passages.
 *
 * `more` and `alternates` are summed and merged across the document's passages, so
 * the group can say "12 more passages in this document" once instead of three times.
 *
 * @param {object} index
 * @param {object[]} results  output of foldResults, descending by score
 * @returns {object[]} one entry a document: {doc, score, cosine, results, more, alternates}
 */
export function groupByDocument(index, results) {
  const groups = [];
  const byDoc = new Map();
  for (const result of results) {
    let group = byDoc.get(result.doc);
    if (!group) {
      group = { doc: result.doc, score: result.score, cosine: result.cosine,
                results: [], more: 0, alternates: [] };
      byDoc.set(result.doc, group);
      groups.push(group);
    }
    group.results.push(result);
    // Both, and for the same passage: `score` is what the group is ranked and
    // labelled by, `cosine` is kept beside it so anything reading the vector
    // number off a group still gets the vector number.
    if (result.score > group.score) {
      group.score = result.score;
      group.cosine = result.cosine;
    }
    group.more += result.more || 0;
    for (const alt of result.alternates || []) {
      if (!group.alternates.some((a) => a.doc === alt.doc)) group.alternates.push(alt);
    }
  }
  // Sorted explicitly rather than trusting the input order. It happens to be
  // descending already, but "the document with the closest chunk comes first" is the
  // contract of this view, and it should not depend on a caller's sort surviving.
  groups.sort((a, b) => b.score - a.score);
  return groups;
}

/**
 * The documents a reader may browse, with no query in play.
 *
 * Ordered by title rather than by score, because there is no score: this is the
 * corpus list, and its job is "find the document I already have in mind". Honours
 * the same facet filters the search does, through the same allowedDocs, so
 * narrowing the filters narrows the list a reader is looking at.
 *
 * Collator rather than a bare string compare: these titles are French, and a plain
 * comparison sorts "Épisode" after "Zonage".
 *
 * @param {object} index
 * @param {object} filters   facet field -> chosen value
 * @param {string} locale    BCP 47 tag driving the collation
 * @returns {number[]} document ids
 */
export function browseDocs(index, filters, locale = "fr", name = "") {
  const allowed = allowedDocs(index, filters);
  // Every word of the filter has to appear SOMEWHERE in the title or the filename,
  // as a substring and in any order: a reader typing "logem" while still typing is
  // narrowing the list, and one typing "insee logement" is naming two things they
  // remember about the document rather than quoting its title. Substring rather
  // than token equality is the whole difference from the lexical rescoring in
  // rescore(), which answers a different question.
  const terms = fold(name).split(/\s+/).filter(Boolean);
  const named = (id) => {
    if (!terms.length) return true;
    const doc = index.docs[id];
    const hay = fold(`${doc.title || ""} ${doc.file || ""}`);
    return terms.every((term) => hay.includes(term));
  };
  const ids = index.docs.map((_, id) => id)
    .filter((id) => (!allowed || allowed.has(id)) && named(id));
  const collator = new Intl.Collator(locale, { sensitivity: "base", numeric: true });
  const label = (id) => index.docs[id].title || index.docs[id].file;
  ids.sort((a, b) => collator.compare(label(a), label(b)));
  return ids;
}

/**
 * Fetch the chunk text and boxes for the documents present in `results`.
 * Lazy on purpose: the corpus's chunk text is 33 MB of UTF-8 and a reader sees
 * forty chunks, so it is fetched per document, on demand, and cached.
 */
export async function attachText(index, results, base = "index") {
  // Read through a map of this call's OWN, filled before the await for what is
  // already cached and after it for what was fetched. Reading the shared cache
  // after the await instead lost results under load in the search service: a
  // concurrent request finishing during the fetch trims the cache to its bound
  // (server/service.mjs, boundTextCache), oldest first, which are exactly the
  // entries this call found cached and so did not fetch. Up to 50 documents a
  // request against a 64-entry cache made that two or three requests away, and
  // the results it hit lost their text and boxes and were rescored on "".
  const docs = [...new Set(results.map((r) => r.doc))];
  const mine = new Map();
  for (const d of docs) if (index.textCache.has(d)) mine.set(d, index.textCache.get(d));
  await Promise.all(docs.filter((d) => !mine.has(d)).map(async (docId) => {
    let chunks = null;
    try {
      // Keyed by the chunk's own index within the document, which is what a
      // result's global row minus the document's offset gives.
      chunks = (await fetch(`${base}/doc/${docId}.json`).then(okJson)).chunks;
    } catch {
      // A missing per-document file costs a snippet, not the result. The reader
      // can still open the page, which is the part that matters.
    }
    mine.set(docId, chunks);
    index.textCache.set(docId, chunks);
  }));
  for (const r of results) {
    const chunks = mine.get(r.doc);
    if (!chunks) continue;
    const local = r.chunk - index.docs[r.doc].chunk_offset;
    const chunk = chunks[local];
    if (chunk) {
      r.text = chunk.text;
      r.pages = chunk.pages;
      r.boxes = chunk.boxes;
      if (chunk.figure) r.figure = chunk.figure;
    }
  }
  return results;
}

/**
 * Lowercase and strip diacritics, so French text compares the way it is typed.
 *
 * Its own function because two different things need the same folding and had to
 * agree about it: `tokenise` below, which feeds BM25, and the document-name filter
 * on the browse page, which matches substrings rather than whole tokens and so
 * cannot go through `tokenise` at all.
 *
 * @param {string} text
 * @returns {string} Lowercased, with combining accents removed.
 */
export function fold(text) {
  return (text || "").toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "");
}

/**
 * Split text into comparable terms.
 *
 * Lowercase, strip diacritics, split on anything that is not a letter or a digit.
 * Diacritic folding is not optional in this corpus: a reader types "depistage"
 * and the guideline says "depistage" with an accent, and an exact-token match
 * that misses on the accent is worse than no lexical signal at all.
 *
 * Digits are kept, because a dose is a term ("300 mg", "ISO-9001", "NF-C15"), and
 * one-character tokens are dropped, because they are punctuation debris after the
 * split rather than words.
 *
 * No stemming and no stopword list. Stemming French and English with one rule set
 * is a project of its own, and stopwords are handled by IDF: a term that appears
 * in every shortlisted passage scores near zero on its own.
 *
 * @param {string} text
 * @returns {string[]}
 */
export function tokenise(text) {
  return fold(text)
    .split(/[^a-z0-9]+/)
    .filter((w) => w.length > 1 && w.length < 40);
}

/**
 * Reorder a shortlist by blending the dense score with BM25 over the chunk text.
 *
 * Why this exists: the embedding is good at meaning and indifferent to exact
 * words, so a query naming a specific molecule, scale or dose can rank a passage
 * about the general topic above the passage that names it. BM25 is the opposite,
 * which is why the two are combined rather than chosen between.
 *
 * The lexical corpus is the SHORTLIST, not the whole index. That is what makes
 * this affordable in a browser (no term statistics are shipped, and the text of
 * these documents is being fetched for display anyway), and it is also defensible:
 * IDF over the shortlist measures what distinguishes these candidates from each
 * other, which is exactly the question at rank time.
 *
 * The blend is `cosine + weight * bm25 / scale`, where `scale` is the best BM25
 * among the candidates in this candidate's own language, floored at
 * LEXICAL_GROUP_FLOOR of the best in the whole shortlist. Normalising keeps the term
 * bounded and scale-free, since a raw BM25 score has no ceiling and depends on query
 * length; normalising PER LANGUAGE is what stops it from being a vote for the
 * reader's language, which is measured and explained at that constant. `cosine` is
 * left untouched, so the relevance floor still means what it meant and so a rescored
 * list can be compared against the dense one.
 *
 * Each candidate's language is read from its `lang`, which `rank` copies off the
 * document. A candidate without one joins the same group as every other candidate
 * without one, which is why a caller that builds results by hand gets exactly the
 * single-scale behaviour.
 *
 * Results whose text has not been attached score 0 lexically rather than being
 * dropped: a missing `doc/<id>.json` costs a tiebreak, not a result.
 *
 * @param {Array<object>} results Candidates carrying `.cosine` and, ideally, `.text`.
 * @param {string} question The reader's query.
 * @param {number} [weight] Override for RESCORE_WEIGHT, 0 to disable.
 * @param {number} [floor] Override for LEXICAL_GROUP_FLOOR. Only the sweeps pass it,
 *   for the same reason they pass `weight`: a tuning parameter has to be measurable
 *   through the shipped ranker rather than by editing this file.
 * @returns {Array<object>} The same array, sorted by the blended score.
 */
export function rescore(results, question, weight = RESCORE_WEIGHT, floor = LEXICAL_GROUP_FLOOR) {
  if (!weight || results.length < 2) return results;
  const terms = [...new Set(tokenise(question))];
  if (!terms.length) return results;

  const docs = results.map((r) => tokenise(r.text || ""));
  const lengths = docs.map((d) => d.length);
  const avgLen = lengths.reduce((a, b) => a + b, 0) / (lengths.length || 1);
  // Term frequencies per candidate, and document frequency across the shortlist.
  const freqs = docs.map((tokens) => {
    const f = new Map();
    for (const tok of tokens) f.set(tok, (f.get(tok) || 0) + 1);
    return f;
  });
  const df = new Map();
  for (const f of freqs) for (const tok of f.keys()) if (terms.includes(tok)) df.set(tok, (df.get(tok) || 0) + 1);

  const n = results.length;
  const scores = results.map((_, i) => {
    let sum = 0;
    for (const term of terms) {
      const tf = freqs[i].get(term) || 0;
      if (!tf) continue;
      const seen = df.get(term) || 0;
      // Probabilistic IDF with the +1 that keeps a term present in every
      // candidate at a small positive weight instead of a negative one.
      const idf = Math.log(1 + (n - seen + 0.5) / (seen + 0.5));
      const norm = 1 - BM25_B + BM25_B * (lengths[i] / (avgLen || 1));
      sum += idf * (tf * (BM25_K1 + 1)) / (tf + BM25_K1 * norm);
    }
    return sum;
  });

  // The scale is per language, floored at LEXICAL_GROUP_FLOOR of the shortlist's
  // best: see that constant for why one global scale made the term a language vote.
  // A candidate with no language, and a shortlist where every candidate shares one,
  // both reduce to the global scale, so nothing changes where there is nothing to
  // separate.
  const max = Math.max(...scores) || 1;
  const bestByLang = new Map();
  results.forEach((r, i) => {
    const lang = r.lang || "";
    bestByLang.set(lang, Math.max(bestByLang.get(lang) || 0, scores[i]));
  });
  results.forEach((r, i) => {
    // No clamp needed: the divisor is at least this candidate's own group best,
    // which is at least its own score.
    const scale = Math.max(bestByLang.get(r.lang || "") || 0, floor * max) || 1;
    r.lexical = scores[i] / scale;
    r.score = r.cosine + weight * r.lexical;
  });
  // Stable enough: ties keep their dense order because sort() is stable in every
  // engine this runs on, and a tie means BM25 had nothing to say.
  return results.sort((a, b) => b.score - a.score);
}

/**
 * Build the facet options actually present in the corpus.
 *
 * Options are the individual values, not the raw cells: a document tagged
 * `recommandation;synthese` contributes to both options rather than creating a
 * third one that reads like a separate kind of document.
 *
 * @returns {{field: string, values: string[]}[]} one entry per field worth a control
 */
export function facets(index, fields) {
  const out = [];
  for (const field of fields) {
    const values = new Set();
    for (const doc of index.docs) {
      for (const v of cellValues(index, doc, field)) values.add(v);
    }
    // A column nobody has curated yet produces no control at all, rather than an
    // empty dropdown. This is what lets the filter panel grow as MANIFEST.tsv is
    // filled in, with no change here: `family` and `rendition` are set on the
    // documents that have a sibling and blank on the rest, so neither offers a
    // control yet.
    if (values.size > 1) out.push({ field, values: [...values].sort(collate) });
  }
  return out;
}

/** Locale-aware compare, so "Émetteur" values sort where a French reader expects. */
const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });
function collate(a, b) { return collator.compare(a, b); }

/** A short, readable label for a page span. */
export function pageLabel(result) {
  const pages = result.pages;
  if (!pages || pages.length <= 1) return t("page", { n: result.page });
  return t("pages", { a: pages[0], b: pages[pages.length - 1] });
}

/**
 * Tidy a chunk for display without flattening it.
 *
 * Runs of spaces and tabs collapse, because a PDF's column padding is not text,
 * but newlines survive: the chunker puts one in front of every row that starts a
 * paragraph or a list item (chunk.py, `Row.starts_block`) and every row whose
 * predecessor ended its line on purpose (`Row.starts_line`), so a list of
 * recommendations and a table of drug names keep their shape instead of reading as
 * one long sentence. Two or more blank lines collapse to one, so a page break
 * inside a chunk does not open a hole in it.
 */
function tidy(text) {
  return (text || "")
    .replace(/[^\S\n]+/g, " ")
    .replace(/ *\n */g, "\n")
    .replace(/\n{2,}/g, "\n")
    .trim();
}

/**
 * Fold text for matching while keeping a map back into the original.
 *
 * `fold` normalises and drops diacritics, which changes the LENGTH of the string
 * ("é" is one character and folds to one, but through NFD it is two), so offsets
 * found in the folded text do not address the text a reader is shown. Folding one
 * character at a time and recording where each folded character came from costs
 * one pass and makes the two addressable together.
 *
 * @param {string} text
 * @returns {{folded: string, at: number[]}} `at[i]` is the index in `text` of the
 *   character that produced `folded[i]`, with `at[folded.length]` the end.
 */
function foldWithOffsets(text) {
  let folded = "";
  const at = [];
  for (let i = 0; i < text.length; i++) {
    const piece = fold(text[i]);
    for (let k = 0; k < piece.length; k++) {
      folded += piece[k];
      at.push(i);
    }
  }
  at.push(text.length);
  return { folded, at };
}

/** Whole-token occurrences of `terms` in already-folded text, in reading order. */
function termHits(folded, terms) {
  const hits = [];
  const wordy = /[a-z0-9]/;
  for (const term of terms) {
    for (let from = 0; ; ) {
      const i = folded.indexOf(term, from);
      if (i < 0) break;
      from = i + term.length;
      // A token, not a substring: "age" must not match inside "dosage", which is
      // the same rule the BM25 tokeniser applies by splitting on non-word runs.
      if (i > 0 && wordy.test(folded[i - 1])) continue;
      if (wordy.test(folded[from] || "")) continue;
      hits.push({ start: i, end: from, term });
    }
  }
  return hits.sort((a, b) => a.start - b.start);
}

/**
 * The window of `width` folded characters covering the most DISTINCT query terms.
 *
 * Distinct rather than most hits: a passage that says "logement" eight times
 * in one paragraph is not a better window than one that says it once beside
 * "loyer" and "zone rurale", and the reader is looking for the place the question
 * comes together. Ties go to the earliest window, which is the one a reader
 * scanning from the top would have found anyway.
 *
 * @returns {{start: number, end: number}|null} folded offsets, or null when no
 *   term appears at all.
 */
function densestWindow(hits, width) {
  if (!hits.length) return null;
  let best = null;
  for (let i = 0; i < hits.length; i++) {
    const terms = new Set();
    let last = i;
    for (let j = i; j < hits.length && hits[j].end - hits[i].start <= width; j++) {
      terms.add(hits[j].term);
      last = j;
    }
    if (!best || terms.size > best.terms) {
      best = { terms: terms.size, start: hits[i].start, end: hits[last].end };
    }
  }
  return best;
}

/**
 * Trim a chunk to a snippet, centred on the question when there is one.
 *
 * Without a question this is the head of the chunk, which is what it always was.
 * With one, the window is the densest run of query terms in the passage, padded
 * out to `max` characters and snapped to word boundaries, with an ellipsis on
 * whichever side was cut. The head was fine while a chunk was 300 tokens and the
 * match was usually in the first quarter of it; at 1024 tokens the first 260
 * characters are about 7% of the passage and routinely show the methodology
 * boilerplate a guideline opens with instead of the sentence that was matched.
 *
 * The terms come from the same `tokenise` the lexical rescoring uses, so what
 * centres the window is what would have scored it. Negatives never reach here:
 * the caller passes the positive half of the query, because a word the reader
 * asked to subtract must not be what the snippet is centred on.
 *
 * @param {string} text - The chunk's text.
 * @param {number} [max] - Characters, not counting the ellipses.
 * @param {string} [question] - The positive half of the query, or "".
 * @returns {string}
 */
export function snippet(text, max = 260, question = "") {
  const clean = tidy(text);
  if (!clean || clean.length <= max) return clean;
  const terms = [...new Set(tokenise(question))];
  const { folded, at } = terms.length ? foldWithOffsets(clean) : { folded: "", at: [] };
  const window = terms.length ? densestWindow(termHits(folded, terms), max) : null;
  if (!window) {
    // No term of the question is in the passage, so there is no better place to
    // start than its beginning.
    const cut = clean.slice(0, max);
    const space = cut.lastIndexOf(" ");
    return (space > max * 0.6 ? cut.slice(0, space) : cut) + "…";
  }
  const span = { start: at[window.start], end: at[window.end] };
  // Centred on what matched, then pushed back inside the text at either end, so a
  // match near the beginning or the end still gets a full-width window.
  const pad = Math.max(0, (max - (span.end - span.start)) / 2);
  let from = Math.max(0, Math.round(span.start - pad));
  let to = Math.min(clean.length, from + max);
  from = Math.max(0, to - max);
  // Snapped outwards to whitespace, never inwards: a window that starts mid-word
  // reads as a typo, and one that cuts off a matched term defeats the point.
  if (from > 0) {
    const space = clean.lastIndexOf(" ", from);
    from = space >= 0 && from - space < 30 ? space + 1 : from;
  }
  if (to < clean.length) {
    const space = clean.indexOf(" ", to);
    to = space >= 0 && space - to < 30 ? space : to;
  }
  return (from > 0 ? "…" : "") + clean.slice(from, to).trim() + (to < clean.length ? "…" : "");
}
