/* The parts of src/search.js that run on text alone.
 *
 * The ranking path proper has a gate of its own (scripts/check_search.mjs), which
 * runs the shipped module against the real dist/index/ and, when an encoder
 * answers, against real queries. What that gate cannot see is the query box:
 * check_search.mjs feeds it vectors, so a query-syntax regression would pass it
 * while the site quietly searched for the wrong string. These tests cover the
 * text-in, text-out half, the query algebra (on a stub encoder) and the BM25
 * blend, none of which needs the real index or a real encoder, so all of it runs
 * on every push rather than only when dist/index/ is built.
 *
 * Written by Claude Code (Opus 5).
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { installDocumentStub } from "../server/lib/browser_shim.mjs";

installDocumentStub();

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const { parseQuery, tokenise, snippet, rescore, pageLabel, embedParsedQuery,
        foldResults, groupByDocument, browseDocs, loadIndex, rank,
        previousPageRows, pageRowMap, pageKey, LOC_STRIDE, NO_PAGE_ROW, checkParsedQuery, searchRemote,
        allowedDocs, matchesGuideline, recencyBonuses, RECENCY_BONUS, RECENCY_HALFLIFE, figureRange,
        attachText }
  = await import(`${root}/src/search.js`);

/* The smallest index the folding and grouping code will accept: it reads document
 * metadata (for families) and the redundant-rendition set, and nothing else. */
function stubIndex(docs) {
  return { docs, redundant: new Set(["synthese", "argumentaire"]) };
}

test("an ordinary question is one positive text", () => {
  const q = parseQuery("prise en charge du bruit routier");
  assert.deepEqual(q.positive, ["prise en charge du bruit routier"]);
  assert.deepEqual(q.negative, []);
  assert.equal(q.lexical, "prise en charge du bruit routier");
});

test("| splits alternatives on both sides", () => {
  const q = parseQuery("isolation murs | isolation combles");
  assert.deepEqual(q.positive, ["isolation murs", "isolation combles"]);
});

test("the three spellings of an alternative are one query", () => {
  /* `A | B`, `(A | B)` and `A +(B)` are the same instruction written three ways,
   * and readers reach for all three. A syntax that accepts only the author's
   * favourite spelling reads as a bug: `A +(B)` used to parse to the single text
   * "A + B", which is a different question and a different vector. */
  const bare = parseQuery("isolation des combles | arret d'un chauffage");
  const parens = parseQuery("(isolation des combles | arret d'un chauffage)");
  const plus = parseQuery("isolation des combles +(arret d'un chauffage)");
  assert.deepEqual(bare.positive, ["isolation des combles", "arret d'un chauffage"]);
  assert.deepEqual(parens.positive, bare.positive);
  assert.deepEqual(plus.positive, bare.positive);
  assert.deepEqual(plus.negative, []);
  assert.equal(plus.lexical, bare.lexical);
});

test("+(...) and -(...) nest and combine", () => {
  const q = parseQuery("humidite +(moisissure murale | odeur persistante) -(enfant)");
  assert.deepEqual(q.positive, ["humidite", "moisissure murale", "odeur persistante"]);
  assert.deepEqual(q.negative, ["enfant"]);
  // Outside first, in the order written: the question, then what was added to it.
  assert.equal(q.lexical, "humidite moisissure murale odeur persistante");
});

test("a plus in ordinary prose is text, like a hyphen", () => {
  // Only `+(` opens a group. "cuivre + zinc" is a question about two metals,
  // not two questions, and must embed as the one sentence the reader wrote.
  assert.deepEqual(parseQuery("cuivre + zinc").positive, ["cuivre + zinc"]);
  assert.deepEqual(parseQuery("alliage cuivre+zinc").positive,
                   ["alliage cuivre+zinc"]);
});

test("an unclosed +( takes the rest of the string, like -(", () => {
  const q = parseQuery("humidite +(moisissure murale");
  assert.deepEqual(q.positive, ["humidite", "moisissure murale"]);
});

test("+() alone is degenerate syntax, not a query", () => {
  assert.deepEqual(parseQuery("+()").positive, []);
  assert.deepEqual(parseQuery("+(   )").positive, []);
});

test("-(...) subtracts and never reaches the lexical text", () => {
  const q = parseQuery("pollution -(enfant | adolescent)");
  assert.deepEqual(q.positive, ["pollution"]);
  assert.deepEqual(q.negative, ["enfant", "adolescent"]);
  /* The point of a separate lexical string: BM25 must not boost a passage for
   * containing the very word the reader asked to subtract. */
  assert.equal(q.lexical, "pollution");
  assert.ok(!q.lexical.includes("enfant"));
});

test("an unclosed group swallows the rest, and nothing throws", () => {
  const q = parseQuery("humidite -(enfant");
  assert.deepEqual(q.positive, ["humidite"]);
  assert.deepEqual(q.negative, ["enfant"]);
});

test("parentheses are structure, not content", () => {
  /* "(isolation)" and "isolation" have to reach the encoder as the same string, or
   * two spellings of one query retrieve two different shortlists. */
  assert.deepEqual(parseQuery("(isolation) murs").positive,
                   parseQuery("isolation murs").positive);
});

test("an empty box asks for nothing", () => {
  assert.deepEqual(parseQuery("").positive, []);
  assert.deepEqual(parseQuery("   ").positive, []);
  assert.deepEqual(parseQuery("-(tout)").positive, []);
});

test("tokenise folds accents and drops single letters", () => {
  assert.deepEqual(tokenise("Dégradation sévère, à 3 mois"),
                   ["degradation", "severe", "mois"]);
  assert.deepEqual(tokenise(""), []);
  assert.deepEqual(tokenise(null), []);
});

test("snippet cuts on a space and marks the cut", () => {
  const long = "mot ".repeat(200);
  const cut = snippet(long, 50);
  assert.ok(cut.length <= 51, `snippet is ${cut.length} characters`);
  assert.ok(cut.endsWith("…"));
  assert.ok(!cut.includes("  "));
  assert.equal(snippet("court"), "court");
  assert.equal(snippet(""), "");
});

test("snippet centres on the question instead of the head of the chunk", () => {
  // The shape that made this necessary: a report chunk opens on grading
  // boilerplate and answers the question 900 characters later.
  const boilerplate = "Les recommandations de grade A reposent sur une preuve scientifique etablie. ".repeat(12);
  const answer = "L'arret d'un chauffage par combustibles chez le locataire age se fait par paliers.";
  const text = boilerplate + answer + " " + boilerplate;
  const asked = "arret des combustibles chez le locataire age";
  const centred = snippet(text, 160, asked);
  assert.ok(centred.includes("combustibles chez le locataire age"),
    `the matched sentence is missing: ${centred}`);
  assert.ok(centred.startsWith("…") && centred.endsWith("…"),
    `both ends were cut, so both should say so: ${centred}`);
  assert.ok(centred.length <= 162 + 2, `snippet is ${centred.length} characters`);
  // Without the question it is the head of the chunk, which is what it always was.
  assert.ok(snippet(text, 160).startsWith("Les recommandations de grade A"));
});

test("snippet matches the question across accents and whole tokens only", () => {
  const text = "a".repeat(300) + " le releve systematique est recommande " + "b".repeat(300);
  // The reader typed it without the accent and the document has one.
  assert.ok(snippet(text, 80, "relevé").includes("releve"));
  // "age" must not match inside "dosage", so there is no window and the head wins.
  const dosage = "z".repeat(300) + " adapter le dosage " + "y".repeat(300);
  assert.ok(snippet(dosage, 80, "age").startsWith("zzz"));
});

test("snippet keeps the paragraph breaks the chunker put in", () => {
  const list = "Principes\n- premier point\n- deuxieme point\n- troisieme point";
  assert.equal(snippet(list, Infinity), list);
  // Runs of spaces are layout and still collapse, and so do blank lines.
  assert.equal(snippet("a   b\n\n\nc", Infinity), "a b\nc");
});

test("rescore promotes the candidate that names the word", () => {
  /* Both passages are equally close in embedding space; only one says
   * "ventilation". That is the case the BM25 blend exists for. */
  const results = [
    { cosine: 0.60, text: "prise en charge de la copropriete ancienne" },
    { cosine: 0.60, text: "la ventilation est indiquee dans la copropriete ancienne" },
  ];
  const sorted = rescore(results, "ventilation");
  assert.match(sorted[0].text, /ventilation/);
  assert.ok(sorted[0].score > sorted[0].cosine, "the blend did not move the score");
  assert.equal(sorted[0].cosine, 0.60, "cosine must survive rescoring untouched");
});

test("rescore scores a passage against its own language, not against the reader's", () => {
  /* The bug this covers, measured over the eval set's crosslingual group: BM25 is
   * divided by the best BM25 in the shortlist, a shortlist for a French question is
   * mostly French documents, and a French question only shares words with French
   * prose. So the English passage that answers it was divided by a French
   * candidate's score and got almost nothing, however good it was for an English
   * passage. Here the query is French, two French candidates match it well, and the
   * English candidate is the best English one there is. */
  const question = "ventilation dans la copropriete ancienne";
  // A technical term is the same word in both languages, which is why a crosslingual
  // query shares 0.837 of its terms with its own shortlist rather than none: the
  // English passage is not lexically invisible, it is lexically outbid.
  const list = [
    { cosine: 0.50, lang: "fr", text: "la ventilation dans la copropriete ancienne: ventilation et suivi" },
    { cosine: 0.50, lang: "fr", text: "copropriete ancienne, place de la ventilation" },
    { cosine: 0.52, lang: "en", text: "ventilation in old condominium buildings: ventilation monitoring" },
  ];
  const perLanguage = rescore(list.map((r) => ({ ...r })), question);
  const single = rescore(list.map(({ lang, ...r }) => r), question);   // no language: one scale
  const englishOf = (sorted) => sorted.find((r) => /monitoring/.test(r.text));

  // The English group sits below the floor, so the floor is its divisor: half the
  // shortlist's best instead of all of it, which is exactly twice the credit. 1 over
  // LEXICAL_GROUP_FLOOR is the most the floor can ever give back, by construction.
  assert.equal(englishOf(perLanguage).lexical, englishOf(single).lexical * 2);
  assert.ok(englishOf(perLanguage).lexical < 1, "partial credit, not a free ride");
  // The French group holds the shortlist's best, so nothing about it moves, and no
  // candidate in any group can pass 1.
  assert.equal(Math.max(...perLanguage.map((r) => r.lexical)), 1);
  assert.deepEqual(perLanguage.filter((r) => r.lang === "fr").map((r) => r.lexical),
                   single.filter((r) => /suivi|place/.test(r.text)).map((r) => r.lexical));
  assert.equal(englishOf(perLanguage).cosine, 0.52, "cosine must survive rescoring untouched");
});

test("a language group with nothing to say still scores nothing", () => {
  /* The floor's other half. Without it, per-language normalisation would hand every
   * language's best candidate the full term, so a shortlist's only English passage
   * would be promoted for being unopposed however little it matched. With it, a
   * group whose best is below half the shortlist's best is divided by that half, and
   * a group that matches no term at all is still zero rather than the best of
   * nothing. */
  const question = "ventilation condensation surveillance hebdomadaire";
  const mute = rescore([
    { cosine: 0.50, lang: "fr", text: "ventilation et condensation: surveillance hebdomadaire" },
    { cosine: 0.49, lang: "en", text: "nothing to do with the question" },
  ], question);
  assert.equal(mute.find((r) => /nothing/.test(r.text)).lexical, 0);
});

test("rescore with one language, or none, is the single-scale blend it always was", () => {
  /* The guarantee that keeps every earlier measurement comparable: per-language
   * normalisation can only matter where there is more than one language to separate,
   * so a hand-built list and a single-language shortlist must both behave exactly as
   * they did before the groups existed. */
  const texts = [
    { cosine: 0.60, text: "prise en charge de la copropriete ancienne" },
    { cosine: 0.60, text: "la ventilation est indiquee dans la copropriete ancienne" },
    { cosine: 0.55, text: "ventilation ventilation ventilation" },
  ];
  const bare = rescore(texts.map((t) => ({ ...t })), "ventilation");
  const tagged = rescore(texts.map((t) => ({ ...t, lang: "fr" })), "ventilation");
  assert.deepEqual(bare.map((r) => r.lexical), tagged.map((r) => r.lexical));
  // One candidate is the scale, so it is exactly 1 and the others are fractions.
  assert.equal(Math.max(...bare.map((r) => r.lexical)), 1);
});

test("rescore leaves a list alone when it has nothing to say", () => {
  const results = [{ cosine: 0.7, text: "a" }, { cosine: 0.6, text: "b" }];
  assert.equal(rescore(results, "ventilation", 0), results);
  assert.equal(rescore([{ cosine: 0.7, text: "a" }], "ventilation")[0].cosine, 0.7);
});

test("pageLabel says one page or a span", () => {
  assert.match(pageLabel({ page: 7 }), /7/);
  const span = pageLabel({ page: 7, pages: [7, 8, 9] });
  assert.match(span, /7/);
  assert.match(span, /9/);
});

test("a term too short to mean anything is refused before the request", async () => {
  // "test" is four characters, the encoder's floor is five, and what the reader
  // used to get was "embed 400". Nothing is stubbed here on purpose: if this ever
  // reaches fetch, the test fails with a connection error rather than passing.
  await assert.rejects(
    () => embedParsedQuery(parseQuery("test"), 1024),
    (error) => error.message === "short" && error.min === 5,
  );
});

test("one short alternative is enough to refuse the whole query", async () => {
  await assert.rejects(
    () => embedParsedQuery(parseQuery("prise en charge du bruit routier | BR"), 1024),
    (error) => error.message === "short",
  );
});

test("a short subtraction is not held to the length of a question", async () => {
  // "-(ado)" says what the question is not about; it is never the question, so
  // the floor that exists to stop keyword queries has no business refusing it.
  // It still has to fail, because there is no encoder here: on "unavailable" or
  // a fetch error the length checks are behind us, which is what this asserts.
  await assert.rejects(
    () => embedParsedQuery(parseQuery("prise en charge du bruit routier -(ado)"), 1024),
    (error) => error.message !== "short",
  );
});

test("a query with too many terms is told that first, short ones included", async () => {
  // The three limits fire in the order long, terms, short: eleven one-letter
  // alternatives are both too many and too short, and "too many terms" is the
  // one that describes what is wrong with the query.
  await assert.rejects(
    () => embedParsedQuery(parseQuery("(a|b|c|d|e|f) -(g|h|i|j|k)"), 1024),
    (error) => error.message === "terms",
  );
});

/* A fake encoder, so what the query box does with several texts can be checked
 * without a model: one deterministic direction per string, in the wire format
 * embed-service.py answers with (base64 int8, fixed scale 127, the `dim` asked
 * for). Two different strings come back as two unrelated directions, which is the
 * one property these tests lean on and the one a real encoder also has. `fixed`
 * pins chosen strings to chosen vectors. */
function stubEncoder(fixed = {}) {
  const asked = [];
  const previous = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    assert.equal(url, "api/sem/embed");
    const { q, dim } = JSON.parse(init.body);
    asked.push(q);
    // A text named in `fixed` answers with that exact vector, for the algebra tests
    // that need to know what the query is made of.
    if (fixed[q]) {
      const vector = fixed[q];
      return {
        ok: true,
        status: 200,
        json: async () => ({ dim, q: Buffer.from(vector.buffer, vector.byteOffset, vector.length).toString("base64") }),
      };
    }
    let seed = 2166136261;
    for (const ch of q) seed = Math.imul(seed ^ ch.codePointAt(0), 16777619) >>> 0;
    const bytes = new Int8Array(dim);
    for (let d = 0; d < dim; d += 1) {
      seed = (Math.imul(seed, 1103515245) + 12345) >>> 0;
      bytes[d] = ((seed >>> 16) % 255) - 127;
    }
    return {
      ok: true,
      status: 200,
      json: async () => ({ dim, q: Buffer.from(bytes.buffer).toString("base64") }),
    };
  };
  return { asked, restore() { globalThis.fetch = previous; } };
}

const A = "isolation des combles";
const B = "arret d'un chauffage";

test("every alternative is embedded on its own, and the query is their mean", async () => {
  /* The point of the syntax: `A | B` must not be the single string "A B". If it
   * were, nothing in the parser's tests would show it, because the difference is
   * one vector away from the box. */
  // Texts of their own: src/search.js caches query vectors by (text, width), so a
  // text any other test has asked for would reach no encoder here and the count
  // below would depend on the order the file happens to run in.
  const first = "prise en charge du bruit aérien";
  const second = "traitement de l'isolation phonique";
  const encoder = stubEncoder();
  try {
    const both = await embedParsedQuery(parseQuery(`${first} | ${second}`), 1024);
    assert.deepEqual(encoder.asked, [first, second],
                     "the two alternatives went out as two requests");
    const joined = await embedParsedQuery(parseQuery(`${first} ${second}`), 1024);
    assert.equal(encoder.asked.at(-1), `${first} ${second}`, "and the joined question as one");
    assert.notDeepEqual(Array.from(both), Array.from(joined),
                        "averaging two texts landed on the same vector as embedding one");
    // And it is a mean, not one of the two: it sits between them.
    const alone = await embedParsedQuery(parseQuery(first), 1024);
    assert.notDeepEqual(Array.from(both), Array.from(alone));
  } finally {
    encoder.restore();
  }
});

test("the three spellings of an alternative embed identically", async () => {
  // The parser says they are the same two texts; this is the same claim one step
  // later, where it is the thing that actually reaches the index.
  const encoder = stubEncoder();
  try {
    const bare = await embedParsedQuery(parseQuery(`${A} | ${B}`), 1024);
    const parens = await embedParsedQuery(parseQuery(`(${A} | ${B})`), 1024);
    const plus = await embedParsedQuery(parseQuery(`${A} +(${B})`), 1024);
    assert.deepEqual(Array.from(parens), Array.from(bare));
    assert.deepEqual(Array.from(plus), Array.from(bare));
  } finally {
    encoder.restore();
  }
});

test("a subtraction moves the query off the vector it would have had", async () => {
  // A text of its own, because src/search.js caches query vectors by (text, width)
  // and a text another test has already asked for reaches no encoder at all.
  const aside = "chez la personne agee";
  const encoder = stubEncoder();
  try {
    const plain = await embedParsedQuery(parseQuery(A), 1024);
    const minus = await embedParsedQuery(parseQuery(`${A} -(${aside})`), 1024);
    assert.ok(encoder.asked.includes(aside), "the subtracted text was never embedded");
    assert.notDeepEqual(Array.from(minus), Array.from(plain),
                        "the negative group changed nothing about the query vector");
  } finally {
    encoder.restore();
  }
});

test("a folded result carries the score that ordered it", () => {
  /* The bug this pins: the list was sorted by `score` (cosine plus the lexical
   * term) while every row displayed `cosine`, so the numbers on screen ran out of
   * order and the sort looked broken. A candidate below RESCORE_DEPTH never went
   * through rescore and has no `score` of its own, which is why the fallback has
   * to be here rather than in the renderer. */
  const index = stubIndex([{ file: "a.pdf" }, { file: "b.pdf" }]);
  const [rescored, untouched] = foldResults(index, [
    { chunk: 0, doc: 0, page: 1, cosine: 0.44, score: 0.51 },
    { chunk: 9, doc: 1, page: 3, cosine: 0.47 },
  ]);
  assert.equal(rescored.score, 0.51);
  assert.equal(rescored.cosine, 0.44, "the vector number has to survive folding");
  assert.equal(untouched.score, 0.47, "a candidate never rescored scores its cosine");
});

test("the grouped view ranks documents by the same number as the passage list", () => {
  /* Same trap one level up: grouping used to sort by the best cosine, so the two
   * views could disagree about which document comes first while claiming to be two
   * presentations of one answer. */
  const index = stubIndex([{ file: "a.pdf" }, { file: "b.pdf" }]);
  const results = foldResults(index, [
    { chunk: 0, doc: 0, page: 1, cosine: 0.44, score: 0.61 },
    { chunk: 5, doc: 1, page: 2, cosine: 0.52, score: 0.52 },
  ]);
  const groups = groupByDocument(index, results);
  assert.deepEqual(groups.map((g) => g.doc), [0, 1]);
  assert.equal(groups[0].score, 0.61);
  assert.equal(groups[0].cosine, 0.44, "the group keeps its best passage's cosine too");
});

/* The document-name filter on the browse page. It is the only way to find a
 * document by its title without an encoder, so it has to keep working when search
 * is down, and it has to fold accents: half these titles carry one. */
const NAMED = [
  { file: "insee_releve_bruit_2021.pdf", title: "Relevé du bruit pour le PCAET" },
  { file: "ons_energy_2020.pdf", title: "Energy poverty: assessment and management" },
  { file: "insee_energie_argumentaire.pdf", title: "Précarité énergétique, argumentaire" },
];

test("the name filter matches a substring, so it narrows while you type", () => {
  const index = stubIndex(NAMED);
  assert.deepEqual(browseDocs(index, {}, "fr", "energ").length, 2);
  assert.deepEqual(browseDocs(index, {}, "fr", "energy").length, 1);
});

test("the name filter ignores accents in both directions", () => {
  const index = stubIndex(NAMED);
  // Typed without the accent, written with one.
  assert.equal(browseDocs(index, {}, "fr", "releve").length, 1);
  // Typed WITH an accent the document does not carry: folding both sides means a
  // reader cannot miss a document by over-accenting it either, which is the case a
  // one-sided fold gets wrong.
  assert.equal(browseDocs(index, {}, "fr", "PCÄET").length, 1);
  assert.equal(browseDocs(index, {}, "fr", "pcaet").length, 1);
});

test("several words all have to appear, in any order and across title and filename", () => {
  const index = stubIndex(NAMED);
  assert.equal(browseDocs(index, {}, "fr", "energetique insee").length, 1);
  assert.equal(browseDocs(index, {}, "fr", "energetique ons").length, 0);
});

test("an empty or blank name filter is no filter at all", () => {
  const index = stubIndex(NAMED);
  assert.equal(browseDocs(index, {}, "fr", "").length, 3);
  assert.equal(browseDocs(index, {}, "fr", "   ").length, 3);
  assert.equal(browseDocs(index, {}, "fr").length, 3);
});

test("the document list loads meta.json alone, and never the matrices", async () => {
  // The browse page cannot rank, and the matrices are 20 MB. A regression here is
  // invisible on a warm cache and costs a first-time reader the whole index for a
  // page that only lists documents, which is why it is asserted rather than trusted.
  const asked = [];
  const meta = {
    format_version: 5, dims: 1024, quant: "binary", n_chunks: 2, n_pages: 1,
    bytes_per_vector: 128, vectors_file: "v.b1", chunks_file: "c.u16",
    pages_file: "p.b1", page_weight: 0.15, sections_file: "s.b1", section_weight: 0.1,
    n_sections: 1, documents: [], renditions: {},
  };
  const realFetch = globalThis.fetch;
  globalThis.fetch = async (url) => {
    asked.push(String(url));
    return { ok: true, url: String(url), json: async () => meta };
  };
  try {
    const index = await loadIndex("index", { vectors: false });
    assert.deepEqual(asked, ["index/meta.json"]);
    assert.equal(index.vectors, null);
    assert.equal(index.pageWeight, 0);
    assert.equal(index.sectionWeight, 0);
    assert.equal(index.sections, null);
    assert.equal(index.docs, meta.documents);
  } finally {
    globalThis.fetch = realFetch;
  }
});

/* The context blend, on an int8 index small enough to do the arithmetic by hand.
 * check_search.mjs proves the same sum on the real one-bit index to 2%; here it
 * is exact, and the missing-row rule (a term with no row takes the chunk's own
 * score) is pinned per term rather than left to the gate's sample. */

/** A chunk's location row: document, page, page-vector row, section row. */
function loc(doc, page, pageRow, sectionRow) {
  return [doc, page, pageRow, sectionRow];
}

test("a chunk's previous page is the page before it in the same document, when that page has a vector", () => {
  // Document 0 has pages 1, 2 and 4 (page 3 has no chunk, a full-page figure);
  // document 1 starts again at page 1, with its own row for it.
  const rows = [loc(0, 1, 0, 0), loc(0, 2, 1, 0), loc(0, 2, 1, 0), loc(0, 4, 2, 0),
                loc(1, 1, 3, 1), loc(1, 2, NO_PAGE_ROW, 1), loc(1, 3, 4, 1)];
  const prev = previousPageRows(Uint16Array.from(rows.flat()), rows.length);
  assert.deepEqual([...prev], [
    NO_PAGE_ROW,  // page 1 has no predecessor
    0, 0,         // both chunks of page 2 point at page 1's row
    NO_PAGE_ROW,  // page 4: page 3 has no chunk, so no row, rather than page 2's
    NO_PAGE_ROW,  // document 1, page 1: not document 0's last page
    3,            // page 2 of document 1, whose own page has no vector, still has a predecessor
    NO_PAGE_ROW,  // page 3: its predecessor exists but has no vector
  ]);
  assert.equal(LOC_STRIDE, 4);
});

test("every page with a vector can be looked up by document and page", () => {
  const rows = [loc(0, 1, 0, 0), loc(0, 2, 1, 0), loc(1, 1, 2, 1), loc(1, 2, NO_PAGE_ROW, 1)];
  const rowOf = pageRowMap(Uint16Array.from(rows.flat()), rows.length);
  assert.equal(rowOf.get(pageKey(0, 2)), 1);
  assert.equal(rowOf.get(pageKey(1, 1)), 2);  // page 1 of document 1, not of document 0
  assert.equal(rowOf.has(pageKey(1, 2)), false);  // a page without a vector is absent
});

/** An int8 index of `dims` 4 whose every stored vector is a unit axis. */
function tinyIndex({ pageWeight = 0, prevPageWeight = 0, sectionWeight = 0 } = {}) {
  const axis = (d) => Int8Array.from([0, 1, 2, 3].map((i) => (i === d ? 127 : 0)));
  // Three chunks of one document on pages 1, 2, 2. Page rows: page 1 -> 0, page 2 -> 1.
  // Chunk 2 sits in section 1, the others in section 0; chunk 1's section is unknown.
  const locations = Uint16Array.from([
    ...loc(0, 1, 0, 0), ...loc(0, 2, 1, NO_PAGE_ROW), ...loc(0, 2, 1, 1),
  ].flat());
  const concat = (...vs) => { const out = new Int8Array(vs.length * 4); vs.forEach((v, i) => out.set(v, i * 4)); return out; };
  return {
    meta: { dims: 4, quant: "int8", n_chunks: 3, n_pages: 2, n_sections: 2 },
    binary: false, stride: 4,
    vectors: concat(axis(0), axis(0), axis(1)),
    pages: concat(axis(1), axis(2)),
    sections: concat(axis(3), axis(1)),
    locations, prevRows: previousPageRows(locations, 3),
    pageWeight, prevPageWeight, sectionWeight,
    docs: [{ chunk_offset: 0, chunk_count: 3 }], redundant: new Set(),
  };
}

/** Cosine of every chunk, by chunk number, for a query along one axis. */
function cosines(index, axis, options = {}) {
  const q = Int8Array.from([0, 1, 2, 3].map((i) => (i === axis ? 127 : 0)));
  const out = [];
  for (const h of rank(index, q, {}, options)) out[h.chunk] = Number(h.cosine.toFixed(4));
  return out;
}

test("the figures filter keeps, drops or isolates a document's trailing figure chunks", () => {
  // Chunks 0 and 1 are text, chunk 2 is a described figure: tinyIndex's third row.
  const index = tinyIndex();
  index.docs[0].figure_count = 1;
  const chunks = (figures) => rank(index, Int8Array.from([127, 127, 0, 0]), {}, { figures })
    .map((h) => h.chunk).sort();
  assert.deepEqual(chunks("include"), [0, 1, 2]);
  assert.deepEqual(chunks(undefined), [0, 1, 2], "an omitted mode means include");
  assert.deepEqual(chunks("exclude"), [0, 1]);
  assert.deepEqual(chunks("only"), [2]);
});

test("a document without figures has nothing to offer the figures-only filter", () => {
  const doc = { chunk_offset: 10, chunk_count: 5 };
  assert.deepEqual(figureRange(doc, "include"), [10, 15]);
  assert.deepEqual(figureRange(doc, "exclude"), [10, 15]);
  assert.deepEqual(figureRange(doc, "only"), [15, 15]);
});

test("with every weight at zero a chunk scores on its own vector alone", () => {
  assert.deepEqual(cosines(tinyIndex(), 0), [1, 1, 0]);
  assert.deepEqual(cosines(tinyIndex(), 1), [0, 0, 1]);
});

test("the page term blends in the chunk's own page", () => {
  // Query along axis 1: chunk 2 matches (1), its page 2 does not (0).
  // Chunk 0 does not match (0) but its page 1 does (1). Chunk 1 matches
  // nothing, on a page that matches nothing.
  const index = tinyIndex({ pageWeight: 0.2 });
  assert.deepEqual(cosines(index, 1), [0.2, 0, 0.8]);
});

test("the previous-page term reads the page before, and page 1 keeps its own score", () => {
  // Query along axis 1: chunks 1 and 2 are on page 2, whose predecessor is page 1
  // (axis 1, a match). Chunk 0 is on page 1, which has no predecessor, so the
  // term takes its own score of 0.
  const index = tinyIndex({ prevPageWeight: 0.5 });
  assert.deepEqual(cosines(index, 1), [0, 0.5, 1]);
});

test("the section term reads the section row, and a chunk with none keeps its own score", () => {
  // Query along axis 3: only section 0 matches. Chunk 0 is in it (0.25);
  // chunk 1 has no section (own score, 0); chunk 2 is in section 1 (0).
  const index = tinyIndex({ sectionWeight: 0.25 });
  assert.deepEqual(cosines(index, 3), [0.25, 0, 0]);
});

test("the four terms sum to the score the module docstring states", () => {
  // Query along axis 1, chunk 2: own 1, page 0, previous page 1, section 1.
  // Chunk 1: own 0, page 0, previous page 1, section unknown (own, 0).
  const index = tinyIndex({ pageWeight: 0.1, prevPageWeight: 0.05, sectionWeight: 0.15 });
  const c = cosines(index, 1);
  assert.equal(c[2], Number((0.7 * 1 + 0.1 * 0 + 0.05 * 1 + 0.15 * 1).toFixed(4)));
  assert.equal(c[1], Number((0.7 * 0 + 0.1 * 0 + 0.05 * 1 + 0.15 * 0).toFixed(4)));
});

test("a weight is ignored when its matrix did not ship", () => {
  const index = tinyIndex({ sectionWeight: 0.25, prevPageWeight: 0.5 });
  index.sections = null;
  index.pages = null;
  assert.deepEqual(cosines(index, 3), [0, 0, 0]);
  assert.deepEqual(cosines(index, 1), [0, 0, 1]);
});

test("the reference penalty costs a chunk its share of a cosine, and only when asked", () => {
  // Chunk 1 is scored a full bibliography (255 of a scale of 1.0), so the penalty
  // takes REFERENCE_PENALTY of a cosine off it: 1 becomes 0.7. Chunks 0 and 2 are
  // scored 0 and keep what they had.
  const withRefs = () => {
    const index = tinyIndex();
    index.refs = Uint8Array.from([0, 255, 0]);
    index.refScale = 1;
    return index;
  };
  assert.deepEqual(cosines(withRefs(), 0, { demoteReferences: true }), [1, 0.7, 0]);
  // Off is this function's default, and the service's is on: the ranker cannot
  // guess whether an index even carries the row, so the caller says.
  assert.deepEqual(cosines(withRefs(), 0), [1, 1, 0]);
});

test("the reference penalty is scaled by the axis the index was built with", () => {
  // Half the scale, so half the penalty: 0.15 of a cosine rather than 0.30.
  const index = tinyIndex();
  index.refs = Uint8Array.from([0, 255, 0]);
  index.refScale = 0.5;
  assert.deepEqual(cosines(index, 0, { demoteReferences: true }), [1, 0.85, 0]);
});

test("the page's own check refuses what the encoder would, before any request", () => {
  assert.throws(() => checkParsedQuery(parseQuery("test")), (e) => e.message === "short" && e.min === 5);
  assert.throws(() => checkParsedQuery(parseQuery("-(rien que du négatif)")), /empty/);
  assert.throws(() => checkParsedQuery(parseQuery("x".repeat(600))), (e) => e.message === "long" && e.max === 512);
  assert.throws(() => checkParsedQuery(parseQuery(Array(11).fill("terme").join(" | "))), (e) => e.message === "terms" && e.got === 11);
  checkParsedQuery(parseQuery("prise en charge du bruit routier -(ado)"));
});

/** Stub fetch with one canned response, run `fn`, and put fetch back. */
async function withResponse(status, payload, fn) {
  const real = globalThis.fetch;
  let seen = null;
  globalThis.fetch = async (url, init) => {
    seen = { url, body: JSON.parse(init.body) };
    if (status === "network") throw new TypeError("fetch failed");
    return { ok: status === 200, status, json: async () => payload };
  };
  try {
    return [await fn(), seen];
  } finally {
    globalThis.fetch = real;
  }
}

test("searchRemote sends the question, the filters and the toggle, and returns the answer whole", async () => {
  const answer = { results: [{ doc: 1, chunk: 2 }], floor: 0.28, dims: 1024 };
  const [got, seen] = await withResponse(200, answer, () => searchRemote("une question", { topic: ["housing"] }, { bm25: true }));
  assert.deepEqual(got, answer);
  assert.equal(seen.url, "api/search");
  assert.deepEqual(seen.body, {
    q: "une question", filters: { topic: ["housing"] }, bm25: true, demote_references: true,
    favour_recent: true, figures: "include",
  });
});

test("searchRemote carries the figures filter", async () => {
  const [, seen] = await withResponse(200, {}, () => searchRemote("une question", {}, { figures: "only" }));
  assert.equal(seen.body.figures, "only");
});

test("searchRemote says so when the reader turned the bibliography penalty off", async () => {
  const [, seen] = await withResponse(200, {}, () => searchRemote("une question", {}, { demoteRefs: false }));
  assert.equal(seen.body.demote_references, false);
});

test("searchRemote says so when the reader turned the recency bonus off", async () => {
  const [, seen] = await withResponse(200, {}, () => searchRemote("une question", {}, { recent: false }));
  assert.equal(seen.body.favour_recent, false);
  assert.equal(seen.body.demote_references, true, "the two toggles travel independently");
});

test("searchRemote turns the service's refusals back into the errors the page translates", async () => {
  const expect = async (status, payload, check) => {
    await assert.rejects(withResponse(status, payload, () => searchRemote("q", {})), check);
  };
  await expect(400, { error: "short", min: 5 }, (e) => e.message === "short" && e.min === 5);
  await expect(400, { error: "terms", got: 12, max: 10 }, (e) => e.message === "terms" && e.got === 12 && e.max === 10);
  await expect(500, { error: "dim", got: 256, want: 1024 }, (e) => e.message === "dim" && e.want === 1024);
  await expect(502, { error: "unavailable" }, /unavailable/);
  await expect(503, { error: "full" }, /busy/);
  await expect("network", null, /unavailable/);
  await expect(418, {}, /search 418/);
});

/* --- the guideline level ---------------------------------------------------
 *
 * The one filter that is on before the reader touches anything, which makes its
 * two departures from a facet worth pinning: the tiers are NESTED (asking for
 * `family` keeps `strict` too) and a BLANK cell PASSES (418 of the 534 documents
 * have no tier yet, and hiding them by default would leave a reader searching a
 * fifth of the corpus with nothing on screen to say so).
 */

/** An index carrying tiers, as build_index.py ships them: narrowest first. */
function tieredIndex(tiers) {
  return {
    docs: tiers.map((guideline, i) => ({ file: `${i}.pdf`, guideline })),
    meta: { guideline_tiers: ["strict", "family", "wide", "no"], separator: ";" },
    redundant: new Set(),
  };
}

const TIERS = ["strict", "family", "wide", "no", ""];

test("a level keeps every tier at or narrower than itself", () => {
  const index = tieredIndex(TIERS);
  // Indices into TIERS: the blank (4) is in every one of these on purpose.
  assert.deepEqual([...allowedDocs(index, { guideline: "strict" })].sort(), [0, 4]);
  assert.deepEqual([...allowedDocs(index, { guideline: "family" })].sort(), [0, 1, 4]);
  assert.deepEqual([...allowedDocs(index, { guideline: "wide" })].sort(), [0, 1, 2, 4]);
});

test("an uncurated document is never hidden by the level", () => {
  /* The rule that separates this from every facet, where a blank cell fails. The
   * site ships with the level on, so the facet rule would have hidden 418 of 534
   * documents from a reader who never opened the filter panel. */
  const index = tieredIndex(TIERS);
  for (const level of ["strict", "family", "wide"]) {
    assert.ok(allowedDocs(index, { guideline: level }).has(4), `${level} hid a blank tier`);
  }
});

test("the tier no is reachable only by asking for no level at all", () => {
  /* "all" is spelled as the field being absent, which is also what lets the
   * whole-corpus case skip the per-document test entirely. */
  const index = tieredIndex(TIERS);
  assert.equal(allowedDocs(index, {}), null, "an absent level must filter nothing");
  assert.ok(!allowedDocs(index, { guideline: "wide" }).has(3), "a `no` document passed `wide`");
});

test("a level the index cannot place filters nothing", () => {
  /* Fails open, deliberately: a link shared with a tier that manifest.py has since
   * renamed shows the corpus rather than an empty list the reader cannot explain.
   * Same for an index built before the tiers existed. */
  const index = tieredIndex(TIERS);
  assert.equal(allowedDocs(index, { guideline: "stricte" }).size, TIERS.length);
  const old = { docs: index.docs, meta: { separator: ";" }, redundant: new Set() };
  assert.equal(allowedDocs(old, { guideline: "strict" }).size, TIERS.length);
});

test("the level narrows alongside the facets, not instead of them", () => {
  /* `allowedDocs` ANDs its active fields, and the level is one of them: a reader
   * with an issuer chosen and the default level must get the intersection. */
  const index = {
    docs: [
      { file: "a.pdf", guideline: "strict", issuer: "INSEE" },
      { file: "b.pdf", guideline: "no", issuer: "INSEE" },
      { file: "c.pdf", guideline: "strict", issuer: "ONS" },
    ],
    meta: { guideline_tiers: ["strict", "family", "wide", "no"], separator: ";" },
    redundant: new Set(),
  };
  assert.deepEqual([...allowedDocs(index, { guideline: "family", issuer: ["INSEE"] })], [0]);
});

test("matchesGuideline is the whole rule, for one document", () => {
  const index = tieredIndex(TIERS);
  assert.ok(matchesGuideline(index, { guideline: "strict" }, "family"));
  assert.ok(!matchesGuideline(index, { guideline: "wide" }, "family"));
  assert.ok(matchesGuideline(index, { guideline: "" }, "strict"));
  assert.ok(matchesGuideline(index, {}, "strict"), "a document with no cell at all");
});

/* The recency bonus: what a document's year is worth to every passage in it.
 *
 * The arithmetic is checked here rather than through rank() because rank() adds
 * it to a blended score and divides by a quantisation constant, which would let a
 * sign error and a scale error cancel. The ranking half is checked below, on the
 * only thing it has to guarantee: order.
 */
test("the newest document gets the whole bonus and older ones half it per half-life", () => {
  const docs = [{ year: "2024" }, { year: "2016" }, { year: "2008" }];
  const got = [...recencyBonuses(docs, { bonus: 1, halfLife: 8 })];
  assert.deepEqual(got.map((v) => Number(v.toFixed(6))), [1, 0.5, 0.25]);
});

test("a document with no year is treated as the median year of the corpus", () => {
  /* The only value that neither rewards nor punishes a blank cell. Two of the 534
   * documents are in this case, so the point is not the magnitude, it is that the
   * two obvious alternatives are silently wrong: year 0 buries the document for
   * good, and "newest" makes a missing cell the best thing that can happen to it. */
  const docs = [{ year: "2024" }, { year: "2016" }, { year: "2008" }, { year: "" }, {}];
  const got = recencyBonuses(docs, { bonus: 1, halfLife: 8 });
  assert.equal(Number(got[3].toFixed(6)), 0.5, "the median of 2008, 2016, 2024");
  assert.equal(Number(got[4].toFixed(6)), 0.5, "a document with no year cell at all");
});

test("a year past the newest cannot buy more than the bonus", () => {
  // A mis-parsed year is a manifest bug, and it should cost a rank, not the list.
  const got = recencyBonuses([{ year: "2024" }, { year: "2031" }], { bonus: 1, halfLife: 8 });
  assert.equal(got[1], 1);
});

test("there is no bonus to compute without a weight or without a year", () => {
  assert.equal(recencyBonuses([{ year: "2024" }], { bonus: 0, halfLife: 8 }), null);
  assert.equal(recencyBonuses([{ year: "2024" }], { bonus: 0.03, halfLife: 0 }), null);
  assert.equal(recencyBonuses([{ year: "" }, { year: "n/a" }], { bonus: 0.03, halfLife: 8 }), null);
  assert.equal(recencyBonuses([], { bonus: 0.03, halfLife: 8 }), null);
});

/** Two documents of one chunk each, the same vector, different years. */
function datedIndex(years) {
  const axis = (d) => Int8Array.from([0, 1, 2, 3].map((i) => (i === d ? 127 : 0)));
  const vectors = new Int8Array(years.length * 4);
  years.forEach((_, i) => vectors.set(axis(0), i * 4));
  const locations = Uint16Array.from(years.flatMap((_, i) => [i, 1, NO_PAGE_ROW, NO_PAGE_ROW]));
  return {
    meta: { dims: 4, quant: "int8", n_chunks: years.length },
    binary: false, stride: 4, vectors, locations, prevRows: null,
    pageWeight: 0, prevPageWeight: 0, sectionWeight: 0,
    docs: years.map((year, i) => ({ chunk_offset: i, chunk_count: 1, year })),
    redundant: new Set(),
  };
}

test("two equally relevant passages are separated by their documents' years, and only when asked", () => {
  const index = datedIndex(["2000", "2024"]);
  const q = Int8Array.from([127, 0, 0, 0]);
  // Off, which is what rank() defaults to: the two are indistinguishable, and the
  // order between them is whatever the scan happened to produce.
  const plain = rank(index, q, {});
  assert.equal(plain[0].cosine, plain[1].cosine);
  // On: the 2024 document leads, by the difference between the two bonuses and
  // not by more. RECENCY_BONUS is the whole budget, spent across 1943 to now.
  const biased = rank(index, q, {}, { favourRecent: true });
  assert.equal(biased[0].doc, 1);
  const expected = RECENCY_BONUS * (1 - Math.pow(0.5, 24 / RECENCY_HALFLIFE));
  assert.ok(Math.abs((biased[0].cosine - biased[1].cosine) - expected) < 1e-6,
    `${biased[0].cosine - biased[1].cosine} should be ${expected}`);
});

test("a sweep can set the bonus on the index without rebuilding it", () => {
  /* The same idiom the context weights use, and scripts/sweep_recency.mjs is the
   * reason it exists: the measurement has to run the SHIPPED ranker, so the knobs
   * have to be reachable from outside it. */
  const index = datedIndex(["2000", "2024"]);
  const q = Int8Array.from([127, 0, 0, 0]);
  index.recencyBonus = 0.5;
  index.recencyHalfLife = 24;
  const biased = rank(index, q, {}, { favourRecent: true });
  assert.equal(Number((biased[0].cosine - biased[1].cosine).toFixed(6)), 0.25);
});

test("attachText keeps a cached document even if the cache is trimmed while it waits", async () => {
  // The search service trims its shared text cache after every answer. A request
  // that found document 0 cached (so did not fetch it) and was still waiting on
  // document 1 used to read document 0 back from the cache AFTER the wait, by
  // which time a concurrent request had evicted it: the result lost its text and
  // boxes and was rescored on an empty string.
  const index = {
    docs: [{ chunk_offset: 0 }, { chunk_offset: 1 }],
    textCache: new Map([[0, [{ text: "cached passage", pages: [1], boxes: {} }]]]),
  };
  let release;
  const gate = new Promise((done) => { release = done; });
  const saved = globalThis.fetch;
  globalThis.fetch = async () => {
    await gate;
    return { ok: true, json: async () => ({ chunks: [{ text: "fetched passage", pages: [2], boxes: {} }] }) };
  };
  try {
    const results = [{ doc: 0, chunk: 0 }, { doc: 1, chunk: 1 }];
    const pending = attachText(index, results);
    await Promise.resolve();
    index.textCache.delete(0);  // what a concurrent boundTextCache does
    release();
    await pending;
    assert.equal(results[0].text, "cached passage");
    assert.equal(results[1].text, "fetched passage");
  } finally {
    globalThis.fetch = saved;
  }
});

/* The parse table. The first rows are the ones that must never break: every query
 * written before this syntax existed is one positive text and no negatives. The
 * last block is punctuation with no text in it: a reader mashing the syntax keys,
 * or a URL carrying ?query=|||, has to land on "ask a question", not on a crash. */
const PARSES = [
  ["arrêt des chaudières", ["arrêt des chaudières"], []],
  ["suivi thermo-hydraulique", ["suivi thermo-hydraulique"], []],   // a bare hyphen is prose
  ["prise en charge - adulte", ["prise en charge - adulte"], []],
  ["chaudières -(personne âgée)", ["chaudières"], ["personne âgée"]],
  ["(isolation | arrêt) -(fioul | charbon)", ["isolation", "arrêt"], ["fioul", "charbon"]],
  ["a -(b) -(c)", ["a"], ["b", "c"]],
  ["dégradation -(humidité", ["dégradation"], ["humidité"]],        // unclosed: take the rest
  ["-(only negative)", [], ["only negative"]],
  ["isolation +(arrêt | fin de chantier)", ["isolation", "arrêt", "fin de chantier"], []],
  ["a +(b) -(c)", ["a", "b"], ["c"]],
  ["   ", [], []],
  ["|||", [], []],
  ["a |||| b", ["a", "b"], []],
  ["()", [], []],
  ["-()", [], []],
  ["+(|||)", [], []],
  ["  |  |  ", [], []],
  ["-(|||)", [], []],
  ["((a))", ["a"], []],
];

for (const [text, positive, negative] of PARSES) {
  test(`parses ${JSON.stringify(text)}`, () => {
    const got = parseQuery(text);
    assert.deepEqual(got.positive, positive);
    assert.deepEqual(got.negative, negative);
  });
}

/* The vector arithmetic, through embedParsedQuery's own plumbing. Two fixed
 * directions stand in for two embedded texts, and "blend" is half of each:
 * two unrelated vectors at 1024 dims are very nearly orthogonal, so subtracting one
 * from the other would prove nothing (removing a direction the query does not have
 * changes nothing). A query that genuinely overlaps the negative is the case the
 * projection has to get right. The texts are five letters, the encoder's floor,
 * and used by no other test, because query vectors are cached by text. */
const DIMS = 1024;
/* A pseudo-random unit vector in the encoder's wire format: scaled by 127 and
   rounded, as embed-service.py answers. At that scale a component is a few counts,
   so a vector that was not already on it would lose a percent of its direction to
   rounding on the round trip and fail the first test below for the wrong reason. */
function direction(seed) {
  const raw = new Float64Array(DIMS);
  let x = seed;
  for (let d = 0; d < DIMS; d += 1) {
    x = (Math.imul(x, 1103515245) + 12345) >>> 0;
    raw[d] = ((x >>> 16) % 255) - 127;
  }
  const norm = Math.hypot(...raw);
  return Int8Array.from(raw, (v) => Math.round((v / norm) * 127));
}
const alpha = direction(1);
const omega = direction(2);
const blend = Int8Array.from(alpha, (a, d) => Math.round((a + omega[d]) / 2));
const FIXED = { alpha, omega, blend };
const cos = (a, b) => {
  let dot = 0, na = 0, nb = 0;
  for (let d = 0; d < a.length; d += 1) { dot += a[d] * b[d]; na += a[d] * a[d]; nb += b[d] * b[d]; }
  return dot / Math.sqrt(na * nb);
};
// How correlated the two directions already are. It bounds what subtracting one
// from the other can achieve: the projected query is re-quantised to int8 at
// scale 127, which re-introduces about this much of the negative. Removal is
// therefore "gone down to the query's own precision", not "gone".
const floorResidual = Math.abs(cos(alpha, omega)) + 0.02;

/** Embed `text` against the fixed directions, or return the refusal's message. */
async function embedFixed(text) {
  const encoder = stubEncoder(FIXED);
  try {
    return await embedParsedQuery(parseQuery(text), DIMS);
  } catch (error) {
    return error.message;
  } finally {
    encoder.restore();
  }
}

test("one positive text round-trips to its own vector", async () => {
  assert.ok(cos(await embedFixed("alpha"), alpha) > 0.999);
});

test("\"|\" lands between its alternatives, not on either", async () => {
  const summed = await embedFixed("(alpha | omega)");
  assert.ok(cos(summed, alpha) > 0 && cos(summed, omega) > 0);
  assert.ok(Math.abs(cos(summed, alpha) - cos(summed, omega)) < 0.02,
            `${cos(summed, alpha)} vs ${cos(summed, omega)}`);
});

test("subtraction removes the negative direction, and only that", async () => {
  assert.ok(cos(blend, alpha) > 0.5 && cos(blend, omega) > 0.5,
            "the test query is not really about both halves");
  const minus = await embedFixed("blend -(omega)");
  assert.ok(Math.abs(cos(minus, omega)) < floorResidual,
            `${cos(minus, omega)} against a floor of ${floorResidual}`);
  // What is left is MORE about the other half than the question was...
  assert.ok(cos(minus, alpha) > cos(blend, alpha));
  // ...without ranking the OPPOSITE of the negative, which plain q - n would do.
  assert.ok(cos(minus, omega) > -floorResidual);
});

test("subtracting the question itself refuses rather than ranking noise", async () => {
  assert.equal(await embedFixed("alpha -(alpha)"), "cancelled");
});

test("exactly the term budget is allowed (10)", async () => {
  // Repeated on purpose: parseQuery does not deduplicate, and it is the COUNT that
  // is under test. One more is refused, see "a query with too many terms" above.
  const result = await embedFixed("(alpha|omega|blend|alpha|omega|blend|alpha) -(omega|blend|alpha)");
  assert.ok(result instanceof Int8Array, result);
});

test("a question longer than the cap refuses, and one exactly at it does not", async () => {
  // The other half of the term budget: one pasted chapter is a single term and
  // would pass every count, so length is capped as well as arity. The encoder is
  // shared with justelesRCP, where a long input costs quadratic attention.
  assert.equal(await embedFixed("a".repeat(513)), "long");
  assert.ok((await embedFixed("a".repeat(512))) instanceof Int8Array);
});

test("a query with nothing but a subtraction refuses", async () => {
  assert.equal(await embedFixed("-(alpha)"), "empty");
});

test("the BM25 blend only ever adds, and the lexical part stays normalised", () => {
  const sorted = rescore([
    { cosine: 0.60, text: "la prise en charge du reseau de chaleur chez l'habitant jeune" },
    { cosine: 0.58, text: "le calcaire impose une dureté mesurée tous les trois mois, et la dureté guide le dosage" },
    { cosine: 0.56, text: "les pompes à chaleur de seconde génération et leurs effets thermiques" },
  ], "surveillance de la dureté");
  assert.match(sorted[0].text, /dureté/, "the passage naming the word did not overtake");
  assert.ok(sorted.every((r) => r.score >= r.cosine && r.lexical >= 0 && r.lexical <= 1));
});

test("a candidate with no attached text is kept, not dropped", () => {
  const kept = rescore([{ cosine: 0.6, text: undefined }, { cosine: 0.5, text: "dureté" }], "dureté");
  assert.equal(kept.length, 2);
});
