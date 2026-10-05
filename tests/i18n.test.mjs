/* The shipped string table, checked for the two ways a bilingual site goes wrong.
 *
 * One: a key added to one language and not the other. `t()` falls back to the
 * other language, so the page keeps working and a French reader is quietly shown
 * an English sentence. Nothing in a browser looks broken.
 *
 * Two: a renamed placeholder. The footer's credit and suggestion sentences are
 * assembled by splitting a translated string on its {placeholders} and dropping DOM
 * nodes into the gaps, so a placeholder spelled differently in one language does
 * not throw: it silently renders that sentence without its links. That is a dead
 * link in the footer of one language only, which is exactly the kind of thing a
 * human reviewer skims past.
 *
 * Run by `uv run tests/run.py`, which is what .githooks/pre-push runs.
 *
 * Written by Claude Code (Opus 5).
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { installDocumentStub } from "../server/lib/browser_shim.mjs";

installDocumentStub();

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const i18n = await import(`${root}/src/i18n.js`);

/* The module exports lookups, not the tables, so the tables are read out of the
 * source. Parsing the shipped file is the point: a test that re-declared the keys
 * would pass while the file it describes drifted. */
function keysOf(language) {
  const text = readFileSync(`${root}/src/i18n.js`, "utf8");
  const start = text.indexOf(`\n  ${language}: {\n`);
  assert.ok(start > 0, `no ${language} table in src/i18n.js`);
  const end = text.indexOf("\n  },", start);
  const body = text.slice(start, end);
  return new Map(
    [...body.matchAll(/^    ([a-z_0-9]+): "((?:[^"\\]|\\.)*)"/gm)]
      .map((match) => [match[1], match[2]]),
  );
}

const fr = keysOf("fr");
const en = keysOf("en");

test("both languages carry the same keys", () => {
  assert.ok(fr.size > 100, `only ${fr.size} French keys found`);
  const onlyFr = [...fr.keys()].filter((key) => !en.has(key));
  const onlyEn = [...en.keys()].filter((key) => !fr.has(key));
  assert.deepEqual(onlyFr, [], "keys with no English translation");
  assert.deepEqual(onlyEn, [], "keys with no French translation");
});

test("a string's placeholders are the same in both languages", () => {
  const holes = (value) => [...value.matchAll(/\{([a-z_]+)\}/g)].map((m) => m[1]).sort();
  for (const [key, value] of fr) {
    assert.deepEqual(holes(value), holes(en.get(key)),
                     `${key}: the two languages take different placeholders`);
  }
});

test("every placeholder is filled by somebody", () => {
  /* A placeholder is filled by a caller passing an object whose key matches, so a
   * typo in either half renders literal braces on the page. Rather than keep a list
   * here (which would go stale the day a string gains an argument), the fill sites
   * are read out of src/: the placeholder has to appear as an object key somewhere
   * in the code that calls t() or fillTemplate(). */
  const code = readdirSync(`${root}/src`)
    .filter((name) => name.endsWith(".js"))
    .map((name) => readFileSync(`${root}/src/${name}`, "utf8"))
    .join("\n");
  const filled = new Set([...code.matchAll(/\b([a-z_]+):/g)].map((m) => m[1]));
  for (const [language, table] of [["fr", fr], ["en", en]]) {
    for (const [key, value] of table) {
      for (const [, hole] of value.matchAll(/\{([a-z_]+)\}/g)) {
        assert.ok(filled.has(hole),
                  `${language}.${key}: nothing in src/ fills {${hole}}`);
      }
    }
  }
});

test("every key the pages ask for exists", () => {
  /* A data-i18n naming a key that is not in the table renders as empty text: the
   * element is there, styled, and says nothing. That is invisible in a diff and
   * nearly invisible in a browser, which is why it is checked from the HTML rather
   * than trusted to review. */
  const pages = readdirSync(`${root}/src`).filter((name) => name.endsWith(".html"));
  assert.ok(pages.length >= 2, `only ${pages.length} pages found`);
  for (const name of pages) {
    const html = readFileSync(`${root}/src/${name}`, "utf8");
    for (const [, key] of html.matchAll(/data-i18n="([a-z_0-9]+)"/g)) {
      assert.ok(fr.has(key), `${name}: data-i18n="${key}" has no French string`);
      assert.ok(en.has(key), `${name}: data-i18n="${key}" has no English string`);
    }
    // The attribute form, "placeholder:search_placeholder,aria-label:search_label".
    for (const [, spec] of html.matchAll(/data-i18n-attr="([^"]+)"/g)) {
      for (const pair of spec.split(",")) {
        const key = pair.split(":")[1]?.trim();
        assert.ok(key && fr.has(key) && en.has(key),
                  `${name}: data-i18n-attr key "${key}" is missing from a dictionary`);
      }
    }
  }
});

test("t() substitutes and falls back", () => {
  i18n.setLang("fr");
  assert.equal(i18n.lang(), "fr");
  assert.match(i18n.t("changelog_version", { v: "0.2.0" }), /0\.2\.0/);
  assert.equal(i18n.t("no_such_key_anywhere"), "no_such_key_anywhere");
});

test("plural agrees with the count", () => {
  i18n.setLang("fr");
  assert.match(i18n.t("ago_days", { n: 1, s: i18n.plural(1) }), /^1 jour$/);
  assert.match(i18n.t("ago_days", { n: 3, s: i18n.plural(3) }), /^3 jours$/);
});

test("the language toggle is a round trip", () => {
  i18n.setLang("en");
  assert.equal(i18n.lang(), "en");
  assert.equal(i18n.otherLang(), "fr");
  i18n.setLang("fr");
  assert.equal(i18n.lang(), "fr");
});

test("a document's identifier becomes a link, and a cell that is not one does not", () => {
  /* identifierLink() is the one place the site turns a hand-curated TSV cell into a
   * URL, so it validates rather than trusts. NO_IDENTIFIER ("-") is a real value in
   * data/MANIFEST.tsv meaning "the extractor looked and found none", and it is what
   * most of the corpus carries: rendered as a link it would open doi.org on nothing,
   * on 460 of the 534 documents. */
  i18n.setLang("en");
  const doi = i18n.identifierLink({ doi: "10.1001/jama.2023.0589", isbn: "" });
  assert.equal(doi.href, "https://doi.org/10.1001/jama.2023.0589");
  assert.equal(doi.textContent, "DOI");
  assert.equal(doi.rel, "noopener noreferrer");
  assert.equal(doi.target, "_blank");

  const isbn = i18n.identifierLink({ doi: "-", isbn: "978-2-11-128504-0" });
  assert.equal(isbn.href, "https://search.worldcat.org/search?q=bn%3A9782111285040");
  assert.equal(isbn.textContent, "ISBN");

  // The three documents that print both are journal articles bound into a book: the
  // DOI resolves to the work, the ISBN only to a catalogue record of the volume.
  assert.equal(i18n.identifierLink({ doi: "10.3917/psye.521.0089", isbn: "9782130572787" }).textContent,
               "DOI");

  for (const doc of [null, {}, { doi: "-", isbn: "-" }, { doi: "", isbn: "" },
                     { doi: "10.1001", isbn: "" }, { doi: "", isbn: "12345" },
                     { doi: "see the cover", isbn: "n/a" }]) {
    assert.equal(i18n.identifierLink(doc), null, `${JSON.stringify(doc)} is not a link`);
  }
});

test("the identifier link says what it opens, in both languages", () => {
  /* The label is the same word in French and English ("DOI"), so the tooltip is the
   * only translated part, and a missing one would be invisible: the link still
   * renders and still works. */
  for (const language of ["fr", "en"]) {
    i18n.setLang(language);
    const link = i18n.identifierLink({ doi: "10.1001/jama.2023.0589" });
    assert.ok(link.title.length > 10 && !link.title.includes("_"),
              `${language}: doi_title is not translated (${link.title})`);
  }
  i18n.setLang("fr");
});

test("docRefLinks puts the document's own identifier before the publisher link", () => {
  const both = i18n.docRefLinks({ doi: "10.1001/jama.2023.0589", source_url: "https://example.org/x" });
  assert.deepEqual(both.map((link) => link.className), ["doc-id", "src-ai"]);
  assert.deepEqual(i18n.docRefLinks({ doi: "-", source_url: "" }), []);
});

test("sourceLink only links an http(s) URL", () => {
  // The manifest is hand-curated: a `javascript:` or relative cell must not become a
  // live link, even though the CSP would block the first one anyway.
  for (const url of ["javascript:alert(1)", "pdf/x.pdf", "www.has-sante.fr", "", "  "]) {
    assert.equal(i18n.sourceLink({ source_url: url }), null, `${url} is not a link`);
  }
  const link = i18n.sourceLink({ source_url: " https://www.has-sante.fr/x " });
  assert.equal(link.href, "https://www.has-sante.fr/x");
  assert.equal(link.rel, "noopener noreferrer");
});
