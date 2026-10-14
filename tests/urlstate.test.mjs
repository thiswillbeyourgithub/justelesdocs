/* The search URL contract in src/urlstate.js, which the page and the search
 * service's Markdown endpoint both read: a link has to mean the same question,
 * filters and order to each of them.
 *
 * Written by Claude Code (Opus 5.5).
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const { readSearchParams, applyLevel, LEVEL_ALL } = await import(`${root}/src/urlstate.js`);

const FIELDS = { facets: ["issuer", "topic", "year"], ranges: new Set(["year"]) };
const read = (query) => readSearchParams(new URLSearchParams(query), FIELDS);

test("q, its alias, repeated facets, a range and the scope field", () => {
  const got = read("query=%20lithium%20&topic=a&topic=b&issuer=HAS&year=2016-&file=x.pdf&unknown=1");
  assert.equal(got.question, "lithium");
  assert.deepEqual(got.filters, { issuer: ["HAS"], topic: ["a", "b"], year: { from: 2016 }, file: "x.pdf" });
});

test("q wins over its alias", () => {
  assert.equal(read("q=one&query=two").question, "one");
});

test("an empty first value does not hide the rest of a facet list", () => {
  assert.deepEqual(read("topic=&topic=b").filters, { topic: ["b"] });
});

test("ranking choices read both boolean spellings and nothing else", () => {
  assert.deepEqual(read("bm25=1&norefs=0&recent=yes&figures=only").prefs,
    { rescore: true, norefs: false, recent: null, figures: "only" });
  assert.equal(read("figures=nonsense").prefs.figures, null);
});

test("level, its old spelling, and silence", () => {
  assert.equal(read("level=strict").level, "strict");
  assert.equal(read("guideline=family").level, "family");
  assert.equal(read("").level, null);
});

test("applyLevel removes the field for the whole corpus", () => {
  const filters = { guideline: "strict" };
  applyLevel(filters, "guideline", LEVEL_ALL);
  assert.deepEqual(filters, {});
  applyLevel(filters, "guideline", "family");
  assert.deepEqual(filters, { guideline: "family" });
});
