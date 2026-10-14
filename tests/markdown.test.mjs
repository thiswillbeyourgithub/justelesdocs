/* server/markdown.mjs: a search answer written for an AI assistant.
 *
 * Written by Claude Code (Opus 5.5).
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const { renderSearch, originOf, metadataLine } = await import(`${root}/server/markdown.mjs`);
const { apiDocHref } = await import(`${root}/src/paths.js`);

const index = { docs: [
  { file: "Open doc.pdf", title: "Open", issuer: "HAS", year: "2020", topic: "a;b", access: "open" },
  { file: "shut.pdf", title: "Shut", issuer: "", year: "2019", access: "restricted" },
] };
const result = (doc, extra = {}) => ({ doc, chunk: 4, page: 3, pages: [3], score: 0.6, more: 0, text: "line one\n\nline two", ...extra });

test("the api/doc path drops the extension and escapes the name (tests/test_stage.py pins the same)", () => {
  assert.equal(apiDocHref("Open doc.pdf"), "api/doc/Open%20doc");
  assert.equal(apiDocHref("Open doc.pdf", 3), "api/doc/Open%20doc/p/3");
});

test("an open passage links to its page text; a restricted one does not", () => {
  const md = renderSearch({ question: "q", answer: { results: [result(0), result(1)] }, index,
    fields: ["issuer", "year", "topic"], origin: "https://x.org", query: "q=q" });
  assert.match(md, /Page text: https:\/\/x\.org\/api\/doc\/Open%20doc\/p\/3/);
  assert.equal(md.match(/Page text:/g).length, 1);
  assert.match(md, /Restricted document/);
  // A blank line inside the passage does not end the quote.
  assert.match(md, /> line one\n>\n> line two/);
});

test("metadata prints the corpus's facets in order, lists joined, blanks left out", () => {
  assert.equal(metadataLine(index.docs[0], ["issuer", "year", "topic"]), "issuer: HAS · year: 2020 · topic: a, b");
  assert.equal(metadataLine(index.docs[1], ["issuer", "year"]), "year: 2019");
});

test("the origin comes from a plausible Host header only", () => {
  assert.equal(originOf("docs.example.org"), "https://docs.example.org");
  assert.equal(originOf("127.0.0.1:8649"), "http://127.0.0.1:8649");
  assert.equal(originOf("evil.org/<script>"), "");
  assert.equal(originOf(undefined), "");
});
