/* The URL helpers in src/paths.js: what a link says, read back the way it was meant.
 *
 * A link is a promise made to someone who is not here yet (a bookmark, a shared
 * search), so a wrong parse fails silently and far from the code that caused it: the
 * reader just sees an empty list.
 *
 * Written by Claude Code (Opus 5.5).
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const { parseRange } = await import(`${root}/src/paths.js`);

test("a closed year range keeps both bounds", () => {
  assert.deepEqual(parseRange("2016-2020"), { from: 2016, to: 2020 });
});

test("a range open at the top has no upper bound, not an upper bound of 0", () => {
  // Number("") is 0: "2016-" used to come back as {from: 2016, to: 0}, which no
  // document can satisfy.
  assert.deepEqual(parseRange("2016-"), { from: 2016 });
});

test("a range open at the bottom has no lower bound", () => {
  assert.deepEqual(parseRange("-2020"), { to: 2020 });
});

test("a range with neither bound is no range at all", () => {
  assert.equal(parseRange("-"), null);
  assert.equal(parseRange("abc-def"), null);
});

/* Viewer links. Three documents is enough to show an id that has moved. */
const { viewHref, viewerDocId, localChunkOf, openingPage } = await import(`${root}/src/paths.js`);
const DOCS = [{ file: "a.pdf" }, { file: "b é.pdf" }, { file: "c.pdf" }];

test("a viewer link carries the file name and the page, encoded", () => {
  const href = viewHref(1, "b é.pdf", { chunk: 42, page: 7, back: "q=x&year=2016-" });
  const url = new URL(href, "https://site.test/");
  assert.equal(url.pathname, "/view.html");
  assert.equal(url.searchParams.get("doc"), "1");
  assert.equal(url.searchParams.get("file"), "b é.pdf");
  assert.equal(url.searchParams.get("chunk"), "42");
  assert.equal(url.searchParams.get("p"), "7");
  assert.equal(url.searchParams.get("back"), "q=x&year=2016-");
});

test("a link with no passage names no chunk and no page", () => {
  assert.equal(viewHref(2, "c.pdf"), "view.html?doc=2&file=c.pdf");
});

test("the file name overrules an id that has moved", () => {
  // A document added before "b é.pdf" shifted it from id 0 to id 1.
  assert.equal(viewerDocId("0", "b é.pdf", DOCS), 1);
  assert.equal(viewerDocId("1", "b é.pdf", DOCS), 1);
});

test("a file no longer in the index is not found, whatever the id says", () => {
  assert.equal(viewerDocId("0", "gone.pdf", DOCS), null);
});

test("a link made before `file` existed is taken at its id", () => {
  assert.equal(viewerDocId("2", null, DOCS), 2);
  assert.equal(viewerDocId("3", null, DOCS), null);
});

test("a link with no document opens nothing, not document 0", () => {
  // Number(null) is 0, which is how a bare view.html used to open the first document.
  assert.equal(viewerDocId(null, null, DOCS), null);
  assert.equal(viewerDocId("", null, DOCS), null);
});

test("a chunk outside the document highlights nothing", () => {
  assert.equal(localChunkOf(105, 100, 10), 5);
  // Used to clamp to the document's first chunk and highlight it as the match.
  assert.equal(localChunkOf(95, 100, 10), -1);
  assert.equal(localChunkOf(110, 100, 10), -1);
  assert.equal(localChunkOf(null, 100, 10), -1);
});

test("the viewer opens on the page the link names", () => {
  const match = { pages: [3, 4], boxes: { 3: [[0, 0, 1, 1]], 4: [[0, 0, 1, 1], [0, 2, 1, 3]] } };
  assert.equal(openingPage("3", match, 10), 3);
  // No `p`, or one that is not a page of the passage: the boxes rule, as before.
  assert.equal(openingPage(null, match, 10), 4);
  assert.equal(openingPage("9", match, 10), 4);
  // No passage: the named page if the document has it, else the first.
  assert.equal(openingPage("5", undefined, 10), 5);
  assert.equal(openingPage("11", undefined, 10), 1);
  assert.equal(openingPage(null, undefined, 10), 1);
});
