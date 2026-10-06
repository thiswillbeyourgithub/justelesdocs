/* src/pagetext.js: the page as text with the passage marked, which the viewer's text
 * mode shows and its copy button copies for a document whose text is not published.
 *
 * Written by Claude Code (Opus 5.5).
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const { inBoxes, toUserSpace, textRuns, passageOf } = await import(`${root}/src/pagetext.js`);

// A 600 x 800 page. A pdf.js item's transform[4..5] is its baseline origin, y up.
const item = (str, x, y, { eol = false, width = 50, height = 10 } = {}) =>
  ({ str, hasEOL: eol, width, height, transform: [10, 0, 0, 10, x, y] });

test("chunk.py's top-left boxes are flipped into pdf.js user space", () => {
  // A line 100 to 112 points from the TOP of an 800-point page is 688 to 700 from the bottom.
  assert.deepEqual(toUserSpace([[50, 100, 300, 112]], [0, 0, 600, 800]), [[50, 688, 300, 700]]);
  assert.deepEqual(toUserSpace([[50, 100, 300, 112]], [20, 0, 620, 800]), [[70, 688, 320, 700]],
    "a cropped page's left edge shifts x");
  assert.deepEqual(toUserSpace(undefined, [0, 0, 600, 800]), []);
});

test("an item is in the passage when its body sits in a box", () => {
  const boxes = [[50, 688, 300, 700]];
  assert.ok(inBoxes(item("dose", 60, 690), boxes));
  assert.ok(!inBoxes(item("header", 60, 750), boxes));
});

test("the page comes back as runs, the passage merged into one marked stretch", () => {
  const boxes = toUserSpace([[50, 100, 300, 130]], [0, 0, 600, 800]);
  const runs = textRuns([
    item("Titre", 60, 750, { eol: true }),
    item("Lithium", 60, 690), item(" 400 mg", 120, 690, { eol: true }),
    item("", 60, 680, { eol: true }),           // a bare line break inside the passage
    item("le soir", 60, 675, { eol: true }),
    item("Note", 60, 300),
    { type: "beginMarkedContent" },              // not text: skipped
  ], boxes);
  assert.deepEqual(runs, [
    { text: "Titre\n", hit: false },
    { text: "Lithium 400 mg\n\nle soir\n", hit: true },
    { text: "Note", hit: false },
  ]);
  assert.equal(passageOf(runs), "Lithium 400 mg\n\nle soir");
});

test("a page with no box marks nothing", () => {
  const runs = textRuns([item("a", 0, 0), item("b", 10, 0)], []);
  assert.deepEqual(runs, [{ text: "ab", hit: false }]);
  assert.equal(passageOf(runs), "");
});
