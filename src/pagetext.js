/* A PDF page as plain text, with the passage marked: what the viewer's "read as text"
   mode shows and what "copy the passage" copies on a document whose text is not
   published.

   The text comes from pdf.js's own text content of the page (the same runs its
   selectable text layer is built from), so it is the document's words in the
   document's reading order, not the indexed chunk text. Which runs belong to the
   passage is decided geometrically, from the boxes chunk.py recorded: the same boxes
   the highlight is drawn from, so the marked text is exactly what the coloured
   highlight covers on the image.

   Pure functions, no DOM and no pdf.js import, so tests/pagetext.test.mjs can run
   them under Node. */

/**
 * Whether a pdf.js text item sits inside one of the passage's boxes.
 *
 * The test point is a little right of the run's start and a third of the way up from
 * its baseline: a run's left edge can touch the box of the line before or after it,
 * and its baseline is at the bottom of its glyphs, so the corner is the one place
 * that is NOT a fair sample of where the run is.
 *
 * @param {{transform: number[], width?: number, height?: number}} item - A pdf.js
 *   text item; `transform[4]`, `transform[5]` are its baseline origin in PDF user
 *   space (origin bottom-left, y up).
 * @param {number[][]} boxes - Boxes in PDF user space as [x0, yLow, x1, yHigh].
 * @returns {boolean}
 */
export function inBoxes(item, boxes) {
  const width = Math.abs(item.width || 0);
  const height = Math.abs(item.height || 0);
  const x = item.transform[4] + Math.min(width / 2, 2);
  const y = item.transform[5] + height / 3;
  const slack = 1;
  return boxes.some(([x0, y0, x1, y1]) =>
    x >= x0 - slack && x <= x1 + slack && y >= y0 - slack && y <= y1 + slack);
}

/**
 * Convert chunk.py's boxes to PDF user space.
 *
 * chunk.py stores what PyMuPDF reports: points from the TOP-LEFT of the page's visible
 * box, y growing downward. pdf.js text items are in user space: origin at the
 * bottom-left of the view box, y growing upward. The same flip viewer.js does before
 * drawing a highlight.
 *
 * @param {number[][]|undefined} boxes - [x0, y0, x1, y1] per line, top-left origin.
 * @param {number[]} viewBox - The page's view box, [x0, y0, x1, y1] in user space.
 * @returns {number[][]} [x0, yLow, x1, yHigh] per box.
 */
export function toUserSpace(boxes, viewBox) {
  const [vx0, , , vy1] = viewBox;
  return (boxes || []).filter((b) => b && b.length === 4)
    .map(([x0, y0, x1, y1]) => [vx0 + x0, vy1 - y1, vx0 + x1, vy1 - y0]);
}

/**
 * The page's text as runs, each marked as in the passage or not.
 *
 * Adjacent items with the same mark are merged, so the caller writes one <mark> per
 * stretch of passage rather than one per pdf.js item (which can be a single glyph).
 * A line break pdf.js reports (`hasEOL`) is kept as "\n", which `white-space:
 * pre-wrap` turns back into the page's own lines.
 *
 * @param {Array<{str?: string, hasEOL?: boolean, transform?: number[]}>} items -
 *   pdf.js `getTextContent().items`; marked-content entries (no `str`) are skipped.
 * @param {number[][]} userBoxes - From `toUserSpace`.
 * @returns {Array<{text: string, hit: boolean}>}
 */
export function textRuns(items, userBoxes) {
  const runs = [];
  for (const item of items) {
    if (typeof item.str !== "string") continue;
    const text = item.str + (item.hasEOL ? "\n" : "");
    if (!text) continue;
    // An empty run (a bare line break) takes the mark of what precedes it, so a
    // passage's own line breaks stay inside its <mark> instead of splitting it.
    const hit = item.str.trim() === "" && runs.length
      ? runs[runs.length - 1].hit
      : inBoxes(item, userBoxes);
    const last = runs[runs.length - 1];
    if (last && last.hit === hit) last.text += text;
    else runs.push({ text, hit });
  }
  return runs;
}

/**
 * Only the passage's words from a list of runs, as one string.
 *
 * @param {Array<{text: string, hit: boolean}>} runs - From `textRuns`.
 * @returns {string}
 */
export function passageOf(runs) {
  return runs.filter((r) => r.hit).map((r) => r.text).join(" ")
    .replace(/[ \t]+\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
}
