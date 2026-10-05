/* The hit view: one PDF page, rendered with pdf.js, with the matching passage
   highlighted where it actually sits on the page.

   Why the real PDF and not a converted page. The corpus is documents full of
   tables, algorithms, grading matrices and figures. Reflowing them into HTML would
   lose exactly the parts a specialist reader needs, and it would also break the promise
   that what is on screen is the document as its authors published it. So the page
   is rendered as the PDF, and the search result is drawn ON it.

   How a highlight lands in the right place. chunk.py records, per chunk and per
   page, the bounding boxes of the text rows it took, in PDF points from the page's
   own coordinate system. pdf.js gives a viewport that maps that system to pixels
   at a chosen scale. Rather than convert to pixels, each box is expressed as a
   PERCENTAGE of the page box: percentages survive a resize, a zoom and a
   device-pixel-ratio change without recomputation, so the highlight cannot drift
   from the words under it.

   The PDF served is the same file the boxes were computed from. For the 14
   documents with scanned pages that is the OCR'd derivative, not the original;
   scripts/stage.py resolves that from what chunk.py recorded, which is why the
   coordinates can be trusted at all. */

import { load, save } from "./store.js";
import { t, apply, docRefLinks, pdfMenu, isRestricted, tapNote, figureNote } from "./i18n.js";
import { setStatus } from "./site.js";
import { pdfHref, pageHref, okJson, viewerDocId, localChunkOf, openingPage } from "./paths.js";

const els = {
  canvas: document.getElementById("canvas"),
  layer: document.getElementById("layer"),
  text: document.getElementById("textlayer"),
  frame: document.getElementById("page-frame"),
  stage: document.getElementById("stage"),
  scroll: document.getElementById("scroll"),
  title: document.getElementById("doc-title"),
  meta: document.getElementById("doc-meta"),
  pageno: document.getElementById("pageno"),
  prev: document.getElementById("prev"),
  next: document.getElementById("next"),
  pdfSlot: document.getElementById("pdf-slot"),
  docSearch: document.getElementById("doc-search"),
  docSearchForm: document.getElementById("doc-search-form"),
  docQ: document.getElementById("doc-q"),
  scopeIssuerRow: document.getElementById("scope-issuer-row"),
  scopeIssuerLabel: document.getElementById("scope-issuer-label"),
  back: document.getElementById("back"),
  hlBtn: document.getElementById("hl-btn"),
  chunkText: document.getElementById("chunk-text"),
  chunkTextBody: document.getElementById("chunk-text-body"),
};

const params = new URL(location.href).searchParams;
// Resolved in `boot` against the index, since the `file` parameter can overrule the
// id (paths.js: viewerDocId). Null until then.
let docId = null;
// Absent on a link that names a document and no passage (the browse page's link to a
// restricted document), which is null here rather than NaN, and opens on page 1.
const chunkRow = params.has("chunk") ? Number(params.get("chunk")) : null;

/**
 * Where "retour aux résultats" goes.
 *
 * The search page hands us its own query string in `back`, so the reader lands on
 * the same question with the same filters rather than on a bare corpus listing.
 * Rebuilt through the URL API with only `search` assigned: `back` arrives from the
 * address bar, and setting `.search` on a URL whose path is already fixed cannot
 * move it to another path, another origin or another scheme, whatever it contains.
 *
 * Falling back to document.referrer covers a hit reached from somewhere that did not
 * set the parameter, and "./" covers a bookmarked viewer URL, which has neither.
 *
 * @returns {string} an href on this origin
 */
function backHref() {
  const back = params.get("back");
  if (back) {
    const url = new URL("./", location.href);
    url.search = back;
    return url.href;
  }
  try {
    const ref = new URL(document.referrer);
    // Same origin and the search page itself, not another viewer page: stepping from
    // one hit to another and then "back" should return to the list, not to the hit.
    if (ref.origin === location.origin && !ref.pathname.endsWith("/view.html")) return ref.href;
  } catch { /* no referrer, or not a URL: fall through */ }
  return "./";
}

let pdfjs = null;        // the pdf.js module, kept for its TextLayer
let pdf = null;          // pdf.js document
let meta = null;         // index meta.json
let doc = null;          // this document's entry
let chunks = null;       // this document's chunk text/pages/boxes
let localChunk = -1;     // the matching chunk's index within the document, -1 for none
let page = 1;
let renderToken = 0;     // guards against out-of-order renders
// The pdf.js render drawing on the canvas right now, if any. pdf.js refuses a second
// render() on a canvas that one is still drawing on ("Cannot use the same canvas
// during multiple render() operations"), and the token alone only drops a stale
// render's RESULT, it does not stop the drawing. So a new render cancels this first.
let renderTask = null;
// Which page `pdf` currently holds, for a restricted document only. Undefined for an
// open one, where `pdf` holds them all.
let loadedPage = 0;
// How many pages the DOCUMENT has, which is not always how many `pdf` holds. For a
// restricted document `pdf` is a single page cut out of the file by the page service,
// so `pdf.numPages` is 1 no matter where the reader is; the real count comes from the
// index, which took it from the file at build time.
let pageCount = 1;
// The page the viewer opened on, which is the passage's page. It is the centre of the
// window a restricted document may be read through, so it is remembered rather than
// recomputed: `page` moves as the reader steps, and the window must not move with it.
let opened = 1;

/**
 * The first and last page this reader may reach from here.
 *
 * A restricted document is one the site may show but not redistribute, so it is read
 * one page either side of the passage and no further: enough to see a sentence that
 * runs over a page break, not enough to walk through the book three pages at a time.
 * Every other document is bounded only by its own length.
 *
 * @returns {{first: number, last: number}} Inclusive page bounds, 1-based.
 */
function pageWindow() {
  const last = pageCount;
  if (!isRestricted(doc)) return { first: 1, last };
  return { first: Math.max(1, opened - 1), last: Math.min(last, opened + 1) };
}

/* Click the page to fill the screen with it, click again to come back.
 *
 * ONE state, not a ladder of zoom levels. A click cannot express "how much", so a
 * ladder makes the reader click through sizes they did not ask for to get back to the
 * one they had, and leaves them wondering which rung they are on. There are only two
 * things anyone wants here: the page next to the rest of the interface, or the page
 * and nothing else.
 *
 * The expanded state is an overlay filling the viewport, and the page is re-rendered
 * to the width that overlay gives it, which on any real screen is more than the 1000
 * CSS px the inline view caps itself at. Native fullscreen is requested on top of it
 * when the browser has it, purely to win back the browser chrome's pixels: the overlay
 * is what does the work, so a refusal (iOS Safari has no element fullscreen) changes
 * nothing the reader can see. */
let expanded = false;

/* How far the reader has pinched the expanded page, as a multiple of its width at
 * fit-to-width. 1 on every expand. Continuous rather than a ladder, which is what
 * the one-state rule above is against: a pinch says "how much" by itself.
 *
 * Handled here rather than left to the browser's own pinch, because the browser's
 * pinch only magnifies the bitmap already drawn: the page stays at the resolution it
 * was rendered for, so zooming in on a small footnote showed bigger blurred pixels
 * instead of the footnote. Here the page is re-rendered at the new size once the
 * fingers stop, so the text is as sharp at 5x as at 1x. */
let zoom = 1;
// Out to "the whole page on screen" (clamped per page in `zoomBounds`), in to 6x,
// where a 7pt footnote on a phone is already larger than body text.
const MAX_ZOOM = 6;
// iOS Safari refuses a canvas over about 16.7M pixels and draws NOTHING rather than
// a smaller one, so the resolution gives way before the page does.
const MAX_CANVAS_PIXELS = 16e6;

/**
 * Device pixels per CSS pixel to render the page at.
 *
 * The device pixel ratio, so text is sharp on a HiDPI screen, times the browser's own
 * pinch zoom of the whole page (`visualViewport.scale`), so a reader who pinches the
 * inline page gets a re-render at the magnified size rather than stretched pixels.
 * The old cap at 2 is gone: on a 2.625 phone it made every page a 1.3x upscale, which
 * is exactly the soft text readers reported. The bound is now the canvas area.
 *
 * @param {{width: number, height: number}} cssViewport - The page's size in CSS px.
 * @returns {number} The ratio to render at.
 */
function pixelRatio(cssViewport) {
  const wanted = (window.devicePixelRatio || 1) * (window.visualViewport?.scale || 1);
  const fits = Math.sqrt(MAX_CANVAS_PIXELS / (cssViewport.width * cssViewport.height));
  return Math.min(wanted, fits);
}

/**
 * The zoom range for the page on screen now.
 *
 * The low end is "the whole page fits": on a landscape monitor fit-to-width makes an
 * A4 page three screens tall, and seeing it whole is a reasonable thing to want. On
 * a phone the page already fits by its width, so the low end is 1.
 *
 * @returns {{min: number, max: number}}
 */
function zoomBounds() {
  const { clientWidth, clientHeight } = els.scroll;
  const aspect = els.canvas.width / els.canvas.height || 0.7;
  return { min: Math.min(1, (clientHeight * aspect) / clientWidth), max: MAX_ZOOM };
}

/**
 * Zoom the expanded page to `next`, keeping the point under (`sx`, `sy`) in place.
 *
 * The point is in the scroller's own coordinates: it is where the fingers or the
 * pointer are, and the reader's eye is on whatever was there.
 *
 * @param {number} next - The zoom asked for, clamped to `zoomBounds`.
 * @param {number} sx - Focal x, CSS px from the scroller's left edge.
 * @param {number} sy - Focal y, same from its top.
 * @returns {Promise<void>} Resolves once the page has been re-rendered.
 */
async function zoomTo(next, sx, sy) {
  const { min, max } = zoomBounds();
  const before = els.stage.getBoundingClientRect();
  const view = els.scroll.getBoundingClientRect();
  // The focal point as a share of the page, measured BEFORE the resize.
  const fx = (view.left + sx - before.left) / before.width;
  const fy = (view.top + sy - before.top) / before.height;
  zoom = Math.min(max, Math.max(min, next));
  await safeRender();
  const after = els.stage.getBoundingClientRect();
  els.scroll.scrollLeft += after.left + fx * after.width - (view.left + sx);
  els.scroll.scrollTop += after.top + fy * after.height - (view.top + sy);
}

// Whether the highlight boxes are drawn over the page. On by default: showing where
// the passage sits is the whole reason this view exists rather than a link to the PDF.
// Off is for reading the page as published, or for seeing what a coloured box covers,
// so the choice is remembered across hits: a reader who turned them off is reading,
// and having them come back on the next result would be the annoying half of a toggle.
const HIGHLIGHTS_KEY = "highlights";
let highlights = true;
highlights = load(HIGHLIGHTS_KEY) !== "off";

/**
 * Apply the highlight setting to the page.
 *
 * Only visibility, never a re-render: the boxes are already in the DOM as percentages
 * of the page box, so hiding the layer costs nothing and turning it back on cannot
 * put a box anywhere but where it was.
 */
function applyHighlights() {
  els.layer.hidden = !highlights;
  // Gone entirely while the page fills the screen. It sits ON the page, and expanding
  // is what a reader does to read the page rather than the interface: the one control
  // that covers the words is the last thing wanted then, and the click that got here
  // is one click from undoing itself.
  els.hlBtn.hidden = expanded;
  els.hlBtn.textContent = t(highlights ? "hide_highlights" : "show_highlights");
  els.hlBtn.setAttribute("aria-pressed", String(highlights));
}

/** Frame the matched passage on `pageNumber`, one frame per block of lines. */
function drawBoxes(pageNumber, viewport) {
  els.layer.textContent = "";
  if (!chunks) return;
  const width = viewport.width;
  const height = viewport.height;

  // ONE chunk is drawn: the one the reader clicked. Neighbouring chunks used to be
  // drawn in a second colour, and on a dense page that is most of the page: the
  // passage the search actually found was a purple island in a field of orange, which
  // is the opposite of what a highlight is for. The other chunks are not matches, they
  // are simply the rest of the document, and the rest of the document is already
  // visible: it is the page.
  const boxes = chunks[localChunk]?.boxes?.[String(pageNumber)];
  if (!boxes) return;
  // NOT merged: chunk.py stores one rectangle per LINE, and one fill per line is
  // exactly what the highlight is meant to show. Merging was what the FRAME needed
  // (a frame per line is a ladder down the page, so the lines had to be grouped
  // into the blocks a reader would draw by hand); a fill has the opposite
  // requirement. A merged block is a bounding box, so it covers the ragged right
  // edge of every short line and the indent of every first line, and the reader
  // can no longer tell where the chunk's text actually stops. Per line, the
  // highlight ends where the words end.
  //
  // No padding either, for the same reason: the merge grew each block by 2 points
  // so the frame would sit clear of the glyphs, which is right for a frame and
  // wrong for a fill that is meant to mark the text itself. `src/boxes.js` held
  // that merge and was deleted on 2026-09-22, unimported since the fill landed;
  // `git log -- src/boxes.js` has it if a frame mode is ever wanted back.
  for (const [x0, y0, x1, y1] of boxes.filter((b) => b && b.length === 4)) {
    // Two different coordinate systems meet here, and getting it wrong mirrors
    // every highlight about the middle of the page, which is easy to miss on a
    // dense page because there is text under the box either way.
    //
    // chunk.py stores boxes the way PyMuPDF reports them: PDF points, origin at
    // the TOP-LEFT of the page's visible box, y growing DOWNWARD. pdf.js's
    // convertToViewportPoint expects PDF USER space: origin at the bottom-left
    // of viewBox, y growing UPWARD. So the y has to be flipped into user space
    // first, and the x shifted by the view box's own origin, which is not always
    // zero (a cropped page has a non-zero left edge).
    //
    // The conversion still goes through the viewport rather than scaling by
    // hand, because the viewport is what carries the page's /Rotate.
    const [vx0, , , vy1] = viewport.viewBox;
    const [ax, ay] = viewport.convertToViewportPoint(vx0 + x0, vy1 - y0);
    const [bx, by] = viewport.convertToViewportPoint(vx0 + x1, vy1 - y1);
    const left = Math.min(ax, bx), top = Math.min(ay, by);
    const box = document.createElement("div");
    box.className = "box best";
    // Percentages, not pixels: the canvas is responsive (max-width:100%) so its
    // rendered size is not its bitmap size, and a pixel offset would be wrong
    // on every screen but the one it was computed for.
    box.style.left = `${(left / width) * 100}%`;
    box.style.top = `${(top / height) * 100}%`;
    box.style.width = `${(Math.abs(bx - ax) / width) * 100}%`;
    box.style.height = `${(Math.abs(by - ay) / height) * 100}%`;
    els.layer.append(box);
  }
}

/**
 * Lay pdf.js's invisible, selectable text over the rendered page.
 *
 * The canvas is a picture: without this, a reader cannot copy a dose, a criterion or
 * a reference out of a document whose whole value is its exact wording. pdf.js
 * positions a transparent span per text run, which is the same mechanism its own
 * viewer uses, so selection follows the real reading order rather than pixel
 * neighbourhood.
 *
 * Sized in CSS pixels, not device pixels. The canvas is rendered at the device
 * pixel ratio and scaled down by CSS; the text layer lives in CSS space, so it gets
 * its own viewport at the CSS scale. Using the canvas's viewport here would place
 * every span at twice its proper offset on a HiDPI screen.
 *
 * `--scale-factor` is pdf.js's contract with the stylesheet: every span's font-size
 * is `calc(var(--scale-factor) * Npx)`, so without it the text collapses to nothing
 * and selection silently stops working.
 *
 * Failure is swallowed on purpose. A document with no text layer at all (several of
 * the scanned ones, before OCR) must still render its page.
 */
async function renderText(pdfPage, cssViewport, token) {
  // Cleared now, because the canvas under it already shows the new page, but filled
  // only at the end and only if this render is still the current one. pdf.js builds
  // the spans into a DETACHED div, so an older render finishing late cannot clear or
  // overwrite the layer a newer one wrote: it simply never commits. The size and
  // --scale-factor are committed with the spans for the same reason.
  els.text.textContent = "";
  const built = document.createElement("div");
  try {
    const layer = new pdfjs.TextLayer({
      textContentSource: pdfPage.streamTextContent(),
      container: built,
      viewport: cssViewport,
    });
    await layer.render();
  } catch {
    if (token === renderToken) els.text.textContent = "";
    return;
  }
  if (token !== renderToken) return;
  els.text.style.setProperty("--scale-factor", String(cssViewport.scale));
  els.text.style.width = `${Math.floor(cssViewport.width)}px`;
  els.text.style.height = `${Math.floor(cssViewport.height)}px`;
  els.text.replaceChildren(...built.childNodes);
}

/**
 * The pdf.js document holding `page`, fetching it first if it has to.
 *
 * An open document is loaded once and paged through in the browser. A restricted one
 * cannot be: the file is not served, so each page is a separate one-page PDF cut by
 * `server/pages.py`. That makes turning a page a network round trip for those 53
 * documents, which is the cost of not handing over the file.
 *
 * @param {number} pageNumber 1-based page to display.
 * @param {number} token The render's `renderToken`, to tell a stale download apart.
 * @returns {Promise<object|null>} A pdf.js document whose page 1 (restricted) or page
 *   `pageNumber` (open) is the one to render, or null when a newer render overtook it.
 */
async function documentFor(pageNumber, token) {
  if (!isRestricted(doc)) return pdf;
  if (pdf && loadedPage === pageNumber) return pdf;
  const next = await pdfjs.getDocument({
    url: pageHref(docId, pageNumber),
    standardFontDataUrl: "vendor/pdfjs/standard_fonts/",
  }).promise;
  // A newer render asked for another page while this one was downloading, and it may
  // already hold its own page in `pdf`: replacing it here would destroy the document
  // that render is drawing from. Two quick steps finishing out of order did exactly that.
  if (token !== renderToken) {
    next.destroy();
    return null;
  }
  // Destroyed rather than dropped: each of these holds a worker-side document, and a
  // reader stepping through a chapter would otherwise leave one behind per page.
  if (pdf) pdf.destroy();
  pdf = next;
  loadedPage = pageNumber;
  return pdf;
}

async function render() {
  const token = ++renderToken;
  const holder = await documentFor(page, token);
  if (!holder || token !== renderToken) return;   // a newer render started while fetching
  const pdfPage = await holder.getPage(isRestricted(doc) ? 1 : page);
  // The scroller is what has the room. Capped at 1000 CSS px inline, so a wide window
  // does not render an A4 page at poster size next to a 700px column of interface;
  // when expanded, the page's width times the reader's own zoom (see `zoom`).
  const unscaled = pdfPage.getViewport({ scale: 1 });
  const cssWidth = expanded ? els.scroll.clientWidth * zoom : Math.min(els.scroll.clientWidth, 1000);
  const cssScale = cssWidth / unscaled.width;
  const cssViewport = pdfPage.getViewport({ scale: cssScale });
  const viewport = pdfPage.getViewport({ scale: cssScale * pixelRatio(cssViewport) });
  if (token !== renderToken) return;   // a newer render started; drop this one

  // Cancelled before the canvas is resized: resizing clears it, and a render still
  // drawing would paint the previous page's remainder over the cleared canvas.
  renderTask?.cancel();
  els.canvas.width = Math.floor(viewport.width);
  els.canvas.height = Math.floor(viewport.height);
  els.canvas.style.width = `${Math.floor(cssViewport.width)}px`;
  const task = pdfPage.render({
    canvasContext: els.canvas.getContext("2d", { alpha: false }),
    viewport,
  });
  renderTask = task;
  try {
    await task.promise;
  } catch (error) {
    // Cancelled by a newer render, which is the normal fate of a page the reader
    // stepped past (an arrow key held down, a zoom mid-render): nothing to report.
    if (error?.name === "RenderingCancelledException") return;
    throw error;
  } finally {
    if (renderTask === task) renderTask = null;
  }
  if (token !== renderToken) return;

  drawBoxes(page, viewport);
  await renderText(pdfPage, cssViewport, token);
  if (token !== renderToken) return;
  els.pageno.textContent = `${page} / ${pageCount}`;
  const window_ = pageWindow();
  // Greyed either way, but dead in two different senses. At the ends of a document
  // there is nothing to explain, so the arrow is plainly `disabled`. At the edge of a
  // restricted window there IS something to explain, and a `disabled` button cannot
  // be clicked, so it stays in the tab order with `aria-disabled` and answers the tap
  // with the reason (`tapNote`, below). The click handlers clamp to the window in
  // both cases, so an arrow that looks dead never turns a page.
  const why = isRestricted(doc) ? t("restricted_pages") : "";
  for (const [button, stop] of [[els.prev, page <= window_.first],
                                [els.next, page >= window_.last]]) {
    button.disabled = stop && !why;
    button.setAttribute("aria-disabled", String(stop));
    // The tooltip is on the buttons themselves, since that is where the reader's
    // pointer already is when a greyed arrow surprises them.
    button.title = stop ? why : "";
  }
  setStatus("");
}

/**
 * `render`, with a failure said on the status line rather than lost.
 *
 * Everything after boot calls the render from an event handler that does not wait
 * for it, so an error had nowhere to go: a restricted page refused by the rate limit
 * left the arrow looking dead and the status line blank.
 *
 * @returns {Promise<void>} Never rejects.
 */
async function safeRender() {
  try {
    await render();
  } catch (error) {
    setStatus(t("viewer_failed", { error: error.message || error }), true);
  }
}

/**
 * Fill the screen with the page, or give the page back its place on the page.
 *
 * Without the scroll correction the reader who clicked on a table halfway down lands
 * at the top-left corner of a page too big to navigate, which is worse than not
 * expanding at all. The fractions are where the click was, as a share of the page, so
 * whatever was under the pointer stays under it.
 *
 * @param {boolean} next - The state to move to.
 * @param {number} [fx] - Horizontal position of the click as a fraction of the page.
 * @param {number} [fy] - Vertical position, same.
 * @returns {Promise<void>} Resolves once the page has been re-rendered.
 */
async function setExpanded(next, fx = 0.5, fy = 0.5) {
  expanded = next;
  zoom = 1;
  els.frame.classList.toggle("expanded", expanded);
  // Before the await, so the control disappears with the click that expanded the page
  // rather than a re-render later.
  applyHighlights();
  // Requested, never depended on. Wrapped because a browser that refuses (no user
  // gesture, no permission, no element fullscreen at all) rejects rather than
  // returning false, and the overlay is already doing the job either way.
  try {
    if (expanded) await els.frame.requestFullscreen?.();
    else if (document.fullscreenElement) await document.exitFullscreen?.();
  } catch { /* the overlay alone, then */ }
  await safeRender();
  const { clientWidth, clientHeight, scrollWidth, scrollHeight } = els.scroll;
  els.scroll.scrollLeft = fx * scrollWidth - clientWidth / 2;
  els.scroll.scrollTop = fy * scrollHeight - clientHeight / 2;
}

function refreshChrome() {
  apply();
  if (doc) {
    els.title.textContent = doc.title || doc.file;
    els.meta.textContent = [doc.issuer, doc.country, doc.year, doc.language]
      .filter(Boolean).join(" · ");
    // Where the document can be found off this site belongs on the metadata line
    // rather than in the PDF menu: that menu is about THIS file, and this is about
    // where the document came from. Rebuilt on every repaint, which is also how it
    // follows a language change.
    for (const link of docRefLinks(doc)) {
      if (els.meta.textContent) els.meta.append(" · ");
      els.meta.append(link);
    }
    // Said in full on the metadata line, not only as a tooltip on the greyed arrows: a
    // touch screen has no tooltip, and a reader who cannot step to the next page is
    // owed the reason without having to hunt for it.
    if (isRestricted(doc)) {
      const note = document.createElement("span");
      note.className = "restricted-note";
      note.textContent = t("restricted_pages");
      if (els.meta.textContent) els.meta.append(" · ");
      els.meta.append(note);
    }
    // Rebuilt here rather than filled through data-i18n: the summary carries the
    // document's page count, so it is not known until the index has loaded, and every
    // string in it has to be rewritten on a language change like the rest of the bar.
    // `ids` is what keeps this one menu addressable by scripts/check_ui.mjs.
    els.pdfSlot.replaceChildren(pdfMenu(doc, { ids: true }));
    // The second scope names the issuer, so it cannot come from data-i18n, and a
    // document without one (a journal article, say) must not offer it at all: an
    // empty issuer filter would return nothing and read as a broken search.
    els.scopeIssuerRow.hidden = !doc.issuer;
    if (doc.issuer) els.scopeIssuerLabel.textContent = t("scope_issuer", { issuer: doc.issuer });
  }
  applyHighlights();
}

(async function boot() {
  refreshChrome();
  setStatus(t("viewer_loading"));
  try {
    // pdf.js is an ES module served from this origin; the worker likewise, so the
    // CSP's `worker-src 'self'` covers it with no blob: needed in the common path.
    pdfjs = await import("./vendor/pdfjs/pdf.mjs");
    pdfjs.GlobalWorkerOptions.workerSrc = "vendor/pdfjs/pdf.worker.mjs";

    // The document JSON for the id the link names is fetched alongside meta.json, as
    // before: that id is right for every link made since the last rebuild. Only when
    // the `file` parameter says the document has moved is it fetched again, by the
    // resolved id.
    const named = params.get("doc");
    const guess = /^\d+$/.test(named || "") ? Number(named) : null;
    const early = guess === null ? null : fetch(`index/doc/${guess}.json`).then(okJson);
    early?.catch(() => {});   // unused, and so unobserved, when the id is overruled
    meta = await fetch("index/meta.json").then(okJson);
    docId = viewerDocId(named, params.get("file"), meta.documents);
    if (docId === null) {
      throw new Error(`document ${params.get("file") || named || "?"} is not in the index`);
    }
    doc = meta.documents[docId];
    const docJson = await (docId === guess ? early : fetch(`index/doc/${docId}.json`).then(okJson));
    chunks = docJson.chunks;

    localChunk = localChunkOf(chunkRow, doc.chunk_offset, chunks.length);
    const match = chunks[localChunk];
    // The passage's own text under the page, for a page that renders too faintly or
    // too small to read. Set once: it is the document's text, not a translation, so
    // a language change has nothing to redo here. Restricted documents ship their
    // geometry without their text (stage.py), so `match.text` is undefined there and
    // the disclosure stays hidden rather than opening onto nothing.
    els.chunkText.hidden = !match?.text;
    els.chunkTextBody.textContent = match?.text || "";
    // Except a figure's: its text is a model's description (scripts/figures.py), and
    // the reader is told so before reading it, in the language the page opened in.
    if (match?.text && match.figure) {
      els.chunkTextBody.textContent = `[${figureNote(match.figure)}] ${match.text}`;
    }
    // The page the result list printed, carried in the link, so the list and the
    // viewer agree by construction (paths.js: openingPage).
    page = openingPage(params.get("p"), match, Math.max(1, Number(doc.pages) || 1));
    opened = page;

    els.back.href = backHref();
    refreshChrome();

    // A restricted document is never loaded as a document: `render` asks the page
    // service for the one page it is about to draw. Its length therefore has to come
    // from the index, which read it off the file at build time, because there is no
    // `pdf.numPages` here to ask. An open document is loaded once, as before, and its
    // own count wins over the manifest's if the two ever disagree.
    pageCount = Math.max(1, Number(doc.pages) || 1);
    if (!isRestricted(doc)) {
      pdf = await pdfjs.getDocument({
        url: pdfHref(doc),
        // Without this, a PDF that references a standard-14 face without embedding
        // it renders with wrong metrics. Several older French documents do.
        standardFontDataUrl: "vendor/pdfjs/standard_fonts/",
      }).promise;
      pageCount = pdf.numPages;
    }
    await render();
  } catch (error) {
    setStatus(t("viewer_failed", { error: error.message || error }), true);
  }
})();

/* Click the page to expand it, except when the click was the end of a selection.
 *
 * Both gestures start with a mousedown on the same pixels, so one of them has to
 * yield. Selecting text is the deliberate act and expanding is the casual one, so a
 * click that leaves a selection behind, or that travelled more than a few pixels, is
 * treated as a drag and the page stays as it is. */
let pressAt = null;
els.scroll.addEventListener("pointerdown", (event) => {
  pressAt = { x: event.clientX, y: event.clientY };
});
els.scroll.addEventListener("click", (event) => {
  const from = pressAt;
  pressAt = null;
  if (!pdf) return;
  if (!from || Math.hypot(event.clientX - from.x, event.clientY - from.y) > 4) return;
  if (String(getSelection?.() || "").length) return;
  // A pinch that just ended is not a tap, even when a browser fires a click for it.
  if (performance.now() - pinchEnded < 400) return;
  const box = els.stage.getBoundingClientRect();
  void setExpanded(!expanded,
                   (event.clientX - box.left) / box.width,
                   (event.clientY - box.top) / box.height);
});

/* Pinch, on the expanded page only (the inline page keeps the browser's own pinch,
 * which `pixelRatio` now follows).
 *
 * Two phases, the way pdf.js's own viewer does it: while the fingers move, the page
 * is scaled with a CSS transform, which costs nothing and follows the fingers at
 * frame rate; when they lift, it is re-rendered once at the size they asked for.
 * Rendering on every touchmove would decode the page sixty times a second.
 *
 * Touch events rather than pointer events, because a touchmove can be cancelled
 * (non-passive listener) and that is what stops the browser zooming the whole
 * overlay at the same time. One finger is left alone, so scrolling stays native. */
let pinch = null;
let pinchEnded = 0;
const fingers = (touches) => {
  const [a, b] = touches;
  const view = els.scroll.getBoundingClientRect();
  return {
    dist: Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY),
    sx: (a.clientX + b.clientX) / 2 - view.left,
    sy: (a.clientY + b.clientY) / 2 - view.top,
  };
};
els.scroll.addEventListener("touchstart", (event) => {
  if (!expanded || event.touches.length !== 2) return;
  const start = fingers(event.touches);
  const stage = els.stage.getBoundingClientRect();
  const view = els.scroll.getBoundingClientRect();
  // The transform grows the page about the point between the fingers, so what they
  // are holding stays under them.
  els.stage.style.transformOrigin =
    `${view.left + start.sx - stage.left}px ${view.top + start.sy - stage.top}px`;
  pinch = { ...start, scale: 1 };
}, { passive: true });
els.scroll.addEventListener("touchmove", (event) => {
  if (!pinch || event.touches.length !== 2) return;
  event.preventDefault();
  const { min, max } = zoomBounds();
  // Clamped during the gesture as well, so the page does not stretch past a limit
  // and then snap back when the fingers lift.
  pinch.scale = Math.min(max / zoom, Math.max(min / zoom, fingers(event.touches).dist / pinch.dist));
  els.stage.style.transform = `scale(${pinch.scale})`;
}, { passive: false });
els.scroll.addEventListener("touchend", () => {
  if (!pinch) return;
  const { scale, sx, sy } = pinch;
  pinch = null;
  pinchEnded = performance.now();
  els.stage.style.transform = "";
  void zoomTo(zoom * scale, sx, sy);
});
// A trackpad pinch arrives as a wheel event with ctrlKey set, in every desktop
// browser. Without this it zooms the whole interface, overlay included, which is the
// same stretched bitmap as a phone pinch. Wheel zoom renders per event, debounced.
let wheelTimer = 0;
let wheelZoom = 0;
els.scroll.addEventListener("wheel", (event) => {
  if (!expanded || !event.ctrlKey) return;
  event.preventDefault();
  const view = els.scroll.getBoundingClientRect();
  wheelZoom = (wheelZoom || zoom) * Math.exp(-event.deltaY / 200);
  clearTimeout(wheelTimer);
  wheelTimer = setTimeout(() => {
    const target = wheelZoom;
    wheelZoom = 0;
    void zoomTo(target, event.clientX - view.left, event.clientY - view.top);
  }, 60);
}, { passive: false });

els.hlBtn.addEventListener("click", () => {
  highlights = !highlights;
  save(HIGHLIGHTS_KEY, highlights ? "on" : "off");
  applyHighlights();
});
/* Whether the last click on an arrow turned a page or was refused, which is what the
   note under that arrow is about. Read rather than recomputed, because the note's
   handler runs after the one that moves the page: asking "is this the edge?" there
   would raise the note on the step that ARRIVES at the edge, which is a step that
   worked, rather than on the one that is refused. */
let refused = false;

/**
 * Move by one page, within the window this reader may read.
 *
 * Clamped to the window rather than to the document, so the keyboard cannot walk past
 * what the buttons refuse: the arrow keys press these same buttons.
 *
 * @param {number} delta - -1 or 1.
 * @returns {boolean} Whether the page moved.
 */
function step(delta) {
  const window_ = pageWindow();
  const target = page + delta;
  if (!pdf || target < window_.first || target > window_.last) return false;
  page = target;
  void safeRender();
  return true;
}

els.prev.addEventListener("click", () => { refused = !step(-1); });
els.next.addEventListener("click", () => { refused = !step(1); });
// Bound once, asked on every click: only a restricted document refuses a step that
// the document itself could have taken, and only there is there anything to say. The
// arrow keys go through the same buttons, so a key at the edge is answered the same.
tapNote(els.prev, () => (refused && isRestricted(doc) ? t("restricted_pages") : ""));
tapNote(els.next, () => (refused && isRestricted(doc) ? t("restricted_pages") : ""));
document.addEventListener("langchange", refreshChrome);
addEventListener("keydown", (event) => {
  // Every shortcut below is a bare key, so none of them may fire while the reader is
  // typing: the "search this document" box (#doc-q) would otherwise lose its "h", its
  // hyphens and its cursor movement to a highlight flip, a zoom and a page turn.
  if (event.target.closest?.("input, textarea, select, [contenteditable]")) return;
  if (event.key === "ArrowLeft") els.prev.click();
  if (event.key === "ArrowRight") els.next.click();
  // "h" for the highlights, so a reader comparing the page with and without them does
  // not have to travel to the bar for each flip.
  if (event.key === "h" || event.key === "H") els.hlBtn.click();
  // "+" and "-" zoom the expanded page about its centre, for a reader with neither a
  // touchscreen nor a trackpad. "=" is "+" without shift on most layouts.
  if (expanded && ["+", "=", "-"].includes(event.key)) {
    event.preventDefault();
    void zoomTo(zoom * (event.key === "-" ? 1 / 1.25 : 1.25),
                els.scroll.clientWidth / 2, els.scroll.clientHeight / 2);
  }
  // Escape brings an expanded page back (site.js closes the menus). Native fullscreen
  // swallows Escape to exit itself, which the fullscreenchange listener below turns
  // into the same thing, so both routes agree.
  if (event.key === "Escape" && expanded && !document.fullscreenElement) void setExpanded(false);
});
// Leaving fullscreen by the browser's own route (Escape, F11, the system control)
// must leave the overlay too, or the page stays covering the window with no way back.
document.addEventListener("fullscreenchange", () => {
  if (!document.fullscreenElement && expanded) void setExpanded(false);
});

// The viewer's own question goes to the search page, because that page IS the answer:
// the ranking, the two views and the filter panel all live there, and a second copy
// of them here would be a second thing to keep correct. The scope travels as the same
// URL parameters the search page already reads, so the reader lands on a result list
// that says what it was narrowed to and can widen it in one click.
els.docSearchForm?.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = els.docQ.value.trim();
  if (!question) return;
  const scope = els.docSearchForm.querySelector("input[name=scope]:checked")?.value || "doc";
  const url = new URL("index.html", location.href);
  url.searchParams.set("q", question);
  if (scope === "doc") url.searchParams.set("file", doc.file);
  else if (scope === "issuer" && doc.issuer) url.searchParams.set("issuer", doc.issuer);
  location.href = url.toString();
});
// Re-render on resize so the bitmap matches the new CSS width. Debounced: a drag
// fires this continuously and each render decodes a PDF page.
let resizeTimer = 0;
const rerender = () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { if (pdf) void safeRender(); }, 200);
};
addEventListener("resize", rerender);
// The browser's own pinch fires this and not the window's resize: it changes
// `visualViewport.scale`, which `pixelRatio` reads, so the pinched page is redrawn
// at the resolution it is now shown at.
window.visualViewport?.addEventListener("resize", rerender);
