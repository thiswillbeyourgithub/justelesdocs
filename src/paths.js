/* The URLs of the things this site serves, written once.
 *
 * Every page builds the same handful of paths: the PDF of a document, the viewer's
 * own module and worker. They were spelled out separately in app.js, viewer.js and
 * i18n.js, which is three places to forget `encodeURIComponent` on a filename full
 * of spaces and accents, and three places to edit the day `stage.py` lays the
 * directory out differently.
 *
 * Deliberately dependency-free, so anything can import it: search.js imports
 * i18n.js, so a URL helper living in i18n.js could not be used by search.js without
 * a cycle.
 *
 * Written by Claude Code (Opus 5).
 */

/**
 * Where one document's PDF is served.
 *
 * @param {{file: string}} doc - A document record out of `index/meta.json`.
 * @returns {string} Path relative to the site root, which is where every page sits.
 */
export function pdfHref(doc) {
  return `pdf/${encodeURIComponent(doc.file)}`;
}

/**
 * Where one page of a RESTRICTED document comes from.
 *
 * Those 53 documents are not under `pdf/` at all: `scripts/stage.py` keeps them out
 * of the web root, and `server/pages.py` cuts the asked-for page out of the file per
 * request. So this is not a variant spelling of `pdfHref`, it is the only route to
 * that tier, and it returns one page rather than a document.
 *
 * @param {number} docId Index id of the document, as `meta.json` numbers it.
 * @param {number} page 1-based page number.
 * @returns {string} A URL returning a one-page PDF.
 */
export function pageHref(docId, page) {
  return `api/page?doc=${encodeURIComponent(docId)}&p=${encodeURIComponent(page)}`;
}

/**
 * The viewer's own downloads, in the order they are needed.
 *
 * 2.8 MB between them, fetched on the first click into the viewer and reused from
 * cache on every click after. Naming them here is what lets the result list warm
 * them while the reader is still reading (see prefetch.js); viewer.js imports pdf.js
 * by module specifier rather than from this list, because an import is an import.
 */
export const VIEWER_ASSETS = [
  "vendor/pdfjs/pdf.mjs",
  "vendor/pdfjs/pdf.worker.mjs",
];

/**
 * A response's JSON, or an error naming the status and the file.
 *
 * Here because both the ranker and the viewer fetch the index's JSON, and without the
 * check a missing file reads as "Unexpected token <" rather than as a 404.
 *
 * @param {Response} response
 * @returns {Promise<any>}
 */
export async function okJson(response) {
  if (!response.ok) throw new Error(`${response.status} ${response.url.split("/").pop()}`);
  return response.json();
}

/**
 * A numeric range as the search page writes it into its URL, read back.
 *
 * `writeUrlState` writes `${from ?? ""}-${to ?? ""}`, so either side may be empty
 * and an empty side means open-ended: "2016-2020", "2016-" (from 2016 on) and
 * "-2020" (up to 2020). An empty side has to come back absent, not as
 * `Number("")`, which is 0: "2016-" read as `{from: 2016, to: 0}` matched no
 * document at all. Split on the first hyphen only, since years are never negative.
 *
 * @param {string} raw The parameter's value, e.g. "2016-".
 * @returns {{from?: number, to?: number}|null} The bounds that were given, or null
 *   when neither side is a number.
 */
export function parseRange(raw) {
  const [from, to] = String(raw).split("-", 2)
    .map((part) => (part.trim() === "" ? NaN : Number(part.trim())));
  const range = {};
  if (Number.isFinite(from)) range.from = from;
  if (Number.isFinite(to)) range.to = to;
  return range.from != null || range.to != null ? range : null;
}

/**
 * A link into the viewer for one document, and optionally one passage in it.
 *
 * `doc` is the document's position in the index's sorted list, which is what every
 * other file the viewer fetches is keyed by, but it is not a stable name: one
 * document added to the corpus shifts every id after it, and a bookmark would open
 * a different guideline. So the link also carries `file`, the one name a document
 * keeps across rebuilds, and the viewer trusts the id only while the two agree
 * (`viewerDocId`).
 *
 * `p` is the page the result list showed for the passage (build_index.py chose it),
 * so the viewer opens where the list said rather than re-deriving the choice.
 *
 * @param {number} docId Index id of the document.
 * @param {string} file The document's file name, as `meta.json` records it.
 * @param {{chunk?: number|null, page?: number|null, back?: string}} [opts] The
 *   passage's chunk row, its page, and the search page's query string to return to.
 * @returns {string} A path relative to the site root.
 */
export function viewHref(docId, file, { chunk = null, page = null, back = "" } = {}) {
  let href = `view.html?doc=${encodeURIComponent(docId)}&file=${encodeURIComponent(file)}`;
  if (Number.isFinite(chunk)) href += `&chunk=${chunk}`;
  if (Number.isFinite(page)) href += `&p=${page}`;
  return back ? `${href}&back=${encodeURIComponent(back)}` : href;
}

/**
 * Which document a viewer link means, or null when it names none in this index.
 *
 * The `file` parameter wins over the id when they disagree, because the id is a
 * position that moves when the corpus grows (`viewHref`). A link without `file`
 * (made before it existed) is taken at its id, as it always was. A link with no
 * `doc` at all used to read as `Number(null)`, which is 0, and opened the first
 * document in the corpus as if it were the one asked for.
 *
 * @param {string|null} docParam The `doc` parameter as read from the URL.
 * @param {string|null} fileParam The `file` parameter, likewise.
 * @param {Array<{file: string}>} documents `meta.json`'s document list.
 * @returns {number|null} The document's index id.
 */
export function viewerDocId(docParam, fileParam, documents) {
  const named = docParam == null || docParam.trim() === "" ? NaN : Number(docParam);
  const inRange = Number.isInteger(named) && named >= 0 && named < documents.length;
  if (fileParam) {
    if (inRange && documents[named].file === fileParam) return named;
    const found = documents.findIndex((d) => d.file === fileParam);
    return found >= 0 ? found : null;
  }
  return inRange ? named : null;
}

/**
 * The linked passage's position within its document, or -1 for none.
 *
 * A chunk row outside the document (a stale link, a hand-edited one) means no
 * highlight. It used to be clamped to the document's first chunk, which drew a
 * highlight on a passage nobody had asked about and called it the match.
 *
 * @param {number|null} chunkRow The `chunk` parameter: a row in the whole index.
 * @param {number} offset The document's `chunk_offset`.
 * @param {number} count How many chunks the document has.
 * @returns {number} 0-based index into the document's chunks, or -1.
 */
export function localChunkOf(chunkRow, offset, count) {
  if (!Number.isInteger(chunkRow)) return -1;
  const local = chunkRow - offset;
  return local >= 0 && local < count ? local : -1;
}

/**
 * The page the viewer opens on.
 *
 * The link's `p` when it is a page of the passage (or, with no passage, of the
 * document): that is the page the result list printed, chosen once by
 * build_index.py. For a link without `p` the viewer falls back to that same rule,
 * the page carrying most of the passage's boxes, so older links still agree with
 * the list.
 *
 * @param {string|null} pParam The `p` parameter as read from the URL.
 * @param {{pages: number[], boxes?: Object<string, Array>}|undefined} match The
 *   linked chunk, if any.
 * @param {number} pageCount How many pages the document has.
 * @returns {number} A 1-based page number.
 */
export function openingPage(pParam, match, pageCount) {
  const asked = pParam == null || pParam.trim() === "" ? NaN : Number(pParam);
  const valid = Number.isInteger(asked) && asked >= 1 && asked <= pageCount
    && (!match || match.pages.includes(asked));
  if (valid) return asked;
  if (!match) return 1;
  return Number(Object.entries(match.boxes || {})
    .sort((a, b) => b[1].length - a[1].length)[0]?.[0] || match.pages[0]);
}
