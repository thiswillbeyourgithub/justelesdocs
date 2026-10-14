/* The search page's URL contract, read in one place for both of its readers.

   The page (src/app.js) reads it to restore a question and its filters from a link,
   and the search service (server/service.mjs) reads the very same query string on
   `GET /api/search.md`, so an assistant handed a result-page link gets the same
   answer as Markdown by changing the path alone. Two parsers would drift: a
   parameter the page learned and the service did not would be dropped silently by
   the one an assistant uses, and it would quote the answer to another question.

   The URL is the page's whole state: the question and every active filter.

   That is what makes a result list a place. It can be bookmarked, sent to a
   colleague, and reached again by the back button from the viewer, which is the
   path that matters most: "retour aux résultats" has to come back to the same
   question AND the same filters, not to a bare corpus listing.

   `q` is the canonical name and `query` is accepted as an alias on the way in,
   because `?query=` is what anyone hand-writing one of these URLs reaches for
   first. Only `q` is ever written, so the address bar stays one shape.

   A filter is one parameter named after its field (`?issuer=WHO&topic=water`), and
   the year range is `?year=2016-2020`. A facet with several values REPEATS its
   parameter (`?topic=water&topic=energy`) rather than joining them with a separator:
   nothing then has to be escaped, a value containing the separator cannot corrupt
   the list, and `?topic=tdah` written by hand or bookmarked before the control
   could express more than one still means exactly what it meant.

   Imports only search.js and paths.js, both already in the search image.

   Written by Claude Code (Opus 5.5). */

import { FIGURE_MODES } from "./search.js";
import { parseRange } from "./paths.js";

export const QUERY_PARAM = "q";
export const QUERY_ALIAS = "query";

/* The guideline level, spelled positively (`?level=strict`, `?level=all`).
   `?guideline=` is what links said before the tier column was configurable, and is
   still read so those links keep their meaning. src/app.js explains why the level is
   its own mechanism rather than one more facet. */
export const LEVEL_PARAM = "level";
export const LEVEL_PARAM_ALIAS = "guideline";
/* Not a tier: the value that means "filter nothing at all". It is spelled out rather
   than left as an empty string so that a URL can say it, which is the only way a
   shared link can override a reader's saved preference in the widening direction. */
export const LEVEL_ALL = "all";

/* The document-name filter on the browse page. Kept out of `filters` on purpose:
   that object is the manifest's facets and goes to allowedDocs() as-is, where an
   unknown key would be read as a column that does not exist. */
export const NAME_PARAM = "name";

/**
 * Filters that exist in the URL but get no control in the panel.
 *
 * `file` is how the viewer asks the search page a question about the document the
 * reader is already in. It is a filter like any other as far as `allowedDocs` is
 * concerned, and deliberately not a facet: a hundred-odd documents make a useless
 * dropdown, and this scope is chosen where the reader already is rather than from a list.
 */
export const SCOPE_FIELDS = ["file"];

const BOOLEAN = {
  fromUrl: (raw) => (raw === "1" ? true : raw === "0" ? false : null),
  toUrl: (value) => (value ? "1" : "0"),
  fromStore: (raw) => (raw === "on" ? true : raw === "off" ? false : null),
  toStore: (value) => (value ? "on" : "off"),
};
const oneOf = (values) => {
  // A value the page does not know is read as "says nothing" rather than sent on,
  // since the service would refuse the whole search over it.
  const known = (raw) => (values.includes(raw) ? raw : null);
  return { fromUrl: known, toUrl: String, fromStore: known, toStore: String };
};

/* The four ranking choices, by the name the page keeps them under: their URL
   parameter, their default, and how each is read and written. src/app.js adds the
   ids of the controls and says why each default is what it is. */
export const URL_PREFS = {
  rescore: { param: "bm25", def: false, ...BOOLEAN },
  norefs: { param: "norefs", def: false, ...BOOLEAN },
  recent: { param: "recent", def: false, ...BOOLEAN },
  figures: { param: "figures", def: FIGURE_MODES[0], ...oneOf(FIGURE_MODES) },
};

/**
 * The ranking choices as `search.searchBody` and `search.searchRemote` take them.
 *
 * @param {{rescore?: boolean|null, norefs?: boolean|null, recent?: boolean|null,
 *   figures?: string|null}} prefs - Null or absent means the default.
 */
export function rankingOptions(prefs) {
  return {
    bm25: prefs.rescore ?? URL_PREFS.rescore.def,
    demoteRefs: prefs.norefs ?? URL_PREFS.norefs.def,
    recent: prefs.recent ?? URL_PREFS.recent.def,
    figures: prefs.figures ?? URL_PREFS.figures.def,
  };
}

/**
 * Read the question, the filters, the ranking choices and the level out of a URL.
 *
 * @param {URLSearchParams} params
 * @param {{facets: string[], ranges: Set<string>}} fields - The index's facet fields
 *   (`search.facetFields`) and the ones among them read as a numeric range
 *   (`search.rangeFields`). A parameter naming neither, nor a scope field, is ignored.
 * @returns {{question: string, filters: object, prefs: object, name: string,
 *   level: string|null}} `prefs` and `level` hold null where the URL says nothing,
 *   which the caller resolves against a saved preference or the default.
 */
export function readSearchParams(params, { facets, ranges }) {
  const question = (params.get(QUERY_PARAM) ?? params.get(QUERY_ALIAS) ?? "").trim();
  const prefs = {};
  for (const [name, pref] of Object.entries(URL_PREFS)) prefs[name] = pref.fromUrl(params.get(pref.param));
  // Not read through the loop below: the loop writes straight into `filters`, and an
  // absent level does not mean "no level".
  const level = (params.get(LEVEL_PARAM) || params.get(LEVEL_PARAM_ALIAS) || "").trim() || null;
  const name = (params.get(NAME_PARAM) || "").trim();
  const filters = {};
  for (const field of [...facets, ...SCOPE_FIELDS]) {
    if (!ranges.has(field) && facets.includes(field)) {
      // Repeated parameters, in the order they appear in the URL. Read before the
      // single-value path below, which takes the FIRST parameter and would drop the
      // rest of the list if an empty one led it.
      const all = params.getAll(field).map((v) => v.trim()).filter(Boolean);
      if (all.length) filters[field] = all;
      continue;
    }
    const raw = (params.get(field) || "").trim();
    if (!raw) continue;
    if (ranges.has(field)) {
      // "2016-2020", and "2016-" or "-2020" for an open end (paths.js: parseRange).
      const range = parseRange(raw);
      if (range) filters[field] = range;
    } else {
      filters[field] = raw;
    }
  }
  return { question, filters, prefs, name, level };
}

/**
 * Put a level into `filters`, which is what the ranker and the service read.
 *
 * LEVEL_ALL is represented by the field being ABSENT rather than by a value meaning
 * "everything": `allowedDocs` skips a field nobody set, so the whole-corpus case
 * costs no per-document test, and `isActive` then agrees with the panel about
 * whether anything is being narrowed.
 *
 * @param {object} filters - Changed in place.
 * @param {string} field - The tier column, `search.tierField(index)`.
 * @param {string} level
 */
export function applyLevel(filters, field, level) {
  if (level === LEVEL_ALL) delete filters[field];
  else filters[field] = level;
}
