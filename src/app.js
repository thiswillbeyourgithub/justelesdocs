/* Wiring for the two reader-facing pages: DOM in, search.js out.

   Kept separate from search.js so the ranking logic has no DOM dependency and can
   be reasoned about (or tested) on its own. This file owns every querySelector on
   the page; search.js owns none.

   One file for both pages rather than two. index.html asks questions, browse.html
   lists documents, and they share the index, the filter panel, the URL contract, the
   footer and every row-rendering function: the only real difference is what the list
   holds. `PAGE` below is the whole of the divergence, and each element is looked up
   on both pages and simply missing on one, which is why the guards are on the
   elements rather than on the page name wherever that reads honestly. */

import { t, apply, lang, plural, docRefLinks, pdfMenu, isRestricted, figureNote } from "./i18n.js";
import { setStatus } from "./site.js";
import { load, save } from "./store.js";
import {
  loadIndex, parseQuery, checkParsedQuery, searchRemote,
  facets, facetFields, rangeFields, tierField, pageLabel, snippet,
  groupByDocument, browseDocs, isActive, fold, FIGURE_MODES,
} from "./search.js";
import { startTour } from "./tour.js";
import { pdfHref, parseRange, viewHref, VIEWER_ASSETS } from "./paths.js";
import { createPrefetcher, prefetchAllowed, warmDocs } from "./prefetch.js";

/* "search" (index.html) or "browse" (browse.html), from body[data-page]. Anything
   unrecognised is the search page, so a page that forgets the attribute degrades to
   the site's front door rather than to a blank list. */
const PAGE = document.body.dataset.page === "browse" ? "browse" : "search";

const els = {
  form: document.getElementById("form"),
  q: document.getElementById("q"),
  results: document.getElementById("results"),
  facets: document.getElementById("facets"),
  filters: document.getElementById("filters"),
  scopeNote: document.getElementById("scope-note"),
  reset: document.getElementById("reset-btn"),
  tourBtn: document.getElementById("tour-btn"),
  viewToggle: document.getElementById("view-toggle"),
  resultControls: document.getElementById("result-controls"),
  advanced: document.getElementById("advanced"),
  hint: document.getElementById("landing-hint"),
  nameFilter: document.getElementById("name-filter"),
  widen: document.getElementById("widen"),
};

/* Facet fields, in the order they are offered, and the ones rendered as a
   two-handle numeric range rather than a dropdown: both declared by the corpus
   (corpus.toml, [facets]) and read from the index's meta.json, so this page holds
   no list of its own to drift from the search service's. Every field may be blank
   across the whole corpus, in which case search.facets() drops it and no control
   appears: the panel grows as the manifest gets curated, with no edit here. Empty
   until the index has loaded, which is also when the first URL is read. */
const facetFieldsNow = () => facetFields(index);
const rangeFieldsNow = () => rangeFields(index);

/**
 * Filters that exist in the URL but get no control in the panel.
 *
 * `file` is how the viewer asks this page a question about the document the reader is
 * already in. It is a filter like any other as far as `allowedDocs` is concerned, and
 * deliberately not a facet: a hundred-odd documents make a useless dropdown, and this
 * scope is chosen where the reader already is rather than from a list.
 */
const SCOPE_FIELDS = ["file"];
const urlFields = () => [...facetFieldsNow(), ...SCOPE_FIELDS];

/* The guideline level: one filter, but not a facet. It defaults to the whole
   corpus (the user's call, 2026-09-28: a reader should find a reference book or an
   article without first learning that a filter hid it), so it narrows only once
   the reader picks a level.

   Its column is the corpus's tier column (corpus.toml, [tiers] column; `guideline`
   in the first corpus, hence the historical name), read from the index by search.tierField.

   Three things make it its own mechanism rather than one more facet field. Its values are NESTED, so picking one keeps the narrower ones too
   (search.js: matchesGuideline). It has a DEFAULT of its own, spelled `all` and
   represented by the field being absent, and a saved preference can override it,
   which the loop over urlFields() knows nothing about. And its options come from the vocabulary in
   meta.json rather than from the values the corpus happens to carry, because a
   level the corpus cannot currently fill is still a level worth offering.

   Spelled positively in the URL (`?level=strict`, `?level=all`) and written only
   when it differs from the default, so an ordinary shared link carries nothing
   about it, and a link that does carry one reproduces that reader's list exactly.
   `?guideline=` is what links said before the tier column was configurable, and
   is still read so those links keep their meaning. The storage key keeps that old
   name too, so no returning reader loses a saved level. */
const LEVEL_PARAM = "level";
const LEVEL_PARAM_ALIAS = "guideline";
const LEVEL_KEY = "guideline";
/* Not a tier: the value that means "filter nothing at all". It is spelled out rather
   than left as an empty string so that a URL can say it, which is the only way a
   shared link can override a reader's saved preference in the widening direction. */
const LEVEL_ALL = "all";
const LEVEL_DEFAULT = LEVEL_ALL;
let guidelineLevel = LEVEL_DEFAULT;
guidelineLevel = load(LEVEL_KEY) || guidelineLevel;

let index = null;
let filters = {};
let searchable = false;

/* The last query's folded results, kept so that flipping the view redraws them
   instead of re-running the search: the two views are two presentations of one
   answer, and re-embedding the same question to change a layout would be both slow
   and a lie, since the encoder is not guaranteed to be up. */
let lastResults = [];
let lastQuestion = "";

/* "passages" (one row a passage) or "documents" (one row a document, its passages
   nested under it). Remembered per browser: a reader who prefers one of them prefers
   it for every query, and the preference is worth nothing if it resets on reload. */
const VIEW_KEY = "view";
let view = "passages";
if (load(VIEW_KEY) === "documents") view = "documents";

/* The four ranking choices a reader can make, and every place each one lives.

   Each is remembered per browser (localStorage) AND kept in the URL. A result list is
   a place here, links to it are meant to be shareable, and two readers opening the
   same link must see the same order, so a shared link beats the saved preference.
   A choice is written into the URL only when it differs from its default, so the
   common case adds nothing to the address bar, and BOTH spellings of a boolean
   (`1` and `0`) are read, so a link shared while a default was the other way round
   still names the order it was shared with.

   - `rescore` (`bm25`): the lexical half of the ranking. OFF by default, so that what
     the page shows first is what the embedding model alone thinks the answer is: that
     is the thing being evaluated, and mixing a word-count into it by default hides
     which half a bad answer came from. It is also the half a reader can disagree
     with, since it rewards a passage for using the words that were typed, which is
     exactly wrong when the question and the answer use two vocabularies. It is worth
     about eight queries of page@1 over the eval set (DESIGN.md).
   - `norefs`: the reference demotion, ON by default, so only `norefs=0` is written.
   - `recent`: the recency bonus (src/search.js, RECENCY_BONUS), ON by default too.
   - `figures`: described figures competing with the text (the default), left out,
     or alone. One of FIGURE_MODES.

   `check` and `label` are element ids: the control, and the words beside it that
   carry the explanation as a tooltip (the box itself is three pixels wide). */
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
const PREFS = {
  rescore: { param: "bm25", def: false, ...BOOLEAN, check: "rescore-check", label: "rescore-toggle" },
  norefs: { param: "norefs", def: true, ...BOOLEAN, check: "norefs-check", label: "norefs-toggle" },
  recent: { param: "recent", def: true, ...BOOLEAN, check: "recent-check", label: "recent-toggle" },
  figures: { param: "figures", def: FIGURE_MODES[0], ...oneOf(FIGURE_MODES),
             check: "figures-select", label: "figures-toggle" },
};

/** Current value of each preference: the saved one, else its default. */
const prefs = {};
for (const [name, pref] of Object.entries(PREFS)) {
  prefs[name] = pref.fromStore(load(name)) ?? pref.def;
}

/** Read a preference's control, a checkbox or a select. */
function controlValue(control) {
  return control.type === "checkbox" ? control.checked : control.value;
}

/** Show `prefs[name]` on its control, when the page has one. */
function showPref(name) {
  const control = document.getElementById(PREFS[name].check);
  if (!control) return;
  if (control.type === "checkbox") control.checked = prefs[name];
  else control.value = prefs[name];
}

/* Warms the viewer's downloads while the reader reads the list. Null when the
   browser says the connection is metered or slow, in which case nothing speculative
   happens at all: see prefetch.js for the three rules this obeys. */
const prefetcher = prefetchAllowed(navigator.connection)
  ? createPrefetcher({ keep: VIEWER_ASSETS })
  : null;

/**
 * Queue what a click on this answer would need.
 *
 * The viewer's own module and worker first, because they are 2.8 MB that every
 * click needs and no click can start without; then the PDFs of the few top results,
 * because that is where a click usually lands. Bounded by PREFETCH_DOCS and by the
 * queue's own byte budget, so a list full of 10 MB argumentaires cannot turn reading
 * into downloading.
 *
 * Called on a painted list rather than on the results themselves, so the grouped
 * view and the passage view warm the same documents in the same order: what is
 * prefetched follows what is on screen.
 */
function warmResults() {
  if (!prefetcher || !lastResults.length) return;
  // Restricted documents have no file to fetch, only pages cut on demand, and warming
  // those would mean asking the page service for pages nobody has asked to read.
  const docs = warmDocs(lastResults, index.docs, isRestricted);
  prefetcher.enqueue([...VIEWER_ASSETS, ...docs.map(pdfHref)]);
}

/**
 * The reader-facing label for one facet value.
 *
 * The manifest stores stable ASCII slugs (`housing`, `safety_notice`, `FR`, `INSEE`)
 * because they are a data format: they must survive a language switch, sort
 * predictably, and never change when the wording does. What the reader sees is
 * looked up here, which is also where acronyms get expanded: "INSEE" alone tells a
 * reader outside French statistics nothing, so the option reads
 * "INSEE (Institut national de la statistique et des études économiques)".
 *
 * A slug with no label falls back to itself rather than to a blank. That is
 * deliberate: an unlabelled option in the UI is a visible bug report, and it means
 * curating a new slug never breaks the panel, it only makes it briefly ugly.
 */
function facetLabel(field, value) {
  const key = `facet_${field}_${value}`;
  const label = t(key);
  return label === key ? value : label;
}

/** A tier's label (guideline_<tier>, from the corpus overlay), or the tier itself. */
function tierLabel(value) {
  const key = `guideline_${value}`;
  const label = t(key);
  return label === key ? value : label;
}

/**
 * A facet as a list of checkboxes: any number of values, or none.
 *
 * It was a dropdown, which could express one value per field. That is the wrong
 * shape for this manifest: `doc_type` and `topic` are multi-valued columns, a
 * document is tagged `tsa;adulte;diagnostic`, and a reader looking for two topics
 * had to pick one and browse past the other. `allowedDocs` has always taken a list
 * and ORed it (see matchesFilter in search.js); only the control could not say one.
 *
 * Checkboxes rather than a `<select multiple>`: the native multi-select needs a
 * modifier key nobody discovers, collapses to a scroll box of about four rows, and
 * on a phone it is a system sheet that hides the list it is filtering. A list of
 * boxes says what is selected without being opened, which is the whole point of
 * the panel no longer being collapsible.
 *
 * @param {string} field - Manifest column.
 * @param {string[]} values - Its distinct values, as search.js collected them.
 * @returns {HTMLElement} A labelled group, scrolling internally if it is long.
 */
function checkFacet(field, values) {
  const group = document.createElement("div");
  group.className = "choices";
  group.id = `facet-${field}`;
  group.setAttribute("role", "group");
  group.setAttribute("aria-labelledby", `facet-${field}-label`);

  // Sorted by what the reader reads, not by the slug: search.js sorted the slugs,
  // which puts "alerte_securite" before "argumentaire" but "Alerte de sécurité"
  // after "Argumentaire". Collated in the current language for the same reason the
  // corpus listing is.
  const collator = new Intl.Collator(lang(), { numeric: true, sensitivity: "base" });
  const labelled = values.map((v) => [v, facetLabel(field, v)])
                         .sort((a, b) => collator.compare(a[1], b[1]));
  const chosen = new Set(selected(field));

  for (const [value, text] of labelled) {
    const choice = document.createElement("label");
    choice.className = "choice";
    const box = document.createElement("input");
    box.type = "checkbox";
    box.value = value;
    box.checked = chosen.has(value);
    const caption = document.createElement("span");
    caption.textContent = text;
    box.addEventListener("change", () => {
      const next = new Set(selected(field));
      if (box.checked) next.add(value);
      else next.delete(value);
      // Absent rather than empty: `isActive` treats an empty list as no filter, but
      // an empty array would still be written into the URL and read back as one.
      if (next.size) filters[field] = [...next];
      else delete filters[field];
      refilter();
    });
    choice.append(box, caption);
    group.append(choice);
  }
  return group;
}

/**
 * Wrap a facet's checkboxes in a button that opens them, with a box to search them.
 *
 * Fourteen document types and thirty-odd topics do not fit next to a search box,
 * and a scrolling column of checkboxes is the worst of both: it takes the room of
 * an open menu while showing four of its options. So the knob a reader sees is one
 * button per facet carrying what that facet is currently doing ("Thème · 2"), and
 * the checkboxes live in a popup under it with a text box to narrow them, which is
 * how a reader finds "qualité de l'eau potable" without reading thirty
 * labels.
 *
 * Ticked values are moved to the top each time the popup opens, so what is in force
 * is visible without scrolling. They are NOT re-sorted while the popup is open: a
 * box that jumps to the top under the pointer as it is ticked takes the next click
 * with it, and the reader unticks something they never chose.
 *
 * @param {string} field
 * @param {string[]} values
 * @param {HTMLElement} group the checkbox group from `checkFacet`
 * @returns {HTMLElement}
 */
function facetPopup(field, values, group) {
  const wrap = document.createElement("div");
  wrap.className = "facet-menu";

  const button = document.createElement("button");
  button.type = "button";
  button.className = "plain facet-btn";
  button.id = `facet-${field}-btn`;
  button.setAttribute("aria-expanded", "false");
  button.setAttribute("aria-controls", `facet-${field}-pop`);

  const pop = document.createElement("div");
  pop.className = "facet-pop";
  pop.id = `facet-${field}-pop`;
  pop.hidden = true;

  const search = document.createElement("input");
  search.type = "search";
  search.className = "facet-search";
  search.id = `facet-${field}-search`;
  search.autocomplete = "off";
  search.placeholder = t("facet_search");
  search.setAttribute("aria-label", `${t(field)} - ${t("facet_search")}`);

  const empty = document.createElement("p");
  empty.className = "facet-empty";
  empty.textContent = t("facet_none");
  empty.hidden = true;

  /** Write the button's caption: the facet's name, plus how many values are on. */
  const label = () => {
    const count = [...group.querySelectorAll("input")].filter((box) => box.checked).length;
    button.textContent = count ? `${t(field)} · ${count}` : t(field);
    button.classList.toggle("narrowed", count > 0);
  };

  /** Show only the choices whose label contains what was typed, accents folded. */
  const narrow = () => {
    const needle = fold(search.value.trim());
    let shown = 0;
    for (const choice of group.querySelectorAll(".choice")) {
      const hit = !needle || fold(choice.textContent).includes(needle);
      choice.hidden = !hit;
      if (hit) shown++;
    }
    empty.hidden = shown > 0;
  };

  const open = () => {
    // Ticked first, then the collated order the group was built in. Sorting on open
    // rather than on change keeps the list still under the reader's pointer.
    const choices = [...group.querySelectorAll(".choice")];
    const ticked = choices.filter((c) => c.querySelector("input").checked);
    group.prepend(...ticked);
    pop.hidden = false;
    button.setAttribute("aria-expanded", "true");
    search.value = "";
    narrow();
    search.focus();
  };

  const close = ({ refocus = false } = {}) => {
    pop.hidden = true;
    button.setAttribute("aria-expanded", "false");
    if (refocus) button.focus();
  };

  button.addEventListener("click", () => {
    if (pop.hidden) {
      // One open at a time: two popups side by side cover each other, and the
      // second one covers the first one's own button.
      closeFacetPops(pop);
      open();
    } else {
      close();
    }
  });
  search.addEventListener("input", narrow);
  group.addEventListener("change", label);
  // Escape closes without undoing anything: the ticks have already taken effect,
  // and a reader pressing Escape means "put this menu away", not "undo my filter".
  wrap.addEventListener("keydown", (event) => {
    if (event.key === "Escape") { close({ refocus: true }); event.stopPropagation(); }
  });
  // A click anywhere else closes it, which is what a menu does. Listening on the
  // document rather than on a backdrop element so the page behind stays usable.
  // Tied to this rendering of the panel: renderFacets runs on every language change,
  // reset and history step, and each run used to leave its popups' listeners behind.
  document.addEventListener("click", (event) => {
    if (!pop.hidden && !wrap.contains(event.target)) close();
  }, { signal: facetsAbort.signal });

  label();
  pop.append(search, group, empty);
  wrap.append(button, pop);
  return wrap;
}

/**
 * Shut every facet popup but `except`, with its button saying so.
 *
 * The one definition of "closed" for a popup: the facets call it to keep one open at
 * a time, and the tour to hand back a page with none open.
 *
 * @param {HTMLElement|null} [except] - A `.facet-pop` to leave as it is.
 */
function closeFacetPops(except = null) {
  for (const pop of document.querySelectorAll(".facet-pop")) {
    if (pop === except) continue;
    pop.hidden = true;
    document.getElementById(`${pop.id.replace(/-pop$/, "")}-btn`)?.setAttribute("aria-expanded", "false");
  }
}

/**
 * The values currently chosen for one field, whatever shape they arrived in.
 *
 * A URL written before this control existed carries `?topic=tdah`, and a `file`
 * scope from the viewer is still a single string, so a plain string has to keep
 * meaning a list of one.
 *
 * @param {string} field
 * @returns {string[]}
 */
function selected(field) {
  const value = filters[field];
  if (!value) return [];
  if (Array.isArray(value)) return value;
  if (typeof value === "object") return [];   // a range, which has no checkboxes
  return [String(value)];
}

/**
 * A range facet: two sliders, a low bound and a high bound.
 *
 * The filter is only recorded when the reader has actually narrowed something, so
 * the default state is genuinely "no year filter" rather than "every year, matched
 * one by one". That matters because a document with a blank year cell is excluded
 * by any active filter, and a slider sitting at its own full extent must not
 * quietly drop the few documents whose year nobody has curated yet.
 *
 * It reads as one slider with a handle at each end, and it is still two native
 * inputs: there is no two-handle element, and writing one would mean rewriting
 * keyboard support, focus and touch targets for a facet with twenty values. They
 * are laid on top of each other over a track drawn by the wrapper (app.css), with
 * the lit segment between the two values positioned here.
 */
function rangeFacet(field, values) {
  const numbers = values.map(Number).filter(Number.isFinite).sort((a, b) => a - b);
  if (numbers.length < 2) return checkFacet(field, values);  // not a range after all
  const min = numbers[0];
  const max = numbers[numbers.length - 1];
  const current = filters[field] || {};
  let from = Number.isFinite(current.from) ? current.from : min;
  let to = Number.isFinite(current.to) ? current.to : max;

  const wrap = document.createElement("div");
  wrap.className = "range";
  const readout = document.createElement("output");
  readout.className = "range-readout";
  readout.htmlFor = `facet-${field}`;

  const slider = (which, value) => {
    const input = document.createElement("input");
    input.type = "range";
    input.min = String(min);
    input.max = String(max);
    input.step = "1";
    input.value = String(value);
    input.id = which === "from" ? `facet-${field}` : `facet-${field}-to`;
    input.setAttribute("aria-label", t(which === "from" ? "range_from" : "range_to"));
    return input;
  };
  const lo = slider("from", from);
  const hi = slider("to", to);

  const track = document.createElement("div");
  track.className = "range-track";
  const fill = document.createElement("div");
  fill.className = "range-fill";

  const sync = () => {
    // Crossed handles are the classic failure of two-input ranges: dragging the low
    // handle past the high one would build a range that matches nothing and look
    // broken rather than empty. Pushing the other handle keeps the range legal.
    from = Number(lo.value);
    to = Number(hi.value);
    if (from > to) {
      if (document.activeElement === lo) { to = from; hi.value = String(to); }
      else { from = to; lo.value = String(from); }
    }
    readout.textContent = from === to ? String(from) : t("range_span", { a: from, b: to });
    // The lit part of the track, as two fractions of the span that app.css turns into
    // a position inside the thumbs' travel, so it follows both handles without the
    // page having to know how wide the facet column or the thumb is.
    const span = max - min || 1;
    track.style.setProperty("--lo", String((from - min) / span));
    track.style.setProperty("--hi", String((to - min) / span));
    // Two thumbs at the same year overlap, and the one on top swallows the drag.
    // At the far right only the low handle can still move inward, so it goes on top
    // there; everywhere else the high handle does.
    lo.style.zIndex = from === max ? "4" : "2";
    hi.style.zIndex = from === max ? "2" : "3";
    if (from === min && to === max) delete filters[field];
    else filters[field] = { from, to };
  };

  // "input" paints the readout while dragging; "change" is what re-runs the search,
  // so a drag across fifteen years costs one query rather than fifteen.
  for (const input of [lo, hi]) {
    input.addEventListener("input", sync);
    input.addEventListener("change", () => { sync(); refilter(); });
  }
  track.append(fill, lo, hi);
  wrap.append(readout, track);
  sync();   // after appending: the fill is positioned from the inputs' own values
  return wrap;
}

/* The URL is the page's whole state: the question and every active filter.

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
   could express more than one still means exactly what it meant. */
/* The document-name filter on the browse page. Kept out of `filters` on purpose:
   that object is the manifest's facets and goes to allowedDocs() as-is, where an
   unknown key would be read as a column that does not exist. */
const NAME_PARAM = "name";
let docName = "";

const QUERY_ALIAS = "query";

/** Read the question and the filters out of the current URL. */
function readUrlState() {
  const params = new URL(location.href).searchParams;
  const question = (params.get("q") ?? params.get(QUERY_ALIAS) ?? "").trim();
  // Null where the URL says nothing, which the caller resolves against the saved
  // preference (boot) or the default (popstate).
  const ranking = {};
  for (const [name, pref] of Object.entries(PREFS)) ranking[name] = pref.fromUrl(params.get(pref.param));
  // Not read through the loop below: the loop writes straight into `filters`, and an
  // absent level does not mean "no level". Null here means "the URL says nothing",
  // which the caller resolves against the saved preference (boot) or the default
  // (popstate, where the URL is the whole truth about that history entry).
  const level = (params.get(LEVEL_PARAM) || params.get(LEVEL_PARAM_ALIAS) || "").trim() || null;
  const name = (params.get(NAME_PARAM) || "").trim();
  const next = {};
  const facetList = facetFieldsNow();
  const ranges = rangeFieldsNow();
  for (const field of urlFields()) {
    if (!ranges.has(field) && facetList.includes(field)) {
      // Repeated parameters, in the order they appear in the URL. Read before the
      // single-value path below, which takes the FIRST parameter and would drop the
      // rest of the list if an empty one led it.
      const all = params.getAll(field).map((v) => v.trim()).filter(Boolean);
      if (all.length) next[field] = all;
      continue;
    }
    const raw = (params.get(field) || "").trim();
    if (!raw) continue;
    if (ranges.has(field)) {
      // "2016-2020", and "2016-" or "-2020" for an open end (paths.js: parseRange).
      const range = parseRange(raw);
      if (range) next[field] = range;
    } else {
      next[field] = raw;
    }
  }
  return { question, filters: next, prefs: ranking, name, level };
}

/**
 * Write the question and the filters into the URL.
 *
 * `push` for a new question, so the back button steps between questions; replace for
 * a filter change, so narrowing twice does not bury the question two entries deep.
 * The state object carries the same thing the URL does, so popstate does not have to
 * re-parse, but popstate reads the URL as a fallback: a bookmarked entry has no state.
 */
function writeUrlState({ push = false } = {}) {
  const url = new URL(location.href);
  url.searchParams.delete(QUERY_ALIAS);   // normalised away; only `q` is written
  if (lastQuestion) url.searchParams.set("q", lastQuestion);
  else url.searchParams.delete("q");
  for (const [name, pref] of Object.entries(PREFS)) {
    if (prefs[name] !== pref.def) url.searchParams.set(pref.param, pref.toUrl(prefs[name]));
    else url.searchParams.delete(pref.param);
  }
  // Written only when it is not the shipped default, so the ordinary link says
  // nothing about a level its reader never chose. `level=all` is a value like
  // any other here, which is what lets a link widen the corpus for someone whose
  // saved preference is narrower.
  url.searchParams.delete(LEVEL_PARAM_ALIAS);   // normalised away; only `level` is written
  if (guidelineLevel !== LEVEL_DEFAULT) url.searchParams.set(LEVEL_PARAM, guidelineLevel);
  else url.searchParams.delete(LEVEL_PARAM);
  if (docName) url.searchParams.set(NAME_PARAM, docName);
  else url.searchParams.delete(NAME_PARAM);
  const ranges = rangeFieldsNow();
  for (const field of urlFields()) {
    const value = filters[field];
    url.searchParams.delete(field);
    if (!value) continue;
    if (ranges.has(field)) {
      url.searchParams.set(field, `${value.from ?? ""}-${value.to ?? ""}`);
    } else {
      // delete-then-append, because `set` on a repeated parameter keeps the first
      // one and drops the rest, which would silently turn a three-value filter into
      // a one-value link.
      for (const one of Array.isArray(value) ? value : [value]) url.searchParams.append(field, one);
    }
  }
  const state = { q: lastQuestion, filters: JSON.parse(JSON.stringify(filters)) };
  if (push) history.pushState(state, "", url);
  else history.replaceState(state, "", url);
}

/* Re-run the question already asked, so changing a filter feels like narrowing
   rather than starting over. With no question the same filters narrow the corpus
   listing, which is the other half of what filters are for.

   The question ASKED (`lastQuestion`), not whatever is in the box: a reader who has
   started typing the next question and then ticks a filter is narrowing the answer
   on screen, not submitting half a sentence.

   Either way the URL follows, because a filtered corpus listing is as much a place
   as a filtered result list. */
function refilter() {
  refreshAdvanced();   // the summary carries the filter state while the panel is shut
  if (lastQuestion) void run(lastQuestion, { push: false });
  else { writeUrlState(); paintResults(); }
}

/**
 * Say when a question was asked of one document only, and offer the way out.
 *
 * The scope arrives in the URL from the viewer's search box, where the reader chose
 * it, and it would otherwise be invisible here: a short result list from a narrowed
 * corpus looks exactly like a short result list from a poor question. The way out is
 * a button rather than a hint about the filter panel, because `file` has no control
 * in that panel to go and undo.
 */
function paintScopeNote() {
  if (!els.scopeNote) return;
  const file = filters.file;
  els.scopeNote.textContent = "";
  els.scopeNote.hidden = !file;
  if (!file) return;
  const doc = index ? index.docs.find((candidate) => candidate.file === file) : null;
  els.scopeNote.append(`${t("scope_limited", { what: (doc && doc.title) || file })} `);
  const clear = document.createElement("button");
  clear.type = "button";
  clear.className = "plain";
  clear.textContent = t("scope_clear");
  clear.addEventListener("click", () => {
    delete filters.file;
    // The URL first: refilter() re-runs the question, which only writes the URL once
    // the encoder has answered, and until then a reload or a copied link would still
    // carry the scope the reader has just dropped.
    writeUrlState();
    refilter();
    paintScopeNote();
  });
  els.scopeNote.append(clear);
}

/**
 * The levels the reader can choose, narrowest first, with "all" last.
 *
 * The last tier is the "not guidance" bucket, and it is deliberately not offered:
 * "documents that are `no` or narrower" is every document in the corpus, which is
 * what the explicit "all" option says in words a reader can act on. Offering both
 * would put two spellings of the same list in one dropdown.
 *
 * @returns {string[]} tier ids plus LEVEL_ALL; empty when the index predates the
 *   tiers, in which case no control is drawn at all rather than one that does
 *   nothing.
 */
function levelOptions() {
  const tiers = index?.meta?.guideline_tiers;
  if (!Array.isArray(tiers) || tiers.length < 2) return [];
  return [...tiers.slice(0, -1), LEVEL_ALL];
}

/**
 * The level selector, at the head of the filter panel.
 *
 * A native `<select>` rather than the popup-of-checkboxes the facets use: the
 * levels are mutually exclusive and there are four of them, so the control that
 * shows the current one without being opened is the one that costs least. It also
 * keeps working with no JavaScript decisions about focus, which the popups do not.
 *
 * @returns {HTMLElement}
 */
function levelControl() {
  const options = levelOptions();
  const wrap = document.createElement("div");
  wrap.className = "facet facet-level";
  // On the wrapper, so hovering either the caption or the box explains the default.
  // Same reasoning as the ranking toggles: the sentence belongs to the words, and
  // the control itself is too small to hover deliberately.
  wrap.title = t("guideline_title");

  const label = document.createElement("label");
  label.className = "facet-label";
  label.htmlFor = `facet-${tierField(index)}`;
  label.textContent = t("guideline_level");

  const select = document.createElement("select");
  select.className = "level-select";
  select.id = `facet-${tierField(index)}`;
  for (const value of options) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = tierLabel(value);
    select.append(option);
  }
  // A level from localStorage or from a URL that this index cannot offer (a tier
  // renamed in manifest.py since the link was shared) falls back to the default
  // rather than leaving the box showing one thing and the list showing another.
  if (!options.includes(guidelineLevel)) {
    guidelineLevel = LEVEL_DEFAULT;
    syncLevelFilter();
  }
  select.value = guidelineLevel;

  select.addEventListener("change", () => setLevel(select.value));

  wrap.append(label, select);
  return wrap;
}

/**
 * Move to a level, from wherever the reader asked for it.
 *
 * Two controls set this: the select at the head of the filter panel, and the offer to
 * widen under the last result. They have to agree about everything that follows a
 * change (the box shows the new level, the preference is remembered, the filter is
 * rewritten, the question is asked again), so there is one of this rather than two
 * sequences that drift.
 *
 * The select is found in the DOM rather than closed over, because the panel is rebuilt
 * whenever the facets are, so the element this function wants is not always the one
 * that existed when a handler was bound.
 *
 * @param {string} next - A tier id, or LEVEL_ALL.
 */
function setLevel(next) {
  guidelineLevel = next;
  // Remembered like the ranking toggles: a reader who works in one level should
  // not have to set it again on the next visit.
  save(LEVEL_KEY, guidelineLevel);
  const select = document.getElementById(`facet-${tierField(index)}`);
  if (select && select.value !== guidelineLevel) select.value = guidelineLevel;
  syncLevelFilter();
  refilter();
}

/**
 * The next level out, or null when the reader is already seeing everything.
 *
 * @returns {string|null} A tier id or LEVEL_ALL, from the same ladder the select
 *   offers, so the two cannot disagree about what "one notch wider" means.
 */
function nextLevel() {
  const options = levelOptions();
  const at = options.indexOf(guidelineLevel);
  return at >= 0 && at < options.length - 1 ? options[at + 1] : null;
}

/**
 * Offer to widen, under the last result.
 *
 * The level filter is the one control on this page that narrows the corpus before the
 * reader has touched anything, and the moment its cost is felt is the end of a list
 * that stopped short: a guideline's own background report, an agency report or a public guide
 * may be the best answer to the question and be kept out of the default by its type.
 * The panel above says what the level IS; this says what it is leaving out, where a
 * reader who did not find their answer is already looking.
 *
 * Shown on an empty list too, deliberately: no results from a narrowed corpus and no
 * results from a corpus that has nothing look exactly alike, and this is the one that
 * can be acted on.
 *
 * One notch, not straight to "all": the ladder's own labels name what each step adds,
 * so a step can be described honestly ("also the summaries") where a
 * jump to the whole corpus could only be described as "everything".
 *
 * A widened list can come back SHORTER, and that is the ranker working as designed
 * rather than a bug here: search.js scores at most CANDIDATES (300) chunks and then
 * folds them at PER_DOC (3) per document, so a long argumentaire admitted by the wider
 * level can take fifty pool slots and contribute three rows, pushing other documents
 * out of the pool entirely. Measured once on a real corpus and question: 40 passages
 * from 20 documents at `strict`, 37 from 18 at `wide`. The corpus searched is wider
 * either way, which is what the button says.
 */
function paintWiden() {
  if (!els.widen) return;
  els.widen.textContent = "";
  // Not on the browse page (that list IS the corpus, and its own control sits beside
  // it), and not before a question: there is no list to have stopped short.
  const next = PAGE === "browse" || !lastQuestion ? null : nextLevel();
  els.widen.hidden = !next;
  if (!next) return;
  const button = document.createElement("button");
  button.type = "button";
  button.className = "plain";
  button.id = "widen-btn";
  // The tier's own label, minus the "+" it carries in the select, where it reads as a
  // ladder rung. Here it is the tail of a sentence.
  button.textContent = next === LEVEL_ALL
    ? t("widen_all")
    : t("widen_add", { what: tierLabel(next).replace(/^\+\s*/, "") });
  button.title = t("widen_title");
  button.addEventListener("click", () => setLevel(next));
  els.widen.append(button);
}

/**
 * Put the chosen level into `filters`, which is what the ranker and the service read.
 *
 * LEVEL_ALL is represented by the field being ABSENT rather than by a value meaning
 * "everything": `allowedDocs` skips a field nobody set, so the whole-corpus case
 * costs no per-document test, and `isActive` then agrees with the panel about
 * whether anything is being narrowed.
 */
function syncLevelFilter() {
  if (guidelineLevel === LEVEL_ALL) delete filters[tierField(index)];
  else filters[tierField(index)] = guidelineLevel;
}

/**
 * Is the corpus actually being narrowed, as the reader would understand it?
 *
 * Plain `some(isActive)`, because the default level ("everything") never enters
 * `filters`: `syncLevelFilter` deletes the field rather than storing it, so an
 * untouched page has nothing active. Were the default ever changed to a narrowing
 * level, this would light the "filters are narrowing this" summary on every fresh
 * page, which is the reason to keep the default and the deletion together.
 *
 * @returns {boolean}
 */
function isNarrowed() {
  return Object.values(filters).some(isActive);
}

/* Aborted at the top of every renderFacets, which removes the document listeners the
   previous panel's popups registered along with the popups themselves. */
let facetsAbort = new AbortController();

/**
 * Render the facet controls for whatever metadata actually exists.
 *
 * Two rows of two different shapes, and the split is the point. The level and the
 * year are captioned controls whose value is on show (a sentence in a <select>, a
 * span of years on a slider); every other facet is a button that is its own caption
 * and opens a list. Laid out in one auto-fit grid, the two shapes interleaved: a tall
 * captioned cell beside a short button, the level spanning two columns and pushing
 * the rest round it, and whichever facet came last alone on a row of its own. So the
 * captioned pair gets a row, and the buttons get a row of equal cells under it.
 */
function renderFacets() {
  facetsAbort.abort();
  facetsAbort = new AbortController();
  els.facets.textContent = "";
  const head = document.createElement("div");
  head.className = "facets-head";
  const menus = document.createElement("div");
  menus.className = "facets-menus";
  // First, and outside the loop: it is not built from the corpus's values, and it is
  // the control most likely to explain a result list that looks short.
  if (levelOptions().length) head.append(levelControl());
  const ranges = rangeFieldsNow();
  for (const { field, values } of facets(index, facetFieldsNow())) {
    const wrap = document.createElement("div");
    wrap.className = "facet";
    // A <label> for the range (it labels the slider) and a plain caption for the
    // checkbox group, which is labelled by id instead: a <label for> pointing at a
    // group of inputs is not a label, and clicking it would do nothing.
    const range = ranges.has(field);
    if (range) {
      const label = document.createElement("label");
      label.className = "facet-label";
      label.id = `facet-${field}-label`;
      label.textContent = t(field);
      label.htmlFor = `facet-${field}`;
      wrap.classList.add("facet-range");
      wrap.append(label, rangeFacet(field, values));
      head.append(wrap);
    } else {
      // No separate caption: the button IS the label, and it says more than a
      // caption could, since it also carries how many values are currently on.
      // The group inside still points at the button for its accessible name.
      const group = checkFacet(field, values);
      group.setAttribute("aria-labelledby", `facet-${field}-btn`);
      wrap.append(facetPopup(field, values, group));
      menus.append(wrap);
    }
  }
  // An empty row would still take its gap, so a corpus with no year column (or no
  // curated facet at all) gets no hole where the row would have been.
  for (const row of [head, menus]) if (row.childElementCount) els.facets.append(row);
}

/* Every row below is built with DOM calls, never innerHTML: chunk text is document
   content and must never be parsed as markup. */

/**
 * A link into the viewer, carrying the search it was clicked from.
 *
 * `back` holds this page's own query string, so "retour aux résultats" comes back to
 * the same question with the same filters rather than to a bare corpus listing. It
 * is carried explicitly rather than left to document.referrer because a result
 * opened in a new tab has no referrer at all, and because a referrer is a header
 * this page does not control.
 *
 * The rest of the link (the file name that keeps a bookmark on its document, the page
 * the list printed) is paths.js's `viewHref`, shared with the browse page's link.
 *
 * @param {{doc: number, chunk: number, page: number}} result One ranked passage.
 */
function resultHref(result) {
  const back = location.search.replace(/^\?/, "");
  return viewHref(result.doc, index.docs[result.doc].file,
                  { chunk: result.chunk, page: result.page, back });
}

/**
 * The document line, identical on a result row and on a corpus row: the PDF menu
 * and, where there is one, the publisher.
 *
 * A div rather than a p, because the menu is a <details> and a paragraph may only
 * hold phrasing content; the class and the styling are the ones the line always had.
 */
function docLinks(doc) {
  const links = document.createElement("div");
  links.className = "hit-extra doc-links";
  links.append(pdfMenu(doc));
  // The PDF menu first: it is this site's own copy, which is what a reader here wants
  // first. Then where else the document can be found (docRefLinks says in which order).
  for (const link of docRefLinks(doc)) links.append(" · ", link);
  return links;
}

/** The "N more passages here / also published as" line, or null when there is none. */
function extrasLine(more, alternates) {
  const extras = [];
  if (more) extras.push(t("also_in_doc", { n: more }));
  if (alternates && alternates.length) {
    const names = alternates
      .map((a) => index.docs[a.doc].title || index.docs[a.doc].file)
      .join(", ");
    extras.push(t("other_renditions", { list: names }));
  }
  if (!extras.length) return null;
  const line = document.createElement("p");
  line.className = "hit-extra";
  line.textContent = extras.join(" · ");
  return line;
}

/**
 * Which passages the reader has opened, by their row in the index.
 *
 * Outside the renderers because paintResults() rebuilds the whole list on a view
 * change, a language change and every repaint of the same query: an expansion that
 * disappeared when the reader flipped to the grouped view would read as a bug. Rows
 * are unique across documents, so nothing has to be scoped per document. Cleared by
 * a new query, where the rows mean something else.
 */
const openPassages = new Set();

/**
 * The passage itself: a snippet the reader can open into the whole chunk.
 *
 * The snippet is the biggest and most readable thing in a row and the thing being
 * judged, so it is also the thing under the cursor: clicking it shows the rest of the
 * chunk rather than leaving the reader to open the PDF page to find out whether the
 * sentence continues the way they hope. Clicking again puts it back, and any number
 * of passages can be open at once, since comparing two is the reason to open them.
 *
 * The page itself is one click away throughout, from the row's title and from its
 * page number, which is a link in both views.
 *
 * A passage shorter than the snippet limit has nothing to open, so it renders as
 * plain text with no control and no hover: an affordance that does nothing is worse
 * than none.
 *
 * @param {object} result - A search result, carrying `doc`, `chunk` and `text`.
 * @param {{block?: boolean, limit?: number}} [options] - Layout and snippet length.
 * @returns {HTMLElement} The passage, ready to append.
 */
function passageText(result, { block = false, limit } = {}) {
  // The positive half only: a word the reader subtracted must not be what the
  // snippet is centred on, the same rule the lexical rescoring follows.
  const asked = lastQuestion ? parseQuery(lastQuestion).lexical : "";
  const short = snippet(result.text, limit || undefined, asked);
  const whole = snippet(result.text, Infinity);
  const className = block ? "hit-text block" : "hit-text";
  if (short === whole) {
    const plain = document.createElement("span");
    plain.className = `${className} plain-text`;
    plain.textContent = whole;
    return withFigureNote(result, plain);
  }
  const button = document.createElement("button");
  button.type = "button";
  button.className = className;
  const paint = () => {
    const open = openPassages.has(result.chunk);
    button.textContent = open ? whole : short;
    button.setAttribute("aria-expanded", String(open));
    button.title = t(open ? "passage_collapse" : "passage_expand");
  };
  button.addEventListener("click", () => {
    if (openPassages.has(result.chunk)) openPassages.delete(result.chunk);
    else openPassages.add(result.chunk);
    paint();
  });
  paint();
  return withFigureNote(result, button);
}

/**
 * Say so when the passage is a model's description of a figure, not the document's words.
 *
 * A flowchart's words come out of a PDF as scattered box labels and a scanned figure
 * has none, so figures are indexed through a description a vision model wrote
 * (scripts/figures.py). A reader must never take that text for a quotation of a
 * guideline: the note names the model, and the page it links to shows the figure.
 *
 * @param {object} result - A search result; `figure` is set on figure chunks.
 * @param {HTMLElement} passage - What `passageText` built.
 * @returns {Node} The passage, preceded by the note when there is one.
 */
function withFigureNote(result, passage) {
  if (!result.figure) return passage;
  const note = document.createElement("span");
  note.className = "figure-note";
  note.textContent = figureNote(result.figure);
  note.title = t("figure_note_title");
  const both = document.createDocumentFragment();
  both.append(note, passage);
  return both;
}

/**
 * The page number as a link to that page, as both list views print it.
 *
 * @param {object} result - A search result.
 * @returns {HTMLAnchorElement}
 */
function pageLink(result) {
  const where = document.createElement("a");
  where.className = "passage-page";
  where.href = resultHref(result);
  where.textContent = pageLabel(result);
  return where;
}

/**
 * A row's head line: the document's title as a link, a meta line, and a score.
 *
 * Shared by the passage row and the document group so the two views cannot drift
 * apart in what a title links to or how its issuer and year are printed.
 *
 * @param {object} doc - The document, from `index.docs`.
 * @param {object} target - The result the title opens: the row's own passage, or a
 *   group's best one, which is what the document is ranked by.
 * @param {number} score - Printed with `scoreLabel`.
 * @param {Array<string|Node>} tail - What follows issuer and year on the meta line.
 * @returns {HTMLDivElement}
 */
function hitHead(doc, target, score, tail) {
  const head = document.createElement("div");
  head.className = "hit-head";
  const link = document.createElement("a");
  link.className = "hit-title";
  link.href = resultHref(target);
  link.textContent = doc.title || doc.file;
  const meta = document.createElement("span");
  meta.className = "hit-meta";
  const parts = [doc.issuer, doc.year, ...tail].filter(Boolean);
  parts.forEach((part, i) => meta.append(...(i ? [" · ", part] : [part])));
  const label = document.createElement("span");
  label.className = "hit-score";
  label.textContent = scoreLabel(score);
  head.append(link, meta, label);
  return head;
}

/** One passage row: the document it is in, where on which page, and the passage. */
function renderResult(result) {
  const doc = index.docs[result.doc];
  const li = document.createElement("li");
  // The page number is a link here as it is in the grouped view: with the passage
  // itself opening in place, the row still needs an obvious way to the page, and
  // "p. 42" is the part of the line that means exactly that.
  li.append(hitHead(doc, result, result.score, [pageLink(result)]),
            passageText(result, { block: true }));

  const extras = extrasLine(result.more, result.alternates);
  if (extras) li.append(extras);
  li.append(docLinks(doc));
  return li;
}

/**
 * One document row in the grouped view: the document once, then its passages.
 *
 * The passages keep their own page numbers and snippets, because "which document"
 * and "where in it" are two different questions and this view must not answer only
 * the first. They are nested in their own list so a reader scanning documents can
 * skip a whole group with one eye movement.
 *
 * Each passage carries its own rank and score, not just the document's. The score on
 * the group head is its BEST passage's, which is what the document is ordered by, so
 * on its own it says nothing about the second and third passage under it: a document
 * can lead on one strong passage and follow it with two weak ones. The rank is the
 * row number in the ungrouped list, so it also says what grouping hid, that passage
 * #1 and passage #34 are both here.
 *
 * @param {object} group - One entry from `groupByDocument`.
 * @param {Map<number, number>} ranks - Chunk index -> its row number in the
 *   ungrouped list. Built once by the caller, because a group cannot see the
 *   order it was taken out of.
 */
function renderGroup(group, ranks) {
  const doc = index.docs[group.doc];
  const li = document.createElement("li");
  li.className = "group";

  li.append(hitHead(doc, group.results[0], group.score,
                    [t("matching_passages", { n: group.results.length, s: plural(group.results.length) })]));

  const passages = document.createElement("ol");
  passages.className = "passages";
  for (const result of group.results) {
    const row = document.createElement("li");
    const where = pageLink(result);
    const rank = document.createElement("span");
    rank.className = "passage-rank";
    rank.textContent = t("passage_rank", { n: ranks.get(result.chunk) ?? "?",
                                           score: scoreLabel(result.score) });
    row.append(where, document.createTextNode(" "), rank, document.createTextNode(" "),
               passageText(result, { limit: 200 }));
    passages.append(row);
  }
  li.append(passages);

  const extras = extrasLine(group.more, group.alternates);
  if (extras) li.append(extras);
  li.append(docLinks(doc));
  return li;
}

/**
 * The score as a percentage, in the reader's typography.
 *
 * Three decimals of a cosine read as a precision the number does not have: 0.471
 * and 0.468 are the same answer, and neither says anything on its own scale,
 * where a relevant hit sits around 0.55 to 0.70 rather than near 1. A percentage
 * is read as an approximation, which is what it is. The unit lives in i18n.js
 * because French puts a space before it and English does not.
 *
 * What is passed in is the result's `score`, not its `cosine`: with the lexical
 * rescoring on they differ, and a list sorted by one while displaying the other
 * looks unsorted. That is what it looked like, and it is what the toggle beside
 * the view switch now makes visible.
 *
 * @param {number} cosine
 * @returns {string} the label and the value, e.g. "score 47%".
 */
function scoreLabel(cosine) {
  return `${t("score")} ${t("score_value", { n: Math.round(cosine * 100) })}`;
}

/** One corpus row: a document with no query in play, so no score and no passage. */
function renderBrowseRow(docId) {
  const doc = index.docs[docId];
  const li = document.createElement("li");

  const head = document.createElement("div");
  head.className = "hit-head";
  // Straight to the PDF, not into the viewer: with no query there is no passage to
  // highlight, and opening the viewer on an arbitrary chunk would claim there is.
  // A restricted document has no file to link to, so it goes to the viewer after all,
  // which opens on its first page and offers the pages either side of it.
  const link = document.createElement("a");
  link.className = "hit-title";
  link.href = isRestricted(doc) ? viewHref(docId, doc.file) : pdfHref(doc);
  link.textContent = doc.title || doc.file;
  const pages = Number(doc.pages);
  const meta = document.createElement("span");
  meta.className = "hit-meta";
  meta.textContent = [doc.issuer, doc.country, doc.year,
                      Number.isFinite(pages) && pages >= 1
                        ? t("page_count", { n: pages, s: plural(pages) }) : ""]
    .filter(Boolean).join(" · ");
  head.append(link, meta);
  li.append(head, docLinks(doc));
  return li;
}

/**
 * Draw whatever the page should be showing: the corpus, the passages, or the
 * documents.
 *
 * One function rather than three call sites, because the view toggle and a new
 * query are the same event as far as the list is concerned, and because the corpus
 * listing has to come back when a query is cleared.
 */
function paintResults() {
  els.results.textContent = "";
  paintScopeNote();
  paintWiden();
  if (els.resultControls) els.resultControls.hidden = !lastResults.length;
  if (!index) return;

  if (PAGE === "browse") {
    // This page IS the corpus listing, question or no question. It stays useful when
    // the encoder is down, which is the other reason it is a page of its own: a
    // reader can still find a document and download it while search is unavailable,
    // and #status keeps whatever reason it was given.
    const ids = browseDocs(index, filters, lang(), docName);
    setStatus(ids.length ? t("browse_count", { n: ids.length }) : t("browse_none"));
    for (const id of ids) els.results.append(renderBrowseRow(id));
    return;
  }

  // The search page before a question: nothing to list. The corpus used to be
  // printed here, which made the first screen a wall of every document under an empty
  // box and buried what the site is for. The hint takes its place and names the page
  // the list moved to.
  if (els.hint) els.hint.hidden = Boolean(lastQuestion);
  if (!lastQuestion) return;

  if (!lastResults.length) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = t("no_results");
    els.results.append(empty);
    return;
  }

  const docCount = new Set(lastResults.map((r) => r.doc)).size;
  setStatus(t("results_count", { n: lastResults.length, docs: docCount }));
  if (view === "documents") {
    // The row number of the ungrouped list, which is the rank each passage is labelled
    // with once grouping has taken the order away. Keyed by chunk, which is the index's
    // own global identifier for a passage and so unique across the whole answer.
    const ranks = new Map(lastResults.map((result, i) => [result.chunk, i + 1]));
    for (const group of groupByDocument(index, lastResults)) {
      els.results.append(renderGroup(group, ranks));
    }
  } else {
    for (const result of lastResults) els.results.append(renderResult(result));
  }
  warmResults();
}

/** Reflect the current view on the toggle, in the current language. */
function refreshViewToggle() {
  for (const button of els.viewToggle.querySelectorAll("button")) {
    button.textContent = t(button.dataset.view === "documents" ? "view_documents" : "view_passages");
    button.setAttribute("aria-pressed", String(button.dataset.view === view));
  }
}

/* Which call of `run` is the current one. Two searches can be in flight at once (a
   filter ticked while the previous answer is still on its way, a question resubmitted),
   and the service does not answer in order: without this, the OLDER answer arriving
   last painted itself under the newer question, or its error over the newer answer. */
let runSeq = 0;

/** Run one query end to end. */
async function run(question, { push = true } = {}) {
  if (!searchable) return;
  // Taken before the empty-question branch too: clearing the box has to win over a
  // search still on its way, or that answer comes back under an empty box.
  const seq = ++runSeq;
  if (!question) {
    // An emptied box goes back to the corpus listing rather than leaving the last
    // answer on screen under a box that no longer says what produced it.
    lastQuestion = "";
    lastResults = [];
    // And the count goes with it: "40 passages" over an empty list described an answer
    // no longer on screen (seen at the end of the guided tour, which empties the box).
    setStatus("");
    writeUrlState();
    paintResults();
    return;
  }
  // Recorded now rather than when the answer lands: it is what a filter change or a
  // ranking toggle re-runs, and a reader can touch those while this is in flight.
  lastQuestion = question;
  setStatus(t("searching"));
  els.results.textContent = "";
  // A new question means new rows: a row number the reader had opened would open an
  // unrelated passage in the next answer.
  openPassages.clear();
  // The advanced panel has done its job the moment a question is asked. Left open it
  // pushes the answer below the fold on a laptop, and a reader who opened it to read
  // about "-(...)" and then typed a question is looking for results, not for the
  // explanation they have just finished with. Reopening it is one click.
  // Not when a filter is set, though: the knobs the reader has just touched are in
  // there, and folding them away as the answer arrives reads as the page undoing the
  // work rather than getting out of the way.
  if (els.advanced && !isNarrowed()) els.advanced.open = false;
  try {
    // Parsed and checked here, before any request: a question the encoder would
    // refuse is refused in the same words without a round trip. The service parses
    // it again with the same function, since it cannot trust what it is sent.
    checkParsedQuery(parseQuery(question));
    // A new question means a new set of likely clicks: drop whatever was queued for
    // the old answer rather than finishing it out of politeness to nobody.
    prefetcher?.reset();
    // The page does not rank. The service holds the index, embeds the question,
    // ranks it under these filters, rescores the head of the list when the lexical
    // toggle is on, folds the renditions and cuts at the deployment's floor. Its
    // query-vector cache is what makes a re-ask, a filter change or the toggle cost
    // no encoder round trip.
    const answer = await searchRemote(question, filters, {
      bm25: prefs.rescore, demoteRefs: prefs.norefs, recent: prefs.recent, figures: prefs.figures,
    });
    if (seq !== runSeq) return;   // a newer search was started: its answer is the one
    lastResults = answer.results;
    if (!answer.results.length) setStatus("");
    // Before painting, not after: every result row's link carries the current search
    // as its way back, and reads it off location.
    writeUrlState({ push });
    paintResults();
  } catch (error) {
    if (seq !== runSeq) return;   // a newer search owns the status line now
    // The previous answer goes with the error. It was already cleared from the list,
    // but kept here a view or language toggle repainted it under a status line saying
    // this question failed, as if the old rows were its answer.
    lastResults = [];
    if (els.resultControls) els.resultControls.hidden = true;
    if (error.message === "empty") setStatus(t("query_only_negative"), true);
    else if (error.message === "terms") {
      setStatus(t("query_terms", { n: error.got, max: error.max }), true);
    } else if (error.message === "long") {
      setStatus(t("query_long", { n: error.got, max: error.max }), true);
    } else if (error.message === "short") {
      // Raised by search.js before the request, and again if the encoder's own
      // floor is higher than the page's copy of it. Either way the reader gets a
      // sentence about what this kind of search needs, not the status code.
      setStatus(t("query_short", { min: error.min }), true);
    } else if (error.message === "cancelled") setStatus(t("query_cancelled"), true);
    else if (error.message === "unavailable") setStatus(t("embed_unavailable"), true);
    else if (error.message === "busy") setStatus(t("search_busy"), true);
    else if (error.message === "dim") {
      setStatus(t("dim_mismatch", { got: error.got, want: error.want }), true);
      searchable = false;
    } else setStatus(String(error.message || error), true);
  }
}

/**
 * Keep the "recherche avancée" summary honest about what it is hiding.
 *
 * The panel is collapsed most of the time, so when it is closed its own summary is
 * the only thing left on the page that can say the corpus is being narrowed. A
 * reader who set two issuers, closed the panel and then wondered why half the
 * answers went missing would otherwise have nothing to look at.
 *
 * The text is written here rather than by `data-i18n` because it has two versions,
 * and `apply()` would overwrite whichever one this chose.
 */
function refreshAdvanced() {
  if (!els.advanced) return;   // the corpus listing has no disclosure, its panel is open
  const narrowed = isNarrowed();
  const summary = els.advanced.querySelector("summary");
  if (summary) summary.textContent = t(narrowed ? "advanced_narrowed" : "advanced_summary");
  els.advanced.classList.toggle("narrowed", narrowed);
}

function refreshChrome() {
  apply();
  refreshAdvanced();
  for (const [name, pref] of Object.entries(PREFS)) {
    showPref(name);
    // On the label rather than the box: the sentence explaining what the ranking does
    // belongs to the words a reader hovers, and the box is three pixels wide.
    const label = document.getElementById(pref.label);
    if (label) label.title = t(`${name}_title`);
  }
  if (index) {
    // Looked up here rather than captured with the other elements at load: site.js owns
    // the footer and creates this paragraph, and this file must not depend on which of
    // the two modules happened to run first.
    const corpusNote = document.getElementById("corpus-note");
    if (corpusNote) {
      corpusNote.textContent = t("index_ready", {
        chunks: index.meta.n_chunks.toLocaleString(t("locale")),
        docs: index.meta.n_documents,
      });
    }
    renderFacets();
    if (els.viewToggle) refreshViewToggle();
    // The rows carry translated strings (page labels, "N more passages", the corpus
    // listing's page counts), so a language change has to redraw them, not just the
    // chrome around them.
    paintResults();
  }
}

els.form?.addEventListener("submit", (event) => {
  event.preventDefault();
  void run(els.q.value.trim());
});
/* The box is a textarea, so three things a single-line input gave for free have to
   be said out loud: Enter submits (Shift+Enter is the newline a textarea is here
   for), a pasted line break does not survive into the question, and the height
   follows the content. `grow` is idempotent and cheap: it resets the height before
   measuring, because scrollHeight only ever grows while an explicit height is set. */
function grow() {
  if (!els.q) return;
  els.q.style.height = "auto";
  els.q.style.height = `${els.q.scrollHeight}px`;
}
els.q?.addEventListener("keydown", (event) => {
  // Right arrow in an EMPTY box takes the placeholder as the question, the way a shell
  // accepts its greyed-out suggestion: the example is the fastest way to see what a
  // good question looks like. Only when empty, so it never steals a cursor move.
  if (event.key === "ArrowRight" && !els.q.value && els.q.placeholder
      && !event.shiftKey && !event.ctrlKey && !event.altKey && !event.metaKey) {
    event.preventDefault();
    els.q.value = els.q.placeholder;
    els.q.setSelectionRange(els.q.value.length, els.q.value.length);
    grow();
    return;
  }
  if (event.key !== "Enter" || event.shiftKey || event.isComposing) return;
  event.preventDefault();
  void run(els.q.value.trim());
});
els.q?.addEventListener("input", grow);
els.nameFilter?.addEventListener("input", () => {
  docName = els.nameFilter.value.trim();
  // No debounce: the list is 483 rows of plain DOM and no request leaves the page,
  // so a keystroke costs a re-render that is far under a frame. A timer here would
  // buy nothing and would make the list lag behind the box.
  refilter();
});
els.reset.addEventListener("click", () => {
  // Including the document scope and the name box: "reset the filters" means the
  // whole corpus, and a filter the panel does not show is exactly the one a reader
  // cannot clear by hand.
  filters = {};
  docName = "";
  if (els.nameFilter) els.nameFilter.value = "";
  // Back to the SHIPPED default rather than to "all". "Réinitialiser" restores the
  // page as it arrives, and the level is part of what arrives; clearing it to the
  // whole corpus would quietly hand a reader the articles and public guides that
  // the default exists to keep out of a search, with the dropdown then
  // disagreeing with what it had been left on.
  // Through setLevel, which also remembers the level, re-syncs the select and
  // refilters, so reset cannot drift from what a change of level does.
  renderFacets();
  setLevel(LEVEL_DEFAULT);
});
for (const [name, pref] of Object.entries(PREFS)) {
  const control = document.getElementById(pref.check);
  control?.addEventListener("change", () => {
    const value = controlValue(control);
    if (pref.fromStore(pref.toStore(value)) === null) return;   // not a value we know
    prefs[name] = value;
    save(name, pref.toStore(value));
    // No question yet: the preference is recorded and the URL follows, and the next
    // search will be ranked with it.
    if (!lastQuestion) { writeUrlState(); return; }
    // Re-asked, not re-ranked here: the ranking lives in the service, and its
    // query-vector cache means this costs no encoder round trip.
    void run(lastQuestion, { push: false });
  });
}
// The tour types its example into the box and searches it, so the result steps point
// at real results; `run` without a history entry, and the reader's own question is
// handed back when the tour ends.
els.tourBtn?.addEventListener("click", () => startTour({
  question: () => lastQuestion,
  closePopups: () => closeFacetPops(),
  search: (question) => { els.q.value = question; grow(); return run(question, { push: false }); },
}));
els.viewToggle?.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-view]");
  if (!button || button.dataset.view === view) return;
  view = button.dataset.view;
  save(VIEW_KEY, view);
  refreshViewToggle();
  paintResults();
});
/**
 * The PDF a link in the result list would lead to, directly or through the viewer.
 *
 * Read off the href rather than out of a data attribute, so every way into a
 * document counts without each renderer having to remember to mark its links: the
 * title, the page number, the group heading and the PDF menu all end up here.
 *
 * @param {EventTarget} target
 * @returns {string|null} the PDF's URL, or null for anything else
 */
function pdfBehind(target) {
  const link = target?.closest?.("a[href]");
  if (!link || !els.results?.contains(link)) return null;
  const url = new URL(link.getAttribute("href"), location.href);
  const here = new URL("./", location.href);
  const path = url.pathname.startsWith(here.pathname)
    ? url.pathname.slice(here.pathname.length) : url.pathname;
  if (path.startsWith("pdf/")) return link.getAttribute("href");
  if (!path.startsWith("view.html")) return null;
  const doc = index?.docs?.[Number(url.searchParams.get("doc"))];
  return doc ? pdfHref(doc) : null;
}

/* Hovering or tabbing onto a result is the strongest hint short of a click, so that
   document jumps the queue. Pointerover rather than mouseover so a touch that starts
   a tap counts, and focusin so a reader on the keyboard gets the same head start. */
for (const type of ["pointerover", "focusin"]) {
  els.results?.addEventListener(type, (event) => {
    const href = pdfBehind(event.target);
    if (href) prefetcher?.promote(href);
  });
}
/* And the click is the commitment: everything speculative is dropped so the
   navigation, and then pdf.js's own range requests, have the connection to
   themselves. Capture phase, because the page is about to leave. */
els.results?.addEventListener("click", (event) => {
  const href = pdfBehind(event.target);
  if (href) prefetcher?.commit(href);
}, true);

document.addEventListener("langchange", refreshChrome);
addEventListener("popstate", (event) => {
  // The URL is authoritative and the state object is the fast path. They agree for
  // every entry this page pushed; an entry restored from a bookmark or a pasted link
  // has no state object at all, which is why the URL is parsed rather than trusted
  // to be there.
  const fromUrl = readUrlState();
  const q = event.state?.q ?? fromUrl.question;
  filters = event.state?.filters ?? fromUrl.filters;
  // The URL says how that entry was ranked, and stepping back has to reproduce it
  // rather than re-rank an old answer the way the reader happens to like it now.
  for (const name of Object.keys(PREFS)) {
    prefs[name] = fromUrl.prefs[name] ?? PREFS[name].def;
    showPref(name);
  }
  // The saved preference is deliberately not consulted here. An entry that carries
  // no level was taken at the default, and stepping back to it has to show the list
  // it showed, not the list a reader who has since changed the dropdown would get.
  // `event.state.filters` already holds the level for entries this page pushed; the
  // line below is what makes a bookmarked or pasted URL agree with them.
  guidelineLevel = fromUrl.level || LEVEL_DEFAULT;
  syncLevelFilter();
  docName = fromUrl.name;
  if (els.nameFilter) els.nameFilter.value = docName;
  renderFacets();   // the controls have to show the filters this entry was taken with
  refreshAdvanced();
  if (!els.q) { paintResults(); return; }   // browse page: only the filters can change
  els.q.value = q;
  grow();
  // run() with an empty question clears the answer, so stepping back past the first
  // search lands on the page as it was before it.
  void run(q, { push: false });
});

(async function boot() {
  refreshChrome();
  setStatus(t("loading_index"));
  try {
    // meta.json only, on both pages: facets, titles and the document list. Neither
    // page ranks, so neither downloads a matrix; the service holds those, and it
    // refuses to start on a width mismatch rather than leaving the page to notice.
    index = await loadIndex(undefined, { vectors: false });
    searchable = PAGE === "search";
    setStatus("");
    // Filters first: they have to be in place before anything paints, or a link
    // carrying both would flash the unfiltered corpus and then narrow it.
    const initial = readUrlState();
    filters = initial.filters;
    docName = initial.name;
    if (els.nameFilter) els.nameFilter.value = docName;
    // A shared link beats this browser's preference: two readers opening the same
    // URL have to see the same list in the same order, which is the whole point of
    // the parameter being there.
    for (const name of Object.keys(PREFS)) prefs[name] = initial.prefs[name] ?? prefs[name];
    // Same rule for the level, over the preference read from localStorage at the top
    // of this file. Applied before the first paint, like the filters above, so the
    // page never shows the whole corpus and then takes documents away.
    if (initial.level !== null) guidelineLevel = initial.level;
    // A question scoped to ONE document is a question about that document, so the
    // corpus-wide level filter has no say in it. The scope arrives from the viewer's
    // own search box, where the reader is already inside the document they mean, and
    // most of what is read that way (an article, a public guide, a classification)
    // sits below `family`: applying a saved narrower level here answers "search this
    // document" with an empty list, under a note saying the corpus was narrowed to a document
    // that is not in it, and nothing on the page explains which filter did it. An
    // explicit `?level=` still wins, because that reader said so in the URL.
    if (initial.filters.file && initial.level === null) guidelineLevel = LEVEL_ALL;
    syncLevelFilter();
    // Paints as a side effect: the document list on the browse page, and the empty
    // search page with its hint on the other.
    refreshChrome();
    if (PAGE === "browse") {
      // A filtered list is a place, so its URL is normalised exactly as the search
      // page normalises a filtered question. A `q=` that arrived here (a hand-edited
      // link, a stale bookmark) is dropped by writeUrlState, since lastQuestion is
      // empty on this page and never becomes anything else.
      writeUrlState();
    } else if (initial.question) {
      els.q.value = initial.question;
      grow();
      void run(initial.question, { push: false });
    } else if (isNarrowed()) {
      // A filtered listing with no question still deserves a normalised URL, so that
      // a shared "?query=" or a stale parameter comes back as this page writes it.
      writeUrlState();
    }
  } catch (error) {
    setStatus(t("index_failed", { error: error.message || error }), true);
  }
})();
