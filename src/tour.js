/* A seven-step guided tour of the search page.

   Search page only: browse.html does not load it, and its ids (the query box, the
   landing hint) do not exist there. `show()` tolerates a missing target anyway, so a
   step whose element is removed degrades to a caption rather than to an exception.

   It exists because the two things that make this site different from a document
   list are both invisible until explained: a query is matched on MEANING, so
   French wording finds English text, and a hit opens the actual PDF page with the
   matching passage highlighted rather than a reconstructed web page.

   Every step runs in two beats. First it SHOWS the element: scrolled into view,
   pulsing, with an arrow pointing at it, and nothing else moving, so the reader
   finds it before anything changes. Only then does it DO what the element does:
   open the "?" panel, unfold the filters, type the example question, run it. Doing
   both at once is what made the first version hard to follow: the panel opened in
   the same instant the eye was still looking for the button.

   Implemented with a scroll-and-pulse on real elements rather than an overlay with
   cut-outs. An overlay has to track element geometry on resize and scroll and gets
   it wrong on mobile; pulsing the element itself cannot desynchronise from it. The
   arrow is the one piece that does need geometry, so it is placed once the scroll
   has settled and again on resize, and a stale position costs a misplaced arrow,
   not a misplaced hole. Under prefers-reduced-motion the stylesheet stops the pulse
   and the arrow's bounce, and this file skips the pauses and the typing, so the tour
   still reads as a sequence of captions for readers who do not want movement. */

import { t } from "./i18n.js";

/** The script, keyed to element ids on the page. The captions are i18n keys
    (tour_*), so a corpus overlay can rewrite them around its own sources and its
    own worked example (tour_example) without touching the steps.

    `id` is an element id, or a selector for what has no id of its own (the first
    result: the list itself starts under the sticky header once scrolled to, so an arrow
    at its top points at the logo). A caption's lines are separated by "\n" and
    rendered as such. `act` is what the step
    does once its element has been shown: "type" writes the example into the box,
    "open" unfolds the target, "search" runs that example so the result steps point at
    real results. */
const FIRST_RESULT = "#results > :first-child";

const STEPS = [
  { id: "q", act: "type", text: "tour_ask" },
  { id: "q", text: "tour_metadata" },
  { id: "syntax-help", act: "open", text: "tour_syntax" },
  { id: "advanced", act: "open", text: "tour_advanced" },
  { id: "landing-hint", text: "tour_browse" },
  { id: "search-btn", act: "search", then: FIRST_RESULT, text: "tour_search" },
  { id: FIRST_RESULT, text: "tour_result" },
];

/* How long an element is shown before the step acts on it, and the typing speed.
   Long enough for the eye to land on the arrow, short enough that "Suivant" never
   feels like it is waiting on an animation. */
const SHOW_MS = 1400;
const TYPE_MS = 40;

let bubble = null;
let arrow = null;
let step = 0;
/* Bumped on every step and on teardown. Each step's sequence is async (pauses,
   typing, a search), and a reader who presses "Suivant" mid-sequence must not have
   the previous step's typing carry on into the next one. */
let generation = 0;
/* Supplied by app.js: the tour cannot reach `run`, nor the facet popups, itself. */
let hooks = { question: () => "", search: async () => {}, closePopups: () => {} };
let savedQuestion = "";
/* Whether the tour has put its own question in the box (typed it, or searched it), so
   teardown knows there is something to give back. */
let touched = false;
/* The tour's own search, once started: a promise, so a step reached while it is still
   in flight waits for its results instead of pointing at a list that is not there. */
let searched = null;
/** Panels placeBubble shortened to clear the caption, given their own height back on leaving. */
const shortened = [];

const find = (key) => document.getElementById(key) || document.querySelector(key);
const calm = () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, calm() ? 0 : ms));

/**
 * Put the page back in the state a first-time reader sees, keeping open only what
 * holds `target`.
 *
 * Run at the start of the tour and again before every step. At the start, because a
 * reader who presses "Visite guidée" with the "?" panel open and "Recherche avancée"
 * unfolded would otherwise get a tour narrated over a page that looks nothing like the
 * one being described, with the panel covering the box the first step points at. On
 * every step, because each step opens what it talks about (the "?" panel, the filter
 * disclosure), and without this the panel opened for step two was still hanging over
 * the page three steps later.
 *
 * Every <details> rather than closeMenus()'s three kinds (i18n.js): that helper runs
 * on every click on the page and must NOT fold "Recherche avancée" away, whereas the
 * tour wants exactly that. The facet popups are not <details> (app.js, facetPopup),
 * so app.js shuts them, through the hook it hands the tour.
 *
 * Unlike the first version, it no longer opens the target's own <details>: opening is
 * the step's second beat (`act: "open"`), after the reader has seen what is opening.
 *
 * @param {Element|null} target  the element the coming step points at, or null
 */
function resetUi(target) {
  for (const details of document.querySelectorAll("details[open]")) {
    if (!target || !details.contains(target) || details === target) details.open = false;
  }
  hooks.closePopups();
  if (target && target.tagName !== "DETAILS") target.closest("details")?.setAttribute("open", "");
}

/** Remove the pulse and the arrow from wherever they are. */
function unmark() {
  document.querySelectorAll(".pulse").forEach((el) => el.classList.remove("pulse"));
  arrow?.remove();
  arrow = null;
  shortened.splice(0).forEach((panel) => panel.style.removeProperty("max-height"));
}

/**
 * Point the arrow at `target` and make it pulse.
 *
 * The arrow sits to the LEFT of a wide element (the box, a result), pointing right,
 * and ABOVE a small control, pointing down: a button's left is its neighbour (the "?"
 * sits right of "Rechercher"), and an arrow there lands on the neighbour's label. Above
 * as well when there is no room on the left, as on a phone where the box spans the
 * width. Positioned in page coordinates, so scrolling carries it with the element.
 */
function mark(target) {
  unmark();
  target.classList.add("pulse");
  arrow = document.createElement("div");
  arrow.className = "tour-arrow";
  arrow.setAttribute("aria-hidden", "true");
  document.body.append(arrow);
  placeArrow(target);
  // Placed first and only then allowed to glide: created with the transition on, it
  // would slide in from the page's top left corner.
  void arrow.offsetWidth;
  arrow.classList.add("tour-arrow-placed");
}

function placeArrow(target) {
  if (!arrow) return;
  const box = target.getBoundingClientRect();
  const size = 28;
  const left = box.width >= 200 && box.left >= size + 8;
  arrow.classList.toggle("tour-arrow-down", !left);
  arrow.textContent = left ? "➜" : "⬇";
  arrow.style.left = `${window.scrollX + (left ? box.left - size - 6
                                               : box.left + Math.min(box.width / 2, 40) - size / 2)}px`;
  arrow.style.top = `${window.scrollY + (left ? box.top + Math.min(box.height, 60) / 2 - size / 2
                                              : box.top - size - 4)}px`;
}

/**
 * Move the caption to the top of the screen when it covers what it is describing.
 *
 * It sits at the bottom by default, which is where it is least in the way, but the
 * "?" panel hangs down from the box and is up to 70vh tall: on most screens the
 * caption was drawn over the panel it was explaining. Checked against the target and
 * against whatever the step opened inside it, after the opening.
 */
function placeBubble(target) {
  if (!bubble) return;
  const wasTop = bubble.classList.contains("tour-bubble-top");
  decideBubble(target);
  // A caption that jumps from one end of the screen to the other reads as a glitch;
  // fading it back in at its new end reads as a move.
  if (bubble.classList.contains("tour-bubble-top") !== wasTop) enter(bubble);
}

/** Replay the entrance animation on `el` (app.css, .tour-enter). */
function enter(el) {
  el.classList.remove("tour-enter");
  void el.offsetWidth;  // a reflow, so removing and re-adding the class restarts it
  el.classList.add("tour-enter");
}

function decideBubble(target) {
  bubble.classList.remove("tour-bubble-top");
  if (!target) return;
  // `:scope[open]` as well: the target is often the <details> itself, and a descendant
  // selector alone never matches its own panel.
  const panels = [...target.querySelectorAll(":scope[open] > :not(summary), [open] > :not(summary)")];
  const covered = [target, ...panels]
    .map((el) => el.getBoundingClientRect())
    .filter((r) => r.width && r.height);
  const overlaps = (b) => covered.some((r) => r.top < b.bottom && r.bottom > b.top);
  if (!overlaps(bubble.getBoundingClientRect())) return;
  bubble.classList.add("tour-bubble-top");
  // If the top covers it too (a panel taller than the space above the caption), the
  // bottom was the lesser evil, since the element's own head is what is near the top.
  if (overlaps(bubble.getBoundingClientRect())) {
    const top = covered.reduce((m, r) => Math.min(m, r.top), Infinity);
    if (top < bubble.getBoundingClientRect().bottom) bubble.classList.remove("tour-bubble-top");
  }
  // Still covered (the "?" panel on a phone is taller than the screen leaves): shorten
  // the panel to the room above the caption. It scrolls on its own, so nothing in it is
  // lost, where under the caption its last half could not be reached at all.
  const room = bubble.getBoundingClientRect().top - 8;
  if (bubble.classList.contains("tour-bubble-top")) return;
  for (const panel of panels) {
    const r = panel.getBoundingClientRect();
    if (r.height && r.bottom > room && room - r.top > 80) {
      panel.style.maxHeight = `${Math.floor(room - r.top)}px`;
      shortened.push(panel);
    }
  }
}

/** Write `text` into the box one character at a time, as a reader would. */
async function type(box, text, gen) {
  box.focus({ preventScroll: true });
  box.value = "";
  if (calm()) {
    box.value = text;
    box.dispatchEvent(new Event("input"));
    return;
  }
  for (const char of text) {
    if (gen !== generation) return;
    box.value += char;
    // `input` is what grows the textarea (app.js, grow).
    box.dispatchEvent(new Event("input"));
    await wait(TYPE_MS);
  }
}

const example = () => t("tour_example");

/** Run the example, once: a second call waits on the first rather than asking again. */
function ensureSearched() {
  touched = true;
  searched ||= hooks.search(example());
  return searched;
}

/**
 * Put the page in the state step `index` is narrated over, whatever came before it.
 *
 * A step cannot assume the one before it finished: "Suivant" pressed mid-typing used
 * to leave "Wh" in the box, and the search step then ran "Wh" and got the red
 * too-short message. Nor that it is reached going forward, now that there is a way
 * back. So the state is a function of the index: past the typing step the box holds
 * the whole example; before the search step there are no results; from it on, there
 * are the example's.
 *
 * @param {number} index  the step about to be shown
 * @param {object[]} steps  the script in the current language
 */
async function settle(index, steps) {
  const typing = steps.findIndex((s) => s.act === "type");
  const searching = steps.findIndex((s) => s.act === "search");
  if (index < searching && searched) {
    await searched;
    searched = null;
    await hooks.search("");
  }
  const box = document.getElementById("q");
  if (box && index > typing && box.value !== example()) {
    touched = true;
    box.value = example();
    box.dispatchEvent(new Event("input"));
  }
  if (index > searching) await ensureSearched();
}

/** The second beat of a step: whatever the element does. */
async function act(spec, target, gen) {
  if (spec.act === "type" && target) {
    touched = true;
    await type(target, example(), gen);
  } else if (spec.act === "open" && target?.tagName === "DETAILS") {
    target.open = true;
  } else if (spec.act === "search") {
    await ensureSearched();
  }
  if (gen !== generation) return;
  const next = spec.then ? find(spec.then) : target;
  if (next && next !== target) {
    mark(next);
    next.scrollIntoView({ behavior: calm() ? "auto" : "smooth", block: "center" });
    await wait(400);
    if (gen !== generation) return;
    placeArrow(next);
  }
  placeBubble(next);
}

function teardown() {
  generation += 1;
  bubble?.remove();
  bubble = null;
  unmark();
  document.body.classList.remove("touring");
  window.removeEventListener("resize", onResize);
  document.removeEventListener("keydown", onKey);
  // The tour opened what it described; the page it hands back is the one it started
  // from, not one with a panel left hanging over the results.
  resetUi(null);
  // Nor with the tour's example question in place of the reader's own: give back what
  // was on screen before, which for a first visit is the empty box and the corpus list.
  if (touched) {
    touched = false;
    const pending = searched;
    searched = null;
    // After the tour's own search lands, or its late answer would repaint over this one.
    void Promise.resolve(pending).then(() => hooks.search(savedQuestion));
  }
}

/* Escape leaves the tour, as it leaves every other floating panel on the page. It has
   to, now that the tour types into the box: a tour left running in the background would
   overwrite whatever the reader started typing instead. */
function onKey(event) {
  if (event.key === "Escape" && bubble) teardown();
}

function onResize() {
  const target = document.querySelector(".pulse");
  if (target) { placeArrow(target); placeBubble(target); }
}

async function show() {
  const steps = STEPS;
  if (step >= steps.length) return teardown();
  generation += 1;
  const gen = generation;
  const spec = steps[step];

  unmark();
  renderBubble(t(spec.text), steps.length);
  await settle(step, steps);
  if (gen !== generation) return;
  const target = find(spec.id);
  resetUi(target);
  if (!target) return;

  // Beat one: show the element and nothing else.
  target.scrollIntoView({ behavior: calm() ? "auto" : "smooth", block: "center" });
  mark(target);
  placeBubble(target);
  // The smooth scroll moves the element after the arrow was placed; place it again
  // once the scroll has settled.
  await wait(400);
  if (gen !== generation) return;
  placeArrow(target);
  await wait(SHOW_MS - 400);
  if (gen !== generation) return;
  // Beat two: what it does.
  await act(spec, target, gen);
}

function renderBubble(text, total) {
  if (!bubble) {
    bubble = document.createElement("div");
    bubble.className = "tour-bubble";
    bubble.setAttribute("role", "dialog");
    bubble.setAttribute("aria-live", "polite");
    bubble.setAttribute("aria-label", t("tour_start"));
    document.body.append(bubble);
  }
  bubble.textContent = "";
  enter(bubble);
  const p = document.createElement("p");
  p.textContent = text;
  const actions = document.createElement("div");
  actions.className = "tour-actions";
  // Where the reader is in the script: without it, "Suivant" is a button that might
  // go on forever, and a reader who does not know how many steps there are quits at
  // the second.
  const count = document.createElement("span");
  count.className = "tour-count";
  count.textContent = `${step + 1} / ${total}`;
  const skip = document.createElement("button");
  skip.type = "button";
  skip.className = "plain tour-skip";
  skip.textContent = t("tour_skip");
  skip.addEventListener("click", (event) => { event.stopPropagation(); teardown(); });
  const next = document.createElement("button");
  next.type = "button";
  next.className = "plain tour-next";
  next.textContent = step === total - 1 ? t("tour_done") : t("tour_next");
  // The click stops here. A step whose target is one of the bar menus (the "?" beside
  // the box, for one) opens that menu, and the page closes every menu on any click
  // outside one (closeMenus, in i18n.js): let this click reach the document and the
  // step pulses a panel that shut itself during the same event.
  next.addEventListener("click", (event) => { event.stopPropagation(); step++; void show(); });
  const prev = document.createElement("button");
  prev.type = "button";
  prev.className = "plain tour-prev";
  prev.textContent = t("tour_prev");
  prev.addEventListener("click", (event) => { event.stopPropagation(); step--; void show(); });
  // The last step has nothing to skip, so it offers only the one way out; the first
  // has nothing to go back to.
  actions.append(count);
  if (step < total - 1) actions.append(skip);
  if (step > 0) actions.append(prev);
  actions.append(next);
  bubble.append(p, actions);
  next.focus({ preventScroll: true });
}

/**
 * Start (or restart) the tour, from a page put back to its untouched state.
 *
 * @param {{question: () => string, search: (q: string) => Promise<void>,
 *          closePopups: () => void}} [given]
 *   app.js's way in: the reader's current question, to give back at the end, a
 *   search that runs without adding a history entry, and a way to shut the facet
 *   popups.
 */
export function startTour(given) {
  if (given) hooks = given;
  if (bubble) teardown();
  savedQuestion = hooks.question() || "";
  touched = false;
  step = 0;
  window.scrollTo({ top: 0 });
  searched = null;
  window.addEventListener("resize", onResize);
  document.addEventListener("keydown", onKey);
  document.body.classList.add("touring");
  void show();
}
