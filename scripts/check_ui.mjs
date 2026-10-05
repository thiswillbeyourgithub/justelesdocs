/**
 * Browser gate for the site: load the real pages in a real browser and check the
 * things that only exist once a page is rendered.
 *
 * check_search.mjs covers the ranking arithmetic headlessly and cannot see any of
 * this: whether the modules load, whether the DOM wiring works, and above all
 * whether a highlight lands on the words it describes. That last one is the reason
 * this file exists. The first browser run of the viewer showed every highlight
 * mirrored about the middle of the page, because chunk.py stores boxes in PyMuPDF's
 * top-down page space while pdf.js's convertToViewportPoint expects PDF user space,
 * which grows upward. It is nearly invisible on a dense page: there is text under
 * the box either way, and the result looks like a slight offset rather than a flip.
 *
 * So the geometry check here is deliberately not a re-implementation of the
 * viewer's arithmetic (which would pass a flip by construction). It reads back the
 * highlight rectangle the browser actually rendered, asks pdf.js which text items
 * fall inside it, and asserts that those words appear in the passage the highlight
 * is supposed to be marking. Flipped, offset or mis-scaled, the words under the box
 * stop being the passage's words.
 *
 * Not part of the deploy chain: it needs a browser and a served tree. Run it after
 * touching anything in src/.
 *
 *   uv run scripts/dev_server.py &                      # dist/ on :8649, with the real CSP
 *   PW=<path to a playwright install> SITE=http://127.0.0.1:8649 \
 *     node scripts/check_ui.mjs
 *
 * PW is the playwright package to load (this repo has no node_modules and no
 * package.json, by the same decision that keeps the site free of a build step).
 * With the search service reachable through SITE (dev_server.py proxies /api/search
 * to it) the search flow is exercised too, against the real ranker and the real
 * encoder; without one those sections report themselves as skipped rather than
 * failing. Skipped only when the service is genuinely ABSENT: SITE's
 * /api/search/health is probed first, and a service that answers health but
 * gives the page no rows is a failure, not a skip (that is the "every query
 * says service unavailable" deploy this gate used to wave through).
 * REQUIRE_SEARCH=1 turns the skip itself into a failure, for a run whose point
 * is the search flow. Nothing here stubs a vector any more: the page does not rank, so a
 * stubbed encoder would exercise nothing the reader sees.
 *
 * Written by Claude Code.
 */

import { createRequire } from "node:module";
import { loadScenarios } from "./lib/scenarios.mjs";

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PW || "playwright");

const SITE = (process.env.SITE || "http://127.0.0.1:8649").replace(/\/$/, "");

// The corpus half of this gate (scripts/lib/scenarios.mjs): real questions and the
// values only a reader of that corpus would know to check. Without it, every
// question falls back to a served document's own title, which the search always
// has something to say about, and the corpus-only checks report themselves skipped.
const SCENARIO = loadScenarios().ui || {};
const servedMeta = await (await fetch(`${SITE}/index/meta.json`)).json();
const FALLBACK_QUERY = (servedMeta.documents || [])
  .map((d) => d.title).find((title) => title && title.length > 12) || "document";
const query = (name) => SCENARIO.queries?.[name] || FALLBACK_QUERY;
// The level ladder the select offers: the corpus's tiers narrowest first, its last
// tier ("not part of any level") left out, and "all" at the end. Its column names
// the control. NARROW, NEXT are its first two rungs.
const TIERS = servedMeta.guideline_tiers || [];
const LEVELS = [...TIERS.slice(0, -1), "all"];
const LEVEL_ID = `#facet-${servedMeta.tier_field || "guideline"}`;
const [NARROW, NEXT] = LEVELS;
const SHOTS = process.env.SHOTS || "";

let failures = 0;
const check = (ok, label, detail = "") => {
  if (ok) console.log(`  ok   ${label}${detail ? ` (${detail})` : ""}`);
  else { failures++; console.error(`  FAIL ${label}${detail ? `: ${detail}` : ""}`); }
};
const note = (s) => console.log(`    ${s}`);

const browser = await chromium.launch();

const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
const consoleErrors = [];
const cspViolations = [];
const badResponses = [];
// CSP violations are reported on the console at a level that varies by browser
// version, so they are matched by text rather than by level, and counted apart:
// "style attribute blocked" is a specific, silent-looking failure that deserves its
// own line rather than being buried among page errors.
const watchConsole = (m) => {
  const text = m.text();
  if (/Content Security Policy|Refused to (load|apply|execute)/i.test(text)) cspViolations.push(text.slice(0, 160));
  else if (m.type() === "error") consoleErrors.push(text);
};
page.on("console", watchConsole);
page.on("pageerror", (e) => consoleErrors.push(`pageerror: ${e.message}`));
page.on("response", (r) => { if (r.status() >= 400 && !r.url().includes("favicon")) badResponses.push(`${r.status()} ${r.url()}`); });

console.log("search page");
await page.goto(`${SITE}/index.html`, { waitUntil: "networkidle" });
check((await page.title()).length > 0, "the page has a title", await page.title());
check(await page.locator("#q").isVisible(), "the search box is there");
// The search page starts empty. It used to print the whole corpus under the box,
// which made the first screen a list of documents rather than an invitation to ask.
check((await page.locator("#results > li").count()) === 0,
      "and nothing is listed until something is asked");
check(await page.locator("#landing-hint").isVisible(),
      "with a line pointing at the document list instead");
/* Both ways into the corpus are in the bar of every page, in the same order, with
   the page you are on marked. Asserted by position rather than by count, because
   the bug this replaces was a bar that showed whichever link you were not on: it
   passed a count of one everywhere and still left the reader guessing. */
const navHrefs = await page.locator("header a.nav").evaluateAll(
  (els) => els.map((el) => `${el.getAttribute("href")}${el.hasAttribute("aria-current") ? "*" : ""}`));
check(JSON.stringify(navHrefs) === JSON.stringify(["./*", "browse.html"]),
      "the bar offers the search and the documents, and marks this one",
      navHrefs.join(" "));

console.log("document page");
await page.goto(`${SITE}/browse.html`, { waitUntil: "networkidle" });
check((await page.locator("#q").count()) === 0,
      "the document page has no search box: it is a list, not a question");
check((await page.locator("#facets .facet").count()) > 0, "facet controls rendered",
      `${await page.locator("#facets .facet").count()} facets`);
const browseNav = await page.locator("header a.nav").evaluateAll(
  (els) => els.map((el) => `${el.getAttribute("href")}${el.hasAttribute("aria-current") ? "*" : ""}`));
check(JSON.stringify(browseNav) === JSON.stringify(["./", "browse.html*"]),
      "the same two links are here, in the same order, with the other one marked",
      browseNav.join(" "));

/* The name filter. The only way to find a document by its title when the encoder
   is down, so it is checked here rather than trusted: it must narrow the list as
   it is typed, fold accents, survive a reload through the URL, and be cleared by
   the reset button along with the facets. */
const rowCount = () => page.locator("#results > li").count();
const allRows = await rowCount();
check(allRows > 100, "the document page lists the whole corpus to begin with", String(allRows));
// A word of a served title unless the corpus names one, so the filter matches.
const NAME_FILTER = SCENARIO.name_filter
  || FALLBACK_QUERY.toLowerCase().split(/\s+/).find((w) => w.length > 5) || FALLBACK_QUERY;
await page.fill("#name-filter", NAME_FILTER);
await page.waitForFunction(
  (total) => document.querySelectorAll("#results > li").length < total, allRows);
const named = await rowCount();
check(named > 0 && named < allRows, "typing a name narrows the list", `${named} of ${allRows}`);
check(new URL(page.url()).searchParams.get("name") === NAME_FILTER,
      "and the URL carries it, so a filtered list is a link");
const accented = await page.evaluate(async () => {
  const input = document.getElementById("name-filter");
  input.value = "depistage";
  input.dispatchEvent(new Event("input", { bubbles: true }));
  await new Promise((r) => setTimeout(r, 50));
  return document.querySelectorAll("#results > li").length;
});
check(accented > 0, "and it folds accents: 'depistage' finds 'dépistage'", String(accented));
await page.reload({ waitUntil: "networkidle" });
check((await page.inputValue("#name-filter")) === "depistage",
      "a reload comes back with the box filled from the URL");
await page.click("#reset-btn");
check((await page.inputValue("#name-filter")) === "" && (await rowCount()) === allRows,
      "and resetting the filters clears it too");

// The facet panel is where the manifest's slugs meet a reader. Three things can
// only be seen here: that the year control is a slider rather than a dropdown,
// that options read as language rather than as filing codes, and that dragging
// the slider actually filters. The listing below is filtered by all of it, so a
// broken option is a document page showing nothing.
//
// This page's panel is not a disclosure at all: filtering IS the task here, so the
// knobs are open next to the name box rather than behind a summary.
check((await page.locator("#filters").evaluate((el) => el.tagName)) === "SECTION",
      "the corpus listing's filters are open, not behind a disclosure");
check((await page.locator("#filters #name-filter").count()) === 1,
      "with the name box inside the same panel, not floating above it");
// textContent, not innerText: assertions about WORDING should not also depend on
// the caption being laid out. The drag and the ticks below are what check the
// panel is really interactive.
const choices = (field) => page.locator(`#facet-${field} .choice span`).allTextContents();
const issuerChoices = await choices("issuer");
if (SCENARIO.issuer_label) {
  const wanted = new RegExp(SCENARIO.issuer_label);
  check(issuerChoices.some((o) => wanted.test(o)), "issuer choices carry the acronym's meaning",
        JSON.stringify(issuerChoices.find((o) => wanted.test(o)) || issuerChoices[0]));
} else note("issuer label: skipped (no ui.issuer_label scenario)");
if (SCENARIO.issuer_absent) {
  check(!issuerChoices.includes(SCENARIO.issuer_absent),
        `and ${SCENARIO.issuer_absent} is not a second choice for the same body`);
}
// The values a URL filter below uses, read off the controls rather than written
// down, unless the corpus names one.
const firstValue = (field) => page.locator(`#facet-${field} input[type=checkbox]`).first().getAttribute("value");
const ISSUER = SCENARIO.issuer || await firstValue("issuer");
// The country is one the issuer's own documents carry: the URL-state check below
// adds it ON TOP of the issuer filter, and an issuer with no document in the first
// listed country (a national agency and another country) leaves an empty list with nothing to click.
const COUNTRY = (servedMeta.documents || [])
  .find((d) => d.issuer === ISSUER && d.country)?.country || await firstValue("country");
const LANGUAGE = await firstValue("language");
// A bare code (FR, fr) in a choice is a slug that fell back to itself: the manifest
// stores codes, and the reader is owed a name.
const countryChoices = await choices("country");
check(countryChoices.length > 0 && !countryChoices.some((o) => /^[A-Z]{2,3}$/.test(o)),
      "countries read as names, not two-letter codes", countryChoices.join(", "));
const languageChoices = await choices("language");
check(languageChoices.length > 0 && !languageChoices.some((o) => /^[a-z]{2}$/.test(o)),
      "so do languages", languageChoices.join(", "));
const typeChoices = await choices("doc_type");
// Counted off the served MANIFEST.tsv rather than written down here: the corpus
// grows, and a hard number turns every added document into a failure of this gate
// instead of a failure of the thing it is meant to watch, which is a multi-valued
// cell ("pnds;recommandation") arriving in the popup as one choice with a
// semicolon in it rather than as two.
const manifestTsv = await (await fetch(`${SITE}/MANIFEST.tsv`)).text();
const manifestRows = manifestTsv.trimEnd().split("\n").map((line) => line.split("\t"));
const typeColumn = manifestRows[0].indexOf("doc_type");
const typesOnDisk = new Set(manifestRows.slice(1)
  .flatMap((row) => (row[typeColumn] || "").split(";"))
  .map((value) => value.trim())
  .filter(Boolean));
check(typeChoices.length === typesOnDisk.size && !typeChoices.some((o) => o.includes(";")),
      "and multi-valued cells are split into separate choices",
      `${typeChoices.length} document types, ${typesOnDisk.size} in MANIFEST.tsv`);
// Checkboxes rather than a <select>, which is the whole point of the rework: a
// reader wanting two themes must be able to tick two. Two values in one facet have
// to mean EITHER, because no document carries two themes at once and an AND would
// answer nothing at all.
// The values are behind a button now, one popup per facet, so every interaction
// below opens the one it needs first. `check()` on a hidden box would pass in
// Playwright's own way (it can tick what it cannot see), which would say nothing
// about whether a reader can reach it.
const openFacet = async (target, field) => {
  await target.click(`#facet-${field}-btn`);
  await target.waitForSelector(`#facet-${field}-pop`, { state: "visible", timeout: 5000 });
};
check((await page.locator("#facet-topic-pop").isVisible()) === false,
      "a facet's values start behind a button, not spread across the page");
await openFacet(page, "topic");
const topics = page.locator("#facet-topic input[type=checkbox]");
check((await topics.count()) > 1, "a facet offers its values as checkboxes",
      `${await topics.count()} topics`);
// The box that searches the values, which is what makes thirty topics usable.
// A piece of the first topic's own label unless the corpus names one, so the
// filter is known to match something.
const topicSearch = SCENARIO.topic_search
  || (await choices("topic"))[0].toLowerCase().split(/\s+/)[0].slice(0, 5);
await page.fill("#facet-topic-search", topicSearch);
await page.waitForTimeout(150);
const visibleTopics = await page.locator("#facet-topic .choice:visible").count();
check(visibleTopics > 0 && visibleTopics < (await topics.count()),
      "typing in the popup narrows the values it offers",
      `${visibleTopics} of ${await topics.count()}`);
await page.fill("#facet-topic-search", "zzz-nothing-matches");
await page.waitForTimeout(150);
check(await page.locator("#facet-topic-pop .facet-empty").isVisible(),
      "and it says so when nothing matches rather than showing an empty box");
await page.fill("#facet-topic-search", "");
await page.waitForTimeout(150);
await topics.nth(0).check();
await page.waitForTimeout(250);
const oneTopic = await rowCount();
await topics.nth(1).check();
await page.waitForTimeout(250);
const twoTopics = await rowCount();
check(oneTopic > 0 && twoTopics > oneTopic, "ticking a second value widens the list, so values are ORed",
      `${oneTopic} -> ${twoTopics}`);
const urlTopics = new URL(page.url()).searchParams.getAll("topic");
check(urlTopics.length === 2, "and both values travel in the URL", urlTopics.join(" + "));
// What the closed button says is the only trace of an active filter once the popup
// is away, so it has to carry the count.
const topicButton = await page.locator("#facet-topic-btn").innerText();
check(/·\s*2/.test(topicButton), "the closed button says how many values are on",
      topicButton);
await page.reload({ waitUntil: "networkidle" });
await page.waitForTimeout(400);
check((await page.locator("#facet-topic input:checked").count()) === 2 && (await rowCount()) === twoTopics,
      "a reload comes back with both boxes ticked and the same list");
// Ticked values are pulled to the top when the popup opens, so a reader who set a
// filter days ago sees it without scrolling a list of thirty.
await openFacet(page, "topic");
const firstTwo = await page.locator("#facet-topic .choice").evaluateAll(
  (els) => els.slice(0, 2).map((el) => el.querySelector("input").checked));
check(firstTwo.every(Boolean), "and what is ticked is at the top of the list",
      JSON.stringify(firstTwo));
// Escape puts the menu away without undoing the ticks: the filter is already in
// force, and a reader pressing Escape means "close this", not "undo that".
await page.keyboard.press("Escape");
await page.waitForTimeout(150);
check(!(await page.locator("#facet-topic-pop").isVisible())
      && (await page.locator("#facet-topic input:checked").count()) === 2,
      "Escape closes the popup and leaves the filter in force");
await page.click("#reset-btn");
await page.waitForTimeout(250);
check((await page.locator("#facet-topic input:checked").count()) === 0,
      "and reset unticks them");

check((await page.locator("#facet-year").getAttribute("type")) === "range",
      "the year facet is a slider, not a dropdown");
check((await page.locator("#facet-year-to").getAttribute("type")) === "range",
      "with a second handle for the upper bound");
// One slider with two ends, not two sliders: both inputs share a track, and the two
// handles sit on the same line. Checked by geometry as well as by markup, because
// the stacked version this replaces also had both inputs and looked like two rows.
check((await page.locator(".range-track input[type=range]").count()) === 2,
      "both handles live on one track");
const handleTops = await page.locator(".range-track input[type=range]")
  .evaluateAll((els) => els.map((el) => Math.round(el.getBoundingClientRect().top)));
check(handleTops.length === 2 && handleTops[0] === handleTops[1],
      "and they are drawn on the same line", handleTops.join(" vs "));
const yearMin = Number(await page.locator("#facet-year").getAttribute("min"));
const yearMax = Number(await page.locator("#facet-year").getAttribute("max"));
check(yearMax > yearMin && yearMax - yearMin > 10, "spanning the corpus's own years",
      `${yearMin}-${yearMax}`);
const allDocs = await page.locator("#results > li").count();
// fill() on a range input sets the value and fires input+change, which is what the
// reader's drag does. The listing below has to shrink, or the range is decorative.
await page.locator("#facet-year").fill(String(yearMax - 2));
await page.waitForTimeout(250);
const recent = await page.locator("#results > li").count();
check(recent > 0 && recent < allDocs, "dragging it narrows the corpus listing",
      `${allDocs} -> ${recent} from ${yearMax - 2}`);
check((await page.locator(".range-readout").innerText()).includes(String(yearMax - 2)),
      "and the readout says what the range is now",
      await page.locator(".range-readout").innerText());
// The lit segment is the only thing that says which part of the track is selected,
// and it is positioned from JS, so it can silently stop following the handles.
// Measured in pixels against where the browser draws each thumb, because both ways
// this went wrong were geometry that fractions alone cannot show: a fill drawn over
// the whole box overshot the handles by half a thumb, and a WebKit thumb sat above
// the rail. A native thumb's centre travels from thumb/2 to width - thumb/2.
const span = await page.evaluate(() => {
  const track = document.querySelector(".range-track");
  const fill = track.querySelector(".range-fill").getBoundingClientRect();
  const box = track.getBoundingClientRect();
  const thumb = parseFloat(getComputedStyle(track).getPropertyValue("--thumb")) *
                parseFloat(getComputedStyle(document.documentElement).fontSize);
  const centre = (input) => {
    const frac = (input.value - input.min) / (input.max - input.min);
    return box.left + thumb / 2 + (box.width - thumb) * frac;
  };
  const [lo, hi] = track.querySelectorAll("input");
  return { fillLeft: fill.left, fillRight: fill.right, lo: centre(lo), hi: centre(hi),
           fillMid: fill.top + fill.height / 2, boxMid: box.top + box.height / 2 };
});
check(Math.abs(span.fillLeft - span.lo) < 1.5 && Math.abs(span.fillRight - span.hi) < 1.5,
      "and the lit part of the track runs from one handle's centre to the other's",
      `fill ${span.fillLeft.toFixed(1)}-${span.fillRight.toFixed(1)}, ` +
      `handles ${span.lo.toFixed(1)}-${span.hi.toFixed(1)}`);
check(Math.abs(span.fillMid - span.boxMid) < 1,
      "and the rail runs through the middle of the track, where the thumbs are centred",
      `rail at ${span.fillMid.toFixed(1)}, track middle at ${span.boxMid.toFixed(1)}`);
await page.click("#reset-btn");
await page.waitForTimeout(250);
check((await page.locator("#results > li").count()) === allDocs,
      "and reset puts every document back");

/* The guideline level. It ships at `all`, and its default has to be seen rather than
   trusted: a wrong default here is a reader searching a subset of a medical corpus
   without being told. `allDocs` above was counted at that default, so every narrower
   level is checked against it.

   Worth checking in a browser rather than in search.test.mjs because none of it is
   the ranker: the options come from meta.json, the value survives a reload through
   the URL, and the reset button has to agree with the dropdown about what "the
   filters, reset" means. */
const level = page.locator(LEVEL_ID);
check((await level.count()) === 1 && (await level.evaluate((el) => el.tagName)) === "SELECT",
      "the level control is a select at the head of the panel");
const levels = await level.locator("option").evaluateAll((els) => els.map((el) => el.value));
check(JSON.stringify(levels) === JSON.stringify(LEVELS),
      "offering the tiers narrowest first, with 'all' last and 'no' not offered",
      levels.join(", "));
check((await level.inputValue()) === "all",
      "and arriving at the level the site ships with, the whole corpus");

await level.selectOption(NEXT);
await page.waitForTimeout(250);
const familyDocs = await page.locator("#results > li").count();
check(familyDocs < allDocs && familyDocs > 0, "choosing a level narrows the listing",
      `${allDocs} -> ${familyDocs}`);
check(new URL(page.url()).searchParams.get("level") === NEXT,
      "and the URL says so, so a narrowed list is a link");

await level.selectOption(NARROW);
await page.waitForTimeout(250);
const strictDocs = await page.locator("#results > li").count();
check(strictDocs < familyDocs && strictDocs > 0,
      "and the narrowest level narrower still", `${strictDocs} < ${familyDocs}`);
await page.reload({ waitUntil: "networkidle" });
check((await page.inputValue(LEVEL_ID)) === NARROW,
      "a reload comes back at the level the URL named");
check((await page.locator("#results > li").count()) === strictDocs,
      "showing the same list it was sharing");

await page.click("#reset-btn");
await page.waitForTimeout(250);
check((await page.inputValue(LEVEL_ID)) === "all"
      && !new URL(page.url()).searchParams.has("guideline"),
      "and reset restores the shipped default, the whole corpus");

// The point of this page being a page: it renders with no service anywhere. This is
// the half of the site that must keep working when the search service is down.
const browsed = await page.locator("#results > li").count();
check(browsed > 100, "the document page lists the corpus", `${browsed} documents`);
check((await page.locator("#view-toggle").count()) === 0,
      "with no view toggle, there being one row per document and nothing to switch");
const firstRow = await page.locator("#results > li").first().innerText();
check(/\.pdf$/i.test(await page.locator("#results > li a").first().getAttribute("href") || ""),
      "and each row links straight to the PDF, there being no passage to highlight");
check(/page/i.test(firstRow), "and says how long the document is", JSON.stringify(firstRow.replace(/\s+/g, " ").slice(0, 90)));

console.log("bilingual chrome");
await page.goto(`${SITE}/index.html`, { waitUntil: "networkidle" });
const lang0 = await page.getAttribute("html", "lang");
const tagline0 = await page.locator("h1").first().innerText();
await page.click("#lang-btn");
await page.waitForTimeout(200);
check((await page.getAttribute("html", "lang")) !== lang0, "the language button switches the document language",
      `${lang0} -> ${await page.getAttribute("html", "lang")}`);
check((await page.locator("h1").first().innerText()) !== tagline0, "visible text follows the language");
await page.click("#lang-btn");
const theme0 = await page.getAttribute("html", "data-theme");
await page.click("#theme-btn");
await page.waitForTimeout(150);
check((await page.getAttribute("html", "data-theme")) !== theme0, "the theme button cycles",
      `${theme0} -> ${await page.getAttribute("html", "data-theme")}`);
await page.click("#theme-btn"); await page.click("#theme-btn");
await page.click("#tour-btn");
await page.waitForTimeout(500);
check(await page.locator("div[role=dialog]").first().isVisible(), "the tour opens a caption");
await page.keyboard.press("Escape");

console.log("a real query");
let searched = false;
// Is a service there at all? Asked of SITE, so it goes through the same proxy the
// page's own request does. Without this, "no rows" could not tell a machine with no
// service (skip) from a service that answers health and fails every search (bug).
const serviceUp = await fetch(`${SITE}/api/search/health`, { signal: AbortSignal.timeout(5000) })
  .then(async (r) => r.ok && (await r.json()).ok === true).catch(() => false);
note(`search service through SITE: ${serviceUp ? "answering" : "absent"}`);
await page.fill("#q", query("answer"));
await page.press("#q", "Enter");
await page.waitForTimeout(1500);
await page.waitForFunction(() => document.querySelectorAll("#results > *").length > 0, { timeout: 45000 }).catch(() => null);
const rows = await page.locator("#results > *").count();
if (rows === 0 && serviceUp) {
  const status = (await page.locator("#status").innerText().catch(() => "")).trim();
  check(false, "a live search service gives the page results", `health says ok, the page says "${status}"`);
} else if (rows === 0) {
  const status = (await page.locator("#status").innerText().catch(() => "")).trim();
  if (process.env.REQUIRE_SEARCH === "1") {
    check(false, "the search service is reachable through SITE (REQUIRE_SEARCH=1)", `the page says "${status}"`);
  } else {
    note(`no results: "${status}". Search needs the search service reachable through SITE; skipping.`);
  }
  // The site is SUPPOSED to fail this way when no service is reachable: the fetch
  // 502s, the client catches it and says so in #status, and the rest of the page
  // keeps working. Without this, running the gate with no search service behind
  // SITE reports two failures for the one condition the run already announced,
  // and a real regression would be indistinguishable from the expected noise.
  // Only this branch forgives it: with a service present, a 502 on the same URL
  // is a genuine failure and still fails below.
  const isExpectedEmbedFailure = (s) => /\/api\/search|502 \(Bad Gateway\)/.test(s);
  const forgivenConsole = consoleErrors.filter(isExpectedEmbedFailure).length;
  const forgivenResponses = badResponses.filter(isExpectedEmbedFailure).length;
  consoleErrors.splice(0, consoleErrors.length, ...consoleErrors.filter((s) => !isExpectedEmbedFailure(s)));
  badResponses.splice(0, badResponses.length, ...badResponses.filter((s) => !isExpectedEmbedFailure(s)));
  if (forgivenConsole || forgivenResponses) {
    note(`ignoring ${forgivenConsole} console error(s) and ${forgivenResponses} failed request(s) from the absent service`);
  }
} else {
  searched = true;
  check(true, "results rendered", `${rows} rows`);
  const first = (await page.locator("#results > *").first().innerText()).replace(/\s+/g, " ");
  note(`first row: ${first.slice(0, 120)}`);
  // A document often writes an abbreviation once it has spelled the word out, and
  // the top hit can be a page that uses only the abbreviation, so the scenario's
  // regex should accept both: the check is that the top row is about the question's
  // subject, not that it spells it the same way.
  if (SCENARIO.answer_expect) {
    check(new RegExp(SCENARIO.answer_expect, "i").test(first), "the top row answers the question asked");
  } else note("top-row relevance: skipped (no ui.answer_expect scenario)");
  check(page.url().includes("q="), "the query is in the URL, so a result list is shareable");
}
if (SHOTS) await page.screenshot({ path: `${SHOTS}/results.png` });

console.log("the two result views (against the real service)");
// What is being checked here is a PRESENTATION invariant: the passage view and the
// document view must show the same passages, one row each versus grouped under
// their documents. That holds for any answer, so nothing here asserts anything
// about which passages come back. It used to run against a stubbed encoder so it
// could run on a machine without one; now that the page ranks nothing itself, a
// stubbed vector would exercise none of the code that paints a result, so this
// section needs the service and skips itself, like the query above, when the
// query above found none.
if (!searched) {
  note("the result views need the search service; skipping");
} else {
  const views = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  views.on("pageerror", (e) => consoleErrors.push(`views: ${e.message}`));
  views.on("console", watchConsole);
  await views.goto(`${SITE}/index.html`, { waitUntil: "networkidle" });
  // A real question, because the real service answers it: a placeholder scores
  // under the floor and comes back as an empty list, which has no views to compare.
  await views.fill("#q", query("views"));
  await views.press("#q", "Enter");
  await views.waitForFunction(() => document.querySelectorAll("#results > li").length > 0 &&
                                    !document.querySelector("#result-controls").hidden, { timeout: 20000 }).catch(() => null);

  check(await views.locator("#view-toggle").isVisible(), "results bring out the view toggle");
  const passageLinksTo = async (sel, what) => {
    const href = (await views.locator(sel).first().getAttribute("href")) || "";
    check(/^view\.html\?doc=.+&chunk=\d+/.test(href), what, href.slice(0, 60));
    check(href.includes("back="), "and carries the search back with it");
  };

  const passageRows = await views.locator("#results > li").count();
  check(passageRows > 1, "the passage view lists one row a passage", `${passageRows} rows`);
  // The snippet is the biggest thing in a row and the thing the reader is judging, so
  // clicking it opens the whole chunk in place; the page number next to it is the way
  // to the page, in this view as in the grouped one.
  await passageLinksTo("#results > li a.passage-page", "and each passage links to its own page");

  // Opened, closed, and two at once, since comparing two passages is the reason to
  // open them. The text has to actually grow: a button that flips a class and shows
  // the same snippet would pass any check that only counted clicks.
  const passages = views.locator("#results > li button.hit-text");
  const opened = await passages.count();
  if (opened > 1) {
    const before = (await passages.nth(0).innerText()).length;
    const shutColour = await passages.nth(0).evaluate((el) => getComputedStyle(el).color);
    await passages.nth(0).click();
    await views.waitForTimeout(80);
    const after = (await passages.nth(0).innerText()).length;
    check(after > before, "clicking a passage shows the whole chunk", `${before} -> ${after} characters`);
    // An open passage is a paragraph of prose, so the state is a frame around it and
    // not a colour through it. Same decision as the viewer's highlighted region, and
    // it has to be checked here or the next stylesheet edit quietly reprints a
    // paragraph in purple.
    const open = await passages.nth(0).evaluate((el) => {
      const style = getComputedStyle(el);
      return { colour: style.color, ring: style.boxShadow };
    });
    check(open.colour === shutColour && open.ring !== "none",
          "and is framed rather than reprinted in purple", open.ring.slice(0, 70));
    check(await passages.nth(0).getAttribute("aria-expanded") === "true", "and says so to a screen reader");
    await passages.nth(1).click();
    await views.waitForTimeout(80);
    check(await passages.nth(0).getAttribute("aria-expanded") === "true" &&
          await passages.nth(1).getAttribute("aria-expanded") === "true",
          "a second one opens without closing the first");
    await passages.nth(0).click();
    await views.waitForTimeout(80);
    check((await passages.nth(0).innerText()).length === before, "and clicking again folds it back");
    await passages.nth(1).click();
    await views.waitForTimeout(80);
  }

  await views.locator("#view-toggle button[data-view=documents]").click();
  await views.waitForTimeout(200);
  const docRows = await views.locator("#results > li.group").count();
  check(docRows > 0 && docRows <= passageRows, "the document view lists fewer rows",
        `${docRows} documents for ${passageRows} passages`);
  const nested = await views.locator("#results > li.group .passages > li").count();
  check(nested === passageRows, "and shows every passage exactly once, under its document",
        `${nested} nested vs ${passageRows} passages`);
  await passageLinksTo("#results > li.group .passages > li > a.passage-page",
                       "so does every passage nested under its document");

  // Identity is the document id in the link, not the title: an agency may publish
  // a summary, a full text and a background report of one guideline under the
  // SAME title, and until `family`/`rendition` is curated for them, two such rows are two different documents that read alike. Grouping
  // by title here would have called that a bug in the grouping, which it is not.
  const groupIds = await views.locator("#results > li.group .passages > li > a.passage-page")
    .evaluateAll((links) => [...new Set(links.map((a) => new URL(a.href).searchParams.get("doc")))]);
  const docRowCount = await views.locator("#results > li.group").count();
  check(groupIds.length === docRowCount, "with no document listed twice",
        `${docRowCount} rows, ${groupIds.length} distinct documents`);
  const titles = await views.locator("#results > li.group .hit-title").allInnerTexts();
  if (new Set(titles).size !== titles.length) {
    note(`${titles.length - new Set(titles).size} title(s) repeated across documents: renditions needing family/rendition`);
  }
  const scores = (await views.locator("#results > li.group .hit-score").allInnerTexts())
    .map((x) => Number(x.replace(/[^0-9.]/g, "")));
  check(scores.every((x, i) => i === 0 || scores[i - 1] >= x),
        "ranked by each document's closest passage", scores.slice(0, 4).join(" >= "));

  // The per-passage rank and score labels. They are the grouped view's only account of
  // where a passage sits in the ungrouped ranking, and the only place the SECOND passage
  // of a document says anything about itself: the head's score is its best passage's.
  // Checked as a set rather than row by row, because grouping deals the ranks out to
  // whichever document each passage belongs to. What has to hold is that 1..n is dealt
  // exactly once, and that each group's own ranks read upwards.
  const labelled = await views.locator("#results > li.group").evaluateAll((groups) =>
    groups.map((g) => ({
      head: (g.querySelector(".hit-score")?.textContent || "").replace(/[^0-9.]/g, ""),
      ranks: [...g.querySelectorAll(".passages > li > .passage-rank")]
        .map((el) => el.textContent || ""),
    })));
  const labels = labelled.flatMap((g) => g.ranks);
  check(labels.length === passageRows, "every nested passage carries a rank label",
        `${labels.length} labels for ${passageRows} passages`);
  check(labels.every((s) => /^(#|n°\s)\d+ \(.*\d+\s?%\)$/.test(s.trim())),
        "reading as a rank and a score of its own", labels.slice(0, 3).join(" / "));
  const number = (s) => Number((s.match(/\d+/) || [])[0]);
  const dealt = labels.map(number).sort((a, b) => a - b);
  check(dealt.every((n, i) => n === i + 1), "numbered 1..n across the whole answer, once each",
        `${dealt.slice(0, 5).join(",")} ... ${dealt.slice(-2).join(",")}`);
  check(labelled.every((g) => g.ranks.map(number).every((n, i) => i === 0 || g.ranks.map(number)[i - 1] < n)),
        "and upwards within each document",
        labelled.slice(0, 3).map((g) => g.ranks.map(number).join("<")).join(" | "));
  check(labelled.length > 0 && number(labelled[0].ranks[0]) === 1,
        "with the best passage overall opening the first group",
        `first group opens at ${labelled[0]?.ranks[0]}`);
  // The head's score IS the first passage's, so the two labels have to agree. If they
  // ever disagree, one of them is showing a number the list was not sorted by.
  const headMismatch = labelled.filter((g) => {
    const first = (g.ranks[0].match(/(\d+[.,]?\d*)\s?%/) || [])[1];
    return !first || first.replace(",", ".") !== g.head;
  });
  check(headMismatch.length === 0, "and the document's score is its first passage's",
        headMismatch.length ? `${headMismatch.length} groups disagree: ${headMismatch[0].head} vs ${headMismatch[0].ranks[0]}`
                            : `${labelled.length} groups agree`);

  // Clicked for real: an href that the browser refuses (or that a handler swallows)
  // reads the same as a working one in the DOM.
  await views.locator("#results > li.group .passages > li > a.passage-page").first().click();
  await views.waitForURL(/view\.html/, { timeout: 20000 }).catch(() => null);
  check(/view\.html\?doc=/.test(views.url()), "and clicking one opens the viewer",
        views.url().split("/").pop().slice(0, 50));
  await views.goBack({ waitUntil: "networkidle" });
  await views.waitForTimeout(300);

  await views.reload({ waitUntil: "networkidle" });
  await views.waitForTimeout(300);
  check(await views.locator("#view-toggle button[data-view=documents]").getAttribute("aria-pressed") === "true",
        "and the chosen view survives a reload");

  // The query algebra, end to end in a browser. Nothing here is about ranking:
  // what is being checked is that a compound query reaches the service as ONE
  // request whatever the arithmetic in it (the service parses the terms and embeds
  // each side; the page never talks to the encoder), that the page survives the
  // arithmetic, and that a refusal reaches the reader as a message instead of a
  // blank page. Counted per path: a page that went back to embedding for itself
  // would show up as a /api/sem/embed request, and Caddy answers that path 404.
  let searchCalls = 0;
  let embedCalls = 0;
  views.on("request", (r) => {
    const url = r.url();
    if (url.includes("/api/sem/embed")) embedCalls++;
    else if (url.includes("/api/search") && !url.includes("/health")) searchCalls++;
  });
  await views.locator("#view-toggle button[data-view=passages]").click();
  await views.fill("#q", SCENARIO.queries?.negated || `${FALLBACK_QUERY} -(annexe)`);
  await views.press("#q", "Enter");
  // Waited for rather than timed: this is the first query of the run, so the encoder
  // behind the service is answering it cold, and on a busy machine that is seconds
  // rather than milliseconds. A fixed delay here failed the two checks below on a
  // machine that was baking at the same time, which says nothing about the page.
  await views.waitForSelector("#results > li", { timeout: 30000 }).catch(() => { /* checked below */ });
  check(searchCalls === 1, "a query with a subtraction is one request to the service", `${searchCalls} requests`);
  check(await views.locator("#results > li").count() > 0, "and still returns results");

  // Every change to the question, its filters or its scoring is one request to the
  // service and nothing else: the filters and the lexical term are applied
  // server-side now, so each costs the same single POST as the question itself,
  // and the encoder round trip behind it is the service's query-vector cache to
  // save, not the page's.
  // The passage list, which is where the numbers a reader reads come from. They have
  // to descend: the list is ordered by the cosine plus the lexical term, and it used
  // to PRINT the cosine alone, so a rescored row sat above one with a higher number
  // and the sort looked broken. Checked here rather than in the grouped view, which
  // had a check of its own and passed throughout.
  const shown = (await views.locator("#results > li .hit-score").allInnerTexts())
    .map((x) => Number(x.replace(/[^0-9.]/g, "")));
  check(shown.length > 1 && shown.every((x, i) => i === 0 || shown[i - 1] >= x),
        "the passage list prints the number it is sorted by", shown.slice(0, 5).join(" >= "));

  // The lexical half is off until a reader asks for it, so what the page has shown
  // so far is the model's own order, and the interesting direction is switching it
  // ON. That is one request, the same POST with bm25 set: the service rescores
  // the head of the list, the page only repaints what comes back.
  check(!(await views.isChecked("#rescore-check")),
        "the lexical half starts off, so the first list is the model's own answer");
  const beforeToggle = searchCalls;
  await views.check("#rescore-check");
  await views.waitForTimeout(600);
  const lexical = (await views.locator("#results > li .hit-score").allInnerTexts())
    .map((x) => Number(x.replace(/[^0-9.]/g, "")));
  check(searchCalls === beforeToggle + 1, "adding the lexical term is one request to the service",
        `${searchCalls - beforeToggle} requests`);
  check(lexical.length > 1 && lexical.every((x, i) => i === 0 || lexical[i - 1] >= x),
        "and the list it leaves behind is still in order", lexical.slice(0, 5).join(" >= "));
  check(new URL(views.url()).searchParams.get("bm25") === "1",
        "and the URL says so, so the link shows what the reader saw");
  await views.uncheck("#rescore-check");
  await views.waitForTimeout(400);
  check(new URL(views.url()).searchParams.get("bm25") === null,
        "and switching it back off leaves no parameter behind");
  // The reference demotion is the other way round: ticked from the first paint, so
  // the interesting direction is switching it OFF. Same single POST, and the URL
  // only ever carries the unusual state.
  check(await views.isChecked("#norefs-check"),
        "bibliographies are set aside from the first list, without anyone asking");
  const beforeRefs = searchCalls;
  await views.uncheck("#norefs-check");
  await views.waitForTimeout(600);
  check(searchCalls === beforeRefs + 1, "letting them back in is one request to the service",
        `${searchCalls - beforeRefs} requests`);
  check(new URL(views.url()).searchParams.get("norefs") === "0",
        "and the URL says so, since that is the state a link has to carry");
  await views.check("#norefs-check");
  await views.waitForTimeout(400);
  check(new URL(views.url()).searchParams.get("norefs") === null,
        "and setting them aside again leaves no parameter behind");

  // The recency bonus, the third ranking toggle and the only one that acts on the
  // DOCUMENT rather than the passage. Same shape as the demotion: ticked from the
  // first paint, one request to change, and the URL carries only the off state.
  // What is checked beyond the plumbing is that it MOVES something, because a
  // tie-breaker worth 0.05 of a cosine is exactly the kind of feature that can be
  // wired end to end and still change nothing a reader would see.
  check(await views.isChecked("#recent-check"),
        "recent documents are favoured from the first list, without anyone asking");
  const yearsOf = async () => (await views.locator("#results > li .hit-doc-year, #results > li .hit-meta")
    .allInnerTexts()).join(" ").match(/\b(19|20)\d\d\b/g) || [];
  const withBonus = await yearsOf();
  const beforeRecent = searchCalls;
  await views.uncheck("#recent-check");
  await views.waitForTimeout(600);
  check(searchCalls === beforeRecent + 1, "ignoring the year is one request to the service",
        `${searchCalls - beforeRecent} requests`);
  check(new URL(views.url()).searchParams.get("recent") === "0",
        "and the URL says so, since that is the state a link has to carry");
  const withoutBonus = await yearsOf();
  const mean = (xs) => (xs.length ? xs.map(Number).reduce((a, b) => a + b, 0) / xs.length : 0);
  check(withBonus.length > 0 && withoutBonus.length > 0,
        "the result list prints the years the bonus acts on", `${withBonus.length} years`);
  check(mean(withBonus) >= mean(withoutBonus),
        "and the biased list is not older than the unbiased one",
        `${mean(withBonus).toFixed(1)} vs ${mean(withoutBonus).toFixed(1)}`);
  await views.check("#recent-check");
  await views.waitForTimeout(400);
  check(new URL(views.url()).searchParams.get("recent") === null,
        "and favouring them again leaves no parameter behind");

  // The negative spelling was what links carried while the default was the other way
  // round, and it still has to mean what it said then.
  await views.goto(`${SITE}/index.html?q=${encodeURIComponent(query("bm25"))}&bm25=0`);
  await views.waitForSelector("#results > li", { timeout: 20000 });
  check(!(await views.isChecked("#rescore-check")),
        "a link written with bm25=0 still ranks on meaning alone");

  const beforeReask = searchCalls;
  await views.press("#q", "Enter");
  await views.waitForTimeout(600);
  check(searchCalls === beforeReask + 1, "re-asking the same question is one request",
        `${searchCalls - beforeReask} requests`);
  // Opened by attribute rather than by clicking the summary: the panel's own open
  // state is asserted further down, and a click here would depend on it.
  await views.locator("#advanced").evaluate((el) => { el.open = true; });
  await openFacet(views, "language");
  await views.locator(`#facet-language input[value="${LANGUAGE}"]`).check();
  await views.waitForTimeout(700);
  check(searchCalls === beforeReask + 2, "and so is narrowing a filter",
        `${searchCalls - beforeReask - 1} requests`);
  check(embedCalls === 0, "and the page never asked the encoder for anything itself",
        `${embedCalls} /api/sem/embed requests`);
  await views.locator("#reset-btn").click();
  await views.waitForTimeout(400);

  // Same vector on both sides, so the negative IS the question: the page must say
  // so rather than rank the rounding error left behind.
  await views.fill("#q", "quelque chose -(quelque chose)");
  await views.press("#q", "Enter");
  // Waited for rather than slept through. A fixed 600 ms was enough while this ran
  // against a 30,746-chunk index and a warm service, and it stopped being enough on
  // 2026-09-26: the FIRST search of a session pays for the encoder connection and a
  // cold text cache, measured at 922 ms against the 1 ms a warm one takes, so the
  // refusal had not arrived when the status was read. That failed this check AND left
  // the service's 400 in the console log for the final check to count as an error,
  // which is a gate that fails on its own timing rather than on the site.
  await views.waitForFunction(
    () => /annule|cancels/i.test(document.querySelector("#status")?.innerText || ""),
    { timeout: 20000 }).catch(() => null);
  const refusal = (await views.locator("#status").innerText()).trim();
  check(/annule|cancels/i.test(refusal), "a query that cancels itself says so", refusal.slice(0, 60));
  // The refusal is the service's 400, which the browser logs as a failed resource
  // whatever the page does with it. That one line is the expected shape of this
  // check passing, not an error; anything else logged stays counted.
  consoleErrors.splice(0, consoleErrors.length, ...consoleErrors.filter((e) => !/400 \(Bad Request\)/.test(e)));

  // Two controls now, one each: "?" beside the box for how to ask, and "Recherche
  // avancée" below it for the filters. The help moved out of the panel because a
  // reader opening it to set a filter was made to scroll past four paragraphs of
  // prose, so the check that matters is that the two are no longer nested.
  check(await views.locator("#advanced #syntax").count() === 0
        && await views.locator("#syntax-help #syntax").count() === 1,
        "the syntax help sits beside the box, not inside the filter panel");
  await views.locator("#advanced").evaluate((el) => { el.open = false; });
  check(await views.locator("#facets").isVisible() === false,
        "closed, the panel hides the filters");
  await views.locator("#advanced > summary").click();
  check(await views.locator("#facets .facet-btn").first().isVisible(),
        "and one click shows them");
  // The help itself: shut, one click open, and a label a reader can act on. The button
  // carries one character, so what it says is in the tooltip and the aria-label, and
  // an empty one would leave a bare "?" on the row with nothing to explain it.
  check(await views.locator("#syntax code").first().isVisible() === false,
        "the syntax help is folded away until it is asked for");
  const helpLabel = await views.locator("#syntax-summary").getAttribute("aria-label");
  check(/aide|help/i.test(helpLabel || ""), "and says what it is, since the button is one character",
        (helpLabel || "").slice(0, 50));
  check((await views.locator("#syntax-summary").getAttribute("title")) === helpLabel,
        "to a pointer as well as to a screen reader");
  await views.locator("#syntax-summary").click();
  await views.waitForTimeout(150);
  check(await views.locator("#syntax code").first().isVisible(),
        "and one click on it shows the query algebra");
  // The cross-language sentence, in whichever language the page is in. It is the
  // only place the site says that a French question reaches an English guideline
  // AND that it does so less well, which is what stops an empty result from reading
  // as "the corpus has nothing on this".
  const languageNote = (await views.locator('#syntax [data-i18n="syntax_language"]').innerText()).trim();
  check(/langue|language/i.test(languageNote)
        && (!SCENARIO.language_note || new RegExp(SCENARIO.language_note).test(languageNote)),
        "the panel says search crosses languages and works best in the document's own",
        languageNote.slice(0, 70));
  await views.keyboard.press("Escape");
  await views.waitForTimeout(150);
  check(await views.locator("#syntax code").first().isVisible() === false,
        "and Escape folds it away again");
  // Opened, then a search: the panel has said what it has to say and the answer needs
  // the room. This is the one piece of its state the page moves by itself.
  await views.fill("#q", query("advanced"));
  await views.press("#q", "Enter");
  await views.waitForTimeout(600);
  check(await views.locator("#advanced").evaluate((el) => el.open) === false,
        "and a search closes it again");
  // Unless a filter is set. Then the knobs stay where the reader left them, and the
  // summary carries the state for the times the panel is shut: a narrowed corpus has
  // to be visible from the page, not only from the URL.
  await views.locator("#advanced > summary").click();
  await openFacet(views, "language");
  await views.locator(`#facet-language input[value="${LANGUAGE}"]`).check();
  await views.waitForTimeout(700);
  const narrowedSummary = await views.locator("#advanced > summary").innerText();
  check(/\(.+\)/.test(narrowedSummary), "the summary says when a filter is narrowing the corpus",
        narrowedSummary);
  await views.fill("#q", query("filtered"));
  await views.press("#q", "Enter");
  await views.waitForTimeout(800);
  check(await views.locator("#advanced").evaluate((el) => el.open),
        "and a search leaves the panel open while a filter is set");
  await views.locator("#reset-btn").click();
  await views.waitForTimeout(400);

  console.log("url state (a search is a place)");
  // ?query= is the alias anyone hand-writing a URL reaches for; ?q= is what the page
  // writes. A filter arrives with it, so the listing under the answer is narrowed
  // before anything paints.
  await views.goto(`${SITE}/index.html?query=${encodeURIComponent(query("filtered"))}&issuer=${encodeURIComponent(ISSUER)}`,
                   { waitUntil: "networkidle" });
  await views.waitForTimeout(900);
  check(await views.locator("#q").inputValue() === query("filtered"),
        "?query= in the URL runs the search on arrival");
  check(await views.locator("#results > li").count() > 0, "and the results are there");
  const normalised = new URL(views.url()).searchParams;
  check(normalised.get("q") && !normalised.get("query"),
        "the alias is normalised to ?q=", views.url().split("?")[1] || "");
  await views.locator("#advanced").evaluate((el) => { el.open = true; });
  check(await views.locator(`#facet-issuer input[value="${ISSUER}"]`).isChecked(),
        "a filter in the URL is applied AND shown in its control");
  await openFacet(views, "country");
  await views.locator(`#facet-country input[value="${COUNTRY}"]`).check();
  await views.waitForTimeout(700);
  check(new URL(views.url()).searchParams.get("country") === COUNTRY,
        "changing a filter puts it in the URL", decodeURIComponent(new URL(views.url()).search));
  await views.locator("#facet-year").fill(String(yearMax - 3));
  await views.waitForTimeout(700);
  check(/^\d{4}-\d{4}$/.test(new URL(views.url()).searchParams.get("year") || ""),
        "and the year range travels as a range", new URL(views.url()).searchParams.get("year"));

  // Into a hit and back out again. This is what "retour aux résultats" has to do:
  // the question and every filter, not a bare corpus listing.
  const backTo = decodeURIComponent(new URL(views.url()).search);
  await views.locator("#results .hit-title").first().click();
  await views.waitForLoadState("networkidle");
  await views.waitForTimeout(500);
  const backHref = await views.locator("#back").getAttribute("href");
  check(decodeURIComponent(new URL(backHref, views.url()).search) === backTo,
        "the viewer's way back carries the whole search", decodeURIComponent(new URL(backHref, views.url()).search));
  await views.locator("#back").click();
  await views.waitForLoadState("networkidle");
  await views.waitForTimeout(900);
  check(await views.locator("#q").inputValue() === query("filtered"),
        "and returning puts the question back in the box");
  check(await views.locator("#results > li").count() > 0, "with its results under it");
  await views.locator("#advanced").evaluate((el) => { el.open = true; });
  check(await views.locator(`#facet-country input[value="${COUNTRY}"]`).isChecked(),
        "and its filters back in their controls");

  /* The offer to widen, under the last result. A narrowed level keeps part of the
     corpus out, and the end of a list that stopped short is where that costs the
     reader something, so this is checked in a browser: that it
     appears at a narrowed level, that it names the NEXT rung rather than "everything"
     (a reader has to know what they are letting in), that clicking it moves the
     select at the head of the panel (two controls, one level: they must not disagree),
     and that it takes itself off the page once there is nothing left to widen. */
  await views.goto(`${SITE}/index.html?q=${encodeURIComponent(query("widen"))}&guideline=${NARROW}`,
                   { waitUntil: "load" });
  await views.waitForFunction(() => document.querySelectorAll("#results > li").length > 0,
                              { timeout: 20000 }).catch(() => null);
  await views.waitForTimeout(400);
  const widenLabel = (await views.locator("#widen-btn").innerText().catch(() => "")).trim();
  check(/synthès|synthes/i.test(widenLabel),
        "a narrowed list offers to widen, and names the tier it would add", widenLabel);
  check(await views.locator("#widen").evaluate(
          (el) => el.compareDocumentPosition(document.querySelector("#results")) === 2),
        "under the last result, not above the list");
  await views.click("#widen-btn");
  await views.waitForTimeout(1200);
  check(await views.inputValue(LEVEL_ID) === NEXT,
        "clicking it moves the level one notch, and the select says so",
        await views.inputValue(LEVEL_ID));
  check(new URL(views.url()).searchParams.get("level") === NEXT,
        "and the URL follows it");
  // The panel shuts itself on a search, so open it before reaching for the select.
  await views.locator("#advanced").evaluate((el) => { el.open = true; });
  await views.selectOption(LEVEL_ID, "all");
  await views.waitForTimeout(1200);
  check(await views.locator("#widen").isVisible() === false,
        "at the widest level there is nothing left to offer");
  // `all` is also the level the site ships with, which matters because this page's
  // localStorage outlives the section: a later check that assumes the default reads
  // the level this one left behind.
  await views.close();
}

console.log("the publisher link");
// source_url came from a language-model lookup, so the UI has to say so where the
// reader meets it, not only in the repository's own documentation. Three things are
// checked here and nowhere else: that every link on the page is one of the URLs the
// index actually carries (a link built by string-joining would pass every other
// check and send a reader somewhere arbitrary), that the caveat rides in the visible
// label rather than only in the tooltip a phone cannot show, and that the documents
// whose publisher withdrew them show no link at all rather than an empty one.
{
  // The document page, and a fresh load of it: the sections above leave the listing
  // filtered by whatever they dragged the facet controls to, and a filtered-to-nothing
  // list has no rows to look at. The search page cannot stand in for it, because a
  // result row only exists once the search service has answered.
  await page.goto(`${SITE}/browse.html`, { waitUntil: "networkidle" });
  await page.waitForFunction(() => document.querySelectorAll("#results > li").length > 0,
                             { timeout: 15000 }).catch(() => null);
  const seen = await page.evaluate(async () => {
    const meta = await fetch("index/meta.json").then((r) => r.json());
    const urls = new Set(meta.documents.map((d) => d.source_url).filter(Boolean));
    const rows = [...document.querySelectorAll("#results > li")];
    const links = [...document.querySelectorAll("#results .src-ai")];
    // The row identifies its document by the file its own links point at.
    const fileOf = (row) => decodeURIComponent((row.querySelector("a[href^='pdf/']")?.getAttribute("href") || "").slice(4));
    const blank = new Set(meta.documents.filter((d) => !d.source_url).map((d) => d.file));
    return {
      withUrl: urls.size,
      docs: meta.documents.length,
      rows: rows.length,
      shown: links.length,
      unknown: links.map((a) => a.href).filter((h) => !urls.has(h)).slice(0, 3),
      blankRowsWithLink: rows.filter((r) => blank.has(fileOf(r)) && r.querySelector(".src-ai")).map(fileOf),
      blankRowsListed: rows.filter((r) => blank.has(fileOf(r))).length,
      first: links[0] ? { text: links[0].textContent, title: links[0].title,
                          target: links[0].target, rel: links[0].rel } : null,
    };
  });
  check(seen.withUrl > 100, "the index carries the publisher URLs", `${seen.withUrl} of ${seen.docs}`);
  check(seen.shown > 0, "the corpus listing shows them", `${seen.shown} links on ${seen.rows} rows`);
  check(seen.unknown.length === 0, "every link is a URL the index actually carries",
        seen.unknown.join(" ") || "all matched");
  check(/\((IA|AI)\)/.test(seen.first?.text || ""),
        "the label itself says the link is machine-found", seen.first?.text);
  check(/modèle de langage|language model/.test(seen.first?.title || ""),
        "and the tooltip says what that means for trusting it",
        (seen.first?.title || "").slice(0, 80));
  check(seen.first?.target === "_blank" && /noopener/.test(seen.first?.rel || ""),
        "it opens elsewhere without handing over this window", `${seen.first?.target} ${seen.first?.rel}`);
  check(seen.blankRowsWithLink.length === 0,
        "a document with no known source shows no link",
        `${seen.blankRowsListed} such rows listed`);
}

// The other link on that line: the DOI or ISBN the document prints about itself,
// read out of the PDF by scripts/manifest.py rather than looked up. Checked in a
// browser because everything about it is DOM wiring, and because the failure it
// guards against is silent: NO_IDENTIFIER ("-") is what 467 of the 535 rows carry,
// and rendered as a link it would send a reader to doi.org/- from most of the
// corpus. The same helper draws it on a result row and in the viewer.
{
  const seen = await page.evaluate(async () => {
    const meta = await fetch("index/meta.json").then((r) => r.json());
    const real = (v) => v && v !== "-";
    const withDoi = meta.documents.filter((d) => real(d.doi));
    const links = [...document.querySelectorAll("#results .doc-id")];
    const rows = [...document.querySelectorAll("#results > li")];
    const fileOf = (row) => decodeURIComponent((row.querySelector("a[href^='pdf/']")?.getAttribute("href") || "").slice(4));
    const none = new Set(meta.documents.filter((d) => !real(d.doi) && !real(d.isbn)).map((d) => d.file));
    const urls = new Set([
      ...withDoi.map((d) => `https://doi.org/${encodeURI(d.doi)}`),
      ...meta.documents.filter((d) => real(d.isbn))
        .map((d) => `https://search.worldcat.org/search?q=bn%3A${d.isbn.replace(/[-\s]/g, "")}`),
    ]);
    return {
      carried: withDoi.length + meta.documents.filter((d) => real(d.isbn)).length,
      shown: links.length,
      unknown: links.map((a) => a.href).filter((h) => !urls.has(h)).slice(0, 3),
      sentinelRowsWithLink: rows.filter((r) => none.has(fileOf(r)) && r.querySelector(".doc-id")).map(fileOf),
      first: links[0] ? { text: links[0].textContent, href: links[0].href,
                          target: links[0].target, rel: links[0].rel } : null,
    };
  });
  check(seen.carried > 100, "the index carries the documents' own identifiers",
        `${seen.carried} DOIs and ISBNs`);
  check(seen.shown > 0, "the corpus listing turns them into links", `${seen.shown} shown`);
  check(seen.unknown.length === 0, "each one points at the identifier the index holds",
        seen.unknown.join(" ") || "all matched");
  check(seen.sentinelRowsWithLink.length === 0,
        "a document that prints neither shows no link",
        seen.sentinelRowsWithLink.slice(0, 3).join(" ") || "none");
  check(/^(DOI|ISBN)$/.test(seen.first?.text || ""),
        "the label is the identifier's own name", seen.first?.text);
  check(seen.first?.target === "_blank" && /noopener/.test(seen.first?.rel || ""),
        "it opens elsewhere without handing over this window",
        `${seen.first?.target} ${seen.first?.rel}`);
}

console.log("viewer geometry (the check that matters)");
// Pick the target from the index rather than hard-coding one: any chunk whose boxes
// live on a single page and which carries enough text to be identified.
const target = await page.evaluate(async () => {
  const meta = await fetch("index/meta.json").then((r) => r.json());
  for (const doc of meta.documents.slice(0, 40)) {
    // An open document: this section checks the PDF menu's two answers, which a
    // restricted document is not supposed to have. The restricted ones get their own
    // section below.
    if (doc.access === "restricted") continue;
    const payload = await fetch(`index/doc/${doc.id}.json`).then((r) => r.json());
    const i = payload.chunks.findIndex((c) => Object.keys(c.boxes || {}).length === 1 && c.text.length > 300
      // A figure's text is a description, not the page's words: nothing to read under it.
      && !c.figure);
    if (i >= 0) {
      const c = payload.chunks[i];
      return { doc: doc.id, chunk: doc.chunk_offset + i, file: doc.file,
               issuer: doc.issuer || "", title: doc.title,
               page: Number(Object.keys(c.boxes)[0]), text: c.text, pages: doc.pages };
    }
  }
  return null;
});
check(!!target, "found a chunk to open", target ? `${target.file} page ${target.page}` : "none");

if (target) {
  const viewer = await browser.newPage({ viewport: { width: 1100, height: 1200 } });
  const viewerErrors = [];
  viewer.on("pageerror", (e) => viewerErrors.push(e.message));
  viewer.on("console", watchConsole);
  viewer.on("response", (r) => { if (r.status() >= 400 && !r.url().includes("favicon")) badResponses.push(`viewer ${r.status()} ${r.url()}`); });
  await viewer.goto(`${SITE}/view.html?doc=${target.doc}&chunk=${target.chunk}`, { waitUntil: "load" });
  await viewer.waitForFunction(() => document.querySelectorAll("#layer > .best").length > 0, { timeout: 30000 }).catch(() => null);

  const canvas = await viewer.locator("#canvas").boundingBox();
  check(canvas && canvas.width > 200, "the PDF page rendered",
        canvas ? `${Math.round(canvas.width)}x${Math.round(canvas.height)}` : "no canvas");
  const drawn = await viewer.locator("#layer > .best").count();
  check(drawn > 0, "the matching passage is highlighted", `${drawn} boxes`);

  // The viewer builds its own copy of the publisher link, on the metadata line rather
  // than in the PDF menu, so it needs its own check: same URL as the index carries,
  // and nothing at all for a document that has none.
  const publisher = await viewer.evaluate(async () => {
    const meta = await fetch("index/meta.json").then((r) => r.json());
    const doc = meta.documents[Number(new URL(location.href).searchParams.get("doc"))];
    const link = document.querySelector("#doc-meta .src-ai");
    return { want: doc.source_url || "", href: link ? link.href : "", title: link ? link.title : "" };
  });
  check(publisher.href === publisher.want,
        publisher.want ? "the viewer shows the same publisher link"
                       : "the viewer shows no publisher link for a document without one",
        publisher.want ? publisher.href.slice(0, 60) : "none, as expected");
  if (publisher.want) {
    check(/modèle de langage|language model/.test(publisher.title),
          "with the same warning on it", publisher.title.slice(0, 60));
  }

  // One control for the whole document, and its label has to say out loud that it
  // hands over the whole thing and how long that thing is: the reader is looking at a
  // single page, so "PDF" on its own reads as "this page".
  const pdfLabel = (await viewer.locator("#pdf-summary").innerText()).trim();
  check(/PDF complet|full PDF/i.test(pdfLabel), "one control offers the whole PDF", pdfLabel);
  if (Number(target.pages) >= 1) {
    check(pdfLabel.includes(String(Number(target.pages))), "and says how many pages that is", pdfLabel);
  }
  // Folded away until asked: two links side by side made the reader choose between
  // opening and saving before knowing that was the choice on offer.
  check(!(await viewer.locator("#dl").isVisible()), "with its two answers folded away");
  await viewer.click("#pdf-summary");
  await viewer.waitForTimeout(100);
  check(await viewer.locator("#pdf").isVisible() && await viewer.locator("#dl").isVisible(),
        "and both answers once it is opened");
  check(await viewer.locator("#dl").getAttribute("download") === target.file,
        "one of which downloads the document rather than navigating to it");
  await viewer.keyboard.press("Escape");
  await viewer.waitForTimeout(100);
  check(!(await viewer.locator("#dl").isVisible()), "and Escape closes it again");

  /* A document the site may show but not hand over.
   *
   * Checked in a browser because the rule lives in three places that have to agree:
   * the control that would offer the file, the arrows that would walk out of the
   * window, and the keyboard that goes through those same arrows. A reader who can
   * still press ArrowRight three times has the book. */
  const restricted = await page.evaluate(async () => {
    const meta = await fetch("index/meta.json").then((r) => r.json());
    for (const doc of meta.documents) {
      if (doc.access !== "restricted") continue;
      const payload = await fetch(`index/doc/${doc.id}.json`).then((r) => r.json());
      // A chunk in the middle of the document, so the window has room on both sides,
      // and one that lives on a SINGLE page: the viewer opens on the page carrying
      // most of a chunk, the rule the result list follows, so on a chunk that
      // straddles two pages "the passage's page" would have to be that rule restated
      // here. It used to be `pages[0]`, and the first chunk past page 20 of the first
      // restricted document is now one spanning pages 21 to 23 with 45 of its 71
      // boxes on page 22, so the gate failed on the viewer being right. Two pages
      // clear of the end as well, or the forward walk is stopped by the document
      // rather than by the window, which is the other rule entirely.
      const last = Number(doc.pages) || 0;
      const i = payload.chunks.findIndex((c) => (c.pages || []).length === 1
                                                && c.pages[0] > 20
                                                && (!last || c.pages[0] <= last - 2));
      if (i >= 0) return { doc: doc.id, chunk: doc.chunk_offset + i, file: doc.file,
                           page: payload.chunks[i].pages[0] };
    }
    return null;
  });
  if (!restricted) {
    console.log("  - no restricted document in the index, section skipped");
  } else {
    const locked = await browser.newPage({ viewport: { width: 1100, height: 1200 } });
    locked.on("console", watchConsole);
    await locked.goto(`${SITE}/view.html?doc=${restricted.doc}&chunk=${restricted.chunk}`,
                      { waitUntil: "load" });
    await locked.waitForFunction(() => document.querySelector("#pageno")?.textContent?.includes("/"),
                                 { timeout: 30000 }).catch(() => null);
    check((await locked.locator("#dl").count()) === 0,
          "a restricted document offers no download link at all", restricted.file);
    check((await locked.locator("#pdf").count()) === 0,
          "and no way to open the whole file either");
    const note = (await locked.locator("#pdf-summary").innerText()).trim();
    check(/non téléchargeable|not downloadable/i.test(note),
          "the control says so rather than going silently missing", note);
    const why = await locked.locator("#pdf-summary").getAttribute("title");
    check(/droits|copyright/i.test(why || ""), "and gives the reason on hover", (why || "").slice(0, 60));

    const pageno = async () => Number((await locked.locator("#pageno").innerText()).split("/")[0].trim());
    /* The page number is written at the END of render, and a press that arrives while
       a render is running cancels that render, so the label stays on the page still
       drawn. A fixed delay per press therefore reads a stale number and calls a walk
       that worked a walk that stopped short. So a press that is allowed waits for its
       own page to appear, and only the presses that must be refused are timed: those
       render nothing, so the label cannot be late for them. */
    const step = async (key, want) => {
      await locked.keyboard.press(key);
      await locked.waitForFunction(
        (n) => document.querySelector("#pageno")?.textContent.trim().startsWith(`${n} /`),
        want, { timeout: 20000 }).catch(() => { /* asserted below, not here */ });
      return pageno();
    };
    const refuse = async (key, times) => {
      for (let i = 0; i < times; i += 1) { await locked.keyboard.press(key); await locked.waitForTimeout(400); }
      return pageno();
    };
    const opened = await pageno();
    check(opened === restricted.page, "it opens on the passage's page", String(opened));
    // Three presses, one page: the arrows stop at the edge of the window and the
    // keyboard cannot get past them.
    await step("ArrowRight", opened + 1);
    const forward = await refuse("ArrowRight", 2);
    check(forward === opened + 1, "and reading forward stops one page later", `${opened} -> ${forward}`);
    await step("ArrowLeft", opened);
    await step("ArrowLeft", opened - 1);
    const back = await refuse("ArrowLeft", 2);
    check(back === opened - 1, "and backward one page earlier", `${forward} -> ${back}`);
    // Standing on the first page of the window, so the backward arrow is the one that
    // must be dead, and it must say why where the pointer already is.
    // Greyed rather than merely inert, and still clickable: `disabled` would be the
    // honest attribute for a dead button, but a disabled button swallows the click,
    // and the click is how a touch screen asks the question a tooltip answers.
    const grey = await locked.evaluate(() => {
      const prev = document.querySelector("#prev"), next = document.querySelector("#next");
      return { aria: prev.getAttribute("aria-disabled"), clickable: !prev.disabled,
               dimmed: getComputedStyle(prev).color !== getComputedStyle(next).color };
    });
    check(grey.aria === "true" && grey.dimmed, "with the arrow at the edge greyed out");
    check(grey.clickable, "and still able to answer a tap");
    // The note is the only answer a finger gets: a touch screen has no hover, so
    // without it the reader is left with an arrow that refuses and says nothing.
    // `force`, because playwright's own actionability check treats aria-disabled as
    // unclickable in some versions and would refuse the very click being tested. It
    // is still a real mouse click at the button's centre, not a dispatched event.
    await locked.click("#prev", { force: true });
    await locked.waitForTimeout(100);
    const tapped = (await locked.locator(".tap-note").innerText().catch(() => "")).trim();
    check(/droits|legal|copyright/i.test(tapped), "and a tap on it says why",
          tapped.slice(0, 60));
    const after = await pageno();
    check(after === back, "without turning the page it refused to turn", `${back} -> ${after}`);
    const tip = await locked.locator("#prev").getAttribute("title");
    check(/droits|legal/i.test(tip || ""), "and that arrow carrying the reason", (tip || "").slice(0, 60));
    /* The PDF control is dead in the same way and has to answer in the same way: the
       reader who wants the file is the one who most needs to be told why there is no
       file. A pointer gets `title`, a finger gets nothing, so the tap is what is
       checked here. */
    await locked.click("#pdf-summary", { force: true });
    await locked.waitForTimeout(100);
    const pdfTap = (await locked.locator(".tap-note").innerText().catch(() => "")).trim();
    check(/droits|legal|copyright/i.test(pdfTap), "and a tap on the PDF control says why too",
          pdfTap.slice(0, 60));
    const meta_ = await locked.locator("#doc-meta").innerText();
    check(/page|droits|legal/i.test(meta_), "and the metadata line says why", meta_.slice(0, 80));
    // And no passage text either. stage.py publishes these documents' geometry without
    // their `text`, and check_served.py refuses a word of one under the web root, so
    // the disclosure has nothing to open onto and must be absent rather than empty.
    check(!(await locked.locator("#chunk-text").isVisible()),
          "and the passage's text is not offered, because it is not published");
    await locked.close();
  }

  // The highlight toggle. Checked in a browser rather than by reading because the
  // interesting half is that the choice survives a reload: a reader who turned the
  // boxes off is reading the page as published, and having them return on the next
  // hit would be the annoying half of a toggle.
  check(await viewer.locator("#layer").isVisible(), "highlights are on by default");
  const onLabel = (await viewer.locator("#hl-btn").innerText()).trim();
  await viewer.click("#hl-btn");
  await viewer.waitForTimeout(100);
  check(!(await viewer.locator("#layer").isVisible()), "the toggle hides them");
  check((await viewer.locator("#hl-btn").innerText()).trim() !== onLabel,
        "and the button offers to put them back",
        (await viewer.locator("#hl-btn").innerText()).trim());
  await viewer.reload({ waitUntil: "load" });
  await viewer.waitForTimeout(600);
  check(!(await viewer.locator("#layer").isVisible()), "and the choice survives a reload");
  await viewer.keyboard.press("h");
  await viewer.waitForTimeout(100);
  check(await viewer.locator("#layer").isVisible(), "and \"h\" flips it back");
  await viewer.waitForFunction(() => document.querySelectorAll("#layer > .best").length > 0, { timeout: 30000 }).catch(() => null);

  const geometry = await viewer.evaluate(async ({ file, pageNumber }) => {
    const pdfjs = await import("./vendor/pdfjs/pdf.mjs");
    const doc = await pdfjs.getDocument(`pdf/${encodeURIComponent(file)}`).promise;
    const pdfPage = await doc.getPage(pageNumber);
    // scale 1: the same space the percentages are fractions of.
    const viewport = pdfPage.getViewport({ scale: 1 });
    const content = await pdfPage.getTextContent();
    const items = content.items.filter((i) => i.str.trim()).map((i) => {
      const [x, baseline] = pdfjs.Util.applyTransform([i.transform[4], i.transform[5]], viewport.transform);
      return { str: i.str, x0: x, x1: x + i.width, top: baseline - i.height, bottom: baseline };
    });
    const boxes = [...document.querySelectorAll("#layer > .best")].map((el) => ({
      x0: (parseFloat(el.style.left) / 100) * viewport.width,
      y0: (parseFloat(el.style.top) / 100) * viewport.height,
      x1: ((parseFloat(el.style.left) + parseFloat(el.style.width)) / 100) * viewport.width,
      y1: ((parseFloat(el.style.top) + parseFloat(el.style.height)) / 100) * viewport.height,
    }));
    // The words the reader sees under the highlights, in reading order.
    const covered = [];
    for (const box of boxes) {
      for (const item of items) {
        const middle = (item.top + item.bottom) / 2;
        const overlap = Math.min(box.x1, item.x1) - Math.max(box.x0, item.x0);
        if (middle > box.y0 && middle < box.y1 && overlap > (item.x1 - item.x0) * 0.5) covered.push(item.str);
      }
    }
    return { covered: covered.join(""), boxes: boxes.length, page: { w: viewport.width, h: viewport.height } };
  }, { file: target.file, pageNumber: target.page });

  // This guards the coordinate conversion: a highlight mirrored about the middle of
  // the page still sits over text, so nothing but reading the words underneath can
  // tell the difference. It guarded a merge too, while the viewer drew one frame per
  // block of lines; the fill is per line now and there is nothing left to merge.
  //
  // Letters only, on both sides. pdf.js and PyMuPDF agree on the words of a page
  // and disagree on everything around them: ligatures are split into their own
  // items (which PyMuPDF renders as a space), a non-breaking hyphen comes back as
  // U+2011 from one and U+2010 from the other, and a journal byline's superscript
  // affiliation digits are emitted between the names by one and after them by the
  // other. None of that is what this check is about: a flipped or offset box
  // covers DIFFERENT WORDS, which survives any amount of punctuation folding.
  const flatten = (s) => s.normalize("NFKD").toLowerCase().replace(/[^a-z]+/g, "");
  const covered = flatten(geometry.covered);
  const passage = flatten(target.text);
  const inside = covered.length > 40 && passage.includes(covered.slice(0, Math.min(covered.length, 400)));
  note(`under the highlights: "${geometry.covered.slice(0, 90).replace(/\s+/g, " ")}..."`);
  check(covered.length > 40, "the highlights cover some text at all", `${covered.length} characters`);
  // When it fails, say WHERE the two diverge. A flipped or offset box diverges at
  // character 0; a punctuation difference between pdf.js and PyMuPDF diverges deep
  // into the string, and the two are worth telling apart without a debugger.
  let diverged = "";
  if (!inside) {
    const probe = covered.slice(0, Math.min(covered.length, 400));
    let common = 0;
    while (common < probe.length && passage.slice(0, common + 1) === probe.slice(0, common + 1)) common++;
    diverged = `covered text is not part of the chunk (a flipped or offset box looks exactly like this). `
      + `They agree for ${common} characters, then the page has ${JSON.stringify(probe.slice(common, common + 24))} `
      + `and the chunk has ${JSON.stringify(passage.slice(common, common + 24))}`;
  }
  check(inside, "the words under the highlights are the passage's own words", diverged);

  const layer = await viewer.evaluate(() => {
    const l = document.querySelector("#layer").getBoundingClientRect();
    const c = document.querySelector("#canvas").getBoundingClientRect();
    return { dx: Math.abs(l.x - c.x), dy: Math.abs(l.y - c.y), dw: Math.abs(l.width - c.width), dh: Math.abs(l.height - c.height) };
  });
  check(layer.dx < 2 && layer.dy < 2 && layer.dw < 2 && layer.dh < 2,
        "the highlight layer is exactly the page box", JSON.stringify(layer));

  // ONE chunk is drawn. Neighbouring chunks used to be drawn in a second colour, and
  // on a dense page that is most of the page: the passage the search found became a
  // purple island in a field of orange.
  check(await viewer.locator("#layer > .other").count() === 0,
        "no chunk but the matching one is coloured in");

  // The page is a picture; without a text layer the reader cannot copy a dose, a
  // criterion or a reference out of a document whose whole value is its exact wording.
  const spans = await viewer.locator("#textlayer span").count();
  check(spans > 10, "the page text is selectable", `${spans} text spans`);

  // Order on the page. The pager belongs under the page: above it, it asked for a
  // decision before the page had been read.
  const order = await viewer.evaluate(() => {
    const box = (s) => document.querySelector(s)?.getBoundingClientRect() || null;
    const frame = box(".page-frame"), pager = box(".pager");
    const btn = box("#hl-btn");
    return { pagerBelow: pager && frame ? pager.top >= frame.bottom - 1 : false,
             // Top right, not just "somewhere over the page": a long page puts its
             // bottom corner a scroll away from the reader who wants the colours off.
             overlayTopRight: btn && frame
               ? btn.top >= frame.top - 1 && btn.top <= frame.top + 60
                 && btn.right <= frame.right + 1 && btn.right >= frame.right - 200
               : false };
  });
  check(order.pagerBelow, "the pager sits under the page");
  check(order.overlayTopRight, "while the highlight toggle sits in the page's top right corner");
  if (SHOTS) await viewer.screenshot({ path: `${SHOTS}/viewer.png` });

  // Percentages, so a narrower window must not move a highlight off its words.
  await viewer.setViewportSize({ width: 780, height: 1200 });
  await viewer.waitForTimeout(2500);
  const after = await viewer.evaluate(() => {
    const l = document.querySelector("#layer").getBoundingClientRect();
    const c = document.querySelector("#canvas").getBoundingClientRect();
    const b = document.querySelector("#layer > .best")?.getBoundingClientRect();
    return { same: Math.abs(l.width - c.width) < 2 && Math.abs(l.height - c.height) < 2,
             insideCanvas: b ? (b.x >= c.x - 2 && b.x + b.width <= c.x + c.width + 2) : false, w: Math.round(c.width) };
  });
  check(after.same && after.insideCanvas, "highlights still fit the page after a resize", `canvas ${after.w}px`);

  // Click to fill the screen with the page, click again to come back: one state, not
  // a ladder of zoom levels. It has to survive the text layer sitting over the whole
  // page, so this is a real mouse click rather than element.click(): the handler
  // ignores anything that travelled more than a few pixels between pointerdown and
  // click, so that dragging out a selection does not also expand, and a synthetic
  // click has no pointerdown at all. Widened first so the expanded page is visibly
  // bigger than the inline one rather than bigger by a page margin.
  await viewer.setViewportSize({ width: 1400, height: 900 });
  await viewer.waitForTimeout(2500);
  const widthOfPage = () => viewer.evaluate(() => document.querySelector("#canvas").getBoundingClientRect().width);
  const inline = await widthOfPage();
  await viewer.locator("#stage").click({ position: { x: 40, y: 40 } });
  await viewer.waitForTimeout(2000);
  const full = await widthOfPage();
  check(await viewer.locator("#page-frame.expanded").count() === 1,
        "clicking the page fills the screen with it");
  check(full > inline + 10, "and the page is re-rendered to the room that gives it",
        `${Math.round(inline)} -> ${Math.round(full)}`);
  // The one control that sits ON the page goes away while the page fills the screen.
  // A reader expands to read the page, and that button covers the words it is about.
  check(await viewer.locator("#hl-btn").isVisible() === false,
        "and the highlight toggle, which sits on the page, is gone while it does");
  // A trackpad pinch (ctrl+wheel) on the expanded page is the page's zoom, not the
  // browser's: the page must come back RE-RENDERED larger, bitmap and all. The
  // browser's own pinch only stretched the bitmap, which is what readers called
  // illegible. Zooming out far enough shows the whole page, centred in the screen.
  const bitmap = () => viewer.evaluate(() => document.querySelector("#canvas").width);
  const before = await bitmap();
  await viewer.mouse.move(700, 450);
  await viewer.keyboard.down("Control");
  for (let i = 0; i < 3; i++) await viewer.mouse.wheel(0, -100);
  await viewer.keyboard.up("Control");
  await viewer.waitForTimeout(2000);
  const zoomedIn = await bitmap();
  check(zoomedIn > before * 1.5 && await viewer.locator("#page-frame.expanded").count() === 1,
        "a pinch on the expanded page re-renders it larger rather than stretching it",
        `${before} -> ${zoomedIn} px`);
  await viewer.keyboard.down("Control");
  for (let i = 0; i < 12; i++) await viewer.mouse.wheel(0, 100);
  await viewer.keyboard.up("Control");
  await viewer.waitForTimeout(2000);
  const fitted = await viewer.evaluate(() => {
    const c = document.querySelector("#canvas").getBoundingClientRect();
    return { top: c.top, bottom: innerHeight - c.bottom, left: c.left, right: innerWidth - c.right };
  });
  check(fitted.top >= -1 && fitted.bottom >= -1 && Math.abs(fitted.left - fitted.right) < 2,
        "and zoomed out, the whole page fits the screen, centred",
        JSON.stringify(Object.fromEntries(Object.entries(fitted).map(([k, v]) => [k, Math.round(v)]))));
  await viewer.locator("#stage").click({ position: { x: 40, y: 40 } });
  await viewer.waitForTimeout(2000);
  check(await viewer.locator("#page-frame.expanded").count() === 0 &&
        Math.abs((await widthOfPage()) - inline) < 2,
        "and clicking again puts it back where it was");
  check(await viewer.locator("#hl-btn").isVisible() === true,
        "and comes back with it");

  /* The passage's own text under the page, for the page that renders too faintly to
     read. Checked in a browser rather than by reading the markup for one reason: what
     it is worth depends on it being the SAME characters the ranker scored, and the
     viewer reaches them through the index's per-document JSON, `chunk_offset` and all.
     An off-by-one in that arithmetic would show a neighbouring passage, which reads
     perfectly well and is the wrong text. */
  check(await viewer.locator("#chunk-text").isVisible(),
        "the passage's own text is offered under the page");
  check(!(await viewer.locator("#chunk-text-body").isVisible()),
        "folded away, since the page is the answer and this is the fallback");
  await viewer.click("#chunk-text > summary");
  await viewer.waitForTimeout(100);
  const shown = (await viewer.locator("#chunk-text-body").innerText()).trim();
  check(shown === target.text.trim(), "and opening it shows that passage, character for character",
        `${shown.length} characters against ${target.text.trim().length}`);
  // `pre-wrap`, not `pre` and not the default: the document's own line breaks are what
  // makes a dose table readable here, and `innerText` collapses them under any other
  // value, so a regression would be silent in the string above.
  check(await viewer.locator("#chunk-text-body").evaluate(
          (el) => getComputedStyle(el).whiteSpace === "pre-wrap"),
        "keeping the document's line breaks");
  check(viewerErrors.length === 0, "no uncaught error in the viewer", viewerErrors.slice(0, 3).join(" | "));

  // Asking the same corpus a second question, from inside a document. The scope is
  // the whole point of this control, so what is checked is where the question lands:
  // a result list narrowed to this file, saying so, with one click back to the corpus.
  console.log("searching from inside a document");
  await viewer.click("#doc-search > summary");
  // The document's own title as the question: the real service applies the
  // deployment's floor, and a one-word question scores under it inside most
  // documents, while every chunk of this one was baked with its title in front.
  await viewer.fill("#doc-q", target.title);
  check(await viewer.locator("#doc-search-form").isVisible(), "the viewer offers its own search box");
  check(await viewer.locator("#scope-issuer-row").isVisible() === Boolean(target.issuer),
        "which offers the issuer scope exactly when the document has an issuer",
        target.issuer || "no issuer");
  await viewer.click("#doc-search-form button[type=submit]");
  await viewer.waitForURL(/index\.html\?/, { timeout: 20000 }).catch(() => null);
  const scoped = new URL(viewer.url());
  check(scoped.searchParams.get("q") === target.title &&
        scoped.searchParams.get("file") === target.file,
        "and sends the question to the search page, scoped to this document",
        scoped.search.slice(0, 80));
  await viewer.waitForFunction(() => document.querySelectorAll("#results > li").length > 0,
                               { timeout: 20000 }).catch(() => null);
  const files = await viewer.evaluate(() => [...document.querySelectorAll("#results > li a.hit-title")]
    .map((a) => a.textContent.trim()));
  check(new Set(files).size === 1, "the answer comes from that document only",
        `${new Set(files).size} documents in ${files.length} rows`);
  // The level control, a corpus-wide filter a reader may have saved at a narrow
  // level, has to stand down for a scoped question: most of what a reader searches
  // from inside a document sits below `family`, and a narrow level silently emptied
  // this list until the scope was made to override it. Read off the select rather than off the result
  // count, because "no rows" has too many other causes to point at this one.
  check(await viewer.locator(LEVEL_ID).inputValue() === "all",
        "with the corpus-wide level filter standing down for it",
        await viewer.locator(LEVEL_ID).inputValue());
  check(await viewer.locator("#scope-note").isVisible(), "and the page says it was narrowed");
  await viewer.click("#scope-note button");
  await viewer.waitForFunction(() => !new URL(location.href).searchParams.get("file"),
                               { timeout: 20000 }).catch(() => null);
  check(!new URL(viewer.url()).searchParams.get("file"), "one click widens it to the corpus",
        new URL(viewer.url()).search.slice(0, 60));
  await viewer.close();
}

console.log("console and network");
// Worth its own check: dev_server.py sends the policy read out of docker/Caddyfile,
// so a violation here is a violation on the deployed site. The one that shipped was
// six style="" attributes in the markup, which style-src 'self' blocks while leaving
// the page rendering, slightly wrong, with no other symptom.
check(cspViolations.length === 0, "no Content-Security-Policy violations",
      cspViolations.slice(0, 3).join(" | "));
check(consoleErrors.length === 0, "no console errors", consoleErrors.slice(0, 4).join(" | "));
check(badResponses.length === 0, "no failed requests", badResponses.slice(0, 4).join(" | "));

await browser.close();
console.log(failures ? `\n${failures} check(s) failed` : `\nbrowser check clean${searched ? "" : " (search flow skipped)"}`);
process.exit(failures ? 1 : 0);
