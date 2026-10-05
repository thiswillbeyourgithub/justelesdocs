/* The "Quoi de neuf ?" / "What's new?" popup.

   The notes are authored per release in docs/changelog/<version>/changelog.md and
   compiled by scripts/changelog.py into /changelog.json, which scripts/stage.py
   writes into the shipped tree. This file only decides WHEN to show them and
   renders them.

   Auto-open rule: the browser remembers the last version whose notes it showed
   (localStorage). On a later visit, if the site's version (window.__APP_VERSION__,
   from the generated app-version.js) is newer, the popup opens with every release
   since the stored one and then stores the new version. A FIRST-time visitor sees
   nothing, because there is no "since" to show: the current version is stored
   silently. That rule is the whole point of the feature, so it is worth stating
   what it is not: this is not a "new here?" greeting, it is a "you have been here
   before and the site has changed since" one.

   Nothing auto-starts on this site except this, which is why there is no guard
   against opening on top of the guided tour the way justelesRCP has one: the tour
   here only ever runs when a reader presses its button.

   Bilingual, like everything else in src/: each bullet ships in both languages and
   the reader's current one is picked, including when the language is toggled while
   the popup is open.

   CSP-safe: same-origin fetch, no inline handlers, no innerHTML, no style
   attributes (docker/Caddyfile serves style-src 'self', which blocks those but not
   el.style.x from script). It is loaded on every page but fetches changelog.json
   only when it actually has something to show.

   Written by Claude Code. */

import { t, lang, externalLink, isHttpUrl } from "./i18n.js";
import { load as loadSetting, save as saveSetting } from "./store.js";

/** Last version whose notes were shown in this browser. */
const SEEN_KEY = "changelogSeen";

let data = null;      // the parsed changelog.json, fetched at most once
let overlay = null;   // the open popup, if any
let since = null;     // version the open popup is showing "since", or null for all
let refresh = null;   // repaints the open popup in the current language

/* store.js swallows a blocked localStorage: a read is null, a write does nothing.
   Here that degrades to "never auto-open" (every visit looks like a first one),
   which is the quiet side to fail on. */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

/** Compare two `major.minor.patch` strings numerically: "0.9.0" is BEFORE "0.10.0". */
function compare(a, b) {
  const left = String(a).split("."), right = String(b).split(".");
  for (let i = 0; i < 3; i++) {
    const delta = (parseInt(left[i], 10) || 0) - (parseInt(right[i], 10) || 0);
    if (delta) return delta < 0 ? -1 : 1;
  }
  return 0;
}

/** A release date in the reader's language, falling back to the ISO string. */
function readableDate(iso) {
  try {
    // Noon, not midnight: a date-only string parsed as UTC and then rendered in a
    // timezone behind UTC shows the previous day.
    return new Date(`${iso}T12:00:00`).toLocaleDateString(t("locale"),
                                                          { day: "numeric", month: "long", year: "numeric" });
  } catch {
    return iso;
  }
}

function renderRelease(release) {
  const box = el("section", "cl-release");
  const heading = el("h3", "cl-version", t("changelog_version", { v: release.version }));
  heading.append(el("span", "cl-date", readableDate(release.date)));
  box.append(heading);
  for (const section of release.sections || []) {
    // The category labels ride in the JSON (changelog.py's CATEGORIES), in both
    // languages, so this file keeps no copy of that table.
    const labels = (data.categories || {})[section.key] || {};
    box.append(el("h4", "cl-cat", labels[lang()] || labels.fr || section.key));
    const list = el("ul", "cl-items");
    for (const item of section.items || []) {
      const row = el("li", null, item[lang()] || item.fr || item.en || "");
      for (const sha of item.commits || []) {
        // Only an http(s) base becomes a link, as every other off-site link here: an
        // empty or odd commit_url would otherwise make a relative href to "/<sha>".
        const link = isHttpUrl(data.commit_url)
          ? externalLink(data.commit_url + sha, sha.slice(0, 7))
          : el("span", null, sha.slice(0, 7));
        link.className = "cl-sha";
        link.title = t("changelog_commit");
        row.append(" ", link);
      }
      list.append(row);
    }
    box.append(list);
  }
  return box;
}

/** Fill the scrolling body with every release newer than `from`, or with all of them. */
function fill(body, from) {
  const fragment = document.createDocumentFragment();
  for (const release of data.releases || []) {
    if (!from || compare(release.version, from) > 0) fragment.append(renderRelease(release));
  }
  body.textContent = "";
  body.append(fragment);
}

function close() {
  if (!overlay) return;
  document.removeEventListener("keydown", onKey, true);
  document.body.classList.remove("cl-open");
  overlay.remove();
  overlay = null;
  since = null;
  refresh = null;
}

function onKey(event) {
  if (event.key === "Escape") { event.preventDefault(); close(); }
}

/**
 * Open the popup.
 * @param {string|null} from  show only releases after this version; null shows all.
 */
function open(from) {
  close();
  since = from;
  overlay = el("div", "cl-overlay");
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) close();   // a click outside the card dismisses it
  });

  const card = el("div", "cl-card");
  card.setAttribute("role", "dialog");
  card.setAttribute("aria-modal", "true");

  const head = el("div", "cl-head");
  const title = el("h2", "cl-title");
  const close_ = el("button", "plain cl-x", "×");
  close_.type = "button";
  close_.addEventListener("click", close);
  head.append(title, close_);

  const sub = el("p", "cl-sub");
  const body = el("div", "cl-body");
  const foot = el("div", "cl-foot");
  const all = el("button", "plain", "");
  all.type = "button";
  all.addEventListener("click", () => {
    since = null;
    fill(body, null);
    body.scrollTop = 0;
    paint();
  });
  const ok = el("button", "plain", "");
  ok.type = "button";
  ok.addEventListener("click", close);
  foot.append(all, ok);

  // Everything the language touches is repainted from one place, so the langchange
  // listener below is a call to this rather than a second set of assignments that
  // could drift from these.
  function paint() {
    title.textContent = since ? t("changelog_title_new") : t("changelog_title_all");
    card.setAttribute("aria-label", title.textContent);
    sub.textContent = since ? t("changelog_since", { v: since }) : "";
    sub.hidden = !since;
    all.textContent = t("changelog_show_all");
    all.hidden = !since;
    ok.textContent = t("changelog_close");
    close_.setAttribute("aria-label", ok.textContent);
  }

  fill(body, from);
  paint();
  card.append(head, sub, body, foot);
  overlay.append(card);
  document.body.append(overlay);
  // Freezes the page behind the overlay: without it, scrolling the popup to its end
  // carries on scrolling the result list underneath.
  document.body.classList.add("cl-open");
  document.addEventListener("keydown", onKey, true);
  ok.focus();

  // Picked up by the one langchange listener at the bottom of this file. Registering
  // a listener here instead would add one per open and leave them all subscribed,
  // repainting popups that no longer exist.
  refresh = () => { fill(body, since); paint(); };
}

/** Fetch changelog.json once, then run `then`. Silent if the site ships no notes. */
async function load(then) {
  if (data) return then();
  try {
    const response = await fetch("changelog.json");
    if (!response.ok) return;
    data = await response.json();
  } catch {
    return;   // no notes served, or offline: say nothing
  }
  then();
}

function boot() {
  // Manual openers work whether or not there is anything new, and show everything.
  for (const node of document.querySelectorAll("[data-changelog]")) {
    node.addEventListener("click", (event) => {
      event.preventDefault();
      load(() => open(null));
    });
  }

  const current = window.__APP_VERSION__;
  // Absent in a plain src/ tree opened without staging: nothing to compare against,
  // so nothing opens. scripts/stage.py generates it into dist/ from VERSION.
  if (!current) return;
  const seen = loadSetting(SEEN_KEY);
  if (!seen) return saveSetting(SEEN_KEY, current);      // first visit: store, stay quiet
  if (compare(seen, current) >= 0) return;            // up to date, or a rollback
  load(() => {
    saveSetting(SEEN_KEY, current);
    open(seen);
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot, { once: true });
} else {
  boot();
}

// Follow the language toggle while the popup is open, keeping the reader where they
// were rather than closing on them.
document.addEventListener("langchange", () => refresh?.());
