/**
 * Runtime chrome that is the same on every page: optional privacy-friendly
 * metrics, the development banner, the footer, and the bar's shared controls (the
 * status line, the language and theme buttons, and closing menus).
 *
 * Both are driven by window.__APP_CONFIG__, which docker/entrypoint.sh renders
 * from the container environment (docker/env.example documents every key). Every
 * value empty means "off", which is what the checked-in app-config.js says, so a
 * local dev tree tracks nothing and shows no banner.
 *
 * This file exists because the container already did all the work for a feature
 * the page never used: the entrypoint validated ANALYTICS_URL, derived
 * ANALYTICS_ORIGIN and rendered five analytics keys, the Caddyfile opened exactly
 * that origin in script-src and connect-src, and nothing read any of it.
 *
 * ONE DELIBERATE DIFFERENCE from justelesRCP, which is where the umami snippet
 * comes from: this site does not let umami auto-track. A result list lives at
 * `/?q=<the question>`, and a question asked of a health or legal corpus is exactly the
 * kind of string that must not end up in an analytics database. Auto-tracking
 * would send the full URL. So the tag is loaded with data-auto-track="false" and
 * one page view is sent by hand with the path only, and no click or event tracking
 * is wired at all: a result snippet is document text and a facet value is a subject
 * category, and neither is worth a metric.
 *
 * Written by Claude Code.
 */

import { t, plural, externalLink, isHttpUrl, setLang, otherLang, closeMenus } from "./i18n.js";
import { cycleTheme, themeLabel } from "./theme.js";
import { load, save } from "./store.js";

const config = window.__APP_CONFIG__ || {};
const set = (value) => typeof value === "string" && value.trim() !== "" && !value.includes("{{");

/**
 * Load the umami tag, if one is configured, and send a single page view whose URL
 * carries no query string.
 */
function startMetrics() {
  if (!set(config.url) || !set(config.websiteId)) return;
  const script = document.createElement("script");
  script.defer = true;
  script.src = config.url.trim();
  script.setAttribute("data-website-id", config.websiteId.trim());
  // Off by default, and only "false" turns it off: an unset or misspelt value
  // must fall on the side that tracks fewer people.
  script.setAttribute("data-do-not-track",
                      String(config.dnt || "").trim().toLowerCase() === "false" ? "false" : "true");
  // The page view is sent below, deliberately, so the query string never leaves.
  script.setAttribute("data-auto-track", "false");
  if (set(config.sri)) {
    script.integrity = config.sri.trim();
    script.crossOrigin = "anonymous";
  }
  script.addEventListener("load", () => {
    try {
      // Path only. location.search holds the reader's question.
      window.umami?.track?.((props) => ({ ...props, url: location.pathname }));
    } catch {
      // Metrics are never worth an exception on a page that works without them.
    }
  });
  document.head.append(script);
}

// One localStorage key, holding the STARTED_AT the reader dismissed. Storing the stamp
// as the VALUE rather than baking it into the key means a long-lived browser accumulates
// one entry, not one per deploy.
const DEV_DISMISS_KEY = "devBannerDismissed";

// Stop showing the banner once the container has been up this long. The notice says
// "prototype, and here is how fresh this deployment is", which stops being news after a
// few days: past the cutoff the site simply looks normal again, and the next deploy
// stamps a new STARTED_AT and brings it back. Matches justelesRCP's cutoff; keep the two
// in step, the banner is meant to read the same on both sites.
const DEV_MAX_AGE_SECONDS = 5 * 24 * 60 * 60;

/** Elapsed seconds rendered as "3 jours" / "3 days", in the reader's language. */
function humanAgo(seconds) {
  const minutes = Math.max(0, Math.round(seconds / 60));
  if (minutes < 1) return t("ago_now");
  if (minutes < 60) return t("ago_minutes", { n: minutes, s: plural(minutes) });
  const hours = Math.round(minutes / 60);
  if (hours < 24) return t("ago_hours", { n: hours, s: plural(hours) });
  const days = Math.round(hours / 24);
  return t("ago_days", { n: days, s: plural(days) });
}

/** Show the "prototype" banner when the container was started with DEV=1.
 *
 * Same content as justelesRCP's src/dev-banner.js, in both languages: what the site is,
 * how long ago the container last restarted, and a source link when one is configured.
 * The restart time is the point of it. STARTED_AT is stamped by the entrypoint at
 * container start, so the reader can tell a five-minute-old deployment (expect churn)
 * from a five-day-old one, and it keeps ticking while the tab stays open. */
function showDevBanner() {
  if (String(config.dev || "").trim() !== "1") return;

  const startedAt = Number(config.startedAt);
  const hasStart = Number.isFinite(startedAt) && startedAt > 0;
  const elapsed = () => Date.now() / 1000 - startedAt;
  // Past the cutoff there is nothing to announce. Without a real stamp we cannot know the
  // age, so the notice shows: a missing STARTED_AT means a deploy in flight, not an old one.
  if (hasStart && elapsed() > DEV_MAX_AGE_SECONDS) return;

  // The dismissal is scoped to THIS container start, so clicking it away keeps it away
  // across navigations and browser restarts, and the next reboot brings it back. With no
  // stamp nothing is remembered, so an in-flight deploy keeps warning until it lands.
  const token = hasStart ? String(startedAt) : "";
  // A browser that refuses storage reads null and still gets the banner, which is the
  // safe way round: it is a warning, not a preference.
  if (token && load(DEV_DISMISS_KEY) === token) return;

  const banner = document.createElement("div");
  banner.setAttribute("role", "note");
  banner.style.cssText = "position:sticky;top:0;z-index:40;background:var(--accent);" +
    "color:#fff;padding:.45rem .75rem;font-size:.88rem;text-align:center;cursor:pointer";

  // Built from DOM nodes rather than innerHTML: the source URL comes from the container
  // environment, and the CSP here forbids inline style attributes anyway.
  const sourceUrl = isHttpUrl(config.sourceUrl) ? config.sourceUrl.trim() : "";
  const render = () => {
    banner.textContent = "";
    const lead = document.createElement("strong");
    lead.textContent = t("dev_banner");
    banner.append(lead, " ");
    banner.append(hasStart ? t("dev_restarted", { ago: humanAgo(elapsed()) })
                           : t("dev_deploying"));
    if (sourceUrl) {
      banner.append(" ");
      const link = externalLink(sourceUrl, t("dev_source"));
      link.style.color = "#fff";
      link.style.textDecoration = "underline";
      banner.append(link, ".");
    }
    banner.append(` ${t("dev_dismiss")}`);
  };
  render();
  document.addEventListener("langchange", render);

  // Keep "il y a X" honest without a reload, and retire the banner the moment it crosses
  // the cutoff rather than leaving a stale one on a tab that has been open for days.
  const timer = hasStart ? setInterval(() => {
    if (elapsed() > DEV_MAX_AGE_SECONDS) { banner.remove(); clearInterval(timer); return; }
    render();
  }, 60 * 1000) : null;

  banner.addEventListener("click", (event) => {
    if (event.target.closest("a")) return; // let the source link navigate
    banner.remove();
    if (timer) clearInterval(timer);
    if (token) {
      save(DEV_DISMISS_KEY, token);
    }
  });
  document.body.prepend(banner);
}

// What the corpus declares about the site (scripts/stage.py writes it into
// site-config.js from corpus.toml's [site] table). Empty in an unstaged src/ tree.
const SITE = globalThis.__SITE__ || {};

// Where the source link points when the container does not configure one: the
// corpus's repository, else the software's own. stage.py always fills it (from
// scripts/lib/site_config.py, the one place the software's URL is written), so
// it is empty only in an unstaged src/ tree.
const REPO_URL = SITE.repo_url || "";

/**
 * Write a translated sentence carrying {name} placeholders into `el`, substituting
 * DOM nodes for the placeholders.
 *
 * Done by splitting rather than by innerHTML for two reasons: the CSP forbids nothing
 * here but a configured URL still has no business being parsed as markup, and keeping
 * the placeholders inside the translated string lets each language put the links where
 * its own grammar wants them.
 */
function fillTemplate(el, template, nodes) {
  el.textContent = "";
  for (const part of template.split(/(\{[a-z_]+\})/i)) {
    if (!part) continue;
    const name = part.startsWith("{") && part.endsWith("}") ? part.slice(1, -1) : null;
    el.append(name && nodes[name] ? nodes[name] : part);
  }
}

// Built once and then repainted on every language change. Rebuilding the whole footer
// instead would replace #corpus-note, whose node app.js captured at load time and
// writes the passage count into: the count would silently stop updating.
let footParts = null;

/**
 * Fill <footer class="bot" id="foot">, which every page carries empty.
 *
 * Five paragraphs in a deliberate order: who made the site, how to write to them,
 * what it was built with, what it holds, and what it is not. The three that name
 * people come first, because a reader who wants to say something about the corpus
 * should not have to read past the tooling and the legal line to find out how. The
 * two that describe the site follow, and the release-notes button closes it.
 * Bilingual, so it is rendered from i18n rather than written into the markup.
 *
 * justelesRCP's footer opens with another one, naming where the documents come from
 * and under what terms. It was carried over and then removed: this site IS the
 * documents, and a paragraph listing every publishing agency on every page told a reader
 * nothing they could not see in the result row in front of them.
 */
function renderFooter() {
  const foot = document.getElementById("foot");
  if (!foot) return;
  if (!footParts) {
    const wrap = document.createElement("div");
    wrap.className = "wrap";
    // app.js fills this one on the search page and nothing fills it in the viewer,
    // where .bot p:empty hides it rather than leaving a blank first line.
    const corpus = document.createElement("p");
    corpus.id = "corpus-note";
    const disclaimer = document.createElement("p");
    // Who made it, then how to reach them, then what it was made with. The three are
    // separate paragraphs rather than one sentence so that the order is a property of
    // the footer instead of a property of one translated string, which would have to
    // be reordered identically in every language.
    const author = document.createElement("p");
    const suggest = document.createElement("p");
    const credits = document.createElement("p");
    // The only way into the release notes that does not depend on having been here
    // before: changelog.js auto-opens them for a returning reader, and this opens
    // them for everyone else. A button rather than a link, because it navigates
    // nowhere; [data-changelog] is what changelog.js binds to, so the two files
    // share a selector rather than an id neither owns.
    //
    // Dressed as one of the footer's links, on a colophon line after the version it
    // is about ("v0.4.0 · Quoi de neuf ?"). It used to be a boxed button on a line of
    // its own under the legal sentence, the only control in a footer made of prose,
    // which made it look like the page's call to action. The version is left out,
    // with its separator, in an unstaged src/ tree where app-version.js is absent.
    const notes = document.createElement("p");
    const version = String(window.__APP_VERSION__ || "").trim();
    if (version) notes.append(`v${version} · `);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "link-btn";
    button.dataset.changelog = "";
    notes.append(button);
    wrap.append(author, suggest, credits, corpus, disclaimer, notes);
    foot.append(wrap);
    footParts = { author, disclaimer, credits, suggest, button };
  }
  footParts.disclaimer.textContent = t("foot_disclaimer");
  footParts.button.textContent = t("foot_changelog");
  // The issue tracker follows whichever repository the credit sentence points at,
  // so a fork's footer sends its readers to the fork's issues rather than here.
  const repo = isHttpUrl(config.sourceUrl) ? config.sourceUrl.trim().replace(/\/+$/, "") : REPO_URL;
  // The author line needs a name and the suggestion line a contact page, and both
  // are the corpus's to give (corpus.toml's [site] author, the overlay's
  // foot_author_url and foot_contact_url): a site that declares neither shows
  // neither line rather than a sentence with a hole in it.
  footParts.author.hidden = !SITE.author;
  if (SITE.author) {
    fillTemplate(footParts.author, t("foot_author"), {
      author: externalLink(t("foot_author_url"), SITE.author),
    });
  }
  fillTemplate(footParts.credits, t("foot_credits"), {
    agent: externalLink("https://claude.com/claude-code", "Claude Code"),
    // The repository's own name, so a fork's credit names the fork.
    repo: externalLink(repo, repo.split("/").pop()),
  });
  const contact = t("foot_contact_url");
  footParts.suggest.hidden = !isHttpUrl(contact);
  fillTemplate(footParts.suggest, t("foot_suggest"), {
    issue: externalLink(`${repo}/issues`, t("foot_suggest_issue")),
    contact: externalLink(contact, t("foot_suggest_contact")),
  });
}

/**
 * Say something on the status line under the bar, which every page carries.
 *
 * @param {string} message - "" to clear it.
 * @param {boolean} [isError] - Styled as an error.
 */
export function setStatus(message, isError = false) {
  const status = document.getElementById("status");
  if (!status) return;
  status.textContent = message;
  status.classList.toggle("error", isError);
}

/**
 * Wire the controls every page's bar has, so no page carries its own copy.
 *
 * The language button's caption comes from data-i18n like any other string; the
 * theme button's is a symbol, set here and on every click.
 */
function wireBar() {
  const themeBtn = document.getElementById("theme-btn");
  if (themeBtn) {
    themeBtn.textContent = themeLabel();
    themeBtn.addEventListener("click", () => { cycleTheme(); themeBtn.textContent = themeLabel(); });
  }
  document.getElementById("lang-btn")?.addEventListener("click", () => setLang(otherLang()));
  // A menu the reader has looked away from is a menu they would have to close by hand,
  // and a list can have one open per row. Here, once, so the pages cannot drift apart
  // on when a menu is done.
  document.addEventListener("click", (event) => closeMenus(event.target));
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeMenus();
  });
}

startMetrics();
// The banner needs i18n's language to have settled, which it has by import time,
// and a body to prepend to.
function start() {
  wireBar();
  showDevBanner();
  renderFooter();
}
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", start, { once: true });
} else {
  start();
}
// The footer is the only chrome here that has to follow the language toggle. The dev
// banner registers its own listener, from inside showDevBanner, because it only has
// something to repaint when it is on screen at all.
document.addEventListener("langchange", renderFooter);
