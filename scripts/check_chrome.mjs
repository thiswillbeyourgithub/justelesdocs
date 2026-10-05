/**
 * Browser gate for the chrome that is the same on every page: the development
 * banner and the footer (src/site.js).
 *
 * Both are driven by window.__APP_CONFIG__, which the container renders from its
 * environment, so neither can be exercised by loading the site as deployed: the
 * checked-in app-config.js is all empty strings and a real deployment has exactly
 * one STARTED_AT, the current one. This file therefore intercepts app-config.js
 * per scenario and serves a synthetic one, which is the only way to see what a
 * reader sees six days after a deploy without waiting six days.
 *
 * The banner is worth a gate rather than a reading because every one of its rules
 * is a time or storage rule, and those are exactly the ones that look right in the
 * source and misbehave in a browser: the dismissal is keyed on the container start
 * stamp (so a reboot must bring the banner back, and a reload must not), and the
 * whole notice retires itself five days after that stamp.
 *
 * The locale is pinned to fr-FR because i18n.js picks its default language from
 * navigator.languages: on a machine whose browser reports English, an unpinned
 * context silently tests the English strings against French expectations.
 *
 *   uv run scripts/dev_server.py &                      # dist/ on :8649
 *   PW=<path to a playwright install> SITE=http://127.0.0.1:8649 \
 *     node scripts/check_chrome.mjs
 *
 * PW is the playwright package to load (this repo has no node_modules and no
 * package.json, by the same decision that keeps the site free of a build step).
 *
 * Written by Claude Code.
 */

import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PW || "playwright");

const SITE = (process.env.SITE || "http://127.0.0.1:8649").replace(/\/$/, "");
const HOUR = 3600;
const DAY = 24 * HOUR;

let failures = 0;
const check = (ok, label, detail = "") => {
  if (ok) console.log(`  ok   ${label}${detail ? `  ${detail}` : ""}`);
  else { failures++; console.error(`  FAIL ${label}${detail ? `: ${detail}` : ""}`); }
};

const browser = await chromium.launch();
// One context per scenario would be simpler, but the dismissal checks need
// localStorage to survive a reload, so the context is the unit of isolation and
// the scenarios that must not see each other's storage get their own.
const newContext = () => browser.newContext({ locale: "fr-FR" });

/** A STARTED_AT stamp for a container that booted `ago` seconds ago. */
const stamp = (ago) => Math.floor(Date.now() / 1000) - ago;

/**
 * Open a page with a synthetic app-config.js.
 *
 * `startedAt` is an absolute stamp rather than an age on purpose: the dismissal is
 * keyed on its exact value, so deriving it from the clock at each call would make two
 * loads of the "same" deployment differ by the second that elapsed between them, and
 * the reload check would see a reboot.
 *
 * @param ctx        browser context to open in
 * @param dev        value of DEV ("1" turns the banner on)
 * @param startedAt  container start stamp in epoch seconds, or null for no STARTED_AT
 * @param source     value of SOURCE_URL
 * @param path       page to open, so the footer can be checked on the viewer too
 */
async function open(ctx, { dev = "1", startedAt = null, source = "", path = "/index.html" } = {}) {
  const page = await ctx.newPage();
  const stampValue = startedAt === null ? "" : String(startedAt);
  await page.route("**/app-config.js", (route) => route.fulfill({
    contentType: "text/javascript",
    body: `window.__APP_CONFIG__ = {dev:'${dev}', startedAt:'${stampValue}', sourceUrl:'${source}'};`,
  }));
  await page.goto(SITE + path, { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(300);
  return page;
}

const bannerText = (page) =>
  page.locator("[role=note]").first().textContent().catch(() => "");
const footText = (page) => page.locator("#foot").first().innerText().catch(() => "");

console.log("development banner");
{
  const ctx = await newContext();
  const page = await open(ctx, { startedAt: stamp(3 * HOUR), source: "https://example.org/repo" });
  const text = await bannerText(page);
  check(/Prototype en d.veloppement\./.test(text), "shows the prototype line", JSON.stringify(text));
  check(/Dernier red.marrage il y a 3 heures\./.test(text), "and the delay since the last reboot");
  check(await page.locator("[role=note] a").count() === 1, "and a source link when SOURCE_URL is set");

  await page.locator("#lang-btn").first().click();
  await page.waitForTimeout(200);
  const en = await bannerText(page);
  check(/Prototype under development\..*Last restarted 3 hours ago\./.test(en),
        "English after the language toggle", JSON.stringify(en));
  await ctx.close();
}
{
  const ctx = await newContext();
  let page = await open(ctx, { startedAt: stamp(6 * DAY) });
  check(await page.locator("[role=note]").count() === 0, "gone once the container is 6 days old");
  await page.close();
  page = await open(ctx, { startedAt: stamp(4 * DAY) });
  check(await page.locator("[role=note]").count() === 1, "still there at 4 days");
  await ctx.close();
}
{
  const ctx = await newContext();
  const startedAt = stamp(2 * HOUR);
  let page = await open(ctx, { startedAt });
  await page.locator("[role=note]").first().click();
  await page.waitForTimeout(150);
  check(await page.locator("[role=note]").count() === 0, "click dismisses it");
  await page.close();
  // Same context AND the same STARTED_AT: this is a reload, not a reboot.
  page = await open(ctx, { startedAt });
  check(await page.locator("[role=note]").count() === 0, "and it stays dismissed across a reload");
  await page.close();
  page = await open(ctx, { startedAt: stamp(1) });
  check(await page.locator("[role=note]").count() === 1, "and comes back after a reboot");
  await ctx.close();
}
{
  const ctx = await newContext();
  let page = await open(ctx, {});
  const text = await bannerText(page);
  check(/D.ploiement en cours\./.test(text), "an unstamped deploy says so instead", JSON.stringify(text));
  await page.locator("[role=note]").first().click();
  await page.waitForTimeout(150);
  await page.close();
  page = await open(ctx, {});
  check(await page.locator("[role=note]").count() === 1, "and cannot be dismissed for good");
  await ctx.close();
}
{
  const ctx = await newContext();
  const page = await open(ctx, { dev: "0" });
  check(await page.locator("[role=note]").count() === 0, "DEV=0 shows nothing at all");
  await ctx.close();
}

console.log("footer");
// What the footer should say is the SITE's: its author, its links and its
// disclaimer come from the corpus's overlay, through the served site-config.js.
// The expectations are read from there and from the software's own string table
// (src/i18n.js, imported with the same __SITE__ the page gets), so this gate holds
// any corpus to its own words rather than to one corpus's.
const SITE_CONFIG = await fetch(`${SITE}/site-config.js`).then((r) => (r.ok ? r.text() : "")).catch(() => "");
globalThis.__SITE__ = SITE_CONFIG
  ? JSON.parse(SITE_CONFIG.slice(SITE_CONFIG.indexOf("{"), SITE_CONFIG.lastIndexOf("}") + 1)) : {};
const { installDocumentStub } = await import(new URL("../server/lib/browser_shim.mjs", import.meta.url).href);
installDocumentStub();  // setLang mirrors the choice into <html lang>; see the shim
const i18n = await import(new URL("../src/i18n.js", import.meta.url).href);
const [LANG_A, LANG_B] = i18n.languages();
const said = (lang, key) => { i18n.setLang(lang); return i18n.t(key); };
// The part of a template before its first placeholder: "Built by {author}." -> "Built by".
const lead = (template) => template.split("{")[0].trim();
const AUTHOR = globalThis.__SITE__.author || "";
{
  const ctx = await newContext();
  const page = await open(ctx, { dev: "0" });
  const hrefs = (p) => p.locator("#foot a").evaluateAll((els) => els.map((el) => el.getAttribute("href") || ""));
  const found = (list, wanted) => list.filter((href) => href === wanted).length === 1;
  const lineAt = (text, needle) => text.split("\n").map((l) => l.trim()).filter(Boolean)
    .findIndex((l) => l.includes(needle));
  for (const [index, lang] of [LANG_A, LANG_B].filter(Boolean).entries()) {
    if (index > 0) {
      await page.locator("#lang-btn").first().click();
      await page.waitForTimeout(200);
    }
    const text = await footText(page);
    const links = await hrefs(page);
    check(!/^Source\s*:/m.test(text), `${lang}: no source paragraph: the site is the documents`,
          JSON.stringify(text.slice(0, 60)));
    check(text.includes(said(lang, "foot_disclaimer").slice(0, 30)), `${lang}: the disclaimer is shown`,
          JSON.stringify(text.split("\n").filter(Boolean).pop()));
    // The order of the lines is the decision, not the wording: the person, then how
    // to write to them, then the tooling. Each line reads well on its own, so only a
    // check on the sequence can catch it being lost.
    const contact = said(lang, "foot_contact_url");
    const order = [AUTHOR && `${lead(said(lang, "foot_author"))} ${AUTHOR}`,
                   /^https?:/.test(contact) && lead(said(lang, "foot_suggest")),
                   lead(said(lang, "foot_credits"))].filter(Boolean);
    check(order.every((needle, i) => lineAt(text, needle) === i), `${lang}: the footer lines in order`,
          JSON.stringify(text.split("\n").filter(Boolean).slice(0, order.length)));
    // Every language edition has to keep its links: each sentence is assembled by
    // splitting the translated string on its placeholders, so a renamed placeholder
    // silently drops the link rather than failing loudly.
    const authorUrl = said(lang, "foot_author_url");
    if (AUTHOR && authorUrl) check(found(links, authorUrl), `${lang}: the author link is the language's own`, links.join(" "));
    const repo = links.find((href) => /^https:\/\/github\.com\/[^/]+\/[^/]+$/.test(href)) || "";
    check(repo !== "", `${lang}: a repository link, the public repo when SOURCE_URL is unset`, links.join(" "));
    if (/^https?:/.test(contact)) {
      check(found(links, contact), `${lang}: a contact page in the reader's language`, links.join(" "));
      check(links.includes(`${repo}/issues`), `${lang}: with the issue tracker of that same repository`,
            links.join(" "));
    }
  }
  await ctx.close();
}
{
  const ctx = await newContext();
  const page = await open(ctx, { dev: "0", source: "https://example.org/fork" });
  const forkLinks = await page.locator("#foot a").evaluateAll((els) => els.map((el) => el.getAttribute("href") || ""));
  check(forkLinks.includes("https://example.org/fork"), "SOURCE_URL overrides the repository link",
        forkLinks.join(" "));
  // A fork should collect its own issues; sending them here would be a bug nobody
  // running the fork would ever see.
  check(forkLinks.includes("https://example.org/fork/issues"), "and the issue link follows it",
        forkLinks.join(" "));
  await ctx.close();
}
{
  const ctx = await newContext();
  const page = await open(ctx, { dev: "0", path: "/view.html" });
  const text = await footText(page);
  check(text.includes(said(LANG_A, "foot_disclaimer").slice(0, 30)), "the viewer carries the same footer");
  // The passage count is written by app.js, which the viewer does not load, so on this
  // page its paragraph stays empty and `.bot p:empty` has to hide it or the footer
  // carries a hole. Asserted on the computed style rather than on the text, because
  // innerText collapses an empty block away whether it is displayed or not: reading
  // the text back would pass even with the rule deleted.
  const emptyShown = await page.locator("#foot p").evaluateAll(
    (els) => els.filter((el) => !el.textContent.trim() && getComputedStyle(el).display !== "none").length);
  check(emptyShown === 0, "with no hole where the corpus count would be",
        `${emptyShown} empty paragraph(s) still displayed`);
  await ctx.close();
}

{
  // The document page is a second page of the site, not a second site: it loads the
  // same site.js, so the footer and the development banner have to be there too.
  // Cheap to check and easy to forget, since nothing else on that page comes from
  // this file.
  const ctx = await newContext();
  const page = await open(ctx, { dev: "1", path: "/browse.html" });
  check(await page.locator("#foot a").count() === 5, "the document page carries the footer");
  check(/Prototype en d.veloppement\./.test(await bannerText(page)),
        "and the development banner");
  await ctx.close();
}

// --- the release-notes popup (src/changelog.js) ------------------------------
// Worth a gate for the same reason the banner is: every rule it has is a storage
// rule against a version stamp, and those look right in the source and misbehave
// in a browser. The three cases below are the three a reader can be in.
// src/store.js prefix (the site id, from the served site-config.js) + changelog.js SEEN_KEY
const SITE_ID = await fetch(`${SITE}/site-config.js`)
  .then((r) => (r.ok ? r.text() : ""))
  .then((text) => (text.match(/"id":\s*"([^"]+)"/) || [])[1] || "justelesdocs")
  .catch(() => "justelesdocs");
const SEEN = `${SITE_ID}.changelogSeen`;

/** Open a page having pretended this browser last saw `version` (null: never here). */
async function withSeen(version, path = "/index.html") {
  const ctx = await newContext();
  if (version !== null) {
    // Seeded only when nothing is stored yet: this runs on EVERY navigation in the
    // context, and overwriting would undo what the popup itself stored, hiding the
    // bug where it forgets to store and opens again on the next visit.
    await ctx.addInitScript(([key, value]) => {
      try {
        if (localStorage.getItem(key) === null) localStorage.setItem(key, value);
      } catch { /* as the popup does */ }
    }, [SEEN, version]);
  }
  const page = await open(ctx, { dev: "0", path });
  return { ctx, page };
}

console.log("release notes");
{
  // A returning reader whose last visit predates this version: the popup opens by
  // itself, showing only what is newer than what they saw.
  const { ctx, page } = await withSeen("0.0.1");
  check(await page.locator(".cl-card").count() === 1, "open themselves for a returning reader");
  check(await page.locator(".cl-title").textContent() === "Quoi de neuf ?", "with the what's-new title");
  check(/version 0\.0\.1/.test(await page.locator(".cl-sub").textContent()),
        "saying which version they last saw");
  const items = await page.locator(".cl-items li").count();
  check(items > 0, "and the bullets of the release itself", `${items} bullets`);
  check(await page.locator(".cl-release").count() === 1, "one release, since only one is newer");
  // The commit links are what make a bullet checkable against the code.
  const sha = page.locator(".cl-sha").first();
  check(/\/commit\/[0-9a-f]{7,40}$/.test(await sha.getAttribute("href")), "each bullet links to its commit");

  // Keyboard, not a click: the overlay covers the page, so the pointer cannot reach
  // the language button while the popup is open. Nothing traps focus in the card, so
  // a keyboard reader still can, and that is the path the langchange listener serves.
  await page.locator("#lang-btn").first().focus();
  await page.keyboard.press("Enter");
  await page.waitForTimeout(200);
  check(await page.locator(".cl-title").textContent() === "What's new?",
        "the open popup follows the language toggle");
  check(/New features/.test(await page.locator(".cl-cat").first().textContent()),
        "category labels included, which ride in changelog.json rather than in the page");

  await page.keyboard.press("Escape");
  await page.waitForTimeout(150);
  check(await page.locator(".cl-card").count() === 0, "and Escape closes them");

  // The version is stored on open, so a reload must be quiet.
  const again = await ctx.newPage();
  await again.goto(SITE + "/index.html", { waitUntil: "domcontentloaded" });
  await again.waitForTimeout(400);
  check(await again.locator(".cl-card").count() === 0, "and they stay closed on the next visit");
  await ctx.close();
}
{
  // A first-time visitor has no "since" to be shown, so the popup must not open on
  // them: it would read as a changelog for a site they have never seen.
  const { ctx, page } = await withSeen(null);
  check(await page.locator(".cl-card").count() === 0, "stay shut for a first-time visitor");
  const stored = await page.evaluate((key) => localStorage.getItem(key), SEEN);
  check(stored === "0.1.0", "but remember this version, so the NEXT release is news", `${stored}`);

  // The footer opener is the way in for everyone the auto-open skips.
  await page.locator("#foot [data-changelog]").click();
  await page.waitForTimeout(200);
  check(await page.locator(".cl-card").count() === 1, "and the footer opens them by hand");
  check(await page.locator(".cl-title").textContent() === "Journal des versions",
        "showing the whole history rather than a since-view");
  await ctx.close();
}
{
  // Same version stored as the site's: nothing to say, on either page.
  const { ctx, page } = await withSeen("0.1.0", "/view.html");
  check(await page.locator(".cl-card").count() === 0, "stay shut for a reader who is up to date");
  check(await page.locator("#foot [data-changelog]").count() === 1,
        "and the viewer carries the opener too");
  await ctx.close();
}

await browser.close();
console.log(failures ? `\n${failures} chrome check(s) failed` : "\nall chrome checks passed");
process.exit(failures ? 1 : 0);
