/* What a corpus's site-config.js changes in src/i18n.js and src/store.js.
 *
 * Both modules read window.__SITE__ once, at import, so this file sets it on
 * globalThis BEFORE importing them, under a query string that gives it module
 * instances of its own (i18n.test.mjs imports the plain ones, with no site).
 *
 * Written by Claude Code (Opus 5.5).
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { installDocumentStub } from "../server/lib/browser_shim.mjs";

installDocumentStub();
globalThis.__SITE__ = {
  id: "mycorpus",
  languages: ["en", "fr", "de"],
  default_language: "en",
  strings: {
    en: { tagline: "Search the widget manuals", facet_topic_gears: "Gears" },
    de: { tagline: "Suche in den Handbüchern" },
  },
};
const i18n = await import("../src/i18n.js?site");
const store = await import("../src/store.js?site");

test("the site id is the storage prefix", () => {
  assert.equal(store.PREFIX, "mycorpus.");
});

test("the overlay replaces a software string and adds its own keys", () => {
  i18n.setLang("en");
  assert.equal(i18n.t("tagline"), "Search the widget manuals");
  assert.equal(i18n.t("facet_topic_gears"), "Gears");
  // A key the overlay leaves alone is still the software's.
  assert.equal(i18n.t("search_button"), "Search");
});

test("a language with no software table falls back to the default language key by key", () => {
  i18n.setLang("de");
  assert.equal(i18n.lang(), "de");
  assert.equal(i18n.t("tagline"), "Suche in den Handbüchern");
  assert.equal(i18n.t("search_button"), "Search");
  assert.equal(i18n.t("facet_topic_gears"), "Gears");
  i18n.setLang("en");
});

test("the toggle cycles through every language, in the corpus's order", () => {
  i18n.setLang("en");
  assert.deepEqual(i18n.languages(), ["en", "fr", "de"]);
  assert.equal(i18n.otherLang(), "fr");
  i18n.setLang("fr");
  assert.equal(i18n.otherLang(), "de");
  i18n.setLang("de");
  assert.equal(i18n.otherLang(), "en");
  i18n.setLang("en");
});

test("a language the site does not offer is refused", () => {
  i18n.setLang("it");
  assert.equal(i18n.lang(), "en");
});
