// The localStorage helper's one-time rename of the keys that predate the prefix.
// Written by Claude Code (Opus 5.5).
import { test } from "node:test";
import assert from "node:assert/strict";
import { migrate, LEGACY_KEYS, PREFIX, load, save } from "../src/store.js";

/* No site-config.js under Node, so the prefix is the software's default. */
const P = PREFIX;

/** A Map with Storage's three methods, which is all migrate touches. */
function fakeStorage(entries) {
  const map = new Map(Object.entries(entries));
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => map.set(k, String(v)),
    removeItem: (k) => map.delete(k),
    dump: () => Object.fromEntries(map),
  };
}

test("an old key moves to its prefixed name and is removed", () => {
  const s = fakeStorage({ "jlr-theme": "dark", "jlr-lang": "en" });
  migrate(s);
  assert.deepEqual(s.dump(), {
    [`${P}theme`]: "dark",
    [`${P}lang`]: "en",
  });
});

test("a prefixed key already set wins over the old one, which is still removed", () => {
  const s = fakeStorage({ "jlr-lang": "fr", [`${P}lang`]: "en" });
  migrate(s);
  assert.deepEqual(s.dump(), { [`${P}lang`]: "en" });
});

test("migrating twice changes nothing, and unrelated keys are left alone", () => {
  const s = fakeStorage({ "jlr-theme": "light", "other": "x" });
  migrate(s);
  migrate(s);
  assert.deepEqual(s.dump(), { [`${P}theme`]: "light", "other": "x" });
  assert.equal(Object.keys(LEGACY_KEYS).length, 2);
});

test("with no window (blocked storage), load is null and save does not throw", () => {
  assert.equal(load("theme"), null);
  assert.doesNotThrow(() => save("theme", "dark"));
});

test("without a site id the prefix is the software's name", () => {
  assert.equal(PREFIX, "justelesdocs.");
});

test("a corpus's own legacy keys come from site-config.js and migrate too", async () => {
  globalThis.__SITE__ = { id: "demo", legacy_keys: { demo_seen_version: "changelogSeen" } };
  try {
    // A query string makes a fresh module instance, which reads __SITE__ at load.
    const store = await import("../src/store.js?legacy");
    const s = fakeStorage({ "jlr-lang": "en", demo_seen_version: "1.4.0" });
    store.migrate(s);
    assert.deepEqual(s.dump(), { "demo.lang": "en", "demo.changelogSeen": "1.4.0" });
  } finally {
    delete globalThis.__SITE__;
  }
});
