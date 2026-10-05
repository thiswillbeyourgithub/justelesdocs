/* What this site remembers per browser, under one prefix, through one guarded accessor.

   Every setting a reader can change (language, theme, ranking toggles, guideline
   level, list view, the changelog version last seen, the dismissed dev banner) is
   kept in localStorage under `<site id>.<name>`. One prefix, because the
   origin may one day host something else and because "which keys are ours" should
   be answerable by reading one line. One accessor, because localStorage THROWS in a
   private window with site data blocked: every caller used to carry its own
   try/catch, and the one that forgot (the dev banner's read) took the whole banner
   down with it. Here a failed read is null and a failed write is a no-op, so the
   setting lasts for the page and nothing else breaks.

   Two keys predate the prefix (`jlr-theme`, `jlr-lang`, which were copied from
   ../justelesRCP), and a corpus can name more of its own (corpus.toml [site]
   legacy_keys, through site-config.js). They are moved to their new names once,
   at import, so a returning reader keeps their language, theme and
   "already seen" changelog instead of being greeted as a stranger. A new name
   that is already set wins over an old one: it can only have been written by this
   code, so it is the newer choice.

   No DOM, so the migration is testable under Node (tests/store.test.mjs). */

/* The prefix is the corpus's site id (corpus.toml [site] id, through
   site-config.js), so two sites served from one origin during development keep
   apart, and a corpus that renames its site can keep its old id here so no
   returning reader loses a setting. */
export const PREFIX = `${globalThis.__SITE__?.id || "justelesdocs"}.`;

/** Old key -> its name under the prefix. Kept until no reader can still have one. */
export const LEGACY_KEYS = Object.freeze({
  "jlr-theme": "theme",
  "jlr-lang": "lang",
  ...(globalThis.__SITE__?.legacy_keys || {}),
});

/**
 * Move every legacy key to its prefixed name, then delete it.
 *
 * @param {Storage} storage - localStorage, or any object with the same three methods.
 */
export function migrate(storage) {
  for (const [old, name] of Object.entries(LEGACY_KEYS)) {
    const value = storage.getItem(old);
    if (value === null) continue;
    if (storage.getItem(PREFIX + name) === null) storage.setItem(PREFIX + name, value);
    storage.removeItem(old);
  }
}

/* `window` rather than `globalThis.localStorage`: under Node (the tests) there is no
   window, the ReferenceError lands in the catch, and Node's own experimental
   localStorage is never touched. */
function storage() {
  try { return window.localStorage; } catch { return null; }
}

try { migrate(storage()); } catch { /* blocked storage: nothing to move */ }

/**
 * A remembered value, or null when there is none or storage is unavailable.
 *
 * @param {string} name - The key without its prefix.
 * @returns {string|null}
 */
export function load(name) {
  try { return storage().getItem(PREFIX + name); } catch { return null; }
}

/**
 * Remember a value. Silently does nothing when storage is unavailable.
 *
 * @param {string} name - The key without its prefix.
 * @param {string} value
 */
export function save(name, value) {
  try { storage().setItem(PREFIX + name, value); } catch { /* session only */ }
}
