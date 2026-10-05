/* The browser's half of src/search.js's world, served from the filesystem.
 *
 * src/search.js talks to the network through fetch(), which is exactly the
 * surface a browser gives it. Everything that runs the module under node goes
 * through this one shim: the search service (server/), whose whole job is to run
 * the shipped ranker against the index on disk, and the node scripts that gate and
 * measure it (scripts/check_search.mjs, the hard gate; scripts/evaluate_rescore.mjs
 * and scripts/sweep_blend.mjs, the measurements). One copy, because a second one
 * would drift, and the point of all four is that they exercise the SHIPPED code
 * path rather than a restatement of it.
 *
 * Two properties matter and are deliberate:
 *   - a fetch of anything outside the index directory THROWS rather than quietly
 *     succeeding, so a module that starts reading something it should not is
 *     caught here rather than in production;
 *   - the encoder request is forwarded to a real embed-service.py when one is
 *     offered, so the wire format (base64 int8, the per-request `dim`, the width
 *     check) is exercised rather than reimplemented.
 *
 * Written by Claude Code (Opus 5); moved under server/ by Claude Code (Fable 5.1).
 */

import { readFile } from "node:fs/promises";

let realFetch = null;

/**
 * Install the shim as globalThis.fetch.
 *
 * @param {object} options
 * @param {string} options.indexDir Absolute path of the index directory, served as
 *   the relative prefix "index/".
 * @param {string} [options.embedUrl] Base URL of a running embed-service.py. When
 *   empty, any "api/" fetch throws.
 * @param {number} [options.apiTimeoutMs] Abort an encoder request after this
 *   long, when the caller passed no signal of its own. 0 means wait forever,
 *   which is fine for a gate and not for a service.
 * @returns {{ nodeFetch: typeof fetch, reachable: (base: string) => Promise<boolean> }}
 */
export function installFetchShim({ indexDir, embedUrl = "", apiTimeoutMs = 0 }) {
  // The platform's fetch, taken once: a second install (a test pointing the same
  // process at another index or encoder) must forward to the network, not to
  // the shim it is replacing.
  realFetch ??= globalThis.fetch;
  const nodeFetch = realFetch;
  const base = embedUrl.replace(/\/$/, "");

  globalThis.fetch = async (url, init) => {
    if (String(url).startsWith("api/")) {
      if (!base) throw new Error(`no EMBED_URL, cannot serve ${url}`);
      const timed = apiTimeoutMs && !init?.signal ? { ...init, signal: AbortSignal.timeout(apiTimeoutMs) } : init;
      return nodeFetch(`${base}/${url.replace(/^api\//, "api/")}`, timed);
    }
    const path = String(url).replace(/^index\//, `${indexDir}/`);
    if (!path.startsWith(indexDir)) {
      throw new Error(`unexpected fetch of ${url}: only the index may be read here`);
    }
    let body;
    try {
      body = await readFile(path);
    } catch {
      return { ok: false, status: 404, url: String(url) };
    }
    return {
      ok: true,
      status: 200,
      url: String(url),
      json: async () => JSON.parse(body.toString("utf-8")),
      // A copy, not a view on Node's pooled Buffer: the pool would hand
      // search.js a window onto unrelated bytes and the byte-count check would
      // pass anyway.
      arrayBuffer: async () => body.buffer.slice(body.byteOffset, body.byteOffset + body.byteLength),
    };
  };

  /** True when something accepts a connection at `url`, whatever it answers. */
  const reachable = async (url) => {
    try {
      // The health endpoint is the polite target, but any response at all proves
      // a listener, and this runs before the wire format is exercised, so a 404
      // from some other service is still worth attempting the real request
      // against.
      await nodeFetch(`${url.replace(/\/$/, "")}/api/sem/health`, { signal: AbortSignal.timeout(2000) });
      return true;
    } catch {
      return false;
    }
  };

  return { nodeFetch, reachable };
}

/**
 * Install just enough `document` for src/i18n.js to switch language under node.
 *
 * `setLang` is not a pure string lookup: it mirrors the choice into `<html lang>`
 * and re-applies every data-i18n binding, because those are the two things that
 * have to happen together in a browser or the page ends up half translated. None
 * of that is meaningful here, but it all has to not throw, so that a node gate can
 * ask the SHIPPED string table what a facet value reads as in each language rather
 * than reimporting the table and checking a copy.
 *
 * Deliberately minimal and deliberately not a DOM: `querySelectorAll` returns
 * nothing, so nothing is rendered and no test can come to depend on it. A gate
 * needing real rendering is a browser gate (check_ui.mjs), not this.
 */
export function installDocumentStub() {
  if (globalThis.document) return;
  const empty = { forEach() {} };
  globalThis.document = {
    documentElement: { lang: "fr" },
    querySelectorAll: () => empty,
    // A property bag, not an element. The string helpers in src/i18n.js that build a
    // link (sourceLink, identifierLink) only ever ASSIGN to what createElement hands
    // back, so a test can read the href and the label off it without a DOM. It is
    // deliberately not more than that: anything here that behaved almost like an
    // element would let a test pass on behaviour the browser does not have.
    createElement: (tag) => ({
      tagName: String(tag).toUpperCase(),
      children: [],
      append(...nodes) { this.children.push(...nodes); },
    }),
    dispatchEvent: () => true,
    addEventListener: () => {},
  };
}
