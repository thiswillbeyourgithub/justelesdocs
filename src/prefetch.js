/* Warm the viewer's downloads while the reader is still reading the result list.
 *
 * What a reader waits for after clicking a result is not the search, which is over:
 * it is pdf.js (643 KB of module plus a 2.2 MB worker), then `index/doc/<id>.json`,
 * then enough of the PDF for one page. The first click of a session pays all of it.
 * None of it depends on WHICH result is clicked except the PDF, and the result list
 * already knows the few documents a click is likely to land on.
 *
 * So this is a queue, and the three rules it exists for:
 *
 *  1. **One request at a time.** The point is to use the gap while someone reads,
 *     not to open six connections to a small VPS that is also serving other people.
 *     `docker/Caddyfile` rate-limits /pdf/* per IP for the same reason, and a
 *     prefetcher that tripped that limit would make the site slower, not faster.
 *  2. **A click beats the queue.** The moment the reader commits to a document,
 *     every speculative request is aborted, so the navigation and pdf.js's own range
 *     requests get the connection to themselves. A prefetch that is still running
 *     for the document being opened, or for the viewer's own module and worker, is
 *     kept: it is no longer speculative.
 *  3. **Speculation stays cheap.** A bounded number of documents, a byte budget, and
 *     nothing at all when the browser says the connection is metered or slow.
 *
 * Everything here is fetch-and-forget: the responses are read to completion and
 * dropped, and what is kept is the browser's HTTP cache. `docker/Caddyfile` serves
 * PDFs `no-cache`, which stores the body and forces a conditional request, so an
 * opened document costs a 304 instead of a download. That is the whole mechanism;
 * there is no other cache in this file.
 *
 * Written by Claude Code (Opus 5).
 */

/** Documents whose PDF may be warmed for one result list. */
export const PREFETCH_DOCS = 3;
/**
 * Bytes one result list may spend on speculation.
 *
 * The corpus median PDF is 0.31 MB and the 90th percentile is 2.5 MB, so this
 * usually covers the top three and stops inside one big argumentaire rather than
 * downloading 10 MB nobody asked for. Counted from Content-Length when the server
 * sends one, which Caddy's file server does.
 */
const PREFETCH_BUDGET = 6 * 1024 * 1024;

/**
 * The documents worth warming for one result list: the first PREFETCH_DOCS distinct
 * ones, in list order, skipping any `skip` rules out.
 *
 * Skipped documents are passed over rather than filtered afterwards, so they do not
 * take a slot from one that can actually be warmed.
 *
 * @param {{doc: number}[]} results - The list as painted.
 * @param {object[]} docs - `index.docs`, which `result.doc` indexes into.
 * @param {(doc: object) => boolean} [skip] - Documents with nothing to fetch.
 * @param {number} [limit]
 * @returns {object[]} At most `limit` document records.
 */
export function warmDocs(results, docs, skip = () => false, limit = PREFETCH_DOCS) {
  const picked = [];
  for (const result of results) {
    if (picked.length >= limit) break;
    const doc = docs[result.doc];
    if (!doc || picked.includes(doc) || skip(doc)) continue;
    picked.push(doc);
  }
  return picked;
}

/**
 * Is speculative fetching welcome on this connection?
 *
 * Save-Data is an explicit "do not spend my bytes", and 2g means the speculation
 * would compete with the request the reader is actually waiting for. Both are absent
 * on most desktop browsers, where the answer is yes.
 *
 * @param {object} [connection] - `navigator.connection`, injectable for tests.
 * @returns {boolean}
 */
export function prefetchAllowed(connection) {
  if (!connection) return true;
  if (connection.saveData) return false;
  return !/(^|-)2g$/.test(String(connection.effectiveType || ""));
}

/**
 * A single-flight queue for speculative GETs.
 *
 * @param {object} [options]
 * @param {Function} [options.fetcher] - `fetch`, injectable for tests.
 * @param {number} [options.budget] - Bytes this queue may spend before it stops.
 * @param {string[]} [options.keep] - URLs every click needs whatever document it
 *   opens (the viewer's own module and worker). A commit never aborts one of these
 *   in flight: the viewer page is about to request it anyway, and aborting a 2.2 MB
 *   worker half way only makes the reader wait for all of it again.
 * @returns {{enqueue: Function, promote: Function, commit: Function, reset: Function,
 *            done: Set<string>, spent: () => number, pending: () => number}}
 */
export function createPrefetcher({ fetcher = fetch, budget = PREFETCH_BUDGET, keep = [] } = {}) {
  /** URLs waiting, in the order they will be fetched. */
  let queue = [];
  /** URLs already fetched (or attempted) in this page's lifetime, never repeated. */
  const done = new Set();
  /** The one in flight, so a commit can tell it apart from the ones to abort. */
  let current = null;
  let controller = null;
  let spent = 0;
  let running = false;
  /* Set by commit(): the reader has clicked, so nothing further is speculated. The
     page is navigating away; anything this queue starts now competes with it. */
  let committed = false;

  async function pump() {
    if (running) return;
    running = true;
    try {
      while (queue.length && !committed && spent < budget) {
        const url = queue.shift();
        if (done.has(url)) continue;
        done.add(url);
        current = url;
        controller = new AbortController();
        try {
          // `priority: "low"` is a hint browsers that know it act on and others
          // ignore; the queue of one is what actually keeps this polite.
          const response = await fetcher(url, { signal: controller.signal, priority: "low" });
          const length = Number(response?.headers?.get?.("content-length") || 0);
          // Read the body to completion even though it is thrown away: an unread
          // body can leave the response out of the HTTP cache, which would make the
          // whole exercise pointless.
          if (response?.arrayBuffer) spent += (await response.arrayBuffer()).byteLength;
          else spent += length;
        } catch {
          /* Aborted, offline, 404 on a document that moved: speculation is allowed
             to fail silently. The click path fetches the same URL again and reports
             what it finds, which is where an error belongs. */
        } finally {
          current = null;
          controller = null;
        }
      }
    } finally {
      running = false;
    }
  }

  return {
    done,
    spent: () => spent,
    pending: () => queue.length,

    /**
     * Add URLs to the end of the queue, skipping anything already fetched.
     * @param {string[]} urls
     */
    enqueue(urls) {
      if (committed) return;
      for (const url of urls) if (url && !done.has(url) && !queue.includes(url)) queue.push(url);
      void pump();
    },

    /**
     * Move one URL to the front: the reader is hovering, or has focused, the link
     * that leads to it, which is the strongest hint short of a click.
     * @param {string} url
     */
    promote(url) {
      if (committed || !url || done.has(url)) return;
      queue = [url, ...queue.filter((u) => u !== url)];
      void pump();
    },

    /**
     * The reader has committed to one document: stop speculating.
     *
     * Everything queued is dropped and anything in flight is aborted, UNLESS it is
     * the very document being opened or one of the `keep` URLs every viewer needs:
     * those are now requests the reader is waiting for rather than guesses about them.
     *
     * @param {string} [url] - The document's URL, if the commitment names one.
     */
    commit(url) {
      committed = true;
      queue = [];
      if (current && current !== url && !keep.includes(current)) controller?.abort();
    },

    /** A new question: speculate again, about the new answer. */
    reset() {
      committed = false;
      queue = [];
      spent = 0;
      if (current) controller?.abort();
    },
  };
}
