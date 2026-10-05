/* The search service: src/search.js behind one HTTP endpoint.
 *
 * The browser used to download the whole index (20.6 MB) and rank in JS. This
 * process holds the index instead and answers `POST /api/search` with the folded
 * result list; the page keeps `parseQuery`, the rendering and the viewer. Nothing
 * about ranking is decided here: `parseQuery`, `embedParsedQuery`, `rank`,
 * `rescore` and `foldResults` are the shipped, measured functions, called in the
 * order src/app.js called them.
 *
 * Shape of a request and its answer:
 *
 *   POST /api/search  { "q": "...", "filters": {...}, "bm25": false,
 *                        "demote_references": true,
 *                        "favour_recent": true,
 *                        "figures": "include" }
 *   -> { "results": [ { "doc", "chunk", "page", "pages", "cosine", "score",
 *                       "lexical", "more", "alternates", "text" } ],
 *        "floor": 0.28, "dims": 1024 }
 *
 *   GET /api/search/health -> { "ok": true, "dims", "n_chunks", "waiting", "rss_mb",
 *                                "ranker_sha", "index_sha" }
 *
 * `ranker_sha` fingerprints the src/search.js THIS PROCESS imported and `index_sha`
 * the index it LOADED, so a caller can tell a service running the current ranker
 * and index from one started before the last edit or rebuild. `node scripts/check_search.mjs` with SVC set does exactly that.
 *
 * Refusals carry the same error names the page already translates ("short",
 * "terms", "long", "empty", "dim", "unavailable"), plus "full" (503, Retry-After)
 * when depth + 1 searches are already in flight and "bad_request" for a body that is not a search.
 *
 * Privacy: the question text is never logged and never written to disk. The
 * query-vector cache in src/search.js lives in this process's memory only.
 *
 * Environment (all documented in docker/env.example):
 *   INDEX_DIR           where meta.json and the matrices are (default /index)
 *   EMBED_URL           the shared encoder (default http://justelesrcp-embed:8461)
 *   SEARCH_PORT         listen port inside the container (default 8650)
 *   SEARCH_DIM          refuse to start unless the index has this width
 *   SEARCH_FLOOR        cosine below which a result is not returned (default 0)
 *   SEARCH_QUEUE_DEPTH  waiting searches before a 503 (default 24)
 *   SEARCH_TEXT_CACHE   per-document text files kept in memory (default 64)
 *
 * Written by Claude Code (Fable 5.1).
 */

import { createServer } from "node:http";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { loadIndexFromDisk, rssMb, RANKER_SHA } from "./load.mjs";
import { createQueue } from "./queue.mjs";

/** Body cap; Caddy applies the same at the edge. */
const MAX_BODY_BYTES = 16 * 1024;
/**
 * Filter fields a request may name: the index's facets, its tier column (the level
 * control, which arrives as a plain string like any other single-valued filter;
 * `allowedDocs` is what knows the tiers are nested, and a level this index cannot
 * place filters nothing, so the bound on the string is all the validation it needs
 * here), plus the page's one scope field. Read from meta.json rather than listed,
 * so the corpus declares its facets once and this service cannot drift from it.
 *
 * @param {object} search - The src/search.js module, which reads those keys.
 * @param {object} index
 * @returns {Set<string>}
 */
export function filterFields(search, index) {
  return new Set([...search.facetFields(index), search.tierField(index), "file"]);
}
const MAX_FILTER_VALUES = 64;
const MAX_FILTER_CHARS = 256;
/** Encoder round trip, after which the request is answered "unavailable". */
const EMBED_TIMEOUT_MS = 10_000;

/**
 * Refuse a width the deployment did not ask for.
 *
 * `SEARCH_DIM` used to let the page refuse to search on a mismatch; here the
 * service refuses to START, which is earlier and louder.
 *
 * @param {{dims: number}} meta
 * @param {string|number|undefined} wanted - SEARCH_DIM as read from the environment.
 */
export function checkWidth(meta, wanted) {
  if (wanted === undefined || wanted === "") return;
  const n = Number(wanted);
  if (!Number.isInteger(n) || n <= 0) throw new Error(`SEARCH_DIM=${wanted} is not a width`);
  if (n !== meta.dims) throw new Error(`index is ${meta.dims} dims, SEARCH_DIM asks for ${n}`);
}

/**
 * Read the service's settings from the environment, refusing a value that is not one.
 *
 * `Number("24 ")` is 24 but `Number("2 4")` is NaN, and NaN passed everything
 * downstream silently: a NaN queue depth makes `waiting.length >= depth` false
 * forever, so the bound that keeps the container answering under a burst was
 * gone with no message. A typo in docker/.env now stops the service at start,
 * which is earlier and louder, the same choice as `checkWidth`.
 *
 * @param {Record<string, string|undefined>} env
 * @returns {{indexDir: string, embedUrl: string, port: number, floor: number,
 *            depth: number, textCache: number}}
 */
export function readConfig(env) {
  const integer = (name, fallback, min, max = Number.MAX_SAFE_INTEGER) => {
    const raw = env[name];
    if (raw === undefined || raw.trim() === "") return fallback;
    const n = Number(raw);
    if (!Number.isInteger(n) || n < min || n > max) {
      throw new Error(`${name}=${raw} is not an integer from ${min} to ${max}`);
    }
    return n;
  };
  const rawFloor = env.SEARCH_FLOOR;
  let floor = 0;
  if (rawFloor !== undefined && rawFloor.trim() !== "") {
    floor = Number(rawFloor);
    // A cosine: anything at 1 or above would hide every result there is.
    if (!Number.isFinite(floor) || floor < 0 || floor >= 1) {
      throw new Error(`SEARCH_FLOOR=${rawFloor} is not a cosine from 0 up to 1`);
    }
  }
  return {
    indexDir: env.INDEX_DIR || "/index",
    embedUrl: (env.EMBED_URL || "http://justelesrcp-embed:8461").replace(/\/$/, ""),
    port: integer("SEARCH_PORT", 8650, 1, 65535),
    floor,
    depth: integer("SEARCH_QUEUE_DEPTH", 24, 1),
    // 0 is legitimate: it was measured (DESIGN.md), at 120 ms per answer.
    textCache: integer("SEARCH_TEXT_CACHE", 64, 0),
  };
}

/**
 * Keep the per-document text cache to `max` documents, oldest out.
 *
 * `attachText` in src/search.js caches every document it reads for the life of
 * the index; in a browser tab that is a session, here it would be the whole
 * corpus's text resident within a day. A Map iterates in insertion order, so the
 * first key is the oldest; a document re-read later is re-inserted at the end.
 *
 * @param {{textCache: Map}} index
 * @param {number} max
 */
export function boundTextCache(index, max) {
  const cache = index.textCache;
  while (cache.size > max) cache.delete(cache.keys().next().value);
}

/**
 * Validate the filters a client sent, keeping only what `allowedDocs` understands.
 *
 * @param {unknown} raw
 * @param {Set<string>} allowed - The field names a request may filter on: `filterFields(index)`.
 * @returns {object} field -> string[] | {from, to}; throws "bad_request" otherwise.
 */
export function cleanFilters(raw, allowed) {
  if (raw == null) return {};
  if (typeof raw !== "object" || Array.isArray(raw)) throw new Error("bad_request");
  const out = {};
  for (const [field, value] of Object.entries(raw)) {
    if (!allowed.has(field)) throw new Error("bad_request");
    if (typeof value === "string") {
      if (value.length > MAX_FILTER_CHARS) throw new Error("bad_request");
      if (value) out[field] = value;
    } else if (Array.isArray(value)) {
      if (value.length > MAX_FILTER_VALUES) throw new Error("bad_request");
      if (!value.every((v) => typeof v === "string" && v.length <= MAX_FILTER_CHARS)) throw new Error("bad_request");
      if (value.length) out[field] = value;
    } else if (value && typeof value === "object") {
      const range = {};
      for (const end of ["from", "to"]) {
        if (value[end] == null) continue;
        if (!Number.isFinite(Number(value[end]))) throw new Error("bad_request");
        range[end] = Number(value[end]);
      }
      if (range.from != null || range.to != null) out[field] = range;
    } else if (value != null) {
      throw new Error("bad_request");
    }
  }
  return out;
}

/** The fields of a folded result the page renders; nothing else leaves the process. */
function publicResult(r) {
  return {
    doc: r.doc, chunk: r.chunk, page: r.page, pages: r.pages ?? null,
    cosine: r.cosine, score: r.score, lexical: r.lexical ?? null,
    more: r.more, alternates: r.alternates, text: r.text ?? null,
    figure: r.figure ?? null,
  };
}

/**
 * The search itself, HTTP left out so the tests and the gate can call it.
 *
 * @param {object} options
 * @param {object} options.search - The src/search.js module.
 * @param {object} options.index - What `loadIndexFromDisk` returned.
 * @param {number} [options.floor] - SEARCH_FLOOR.
 * @param {number} [options.depth] - Queue depth.
 * @param {number} [options.textCache] - Documents kept in the text cache.
 */
export function createSearchService({ search, index, floor = 0, depth = 24, textCache = 64 }) {
  const queue = createQueue({ depth });
  const dims = index.meta.dims;
  const allowedFields = filterFields(search, index);
  // The queue bounds only the ranking. The encoder round trip before it and the
  // text reads after it are awaits that any number of requests could sit in at
  // once, each holding a socket, a pending fetch to the shared encoder and,
  // after ranking, up to a shortlist of parsed documents. So the WHOLE answer
  // is bounded too, by the same number the queue promises: one being ranked
  // plus `depth` waiting. A request past that is refused "full" before it
  // costs anything, exactly as one refused by the queue would be.
  const maxInFlight = depth + 1;
  let inFlight = 0;

  /**
   * @param {{q: unknown, filters?: unknown, bm25?: unknown,
   *   demote_references?: unknown, favour_recent?: unknown, figures?: unknown}} body
   * @param {{signal?: AbortSignal}} [options]
   */
  async function answer(body, { signal } = {}) {
    if (inFlight >= maxInFlight) throw new Error("full");
    inFlight += 1;
    try {
      return await answerOne(body, { signal });
    } finally {
      inFlight -= 1;
    }
  }

  async function answerOne(body, { signal }) {
    if (!body || typeof body !== "object" || typeof body.q !== "string") throw new Error("bad_request");
    const filters = cleanFilters(body.filters, allowedFields);
    const bm25 = body.bm25 === true;
    // Defaults ON, so a client that predates the flag (or one that simply omits it)
    // gets the demotion rather than silently losing it. Every other option here
    // defaults off, and this one is the exception on purpose: it is the behaviour
    // the page ships with its box ticked, and the two have to agree.
    const demoteReferences = body.demote_references !== false;
    // Same rule, same reason: the recency bonus is the page's default too, so an
    // omitted flag has to mean on. A client that wants the raw embedding order
    // says so explicitly, with both of these false.
    const favourRecent = body.favour_recent !== false;
    // Described figures compete with the text by default, like the page's own
    // default. "exclude" drops them and "only" keeps nothing else; anything but
    // those three is a client that is not the page.
    const figures = body.figures ?? search.FIGURE_MODES[0];
    if (!search.FIGURE_MODES.includes(figures)) throw new Error("bad_request");
    // The page parsed the same text with the same function before sending it, so a
    // refusal here means a client that is not the page. Same error names either way.
    const parsed = search.parseQuery(body.q);
    // The encoder round trip is network I/O and overlaps freely; only ranking is
    // serialised. A client gone by the time the vector is back is not ranked for.
    const vector = await search.embedParsedQuery(parsed, dims);
    const candidates = await queue.run(
      () => search.rank(index, vector, filters, { demoteReferences, favourRecent, figures }), { signal });
    // Copies: `rescore` writes onto what it is given, and candidates are not kept.
    const shortlist = candidates.slice(0, search.RESCORE_DEPTH).map((c) => ({ ...c }));
    if (bm25) {
      await search.attachText(index, shortlist);
      search.rescore(shortlist, parsed.lexical);
    }
    // Folded here, not on the page: folding drops redundant renditions, and doing
    // it after the list was cut to size would silently shorten it.
    const results = search.foldResults(index, shortlist.concat(candidates.slice(search.RESCORE_DEPTH)), floor);
    await search.attachText(index, results);
    boundTextCache(index, textCache);
    return { results: results.map(publicResult), floor, dims };
  }

  return { answer, queue, dims, inFlight: () => inFlight };
}

/** HTTP status and body for one of the errors `answer` throws. */
export function describeError(error) {
  const name = error?.message || "error";
  switch (name) {
    case "bad_request": return [400, { error: name }];
    case "empty": case "short": case "terms": case "long":
      return [400, { error: name, ...(error.min != null && { min: error.min }),
        ...(error.got != null && { got: error.got, max: error.max }) }];
    case "dim": return [500, { error: name, got: error.got, want: error.want }];
    case "unavailable": return [502, { error: name }];
    case "full": return [503, { error: name }];
    // The reader's own doing: what they subtracted cancels the whole question. A
    // 400 with the same name the page's own check would have raised, so it reads
    // the same sentence. NOT 499, which the handler answers with silence.
    case "cancelled": return [400, { error: name }];
    // The client hung up before its turn in the queue: nobody is listening.
    case "gone": return [499, { error: name }];
    default:
      // An encoder that timed out, refused, or answered something else: the same
      // thing to the reader, a search that cannot run right now.
      if (/^embed |timeout|abort|fetch failed/i.test(name) || error?.name === "TimeoutError") {
        return [502, { error: "unavailable" }];
      }
      return [500, { error: "error" }];
  }
}

/** Read a JSON body of at most MAX_BODY_BYTES, or throw "bad_request". */
function readJson(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on("data", (chunk) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        reject(new Error("bad_request"));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => {
      try {
        resolve(JSON.parse(Buffer.concat(chunks).toString("utf-8")));
      } catch {
        reject(new Error("bad_request"));
      }
    });
    req.on("error", () => reject(new Error("bad_request")));
  });
}

/**
 * The request handler, for `createServer` or a test that calls it with fakes.
 *
 * @param {ReturnType<typeof createSearchService>} service
 * @param {{index: object, indexSha?: string, log?: (line: string) => void}} options
 *   `indexSha` is `indexFingerprint` of the meta.json this process loaded.
 */
export function createHandler(service, { index, indexSha = "", log = () => {} }) {
  const send = (res, status, payload, headers = {}) => {
    if (res.writableEnded) return;
    res.writeHead(status, { "Content-Type": "application/json; charset=utf-8", ...headers });
    res.end(JSON.stringify(payload));
  };

  return async (req, res) => {
    const url = req.url.split("?")[0];
    if (req.method === "GET" && url === "/api/search/health") {
      send(res, 200, { ok: true, dims: service.dims, n_chunks: index.meta.n_chunks,
        waiting: service.queue.waiting(), rss_mb: rssMb(), ranker_sha: RANKER_SHA,
        index_sha: indexSha });
      return;
    }
    if (url !== "/api/search") {
      send(res, 404, { error: "not_found" });
      return;
    }
    if (req.method !== "POST") {
      send(res, 405, { error: "method" }, { Allow: "POST" });
      return;
    }
    // Gone before its turn: the queue drops it, nothing is ranked for nobody.
    const controller = new AbortController();
    res.on("close", () => { if (!res.writableFinished) controller.abort(); });
    const started = performance.now();
    try {
      const body = await readJson(req);
      const answer = await service.answer(body, { signal: controller.signal });
      send(res, 200, answer, { "Cache-Control": "no-store" });
      log(`search 200 ${answer.results.length} results ${(performance.now() - started).toFixed(0)} ms`);
    } catch (error) {
      const [status, payload] = describeError(error);
      if (status === 499) return; // the client hung up; nobody is listening
      send(res, status, payload, status === 503 ? { "Retry-After": "2" } : {});
      log(`search ${status} ${payload.error} ${(performance.now() - started).toFixed(0)} ms`);
    }
  };
}

async function main() {
  const env = process.env;
  const { indexDir, embedUrl, port, floor, depth, textCache } = readConfig(env);
  const log = (line) => console.log(`${new Date().toISOString()} ${line}`);

  const { search, index, indexSha, loadMs, reachable } = await loadIndexFromDisk({ indexDir, embedUrl, apiTimeoutMs: EMBED_TIMEOUT_MS });
  checkWidth(index.meta, env.SEARCH_DIM);
  log(`index ${indexDir}: ${index.meta.n_documents} documents, ${index.meta.n_chunks} chunks at ${index.meta.dims} dims, loaded in ${loadMs.toFixed(0)} ms; rss ${rssMb()} MB`);
  log(`encoder ${embedUrl}: ${await reachable(embedUrl) ? "answering" : "NOT answering (searches will fail with 502 until it does)"}`);

  const service = createSearchService({ search, index, floor, depth, textCache });
  const server = createServer(createHandler(service, { index, indexSha, log }));
  server.keepAliveTimeout = 65_000;
  server.listen(port, "0.0.0.0", () => log(`listening on :${port}, floor ${floor}, queue depth ${depth}, text cache ${textCache} documents`));

  const stop = () => { log("stopping"); server.close(() => process.exit(0)); setTimeout(() => process.exit(0), 2000).unref(); };
  process.on("SIGTERM", stop);
  process.on("SIGINT", stop);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
