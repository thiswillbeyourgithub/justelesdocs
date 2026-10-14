/* Server-side page views for the routes an AI assistant reads, sent to umami.
 *
 * The site's analytics are a page view the BROWSER sends (src/site.js). An
 * assistant fetching /llms.txt, /api/search.md or /api/doc/ runs no script, so
 * without this nobody would know whether those routes are used at all. The search
 * service sends one page view per such request instead: `/api/search.md` itself,
 * and the static ones through Caddy's `forward_auth`, which asks `/api/hit` before
 * serving the file (docker/Caddyfile).
 *
 * What is sent is the PATH and nothing else of the request line: never the query
 * string, which holds the question. Plus the request's User-Agent, and the
 * assistant's name as umami's `browser`, so umami's Browser breakdown reads
 * "Claude-User", "ChatGPT-User"... instead of "unknown". No client IP is forwarded:
 * for these requests it is an AI provider's data centre, which says nothing.
 * umami drops bot user agents unless its instance sets DISABLE_BOT_CHECK.
 *
 * Configured by the same two variables as the browser's analytics, ANALYTICS_URL
 * (the tracker script, whose origin also answers /api/send) and
 * ANALYTICS_WEBSITE_ID. Either unset: nothing is sent. A failed send is logged and
 * never retried, and never delays or fails the answer it is about.
 *
 * Written by Claude Code (Opus 5.5).
 */

/* The network fetch, taken when this module loads. server/load.mjs then replaces
   globalThis.fetch with a shim that reads the index from disk and refuses every
   other URL, so a fetch looked up at send time would never reach umami. */
const networkFetch = globalThis.fetch;

/* Requests in flight to umami at once, beyond which a page view is dropped: a
   crawler hammering /api/doc/ must not pile up sockets in a 256 MB container. */
const MAX_IN_FLIGHT = 8;
const SEND_TIMEOUT_MS = 3000;

/* User agents worth naming, first match wins. A browser matches none and is left
   to umami's own parser. */
const AGENTS = [
  "Claude-User", "Claude-SearchBot", "ClaudeBot", "ChatGPT-User", "OAI-SearchBot", "GPTBot",
  "Perplexity-User", "PerplexityBot", "MistralAI-User", "Gemini", "Google-Extended", "Googlebot",
  "bingbot", "Applebot", "Amazonbot", "meta-externalagent", "Bytespider", "CCBot", "DuckAssistBot",
  "curl", "Wget", "python-requests", "python-httpx", "aiohttp", "Go-http-client", "node-fetch", "undici",
];
const AGENT_RE = new RegExp(AGENTS.map((a) => a.replace(/[.-]/g, "\\$&")).join("|"), "i");

/**
 * The name umami shows as the browser, or undefined for an ordinary browser.
 *
 * @param {string|undefined} userAgent
 */
export function agentOf(userAgent) {
  const found = String(userAgent || "").match(AGENT_RE);
  return found ? AGENTS.find((a) => a.toLowerCase() === found[0].toLowerCase()) : undefined;
}

/**
 * The /api/send body for one page view.
 *
 * @param {object} o
 * @param {string} o.websiteId
 * @param {string} o.host - The request's Host, already checked (markdown.originOf).
 * @param {string} o.path - The path alone; a query string is cut off here too.
 * @param {string|undefined} o.userAgent
 */
export function pageview({ websiteId, host, path, userAgent }) {
  const payload = { website: websiteId, hostname: host.replace(/:\d+$/, ""), url: path.split("?")[0],
    userAgent: String(userAgent || "unknown").slice(0, 500) };
  const agent = agentOf(userAgent);
  if (agent) payload.browser = agent;
  return { type: "event", payload };
}

/**
 * A page-view sender, or a no-op when analytics are not configured.
 *
 * @param {Record<string, string|undefined>} env
 * @param {{fetch?: typeof fetch, log?: (line: string) => void}} [deps]
 * @returns {{enabled: boolean, track: (req: import("node:http").IncomingMessage, path: string) => void}}
 */
export function createTracker(env, { fetch: send = networkFetch, log = () => {} } = {}) {
  const websiteId = (env.ANALYTICS_WEBSITE_ID || "").trim();
  let endpoint = "";
  try {
    endpoint = new URL("/api/send", env.ANALYTICS_URL).href;
  } catch {
    // unset or not a URL: analytics off
  }
  if (!endpoint || !websiteId) return { enabled: false, track: () => {} };
  let inFlight = 0;
  return {
    enabled: true,
    track(req, path) {
      const host = req.headers.host || "";
      // Same rule as the browser's tracker (ANALYTICS_DNT), for the rare person
      // who opens these routes by hand.
      if (req.headers.dnt === "1" || !/^[a-z0-9.-]+(:\d{1,5})?$/i.test(host) || inFlight >= MAX_IN_FLIGHT) return;
      inFlight += 1;
      send(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json", "User-Agent": String(req.headers["user-agent"] || "unknown").slice(0, 500) },
        body: JSON.stringify(pageview({ websiteId, host, path, userAgent: req.headers["user-agent"] })),
        signal: AbortSignal.timeout(SEND_TIMEOUT_MS),
      })
        .then((res) => { if (!res.ok) log(`umami ${res.status} for ${path.split("?")[0]}`); })
        .catch((error) => log(`umami failed: ${error.message}`))
        .finally(() => { inFlight -= 1; });
    },
  };
}
