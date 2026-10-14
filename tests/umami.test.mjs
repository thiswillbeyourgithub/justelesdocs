/* Server-side umami page views for the AI routes (server/umami.mjs), and the two
 * places the search service calls them: /api/search.md and Caddy's forward_auth
 * hook, /api/hit. What matters most is what is NOT sent: the question.
 *
 * Written by Claude Code (Opus 5.5).
 */

import { test, after } from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const { agentOf, pageview, createTracker } = await import(`${root}/server/umami.mjs`);
const { createHandler } = await import(`${root}/server/service.mjs`);

const ENV = { ANALYTICS_URL: "https://u.example.org/script.js", ANALYTICS_WEBSITE_ID: "00000000-0000-0000-0000-000000000001" };
const req = (headers) => ({ headers: { host: "docs.example.org", ...headers } });

/** A tracker whose fetch records what it was asked to send. */
function recording(env = ENV) {
  const sent = [];
  const fetch = (url, init) => { sent.push({ url, init, body: JSON.parse(init.body) }); return Promise.resolve({ ok: true }); };
  return { sent, tracker: createTracker(env, { fetch }) };
}

test("assistants are named, browsers are left to umami", () => {
  assert.equal(agentOf("Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko); compatible; Claude-User/1.0; +Claude-User@anthropic.com)"), "Claude-User");
  assert.equal(agentOf("Mozilla/5.0 (compatible; ChatGPT-User/1.0; +https://openai.com/bot)"), "ChatGPT-User");
  assert.equal(agentOf("curl/8.5.0"), "curl");
  assert.equal(agentOf("Mozilla/5.0 (X11; Linux x86_64) Gecko/20100101 Firefox/131.0"), undefined);
});

test("a page view carries the path without its query, and the assistant as browser", () => {
  const { payload } = pageview({ websiteId: "w", host: "docs.example.org:443", path: "/api/search.md?q=secret question", userAgent: "ChatGPT-User/1.0" });
  assert.deepEqual(payload, { website: "w", hostname: "docs.example.org", url: "/api/search.md", userAgent: "ChatGPT-User/1.0", browser: "ChatGPT-User" });
});

test("sent to the tracker's own origin, and nothing at all when unconfigured", () => {
  const { sent, tracker } = recording();
  tracker.track(req({ "user-agent": "Claude-User" }), "/llms.txt");
  assert.equal(sent[0].url, "https://u.example.org/api/send");
  assert.equal(sent[0].init.headers["User-Agent"], "Claude-User");
  for (const env of [{}, { ANALYTICS_URL: ENV.ANALYTICS_URL }, { ANALYTICS_WEBSITE_ID: "x", ANALYTICS_URL: "not a url" }]) {
    const off = recording(env);
    assert.equal(off.tracker.enabled, false);
    off.tracker.track(req({}), "/llms.txt");
    assert.equal(off.sent.length, 0);
  }
});

test("DNT, a forged Host and a backlog of sends are skipped", () => {
  const { sent, tracker } = recording();
  tracker.track(req({ dnt: "1" }), "/llms.txt");
  tracker.track(req({ host: "evil host/" }), "/llms.txt");
  assert.equal(sent.length, 0);
  const stuck = [];
  const slow = createTracker(ENV, { fetch: () => new Promise((done) => stuck.push(done)) });
  for (let i = 0; i < 20; i++) slow.track(req({}), "/llms.txt");
  assert.equal(stuck.length, 8);
});

test("the service counts /api/search.md and Caddy's /api/hit, never the question", async () => {
  const tracked = [];
  const service = { answerMarkdown: async () => "# ok\n" };
  const server = createServer(createHandler(service, { index: { meta: {} }, track: (_req, path) => tracked.push(path) }));
  await new Promise((done) => server.listen(0, "127.0.0.1", done));
  after(() => server.close());
  const base = `http://127.0.0.1:${server.address().port}`;
  await (await fetch(`${base}/api/search.md?q=lithium%20grossesse`)).text();
  const hit = (uri) => fetch(`${base}/api/hit`, { headers: { "X-Forwarded-Uri": uri } });
  assert.equal((await hit("/api/doc/some-doc/p/3")).status, 204);
  assert.equal((await hit("/llms.txt?x=1")).status, 204);
  assert.equal((await hit("/api/search")).status, 204, "answered, but not counted");
  assert.equal((await hit("/llms.txt.bak")).status, 204);
  assert.deepEqual(tracked, ["/api/search.md", "/api/doc/some-doc/p/3", "/llms.txt"]);
});

test("sends over the network even after the index loader replaced globalThis.fetch", async () => {
  const bodies = [];
  const umami = createServer((r, res) => { let b = ""; r.on("data", (c) => { b += c; }); r.on("end", () => { bodies.push(JSON.parse(b)); res.end("{}"); }); });
  await new Promise((done) => umami.listen(0, "127.0.0.1", done));
  const saved = globalThis.fetch;
  globalThis.fetch = () => Promise.reject(new Error("only the index may be read here"));
  const failures = [];
  try {
    const tracker = createTracker({ ...ENV, ANALYTICS_URL: `http://127.0.0.1:${umami.address().port}/script.js` },
      { log: (line) => failures.push(line) });
    tracker.track(req({ "user-agent": "Claude-User" }), "/llms.txt");
    for (let i = 0; i < 50 && !bodies.length && !failures.length; i++) await new Promise((done) => setTimeout(done, 20));
  } finally {
    globalThis.fetch = saved;
    umami.close();
  }
  assert.deepEqual(failures, []);
  assert.equal(bodies[0].payload.url, "/llms.txt");
});
