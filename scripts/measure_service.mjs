/* Load and memory measurements of the search service, for DESIGN.md.
 *
 * Not a gate: nothing here passes or fails, it prints numbers. The numbers in
 * DESIGN.md ("Measured: the ranking on the server") came from these four modes,
 * run on the development machine; the VPS numbers are still owed and come from
 * running the same thing over an ssh forward of the service's port.
 *
 *   SVC=http://127.0.0.1:8650 node scripts/measure_service.mjs latency    # sequential, warm and cold
 *   SVC=http://127.0.0.1:8650 node scripts/measure_service.mjs burst 50   # 50 at once, same and distinct questions
 *   SVC=http://127.0.0.1:8650 node scripts/measure_service.mjs rss        # 100 distinct bm25 questions, RSS before and after
 *   PW=<playwright> SITE=http://127.0.0.1:8649 node scripts/measure_service.mjs visit
 *                                                                          # bytes and ms until the search box works
 *
 * SVC is the service itself (or dev_server.py, which proxies it); `visit` needs a
 * served site and a playwright install, like check_ui.mjs.
 *
 * Written by Claude Code (Fable 5.1).
 */
import { createRequire } from "node:module";
import { gzipSync } from "node:zlib";
import { loadScenarios } from "./lib/scenarios.mjs";

const SVC = process.env.SVC || "http://127.0.0.1:8650";
const SITE = process.env.SITE || "http://127.0.0.1:8649";
const mode = process.argv[2];

// The questions come from the corpus (scenarios.json measure.*, see
// scripts/lib/scenarios.mjs), since only a corpus knows what its readers ask; the
// neutral defaults below still exercise the service, they just find less.
const SCENARIOS = loadScenarios().measure || {};
const BASE = SCENARIOS.queries || [
  "qualité de l'air en ville et santé respiratoire",
  "consommation d'énergie des logements anciens",
  "prix moyen du loyer par région",
  "accès à l'eau potable en zone rurale",
  "émissions de gaz à effet de serre des transports",
  "household energy use by income group",
  "taux de chômage des jeunes diplômés",
  "rénovation thermique et aides publiques",
  "fréquentation des transports en commun",
  "gestion des déchets ménagers et recyclage",
];
const TOPICS = SCENARIOS.topics || ["énergie", "transport", "logement", "eau", "air", "déchets", "emploi", "revenu", "éducation", "santé", "climat", "agriculture", "industrie", "tourisme", "population", "commerce", "numérique", "sécurité", "culture", "territoire"];
const ASPECTS = SCENARIOS.aspects || ["mesure", "évolution", "comparaison", "prévision", "financement"];
const distinct = (n) => Array.from({ length: n }, (_, i) => `${ASPECTS[i % ASPECTS.length]} ${TOPICS[Math.floor(i / ASPECTS.length) % TOPICS.length]} ${Math.floor(i / (ASPECTS.length * TOPICS.length)) || ""}`.trim());

const post = async (q, bm25 = false) => {
  const t0 = performance.now();
  let status = 0, n = -1;
  try {
    const r = await fetch(`${SVC}/api/search`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ q, filters: {}, bm25 }) });
    status = r.status;
    const body = await r.json().catch(() => ({}));
    n = body.results ? body.results.length : -1;
  } catch { status = -1; }
  return { status, n, ms: performance.now() - t0 };
};
const health = async () => (await fetch(`${SVC}/api/search/health`)).json();
const pct = (xs, p) => { const s = [...xs].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
const summary = (rs) => {
  const by = {};
  for (const r of rs) by[r.status] = (by[r.status] || 0) + 1;
  const ok = rs.filter((r) => r.status === 200).map((r) => r.ms);
  const all = rs.map((r) => r.ms);
  return { statuses: by, ok_ms: ok.length ? { min: pct(ok, 0).toFixed(0), p50: pct(ok, 0.5).toFixed(0), p95: pct(ok, 0.95).toFixed(0), max: pct(ok, 1).toFixed(0) } : null,
           all_ms: { p50: pct(all, 0.5).toFixed(0), max: pct(all, 1).toFixed(0) } };
};

if (mode === "visit") {
  const require = createRequire(import.meta.url);
  const { chromium } = require(process.env.PW);
  const browser = await chromium.launch();
  const page = await browser.newPage();
  const rows = [];
  page.on("response", async (r) => {
    try { const b = await r.body(); rows.push({ url: r.url().replace(SITE, ""), status: r.status(), bytes: b.length, gz: gzipSync(b).length, type: r.headers()["content-type"] || "" }); } catch {}
  });
  const t0 = performance.now();
  await page.goto(`${SITE}/index.html`, { waitUntil: "load" });
  await page.waitForFunction(() => { const q = document.querySelector("#q"); return q && !q.disabled; }, { timeout: 30000 });
  const tBox = performance.now() - t0;
  await page.waitForLoadState("networkidle");
  const tIdle = performance.now() - t0;
  const atBox = rows.length;
  await page.fill("#q", BASE[0]); await page.press("#q", "Enter");
  const t1 = performance.now();
  await page.waitForFunction(() => document.querySelectorAll("#results > li").length > 0, { timeout: 30000 });
  const tFirst = performance.now() - t1;
  await browser.close();
  const tot = (rs) => rs.reduce((a, r) => a + r.bytes, 0), totgz = (rs) => rs.reduce((a, r) => a + r.gz, 0);
  console.log("first visit, response by response (bytes raw / gzip):");
  for (const r of rows.slice(0, atBox)) console.log(`  ${r.status} ${String(r.bytes).padStart(8)} ${String(r.gz).padStart(8)}  ${r.url}`);
  console.log(`until the search box is usable: ${atBox} responses, ${tot(rows.slice(0, atBox))} B raw, ${totgz(rows.slice(0, atBox))} B gzip, ${tBox.toFixed(0)} ms`);
  console.log(`until network idle: ${rows.length} responses, ${tot(rows)} B raw, ${totgz(rows)} B gzip, ${tIdle.toFixed(0)} ms`);
  console.log(`enter to first result row painted: ${tFirst.toFixed(0)} ms (question not yet in the service's vector cache)`);
}

if (mode === "latency") {
  console.log("warm-up (each question once, fills the vector cache):");
  for (const q of BASE) await post(q);
  const cached = [];
  for (let i = 0; i < 3; i++) for (const q of BASE) cached.push(await post(q));
  console.log("cached vector (queue + rank + fold + text), 30 sequential:", JSON.stringify(summary(cached)));
  const cold = [];
  for (const q of distinct(30).map((s) => `${s} cold`)) cold.push(await post(q));
  console.log("cold question (encoder round trip included), 30 sequential:", JSON.stringify(summary(cold)));
  const lex = [];
  for (let i = 0; i < 2; i++) for (const q of BASE) lex.push(await post(q, true));
  console.log("cached vector with bm25 rescoring, 20 sequential:", JSON.stringify(summary(lex)));
  console.log("health:", JSON.stringify(await health()));
}

if (mode === "burst") {
  const n = Number(process.argv[3] || 50);
  console.log(`burst A: ${n} concurrent, the SAME question (vector cached: pure queue pressure)`);
  await post(BASE[0]);
  let hp = null;
  const t0 = performance.now();
  const a = await Promise.all(Array.from({ length: n }, (_, i) => (i === 10 ? (async () => { const t = performance.now(); const h = await health(); hp = { ms: (performance.now() - t).toFixed(0), waiting: h.waiting }; return post(BASE[0]); })() : post(BASE[0]))));
  console.log("  ", JSON.stringify(summary(a)), `wall ${(performance.now() - t0).toFixed(0)} ms`, "health mid-burst:", JSON.stringify(hp));
  console.log(`burst B: ${n} concurrent, ${n} DISTINCT questions (each one an encoder call first)`);
  const qs = distinct(n).map((s) => `${s} burst${Date.now() % 1000}`);
  const t1 = performance.now();
  const b = await Promise.all(qs.map((q) => post(q)));
  console.log("  ", JSON.stringify(summary(b)), `wall ${(performance.now() - t1).toFixed(0)} ms`);
  console.log("after:", JSON.stringify(await health()));
  const after = await post(BASE[1]);
  console.log("one more request after the burst:", JSON.stringify(after));
}

if (mode === "rss") {
  console.log("before:", JSON.stringify(await health()));
  const qs = distinct(100);
  const rs = [];
  for (const q of qs) rs.push(await post(q, true));
  console.log("100 distinct bm25 questions:", JSON.stringify(summary(rs)));
  console.log("after:", JSON.stringify(await health()));
  for (const q of BASE) await post(q, true);
  console.log("after 10 more:", JSON.stringify(await health()));
}
