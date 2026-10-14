/* A search answer as Markdown, for an AI assistant reading the site.
 *
 * `GET /api/search.md` takes the search page's own query string (src/urlstate.js)
 * and answers with the same folded list `POST /api/search` returns, written for
 * something that reads text rather than renders it: one block per passage with
 * its document's metadata, the passage itself, a link to the passage's page as
 * Markdown (open documents) and a link a person can open in the viewer.
 *
 * Nothing leaves here that the POST does not already return: a restricted
 * document's passage is the same one excerpt, and its links go to the viewer
 * rather than to an `api/doc/` page, which does not exist for that tier.
 *
 * Written by Claude Code (Opus 5.5).
 */

import { apiDocHref, viewHref } from "../src/paths.js";

/** How many passages one answer lists. The page shows more, a reader scrolls. */
export const MD_RESULTS = 10;

/* Host names a link may be built on. Anything else (a forged Host header) gets
   root-relative links, which only cost the asker their own absolute URLs. */
const HOST = /^[a-z0-9.-]+(:\d{1,5})?$/i;

/**
 * The site's origin as the request saw it, or "" when it cannot be trusted.
 *
 * TLS ends upstream of Caddy, so the scheme Caddy forwards is always http and says
 * nothing. A public host is served over https; a loopback one is a developer's.
 *
 * @param {string|undefined} host - The request's Host header.
 * @returns {string}
 */
export function originOf(host) {
  if (!host || !HOST.test(host)) return "";
  const local = /^(localhost|127\.|\[::1\])/.test(host);
  return `${local ? "http" : "https"}://${host}`;
}

/** Collapse a cell's `;`-separated values to a readable list. */
const cell = (value) => String(value ?? "").split(";").map((v) => v.trim()).filter(Boolean).join(", ");

/**
 * One document's metadata as a single line: the corpus's own facets, in its order.
 *
 * @param {object} doc - A record out of meta.json.
 * @param {string[]} fields - `search.facetFields(index)`.
 */
export function metadataLine(doc, fields) {
  return fields.map((f) => [f, cell(doc[f])]).filter(([, v]) => v && v !== "-")
    .map(([f, v]) => `${f}: ${v}`).join(" · ");
}

/** A passage as a Markdown quote, every line prefixed so a blank line cannot end it. */
const quote = (text) => String(text || "").trim().split("\n").map((line) => `> ${line}`.trimEnd()).join("\n");

/**
 * Render an answer.
 *
 * @param {object} options
 * @param {string} options.question - The question as asked.
 * @param {{results: object[]}} options.answer - What the search service returned.
 * @param {object} options.index - The loaded index (`docs`, `meta`).
 * @param {string[]} options.fields - The facet fields to print per document.
 * @param {string} options.origin - `originOf(host)`, possibly "".
 * @param {string} options.query - The request's query string, for the human link.
 * @returns {string}
 */
export function renderSearch({ question, answer, index, fields, origin, query }) {
  const at = (path) => `${origin}/${path}`;
  const shown = answer.results.slice(0, MD_RESULTS);
  const lines = [
    `# Search: ${question}`,
    "",
    `Passages matched: ${answer.results.length}. The best ${shown.length} follow, best first. ` +
      `A person can see the same list at ${at(`?${query}`)}`,
    "",
  ];
  shown.forEach((r, i) => {
    const doc = index.docs[r.doc];
    const pages = r.pages && r.pages.length > 1 ? `pp. ${r.pages[0]}-${r.pages[r.pages.length - 1]}` : `p. ${r.page}`;
    lines.push(`## ${i + 1}. ${doc.title || doc.file}, ${pages}`, "");
    const meta = metadataLine(doc, fields);
    if (meta) lines.push(meta, "");
    lines.push(`Score ${r.score.toFixed(3)}${r.figure ? " · a described figure (an AI-written description, not the document's own words)" : ""}` +
      (r.more ? ` · ${r.more} more matching passages in this document` : ""), "");
    lines.push(quote(r.text), "");
    if (doc.access !== "restricted") lines.push(`Page text: ${at(apiDocHref(doc.file, r.page))}`);
    else lines.push("Restricted document: no text page, and the PDF is not downloadable. A person can read it page by page in the viewer.");
    lines.push(`Viewer, passage highlighted: ${at(viewHref(r.doc, doc.file, { chunk: r.chunk, page: r.page }))}`, "");
  });
  if (!shown.length) lines.push("No passage matched. Try other words, another language, or fewer filters.", "");
  return lines.join("\n");
}

/**
 * A refusal as Markdown: what the error means and how to ask again.
 *
 * @param {string} error - The name `describeError` gave.
 * @param {object} payload - Its details (`min`, `got`, `max`).
 */
export function renderError(error, payload) {
  const why = {
    bad_request: "The request could not be read. Ask with `?q=<question>` and the filters the search page puts in its URL.",
    empty: "The question has nothing left to search for once its `-(...)` groups are taken out.",
    short: `The question is too short: at least ${payload.min ?? "a few"} characters.`,
    long: `The question is too long: ${payload.got} characters, the limit is ${payload.max}.`,
    terms: `The question has ${payload.got} alternatives, the limit is ${payload.max}.`,
    unavailable: "Search is unavailable right now (the encoder is not answering). Try again later.",
    full: "Search is busy. Wait a few seconds and ask again.",
  }[error] || "Search failed.";
  return `# Search error: ${error}\n\n${why}\n`;
}
