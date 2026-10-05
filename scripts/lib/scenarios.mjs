/* What a corpus tells the gates about itself: $CORPUS_DIR/scenarios.json.
 *
 * The browser and search gates (check_ui.mjs, check_search.mjs) and the service
 * measurements (measure_service.mjs) have a generic
 * half, which derives what it expects from the served index and the page itself,
 * and a corpus half, which needs real questions the corpus can answer and values
 * only a reader of that corpus would know to check (an issuer's expanded label,
 * the regex a good answer matches). The second half lives with the corpus, in one
 * JSON file, so the software names no corpus and a corpus with no file still gets
 * the generic half: every key is optional and its absence is a fallback or a
 * reported skip, never a failure.
 *
 * Keys read today (see each gate for what it does with them):
 *   ui.queries.{answer,views,negated,bm25,advanced,filtered,widen}  questions
 *   ui.answer_expect     regex the top row of ui.queries.answer must match
 *   ui.issuer            an issuer slug to filter on via the URL
 *   ui.issuer_label      regex one issuer choice must match (an expanded acronym)
 *   ui.issuer_absent     a choice that must NOT be offered (a merged alias)
 *   ui.topic_search      a substring to type into the topic facet's search box
 *   ui.language_note     regex the syntax panel's language advice must match
 *   ui.name_filter       a word typed into the browse page's title filter
 *   search.queries       live questions check_search.mjs prints the top hits of
 *   smoke.question       the question deploy.sh passes to smoke_deployed.sh
 *   measure.queries      ten questions measure_service.mjs times (latency, burst)
 *   measure.topics, measure.aspects  words it combines into distinct questions
 *
 * Unlike corpusPath, an unset CORPUS_DIR is not an error here: check_ui.mjs reads
 * everything else over HTTP from SITE and can run against any site.
 *
 * Written by Claude Code (Opus 5.5).
 */
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

/** The parsed scenarios file, or {} when there is no corpus or no file. */
export function loadScenarios() {
  const dir = (process.env.CORPUS_DIR || "").trim();
  if (!dir) return {};
  const path = resolve(dir, "scenarios.json");
  return existsSync(path) ? JSON.parse(readFileSync(path, "utf8")) : {};
}
