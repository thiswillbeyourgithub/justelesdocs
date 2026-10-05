/* Where the corpus lives, for the node tools: the JS half of scripts/lib/corpus.py.
 *
 * The built index, the eval queries and the grid artefacts belong to one
 * deployment's corpus, named by the CORPUS_DIR environment variable, not to this
 * checkout. Every node tool that used to default to `<repo>/dist/...` or
 * `<repo>/data/...` defaults to the same relative path under $CORPUS_DIR instead,
 * so CORPUS_DIR=. reproduces the old layout exactly. An explicit path (argv,
 * INDEX_DIR, QUERIES, OUT) never consults it.
 *
 * Required rather than defaulted, as in the Python half: a silent fallback to the
 * checkout would read like an empty corpus rather than a missing setting. The
 * search service itself never calls this; it is told INDEX_DIR by compose.
 *
 * Written by Claude Code (Opus 5.5).
 */
import { resolve } from "node:path";

/**
 * `relative` inside $CORPUS_DIR, resolved against the working directory.
 * Exits 2 with a message naming the variable when it is unset, since every
 * caller is a command-line tool with nothing sensible to do without a corpus.
 *
 * @param {string} relative - e.g. "dist/index" or "data/EVAL_QUERIES.tsv"
 * @returns {string} an absolute path
 */
export function corpusPath(relative) {
  const dir = (process.env.CORPUS_DIR || "").trim();
  if (!dir) {
    console.error("CORPUS_DIR is not set. It names the corpus directory (data/, dist/); "
      + "for this checkout that is usually `export CORPUS_DIR=local/psydocs`.");
    process.exit(2);
  }
  return resolve(dir, relative);
}
