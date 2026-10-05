// The search image holds a hand-picked subset of src/, and that subset has to be
// the ranker plus every module it imports, transitively. Nothing at build time
// checks it: a module missing from docker/search.Dockerfile's COPY still builds,
// and the container then dies at start with ERR_MODULE_NOT_FOUND. That is what
// would have shipped when src/i18n.js began importing src/store.js, because the
// local service runs from the full tree and never notices.
//
// So this rebuilds the image's file set in a temporary directory (only what the
// COPY lines name, nothing else from src/) and imports the ranker from it under
// node, exactly as server/load.mjs does in the container. It also checks that
// the root .dockerignore re-admits every one of those files, since a COPY of a
// file the build context does not contain fails the build on the VPS instead.
import { test } from "node:test";
import assert from "node:assert/strict";
import { cpSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

/** Every `src/...` path a `COPY ... /app/src/` line of the Dockerfile names. */
function copiedSources() {
  const dockerfile = readFileSync(join(ROOT, "docker/search.Dockerfile"), "utf8");
  return dockerfile.split("\n")
    .filter((line) => /^COPY\s/.test(line) && line.trim().endsWith("/app/src/"))
    .flatMap((line) => line.trim().split(/\s+/).slice(1, -1));
}

test("the Dockerfile copies at least the ranker", () => {
  assert.ok(copiedSources().includes("src/search.js"), copiedSources().join(" "));
});

test("the ranker imports from the image's own file set alone", async () => {
  const dir = mkdtempSync(join(tmpdir(), "search-image-"));
  try {
    for (const file of copiedSources()) cpSync(join(ROOT, file), join(dir, file));
    // Imported, not just parsed: a module-level import of a file that was not
    // copied fails here exactly as it would in the container.
    const search = await import(pathToFileURL(join(dir, "src/search.js")).href);
    assert.equal(typeof search.rank, "function");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("the root .dockerignore re-admits every file the Dockerfile copies", () => {
  const admitted = new Set(readFileSync(join(ROOT, ".dockerignore"), "utf8")
    .split("\n").filter((line) => line.startsWith("!")).map((line) => line.slice(1).trim()));
  for (const file of copiedSources()) {
    assert.ok(admitted.has(file), `${file} is COPYd but .dockerignore does not re-admit it`);
  }
});
