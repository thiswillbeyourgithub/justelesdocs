/* Every JavaScript module in the repository, parsed.
 *
 * Most of src/ is DOM code that no other test imports, so a syntax error in it
 * passes the whole suite and only shows up in a browser, as a page whose module
 * never ran: every data-i18n label empty, every button blank. That shipped once,
 * when removing a listener from app.js left its closing `});` behind.
 *
 * The build and serving scripts (scripts/, server/, and their lib/) get the same
 * check for a different reason: a gate or a sweep that no test imports fails only
 * when someone runs it, which for an evaluation script can be weeks later.
 * tests/test_syntax.py does the same for the Python and shell files.
 *
 * Run by `uv run tests/run.py`, which is what .githooks/pre-push runs.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const SRC = resolve(ROOT, "src");

for (const name of readdirSync(SRC).filter((f) => f.endsWith(".js"))) {
  test(`src/${name} parses as a module`, () => {
    // On stdin with --input-type=module, not `node --check <file>`: for a .js file with
    // no package.json, node 22 guesses the module type and that guess exits 0 on the
    // very app.js this test was written for.
    const result = spawnSync(process.execPath, ["--check", "--input-type=module"],
                             { input: readFileSync(resolve(SRC, name)), encoding: "utf8" });
    assert.equal(result.status, 0, result.stderr);
  });
}

/* .mjs is unambiguous, so `node --check <file>` is reliable here, unlike for src/'s
   .js files above. A directory that does not exist is skipped rather than failed:
   the list is what may hold modules, not a claim that each one does. */
for (const dir of ["scripts", "scripts/lib", "server", "server/lib"]) {
  let names = [];
  try { names = readdirSync(resolve(ROOT, dir)).filter((f) => f.endsWith(".mjs")); } catch { continue; }
  for (const name of names) {
    test(`${dir}/${name} parses`, () => {
      const result = spawnSync(process.execPath, ["--check", resolve(ROOT, dir, name)], { encoding: "utf8" });
      assert.equal(result.status, 0, result.stderr);
    });
  }
}
