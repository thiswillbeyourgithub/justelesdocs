# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "loguru", "numpy", "pytest", "pymupdf==1.28.2", "tokenizers"]
# ///
"""Run the whole test suite: the Python half, then the JavaScript half.

The repository has no package.json, no node_modules and no linter config (see
CLAUDE.md), and the suite is written to keep it that way. The Python tests run
under pytest, which `uv run` installs into this script's own ephemeral
environment; the front-end tests run under node's built-in test runner, which
needs nothing installed at all. So a fresh clone can run everything with `uv run
tests/run.py` and no setup step.

This is the entry point `.githooks/pre-push` calls. It is deliberately a single
command rather than three, so there is one answer to "did the tests pass".

The third tier is the CORPUS's: when $CORPUS_DIR is set and holds a `tests/`
directory, its tests run after the software's, in their own process (see
`run_corpus_tests`). The software's own tests never need a corpus.

The suite covers the build-time Python scripts and the pure-text half of the
front end. It does NOT replace the six gates (verify_chunks.py, check_served.py,
check_search.mjs, check_ui.mjs, check_chrome.mjs, smoke_deployed.sh): those check
the corpus, the staged tree, the shipped index and the rendered page, which no unit
test can see.

Written by Claude Code (Opus 5).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import click
import pytest
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
# The caller's corpus, read before the Python half runs: tests/conftest.py repoints
# CORPUS_DIR at the example corpus for the whole process when pytest imports it, so
# reading the variable after `pytest.main` would find that temporary directory.
CALLER_CORPUS = os.environ.get("CORPUS_DIR", "").strip()


def run_python_tests(*, verbose: bool) -> int:
    """Run the pytest half in this interpreter.

    Calling ``pytest.main`` rather than spawning a subprocess is what lets the
    tests import click, loguru and pymupdf: those are dependencies of THIS script,
    so they exist in the environment uv built for it and nowhere else.

    Parameters
    ----------
    verbose : bool
        Pass ``-v`` instead of ``-q``.

    Returns
    -------
    int
        The pytest exit code (0 when every test passed, 5 when it collected none).
    """
    # -p no:cacheprovider keeps pytest from writing a .pytest_cache directory into
    # the repository root. Nothing here uses --lf or --ff, and an untracked cache
    # directory would either show up in every `git status` or need a .gitignore entry.
    args = [str(ROOT / "tests"), "-v" if verbose else "-q", "-p", "no:cacheprovider"]
    logger.info("pytest {}", " ".join(args))
    return pytest.main(args)


def run_corpus_tests(*, verbose: bool) -> int:
    """Run `$CORPUS_DIR/tests`, a corpus's own tests, in a subprocess.

    A corpus can carry tests of its own records and policy (its manifest, its eval
    sets, the issuers its corpus.toml must recognise). They run in a SEPARATE
    process because the policy modules read corpus.toml once, at import: in the
    process above they already carry the software's example corpus, and a second
    `pytest.main` here would reuse those modules rather than read the corpus's.
    `sys.executable` is this script's own uv environment, so the subprocess has
    pytest, click, loguru and pymupdf exactly as the software's half does.

    Returns
    -------
    int
        pytest's exit code, or 0 when there is no corpus or it has no tests.
    """
    tests = Path(CALLER_CORPUS).resolve() / "tests" if CALLER_CORPUS else None
    if tests is None or not tests.is_dir():
        logger.info("no $CORPUS_DIR/tests: no corpus tests to run")
        return 0
    command = [sys.executable, "-m", "pytest", str(tests), "-v" if verbose else "-q",
               "-p", "no:cacheprovider", "--rootdir", str(tests)]
    logger.info("pytest {} (corpus tests, own process)", tests)
    # JUSTELESDOCS_DIR tells the corpus's conftest where the software's helpers are,
    # wherever the corpus directory happens to live.
    env = {**os.environ, "JUSTELESDOCS_DIR": str(ROOT), "CORPUS_DIR": CALLER_CORPUS}
    return subprocess.run(command, cwd=tests, env=env, check=False).returncode


def run_node_tests(*, verbose: bool) -> int:
    """Run the node half as a subprocess.

    Returns
    -------
    int
        node's exit code, or 127 when node is not installed.
    """
    files = sorted(str(p) for p in (ROOT / "tests").glob("*.test.mjs"))
    if not files:
        logger.warning("no *.test.mjs files: skipping the front-end tests")
        return 0
    command = ["node", "--test", *([] if verbose else ["--test-reporter=dot"]), *files]
    logger.info("node --test ({} files)", len(files))
    try:
        return subprocess.run(command, cwd=ROOT, check=False).returncode
    except FileNotFoundError:
        logger.error("node is not installed: the front-end tests cannot run")
        return 127


@click.command()
@click.option("--verbose", is_flag=True, help="Name every test instead of summarising.")
@click.option("--python-only", is_flag=True, help="Skip the front-end tests.")
@click.option("--node-only", is_flag=True, help="Skip the Python tests.")
def main(*, verbose: bool, python_only: bool, node_only: bool) -> None:
    """Run the test suite and exit non-zero if any part of it fails."""
    failures: list[str] = []

    # Both halves run even when the first one fails: a push that breaks two things
    # should report two things, not send the author round the loop twice.
    if not node_only:
        if run_python_tests(verbose=verbose) != 0:
            failures.append("python")
        if run_corpus_tests(verbose=verbose) != 0:
            failures.append("corpus")
    if not python_only:
        if run_node_tests(verbose=verbose) != 0:
            failures.append("javascript")

    if failures:
        logger.error("FAILED: {}", ", ".join(failures))
        sys.exit(1)
    logger.success("test suite passed")


if __name__ == "__main__":
    main()
