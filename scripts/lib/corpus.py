"""Where the corpus lives: one directory, named by the `CORPUS_DIR` environment variable.

The software (this repository) and the corpus it serves are separate things: the
documents, their manifest, the curation records, the chunks, the bakes and the
built `dist/` all belong to one deployment, and another deployment of the same
software has its own. Every script used to default to `data/...` and `dist/...`
relative to the working directory, which tied the code to one corpus sitting
inside its checkout. Now every such default is a path INSIDE `$CORPUS_DIR`, with
the same relative spelling as before, so `CORPUS_DIR=.` reproduces the old layout
exactly and `CORPUS_DIR=local/psydocs` points the same commands at a corpus kept in
its own repository.

It is required rather than defaulted, deliberately: a forgotten variable that
fell back to `./data` would make `chunk.py` create an empty `data/chunks/` in the
software checkout and report zero documents, which reads like a corpus problem.
An unset variable fails at the first default it is asked for, saying so.

Resolution is lazy, at the moment a path is used, never at import: the test suite
imports these scripts with no corpus anywhere, and an absolute path (a test's
tmp_path, an explicit `--out`) never consults the variable at all.

Module imported, not run. Standard library plus click, which every caller has.

Written by Claude Code (Opus 5.5).
"""

from __future__ import annotations

import os
from pathlib import Path

import click

VARIABLE = "CORPUS_DIR"


def corpus_dir(*, required: bool = True) -> Path | None:
    """The corpus root named by `$CORPUS_DIR`, relative to the working directory.

    Parameters
    ----------
    required : bool, optional
        Raise when the variable is unset (the default). False returns None
        instead, for the few import-time readers that have a sensible empty
        answer without a corpus.

    Returns
    -------
    Path or None
        The directory, not checked for existence: a missing one fails at the
        first file read under it, with that file's name in the message.

    Raises
    ------
    click.UsageError
        When the variable is unset or empty and `required` is true.
    """
    value = os.environ.get(VARIABLE, "").strip()
    if value:
        return Path(value)
    if required:
        raise click.UsageError(
            f"{VARIABLE} is not set. It names the corpus directory (data/, dist/, the "
            f"curation records); for this checkout that is usually `export {VARIABLE}=local/psydocs`.")
    return None


def corpus_path(path: str | Path) -> Path:
    """`path` inside the corpus, or `path` itself when it is already absolute.

    Parameters
    ----------
    path : str or Path
        A corpus-relative path such as ``data/MANIFEST.tsv``.

    Returns
    -------
    Path
        ``$CORPUS_DIR/path``. An absolute `path` is returned unchanged without
        reading the variable, which is what lets a test monkeypatch a module's
        relative default with a tmp_path and run with no corpus set.
    """
    path = Path(path)
    return path if path.is_absolute() else corpus_dir() / path


def in_corpus(path: str) -> dict[str, object]:
    """A click option's `default` and `show_default` for a corpus-relative path.

    Spread into the option, ``@click.option("--out", **in_corpus("data/chunks"))``,
    so the default is resolved when the command runs (and only if the option was
    not given) while `--help` still shows where it points.

    Parameters
    ----------
    path : str
        The corpus-relative default.

    Returns
    -------
    dict
        ``default`` (a callable click invokes lazily) and ``show_default``.
    """
    return {"default": lambda: corpus_path(path), "show_default": f"${VARIABLE}/{path}"}
