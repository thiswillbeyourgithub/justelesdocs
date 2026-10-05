"""The corpus's own configuration: `$CORPUS_DIR/corpus.toml`.

Everything that describes ONE corpus rather than the software lives in that file:
the closed vocabularies of its facets, the tier a document type puts a document
in, which renditions of one guidance restate each other, which issuing bodies to
recognise on a cover and how, and which cover lines are labels rather than
titles. The software reads it; it never restates a value from it.

The file is read once, at first use, and cached. It is REQUIRED: a corpus with no
configuration has no vocabulary to validate a manifest against, and guessing one
would let every typo through as a new facet option. `corpus.example/corpus.toml`
in the software repository is a complete, commented example to start from, and is
what the test suite runs against.

Module imported, not run. Standard library (tomllib, Python 3.11) plus click,
through lib.corpus.

Written by Claude Code (Opus 5.5).
"""

from __future__ import annotations

import functools
import tomllib
from pathlib import Path
from typing import Any

from lib.corpus import corpus_path

FILE_NAME = "corpus.toml"


@functools.cache
def load() -> dict[str, Any]:
    """The parsed `$CORPUS_DIR/corpus.toml`.

    Returns
    -------
    dict
        The TOML document as plain dicts and lists. Cached for the life of the
        process: every caller sees the same object, so treat it as read-only.

    Raises
    ------
    FileNotFoundError
        When the corpus has no configuration file, naming the example to copy.
    tomllib.TOMLDecodeError
        When the file is not valid TOML.
    """
    path = corpus_path(FILE_NAME)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist. Every corpus needs one: copy "
            f"corpus.example/{FILE_NAME} from the software repository and edit it.")
    with path.open("rb") as handle:
        return tomllib.load(handle)


def section(name: str) -> dict[str, Any]:
    """One top-level table of the configuration, refusing a missing one.

    Parameters
    ----------
    name : str
        The table's name, such as ``"facets"``.

    Returns
    -------
    dict
        That table.

    Raises
    ------
    KeyError
        When the file has no such table, naming the file, so the error says where
        to add it rather than which line of Python wanted it.
    """
    config = load()
    if name not in config:
        raise KeyError(f"{corpus_path(FILE_NAME)} has no [{name}] table")
    return config[name]


def path_of() -> Path:
    """Where the configuration is read from, for messages."""
    return corpus_path(FILE_NAME)
