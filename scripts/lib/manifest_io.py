"""Read `data/MANIFEST.tsv`, once, the way `manifest.py` writes it.

Seven scripts needed the manifest and seven read it themselves. Six agreed, which
is the ordinary cost of a copied three-liner; the seventh, `build_index.py`, had
grown its own parser that split each line on tabs, and that one was a bug waiting
for a curator.

`manifest.py` writes the file with `csv.DictWriter`, which QUOTES any cell holding
a `"`, a tab or a newline and doubles the quotes inside it. A title containing a
quotation mark is an ordinary thing for a curator to type, and splitting on tabs
reads it back as `Guide ""TSA""` (or, for a newline, as two broken rows). Nothing
raises: the mangled value goes into `meta.json`, onto the result card, and into the
text `embed.py --variant meta` prepends to every chunk of that document.

So the rule this module exists to hold is small and absolute: **the manifest is a
CSV file with a tab delimiter, and it is read with the `csv` module.** No caller
needs to know that, which is the point.

This module is imported, not run. It needs nothing outside the standard library,
so any script can use it whatever its PEP 723 header says; uv puts the script's own
directory on sys.path, which is what makes `from lib.manifest_io import ...` work
from `scripts/`.

Written by Claude Code (Opus 5).
"""

from __future__ import annotations

import csv
from pathlib import Path

#: The manifest's delimiter. Named rather than spelled out at each call, so that
#: "the manifest is a TSV" is stated once here and nowhere else.
DELIMITER = "\t"


def read_rows(path: Path) -> list[dict[str, str]]:
    """Every manifest row, in file order.

    Parameters
    ----------
    path
        `data/MANIFEST.tsv`, or another file with the same shape.

    Returns
    -------
    list of dict
        One dict per row, column name to value. A cell absent from a short row
        reads as `""` rather than as None, so callers can treat "blank" and
        "missing" alike, which is what every one of them already did.

    Notes
    -----
    `restval=""` is what tolerates a trailing empty column dropped by a
    spreadsheet's export, the tolerance `build_index.py` used to implement by
    padding its split. Any column beyond the header is collected under
    `restkey` and discarded here: a row longer than the header is a corrupt row,
    and carrying its overflow forward under a None key would put `None` into
    `meta.json`.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=DELIMITER, restval="",
                                restkey="__overflow__")
        rows = []
        for row in reader:
            row.pop("__overflow__", None)
            rows.append({key: (value if value is not None else "")
                         for key, value in row.items()})
        return rows


def read_by_file(path: Path) -> dict[str, dict[str, str]]:
    """Manifest rows keyed by the document's filename.

    Parameters
    ----------
    path
        `data/MANIFEST.tsv`.

    Returns
    -------
    dict
        `{filename: {column: value}}`, the `file` column included so a row stays
        a complete record of itself.

    Raises
    ------
    KeyError
        If the file has no `file` column, which means it is not a manifest.

    Notes
    -----
    Last row wins on a duplicated filename, which is the behaviour every caller
    already had. `manifest.py` writes one row per PDF on disk, so a duplicate
    means the file was edited by hand into a state `check_titles` would also
    complain about.
    """
    return {row["file"]: row for row in read_rows(path)}


#: The two `access` tiers. A document is either served as a file or cut page by
#: page by the page service; there is no third behaviour to fall back on.
ACCESS_TIERS = ("open", "restricted")


def read_access(path: Path) -> dict[str, str]:
    """Map each document's filename to its `access` tier, refusing any doubt.

    Parameters
    ----------
    path
        `data/MANIFEST.tsv`.

    Returns
    -------
    dict[str, str]
        Filename to `"open"` or `"restricted"`.

    Raises
    ------
    ValueError
        When the column is missing, or a row's tier is blank or neither value.

    Notes
    -----
    One reader for the tier, shared by `stage.py` (which decides where a file
    goes), `check_served.py` (which refuses a tree that contradicts it) and
    `catalog.py`. Until 2026-10-03 the first two each had their own, and they had
    drifted: only one validated the values. The tier is the project's legal
    boundary, so a blank cell is an error rather than a default: reading it as
    `"open"` is exactly how a restricted book would end up served as a file.
    `manifest.py` writes `open` explicitly on every row it creates, so a blank
    here means a hand edit, not a new document.
    """
    rows = read_rows(path)
    if rows and "access" not in rows[0]:
        raise ValueError(f"{path} has no 'access' column. Re-run manifest.py.")
    tiers: dict[str, str] = {}
    for row in rows:
        tier = row["access"].strip()
        if tier not in ACCESS_TIERS:
            raise ValueError(
                f"{row['file']} has access={tier!r} in {path}, which is neither "
                "'open' nor 'restricted'. Guessing would either hide a guideline "
                "or publish a book.")
        tiers[row["file"]] = tier
    return tiers
