"""Reading the manifest back exactly as manifest.py wrote it.

The manifest is written with `csv.DictWriter` and was read, in one of seven
places, by splitting each line on tabs. Those two agree until a curator types a
character `csv` has to quote, and then they disagree silently: the value reaches
`meta.json`, the result card and the text every chunk of that document is embedded
with, mangled, without anything raising.

So these tests are mostly round trips. What is written must be what is read.

Written by Claude Code (Opus 5).
"""

from __future__ import annotations

import csv

import pytest

COLUMNS = ["file", "title", "issuer", "year", "notes"]


def write_manifest(path, rows, columns=COLUMNS):
    """Write rows exactly as manifest.py does, quoting and all."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("title", [
    'Guide "PLU" chez l\'adulte',        # the quote a curator actually types
    'Relevé\tet diagnostic',          # a tab pasted from a spreadsheet
    "Two\nlines",                        # a newline pasted from a PDF
    'She said "yes", twice',             # a quote and a comma
    "Ordinary title",                    # and the control
])
def test_a_cell_survives_the_round_trip(manifest_io, tmp_path, title):
    """Anything csv quotes on the way out has to come back unquoted."""
    path = tmp_path / "MANIFEST.tsv"
    write_manifest(path, [{"file": "a.pdf", "title": title, "issuer": "INSEE",
                           "year": "2024", "notes": ""}])

    rows = manifest_io.read_by_file(path)

    assert rows["a.pdf"]["title"] == title
    assert rows["a.pdf"]["issuer"] == "INSEE", "a quoted cell shifted the columns"
    assert len(rows) == 1, "a quoted newline was read as a second document"


def test_splitting_on_tabs_is_what_this_replaces(manifest_io, tmp_path):
    """The bug, pinned: the old parser and the file disagree.

    Not a test of production code, but of the premise. If csv ever stopped
    quoting these, the round trip above would pass for the wrong reason.
    """
    path = tmp_path / "MANIFEST.tsv"
    write_manifest(path, [{"file": "a.pdf", "title": 'Guide "PLU"', "issuer": "INSEE",
                           "year": "2024", "notes": ""}])
    naive = path.read_text(encoding="utf-8").splitlines()[1].split("\t")
    assert naive[1] == '"Guide ""PLU"""'
    assert manifest_io.read_by_file(path)["a.pdf"]["title"] == 'Guide "PLU"'


def test_a_short_row_reads_as_blanks_not_none(manifest_io, tmp_path):
    """A trailing column dropped by a spreadsheet export is not an error.

    build_index.py used to pad its split for this; the tolerance has to survive,
    and `None` in a metadata cell would reach meta.json as a JSON null.
    """
    path = tmp_path / "MANIFEST.tsv"
    path.write_text("file\ttitle\tissuer\tyear\tnotes\na.pdf\tTitre\tINSEE\n",
                    encoding="utf-8")
    row = manifest_io.read_by_file(path)["a.pdf"]
    assert row["year"] == "" and row["notes"] == ""
    assert None not in row.values()


def test_a_long_row_does_not_leak_an_overflow_key(manifest_io, tmp_path):
    """A row longer than the header is corrupt; it must not add a None column."""
    path = tmp_path / "MANIFEST.tsv"
    path.write_text("file\ttitle\na.pdf\tTitre\tstray\tvalues\n", encoding="utf-8")
    row = manifest_io.read_by_file(path)["a.pdf"]
    assert set(row) == {"file", "title"}
    assert all(isinstance(key, str) for key in row)


def test_read_rows_keeps_file_order(manifest_io, tmp_path):
    path = tmp_path / "MANIFEST.tsv"
    write_manifest(path, [{"file": f"{n}.pdf", "title": f"T{n}"} for n in "cab"])
    assert [row["file"] for row in manifest_io.read_rows(path)] == \
        ["c.pdf", "a.pdf", "b.pdf"]
