"""`figures.py` bookkeeping that a concurrent deploy depends on.

The figure and table passes run for days and a deploy can start at any moment,
so `chunk.py` may read `descriptions.tsv` while a batch is being applied.

Written by Claude Code (Opus 5.5).
"""

from __future__ import annotations

import csv

import pytest

pytest.importorskip("pymupdf")
pytest.importorskip("click")


def read(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_append_writes_a_header_once_and_keeps_earlier_rows(figures, tmp_path):
    path = tmp_path / "descriptions.tsv"
    figures.append_tsv(path, ["key", "description"], [{"key": "a", "description": "un"}])
    figures.append_tsv(path, ["key", "description"], [{"key": "b", "description": "deux"}])
    assert read(path) == [{"key": "a", "description": "un"}, {"key": "b", "description": "deux"}]
    assert not (tmp_path / "descriptions.tsv.tmp").exists()


def test_append_replaces_the_file_rather_than_writing_into_it(figures, tmp_path):
    # A reader holding the old file must keep seeing it whole: that is only true
    # when the append lands in a new inode, not at the end of the one being read.
    path = tmp_path / "descriptions.tsv"
    figures.append_tsv(path, ["key"], [{"key": "a"}])
    with path.open(encoding="utf-8") as held:
        figures.append_tsv(path, ["key"], [{"key": "b"}])
        assert held.read() == "key\na\n"
    assert [r["key"] for r in read(path)] == ["a", "b"]
