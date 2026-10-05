"""The eval sample is a tracked file, so it may only quote open documents.

The corpus holds books that are served but never committed. Nothing stopped the
sampler from drawing a page of one and writing it into data/EVAL_PASSAGES.tsv,
which would have published the content by another route. These tests hold that
gate shut, on both samplers: the prose one walks the PDFs, the table and dose one
walks the chunk files, and a filter added to one of them only is the way this
comes back.

Written by Claude Code (Opus 5).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest


@pytest.fixture()
def meta() -> dict[str, dict[str, str]]:
    return {
        "open.pdf": {"access": "open", "language": "fr", "issuer": "INSEE"},
        "book.pdf": {"access": "restricted", "language": "en", "issuer": ""},
    }


def chunk_file(directory: Path, name: str, text: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{Path(name).stem}.json").write_text(json.dumps(
        {"file": name, "pages_total": 10,
         "chunks": [{"text": text, "page": 3, "pages": [3]}]}), encoding="utf-8")


def test_an_open_document_is_shareable(sample_passages, meta):
    assert sample_passages.shareable("open.pdf", meta)


def test_a_restricted_document_is_not(sample_passages, meta):
    assert not sample_passages.shareable("book.pdf", meta)


def test_a_document_the_manifest_does_not_know_is_not(sample_passages, meta):
    """Unknown is not open. A document added but not yet in the manifest is refused."""
    assert not sample_passages.shareable("brand-new.pdf", meta)


def test_the_table_sample_skips_a_restricted_document(sample_passages, meta, tmp_path):
    chunks = tmp_path / "chunks"
    # The caption chunk.py writes above a table it has flattened.
    chunk_file(chunks, "open.pdf",
               "Tableau de 5 lignes et 3 colonnes\nTarifs usuels par commune")
    chunk_file(chunks, "book.pdf",
               "Table of 7 rows and 4 columns\nMaximum daily loads by region")
    rows = sample_passages.chunk_rows(
        chunks, meta=meta, n=10, rng=random.Random(1),
        match=lambda text: sample_passages.TABLE_CAPTION.match(text.lstrip()),
        prefix="t")
    assert [row["file"] for row in rows] == ["open.pdf"]
