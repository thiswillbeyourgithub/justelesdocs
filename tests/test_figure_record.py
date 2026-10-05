"""Figure descriptions read back as chunks: which ones count, and where their box lands.

`figures.py` records crops and the descriptions a model wrote of them; `chunk.py`
turns each described figure into a chunk through `lib/figure_record.py`. What can go
wrong is quiet: a superseded description indexed instead of the newer one, a
description of a PDF that has since been replaced pinned to its new page, a
document's hash moving (and its bake going stale) when it has no figure at all, or a
box drawn in the rotated page's space on a page every other box treats unrotated.

Written by Claude Code (Opus 5.5).
"""

from __future__ import annotations

import csv
import hashlib

import pytest

pymupdf = pytest.importorskip("pymupdf")


def write(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fields, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def corpus(tmp_path):
    """One two-page PDF, the second page rotated a quarter turn."""
    root = tmp_path / "GUIDELINES"
    root.mkdir()
    doc = pymupdf.open()
    doc.new_page(width=600, height=800)
    doc.new_page(width=600, height=800).set_rotation(90)
    doc.save(root / "a.pdf")
    return root


def record(figure_record, corpus, work, rows, descriptions):
    """figures.tsv rows given as (page, bbox), keyed the way figures.py keys them."""
    sha = hashlib.sha256((corpus / "a.pdf").read_bytes()).hexdigest()
    figures = [{"key": figure_record.figure_key(sha, page, bbox), "file": "a.pdf",
                "page": page, "bbox": bbox} for page, bbox in rows]
    write(work / "figures.tsv", ["key", "file", "page", "bbox"], figures)
    write(work / "descriptions.tsv", ["key", "is_figure", "kind", "language", "description",
                                       "model"],
          [{"key": figures[i]["key"], "is_figure": yes, "kind": "flowchart", "language": "fr",
            "description": text, "model": model} for i, yes, text, model in descriptions])
    return [f["key"] for f in figures]


def test_the_latest_description_wins_and_non_figures_are_left_out(figure_record, corpus, tmp_path):
    record(figure_record, corpus, tmp_path, [(1, "10,10,100,100"), (1, "10,200,100,300")], [
        (0, "yes", "Sonnet's reading", "claude-sonnet-5"),
        (0, "yes", "Opus's reading", "claude-opus-5-5"),
        (1, "no", "logo", "claude-sonnet-5"),
    ])
    found = figure_record.described(corpus, tmp_path)["a.pdf"]
    assert [(f.description, f.model) for f in found] == [("Opus's reading", "claude-opus-5-5")]


def test_a_replaced_pdf_drops_its_old_descriptions(figure_record, corpus, tmp_path):
    record(figure_record, corpus, tmp_path, [(1, "10,10,100,100")],
           [(0, "yes", "a flowchart", "claude-sonnet-5")])
    doc = pymupdf.open()
    doc.new_page()
    doc.save(corpus / "a.pdf")
    assert figure_record.described(corpus, tmp_path) == {}


def test_no_figure_leaves_the_hash_alone(figure_record):
    """The cached chunks and vectors of a figureless document must stay valid."""
    assert figure_record.fold("abc", []) == "abc"


def test_a_corrected_language_moves_the_hash(figure_record):
    """The language picks the kind word embed.py puts before the description, so a
    corrected one has to rechunk the document or its vectors keep the wrong word."""
    fr = figure_record.Figure("k", 1, (0, 0, 1, 1), "texte", "table", "claude-sonnet-5", "fr")
    en = figure_record.Figure("k", 1, (0, 0, 1, 1), "texte", "table", "claude-sonnet-5", "en")
    assert figure_record.fold("abc", [fr]) != figure_record.fold("abc", [en])


def test_a_new_description_moves_the_hash(figure_record):
    one = figure_record.Figure("k", 1, (0, 0, 1, 1), "first", "chart", "claude-sonnet-5")
    two = figure_record.Figure("k", 1, (0, 0, 1, 1), "second", "chart", "claude-opus-5-5")
    assert len({figure_record.fold("abc", [one]), figure_record.fold("abc", [two]), "abc"}) == 3


class Words:
    def __init__(self, n):
        self.ids = list(range(n))


class Tokenizer:
    def encode(self, text):
        return Words(len(text.split()))


def test_a_chunk_per_figure_boxed_in_unrotated_space(figure_record, corpus, tmp_path):
    record(figure_record, corpus, tmp_path, [(1, "10,20,110,220"), (2, "0,0,100,50")], [
        (0, "yes", "upright figure", "claude-sonnet-5"),
        (1, "yes", "sideways figure here", "claude-sonnet-5"),
    ])
    figures = figure_record.described(corpus, tmp_path)["a.pdf"]
    entries = figure_record.chunk_entries(corpus / "a.pdf", figures, Tokenizer(), start=7)
    assert [e["i"] for e in entries] == [7, 8]
    assert entries[0]["boxes"] == {"1": [[10, 20, 110, 220]]}
    assert entries[0]["pages"] == [1] and entries[0]["n_tokens"] == 2
    assert entries[0]["figure"] == {"model": "claude-sonnet-5", "kind": "flowchart",
                                    "language": "fr"}
    # Page 2 displays 800 wide and 600 tall; its top-left 100 x 50 corner is, unrotated,
    # a 50 x 100 strip along the page's left edge at the bottom of the 600 x 800 box.
    [[x0, y0, x1, y1]] = entries[1]["boxes"]["2"]
    assert (x1 - x0, y1 - y0) == (50, 100)
    assert 0 <= x0 and x1 <= 600 and 0 <= y0 and y1 <= 800
