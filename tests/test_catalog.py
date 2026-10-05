"""The corpus catalog's formatting helpers.

The tracked CATALOG.md itself (well formed, anchors that resolve) is checked by the
corpus's own tests, since it belongs to the corpus.

Written by Claude Code (Opus 5).
"""

from __future__ import annotations


def test_anchor_follows_github_slugs(catalog):
    assert catalog.anchor("INSEE") == "insee"
    assert catalog.anchor("Institut national de la statistique") == "institut-national-de-la-statistique"
    assert catalog.anchor("Questions?") == "questions"
    assert catalog.anchor("Des questions ?") == "des-questions-"


def test_human_size(catalog):
    assert catalog.human_size(1_200_000) == "1.2 MB"
    assert catalog.human_size(12_345) == "12 kB"
    assert catalog.human_size(0) == "0 kB"


def test_cell_cannot_break_the_table(catalog):
    """A pipe in a title would end the cell and shift every column after it."""
    assert catalog.cell("Depression | a review") == r"Depression \| a review"
    assert catalog.cell("two\nlines") == "two lines"
    assert catalog.cell("") == "-"
    assert catalog.cell("  ") == "-"


def test_multi(catalog):
    assert catalog.multi("recommandation;synthese") == "recommandation, synthese"
    assert catalog.multi("") == "-"
