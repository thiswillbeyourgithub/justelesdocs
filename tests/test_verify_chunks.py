"""Invariant 6: every highlight box lies inside the page it is drawn on.

The docstring of `verify_chunks.py` has claimed this check since the first version
and could not make it: the chunk file said how big a box was and never how big a
page is. Recording the page sizes (chunk format version 10) is what made it
answerable, and the first answer was 13888 boxes, every one of them on a page with
a /Rotate.

Written by Claude Code (Opus 5).
"""

from __future__ import annotations

import pytest

# One A4 page followed by a landscape one, as `chunk.py` records them: upright,
# whole points, in document order.
SIZES = [[595, 842], [842, 595]]


def test_a_box_on_its_page_passes(verify_chunks):
    assert verify_chunks.box_outside_page([72, 100, 500, 120], "1", SIZES) is None


def test_a_box_over_the_bottom_edge_is_named(verify_chunks):
    problem = verify_chunks.box_outside_page([72, 800, 500, 900], "1", SIZES)
    assert problem and "page 1" in problem and "595x842" in problem


def test_each_page_is_measured_against_its_own_size(verify_chunks):
    """A landscape page in a portrait document, which is where the bug lived."""
    assert verify_chunks.box_outside_page([72, 100, 800, 120], "2", SIZES) is None
    assert verify_chunks.box_outside_page([72, 100, 800, 120], "1", SIZES)


@pytest.mark.parametrize("box", [[-1, 100, 500, 120], [72, -2, 500, 120],
                                 [72, 100, 596, 120], [72, 100, 500, 843]])
def test_a_point_out_is_rounding_and_not_a_failure(verify_chunks, box):
    """Boxes and page sizes are both rounded to whole points on the way in."""
    assert verify_chunks.box_outside_page(box, "1", SIZES) is None


def test_a_page_the_document_does_not_have_is_a_failure_not_a_crash(verify_chunks):
    problem = verify_chunks.box_outside_page([72, 100, 500, 120], "9", SIZES)
    assert problem and "2 pages" in problem


def test_nothing_is_claimed_without_page_sizes(verify_chunks):
    """An old chunk file is reported once, per document, not once per box."""
    assert verify_chunks.box_outside_page([72, 100, 500, 120], "1", []) is None
