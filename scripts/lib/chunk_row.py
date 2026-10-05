"""The row every chunking stage passes along, and the median they all take.

A `Row` is one visual line of a PDF with the page and box it came from. It is
made by extraction, rewritten by the text repairs, cut by table reading and
consumed by packing, so it lives below all of them: every other `chunk_*`
module imports it from here and none of them has to import another stage to
name it. `median` sits here for the same reason, being the one helper the
glyph, table and packing code all lean on.

Written by Claude Code (Opus 5.5), split out of chunk.py.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Row:
    """One visual line of text with its provenance.

    Attributes
    ----------
    page : int
        Zero-based page index. Converted to 1-based only when written out,
        so that the stored value matches what a PDF viewer shows.
    bbox : tuple of float
        (x0, y0, x1, y1) in PDF points, origin top-left, as pymupdf reports.
    text : str
        The row's text, whitespace-normalised.
    size : float
        Font size of the row's first span, used only for heading detection.
    starts_block : bool
        Whether a reader would say a new paragraph, list item or section starts
        here, which is `break_strengths` scoring this row 2 or more. Set by `pack`
        once the scores are computed, and read only by `Chunk.text`, which puts a
        newline in front of such a row and a space in front of every other one.
        False everywhere `--boundaries` is off, where the text joins as it always
        did.
    starts_line : bool
        Whether the line BEFORE this row was ended by the author rather than by
        the right margin, which `line_break_flags` reads off the geometry. Like
        `starts_block` it only ever puts a newline in front of this row; unlike it,
        it says nothing about where a chunk may end, so it changes what a passage
        looks like and not how the corpus is cut up.
    right_edge : float
        The margin this row's line would have run to had it wrapped, in PDF points,
        which is the nearest margin of its flow at or beyond its own end
        (`flow_margins`, `margin_for`). 0.0 when there is none: a row out of a
        serialised table, a flow with no margin to speak of, or a line already past
        every margin. Set by `extract_rows`, where the columns are known, and read
        only by `line_break_flags`.
    vertical : bool
        Whether the line is written up or down the page rather than across it,
        read from pymupdf's writing direction. Sideways tables are common in
        annexes, and on such a page "the same visual line" is a column of the
        image rather than a band of it, so the merge test has to know which way
        the text runs. Everything else in this file reads a page top to bottom
        and is unaffected.
    """

    page: int
    bbox: tuple[float, float, float, float]
    text: str
    size: float
    vertical: bool = False
    starts_block: bool = False
    starts_line: bool = False
    right_edge: float = 0.0


def median(values: list[float]) -> float:
    """Middle value of a list, or 0.0 when it is empty.

    A local definition rather than `statistics.median` so that the empty case is
    a value instead of an exception: several of the inputs here are legitimately
    empty (a one-row document has no pitch to measure) and the callers all want
    "no signal" rather than a raised error.

    Parameters
    ----------
    values : list of float
        Unsorted values.

    Returns
    -------
    float
        The lower of the two middle values for an even-length list, which is
        close enough for a threshold and avoids inventing a value that is not in
        the data.
    """
    if not values:
        return 0.0
    return sorted(values)[len(values) // 2]
