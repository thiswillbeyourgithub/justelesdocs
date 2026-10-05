"""Row extraction: from a PDF page to rows in reading order.

`extract_rows` reads every page into `Row`s: it finds the columns of a
two-column page (`reading_groups`, `find_gutter`), merges the lines of one
visual row within a flow (`_same_row`, `_merge`), clips boxes to their page,
hands ruled and layout tables to `chunk_tables`, and records each flow's right
margins (`flow_margins`, `margin_for`) for line-break detection.
`boilerplate_keys` then finds running headers and footers over the whole
document. It is the page-geometry half of chunking; packing never reads a page.

Written by Claude Code (Opus 5.5), split out of chunk.py.
"""

from __future__ import annotations

import re
from dataclasses import replace

import pymupdf

from lib.chunk_row import Row
from lib.chunk_text import GLUE, clean_text
from lib.chunk_glyphs import (
    ROW_MERGE_OVERLAP, continues_word, line_text, reading_blocks, upright_rect,
)
from lib.chunk_tables import LayoutGrids, TableCarry, apply_tables


# Minimum width and height, in PDF points, for a line box to be usable. A box
# thinner than this cannot be highlighted visibly, and in practice such boxes
# are not text at all: OCR reads the vertical rules of a table as pipe
# characters, which arrive as zero-width lines containing "|". The 14 OCR'd
# derivatives carry 1125 of them. They break the "every chunk has a drawable
# box" invariant and add nothing to an embedding.
MIN_BOX_SIDE = 1.0

# Fraction of a page's height counted as margin when deciding whether a
# recurring line is a running header or footer. Body text that happens to
# repeat (a table header inside a long table) sits outside these bands and is
# therefore kept, which is the point of combining position with frequency.
MARGIN_BAND = 0.10

# A line must recur on at least this share of pages to count as boilerplate.
# Set well below 1.0 because front matter, annexes and landscape pages
# routinely break the running header, and well above 0.1 so that a phrase
# genuinely repeated in a few sections is not silently deleted.
BOILERPLATE_PAGE_SHARE = 0.35

# Documents shorter than this are never boilerplate-stripped: with a handful
# of pages, "recurs on a third of pages" is satisfied by ordinary repetition
# and the running header costs almost nothing anyway.
MIN_PAGES_FOR_BOILERPLATE = 6

# --- Column detection -------------------------------------------------------------
#
# A two-column page is the worst thing that happens to text on its way out of a PDF,
# and it breaks two separate things at once. Reading the page top to bottom alternates
# between two unrelated sentences, because the lines of both columns sit at the same
# heights. And two lines at the same height MERGE: _same_row sees a vertical overlap
# and nothing else, so the end of a left-column line is welded to the middle of a
# right-column one ("la gestion des ressour- gestion de l'eau potable :
# résultats"). The same blindness welds a running footer to whatever body text shares
# its height, which then makes the footer's row text near unique on every page and
# hides it from boilerplate_keys, whose whole test is that a line REPEATS.
#
# Both follow from the page's columns being unknown, so they are fixed in one place:
# find the gutter first, then treat each side as its own flow. The gutter is found by
# projection, as a vertical band in the middle of the page that almost no line
# crosses. Every threshold below is a share of the page's own width or of its own line
# count, so a landscape annex and an A5 leaflet are judged by their layout rather than
# by their dimensions.

# How wide a gap must be, as a share of page width, to be a column gutter rather than
# the space between two words or between two cells of the same table row.
GUTTER_MIN_WIDTH = 0.03

# Where a gutter may sit, as a share of page width. A two-column layout splits a page
# near its middle; a band of whitespace at 12% of the width is the indent of a
# bulleted list, and splitting there would put every bullet in its own column.
GUTTER_BAND = (0.30, 0.70)

# Share of a page's lines allowed to cross the gutter. A title, a running header, a
# full-width table and a figure caption all span both columns without making the page
# single-column, so the band is required to be nearly empty rather than empty.
GUTTER_CROSSING_SHARE = 0.10

# What each side has to carry before the page counts as two columns: this many lines,
# and this share of the page's lines. Without both, a page holding one wide paragraph
# and a marginal note is split down the middle, and the note is read as a column.
COLUMN_MIN_LINES = 5
COLUMN_MIN_SHARE = 0.25

# How the right margin of a column is found: lines whose ends are within this many
# points of each other stopped at the same margin, and a margin has to be where at
# least this many lines and this share of the flow's lines stop. Without the share, a
# run of three table rows of similar length is a "margin" of its own, sitting in front
# of the page's real one and hiding every short line behind it.
MARGIN_TOLERANCE = 3.0
MARGIN_MIN_LINES = 4
MARGIN_SHARE = 0.08


def clip_to_page(rows: list[Row], page_rect: pymupdf.Rect) -> tuple[list[Row], int, int]:
    """Cut every row's box down to the page it is drawn on.

    Parameters
    ----------
    rows : list of Row
        The page's rows, boxes upright.
    page_rect : pymupdf.Rect
        The page, from `upright_rect`.

    Returns
    -------
    tuple of (list of Row, int, int)
        The rows, how many boxes were cut, and how many rows were dropped for
        having no visible part at all.

    Notes
    -----
    Scanned books put text outside the page: the OCR layer under a slightly
    oversized scan, a glyph with a bbox 600 points wide. 277 boxes of 1.77 million
    over the corpus, 19 documents, all of them books. A box outside the page is a
    highlight pdf.js draws where nobody can look, on a result the ranking counted
    as a hit, so the reader opens the page and does not see what they searched for.
    Cut to the edge, it lands on the part of the line that IS on the page.

    The count is returned rather than swallowed because clipping is exactly the
    kind of repair that hides the bug it is repairing: the 13888 boxes that came
    from reading table geometry in the page's rotated space would have been clipped
    to nonsense just as quietly. `chunk.py` logs the total per run, so a spike is
    visible without waiting for someone to look at a highlight.
    """
    kept: list[Row] = []
    clipped = dropped = 0
    for row in rows:
        box = pymupdf.Rect(row.bbox) & page_rect
        if tuple(box) == tuple(row.bbox):
            kept.append(row)
            continue
        if box.is_empty or box.width < MIN_BOX_SIDE or box.height < MIN_BOX_SIDE:
            # Nothing of this line is on the page, so nothing can be highlighted
            # for it. Keeping the text would mean a chunk whose provenance points
            # at blank paper; see MIN_BOX_SIDE for the same argument at line level.
            dropped += 1
            continue
        clipped += 1
        kept.append(replace(row, bbox=tuple(box)))
    return kept, clipped, dropped


def extract_rows(doc: pymupdf.Document, *, tables: bool = False,
                 layout: LayoutGrids | None = None, table_boxes: list | None = None
                 ) -> tuple[list[list[Row]], int, int, int]:
    """Extract merged text rows, page by page.

    Parameters
    ----------
    doc : pymupdf.Document
        An open PDF.
    tables : bool, optional
        Detect tables and replace their lines with the cells read out in words.
        Off by default because it costs a table scan per candidate page; see
        `apply_tables`.
    layout : LayoutGrids, optional
        With `tables`, also read the tables the layout model finds.

    Returns
    -------
    tuple of (list of list of Row, int, int, int)
        One list of rows per page in reading order, how many tables were
        serialised, how many boxes were cut down to their page, and how many rows
        were dropped as drawn entirely off it. Pages with no text layer yield an
        empty list, which the caller must treat as a failure rather than as an
        empty page.
    """
    pages: list[list[Row]] = []
    tables_found = 0
    boxes_clipped = 0
    rows_off_page = 0
    # Threaded down the document so a table running over a page break keeps its
    # column names; see resolve_headers. Reset by any page that ends without a
    # table reaching its bottom band, which apply_tables reports as None.
    carried: TableCarry | None = None
    for index, page in enumerate(doc):
        lines: list[Row] = []
        # sort=True asks pymupdf for reading order rather than the PDF's
        # internal drawing order, which is frequently arbitrary. It sorts blocks
        # by position, which on a two-column page interleaves the columns, so the
        # order it gives is only the starting point for reading_groups below.
        for block in reading_blocks(page):
            if block["type"] != 0:  # images carry no text
                continue
            # The line lines[-1] was made from, while it is this block's last one,
            # and its raw text: a line that carries on its word is joined to it.
            previous: dict | None = None
            raw_previous = ""
            for line in block["lines"]:
                raw = line_text(line)
                text = clean_text(raw)
                if not text:
                    continue
                x0, y0, x1, y1 = line["bbox"]
                if x1 - x0 < MIN_BOX_SIDE or y1 - y0 < MIN_BOX_SIDE:
                    previous = None
                    continue  # cannot be highlighted, see MIN_BOX_SIDE
                if previous is not None and continues_word(previous, line):
                    last = lines[-1]
                    raw_previous += GLUE + raw
                    lines[-1] = replace(last, text=clean_text(raw_previous), bbox=(
                        min(last.bbox[0], x0), min(last.bbox[1], y0),
                        max(last.bbox[2], x1), max(last.bbox[3], y1)))
                    previous = line
                    continue
                previous, raw_previous = line, raw
                size = line["spans"][0]["size"] if line["spans"] else 0.0
                # "dir" is the unit vector the text runs along: (1, 0) across the
                # page, (0, +-1) up or down it. Annex tables are routinely printed
                # sideways, and their lines are tall thin boxes that every other
                # line of the page overlaps vertically.
                direction = line.get("dir", (1.0, 0.0))
                vertical = abs(direction[1]) > abs(direction[0])
                lines.append(Row(page=index, bbox=tuple(line["bbox"]), text=text,
                                 size=size, vertical=vertical))
        rows: list[Row] = []
        # Merging happens inside a group, never across one. On a single-column page
        # there is exactly one group holding every line, which is the behaviour this
        # replaces; on a two-column page the groups are the two columns and the
        # full-width lines that divide them, so nothing is welded across the gutter.
        for group in reading_groups(lines, upright_rect(page).width):
            merged: list[Row] = []
            for row in group:
                # Merge with the previous row when they share a visual line,
                # which is how bullets and drop caps arrive.
                if merged and _same_row(merged[-1], row):
                    merged[-1] = _merge(merged[-1], row)
                else:
                    merged.append(row)
            # Here rather than later because this is the last place the COLUMNS are
            # known: after this the rows are one flat list and the right margin of a
            # two-column page's left column is indistinguishable from a short line.
            margins = flow_margins(merged)
            for row in merged:
                row.right_edge = margin_for(row.bbox[2], margins)
            rows.extend(merged)
        if tables:
            rows, found, carried = apply_tables(page, rows, carried=carried,
                                                   layout=layout, boxes_out=table_boxes)
            tables_found += found
        # Last, so that a table box and a line box are cut by the same rule and
        # neither can reach a chunk file describing paper that is not there.
        rows, clipped, dropped = clip_to_page(rows, upright_rect(page))
        boxes_clipped += clipped
        rows_off_page += dropped
        pages.append(rows)
    return pages, tables_found, boxes_clipped, rows_off_page


def reading_groups(lines: list[Row], page_width: float) -> list[list[Row]]:
    """Split a page's lines into flows that may be read, and merged, on their own.

    A single-column page is one group holding every line in the order it arrived.
    A two-column page is a sequence of groups: the lines above the columns, the
    left column, the right column, then whatever a full-width line divides off
    below it, and so on down the page. A full-width line (a title, a section
    heading, a figure caption, a table spanning both columns) is its own group,
    which is what keeps a heading attached to the text it introduces rather than
    landing in whichever column happens to be flushed next.

    Parameters
    ----------
    lines : list of Row
        One page's lines, unmerged, in the order pymupdf gave them.
    page_width : float
        The page's own width, in points. Every threshold is a share of it.

    Returns
    -------
    list of list of Row
        The groups, in reading order. Concatenated they hold every input line
        exactly once.
    """
    # A sideways page has columns too, but they are bands of the image rather than
    # of the text, and finding them means reading the page in a rotated frame that
    # nothing downstream shares (the paragraph gaps, the margin bands and the
    # heading sizes are all measured top to bottom). The merge fix above is what
    # such a page needed; the grouping stays out of it.
    if any(row.vertical for row in lines):
        return [lines]
    gutter = find_gutter(lines, page_width)
    if gutter is None:
        return [lines]
    left_edge, right_edge = gutter
    middle = (left_edge + right_edge) / 2

    def crosses(row: Row) -> bool:
        return row.bbox[0] < left_edge and row.bbox[2] > right_edge

    sides = [row for row in lines if not crosses(row)]
    left = [row for row in sides if (row.bbox[0] + row.bbox[2]) / 2 < middle]
    right = [row for row in sides if (row.bbox[0] + row.bbox[2]) / 2 >= middle]
    # A gutter with almost nothing on one side of it is a margin, not a column: a
    # page of notes in the outer margin, or a table of contents whose page numbers
    # all sit at the right edge. Reading such a page as two columns would move the
    # margin's text away from the text it annotates.
    if (len(left) < COLUMN_MIN_LINES or len(right) < COLUMN_MIN_LINES
            or min(len(left), len(right)) < len(lines) * COLUMN_MIN_SHARE):
        return [lines]

    groups: list[list[Row]] = []
    zone_left: list[Row] = []
    zone_right: list[Row] = []

    def flush() -> None:
        """Emit the columns collected so far, left then right, and start a zone."""
        for column in (zone_left, zone_right):
            if column:
                groups.append(list(column))
        zone_left.clear()
        zone_right.clear()

    for row in sorted(lines, key=lambda r: (r.bbox[1], r.bbox[0])):
        if crosses(row):
            flush()
            groups.append([row])
        elif (row.bbox[0] + row.bbox[2]) / 2 < middle:
            zone_left.append(row)
        else:
            zone_right.append(row)
    flush()
    return groups


def find_gutter(lines: list[Row], page_width: float) -> tuple[float, float] | None:
    """Find the vertical band that separates two columns of text.

    The page's lines are projected onto the x axis and the widest band inside
    GUTTER_BAND that nearly no line covers is returned. "Nearly" rather than "none"
    because a two-column page still has a title and a running header across the top.

    Parameters
    ----------
    lines : list of Row
        One page's lines, unmerged.
    page_width : float
        The page's own width, in points.

    Returns
    -------
    tuple of float, or None
        The gutter's left and right edges, or None when the page reads as one
        column, which is the common case and the safe answer.
    """
    if page_width <= 0 or len(lines) < 2 * COLUMN_MIN_LINES:
        return None
    # 200 bins puts the resolution at about 3pt on A4, finer than the gutter being
    # looked for and coarse enough that the projection stays cheap on a 300-page PDF.
    bins = 200
    step = page_width / bins
    covered = [0] * bins
    for row in lines:
        first = max(0, min(bins - 1, int(row.bbox[0] / step)))
        last = max(0, min(bins - 1, int(row.bbox[2] / step)))
        for index in range(first, last + 1):
            covered[index] += 1
    allowed = len(lines) * GUTTER_CROSSING_SHARE

    low = int(bins * GUTTER_BAND[0])
    high = int(bins * GUTTER_BAND[1])
    best: tuple[int, int] | None = None
    start: int | None = None
    for index in range(low, high + 1):
        if covered[index] <= allowed:
            if start is None:
                start = index
        elif start is not None:
            best = _wider(best, (start, index - 1), bins)
            start = None
    if start is not None:
        best = _wider(best, (start, high), bins)
    if best is None:
        return None
    left_edge, right_edge = best[0] * step, (best[1] + 1) * step
    if right_edge - left_edge < page_width * GUTTER_MIN_WIDTH:
        return None
    return left_edge, right_edge


def _wider(best: tuple[int, int] | None, candidate: tuple[int, int],
           bins: int) -> tuple[int, int]:
    """Keep the wider of two candidate gutters, the more central one on a tie.

    A page can offer several empty bands: the gutter itself, and the ragged right
    edge of a column whose lines all stop short. The widest wins, and when two are
    equally wide the one nearer the middle of the page does, because that is where
    a column gutter is and a coincidence is not.
    """
    if best is None:
        return candidate
    width = lambda band: band[1] - band[0]
    if width(candidate) != width(best):
        return candidate if width(candidate) > width(best) else best
    centre = bins / 2
    offset = lambda band: abs((band[0] + band[1]) / 2 - centre)
    return candidate if offset(candidate) < offset(best) else best


def _same_row(left: Row, right: Row) -> bool:
    """Whether two rows sit on the same visual line.

    Two lines share a visual line when they overlap ACROSS the direction the text
    runs in: for ordinary text that is a vertical overlap, and for the sideways
    text of a rotated annex table it is a horizontal one. Measuring it the same
    way for both was how a sideways table came out as a single row holding every
    cell of the page: its lines are tall, thin boxes that all overlap vertically,
    so each one looked like a continuation of the one before.

    Text running one way is never merged with text running the other, whatever the
    boxes do: a rotated caption beside an upright paragraph is not a line of it.
    """
    if left.vertical != right.vertical:
        return False
    axis = 0 if left.vertical else 1    # x for sideways text, y for upright text
    near, far = max(left.bbox[axis], right.bbox[axis]), min(left.bbox[axis + 2], right.bbox[axis + 2])
    overlap = far - near
    shortest = min(left.bbox[axis + 2] - left.bbox[axis],
                   right.bbox[axis + 2] - right.bbox[axis])
    return shortest > 0 and overlap / shortest > ROW_MERGE_OVERLAP


def _merge(left: Row, right: Row) -> Row:
    """Combine two rows of one visual line into a single row."""
    return Row(
        page=left.page,
        bbox=(
            min(left.bbox[0], right.bbox[0]), min(left.bbox[1], right.bbox[1]),
            max(left.bbox[2], right.bbox[2]), max(left.bbox[3], right.bbox[3]),
        ),
        text=f"{left.text} {right.text}".strip(),
        size=max(left.size, right.size),
        vertical=left.vertical,
        right_edge=left.right_edge,
    )


def boilerplate_keys(pages: list[list[Row]], heights: list[float]) -> set[str]:
    """Identify running headers and footers.

    Combines two weak signals into one reliable one: a line must both recur
    across many pages AND live in a margin band. Frequency alone would delete
    repeated table headers inside long tables; position alone would delete
    the first real line of every page in documents with narrow margins.

    Parameters
    ----------
    pages : list of list of Row
        Rows per page, as returned by `extract_rows`.
    heights : list of float
        Each page's own height, in points. One height for the whole document was
        wrong whenever a document mixes orientations, which annexes do constantly:
        measured against a portrait page, the bottom margin of a landscape one
        starts below its own last line, so its footer was never in a margin at all
        and a 274-page guideline kept its running footer on 120 pages.

    Returns
    -------
    set of str
        Digit-normalised texts to drop. Digits are normalised so that
        "page 37" and "page 38" collapse to one key.
    """
    if len(pages) < MIN_PAGES_FOR_BOILERPLATE:
        return set()
    counts: dict[str, int] = {}
    for rows, page_height in zip(pages, heights):
        top_band = page_height * MARGIN_BAND
        bottom_band = page_height * (1 - MARGIN_BAND)
        # A page contributes each distinct key at most once, so a word
        # repeated many times on one page cannot reach the threshold alone.
        seen = {
            normalise_key(row.text)
            for row in rows
            if row.bbox[1] < top_band or row.bbox[3] > bottom_band
        }
        for key in seen:
            counts[key] = counts.get(key, 0) + 1
    threshold = max(2, int(len(pages) * BOILERPLATE_PAGE_SHARE))
    return {key for key, count in counts.items() if count >= threshold and key}


def normalise_key(text: str) -> str:
    """Reduce a line to a comparison key by collapsing digits."""
    return re.sub(r"\d+", "#", text).strip().lower()


def flow_margins(rows: list[Row]) -> list[float]:
    """Where lines in this flow habitually stop, in PDF points.

    A wrapped block of text stops at the same x on line after line, so a margin is
    a value many lines SHARE, not the widest line or a percentile of them. That
    distinction is the whole reason this is a list: a page whose two columns were
    not separated into two flows (no gutter found, because a full-width table sits
    on it) has two margins in one flow, and measuring its left column against the
    right column's margin would call every line of it short.

    Parameters
    ----------
    rows : list of Row
        The rows of one flow, in any order.

    Returns
    -------
    list of float
        The margins, ascending. Empty when no group of lines is large enough to be
        one, which is the conservative answer: `margin_for` then knows nothing
        about any row and nothing is flagged.
    """
    ends = sorted(row.bbox[2] for row in rows if not row.vertical)
    if not ends:
        return []
    # A margin has to be where a real share of the flow stops, or three table rows
    # of similar length become a margin of their own and hide the page's real one.
    least = max(MARGIN_MIN_LINES, int(len(ends) * MARGIN_SHARE))
    margins: list[float] = []
    group = [ends[0]]
    for end in ends[1:]:
        if end - group[0] <= MARGIN_TOLERANCE:
            group.append(end)
            continue
        if len(group) >= least:
            margins.append(group[-1])
        group = [end]
    if len(group) >= least:
        margins.append(group[-1])
    return margins


def margin_for(x1: float, margins: list[float]) -> float:
    """The margin a line ending at `x1` would have run to, or 0.0 if none does.

    The nearest margin AT OR BEYOND the line's own end: a left-column line is
    measured against the left column's margin and never against the right
    column's, and a line that reaches past every margin (a heading in the gutter,
    a rule caught as text) is measured against nothing at all.
    """
    for margin in margins:
        if margin >= x1:
            return margin
    return 0.0
