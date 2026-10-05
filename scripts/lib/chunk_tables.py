"""Table detection and serialisation for chunking.

A table read line by line loses which value belongs to which column, so a
detected table is replaced by rows that each name their columns. Two detectors
feed this: pymupdf's ruled-line `find_tables`, and the pymupdf-layout model's
grids, cached per PDF under LAYOUT_CACHE (`LayoutGrids`). The rest decides
which detections are tables at all (`is_tabular`, `prose_columns`,
`contents_listing`, `runs_across_rows`) and carries a header across a page
break (`TableCarry`, `resolve_headers`). `apply_tables` is the entry point that
row extraction calls once per page.

Written by Claude Code (Opus 5.5), split out of chunk.py.
"""

from __future__ import annotations

import bisect
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from lib import atomic, hashing
from lib.chunk_row import Row, median
from lib.chunk_text import SENTENCE_FINAL, clean_text, looks_french
from lib.chunk_glyphs import (
    overlap_share, page_words, reread_symbols, symbol_readings, upright_rect,
)


# How many line-like drawings a page must carry before it is worth asking pymupdf to
# look for a table on it. find_tables costs about 0.14s per page against 0.004s for
# the text extraction, 35 times the price of everything else this script does, and
# most pages of a guideline are prose. Counting ruled lines first costs 0.003s per
# page and, measured on the three most table-heavy documents in the corpus, misses
# none of the tables at a threshold of 2 while skipping the scan on 39% to 74% of the
# prose pages. The threshold agrees with what the detector can find anyway: its
# default strategy looks for lines, so a table drawn with no rules at all is invisible
# to both.
TABLE_MIN_RULES = 2

# A "table" of one column or one row is a layout accident: a boxed paragraph, a
# sidebar, a single ruled line under a title. Serialising those as tables would turn
# ordinary prose into "Tableau (1 ligne, 1 colonne)".
TABLE_MIN_COLS = 2
TABLE_MIN_ROWS = 2

# What separates a table from a page whose ruling happens to look like one. The
# detector is generous: on the opioid argumentaire it reports 7 to 13 "columns" over
# pages of ordinary prose, with one cell filled per row and the rest empty, because a
# boxed paragraph and a few rules are enough for it. Serialising those would cut
# sentences into cells and put semicolons through them. A real table has most of its
# grid filled and most of its rows holding more than one value; the agency dosage tables
# that motivated it score 0.8 and 1.0 on these two, the phantom ones 0.14 and 0.0.
TABLE_MIN_FILL = 0.4
TABLE_MIN_MULTI_CELL_ROWS = 0.6

# A detected header is used for the column names only when it looks like one. pymupdf
# often reports the table's title block as the header, with the other cells None, and
# naming a column "TROUBLE ANXIEUX GENERALISE(TAG)\nPosologie Fourchette" in front of
# every value would poison the text far worse than positional cells do.
TABLE_NAME_MAX_CHARS = 60

# A table that runs off the bottom of one page and picks up at the top of the next is
# two tables to the detector, and the second one almost never has a header: the column
# names were printed once, on the page before. Serialised on its own it comes out as
# "Escitalopram ; 10 ; 5-20." with nothing saying which number is the starting dose,
# which is the failure that reading tables cell by cell exists to prevent in the first
# place. These two fractions of the page height say what "runs off the bottom" and
# "picks up at the top" mean. They are loose because a continuation normally sits just
# under the running head and the fragment before it normally reaches into the footer
# margin, and they are not the only test: see apply_tables for the rest.
TABLE_CONTINUATION_STARTS_ABOVE = 0.30
TABLE_CONTINUATION_ENDS_BELOW = 0.70
# How far the two fragments' left and right edges may differ, as a fraction of the
# page width, before they are taken for different tables that merely sit at a page
# boundary. Column names attached to the wrong table are worse than no column names,
# so this is the guard that stops a run of unrelated ruled boxes down a document from
# inheriting each other's headers.
TABLE_CONTINUATION_EDGE_SLACK = 0.05

# Where the layout model's table grids are kept, one JSON per served PDF, keyed by the
# file's sha256. The ruled-line detector above cannot see a table drawn with rules
# only under its header, which is how some guidelines print their dosing tables: one such page
# came out as nine columns read in a zigzag. The
# layout model (pymupdf-layout) predicts the table's box AND its row and column cuts,
# at about 1.2s a page against 0.14s for find_tables, and chunking runs once per
# chunking strategy, so the grids are computed once per file and reused. Untracked
# like everything under data/private/: nothing in it is ever shipped.
LAYOUT_CACHE = Path("data/private/layout")

# Two words on one printed line, in different grid columns, with a gap under this
# many line heights between them are one cell spanning columns rather than two cells:
# a section title across the table ("Second-generation biofuels") would
# otherwise read "Second-generation" in one column and "biofuels" in another.
# A word space is about a quarter of a line height, a column gutter well over one.
TABLE_SPAN_GAP = 0.6

# A layout "table" whose every column holds prose is two columns of running text:
# the model sees a reference list printed in two columns as a 38-row grid, and
# serialising it interleaves the two columns row by row. A real table has at least
# one column of short values (a drug, a criterion number, a dose), so a table where
# the median filled cell of EVERY column runs past this many words is refused.
TABLE_PROSE_WORDS = 8

# A grid whose rows a paragraph runs straight through: the model drew one table
# over a table AND the prose column printed beside it, and serialising it wove
# "Ziprasidone ; A slower approach to titration is to continue the first." out of a
# drug list and a sentence about something else. The test is the one thing a real
# table never does: a cell's text carrying on, mid-sentence and line to line within
# one paragraph of the PDF, into the row where the first column starts a new entry.
# Refused at TABLE_RUN_ON_MIN such crossings making up TABLE_RUN_ON_SHARE of the new
# entries. Over the corpus's 3,459 grids that refuses nine: six prose spills (two
# abstracts beside their "Article history" column, sidebars, this article) and three
# tables whose predicted rows cut through their own cells, where every pairing of a
# value with its row is already wrong. It misses at least one spill, a page of
# two-column criteria in WPS-23-4. Two crossings would add eleven small tables,
# where the evidence is too thin to call.
# A cell counts only at TABLE_RUN_ON_WORDS words or more on both sides of the cut,
# because short lowercase values ("modéré", "oui", "méthylphénidate") also neither
# end a sentence nor start one.
TABLE_RUN_ON_MIN = 3
TABLE_RUN_ON_SHARE = 0.5
TABLE_RUN_ON_WORDS = 3

# A table of contents is a grid to the layout model too: a title, a row of dot
# leaders, a page number, forty times over. Read out cell by cell it is forty
# "ÉQUIPE...... ; 3" fragments that retrieve on nothing, so a grid where at least
# this share of the filled rows carries a run of leaders is refused.
TABLE_LEADER_ROWS = 0.5
TABLE_LEADER = re.compile(r"(?:\.\s?){5,}|…{3,}")


def rule_count(page: pymupdf.Page) -> int:
    """How many ruled lines a page draws.

    Parameters
    ----------
    page : pymupdf.Page
        The page to inspect.

    Returns
    -------
    int
        Drawings that are long and thin, horizontally or vertically.

    Notes
    -----
    This is a filter in front of `find_tables`, not a table detector: it answers
    "could there be a ruled table here" for a thirtieth of the price. The 30pt and
    20pt minimums keep underlined words and list bullets out of the count.
    """
    found = 0
    for drawing in page.get_drawings():
        box = drawing["rect"]
        if (box.width > 30 and box.height < 3) or (box.height > 20 and box.width < 3):
            found += 1
    return found


def is_tabular(data: list[list[str | None]], *, columns: int) -> bool:
    """Whether extracted cells look like a table rather than like boxed prose.

    Parameters
    ----------
    data : list of list of str or None
        The table's cells, as `Table.extract` returns them.
    columns : int
        How many columns the detector claims.

    Returns
    -------
    bool
        True when the grid is filled enough, and enough of its rows hold more than
        one value, for reading it out cell by cell to mean anything.
    """
    if not data or columns < TABLE_MIN_COLS:
        return False
    filled = [sum(1 for cell in row if (cell or "").strip()) for row in data]
    cells = sum(filled)
    if cells / (len(data) * columns) < TABLE_MIN_FILL:
        return False
    multi = sum(1 for count in filled if count > 1)
    return multi / len(data) >= TABLE_MIN_MULTI_CELL_ROWS


def table_names(table: object, data: list[list[str | None]]) -> tuple[list[str], int]:
    """Column names of a detected table, and how many data rows they consumed.

    Parameters
    ----------
    table : pymupdf.table.Table
        A table as `find_tables` reports it.
    data : list of list of str or None
        The table's cells, as `Table.extract` returns them.

    Returns
    -------
    tuple of (list of str, int)
        One name per column and 0 when the detector's own header is usable, the
        first row and 1 when that row looks like a header instead, or ([], 0) when
        neither does.

    Notes
    -----
    Two sources because the detector's header is unreliable on exactly the tables
    that need names most. On some agency dosage tables it reports the table's title
    block as the header, with every other cell None; on 145 of the tables in one
    long background report it reports nothing usable at all. A first row whose cells
    are all short and all filled is what a header looks like, and reading it as one
    puts "colonnes : DCI ; Posologie initiale (mg/j) ; Fourchette posologique
    (mg/j)" in the caption instead of a data row "DCI ; Posologie initiale ...".
    """
    def usable(names: list[str]) -> bool:
        # The first name may be blank: a stub column (the drug names down the left)
        # is routinely printed with no heading, as in some guidelines' dosing tables. Only
        # when there are other names to carry the header, and none of them holds a
        # digit, because a data row that wrapped ("", "2 weeks", "2-4 weeks") also
        # opens on an empty cell and must not be eaten as the column names.
        stub = (not names[0] and table.col_count >= 3
                and not any(ch.isdigit() for name in names for ch in name)) if names else False
        return (len(names) == table.col_count and all(names[1:]) and (names[0] or stub)
                and all(len(name) <= TABLE_NAME_MAX_CHARS for name in names))

    header = getattr(table, "header", None)
    names = [clean_text((name or "").replace("\n", " ")) for name in getattr(header, "names", [])]
    if usable(names):
        # `header.external` says whether the header sits above the table body. When it is
        # False the header IS the first row of the body, and `Table.extract` hands it back
        # as data, so it has to be consumed or the table reads "Recommandation :
        # Recommandation ; Traitement : Traitement" before its real first row. 341 of the
        # 442 named tables in the corpus were doing exactly that.
        return names, 0 if getattr(header, "external", True) else 1
    first = [clean_text((cell or "").replace("\n", " ")) for cell in (data[0] if data else [])]
    if usable(first):
        return first, 1
    return [], 0


@dataclass(frozen=True)
class TableCarry:
    """What one page's last table offers the next page's first one.

    Attributes
    ----------
    names : list of str
        The column names in force at the bottom of the page, which may themselves
        have been inherited: a table spanning four pages passes one header along all
        of them.
    columns : int
        Column count, which a continuation has to match.
    left, right : float
        The fragment's horizontal extent in PDF points. Two fragments of one table
        are printed in the same column positions; two unrelated ruled boxes at a
        page boundary usually are not.
    """

    names: list[str]
    columns: int
    left: float
    right: float


@dataclass
class Detected:
    """One accepted table, with its geometry already in the rows' coordinates.

    Attributes
    ----------
    table
        The table as `find_tables` reports it, read for its counts and its header
        and NEVER for its boxes: `table.bbox` is in the page's rotated space.
    data
        Its cells, already extracted, so the caller's checks and the serialiser
        read the same values.
    box
        `table.bbox`, upright. This is the one that may meet a row's bbox.
    row_boxes
        One upright box per table row, in the same order.

    Notes
    -----
    This type exists so that the derotation happens once, where the page is in
    hand, and cannot be forgotten afterwards. Everything downstream reads `box`
    and `row_boxes` and never learns that rotation exists.
    """

    table: object
    data: list[list[str | None]]
    box: pymupdf.Rect
    row_boxes: list[pymupdf.Rect]


def serialise_table(found: Detected, *, page: int,
                    french: bool, size: float,
                    header: tuple[list[str], int] | None = None,
                    continued: bool = False) -> list[Row]:
    """Turn a detected table into rows of readable text.

    Parameters
    ----------
    found : Detected
        A table, its cells, and its boxes in the rows' coordinate system.
    page : int
        Zero-based page index, for the rows' provenance.
    french : bool
        Whether to describe the table in French.
    size : float
        Font size to claim, so the result is not mistaken for a heading.
    header : tuple of (list of str, int), optional
        Column names and how many data rows they consume, already resolved. The
        default, None, works them out with `table_names`. `apply_tables` passes them
        in because it is the only place that knows whether the page before offered a
        header this table should inherit.
    continued : bool, optional
        Whether these names were inherited from the previous page, which the caption
        says out loud so a reader of the chunk is not told the table starts here.

    Returns
    -------
    list of Row
        A caption row followed by one row per table row, each carrying the bbox of
        what it describes. Empty when the table has nothing in it.

    Notes
    -----
    One row per table row rather than one blob per table, for three reasons. The
    packer can then cut a long table between its rows instead of emitting a single
    oversized chunk; each row ends in a full stop, so those cuts are clean by the
    same test as prose; and each row keeps its own bbox, so the viewer highlights
    the lines a result actually came from instead of the whole table.

    A row whose cells are all empty but one is a subtitle spanning the table, which
    is what some agency tables use to separate sections ("ÉNERGIES RENOUVELABLES"), and it
    is emitted as the text it is rather than as a one-cell table row.
    """
    table, data = found.table, found.data
    names, consumed = table_names(table, data) if header is None else header
    if continued:
        caption = (f"Suite du tableau de la page precedente, {table.row_count} lignes "
                   f"et {table.col_count} colonnes"
                   if french else
                   f"Continuation of the table on the previous page, {table.row_count} "
                   f"rows and {table.col_count} columns")
    else:
        caption = (f"Tableau de {table.row_count} lignes et {table.col_count} colonnes"
                   if french else
                   f"Table of {table.row_count} rows and {table.col_count} columns")
    if names:
        joined = " ; ".join(name for name in names if name)
        caption += (f", colonnes : {joined}." if french else f", columns: {joined}.")
    else:
        caption += "."
    rows = [Row(page=page, bbox=tuple(found.box), text=caption, size=size)]
    for index, cells in enumerate(data):
        if index < consumed:
            # Read as the column names above, and repeating it as a data row would
            # put the header in the text twice.
            continue
        values = [clean_text((cell or "").replace("\n", " ")) for cell in cells]
        filled = [(position, value) for position, value in enumerate(values) if value]
        if not filled:
            continue
        if len(filled) == 1:
            # A subtitle spanning the whole table, not a row of one cell.
            text = filled[0][1]
            if not SENTENCE_FINAL.search(text):
                text += "."
        else:
            # Unlabelled: the caption names the columns once. A label before every
            # cell spent a 256-token chunk on the same few words, and on the one
            # table measured it cost the passage 0.066 of cosine to its own
            # question (DESIGN.md, chunk format 19).
            text = " ; ".join(value for _, value in filled)
        if len(filled) > 1 and not text.endswith("."):
            # A last cell that already ends its sentence keeps its own full stop
            # rather than gaining a second one ("are needed..").
            text += "."
        box = (found.row_boxes[index] if index < len(found.row_boxes) else found.box)
        rows.append(Row(page=page, bbox=tuple(box), text=text, size=size))
    return rows


@dataclass
class GridTable:
    """A table the layout model found, standing in for a `find_tables` one.

    Carries only what `serialise_table`, `table_names` and `resolve_headers` read of
    a table: its counts. It has no `header`, so `table_names` falls back to reading
    the first row, which on the tables this path exists for is where the names are.
    """

    row_count: int
    col_count: int


def grid_cells(words: list[tuple], box: tuple[float, float, float, float],
               row_cuts: list[float], col_cuts: list[float]
               ) -> tuple[list[list[str]], list[pymupdf.Rect], dict]:
    """Fill a predicted table grid with the page's words.

    Parameters
    ----------
    words : list of tuple
        `(x0, y0, x1, y1, text, block, line, word)` as `get_text("words")` gives
        them, in the SAME frame as the grid: the page as it is displayed.
    box : tuple of float
        The table's box.
    row_cuts, col_cuts : list of float
        The interior cuts between rows (y) and between columns (x), absolute, in
        increasing order. n cuts make n + 1 rows or columns.

    Returns
    -------
    tuple of (list of list of str, list of pymupdf.Rect, dict)
        The cells row by row, one box per row spanning the table's width, and for
        each filled cell `(row, column)` the `(block, line)` its text starts and
        ends on, which is what `runs_across_rows` reads.

    Notes
    -----
    Whole words are placed by their centre, never characters by their extent, which
    is the difference from `Table.extract`: clipping characters to a cell cut "For
    each" into "F" and "or each" wherever the grid ran through a word. A printed line
    crossing a column cut with only a word space in the gap is one spanning cell
    (see TABLE_SPAN_GAP), and its text goes to the leftmost column it touches.

    Examples
    --------
    >>> words = [(0, 0, 20, 10, "Fuel", 0, 0, 0), (50, 0, 70, 10, "Rate", 1, 0, 0),
    ...          (0, 20, 20, 30, "Bioethanol", 2, 0, 0), (50, 20, 60, 30, "10", 3, 0, 0),
    ...          (0, 40, 30, 50, "Second-generation", 4, 0, 0),
    ...          (32, 40, 60, 50, "biofuels", 4, 0, 1)]
    >>> data, boxes, _ends = grid_cells(words, (0, 0, 80, 50), [15, 35], [40])
    >>> data
    [['Fuel', 'Rate'], ['Bioethanol', '10'], ['Second-generation biofuels', '']]
    """
    x0, y0, x1, y1 = box
    placed: dict[tuple[int, int], list[tuple]] = {}
    lines: dict[tuple[int, int, int], list[tuple[int, tuple]]] = {}
    for word in words:
        cx, cy = (word[0] + word[2]) / 2, (word[1] + word[3]) / 2
        if not (x0 <= cx <= x1 and y0 <= cy <= y1):
            continue
        row, column = bisect.bisect(row_cuts, cy), bisect.bisect(col_cuts, cx)
        placed.setdefault((row, column), []).append(word)
        lines.setdefault((word[5], word[6], row), []).append((column, word))
    owner: dict[tuple[int, int], int] = {}
    for (_block, _line, row), members in lines.items():
        members.sort(key=lambda member: member[1][0])
        for (left, a), (right, b) in zip(members, members[1:]):
            if left != right and b[0] - a[2] < TABLE_SPAN_GAP * (a[3] - a[1]):
                root = owner.get((row, left), left)
                for column in range(left, right + 1):
                    owner[(row, column)] = root
    data = [["" for _ in range(len(col_cuts) + 1)] for _ in range(len(row_cuts) + 1)]
    ends: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]] = {}
    for (row, column), found in sorted(placed.items()):
        target = owner.get((row, column), column)
        found = sorted(found, key=lambda w: (w[5], w[6], w[7]))
        text = " ".join(word[4] for word in found)
        data[row][target] = f"{data[row][target]} {text}".strip()
        first, last = found[0][5:7], found[-1][5:7]
        if (row, target) in ends:
            first = min(first, ends[(row, target)][0])
            last = max(last, ends[(row, target)][1])
        ends[(row, target)] = (tuple(first), tuple(last))
    edges = [y0, *row_cuts, y1]
    boxes = [pymupdf.Rect(x0, edges[i], x1, edges[i + 1]) for i in range(len(edges) - 1)]
    return data, boxes, ends


def runs_across_rows(data: list[list[str]], ends: dict) -> bool:
    """Whether a paragraph runs on through the rows where the grid's entries start.

    Parameters
    ----------
    data : list of list of str
        The cells, as `grid_cells` returns them.
    ends : dict
        `(row, column)` to the `(block, line)` a cell's text starts and ends on, the
        third thing `grid_cells` returns.

    Returns
    -------
    bool
        True when at least TABLE_RUN_ON_MIN of the rows that open a new entry in the
        first column, and TABLE_RUN_ON_SHARE of them, are rows some other column's
        text flows into from the row above without a break: the upper cell does not
        end a sentence, the lower one starts in lowercase, both carry
        TABLE_RUN_ON_WORDS words or more, and the lower one opens on the very next
        line of the same text block the upper one closed on.

    Notes
    -----
    A new entry is a filled first cell under an empty one, a finished one, or one
    it does not continue in lowercase. Wrapped cells do not trip it, since a cell
    that wraps onto the next grid row does so beside an EMPTY first cell.
    """
    def finished(text: str) -> bool:
        return bool(SENTENCE_FINAL.search(text)) or text.rstrip().endswith((":", ";"))

    first = [row[0].strip() for row in data]
    entries = [i for i in range(1, len(data))
               if first[i] and (not first[i - 1] or finished(first[i - 1])
                                or not first[i][0].islower())]
    if len(entries) < TABLE_RUN_ON_MIN:
        return False
    for column in range(1, len(data[0])):
        crossings = 0
        for i in entries:
            above, below = data[i - 1][column].strip(), data[i][column].strip()
            if (len(above.split()) < TABLE_RUN_ON_WORDS
                    or len(below.split()) < TABLE_RUN_ON_WORDS
                    or finished(above) or not below[0].islower()):
                continue
            upper, lower = ends.get((i - 1, column)), ends.get((i, column))
            if (upper and lower and lower[0][0] == upper[1][0]
                    and lower[0][1] == upper[1][1] + 1):
                crossings += 1
        if crossings >= TABLE_RUN_ON_MIN and crossings >= TABLE_RUN_ON_SHARE * len(entries):
            return True
    return False


def prose_columns(data: list[list[str]]) -> bool:
    """Whether every column of a grid holds running text rather than values.

    Parameters
    ----------
    data : list of list of str
        The cells, as `grid_cells` returns them.

    Returns
    -------
    bool
        True when the median filled cell of every column runs past
        TABLE_PROSE_WORDS words, which is two columns of prose, not a table.
    """
    for column in zip(*data):
        lengths = [len(cell.split()) for cell in column if cell]
        if lengths and median(lengths) <= TABLE_PROSE_WORDS:
            return False
    return True


def contents_listing(data: list[list[str]]) -> bool:
    """Whether a grid is a table of contents, told by its dot leaders.

    Parameters
    ----------
    data : list of list of str
        The cells, as `grid_cells` returns them.

    Returns
    -------
    bool
        True when at least TABLE_LEADER_ROWS of the rows holding any text carry
        a run of leaders, as in ``["INTRODUCTION.......", "7"]``.
    """
    filled = [row for row in data if any(row)]
    leaders = sum(1 for row in filled if any(TABLE_LEADER.search(cell) for cell in row))
    return bool(filled) and leaders / len(filled) >= TABLE_LEADER_ROWS


def layout_version() -> str:
    """The installed pymupdf-layout, whose model decides every grid it predicts."""
    from importlib.metadata import version
    return version("pymupdf-layout")


class LayoutGrids:
    """The layout model's table grids for one PDF, cached on disk.

    Parameters
    ----------
    pdf : Path
        The file the chunker reads, which for a scanned document is its OCR
        derivative: the grid has to describe the text layer the words come from.
    cache_dir : Path
        Where the per-file JSON lives.

    Notes
    -----
    Grids are stored in the frame of the page as DISPLAYED, the only one the model
    can read: on a page with /Rotate the text runs sideways in the unrotated frame
    and the model returns nonsense, so inference runs on a copy of the page with its
    rotation removed, which is what pymupdf4llm does too. The caller maps the boxes
    back into the unrotated frame every row lives in. A page is only asked about once
    it passed the ruled-line gate in `apply_tables`, so the file holds an entry per
    page that was asked, empty or not, and a page missing from it is one never asked.
    """

    def __init__(self, pdf: Path, cache_dir: Path) -> None:
        self.pdf = pdf
        self.version = layout_version()
        self.path = cache_dir / f"{hashing.file_sha256(pdf)}.json"
        self.pages: dict[str, list] = {}
        self.dirty = False
        self._doc: pymupdf.Document | None = None
        if self.path.exists():
            cached = json.loads(self.path.read_text(encoding="utf-8"))
            if cached.get("layout") == self.version:
                self.pages = cached["pages"]

    def grids(self, index: int) -> list[tuple[list[float], list[float], list[float]]]:
        """`(box, row_cuts, col_cuts)` for every table on one page, displayed frame."""
        key = str(index)
        if key not in self.pages:
            import pymupdf.layout  # noqa: F401  (activates the model; slow, so only here)
            if self._doc is None:
                self._doc = pymupdf.open(self.pdf)
            page = self._doc[index]
            if page.rotation:
                page.remove_rotation()
            page.get_layout(return_raw=True)
            found = []
            for item in page.layout_information or []:
                if item["class_name"] != "table" or not item["table_grid"]:
                    continue
                x0, y0, _x1, _y1 = item["group_bbox"]
                grid = item["table_grid"]
                found.append([[float(v) for v in item["group_bbox"]],
                              [float(y0 + cut) for cut in grid.h_lines],
                              [float(x0 + cut) for cut in grid.v_lines]])
            self.pages[key] = found
            self.dirty = True
        return [tuple(entry) for entry in self.pages[key]]

    def close(self) -> None:
        """Write the cache if anything was computed, and release the copy."""
        if self.dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            atomic.write_text_atomic(self.path, json.dumps({"layout": self.version,
                                                            "pages": self.pages}))
            self.dirty = False
        if self._doc is not None:
            self._doc.close()
            self._doc = None


def apply_tables(page: pymupdf.Page, rows: list[Row], *,
                 carried: TableCarry | None = None,
                 layout: LayoutGrids | None = None,
                 boxes_out: list | None = None) -> tuple[list[Row], int, TableCarry | None]:
    """Replace the rows inside each detected table with the table read out in words.

    Parameters
    ----------
    page : pymupdf.Page
        The page the rows came from.
    rows : list of Row
        The page's rows, in reading order.
    carried : TableCarry, optional
        What the previous page's last table offered, or None when that page ended
        without one. The caller threads this through the document.
    layout : LayoutGrids, optional
        The layout model's grids for this document. A table it finds that the
        ruled-line detector did not is read out too; see LAYOUT_CACHE. None reads
        ruled tables only, which is what this function did before the model.
    boxes_out : list, optional
        Receives ``(page, box)`` for every table read out, 1-based page and upright
        box in whole points, so `figures.py tables` can crop exactly what was read.

    Returns
    -------
    tuple of (list of Row, int, TableCarry or None)
        The rows with each table's lines replaced, how many tables were found, and
        what this page's last table offers the next page, or None when it offers
        nothing (no table, no names, or the last one stopped well above the foot of
        the page).

    Notes
    -----
    A table's lines are extracted left to right, one visual line at a time, so a
    three-column table arrives as "Escitalopram Traitement du trouble anxieux 10"
    with no way to tell which number belongs to which drug. Reading the cells out
    one row at a time, under a caption that names the columns, is the difference
    between a passage that answers "what is the starting dose of escitalopram" and
    one that merely contains the words.
    """
    if rule_count(page) < TABLE_MIN_RULES:
        return rows, 0, None
    # find_tables reports boxes in the page's ROTATED space, and every row bbox on
    # this page is unrotated. The two only agree on a page with no /Rotate, and
    # where they disagree the midpoint test below matches the wrong lines, the
    # continuation heuristics measure against the wrong edge, and the box that
    # reaches the viewer highlights nothing. Derotating here is the one place that
    # has the page, so nothing downstream has to know rotation exists.
    derotate = page.derotation_matrix

    def upright(box) -> pymupdf.Rect:
        """A table box in the same coordinates as the rows around it."""
        return (pymupdf.Rect(box) * derotate).normalize()

    tables = []
    symbols = None
    # use_layout=False keeps this the ruled-line detector whether or not the layout
    # model is loaded in the process: once it is, find_tables defaults to trusting
    # the model's boxes and returns nothing where the model saw no table.
    for table in page.find_tables(use_layout=False).tables:
        if table.col_count < TABLE_MIN_COLS or table.row_count < TABLE_MIN_ROWS:
            continue
        if symbols is None:
            symbols = symbol_readings(page)
        data = reread_symbols(table, table.extract(), symbols, upright)
        if not is_tabular(data, columns=table.col_count):
            continue
        tables.append(Detected(table=table, data=data, box=upright(table.bbox),
                               row_boxes=[upright(r.bbox) for r in table.rows]))
    if layout is not None:
        # Ruled tables first, because that detector's output is what every table
        # measurement in DESIGN.md was taken on; the model adds the tables it could
        # not see. Its grid lives in the displayed frame, like the words it is filled
        # with, and `upright` brings both boxes back to the rows' frame.
        words = None
        for box, row_cuts, col_cuts in layout.grids(page.number):
            where = upright(box)
            if any(overlap_share(where, found.box) > 0.5 for found in tables):
                continue
            if words is None:
                rotate = page.rotation_matrix
                words = [(*(pymupdf.Rect(w[:4]) * rotate).normalize(), *w[4:])
                         for w in page_words(page)]
            data, row_boxes, ends = grid_cells(words, box, row_cuts, col_cuts)
            columns = len(col_cuts) + 1
            if (columns < TABLE_MIN_COLS or len(data) < TABLE_MIN_ROWS
                    or not is_tabular(data, columns=columns) or prose_columns(data)
                    or contents_listing(data) or runs_across_rows(data, ends)):
                continue
            tables.append(Detected(table=GridTable(row_count=len(data), col_count=columns),
                                   data=data, box=where,
                                   row_boxes=[upright(r) for r in row_boxes]))
    if not tables:
        return rows, 0, None
    french = looks_french(" ".join(row.text for row in rows))
    sizes = [row.size for row in rows if row.size]
    size = median(sizes) if sizes else 0.0
    # Sorted down the page so that "the first table" and "the last table" mean what
    # the continuation test needs them to mean, whatever order find_tables reported.
    tables.sort(key=lambda found: found.box.y0)
    page_rect = upright_rect(page)
    headers, continued = resolve_headers(tables, carried=carried, page_rect=page_rect)
    boxes = [found.box for found in tables]
    replaced: list[Row] = []
    done: set[int] = set()
    for row in rows:
        middle = pymupdf.Point((row.bbox[0] + row.bbox[2]) / 2,
                               (row.bbox[1] + row.bbox[3]) / 2)
        inside = next((index for index, box in enumerate(boxes) if middle in box), None)
        if inside is None:
            replaced.append(row)
            continue
        # The table takes the place of its first line, so the page keeps its reading
        # order: everything above the table is already emitted, everything below
        # follows. The other lines of the table are dropped, their content having
        # been read out cell by cell.
        if inside not in done:
            done.add(inside)
            replaced.extend(serialise_table(tables[inside], page=row.page,
                                            french=french, size=size,
                                            header=headers[inside],
                                            continued=continued[inside]))
    if boxes_out is not None:
        boxes_out.extend((page.number + 1, [int(round(v)) for v in tables[index].box])
                         for index in sorted(done))
    last = tables[-1]
    names = headers[-1][0]
    offer = (TableCarry(names=names, columns=last.table.col_count,
                        left=last.box.x0, right=last.box.x1)
             if names and last.box.y1 >= page_rect.height * TABLE_CONTINUATION_ENDS_BELOW
             else None)
    return replaced, len(done), offer


def resolve_headers(tables: list[Detected], *,
                    carried: TableCarry | None,
                    page_rect: object) -> tuple[list[tuple[list[str], int]], list[bool]]:
    """Column names for each table on a page, inheriting across the page break.

    Parameters
    ----------
    tables : list of Detected
        The page's accepted tables, sorted down the page.
    carried : TableCarry or None
        What the previous page's last table offered.
    page_rect : pymupdf.Rect
        The page as `upright_rect` gives it, for the "top of the page" and
        page-width tests. Not `page.rect`, which is a quarter turn out on a
        rotated page while every box here is upright.

    Returns
    -------
    (list of (list of str, int), list of bool)
        Per table, the names and how many data rows they consume, and whether those
        names were inherited rather than read off the page.

    Notes
    -----
    Only the FIRST table on a page can inherit, and only when four things hold at
    once: it has no header of its own printed above its body, the previous page ended
    with a table that reached the bottom band, the column counts match, and the two
    fragments are printed at the same horizontal extent. Wrong column names are a
    worse outcome than none (every value in the table would be labelled with somebody
    else's column), which is what all four guards are paying for.

    What this deliberately does NOT do is join the two fragments into one table. They
    stay two, with two captions, and a query that needs a row from each still gets
    two passages. Joining would mean deciding that two detections on two pages are
    one object, and then holding a partly built table across the page loop; carrying
    the names is what recovers the information that was actually being lost.
    """
    headers: list[tuple[list[str], int]] = []
    continued: list[bool] = []
    for position, found in enumerate(tables):
        names, consumed = table_names(found.table, found.data)
        slack = page_rect.width * TABLE_CONTINUATION_EDGE_SLACK
        # `consumed == 0` with names is the only case where the names were printed
        # ABOVE the body; every other non-empty result is table_names guessing that
        # the first row looks like a header. On a continuation that guess is not just
        # unhelpful, it is destructive: the first row is data, and reading it as
        # column names both loses it and labels every row below it with drug names.
        printed_header = bool(names) and consumed == 0
        inherits = bool(
            position == 0 and not printed_header and carried is not None
            and carried.columns == found.table.col_count
            and found.box.y0 <= page_rect.height * TABLE_CONTINUATION_STARTS_ABOVE
            and abs(carried.left - found.box.x0) <= slack
            and abs(carried.right - found.box.x1) <= slack)
        if inherits:
            # consumed stays 0: the continuation's first row is data, the header
            # having been printed on the page before.
            names, consumed = list(carried.names), 0
        headers.append((names, consumed))
        continued.append(inherits)
    return headers, continued
