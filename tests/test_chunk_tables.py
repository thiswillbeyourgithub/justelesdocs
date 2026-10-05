"""Table detection and serialisation in scripts/chunk.py (scripts/lib/chunk_tables.py).

A table read line by line loses which value belongs to which column, so a detected
table is replaced by rows that each name their columns. These tests cover that
serialisation, a header carried across a page break, which detections are not
tables at all (boxed prose, a contents page, a paragraph running through a grid),
and, on small PDFs made here, the two detectors end to end: ruled tables on a
rotated page and the layout model's grids, answered by a fake so no model loads.

Written by Claude Code (Opus 5). Split out of test_chunk.py by Claude Code (Opus 5.5), along the lib/chunk_*.py stages.
"""

from __future__ import annotations


class FakeTableRow:
    """What pymupdf's TableRow gives this code: a bbox and nothing else it reads."""

    def __init__(self, bbox: tuple[float, float, float, float]) -> None:
        self.bbox = bbox


class FakeHeader:
    def __init__(self, names: list[str | None], *, external: bool = True) -> None:
        self.names = names
        # pymupdf sets external False when the header it reports is the table's own
        # first row, which Table.extract then hands back again as data.
        self.external = external


class FakeTable:
    """A detected table, reduced to what serialise_table and table_names read."""

    def __init__(self, data: list[list[str | None]], *,
                 names: list[str | None] | None = None, external: bool = True) -> None:
        self.row_count = len(data)
        self.col_count = len(data[0])
        self.bbox = (72.0, 100.0, 500.0, 300.0)
        self.rows = [FakeTableRow((72.0, 100.0 + 20 * i, 500.0, 120.0 + 20 * i))
                     for i in range(len(data))]
        self.header = FakeHeader(names if names is not None else [None] * self.col_count,
                                 external=external)


def detected(chunk, data, **kwargs):
    """A FakeTable wrapped the way apply_tables wraps a real one.

    The boxes are already upright here, which on a real page is where the
    derotation has happened: `Detected` is the type that guarantees it.
    """
    table = kwargs.pop("table", None) or FakeTable(data, **kwargs)
    import pymupdf
    return chunk.Detected(table=table, data=data, box=pymupdf.Rect(table.bbox),
                          row_boxes=[pymupdf.Rect(r.bbox) for r in table.rows])


def test_a_table_names_its_columns_once_in_its_caption(chunk):
    data = [["Région", "Débit initial", "Fourchette"],
            ["Bretagne", "10", "5-20"],
            ["Normandie", "75", "75-150"]]
    rows = chunk.serialise_table(detected(chunk, data), page=3, french=True, size=10.0)
    # Caption, then one row per data row, minus the row read as the column names.
    assert len(rows) == 3
    assert rows[0].text.startswith("Tableau de 3 lignes et 3 colonnes, colonnes : Région")
    assert rows[0].text.endswith("colonnes : Région ; Débit initial ; Fourchette.")
    # Not before every cell: a label per cell spent a 256-token chunk on the same
    # few words, and cost one measured passage 0.066 of cosine to its question.
    assert rows[1].text == "Bretagne ; 10 ; 5-20."
    # Every row carries the band of the page it describes, so the viewer can
    # highlight the line a result came from rather than the whole table.
    assert rows[1].bbox == (72.0, 120.0, 500.0, 140.0)
    assert all(row.page == 3 for row in rows)


def test_the_header_row_is_not_repeated_as_the_first_cell_row(chunk):
    # The detector reports the header and extract() returns it again as data, so a
    # table read without consuming it opened on "Recommandation : Recommandation".
    data = [["Recommandation", "Équipement"],
            ["1re intention", "PAC"]]
    rows = chunk.serialise_table(detected(chunk, data,
                                          names=["Recommandation", "Équipement"],
                                          external=False),
                                 page=1, french=True, size=10.0)
    assert len(rows) == 2
    assert rows[1].text == "1re intention ; PAC."


def test_a_header_above_the_table_leaves_every_row_as_data(chunk):
    # The same header, reported as external: the body starts at the first row and
    # nothing may be dropped from it.
    data = [["1re intention", "PAC"],
            ["2e intention", "VMC"]]
    rows = chunk.serialise_table(detected(chunk, data,
                                          names=["Recommandation", "Équipement"]),
                                 page=1, french=True, size=10.0)
    assert len(rows) == 3
    assert rows[1].text == "1re intention ; PAC."


def test_a_table_row_that_spans_the_table_is_a_subtitle_not_a_cell(chunk):
    data = [["HABITAT SOCIAL (HS)", "", ""],
            ["Occitanie", "20", "20-50"],
            ["Bretagne", "10", "5-20"]]
    rows = chunk.serialise_table(detected(chunk, data), page=0, french=True, size=10.0)
    # No full stop added: the closing bracket already ends it, by the same test the
    # packer and chunk_quality.py both use.
    assert rows[1].text == "HABITAT SOCIAL (HS)"
    assert rows[2].text == "Occitanie ; 20 ; 20-50."


class FakeRect:
    """Just the page geometry resolve_headers reads."""

    def __init__(self, width: float = 595.0, height: float = 842.0) -> None:
        self.width = width
        self.height = height


def continuation(chunk, table, *, carried, rect=None):
    """resolve_headers for a single table, since that is the only one that inherits."""
    data = [["x"] * table.col_count]
    headers, flags = chunk.resolve_headers([detected(chunk, data, table=table)],
                                           carried=carried,
                                           page_rect=rect or FakeRect())
    return headers[0], flags[0]


def carry_for(chunk, names, *, columns=3, left=72.0, right=500.0):
    return chunk.TableCarry(names=names, columns=columns, left=left, right=right)


def test_a_table_continued_on_the_next_page_keeps_its_column_names(chunk):
    """The defect, fixed on 2026-09-19: the detector sees two tables, and the second
    one has no header because the column names were printed on the page before. Read
    on its own it came out as "Bretagne ; 10 ; 5-20." with nothing saying which
    number was the base rate, which is the exact failure that reading tables cell
    by cell exists to prevent."""
    table = FakeTable([["Bretagne", "10", "5-20"]])
    (names, consumed), inherited = continuation(
        chunk, table, carried=carry_for(chunk, ["Région", "Débit", "Fourchette"]))
    assert inherited
    assert names == ["Région", "Débit", "Fourchette"]
    # The continuation's first row is data: the header was consumed a page ago.
    assert consumed == 0


def test_a_continuation_says_so_in_its_caption(chunk):
    """A chunk that opens "Tableau de 4 lignes" when the table began two pages back
    tells a reader the wrong thing about what they are looking at."""
    data = [["Normandie", "75", "75-150"]]
    rows = chunk.serialise_table(detected(chunk, data), page=4, french=True, size=10.0,
                                 header=(["Région", "Débit", "Fourchette"], 0),
                                 continued=True)
    assert rows[0].text.startswith("Suite du tableau de la page precedente")
    assert "colonnes : Région" in rows[0].text
    assert rows[1].text == "Normandie ; 75 ; 75-150."


def test_a_table_with_its_own_header_ignores_what_the_page_before_offered(chunk):
    table = FakeTable([["a", "b", "c"]], names=["Station", "Débit", "Plage"])
    (names, _consumed), inherited = continuation(
        chunk, table, carried=carry_for(chunk, ["Région", "Débit", "Fourchette"]))
    assert not inherited
    assert names == ["Station", "Débit", "Plage"]


def test_a_table_of_a_different_shape_does_not_inherit(chunk):
    """Column names attached to the wrong table are worse than no column names: every
    value in it would be labelled with somebody else's column."""
    table = FakeTable([["a", "b", "c"]])
    (names, _consumed), inherited = continuation(
        chunk, table, carried=carry_for(chunk, ["Région", "Débit"], columns=2))
    assert not inherited and names != ["Région", "Débit"]


def test_a_table_printed_elsewhere_on_the_page_does_not_inherit(chunk):
    """Same column count, different column positions: two ruled boxes that happen to
    sit either side of a page break, not one table."""
    table = FakeTable([["a", "b", "c"]])
    (_names, _consumed), inherited = continuation(
        chunk, table, carried=carry_for(chunk, ["Région", "Débit", "Fourchette"],
                                        left=300.0, right=560.0))
    assert not inherited


def test_a_table_starting_halfway_down_the_page_does_not_inherit(chunk):
    """A continuation is printed under the running head. One that starts below the
    middle of the page had something else above it, so it started there."""
    table = FakeTable([["a", "b", "c"]])
    table.bbox = (72.0, 500.0, 500.0, 700.0)
    (_names, _consumed), inherited = continuation(
        chunk, table, carried=carry_for(chunk, ["Région", "Débit", "Fourchette"]))
    assert not inherited


def test_only_the_first_table_on_a_page_can_inherit(chunk):
    """The second table on a page is separated from the page break by the first."""
    first = FakeTable([["a", "b", "c"]])
    second = FakeTable([["d", "e", "f"]])
    second.bbox = (72.0, 400.0, 500.0, 600.0)
    data = [["a", "b", "c"]]
    headers, flags = chunk.resolve_headers(
        [detected(chunk, data, table=first), detected(chunk, data, table=second)],
        carried=carry_for(chunk, ["Région", "Débit", "Plage"]),
        page_rect=FakeRect())
    assert flags == [True, False]
    assert headers[0][0] == ["Région", "Débit", "Plage"]
    assert headers[1][0] != ["Région", "Débit", "Plage"]


def test_boxed_prose_is_not_read_as_a_table(chunk):
    # What the detector reports on a page of a long report: nine columns,
    # one cell filled per row. Serialising it would cut the sentences into cells.
    data = [[""] * 4 + ["une phrase entiere de texte courant"] + [""] * 4
            for _ in range(8)]
    assert not chunk.is_tabular(data, columns=9)


def test_a_filled_grid_is_read_as_a_table(chunk):
    data = [["Région", "Débit"], ["Bretagne", "10"], ["Normandie", "75"]]
    assert chunk.is_tabular(data, columns=2)


def rotated_table_pdf(path):
    """One ruled table on a page with /Rotate 90, the shape that exposed the bug."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    columns, lines = [60, 220, 380, 535], [100, 130, 160, 190]
    for x in columns:
        page.draw_line(pymupdf.Point(x, lines[0]), pymupdf.Point(x, lines[-1]))
    for y in lines:
        page.draw_line(pymupdf.Point(columns[0], y), pymupdf.Point(columns[-1], y))
    cells = [["Région", "Débit", "Fourchette"],
             ["Bretagne", "10 m3", "5-20 m3"],
             ["Normandie", "75 m3", "75-150 m3"]]
    for row, values in enumerate(cells):
        for column, text in enumerate(values):
            page.insert_text(pymupdf.Point(columns[column] + 4, lines[row] + 20),
                             text, fontsize=11)
    page.set_rotation(90)
    doc.save(str(path))
    doc.close()
    return path


def test_a_table_on_a_rotated_page_is_highlighted_where_it_is_printed(chunk, tmp_path):
    """The defect found on 2026-09-19 by the invariant that could not be checked.

    find_tables reports its boxes in the page's ROTATED space and every text line
    on the same page comes out unrotated. 13888 boxes over the corpus described a
    quarter turn away from the text they came from: highlights drawn off the page,
    on results the ranking had counted.
    """
    import pymupdf

    path = rotated_table_pdf(tmp_path / "rotated.pdf")
    with pymupdf.open(path) as doc:
        page = doc[0]
        upright = chunk.upright_rect(page)
        assert tuple(page.rect) == (0, 0, 842, 595), "the fixture is not rotated"
        raw = [pymupdf.Rect(t.bbox) for t in page.find_tables().tables]
        assert raw and raw[0] not in upright, \
            "pymupdf stopped reporting table boxes rotated; this test is now moot"

        pages, serialised, clipped, dropped = chunk.extract_rows(doc, tables=True)

    assert serialised == 1, "the fixture's table is no longer detected"
    assert (clipped, dropped) == (0, 0), "a correct box needed no repair"
    outside = [row.bbox for row in pages[0] if pymupdf.Rect(row.bbox) not in upright]
    assert not outside, f"boxes still in the rotated space: {outside}"


def test_a_table_box_is_recorded_upright_and_survives_the_crop_round_trip(chunk, tmp_path):
    """`table_boxes` is what `figures.py tables` crops, and its description chunk
    must highlight the same place as the table's own rows.

    figures.py turns the upright box into the page as displayed to crop it, and
    figure_record derotates the crop back: on a /Rotate 90 page that round trip is
    the only thing keeping the description's highlight on the table.
    """
    import pymupdf

    path = rotated_table_pdf(tmp_path / "rotated.pdf")
    boxes: list = []
    with pymupdf.open(path) as doc:
        page = doc[0]
        pages, serialised, _c, _d = chunk.extract_rows(doc, tables=True, table_boxes=boxes)
        assert serialised == 1 and len(boxes) == 1
        number, box = boxes[0]
        assert number == 1
        rows = pymupdf.Rect()
        for row in pages[0]:
            rows |= pymupdf.Rect(row.bbox)
        assert pymupdf.Rect(box).intersects(rows), "the box is not where the rows are"
        shown = (pymupdf.Rect(box) * page.rotation_matrix).normalize()
        back = (shown * page.derotation_matrix).normalize()
    assert [round(v) for v in back] == box


class FakeLayout:
    """What LayoutGrids answers, without the model: one grid on page 0."""

    def __init__(self, grids):
        self._grids = grids

    def grids(self, index):
        return self._grids if index == 0 else []


def header_ruled_rotated_pdf(path, *, title=None):
    """A landscape table page from a long report, reduced to its shape.

    A landscape table on a page with /Rotate 90, text upright as displayed, ruled
    only above and under its header, with a blank stub heading and a section title
    running across two columns. find_tables sees no table in it, and before the
    layout model the chunker read it as its lines, columns interleaved. ``title``
    goes above the table, flush with its left edge, as the source document prints "TABLE 8".
    """
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.set_rotation(90)
    shown = page.derotation_matrix  # displayed coordinates -> the page's own
    for y in (90, 113):
        page.draw_line(pymupdf.Point(90, y) * shown, pymupdf.Point(700, y) * shown)
    cells = [(70, [(90, title)] if title else []),
             (105, [(300, "Output"), (500, "Frequency")]),
             (130, [(200, "Second-generation heat pumps")]),
             (155, [(100, "Air-source"), (300, "10 kW"), (500, "Daily")]),
             (180, [(100, "Ground-source"), (300, "5 kW"), (500, "Monthly")])]
    for baseline, row in cells:
        for x, text in row:
            page.insert_text(pymupdf.Point(x, baseline) * shown, text, fontsize=11,
                             rotate=90)
    doc.save(str(path))
    doc.close()
    return path


# The grid the model predicts for that page, in the displayed frame it works in.
HEADER_RULED_GRID = [((90.0, 90.0, 700.0, 190.0), [113.0, 138.0, 163.0], [280.0, 480.0])]


def test_a_table_ruled_only_under_its_header_is_read_from_the_layout_grid(chunk, tmp_path):
    """The defect reported on 2026-09-28 on a landscape table page of a long report."""
    import pymupdf

    path = header_ruled_rotated_pdf(tmp_path / "landscape.pdf")
    with pymupdf.open(path) as doc:
        assert not doc[0].find_tables(use_layout=False).tables, \
            "find_tables now sees this table; the fixture no longer reproduces the bug"
        _pages, before, _c, _d = chunk.extract_rows(doc, tables=True)
        pages, after, clipped, dropped = chunk.extract_rows(
            doc, tables=True, layout=FakeLayout(HEADER_RULED_GRID))
        upright = chunk.upright_rect(doc[0])

    assert (before, after) == (0, 1)
    texts = [row.text for row in pages[0]]
    assert texts[0].startswith("Tableau de 4 lignes et 3 colonnes, colonnes : Output ; Frequency")
    # The section title spanned two grid columns and stays one subtitle; the stub
    # column has no heading.
    assert texts[1:] == ["Second-generation heat pumps.",
                         "Air-source ; 10 kW ; Daily.",
                         "Ground-source ; 5 kW ; Monthly."]
    # Boxes come back in the frame every row lives in, not the displayed one.
    assert (clipped, dropped) == (0, 0)
    assert all(pymupdf.Rect(row.bbox) in upright for row in pages[0])


def test_a_sideways_page_reads_its_title_before_its_table(chunk, tmp_path):
    """On that landscape page "TABLE 8" came out after its table.

    pymupdf sorts blocks by their bottom edge in the UNROTATED page, which on this
    page is the displayed LEFT edge, so the title (from x = 90) sorted after every
    table line (from x = 100).
    """
    import pymupdf

    title = "TABLE 1. Maintenance intervals of two heat pumps"
    path = header_ruled_rotated_pdf(tmp_path / "landscape.pdf", title=title)
    with pymupdf.open(path) as doc:
        pages, found, _c, _d = chunk.extract_rows(
            doc, tables=True, layout=FakeLayout(HEADER_RULED_GRID))

    assert found == 1
    texts = [row.text for row in pages[0]]
    assert texts[0] == title
    assert texts[1].startswith("Table of 4 rows and 3 columns")


def test_a_layout_grid_under_a_ruled_table_does_not_read_it_twice(chunk, tmp_path):
    """The ruled detector keeps a table it found; the model only adds the ones it missed."""
    import pymupdf

    path = rotated_table_pdf(tmp_path / "ruled.pdf")
    with pymupdf.open(path) as doc:
        page = doc[0]
        ruled = pymupdf.Rect(page.find_tables(use_layout=False).tables[0].bbox)
        grid = [(tuple(ruled), [ruled.y0 + 30, ruled.y0 + 60], [ruled.x0 + 160, ruled.x0 + 320])]
        _pages, found, _c, _d = chunk.extract_rows(doc, tables=True, layout=FakeLayout(grid))
    assert found == 1


def test_two_columns_of_prose_are_not_a_table(chunk):
    """A two-column reference list, which the model returns as a 38-row grid."""
    reference = "Durand MA et al (1994) Infrared imaging of the outer walls in cold climates"
    assert chunk.prose_columns([[reference, reference]] * 4)
    assert not chunk.prose_columns([["Criterion 1:", reference]] * 4)


def test_a_table_of_contents_is_not_a_table(chunk):
    """A dot-leader contents page came out as a 40-row, two-column "table"."""
    assert chunk.contents_listing([["ÉQUIPE.............. 3", ""], ["LEXIQUE", "……… 6"],
                                   ["", ""], ["Annexe 1", "40"]])
    assert not chunk.contents_listing([["Air-source", "10 kW"], ["Output...", "5 kW"]])
    assert not chunk.contents_listing([])


def spill_words(prose_block: int, *, entries: list[str], prose: list[str]):
    """Words for a two-column grid, first-column entries on every row and one
    line of `prose` per row in column 1, every prose line in text block
    `prose_block` when it is an int, or each line its own block when None."""
    words = []
    for row, (entry, line) in enumerate(zip(entries, prose)):
        y = 20.0 * row
        words.append((0, y, 40, y + 10, entry, 100 + row, 0, 0))
        block, number = ((prose_block, row) if prose_block is not None else (200 + row, 0))
        for k, word in enumerate(line.split()):
            words.append((60 + 30 * k, y, 85 + 30 * k, y + 10, word, block, number, k))
    return words


SPILL_ENTRIES = ["Ventilation", "Condensation", "Heat loss", "Corrosion", "Noise"]
SPILL_PROSE = ["A slower approach to replacement", "is to continue the first",
               "boiler for a period", "at its usual rate while the",
               "second one is introduced"]


def test_a_paragraph_running_through_a_grid_s_entries_refuses_the_grid(chunk):
    """The layout model drew one table over a table and the prose column beside it,
    and the chunk read "Noise ; A slower approach to replacement is to continue
    the first." (a heating-system switchover guide, page 5)."""
    words = spill_words(7, entries=SPILL_ENTRIES, prose=SPILL_PROSE)
    data, _boxes, ends = chunk.grid_cells(words, (0, 0, 400, 100),
                                          [15, 35, 55, 75], [50])
    assert data[1] == ["Condensation", "is to continue the first"]
    assert chunk.runs_across_rows(data, ends)


def test_the_same_words_as_separate_cells_are_a_table(chunk):
    # Each line its own text block: cells the PDF does not flow into each other.
    words = spill_words(None, entries=SPILL_ENTRIES, prose=SPILL_PROSE)
    data, _boxes, ends = chunk.grid_cells(words, (0, 0, 400, 100),
                                          [15, 35, 55, 75], [50])
    assert not chunk.runs_across_rows(data, ends)


def test_a_cell_wrapping_beside_an_empty_first_cell_is_a_table(chunk):
    entries = ["Ventilation", "", "Heat loss", "", "Corrosion"]
    words = [w for w in spill_words(7, entries=entries, prose=SPILL_PROSE) if w[4]]
    data, _boxes, ends = chunk.grid_cells(words, (0, 0, 400, 100),
                                          [15, 35, 55, 75], [50])
    assert not chunk.runs_across_rows(data, ends)


def test_a_blank_stub_heading_still_names_the_other_columns(chunk):
    data = [["", "Output", "Frequency"], ["Air-source", "10 kW", "Daily"]]
    assert chunk.table_names(FakeTable(data), data) == (["", "Output", "Frequency"], 1)


def test_a_wrapped_data_row_opening_on_a_blank_cell_is_not_a_header(chunk):
    data = [["", "2 weeks", "2-4 weeks"], ["Air-source", "10 kW", "Daily"]]
    assert chunk.table_names(FakeTable(data), data) == ([], 0)


def test_a_cell_that_ends_its_sentence_gets_no_second_full_stop(chunk):
    data = [["Unit", "Comment"], ["Ground-source", "Split loads over 100 kW."]]
    rows = chunk.serialise_table(detected(chunk, data), page=0, french=False, size=10.0)
    assert rows[1].text == "Ground-source ; Split loads over 100 kW."
