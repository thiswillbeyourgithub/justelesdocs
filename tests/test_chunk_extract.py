"""Row extraction in scripts/chunk.py (scripts/lib/chunk_extract.py).

The page-geometry half of chunking: reading a two-column page one column at a time,
merging the lines of one visual row without merging across a gutter or across
sideways text, measuring a flow's right margins, finding running headers and
footers on pages of mixed orientation, and clipping boxes to the page so every
highlight can be drawn.

Written by Claude Code (Opus 5). Split out of test_chunk.py by Claude Code (Opus 5.5), along the lib/chunk_*.py stages.
"""

from __future__ import annotations

from conftest import flow_row, line


def test_a_margin_is_where_many_lines_stop(chunk):
    """A single line reaching past the column (a marginal note, a figure label)
    must not move the margin out and make every ordinary line look short."""
    rows = [flow_row(chunk, 0, 100.0 + 12 * i, 14.0, 379.0 + (i % 3), f"ligne {i}", 0.0)
            for i in range(12)]
    rows.append(flow_row(chunk, 0, 300.0, 400.0, 520.0, "note en marge", 0.0))
    assert chunk.flow_margins(rows) == [381.0]


def test_two_columns_in_one_flow_keep_two_margins(chunk):
    """The case that decided the design: a page whose columns were not separated
    (a full-width table defeats the gutter) must not measure its left column
    against the right column's margin, or every line of it reads as short."""
    left = [flow_row(chunk, 0, 100.0 + 12 * i, 37.0, 335.0, f"gauche {i}", 0.0)
            for i in range(10)]
    right = [flow_row(chunk, 0, 100.0 + 12 * i, 351.0, 558.0, f"droite {i}", 0.0)
             for i in range(10)]
    margins = chunk.flow_margins(left + right)
    assert margins == [335.0, 558.0]
    # A left-column line is measured against 335, not against 558.
    assert chunk.margin_for(320.0, margins) == 335.0
    assert chunk.margin_for(540.0, margins) == 558.0
    # And a line past every margin is measured against nothing.
    assert chunk.margin_for(560.0, margins) == 0.0


def test_a_flow_too_small_to_have_a_margin_has_none(chunk):
    """Which flags nothing, which is the conservative answer."""
    rows = [flow_row(chunk, 0, 100.0, 14.0, 180.0, "un titre court", 0.0)]
    assert chunk.flow_margins(rows) == []
    assert chunk.flow_margins([]) == []
    assert chunk.margin_for(100.0, []) == 0.0


def test_a_few_table_rows_of_similar_length_are_not_a_margin(chunk):
    """A handbook page: three table lines end within a point of each
    other. Read as a margin they would sit in front of the page's real one and
    hide every short line behind it."""
    prose = [flow_row(chunk, 0, 400.0 + 12 * i, 14.0, 379.0 + (i % 3), f"prose {i}", 0.0)
             for i in range(20)]
    table = [flow_row(chunk, 0, 100.0 + 12 * i, 91.0, end, f"item {i}", 0.0)
             for i, end in enumerate([232.5, 233.5, 234.9])]
    margins = chunk.flow_margins(prose + table)
    assert margins == [381.0]
    assert chunk.margin_for(233.5, margins) == 381.0


def column_line(chunk, index: int, text: str, *, left: float, right: float,
                page: int = 0) -> object:
    """One line at a chosen horizontal span, stacked under the previous one.

    The column tests need the x axis, which conftest's `line` fixes at the full
    width of the page, so they build their rows here instead.
    """
    return chunk.Row(page=page, bbox=(left, 100.0 + 12 * index, right,
                                      112.0 + 12 * index), text=text, size=10.0)


def two_column_page(chunk, *, rows_per_column: int = 8) -> list[object]:
    """A page in the layout that used to come out scrambled.

    A4 is 595 points wide: the left column runs 60 to 285, the right 310 to 535,
    with the gutter between them. Every line of the left column shares its height
    with a line of the right one, which is what makes the two of them merge.
    """
    lines = []
    for index in range(rows_per_column):
        lines.append(column_line(chunk, index, f"gauche {index}", left=60.0, right=285.0))
        lines.append(column_line(chunk, index, f"droite {index}", left=310.0, right=535.0))
    return lines


def test_a_two_column_page_is_read_one_column_at_a_time(chunk):
    groups = chunk.reading_groups(two_column_page(chunk), 595.0)
    assert [row.text for group in groups for row in group] == (
        [f"gauche {i}" for i in range(8)] + [f"droite {i}" for i in range(8)])


def test_nothing_merges_across_the_gutter(chunk):
    # The bug this replaces: two lines at the same height, one per column, merged
    # into a single row whose text reads "gauche 0 droite 0".
    groups = chunk.reading_groups(two_column_page(chunk), 595.0)
    merged = []
    for group in groups:
        for row in group:
            if merged and chunk._same_row(merged[-1], row):
                merged[-1] = chunk._merge(merged[-1], row)
            else:
                merged.append(row)
    assert all(" droite " not in row.text for row in merged)
    assert len(merged) == 16


def test_a_full_width_line_divides_the_columns_into_zones(chunk):
    # A heading spanning both columns belongs with what follows it, not at the end
    # of whichever column is flushed next.
    lines = two_column_page(chunk)
    heading = column_line(chunk, 4, "2. Traitements", left=60.0, right=535.0)
    lines.insert(8, heading)   # after four lines of each column
    groups = chunk.reading_groups(lines, 595.0)
    order = [row.text for group in groups for row in group]
    assert order.index("2. Traitements") == 8
    assert order[:8] == [f"gauche {i}" for i in range(4)] + [f"droite {i}" for i in range(4)]
    assert order[9:] == [f"gauche {i}" for i in range(4, 8)] + [f"droite {i}" for i in range(4, 8)]


def test_an_ordinary_page_is_left_in_one_piece(chunk):
    lines = [column_line(chunk, i, f"ligne {i}", left=60.0, right=535.0) for i in range(20)]
    groups = chunk.reading_groups(lines, 595.0)
    assert len(groups) == 1
    assert [row.text for row in groups[0]] == [f"ligne {i}" for i in range(20)]


def test_a_table_row_still_merges(chunk):
    # Two cells of one table row sit far apart with a wide gap between them, and
    # they must still read as one row: the page has no gutter, because the lines
    # above and below it cross the middle.
    lines = [column_line(chunk, i, f"paragraphe {i}", left=60.0, right=535.0) for i in range(10)]
    lines.append(column_line(chunk, 10, "Clozapine", left=60.0, right=200.0))
    lines.append(column_line(chunk, 10, "300 mg", left=400.0, right=500.0))
    groups = chunk.reading_groups(lines, 595.0)
    assert len(groups) == 1
    merged = []
    for row in groups[0]:
        if merged and chunk._same_row(merged[-1], row):
            merged[-1] = chunk._merge(merged[-1], row)
        else:
            merged.append(row)
    assert merged[-1].text == "Clozapine 300 mg"


def test_a_margin_note_is_not_a_column(chunk):
    # Text in the outer margin leaves a gap of its own, and a page split there would
    # move the notes away from the paragraphs they annotate.
    lines = [column_line(chunk, i, f"corps {i}", left=60.0, right=400.0) for i in range(20)]
    lines.append(column_line(chunk, 3, "note", left=460.0, right=535.0))
    lines.append(column_line(chunk, 9, "autre note", left=460.0, right=535.0))
    assert len(chunk.reading_groups(lines, 595.0)) == 1


def test_a_short_page_is_never_split(chunk):
    # Under COLUMN_MIN_LINES per side there is not enough evidence: a title page
    # with a logo on one side and a date on the other is not a two-column page.
    lines = two_column_page(chunk, rows_per_column=3)
    assert len(chunk.reading_groups(lines, 595.0)) == 1


def test_every_line_survives_the_split(chunk):
    # Whatever the grouping decides, the page still holds every line exactly once:
    # a group that drops or duplicates a line would drop or duplicate its boxes.
    lines = two_column_page(chunk)
    lines.insert(6, column_line(chunk, 3, "Tableau 1", left=60.0, right=535.0))
    groups = chunk.reading_groups(lines, 595.0)
    regrouped = [row for group in groups for row in group]
    assert sorted(row.text for row in regrouped) == sorted(row.text for row in lines)
    assert len(regrouped) == len(lines)


def sideways_line(chunk, index: int, text: str, *, top: float, bottom: float) -> object:
    """One line of a table printed sideways: a tall, thin box, read bottom to top."""
    return chunk.Row(page=0, bbox=(72.0 + 14 * index, top, 84.0 + 14 * index, bottom),
                     text=text, size=10.0, vertical=True)


def test_sideways_lines_do_not_all_become_one_row(chunk):
    # Every line of a rotated annex table overlaps every other one vertically, so
    # the old test welded the whole page into a single row holding every cell.
    cells = [sideways_line(chunk, i, f"cellule {i}", top=200.0, bottom=600.0) for i in range(6)]
    assert not any(chunk._same_row(cells[i], cells[i + 1]) for i in range(5))


def test_sideways_lines_sharing_a_column_still_merge(chunk):
    # The same rule as upright text, one quarter turn: these two boxes are the two
    # halves of one sideways line, so they belong together.
    left = sideways_line(chunk, 0, "Échelle", top=200.0, bottom=400.0)
    right = sideways_line(chunk, 0, "de dégradation", top=410.0, bottom=600.0)
    assert chunk._same_row(left, right)
    assert chunk._merge(left, right).text == "Échelle de dégradation"
    assert chunk._merge(left, right).vertical is True


def test_text_running_the_other_way_is_never_the_same_row(chunk):
    upright = line(chunk, 0, "Tableau 12. Échelles")
    sideways = chunk.Row(page=0, bbox=(72.0, 100.0, 84.0, 112.0), text="Score",
                         size=10.0, vertical=True)
    assert not chunk._same_row(upright, sideways)


def test_a_landscape_page_keeps_its_own_margins(chunk):
    # A document that mixes orientations: eight portrait pages 842 tall, two
    # landscape ones 595 tall, with the same footer at the foot of each. Measured
    # against the portrait height, the landscape footer sits nowhere near a margin.
    pages = []
    heights = []
    for index in range(10):
        landscape = index >= 8
        height = 595.0 if landscape else 842.0
        foot = height - 40
        pages.append([
            chunk.Row(page=index, bbox=(72.0, 200.0, 500.0, 214.0),
                      text=f"corps de page {index}", size=10.0),
            chunk.Row(page=index, bbox=(140.0, foot, 458.0, foot + 14),
                      text=f"INSEE / Service des statistiques / Novembre 2014", size=8.0),
        ])
        heights.append(height)
    keys = chunk.boilerplate_keys(pages, heights)
    assert chunk.normalise_key("INSEE / Service des statistiques / Novembre 2014") in keys
    # And the body text, which repeats just as often, is kept: it is not in a margin.
    assert chunk.normalise_key("corps de page 3") not in keys


class FakePage:
    """Just the geometry upright_rect and clip_to_page read."""

    def __init__(self, width: float, height: float, rotation: int = 0) -> None:
        import pymupdf

        self.cropbox = pymupdf.Rect(0, 0, width, height)
        self.rotation = rotation
        # What pymupdf reports: the page as displayed, rotation applied.
        self.rect = (pymupdf.Rect(0, 0, height, width) if rotation in (90, 270)
                     else pymupdf.Rect(0, 0, width, height))


def test_the_upright_page_is_the_one_the_boxes_live_in(chunk):
    """A rotated page's `rect` is a quarter turn away from its own text boxes."""
    assert tuple(chunk.upright_rect(FakePage(595, 842))) == (0, 0, 595, 842)
    turned = FakePage(595, 842, rotation=90)
    assert tuple(turned.rect) == (0, 0, 842, 595), "the fake is not faking it"
    assert tuple(chunk.upright_rect(turned)) == (0, 0, 595, 842)


def row_at(chunk, box):
    return chunk.Row(page=0, bbox=box, text="une ligne de texte", size=10.0)


def test_a_box_that_runs_off_the_page_is_cut_to_it(chunk):
    """Scanned books draw text outside the page; pdf.js cannot draw a highlight there."""
    import pymupdf

    page = pymupdf.Rect(0, 0, 420, 617)
    rows, clipped, dropped = chunk.clip_to_page([row_at(chunk, (74, 195, 550, 244))], page)

    assert (clipped, dropped) == (1, 0)
    assert rows[0].bbox == (74, 195, 420, 244)
    assert rows[0].text == "une ligne de texte", "the text is not the problem"


def test_a_box_inside_the_page_is_left_exactly_alone(chunk):
    import pymupdf

    rows, clipped, dropped = chunk.clip_to_page([row_at(chunk, (10.5, 20.25, 100, 40))],
                                                pymupdf.Rect(0, 0, 420, 617))
    assert (clipped, dropped) == (0, 0)
    assert rows[0].bbox == (10.5, 20.25, 100, 40)


def test_a_row_drawn_entirely_off_the_page_is_dropped(chunk):
    """Its text would be a chunk whose provenance points at blank paper."""
    import pymupdf

    rows, clipped, dropped = chunk.clip_to_page([row_at(chunk, (500, 700, 600, 720))],
                                                pymupdf.Rect(0, 0, 420, 617))
    assert (rows, clipped, dropped) == ([], 0, 1)


def test_a_sliver_left_on_the_page_is_dropped_like_any_unhighlightable_row(chunk):
    """Clipping must not manufacture the hairline boxes MIN_BOX_SIDE exists to refuse."""
    import pymupdf

    box = (419.9, 100, 480, 120)
    rows, _clipped, dropped = chunk.clip_to_page([row_at(chunk, box)],
                                                 pymupdf.Rect(0, 0, 420, 617))
    assert (rows, dropped) == ([], 1)
