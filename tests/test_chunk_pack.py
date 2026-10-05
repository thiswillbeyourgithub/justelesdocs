"""Packing in scripts/chunk.py (scripts/lib/chunk_pack.py).

The chunker is measured rather than asserted (DESIGN.md holds the sweeps), so what
is worth testing here is the mechanics underneath the measurements: which rows count
as headings, which heading a chunk sits under, where a line break is the author's,
where a chunk may end (section, paragraph, list item, sentence), the page and
section bakes, and the invariant that the whole format depends on, which is that a
chunk's boxes are an exact quotation of the page.

Rows are built by hand rather than read from a PDF: pymupdf's output is the one part
of this file that is not ours to test, and a synthetic page makes the size and gap
signals explicit instead of hoping a real document happens to contain them.

Written by Claude Code (Opus 5). Split out of test_chunk.py by Claude Code (Opus 5.5), along the lib/chunk_*.py stages.
"""

from __future__ import annotations

import pytest

from conftest import WordTokenizer, flow_row, line, load_lib


@pytest.fixture
def rows(chunk):
    """A page of body text with two headings in it.

    Row 0 is a heading in two lines (the wrap case), rows 2-5 its body, row 6 a
    second heading, rows 7-10 its body. The body rows are deliberately long enough
    that a small budget has to cut the second section in two.
    """
    def row(index: int, text: str, size: float) -> object:
        return chunk.Row(page=0, bbox=(72.0, 100.0 + 12 * index, 500.0,
                                       112.0 + 12 * index), text=text, size=size)

    body = "mot " * 20
    return [
        row(0, "5. Prise en charge de la crue fluviale", 14.0),
        row(1, "chez l'habitant", 14.0),
        row(2, body.strip(), 10.0),
        row(3, body.strip(), 10.0),
        row(4, body.strip(), 10.0),
        row(5, body.strip(), 10.0),
        row(6, "6. Tarifications", 14.0),
        row(7, body.strip(), 10.0),
        row(8, body.strip(), 10.0),
        row(9, body.strip(), 10.0),
        row(10, body.strip(), 10.0),
    ]


def test_heading_flags_find_the_big_type(chunk, rows):
    assert chunk.heading_flags(rows) == [
        True, True, False, False, False, False, True, False, False, False, False
    ]


def test_heading_flags_abstain_without_sizes(chunk, rows):
    blank = [chunk.Row(page=0, bbox=r.bbox, text=r.text, size=0.0) for r in rows]
    assert chunk.heading_flags(blank) == [False] * len(blank)


def test_a_wrapped_heading_is_one_run(chunk, rows):
    context = chunk.heading_context(rows)
    # Both lines of the heading see the whole of it, so a chunk starting on either
    # is recognised as already holding it.
    assert context[0] == [0, 1]
    assert context[1] == [0, 1]
    # The body below sits under that run, and the next section under its own.
    assert context[3] == [0, 1]
    assert context[7] == [6]
    assert context[6] == [6]


def test_rows_before_the_first_heading_sit_under_nothing(chunk, rows):
    assert chunk.heading_context(rows[2:])[0] == []


def test_a_continuing_chunk_carries_the_heading(chunk, rows):
    chunks = chunk.pack(rows, WordTokenizer(), target=60, overlap=10,
                        boundaries=True, headings=chunk.HEADING_MAX_TOKENS)
    assert len(chunks) > 2, "the budget should have cut both sections"
    # Every chunk names the section it belongs to, including the ones that start
    # in the middle of it. That is the whole point of the option.
    for c in chunks:
        assert ("crue fluviale" in c.text) or ("Tarifications" in c.text)


def test_the_heading_is_never_repeated(chunk, rows):
    for c in chunk.pack(rows, WordTokenizer(), target=60, overlap=10,
                        boundaries=True, headings=chunk.HEADING_MAX_TOKENS):
        assert c.text.count("Tarifications") <= 1
        assert c.text.count("crue fluviale") <= 1


def test_without_the_option_nothing_is_carried(chunk, rows):
    chunks = chunk.pack(rows, WordTokenizer(), target=60, overlap=10,
                        boundaries=True, headings=0)
    assert all(not c.heading for c in chunks)
    assert any("Tarifications" not in c.text and "crue fluviale" not in c.text
               for c in chunks), "some chunk should be anonymous without the option"


def test_a_carried_heading_is_text_only(chunk, rows):
    """The invariant the stored format rests on: boxes quote the page exactly.

    A heading can be pages behind the passage. Putting it in `rows` would add its
    page to the chunk's provenance, move the page the site opens on, and draw a
    highlight where the passage is not.
    """
    plain = chunk.pack(rows, WordTokenizer(), target=60, overlap=10,
                       boundaries=True, headings=0)
    carried = chunk.pack(rows, WordTokenizer(), target=60, overlap=10,
                         boundaries=True, headings=chunk.HEADING_MAX_TOKENS)
    assert [c.boxes_by_page() for c in plain] == [c.boxes_by_page() for c in carried]
    assert [sorted({r.page for r in c.rows}) for c in plain] == \
           [sorted({r.page for r in c.rows}) for c in carried]


def test_a_chunk_keeps_the_paragraph_breaks_and_reflows_the_rest(chunk):
    """A wrapped paragraph comes back as one line; a list keeps one line per item."""
    made = [chunk.Row(page=0, bbox=(72.0, y, 500.0, y + 12.0), text=text, size=10.0)
            for y, text in [(100.0, "Le traitement de premiere intention repose sur"),
                            (112.0, "une isolation structuree."),
                            (140.0, "- premier point"),
                            (152.0, "- deuxieme point")]]
    packed = chunk.pack(made, WordTokenizer(), target=200, overlap=0,
                        boundaries=True, headings=0)
    assert len(packed) == 1
    lines = packed[0].text.split("\n")
    # The two wrapped rows of one sentence are one line again; each bullet is its own.
    assert lines[0] == "Le traitement de premiere intention repose sur une isolation structuree."
    assert lines[1] == "- premier point"
    assert lines[2] == "- deuxieme point"


def test_a_line_that_stops_short_with_room_to_spare_ended_on_purpose(chunk):
    """The bug: a table of one item per line arrived as one line of item names.

    A handbook page, reduced: rows at x0=91 in a flow whose margin
    is at 380, each stopping 120 points or more short of it. Nothing else about
    them says "list": no bullet, no sentence end, no wider gap, no larger type.
    """
    rows = [
        flow_row(chunk, 0, 100.0, 91.0, 263.8, "Thermoregulation (Chauffe)", 379.8),
        flow_row(chunk, 0, 112.0, 91.0, 233.5, "Chaudiere (Gazole)", 379.8),
        flow_row(chunk, 0, 124.0, 91.0, 222.7, "Plancher (Chaud)", 379.8),
    ]
    assert chunk.line_break_flags(rows) == [False, True, True]


def test_a_ragged_line_that_ran_out_of_room_is_a_wrap(chunk):
    """What keeps ordinary prose out of this: the next word did not fit."""
    rows = [
        flow_row(chunk, 0, 100.0, 14.0, 340.0, "le traitement de premiere intention repose", 379.8),
        flow_row(chunk, 0, 112.0, 14.0, 379.5, "sur une isolation structuree et suivie", 379.8),
    ]
    # "sur" is short, but 40 points of room is less than the word plus its space
    # plus the margin the estimate is given.
    assert chunk.line_break_flags(rows) == [False, False]


def test_the_last_line_of_a_paragraph_ends_a_line(chunk):
    """And a paragraph's short last line is a line its author ended, which is the
    same flag: what follows starts on a line of its own either way."""
    rows = [
        flow_row(chunk, 0, 100.0, 14.0, 215.7, "provoquant son inhibition.", 379.8),
        flow_row(chunk, 0, 112.0, 25.6, 379.5, "Proche du site pour le GABA se trouve un autre site", 379.8),
    ]
    assert chunk.line_break_flags(rows)[1] is True


def test_two_sentences_of_one_line_are_not_two_lines(chunk):
    """`split_sentences` gives the pieces of a line one bbox, so the second piece
    must not be read as a line whose predecessor stopped short."""
    rows = [
        flow_row(chunk, 0, 100.0, 14.0, 200.0, "Fin de la phrase.", 379.8),
        flow_row(chunk, 0, 100.0, 14.0, 200.0, "Debut de la suivante", 379.8),
    ]
    assert chunk.line_break_flags(rows) == [False, False]


def test_a_page_change_is_not_a_line_break(chunk):
    """It is a block break, which `break_strengths` already scores 3."""
    rows = [
        flow_row(chunk, 0, 700.0, 14.0, 180.0, "derniere ligne de la page", 379.8),
        flow_row(chunk, 1, 100.0, 14.0, 379.5, "premiere ligne de la suivante", 379.8),
    ]
    assert chunk.line_break_flags(rows) == [False, False]


def test_without_a_known_margin_nothing_is_flagged(chunk):
    """A row out of a serialised table carries no flow geometry, and a guess about
    where its column ended would put newlines inside sentences."""
    rows = [
        flow_row(chunk, 0, 100.0, 14.0, 180.0, "une ligne courte", 0.0),
        flow_row(chunk, 0, 112.0, 14.0, 200.0, "et la suivante", 0.0),
    ]
    assert chunk.line_break_flags(rows) == [False, False]


def test_a_sideways_row_is_left_alone(chunk):
    """On a sideways table a "line" runs down the page, so a right margin says
    nothing about where it ended."""
    rows = [
        chunk.Row(page=0, bbox=(100.0, 100.0, 112.0, 400.0), text="colonne", size=10.0,
                  vertical=True, right_edge=379.8),
        flow_row(chunk, 0, 100.0, 120.0, 200.0, "la suite", 379.8),
    ]
    assert chunk.line_break_flags(rows) == [False, False]


def test_a_chunk_breaks_the_line_where_the_author_did(chunk):
    """`Chunk.text` honours the flag, which is the whole point of computing it."""
    made = [flow_row(chunk, 0, 100.0, 91.0, 263.8, "Chaudiere (Gazole)", 379.8),
            flow_row(chunk, 0, 112.0, 91.0, 233.5, "Plancher (Chaud)", 379.8)]
    for row, starts in zip(made, chunk.line_break_flags(made)):
        row.starts_line = starts
    packed = chunk.pack(made, WordTokenizer(), target=200, overlap=0,
                        boundaries=True, headings=0)
    assert packed[0].text == "Chaudiere (Gazole)\nPlancher (Chaud)"


def test_without_boundaries_a_chunk_joins_the_way_it_always_did(chunk, rows):
    """The page and section bakes are text nobody reads, and they are unchanged."""
    packed = chunk.pack(rows, WordTokenizer(), target=60, overlap=10,
                        boundaries=False, headings=0)
    assert all("\n" not in c.text for c in packed)


def test_the_token_count_matches_the_text(chunk, rows):
    for c in chunk.pack(rows, WordTokenizer(), target=60, overlap=10,
                        boundaries=True, headings=chunk.HEADING_MAX_TOKENS):
        assert c.n_tokens == len(c.text.split())


def test_an_over_long_heading_run_is_dropped(chunk, rows):
    """A title page is not a heading, and prefixing one would drown the passage."""
    huge = [chunk.Row(page=0, bbox=(72.0, 100.0, 500.0, 112.0),
                      text="mot " * (chunk.HEADING_MAX_TOKENS + 5), size=14.0)]
    # rows[2:6] is body only, so the oversized row is the single heading candidate
    # and any carry would have to be it.
    chunks = chunk.pack(huge + rows[2:6], WordTokenizer(), target=60, overlap=10,
                        boundaries=True, headings=chunk.HEADING_MAX_TOKENS)
    assert all(not c.heading for c in chunks)


def test_every_row_still_appears_exactly_once(chunk, rows):
    """Overlap may repeat a row; the heading carry must not add to that count."""
    chunks = chunk.pack(rows, WordTokenizer(), target=1000, overlap=0,
                        boundaries=True, headings=chunk.HEADING_MAX_TOKENS)
    assert sum(len(c.rows) for c in chunks) == len(rows)


def test_a_chunk_ends_on_a_sentence_when_no_paragraph_is_near(chunk):
    """Unbroken prose: no paragraph break to rewind to, so the sentence end serves.

    Without the fallback the budget cuts wherever it runs out, which on this page
    is the middle of the second sentence.
    """
    sentence = ("un long paragraphe de texte sans aucune coupure visible qui continue "
                "encore et encore sur plusieurs lignes de suite.")
    rows = [line(chunk, i, sentence) for i in range(8)]
    chunks = chunk.pack(rows, WordTokenizer(), target=40, overlap=6, boundaries=True)
    assert len(chunks) > 1
    for packed in chunks[:-1]:
        assert packed.text.rstrip().endswith("."), packed.text[-60:]


def test_the_budget_still_cuts_where_there_is_no_sentence_at_all(chunk):
    # A dense table row has no punctuation to rewind to; the chunk must still close.
    rows = [line(chunk, i, "12 34 56 78 90 " * 4) for i in range(6)]
    chunks = chunk.pack(rows, WordTokenizer(), target=40, overlap=6, boundaries=True)
    assert len(chunks) > 1
    assert all(c.rows for c in chunks)


def test_the_overlap_opens_on_a_sentence_too(chunk):
    """A budget cut carries context forward; it must not carry half a clause."""
    rows = [line(chunk, 0, "premiere phrase complete du paragraphe."),
            line(chunk, 1, "deuxieme phrase qui se poursuit sur la ligne"),
            line(chunk, 2, "suivante sans ponctuation avant la fin ici."),
            line(chunk, 3, "troisieme phrase entierement nouvelle du texte.")]
    carried = chunk._carry_over(chunk.pack(rows, WordTokenizer(), target=1000, overlap=12,
                                           boundaries=True)[0], overlap=12)
    assert carried.rows
    assert carried.rows[0].text.startswith(("deuxieme", "troisieme"))


def test_a_chunk_reads_on_to_finish_its_sentence(chunk, monkeypatch):
    """No boundary behind, one just ahead: the chunk runs a little past its budget.

    Rewinding cannot help here, because the only sentence end in the whole run is
    the last row's. Before the look-ahead this cut in the middle of a clause.

    The look-ahead is OFF by default since 2026-09-30 (`OVERSHOOT` 0, the target is
    a ceiling), so this pins the value it shipped with: the mechanism is still there
    for any chunking that turns it back on.
    """
    # Patched where `pack` reads it: chunk.py only re-exports what lives in lib/chunk_pack.py.
    monkeypatch.setattr(load_lib("chunk_pack"), "OVERSHOOT", 0.2)
    # Six full rows fill the 40-token budget; the seventh would overflow it and is
    # the one that ends the sentence. It is five tokens, inside the 8-token slack.
    rows = [line(chunk, i, "mot mot mot mot mot mot") for i in range(6)]
    rows.append(line(chunk, 6, "et la phrase termine ici."))
    chunks = chunk.pack(rows, WordTokenizer(), target=40, overlap=6, boundaries=True)
    assert chunks[0].text.rstrip().endswith("."), chunks[0].text[-40:]
    assert chunks[0].n_tokens > 40, chunks[0].n_tokens


def test_a_section_up_to_the_target_stays_whole_and_none_goes_past_it(chunk):
    """The 2026-09-30 window: the floor a third of the target, the target a ceiling.

    Three sections of 40, 70 and 40 tokens (one per page, a page change being a
    section boundary) against a 100-token target, so a 33-token floor. Each section
    fits the target and clears the floor, so each is exactly one chunk: the packer
    fills towards 100, meets the next section's overflow, and rewinds to where the
    section began rather than cutting inside it. And nothing reads past 100.
    """
    sentence = "mot mot mot mot mot mot mot mot mot fin."   # 10 tokens
    rows = ([line(chunk, i, sentence, page=0) for i in range(4)]
            + [line(chunk, i, sentence, page=1) for i in range(7)]
            + [line(chunk, i, sentence, page=2) for i in range(4)])
    chunks = chunk.pack(rows, WordTokenizer(), target=100, overlap=0, boundaries=True)
    assert all(c.n_tokens <= 100 for c in chunks), [c.n_tokens for c in chunks]
    assert [sorted({r.page for r in c.rows}) for c in chunks] == [[0], [1], [2]], \
        [[r.page for r in c.rows] for c in chunks]


def test_overlap_at_boundaries_opens_a_section_on_the_end_of_the_last(chunk):
    """The switch the grid's overlap rows measure, against the same three sections.

    Without it the sections stay disjoint whatever the overlap (the cuts are all
    at boundaries). With it, every chunk after the first opens on the previous
    one's last sentence, and still no row is lost and no chunk is only overlap.
    """
    sentence = "mot mot mot mot mot mot mot mot mot fin."   # 10 tokens
    rows = ([line(chunk, i, sentence, page=0) for i in range(4)]
            + [line(chunk, i, sentence, page=1) for i in range(7)]
            + [line(chunk, i, sentence, page=2) for i in range(4)])
    plain = chunk.pack(rows, WordTokenizer(), target=100, overlap=10, boundaries=True)
    assert [sorted({r.page for r in c.rows}) for c in plain] == [[0], [1], [2]]
    carried = chunk.pack(rows, WordTokenizer(), target=100, overlap=10, boundaries=True,
                         overlap_boundaries=True)
    assert [[r.page for r in c.rows] for c in carried] == \
        [[0] * 4, [0] + [1] * 7, [1] + [2] * 4]
    assert {id(r) for c in carried for r in c.rows} == {id(r) for r in rows}


def test_a_list_moves_whole_to_the_next_chunk_rather_than_being_split(chunk):
    """A bulleted list is cut only when nothing outside it is in reach.

    A 40-token paragraph, then a 70-token list whose last four items are on the
    next page, against a 100-token target. The page change between two items used
    to be the best-ranked cut there is (a section boundary), which left three items
    in one chunk and four in the next. The start of the list is a fine cut too,
    with 40 tokens behind it and a 33-token floor, so the whole list goes on.
    """
    prose = "mot mot mot mot mot mot mot mot mot fin."    # 10 tokens
    item = "- mot mot mot mot mot mot mot mot fin."       # 10 tokens
    rows = ([line(chunk, i, prose) for i in range(4)]
            + [line(chunk, 4 + i, item) for i in range(3)]
            + [line(chunk, i, item, page=1) for i in range(4)])
    chunks = chunk.pack(rows, WordTokenizer(), target=100, overlap=0, boundaries=True)
    assert [c.text.count("- mot") for c in chunks] == [0, 7], [c.text for c in chunks]


def test_a_list_too_long_for_one_chunk_is_cut_between_items_not_inside_one(chunk):
    """When the list has to be split, an item stays whole.

    Items of two sentences, one per row, so a sentence end falls inside every item
    and would be as good a place to cut as any other finished sentence.
    """
    rows = []
    for i in range(12):
        rows.append(line(chunk, 2 * i, "- mot mot mot fin."))          # 5 tokens
        rows.append(line(chunk, 2 * i + 1, "mot mot mot mot fin."))    # 5 tokens
    chunks = chunk.pack(rows, WordTokenizer(), target=100, overlap=0, boundaries=True)
    assert len(chunks) > 1
    assert all(c.rows[0].text.startswith("- ") for c in chunks), [c.rows[0].text for c in chunks]


def test_what_counts_as_inside_a_list(chunk):
    rows = [line(chunk, 0, "Les criteres sont les suivants."),
            line(chunk, 1, "- premier critere,"),
            line(chunk, 2, "qui continue ici."),
            line(chunk, 3, "- second critere."),
            line(chunk, 20, "Un nouveau paragraphe apres la liste.")]
    breaks = chunk.break_strengths(rows)
    assert chunk.list_cuts(rows, breaks) == [0, 0, 2, 1, 0]


def test_a_bullet_after_an_unfinished_line_is_not_a_place_to_cut(chunk):
    """The layout says a list starts here; the text says a sentence does not end here.

    This is the cut that produced chunks ending in "les resultats sont": a bullet,
    a wider line gap and a heading all score as block boundaries, and none of them
    says anything about the line BEFORE them. The packer now prefers a shorter
    chunk that ends on a finished sentence.
    """
    rows = [line(chunk, 0, "mot " * 17 + "fin."),
            line(chunk, 1, "puis vient une seconde phrase dont les resultats sont"),
            line(chunk, 2, "- une liste qui commence apres"),
            line(chunk, 3, "mot " * 13 + "fin.")]
    chunks = chunk.pack(rows, WordTokenizer(), target=40, overlap=6, boundaries=True)
    assert chunks[0].text.rstrip().endswith("."), chunks[0].text[-60:]
    # And the unfinished line is where the next chunk starts, not where it is lost.
    assert "les resultats sont" in chunks[1].text


def test_the_look_ahead_gives_up_rather_than_swallowing_a_table(chunk):
    """A table has no sentence to finish, so the budget has to stay in charge."""
    rows = [line(chunk, i, "12 34 56 78 90 " * 4) for i in range(8)]
    chunks = chunk.pack(rows, WordTokenizer(), target=40, overlap=6, boundaries=True)
    assert max(c.n_tokens for c in chunks) < 40 * 2, [c.n_tokens for c in chunks]


def test_every_row_still_appears_after_the_look_ahead(chunk, rows):
    """The absorbed rows are in one chunk and one only: no row lost, none doubled."""
    packed = chunk.pack(rows, WordTokenizer(), target=45, overlap=0, boundaries=True)
    seen = [row.text for one in packed for row in one.rows]
    assert seen == [row.text for row in rows]


def test_no_chunk_comes_out_empty(chunk, rows):
    """An empty chunk is a box with nothing in it, which verify_chunks refuses.

    The look-ahead can close a chunk on the row BEFORE the one being read, and an
    earlier version then also ran the budget path on the chunk it had just emptied.
    """
    for target in (30, 45, 60, 120):
        packed = chunk.pack(rows, WordTokenizer(), target=target, overlap=10, boundaries=True)
        assert all(one.rows and one.text.strip() for one in packed), target


# --- sections ---------------------------------------------------------------
# `pack_sections` is the bake behind the section term of the blend and the
# literal "a chunk is a section" experiment (DESIGN.md). What it promises: one
# chunk per heading run and what sits under it, the preamble as a section of its
# own, no row lost or repeated, and a section the encoder could not read whole cut
# into pieces it can.


def test_one_section_per_heading_run(chunk, rows):
    """The synthetic page has a preamble-free layout: two headings, two sections."""
    sections = chunk.pack_sections(rows, WordTokenizer())
    assert len(sections) == 2
    assert sections[0].rows[0].text.startswith("5. Prise en charge")
    # The wrapped heading is one run, so its second line is in the same section.
    assert sections[0].rows[1].text == "chez l'habitant"
    assert sections[1].rows[0].text == "6. Tarifications"
    assert len(sections[1].rows) == 5


def test_rows_before_the_first_heading_are_a_section(chunk, rows):
    body = chunk.Row(page=0, bbox=(72.0, 50.0, 500.0, 62.0), text="avant tout", size=10.0)
    sections = chunk.pack_sections([body] + rows, WordTokenizer())
    assert len(sections) == 3
    assert sections[0].text == "avant tout"


def test_a_section_over_the_window_is_cut_at_a_row(chunk, rows):
    """Six 20-word rows under one heading, a 50-token ceiling: pieces, not truncation."""
    sections = chunk.pack_sections(rows, WordTokenizer(), limit=50)
    # Section 1 is 7 + 80 words, section 2 is 2 + 80: each becomes several pieces.
    assert len(sections) > 2
    assert all(s.n_tokens <= 50 or len(s.rows) == 1 for s in sections)
    # Nothing was cut inside a row, and no piece crosses a heading.
    for piece in sections:
        heads = [r for r in piece.rows if r.size > 10.0]
        assert not heads or piece.rows[0] in heads


def test_every_row_is_in_exactly_one_section(chunk, rows):
    sections = chunk.pack_sections(rows, WordTokenizer(), limit=50)
    seen = [r for s in sections for r in s.rows]
    assert seen == rows


def test_a_document_without_sizes_is_one_section(chunk, rows):
    blank = [chunk.Row(page=r.page, bbox=r.bbox, text=r.text, size=0.0) for r in rows]
    sections = chunk.pack_sections(blank, WordTokenizer())
    assert len(sections) == 1
    assert chunk.pack_sections([], WordTokenizer()) == []


@pytest.fixture
def ladder_rows(chunk):
    """A page where the two boundary policies have to disagree.

    Three body rows of 20 words, then a short heading, then a short bullet, then
    another body row. With a 90-token budget the sixth row overflows, and the two
    candidates behind it are a SECTION boundary at row 3 (60 tokens behind it, just
    over the 0.66 floor) and a PARAGRAPH boundary at row 4 (70 tokens). Every row
    ends in a full stop, so every candidate has a finished sentence behind it and
    the ladder's `clean` test cannot be what decides between them.
    """
    def row(index: int, text: str, size: float) -> object:
        return chunk.Row(page=0, bbox=(72.0, 100.0 + 12 * index, 500.0,
                                       112.0 + 12 * index), text=text, size=size)

    body = ("mot " * 19).strip() + " fin."
    return [
        row(0, body, 10.0),
        row(1, body, 10.0),
        row(2, body, 10.0),
        row(3, "Tarifications et conduite a tenir en urgence chez l'usager.", 14.0),
        row(4, "- premiere puce de la liste des tarifs usuels.", 10.0),
        row(5, body, 10.0),
    ]


def test_the_two_boundary_policies_cut_the_same_page_in_different_places(chunk, ladder_rows):
    """--section-first rewinds past the bullet to the heading; the default policy.

    This is the whole content of the flag: both policies end a chunk on a block
    boundary, but only this one gives up tokens to reach a boundary a reader would
    call a section. The heading therefore OPENS the second chunk rather than closing
    the first, which is what "a chunk is one piece of the document's structure"
    means in practice.
    """
    strengths = chunk.break_strengths(ladder_rows)
    assert strengths[3] == 3 and strengths[4] == 2, strengths

    packed = chunk.pack(ladder_rows, WordTokenizer(), target=90, overlap=10,
                        boundaries=True, sections=True)
    assert [len(c.rows) for c in packed] == [3, 3]
    assert "Tarifications" in packed[1].text and "Tarifications" not in packed[0].text


def test_paragraph_preferred_packing_keeps_the_heading_with_what_precedes_it(
        chunk, ladder_rows):
    """--no-section-first takes the LAST block boundary, whatever kind it is.

    Row 4 is a bullet, one rung below a heading and 10 tokens further along, so the
    chunk closes there instead: fuller chunks, and a section that starts mid-chunk.
    Measuring which of the two retrieves better is the reason the flag exists.
    """
    packed = chunk.pack(ladder_rows, WordTokenizer(), target=90, overlap=10,
                        boundaries=True, sections=False)
    assert [len(c.rows) for c in packed] == [4, 2]
    assert "Tarifications" in packed[0].text
    assert packed[1].text.startswith("- premiere")
