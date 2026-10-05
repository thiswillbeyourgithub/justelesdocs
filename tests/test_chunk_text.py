"""Text repair, glyph reading and sentences in scripts/chunk.py.

What a PDF hands back as text is not what the page shows: words cut at a line
break, false spaces inside words, accents drawn as separate glyphs, ligatures and
bullets arriving as control codes or dingbat fonts. These tests cover the repairs
(scripts/lib/chunk_text.py and the word level of scripts/lib/chunk_glyphs.py), the
French/English test, and the sentence split that lets a chunk end where a sentence
does. Each defect is pinned with the document it was found in.

Rows and rawdict lines are built by hand, so each case shows exactly the text or
glyph geometry that triggers it.

Written by Claude Code (Opus 5). Split out of test_chunk.py by Claude Code (Opus 5.5), along the lib/chunk_*.py stages.
"""

from __future__ import annotations

from conftest import line


def test_a_word_split_by_a_line_break_is_rejoined(chunk):
    rows = [line(chunk, 0, "Le suivi des per-"), line(chunk, 1, "sonnes agees demande du temps.")]
    assert chunk.dehyphenate(rows) == 1
    assert rows[0].text.endswith("personnes")
    assert rows[1].text == "agees demande du temps."


def test_the_document_decides_whether_the_hyphen_survives(chunk):
    # The same break, twice, in two documents that spell the word differently
    # elsewhere. Nothing but that context can tell the two cases apart.
    closed = [line(chunk, 0, "une prise en charge medico-"),
              line(chunk, 1, "sociale adaptee."),
              line(chunk, 2, "La prise en charge medicosociale est decrite ici.")]
    hyphenated = [line(chunk, 0, "une prise en charge medico-"),
                  line(chunk, 1, "sociale adaptee."),
                  line(chunk, 2, "L'offre medico-sociale est decrite ici.")]
    chunk.dehyphenate(closed)
    chunk.dehyphenate(hyphenated)
    assert closed[0].text.endswith("medicosociale")
    assert hyphenated[0].text.endswith("medico-sociale")


def test_an_unattested_break_closes_up(chunk):
    rows = [line(chunk, 0, "les reseaux hydrogra-"), line(chunk, 1, "phiques sont denses.")]
    chunk.dehyphenate(rows)
    assert rows[0].text.endswith("hydrographiques")


def test_a_capital_after_the_break_is_left_alone(chunk):
    # "Haussmann- Type" is two things, not one word cut in half.
    rows = [line(chunk, 0, "immeuble de type Haussmann-"), line(chunk, 1, "Type II selon la classification.")]
    assert chunk.dehyphenate(rows) == 0
    assert rows[0].text.endswith("Haussmann-")


def test_words_are_not_rejoined_across_a_page_break(chunk):
    rows = [line(chunk, 0, "le traitement anti-", page=0),
            line(chunk, 1, "calcaire est poursuivi.", page=1)]
    assert chunk.dehyphenate(rows) == 0


def test_a_row_that_is_only_the_second_half_keeps_its_text(chunk):
    # Emptying it would leave a box with nothing to highlight.
    rows = [line(chunk, 0, "un traitement anti-"), line(chunk, 1, "calcaire")]
    assert chunk.dehyphenate(rows) == 0
    assert rows[1].text == "calcaire"


def test_the_table_is_described_in_the_language_of_its_page(chunk):
    assert chunk.looks_french("Le traitement de la ressource en eau chez l'habitant et les autres")
    assert not chunk.looks_french("The treatment of water in homes and the others")


def test_a_word_broken_inside_one_row_is_rejoined(chunk):
    """A table cell built from two visual lines carries the break inside itself."""
    rows = [line(chunk, 0, "Se rendre a un rendez- vous ; 28,5 ; 17,7."),
            line(chunk, 1, "Prendre un rendez-vous demande de la preparation.")]
    assert chunk.dehyphenate(rows) == 1
    # The document spells it with the hyphen elsewhere, so the hyphen stays.
    assert "rendez-vous ; 28,5" in rows[0].text


def test_the_word_is_rejoined_over_a_line_that_ends_a_list_item(chunk):
    """The second half plus a semicolon is still a row with text in it.

    This is the end of every French bulleted list ("... activites signifi-" /
    "catives ;"), and refusing to join left the broken word in the chunk.
    """
    rows = [line(chunk, 0, "la participation a d'autres activites signifi-"),
            line(chunk, 1, "catives ;")]
    assert chunk.dehyphenate(rows) == 1
    assert rows[0].text.endswith("significatives")
    assert rows[1].text == ";"


def test_a_numbered_heading_does_not_end_a_sentence(chunk):
    """Section numbers end on a full stop and end nothing.

    Some French reports number their parts "VII.2.3.", which used to satisfy the
    sentence test and let a chunk close on the number with its title in the next
    one.
    """
    assert not chunk.ends_sentence("VII.2.3.")
    assert not chunk.ends_sentence("3.3.3.")
    assert not chunk.ends_sentence("Les criteres sont :")
    assert chunk.ends_sentence("Le traitement a ete arrete en 2016.")


def raw_line(*runs, y=100.0, size=9.0, width=5.0, direction=(1.0, 0.0)):
    """A rawdict line laid out left to right, as ``get_text("rawdict")`` gives one.

    Each run is ``(text, font)`` or ``(text, font, x)``: a span whose glyphs are
    ``width`` apart, starting at ``x`` or straight after the previous run. A glyph's
    box is exactly its advance, so two letters of a word touch (a gap of 0).
    """
    spans, x = [], 50.0
    for run in runs:
        text, font, *start = run
        x = start[0] if start else x
        chars = []
        for char in text:
            chars.append({"c": char, "bbox": (x, y, x + width, y + size)})
            x += width
        spans.append({"font": font, "size": size, "chars": chars})
    boxes = [char["bbox"] for span in spans for char in span["chars"]]
    return {"dir": direction, "spans": spans,
            "bbox": (min(b[0] for b in boxes), y, max(b[2] for b in boxes), y + size)}


def words_of(chunk, line):
    return [chunk.word_text(word) for word in chunk.split_words(line)]


def test_a_space_drawn_inside_a_word_is_not_a_word_break(chunk):
    """One typeset report's "ar e needed": the space glyph adds nothing to the letter gap."""
    line = raw_line(("ar", "Palatino"), (" ", "Palatino", 60.0), ("e", "Palatino", 60.0),
                    (" ", "Palatino", 65.0), ("needed", "Palatino", 67.0))
    assert words_of(chunk, line) == ["ar\u2060e", "needed"]
    assert chunk.line_text(line) == "ar\u2060e needed"


def test_a_real_space_is_kept_even_where_every_glyph_box_overlaps(chunk):
    """Boxes 0.36 em wider than the letters: only the gap RELATIVE to the line counts."""
    wide = raw_line(("J", "Times"), (" ", "Times"), ("Hydrologist", "Times"))
    for span in wide["spans"]:
        for char in span["chars"]:
            x0, y0, x1, y1 = char["bbox"]
            char["bbox"] = (x0, y0, x1 + 0.36 * 9, y1)
    # Letter to letter the boxes overlap by 3.2 points; across the space they are
    # a whole glyph apart, which is a space.
    assert words_of(chunk, wide) == ["J", "Hydrologist"]


def test_the_document_decides_a_doubted_space(chunk):
    """Geometry nominates; the document's own words decide, as for a hyphen break."""
    rows = [line(chunk, 0, "if doses ar\u2060e needed, give more of\u2060Action"),
            line(chunk, 1, "doses are given; the action of each dose"),
            line(chunk, 2, "reinit\u2060iate the treatment")]
    assert chunk.resolve_glue(rows) == 2
    assert [row.text for row in rows] == ["if doses are needed, give more of Action",
                                          "doses are given; the action of each dose",
                                          "reinitiate the treatment"]


def test_a_fault_on_every_occurrence_cannot_vouch_for_itself(chunk):
    """Every "fi" in one printed guide is followed by a false space."""
    rows = [line(chunk, 0, "fi\u2060nal benefi\u2060ts"), line(chunk, 1, "fi\u2060nal dose")]
    assert chunk.resolve_glue(rows) == 3
    assert [row.text for row in rows] == ["final benefits", "final dose"]


def test_a_space_between_two_fonts_is_kept_however_narrow(chunk):
    """An italic's overhang closes the gap before the roman word: "Insight et"."""
    line = raw_line(("Insight", "Arial-BoldItalic"), (" ", "Arial-Bold", 85.0),
                    ("et", "Arial-Bold", 85.0))
    assert words_of(chunk, line) == ["Insight", "et"]


def test_glyph_boxes_of_no_width_decide_nothing(chunk):
    """One leaflet gives every glyph a zero-width box: no gap can be measured."""
    line = raw_line(("retenues", "Arial"), (" ", "Arial"), ("dans", "Arial"), width=0.0)
    assert words_of(chunk, line) == ["retenues", "dans"]


def test_two_touching_lines_are_one_word(chunk):
    """One typeset report's "reinit" and "iate": two lines, one ending where the next starts."""
    first = raw_line(("reinit", "Palatino"))
    second = raw_line(("iate", "Palatino", 80.0), (" ", "Palatino"))
    apart = raw_line(("iate", "Palatino", 82.0))
    assert chunk.continues_word(first, second)
    assert not chunk.continues_word(first, apart)
    assert not chunk.continues_word(first, raw_line(("iate", "Palatino", 80.0), y=120.0))


def test_an_invisible_hyphen_at_a_line_end_becomes_a_visible_one(chunk):
    # One typesetter breaks words with a soft hyphen, which used to be deleted
    # outright: "hy" and "draulic" then reached the encoder as two words.
    assert chunk.clean_text("3% of homes with electric heating re­") .endswith("re-")


def test_an_invisible_hyphen_at_a_line_end_inside_a_table_cell_is_rejoined(chunk):
    # A cell's printed lines are joined with spaces before clean_text sees them, so
    # the break is followed by a space rather than by the end of the text, and the
    # cells of one report read "hydrogra phie" and "Sodium carbonate monohy drate".
    cell = chunk.clean_text("Sodium carbonate monohy\u00ad\ndrate".replace("\n", " "))
    assert cell == "Sodium carbonate monohy- drate"
    assert chunk._join_inside(cell, attested="", decided={})[0] == "Sodium carbonate monohydrate"


def test_an_invisible_hyphen_inside_a_word_still_closes_it_up(chunk):
    assert chunk.clean_text("co­operation entre professionnels") == "cooperation entre professionnels"


def test_a_soft_hyphen_break_is_then_rejoined(chunk):
    rows = [line(chunk, 0, chunk.clean_text("during hy­")),
            line(chunk, 1, chunk.clean_text("draulic tests, hydraulic systems"))]
    assert chunk.dehyphenate(rows) == 1
    assert rows[0].text.endswith("hydraulic")


def test_a_break_whose_second_half_ends_a_clause_is_rejoined(chunk):
    # "condi-" / "tions, and high rates" used to be refused outright, because the
    # rule wanted whitespace straight after the word that finishes the break. That
    # is exactly where a line ends most often, so the refusal was the common case.
    rows = [line(chunk, 0, "load overlap with other condi-"),
            line(chunk, 1, "tions, and high rates of leakage in these conditions")]
    assert chunk.dehyphenate(rows) == 1
    assert rows[0].text.endswith("conditions,")
    assert rows[1].text == "and high rates of leakage in these conditions"


def test_a_superscript_citation_stuck_to_the_second_half_travels_with_it(chunk):
    # From a journal-style guideline: the reader was shown "insula- tion,5"
    # in a snippet, and the tokenizer three pieces instead of one word. The rule
    # used to move only the punctuation after the word, so anything glued behind
    # the punctuation (here a superscript reference number) kept the pair apart.
    rows = [line(chunk, 0, "questioning the efficacy of insula-"),
            line(chunk, 1, "tion,5 subsequent meta-analyses have confirmed")]
    assert chunk.dehyphenate(rows) == 1
    assert rows[0].text.endswith("insulation,5")
    assert rows[1].text == "subsequent meta-analyses have confirmed"


def test_a_compound_broken_at_its_own_hyphen_is_put_back_whole(chunk):
    # "ener-" / "gy-efficient/renewable": the second hyphen belongs to the
    # compound, so the whole token has to follow the word rather than stopping at
    # the first character that is not a letter.
    rows = [line(chunk, 0, "rising costs,37 ener-"),
            line(chunk, 1, "gy-efficient/renewable housing,38 public transport")]
    assert chunk.dehyphenate(rows) == 1
    assert rows[0].text.endswith("energy-efficient/renewable")
    assert rows[1].text == "housing,38 public transport"


# --- sentence-level rows ------------------------------------------------------
# A row is a visual line, so a chunk could only ever end where a line ended, and
# 46% of chunks ended mid-sentence. These cover the split that fixed it.


def sentence_row(chunk, text: str, *, index: int = 0, page: int = 0) -> object:
    """One ordinary body line, stacked under the previous one."""
    return chunk.Row(page=page, bbox=(72.0, 100.0 + 12 * index, 500.0,
                                      112.0 + 12 * index), text=text, size=10.0)


def test_a_row_holding_two_sentences_becomes_two_rows(chunk):
    rows = [sentence_row(chunk, "Le chantier va bien. Il rentre chez lui.")]
    out, added = chunk.split_sentences(rows)
    assert [r.text for r in out] == ["Le chantier va bien.", "Il rentre chez lui."]
    assert added == 1


def test_the_pieces_keep_the_line_they_came_from(chunk):
    source = sentence_row(chunk, "Premiere phrase. Seconde phrase.")
    out, _ = chunk.split_sentences([source])
    assert {r.bbox for r in out} == {source.bbox}
    assert {r.page for r in out} == {source.page}


def test_one_box_is_drawn_once_however_many_sentences_share_the_line(chunk):
    out, _ = chunk.split_sentences(
        [sentence_row(chunk, "Une phrase. Deux phrases. Trois phrases.")])
    packed = chunk.Chunk()
    for piece in out:
        packed.add(piece, 5)
    assert sum(len(v) for v in packed.boxes_by_page().values()) == 1


def test_an_abbreviation_is_not_a_sentence_end(chunk):
    for text in ("Voir Martin J. P. et al. 2016 pour le detail.",
                 "Le traitement (cf. tableau 2) est efficace.",
                 "Un debit de 5 m3 env. par jour suffit."):
        assert chunk.split_sentences([sentence_row(chunk, text)])[0][0].text == text


def test_a_numbered_heading_is_not_a_sentence_end(chunk):
    text = "3.3.3. Gestion du reseau d'eau potable"
    assert chunk.split_sentences([sentence_row(chunk, text)])[0][0].text == text


def test_a_year_at_the_end_of_a_sentence_still_splits(chunk):
    out, _ = chunk.split_sentences(
        [sentence_row(chunk, "Publie en 2016. Le resultat est clair.")])
    assert [r.text for r in out] == ["Publie en 2016.", "Le resultat est clair."]


def test_splitting_never_loses_a_word(chunk):
    rows = [sentence_row(chunk, "Une phrase. Une autre phrase! Et une question ? Fin."),
            sentence_row(chunk, "Une ligne sans fin de phrase", index=1),
            sentence_row(chunk, "M. Dupont a vu Mme Martin. Ils sont d'accord.", index=2)]
    out, _ = chunk.split_sentences(rows)
    assert " ".join(r.text for r in out).split() == " ".join(r.text for r in rows).split()


def test_a_sentence_end_is_now_a_place_a_chunk_can_stop(chunk):
    """The point of the split: break_strengths scores the row after a sentence."""
    rows, _ = chunk.split_sentences(
        [sentence_row(chunk, "Premiere phrase. Seconde phrase.")])
    assert chunk.break_strengths(rows)[1] >= 1


def test_an_accent_that_lost_its_letter_is_put_back(chunk):
    assert chunk.clean_text("Daniel J. Mu ̈ller") == "Daniel J. Müller"
    assert chunk.clean_text("S. Vale ́rie Tourneur") == "S. Valérie Tourneur"


def accented(chunk, text, accent_at, over):
    """`text` as a rawdict line whose char `accent_at` is drawn over char `over`."""
    line = raw_line((text, "AdvPSA334"))
    chars = line["spans"][0]["chars"]
    chars[accent_at]["bbox"] = chars[over]["bbox"]
    return chunk.clean_text(chunk.line_text(line))


def test_an_accent_drawn_on_its_letter_is_attached_to_it(chunk):
    """TeX-set PDFs: "Clı´nica", "Hoˆpital", "Fe`ve" in the text layer."""
    assert accented(chunk, "Clı´nica", 3, 2) == "Clínica"
    assert accented(chunk, "Hoˆpital", 2, 1) == "Hôpital"
    assert accented(chunk, "a` la", 1, 0) == "à la"
    # Drawn BEFORE its letter in the stream, the mark still lands on that letter.
    assert accented(chunk, "¸ca", 0, 1) == "ça"


def test_an_acute_between_two_letters_it_does_not_sit_on_is_an_apostrophe(chunk):
    """One classification's "individual´s" draws the acute after the "l", not on it."""
    line = raw_line(("individual´s", "Arial"))
    assert chunk.clean_text(chunk.line_text(line)) == "individual’s"


def test_characters_that_carry_no_text_are_dropped_or_mapped(chunk):
    assert chunk.clean_text("ACKNO\u200bWLE\u200bDGE\ufeffMENTS") == "ACKNOWLEDGEMENTS"
    assert chunk.clean_text("ACME : \ufffd\ufffd\ufffdAssociation") == "ACME : Association"
    assert chunk.clean_text("solu\u019fons") == "solutions"
    assert chunk.clean_text("Documentation \u0154 Information") == "Documentation \u2013 Information"
    # GLUE is invisible too, and must survive for resolve_glue to decide it.
    assert chunk.clean_text("ar\u2060e") == "ar\u2060e"


def test_a_control_code_that_is_not_inside_a_word_is_a_bullet(chunk):
    """A Wingdings list bullet, a table's inhibition marks, an InDesign indent."""
    assert chunk.clean_text("\x01 Quels sont les risques ?") == "\u2022 Quels sont les risques ?"
    assert chunk.BULLET_START.match(chunk.clean_text("\x01 Quels sont les risques ?"))
    # A compatibility table: the marks ARE the table's content.
    assert chunk.clean_text("EN1092 \x02 \x02 \x02 ; Flanges") == (
        "EN1092 \u2022 \u2022 \u2022 ; Flanges")
    assert chunk.clean_text("1. \x07FORMALISER le projet") == "1. \u2022 FORMALISER le projet"


def test_a_control_code_inside_a_word_is_read_as_the_ligature_the_document_votes_for(chunk):
    """A report's flowchart: 0x02 draws "ti" and 0x05 draws "ft"."""
    rows = [line(chunk, 0, chunk.clean_text("Consider local factors in selec\x02ng an an\x02freeze")),
            line(chunk, 1, chunk.clean_text("and ini\x02ate a first-line one, a\x05er 2-4 weeks")),
            line(chunk, 2, chunk.clean_text("selecting an antifreeze after the first")),
            line(chunk, 3, chunk.clean_text("Preface\x08xiii"))]
    assert chunk.resolve_ligatures(rows) == 4
    assert [row.text for row in rows] == [
        "Consider local factors in selecting an antifreeze",
        # "initiate" is written nowhere else: 0x02's other words decided it.
        "and initiate a first-line one, after 2-4 weeks",
        "selecting an antifreeze after the first",
        # Nothing votes for 0x08, a tab here: it is the space the page shows.
        "Preface xiii"]


def test_a_dingbat_glyph_is_a_bullet_whatever_code_it_extracts_as(chunk):
    """Wingdings3's arrow reads as an acute, Wingdings' check box as a diaeresis."""
    arrow = raw_line(("\u00b4", "Wingdings3"), (" Y penser", "RotisSemiSansStd"))
    assert chunk.clean_text(chunk.line_text(arrow)) == "\u2022 Y penser"
    boxes = raw_line(("\u00a8", "Wingdings"), ("1 ; ", "TimesNewRoman"),
                     ("\u00a8", "Wingdings"), ("2", "TimesNewRoman"))
    assert chunk.clean_text(chunk.line_text(boxes)) == "\u20221 ; \u20222"
    assert chunk.clean_text(chunk.line_text(raw_line(("\u00af", "MonotypeSorts"), (" une", "Times")))) == "\u2022 une"


def test_a_symbol_code_pymupdf_mis_maps_is_read_as_the_symbol(chunk):
    """One leaflet: "May \u00af flow rate" is a down arrow."""
    line = raw_line(("May ", "Arial"), ("\u00af", "SymbolMT"), (" flow", "Arial"))
    assert chunk.clean_text(chunk.line_text(line)) == "May \u2193 flow"
    # The same code in a text font is a macron, and left to the accent handling.
    assert chunk.symbol_glyph({"c": "\u00af"}, "Arial") == {"c": "\u00af"}


def test_a_ruled_table_cell_reads_symbols_as_running_text_does(chunk):
    """find_tables extracts cell text itself, so the glyph reading is applied to it.

    Before, one r\u00e9f\u00e9rentiel's tables kept 80 Wingdings bullets as acute
    accents and another report's table 11 Symbol arrows as macrons.
    """
    import pymupdf
    from types import SimpleNamespace
    row = SimpleNamespace(cells=[(0, 0, 50, 10), (50, 0, 100, 10), None])
    table = SimpleNamespace(rows=[row])
    data = [["\u00b4 un", "May \u00af flow \u00af", None]]
    # One glyph per cell, each at the centre of the cell it was drawn in.
    symbols = [(pymupdf.Point(5, 5), "\u00b4", "\u2022"),
               (pymupdf.Point(60, 5), "\u00af", "\u2193")]
    out = chunk.reread_symbols(table, data, symbols, pymupdf.Rect)
    # Only as many occurrences as glyphs were drawn there: the second macron
    # came from a text font and is left for the accent handling.
    assert out == [["\u2022 un", "May \u2193 flow \u00af", None]]


def test_an_ordinary_accent_is_left_alone(chunk):
    assert (chunk.clean_text("prise en charge énergétique")
            == "prise en charge énergétique")


def test_the_document_decides_whether_a_broken_word_keeps_its_hyphen(chunk):
    """Not a dictionary: whichever form the document itself uses elsewhere wins."""
    decided = {}
    assert chunk.glue_for("médico", "sociaux",
                          attested="les services médico-sociaux du secteur",
                          decided=decided) == "-"
    assert chunk.glue_for("per", "sonnes", attested="les personnes concernées",
                          decided={}) == ""
    # Attested nowhere: the default closes it up, which is right for 78% of them.
    assert chunk.glue_for("hydrogra", "phie", attested="rien de tel ici",
                          decided={}) == ""


def test_a_pair_is_only_counted_once_per_document(chunk):
    """Each answer is two scans of the whole document, and the pairs repeat.

    Observable by asking again against text that would now answer differently: a
    second count never happens, because the snapshot cannot change under it.
    """
    decided = {}
    assert chunk.glue_for("Éner", "gétique", attested="la éner-gétique",
                          decided=decided) == "-"
    assert decided == {("éner", "gétique"): "-"}
    assert chunk.glue_for("éner", "gétique", attested="énergétique énergétique",
                          decided=decided) == "-", "the pair was counted twice"
