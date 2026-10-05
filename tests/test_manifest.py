"""The manifest builder's guesses and its two gates.

The one bug worth a permanent test is recorded in DESIGN.md: `ISSUER_PATTERNS` was
searched case-insensitively, so an acronym pattern such as `\\bWHO\\b` matched the
English pronoun "who" (and an issuer spelled like a verb or an adjective would match
that word), and documents were labelled with an issuing body that never published
them. A wrong facet value is worse than a blank one
(it excludes documents that should be included and includes ones that should not),
and `manifest.py` fills blanks only, so a wrong value is then protected from ever
being corrected by the script. Hence the prose below.

Written by Claude Code (Opus 5).
"""

from __future__ import annotations

from collections import Counter

import pytest
from conftest import cover

# Ordinary prose that a case-insensitive bare-acronym pattern reads as an issuer.
ENGLISH_PROSE = (
    "This report has been prepared by the working group, who met twice. It is nice to "
    "have a clear summary, and the committee has agreed that the iso-lines are what "
    "ANY reader can use."
)
FRENCH_PROSE = (
    "Ce document a été préparé par le groupe de travail. Il a fait l'objet d'une "
    "relecture, et la commission a validé les propositions."
)


def test_bare_acronyms_do_not_fire_on_prose(manifest):
    """The regression: "who" is a pronoun, "iso" a prefix, "has" a verb."""
    assert manifest.guess_issuer(ENGLISH_PROSE) == ("", "")
    assert manifest.guess_issuer(FRENCH_PROSE) == ("", "")


def test_patterns_are_checked_at_import(manifest):
    """`_check_issuer_patterns` runs on import and is what keeps the above true."""
    manifest._check_issuer_patterns()   # must not raise


def test_a_real_issuer_is_still_recognised(manifest):
    """An acronym in its own case, and a full name in any case, both still match."""
    issuer, country = manifest.guess_issuer(
        "World Health Organization - Guidelines on housing and health - 2018")
    assert (issuer, country) == ("WHO", "INT")
    issuer, country = manifest.guess_issuer("ISO 21542:2021 Building construction")
    assert (issuer, country) == ("ISO", "INT")


def test_language_guess(manifest):
    """Counted over function words, and blank rather than a coin flip when scarce."""
    assert manifest.guess_language(FRENCH_PROSE * 4) == "fr"
    assert manifest.guess_language(ENGLISH_PROSE * 4) == "en"
    assert manifest.guess_language("Titre") == "", "too little text must stay blank"


def test_a_language_tie_stays_blank(manifest):
    """Equal evidence for two languages is a coin flip, and a coin flip is what the
    blank exists to avoid. The two-language version answered "en" on a tie."""
    assert manifest.guess_language("le la les des une the of and for with") == ""


def test_read_existing_keeps_curated_cells(manifest, tmp_path):
    """The blank-only fill rests on this: what is written must come back unchanged."""
    path = tmp_path / "MANIFEST.tsv"
    path.write_text(
        "file\ttitle\tissuer\n"
        "a.pdf\tUn titre écrit à la main\tINSEE\n"
        "b.pdf\t\t\n",
        encoding="utf-8")
    rows = manifest.read_existing(path)
    assert rows["a.pdf"]["title"] == "Un titre écrit à la main"
    assert rows["a.pdf"]["issuer"] == "INSEE"
    assert rows["b.pdf"]["title"] == ""


def test_read_existing_of_a_missing_file(manifest, tmp_path):
    assert manifest.read_existing(tmp_path / "nope.tsv") == {}


def test_clean_title_is_only_a_fallback(manifest):
    """It makes a filename presentable; every shipped title is written by hand."""
    assert "_" not in manifest.clean_title("rapport_17_plu_lyon_vd.pdf")
    assert not manifest.clean_title("x.pdf").endswith(".pdf")


def test_an_edition_is_never_collapsible(manifest):
    """Collapse keeps the best-scoring sibling, which for editions is the wrong one.

    If "edition" ever joins REDUNDANT_RENDITIONS, a query whose wording happens
    to suit the 11th edition of a handbook would fold the 15th away behind it, and
    the reader would be shown withdrawn guidance with no sign that a newer edition of
    the same book was in the corpus.
    """
    assert "edition" not in manifest.REDUNDANT_RENDITIONS
    assert "edition" in manifest.EDITION_RENDITIONS


def test_every_curated_family_has_a_sibling(manifest):
    """A family of one collapses nothing and usually means a slug was mistyped."""
    counts = Counter(family for family, _rendition in manifest.CURATED_FAMILIES.values()
                     if family)
    alone = sorted(family for family, n in counts.items() if n < 2)
    assert not alone, f"family slugs with a single member: {alone}"


def test_two_documents_cannot_ship_under_one_title(manifest):
    """The title is all the result card gives the reader to choose by."""
    rows = [{"file": "a.pdf", "title": "Rénovation du parc de logements sociaux"},
            {"file": "b.pdf", "title": "Rénovation du parc de logements sociaux"}]
    with pytest.raises(Exception) as caught:
        manifest.check_titles(rows)
    assert "a.pdf" in str(caught.value) and "b.pdf" in str(caught.value)


def test_accents_and_punctuation_do_not_hide_a_shared_title(manifest):
    """Two spellings of one title read as one title on screen."""
    rows = [{"file": "a.pdf", "title": "Qualité de l'eau potable en région"},
            {"file": "b.pdf", "title": "Qualite de l eau potable en region"}]
    with pytest.raises(Exception):
        manifest.check_titles(rows)


def test_a_tolerated_pair_does_not_stop_a_run(manifest, monkeypatch):
    """The hatch is empty since the last known duplicate was resolved, so it gets its own pair.

    Testing it against whatever the corpus happens to tolerate today would make the
    test pass by accident once the list is empty, which is exactly when the mechanism
    is least exercised and most likely to rot.
    """
    pair = frozenset({"a.pdf", "b.pdf"})
    monkeypatch.setattr(manifest, "KNOWN_DUPLICATE_TITLES", frozenset({pair}))
    manifest.check_titles([{"file": name, "title": "Plan de prevention du bruit"} for name in sorted(pair)])


def test_a_third_document_joining_a_tolerated_pair_still_stops_the_run(manifest, monkeypatch):
    """The entry excuses that pair, not that title."""
    pair = frozenset({"a.pdf", "b.pdf"})
    monkeypatch.setattr(manifest, "KNOWN_DUPLICATE_TITLES", frozenset({pair}))
    rows = [{"file": name, "title": "Plan de prevention du bruit"} for name in sorted(pair)]
    rows.append({"file": "c.pdf", "title": "Plan de prevention du bruit"})
    with pytest.raises(Exception):
        manifest.check_titles(rows)


def test_blank_titles_are_not_duplicates_of_each_other(manifest):
    """An empty cell is caught by the title check in the shipped-manifest test."""
    manifest.check_titles([{"file": "a.pdf", "title": ""}, {"file": "b.pdf", "title": ""}])


def test_the_title_comes_off_the_cover_not_the_filename(manifest, tmp_path):
    """The reason the reader sees a title at all on an agency dump."""
    pdf = cover(tmp_path, [(120, 22, "Mobilites douces a l'echelle locale"),
                           (200, 9, "Methode Rapport pour la planification urbaine")],
                name="2014-12-15_mobilites_douces_rapport.pdf")
    assert manifest.title_from_first_page(pdf) == "Mobilites douces a l'echelle locale"


def test_the_collection_label_is_not_the_title(manifest, tmp_path):
    """Some covers set "EXECUTIVE SUMMARY" bigger than the title."""
    pdf = cover(tmp_path, [(100, 30, "EXECUTIVE SUMMARY"),
                           (160, 18, "Accessible housing for older people")])
    assert manifest.title_from_first_page(pdf) == "Accessible housing for older people"


def test_a_title_that_wraps_is_joined(manifest, tmp_path):
    """Three cover lines of one sentence, the continuation set smaller."""
    pdf = cover(tmp_path, [(100, 24, "Reduction de la consommation d'eau :"),
                           (128, 24, "du diagnostic individuel"),
                           (156, 16, "au maintien des economies")])
    assert manifest.title_from_first_page(pdf) == (
        "Reduction de la consommation d'eau : du diagnostic individuel "
        "au maintien des economies"
    )


def test_a_far_away_line_is_not_joined(manifest, tmp_path):
    """A gap wider than the title's own font size ends the run."""
    pdf = cover(tmp_path, [(100, 20, "Reseaux de chaleur et systemes apparentes"),
                           (400, 20, "Service des statistiques professionnelles")])
    assert manifest.title_from_first_page(pdf) == "Reseaux de chaleur et systemes apparentes"


def test_a_page_with_no_text_layer_yields_nothing(manifest, tmp_path):
    """Scanned covers exist; the caller falls back to the filename."""
    assert manifest.title_from_first_page(cover(tmp_path, [])) == ""


def test_an_unreadable_file_yields_nothing(manifest, tmp_path):
    """Never raises: a title is a nicety, and the fallback is always available."""
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf at all")
    assert manifest.title_from_first_page(broken) == ""


def test_folding_accents_is_what_lets_the_labels_be_written_in_ascii(manifest):
    assert manifest.fold_accents("Rapport d'élaboration") == "Rapport d'elaboration"


def test_a_leading_publication_date_is_not_part_of_the_title(manifest):
    """Every file in one agency's bulk download is named after its publication date."""
    assert manifest.clean_title("2015-02-12_bruit_synthese.pdf") == "bruit synthese"


def test_a_generic_collection_label_is_refused_as_a_title(manifest, tmp_path):
    """A cover whose only large line is "Rapport final" names nothing."""
    pdf = cover(tmp_path, [(100, 24, "Rapport final")])
    assert manifest.title_from_first_page(pdf) == ""


def test_a_title_cut_in_half_is_refused(manifest, tmp_path):
    """"Bon usage des" reads like a title and names nothing."""
    pdf = cover(tmp_path, [(100, 24, "Bon usage des"), (300, 9, "compteurs d'eau")])
    assert manifest.title_from_first_page(pdf) == ""


def test_a_title_starting_mid_sentence_is_refused(manifest, tmp_path):
    """The same cut seen from the other end."""
    pdf = cover(tmp_path, [(100, 24, "proprietaire bailleur et copropriete")])
    assert manifest.title_from_first_page(pdf) == ""


def test_the_tail_of_a_label_is_not_a_title(manifest, tmp_path):
    """"RAPPORT FINAL / DE LA COMMISSION DES TRANSPORTS": a label, then its tail."""
    pdf = cover(tmp_path, [(100, 20, "RAPPORT FINAL"),
                           (124, 20, "DE LA COMMISSION DES TRANSPORTS")])
    assert manifest.title_from_first_page(pdf) == ""


def test_a_collection_label_above_the_title_is_not_a_reason_to_refuse_it(manifest, tmp_path):
    """The usual cover: the collection label sits right on top of the title."""
    pdf = cover(tmp_path, [(100, 20, "Executive Summary"),
                           (124, 20, "Water quality in rural schools")])
    assert manifest.title_from_first_page(pdf) == "Water quality in rural schools"


def run_manifest(manifest_module, tmp_path, *extra):
    """manifest.py over an EMPTY corpus directory, so no PDF is ever opened."""
    from click.testing import CliRunner

    guidelines = tmp_path / "GUIDELINES"
    guidelines.mkdir()
    path = tmp_path / "MANIFEST.tsv"
    path.write_text("file\ttitle\tsource_url\n"
                    "gone.pdf\tUn titre écrit à la main\thttps://example.org/x\n",
                    encoding="utf-8")
    result = CliRunner().invoke(manifest_module.main, [
        "--guidelines", str(guidelines),
        "--manifest", str(path),
        "--sorting-log", str(tmp_path / "absent.tsv"),
        *extra,
    ])
    return result, path


def test_a_curated_row_whose_pdf_is_gone_stops_the_run(manifest, tmp_path):
    """The row is the only copy of its curation, and rewriting the file drops it.

    Moving a document out of GUIDELINES/ for a moment used to cost the hand-written
    title, source_url, doc_type and topic, with a warning nobody had to read.
    """
    result, path = run_manifest(manifest, tmp_path)

    assert result.exit_code != 0
    assert "gone.pdf" in result.output
    assert "Un titre écrit à la main" in path.read_text(encoding="utf-8"), \
        "the refusal must leave the manifest untouched"


def test_drop_orphans_says_it_out_loud_and_writes(manifest, tmp_path):
    """The escape hatch, for a document that really has left the corpus."""
    result, path = run_manifest(manifest, tmp_path, "--drop-orphans")

    assert result.exit_code == 0, result.output
    assert "gone.pdf" not in path.read_text(encoding="utf-8")


def test_a_row_missing_only_its_page_count_does_not_open_the_pdf(manifest, tmp_path):
    """The page count comes from the sorting log, so nothing has to be extracted.

    Observable because `extract_head_text` warns on a file it cannot parse: this
    corpus holds one, and a run that stays quiet is a run that never opened it.
    """
    from click.testing import CliRunner

    guidelines = tmp_path / "GUIDELINES"
    guidelines.mkdir()
    (guidelines / "a.pdf").write_text("not a PDF at all", encoding="utf-8")
    log = tmp_path / "SORTING_LOG.tsv"
    log.write_text("file\tpages\na.pdf\t42\n", encoding="utf-8")
    path = tmp_path / "MANIFEST.tsv"
    path.write_text("file\ttitle\tissuer\tcountry\tyear\tlanguage\tpages\n"
                    "a.pdf\tUn titre\tINSEE\tFR\t2024\tfr\t\n", encoding="utf-8")

    result = CliRunner().invoke(manifest.main, [
        "--guidelines", str(guidelines), "--manifest", str(path),
        "--sorting-log", str(log),
    ])

    assert result.exit_code == 0, result.output
    assert manifest.read_existing(path)["a.pdf"]["pages"] == "42", "page count not filled"
    assert "could not read" not in result.output, \
        "the PDF was opened to fill a cell that comes from the sorting log"


# --- the guideline tier ----------------------------------------------------------
# The level control on the site is ON by default, so every one of these is a question
# about what a reader who never touched a filter is allowed to find. The tier is
# derived from `doc_type`, and the two failure directions are not symmetrical: a
# document wrongly tiered "no" disappears from the default search with nothing on
# screen to explain it, while one wrongly tiered "strict" merely shows up where a
# purist would not want it. That asymmetry is why a blank doc_type yields a BLANK
# tier rather than "no", and why the test below pins it.


def test_every_doc_type_has_a_tier(manifest):
    """`_check_tiers` runs at import; this says what it is for.

    A slug added to the tier's source vocabulary and forgotten in its by_value table
    would autofill
    to nothing, and its documents would sit among the 418 other blank cells where
    nobody would spot them.
    """
    from conftest import load_lib
    load_lib("manifest_policy")._check_tiers()   # must not raise
    assert set(manifest.CLOSED_VOCABULARIES[manifest.TIER_SOURCE]) == manifest.TIER_BY_VALUE.keys()


def test_the_tiers_are_ordered_narrowest_first(manifest):
    """`build_index.py` ships this list as-is and `src/search.js` compares positions.

    Reordering these four strings changes what the site shows, so the order is a
    fact about the format rather than a detail of this module.
    """
    assert tuple(manifest.TIERS) == ("core", "related", "background", "no")


def test_a_multi_valued_doc_type_takes_the_narrowest_tier(manifest):
    """"standard;tool" is a standard that contains a tool."""
    assert manifest.tier_from_cell("standard") == "core"
    assert manifest.tier_from_cell("standard;tool") == "core"
    assert manifest.tier_from_cell("tool;standard") == "core"
    assert manifest.tier_from_cell("summary;tool") == "related"
    assert manifest.tier_from_cell("tool") == "no"


def test_an_uncurated_doc_type_yields_a_blank_tier_not_no(manifest):
    """Blank means "nobody has judged this", which the site never hides.

    A document added to the corpus is in exactly this state until it is reviewed,
    and 418 of the first 534 were until 2026-09-27, so answering "no" here would
    empty the default view of the corpus.
    """
    assert manifest.tier_from_cell("") == ""
    assert manifest.tier_from_cell("  ") == ""


def test_the_tier_is_filled_from_doc_type_and_never_overwritten(manifest, tmp_path):
    """The autofill rule, both halves of it, through the real command.

    The second half is what lets a curator disagree with the mapping: a "report"
    that really is the standard keeps the tier written into the TSV by hand.
    """
    from click.testing import CliRunner

    guidelines = tmp_path / "GUIDELINES"
    guidelines.mkdir()
    for name in ("derived.pdf", "curated.pdf", "blank.pdf"):
        (guidelines / name).write_text("not a PDF", encoding="utf-8")
    path = tmp_path / "MANIFEST.tsv"
    path.write_text(
        "file\ttitle\tdoc_type\tlevel\n"
        "derived.pdf\tUn titre\tstandard\t\n"
        "curated.pdf\tUn autre\treport\tcore\n"
        "blank.pdf\tUn troisième\t\t\n",
        encoding="utf-8")

    result = CliRunner().invoke(manifest.main, [
        "--guidelines", str(guidelines), "--manifest", str(path),
        "--sorting-log", str(tmp_path / "absent.tsv"),
    ])

    assert result.exit_code == 0, result.output
    rows = manifest.read_existing(path)
    assert rows["derived.pdf"]["level"] == "core", "not derived from doc_type"
    assert rows["curated.pdf"]["level"] == "core", "a hand-written tier was overwritten"
    assert rows["blank.pdf"]["level"] == "", "a blank doc_type must not guess a tier"


def test_a_tier_outside_the_vocabulary_stops_the_run(manifest, tmp_path):
    """Same gate as doc_type and topic: a typo fails the build, not the reader.

    "kore" in a cell would otherwise place the document at no tier the level
    control offers, and `matchesGuideline` fails open, so it would silently appear
    at every level including the narrowest.
    """
    from click.testing import CliRunner

    guidelines = tmp_path / "GUIDELINES"
    guidelines.mkdir()
    (guidelines / "a.pdf").write_text("not a PDF", encoding="utf-8")
    path = tmp_path / "MANIFEST.tsv"
    path.write_text("file\ttitle\tlevel\na.pdf\tUn titre\tkore\n", encoding="utf-8")

    result = CliRunner().invoke(manifest.main, [
        "--guidelines", str(guidelines), "--manifest", str(path),
        "--sorting-log", str(tmp_path / "absent.tsv"),
    ])

    assert result.exit_code != 0
    assert "kore" in result.output


# --- the identifiers -------------------------------------------------------------
# A DOI has no check digit and is reprinted in full inside every reference entry that
# cites the work, so reading one off a page is a guess about WHOSE it is. An ISBN has
# a check digit and appears once, on the copyright page or the back cover, so reading
# one is arithmetic. The tests below are shaped by that asymmetry, and every failing
# case in them came off a real corpus document.

CITING_PAGE = (
    "Fiche memo. Bruit de voisinage.\n"
    "1. Smith J et al. Appl Acoust 2016. doi:10.1016/j.apacoust.2016.08.005\n"
    "2. Dupont A et al. Noise Health 2019. https://doi.org/10.1016/j.noise.2019.01.002\n"
)


def test_a_doi_is_only_taken_when_the_page_prints_one(manifest):
    """Two DOIs on a page means a reference list, and the page's own DOI is unknown.

    The regression: a fiche memo prints two references in a page-1 footnote and
    came out labelled with the first one, so the card offered a link to somebody
    else's journal paper under the fiche's own title.
    """
    assert manifest.find_doi([CITING_PAGE]) == ""
    assert manifest.find_doi(["Article. doi: 10.1001/jama.2023.0589 . Published."]) \
        == "10.1001/jama.2023.0589"


def test_a_doi_without_its_marker_is_not_a_doi(manifest):
    """`10.1016/...` alone is a string; only `doi:` or doi.org makes it a claim.

    Page text is full of numbers that a bare pattern reads as a DOI. Requiring the
    marker is what keeps the extractor from inventing one out of a reference list
    that omits the prefix, and costs nothing: a document that prints its own DOI
    prints the word next to it.
    """
    assert manifest.find_doi(["Reference: 10.1016/j.apacoust.2016.08.005 (2016)"]) == ""


def test_a_doi_is_cleaned_of_what_the_text_layer_did_to_it(manifest):
    """Unicode dashes, control characters and the punctuation of the sentence.

    All three are real: `10.1136/bmj-2024-082507` came out of a BMJ PDF with a
    U+2011 in place of its hyphen and a `\\x08` after its last digit, and a DOI
    printed inside brackets keeps the closing one.
    """
    assert manifest.clean_doi("10.1136/bmj‑2024-082507") == "10.1136/bmj-2024-082507"
    assert manifest.clean_doi("10.1136/bmj-2024-082507\x08rest") == "10.1136/bmj-2024-082507"
    assert manifest.clean_doi("10.1001/jama.2023.0589).") == "10.1001/jama.2023.0589"
    assert manifest.clean_doi("10.1001/JAMA.2023.0589") == "10.1001/jama.2023.0589"
    assert manifest.clean_doi("\x08") == ""


def test_a_doi_is_looked_for_on_the_first_pages_only(manifest):
    """Page three is body text, and body text cites. The title page does not."""
    pages = ["cover", "summary", "Bibliography. doi:10.1001/jama.2023.0589"]
    assert manifest.find_doi(pages) == ""


def test_an_isbn_has_to_pass_its_check_digit(manifest):
    """Both schemes, including the ISBN-10 whose check digit is X."""
    assert manifest.valid_isbn("9782111285040")      # a French public report
    assert manifest.valid_isbn("080442957X")         # check digit X
    assert not manifest.valid_isbn("9782111285041")  # last digit bumped
    assert not manifest.valid_isbn("0804429570")
    assert not manifest.valid_isbn("12345")


def test_an_isbn_is_read_off_the_page_it_is_printed_on(manifest):
    """The marker locates it, the checksum confirms it, and the first valid one wins.

    A book that prints several (paper, PDF, EPUB) is the same work under all of
    them, so which one the column holds does not matter as long as it resolves.
    """
    assert manifest.find_isbn(["", "ISBN : 978-2-11-128504-0\nDepot legal"]) == "9782111285040"
    assert manifest.find_isbn(["ISBN-13: 978‑2‑11‑128504‑0"]) == "9782111285040"
    # A number that looks like one and is not: no link is better than a wrong one.
    assert manifest.find_isbn(["ISBN 978-2-11-128504-1"]) == ""
    assert manifest.find_isbn(["Tirage 9782111285040 exemplaires"]) == ""


def test_a_document_with_no_identifier_is_marked_as_looked_at(manifest, tmp_path, monkeypatch):
    """The sentinel, and the whole reason it exists: the second run reads nothing.

    A blank cell means "nobody has looked". Left to mean both that and "there is
    nothing there", every run of the script would re-open the ~460 documents that
    carry neither identifier, for a script whose whole point is that re-running it
    costs thirty seconds.
    """
    from click.testing import CliRunner

    guidelines = tmp_path / "GUIDELINES"
    guidelines.mkdir()
    (guidelines / "a.pdf").write_text("not a PDF at all", encoding="utf-8")
    path = tmp_path / "MANIFEST.tsv"
    path.write_text("file\ttitle\ta.pdf\tUn titre\n", encoding="utf-8")
    # The PDF scan is counted rather than read out of the log, so the test says
    # "it did not open the file" rather than "it did not print that it had".
    opened = []
    real = manifest.extract_identifier_pages
    monkeypatch.setattr(manifest, "extract_identifier_pages",
                        lambda pdf, **kw: opened.append(pdf.name) or real(pdf, **kw))
    run = lambda *extra: CliRunner().invoke(manifest.main, [
        "--guidelines", str(guidelines), "--manifest", str(path),
        "--sorting-log", str(tmp_path / "absent.tsv"), *extra,
    ])

    first = run()
    assert first.exit_code == 0, first.output
    rows = manifest.read_existing(path)
    assert rows["a.pdf"]["doi"] == manifest.NO_IDENTIFIER
    assert rows["a.pdf"]["isbn"] == manifest.NO_IDENTIFIER
    assert opened == ["a.pdf"]

    second = run()
    assert second.exit_code == 0, second.output
    assert opened == ["a.pdf"], "a sentinel means the PDF is not opened again"
    assert run("--rescan-identifiers").exit_code == 0
    assert opened == ["a.pdf", "a.pdf"], "--rescan-identifiers looks again"


def test_a_curated_identifier_is_never_re_read(manifest, tmp_path, monkeypatch):
    """Blank-only, like every other cell this script fills, even on a rescan."""
    from click.testing import CliRunner

    guidelines = tmp_path / "GUIDELINES"
    guidelines.mkdir()
    (guidelines / "a.pdf").write_text("not a PDF at all", encoding="utf-8")
    path = tmp_path / "MANIFEST.tsv"
    path.write_text("file\ttitle\tdoi\tisbn\n"
                    "a.pdf\tUn titre\t10.1001/jama.2023.0589\t9782111285040\n",
                    encoding="utf-8")
    opened = []
    monkeypatch.setattr(manifest, "extract_identifier_pages",
                        lambda pdf, **kw: opened.append(pdf.name) or [])
    result = CliRunner().invoke(manifest.main, [
        "--guidelines", str(guidelines), "--manifest", str(path),
        "--sorting-log", str(tmp_path / "absent.tsv"), "--rescan-identifiers",
    ])
    assert result.exit_code == 0, result.output
    rows = manifest.read_existing(path)
    assert rows["a.pdf"]["doi"] == "10.1001/jama.2023.0589"
    assert rows["a.pdf"]["isbn"] == "9782111285040"
    assert opened == [], "a curated cell is not a blank one, even on a rescan"


def test_a_curated_table_refuses_a_duplicated_file(manifest, tmp_path):
    """The curated tables are TSVs edited by hand; a second row for one file would
    make the answer depend on row order, so it is an error, not last-row-wins."""
    table = tmp_path / "CURATED_ISSUERS.tsv"
    table.write_text("file\tissuer\tcountry\tnote\n"
                     "a.pdf\tINSEE\tFR\t\n"
                     "a.pdf\t\t\tno issuing body\n", encoding="utf-8")
    with pytest.raises(ValueError, match="appears twice"):
        manifest.read_curated(table, columns=("issuer", "country"))
    table.write_text("file\tissuer\tcountry\tnote\na.pdf\t\t\tjournal\n", encoding="utf-8")
    assert manifest.read_curated(table, columns=("issuer", "country")) == {"a.pdf": ("", "")}


def test_page_count_is_read_off_the_pdf_when_no_log_has_it(manifest, tmp_path):
    """Without a sorting log the pages cell is counted from the file, and a file
    that cannot be opened leaves it blank for check_pdfs.py to refuse."""
    import pymupdf
    pdf = tmp_path / "three.pdf"
    with pymupdf.open() as doc:
        for _ in range(3):
            doc.new_page()
        doc.save(pdf)
    assert manifest.count_pages(pdf) == "3"
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf")
    assert manifest.count_pages(broken) == ""
