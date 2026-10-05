"""The gate that refuses a corpus file the browser cannot open.

The bug it exists for: a downloaded `2013-01-07_rapport_logement_web2009.pdf` was
the publisher website's error page saved with a .pdf extension. pymupdf opened it and
reported four pages of navigation menu, which were chunked, embedded and served,
and the reader who clicked the result got "Invalid PDF structure" from pdf.js. So
the cases below are written around what a WRONG file looks like rather than around
what a PDF looks like.

Written by Claude Code (Opus 5).
"""

from __future__ import annotations

import pytest

HTML_ERROR_PAGE = b"""<!DOCTYPE html>
<html lang="fr"><head><title>Page introuvable</title></head>
<body><div>Recherche Menu principal Contenu principal Professionnels</div></body></html>
"""


@pytest.fixture
def corpus(tmp_path):
    """A directory standing in for the served corpus folder."""
    return tmp_path


def test_a_web_page_saved_as_a_pdf_is_named_as_one(check_pdfs, corpus):
    bad = corpus / "rapport_logement_web.pdf"
    bad.write_bytes(HTML_ERROR_PAGE)
    assert check_pdfs.why_unusable(bad) == "an HTML page saved as .pdf"


def test_a_file_with_no_header_is_refused_without_being_parsed(check_pdfs, corpus):
    bad = corpus / "truncated.pdf"
    bad.write_bytes(b"\x00\x01\x02 not a document at all")
    assert check_pdfs.why_unusable(bad) == "no %PDF- header"


def test_a_real_pdf_passes(check_pdfs, corpus):
    pymupdf = pytest.importorskip("pymupdf")
    good = corpus / "recommandation.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Rénovation du parc de logements")
    doc.save(good)
    doc.close()
    assert check_pdfs.why_unusable(good) is None


def test_a_pdf_with_no_pages_is_refused(check_pdfs, corpus):
    # pdf.js renders nothing and the viewer's page controls have nothing to point
    # at, so an empty document is as unusable as an unparseable one. Written by
    # hand because pymupdf refuses to SAVE a document with no pages, while opening
    # one without complaint, which is the asymmetry this check covers.
    empty = corpus / "empty.pdf"
    empty.write_bytes(b"%PDF-1.4\n"
                      b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
                      b"2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\n"
                      b"trailer<</Root 1 0 R>>\n%%EOF\n")
    assert check_pdfs.why_unusable(empty) == "has no pages"


def write_pdf(path, lines, *, pages=1):
    """A PDF carrying `lines` on its first page, and blank pages after it."""
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    for index in range(pages):
        page = doc.new_page()
        if index == 0:
            for offset, line in enumerate(lines):
                page.insert_text((72, 100 + 14 * offset), line)
    doc.save(path)
    doc.close()


# Enough text to clear FINGERPRINT_MINIMUM, which exists to keep image-only scans
# from all looking alike.
BODY = [f"Recommandation de bonne pratique, paragraphe {i} du texte." for i in range(8)]


def test_the_same_document_under_two_names_is_reported(check_pdfs, corpus):
    """The real case: one report filed under its own name and under its publisher's."""
    write_pdf(corpus / "rapport_logement_social.pdf", BODY, pages=3)
    write_pdf(corpus / "INSEE.pdf", BODY, pages=3)
    assert check_pdfs.same_document(sorted(corpus.glob("*.pdf"))) == [
        ["INSEE.pdf", "rapport_logement_social.pdf"]]


def test_two_documents_sharing_a_page_count_are_not_duplicates(check_pdfs, corpus):
    write_pdf(corpus / "logement.pdf", BODY, pages=3)
    write_pdf(corpus / "transport.pdf", [line.replace("paragraphe", "section") for line in BODY],
              pages=3)
    assert check_pdfs.same_document(sorted(corpus.glob("*.pdf"))) == []


def test_two_image_only_scans_are_not_called_copies_of_each_other(check_pdfs, corpus):
    """They extract to nothing, so hashing their text would group every scan."""
    write_pdf(corpus / "scan_a.pdf", [], pages=2)
    write_pdf(corpus / "scan_b.pdf", [], pages=2)
    assert check_pdfs.same_document(sorted(corpus.glob("*.pdf"))) == []


def test_a_different_page_count_is_never_compared(check_pdfs, corpus):
    """Two printings of one document share their page count; the filter rests on it."""
    write_pdf(corpus / "short.pdf", BODY, pages=2)
    write_pdf(corpus / "long.pdf", BODY, pages=3)
    assert check_pdfs.same_document(sorted(corpus.glob("*.pdf"))) == []


def test_a_missing_file_is_reported_rather_than_raised(check_pdfs, corpus):
    reason = check_pdfs.why_unusable(corpus / "absent.pdf")
    assert reason and reason.startswith("cannot be read")
