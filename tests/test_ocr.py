"""What ocr.py does when ocrmypdf refuses a document.

The bug these cover: ocrmypdf failed on a 137-page book with "the generated PDF is
INVALID" and exit 4, and the 1.6 MB broken file it had already written stayed in
data/OCR. That directory is not a cache, it is a source: chunk.py reads the derivative
in preference to the original and stage.py serves it, so a failed OCR run left the site
about to ship a document no reader could open.

The tests drive the real command through `--ocrmypdf-cmd`, which is how the script lets
a caller say where ocrmypdf lives, so nothing here needs tesseract installed.

Written by Claude Code (Opus 5).
"""

from __future__ import annotations

import json

import pytest

from click.testing import CliRunner

pymupdf = pytest.importorskip("pymupdf")


def image_only_pdf(path):
    """A two-page PDF with no text layer, which is what `classify` looks for."""
    doc = pymupdf.open()
    for _ in range(2):
        page = doc.new_page()
        page.draw_rect(pymupdf.Rect(50, 50, 300, 300), fill=(0.6, 0.6, 0.6))
    doc.save(path)
    doc.close()


@pytest.fixture
def corpus(tmp_path):
    """A corpus of one scanned document, and the manifest naming its language."""
    guidelines = tmp_path / "GUIDELINES"
    guidelines.mkdir()
    image_only_pdf(guidelines / "scanned.pdf")
    manifest = tmp_path / "MANIFEST.tsv"
    manifest.write_text("file\tlanguage\nscanned.pdf\tfr\n", encoding="utf-8")
    return guidelines, manifest, tmp_path / "OCR"


def run(ocr, corpus, command):
    guidelines, manifest, out = corpus
    return CliRunner().invoke(ocr.main, [
        "--guidelines", str(guidelines), "--manifest", str(manifest),
        "--out", str(out), "--ocrmypdf-cmd", command,
    ])


def test_a_failed_run_leaves_no_half_written_derivative(ocr, corpus, tmp_path):
    # ocrmypdf writes the file and THEN decides it is invalid, so "it failed" and "it
    # wrote something" are both true at once. Reproduced with a command that does the
    # same: write a file, then fail.
    out = corpus[2]
    # The destination is the LAST argument, wherever ocr.py puts it in the line.
    # "$6" happened to name the string "pdf", so this test wrote a file called pdf
    # into the working directory and passed because the derivative it was looking
    # for had never been written anywhere near the place it looked.
    failing = "sh -c 'for last; do :; done; printf broken > \"$last\"; exit 4' --"
    result = run(ocr, corpus, failing)
    assert result.exit_code != 0
    assert not (out / "scanned.pdf").exists(), "a broken derivative was left behind"


def text_pdf(path, pages=2):
    """A PDF whose every page carries well over MIN_PAGE_CHARS of text.

    Stands in for what ocrmypdf hands back: the verification in `main` counts pages
    that crossed the threshold, so a derivative has to really carry text.
    """
    doc = pymupdf.open()
    for number in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {number} du document, avec assez de texte "
                                   "pour franchir le seuil de cinquante caracteres.")
    doc.save(path)
    doc.close()


def test_a_failure_keeps_the_index_of_what_already_succeeded(ocr, tmp_path):
    """OCR costs seconds per page, so a late failure must not discard early work.

    The index is the only record of which derivative came from which source bytes,
    and it used to be written after the loop that exits the process on a failure:
    document ten failing threw away the nine that had succeeded, whose derivatives
    were sitting on disk, so the next run re-OCR'd all of them.
    """
    guidelines = tmp_path / "GUIDELINES"
    guidelines.mkdir()
    image_only_pdf(guidelines / "a-works.pdf")
    image_only_pdf(guidelines / "b-fails.pdf")
    manifest = tmp_path / "MANIFEST.tsv"
    manifest.write_text("file\tlanguage\na-works.pdf\tfr\nb-fails.pdf\tfr\n",
                        encoding="utf-8")
    out = tmp_path / "OCR"
    ocred = tmp_path / "ocred.pdf"
    text_pdf(ocred)

    # `for last; do :; done` leaves the destination in $last whatever flags the
    # script decided to pass, which is steadier than counting positions.
    command = ("sh -c 'for last; do :; done; "
               f'case "$*" in *b-fails*) exit 4;; esac; cp "{ocred}" "$last"\' --')
    result = CliRunner().invoke(ocr.main, [
        "--guidelines", str(guidelines), "--manifest", str(manifest),
        "--out", str(out), "--ocrmypdf-cmd", command,
    ])

    assert result.exit_code != 0, "the failing document should still fail the run"
    index = json.loads((out / "ocr_index.json").read_text(encoding="utf-8"))
    assert "a-works.pdf" in index, "the completed document was dropped from the index"
    assert index["a-works.pdf"]["pages_gained"] == 2
    assert "b-fails.pdf" not in index, "a failed document must not look cached"
    # And the work it records is really on disk, so a re-run can trust the hash.
    assert (out / "a-works.pdf").exists()
    assert not (out / "b-fails.pdf").exists()


def test_a_document_ocrmypdf_cannot_handle_is_skipped_by_name(ocr, corpus, monkeypatch):
    # The escape hatch for the document above: named one by one, because a blanket
    # "carry on after a failure" would hide the case where OCR failing means the
    # document is about to ship unsearchable.
    monkeypatch.setitem(ocr.UNOCRABLE, "scanned.pdf", "cannot be repaired by qpdf")
    # loguru's sink is bound to the real stderr at import, so neither CliRunner nor
    # capsys sees it: the messages are collected from loguru itself.
    said: list[str] = []
    sink = ocr.logger.add(said.append, format="{message}")
    try:
        # A command that would fail loudly if it ran at all, so the test cannot pass by
        # the OCR merely succeeding.
        result = run(ocr, corpus, "sh -c 'exit 9' --")
    finally:
        ocr.logger.remove(sink)
    assert result.exit_code == 0, result.output
    assert any("skipping scanned.pdf" in line for line in said), said
    assert not (corpus[2] / "scanned.pdf").exists()


def test_an_unchanged_corpus_is_neither_reclassified_nor_announced(ocr, tmp_path, monkeypatch):
    """A deploy with nothing new must not re-inspect every page, nor claim to OCR.

    Every deploy used to spend about 340 s of CPU in `classify`, walking all 534
    PDFs page by page to conclude that the same 49 were already done, and it logged
    those 49 as needing a text layer before skipping them, which read as OCR running
    again on every deploy.
    """
    guidelines = tmp_path / "GUIDELINES"
    guidelines.mkdir()
    image_only_pdf(guidelines / "scanned.pdf")
    manifest = tmp_path / "MANIFEST.tsv"
    manifest.write_text("file\tlanguage\nscanned.pdf\tfr\n", encoding="utf-8")
    out = tmp_path / "OCR"
    ocred = tmp_path / "ocred.pdf"
    text_pdf(ocred)
    args = ["--guidelines", str(guidelines), "--manifest", str(manifest), "--out", str(out),
            "--ocrmypdf-cmd", f'sh -c \'for last; do :; done; cp "{ocred}" "$last"\' --']
    first = CliRunner().invoke(ocr.main, args)
    assert first.exit_code == 0, first.output

    def must_not_run(pdf):
        raise AssertionError(f"classify re-ran on unchanged {pdf.name}")
    monkeypatch.setattr(ocr, "classify", must_not_run)
    said: list[str] = []
    sink = ocr.logger.add(said.append, format="{message}")
    try:
        # ocrmypdf fails if it is called at all, so a re-OCR cannot pass silently.
        second = CliRunner().invoke(ocr.main, [*args[:-1], "sh -c 'exit 9' --"])
    finally:
        ocr.logger.remove(sink)
    assert second.exit_code == 0, second.output
    assert not any("need OCR" in line or "page(s)" in line for line in said), said
    assert any("already have a current OCR text layer" in line for line in said), said


def test_a_changed_file_is_classified_again(ocr, tmp_path):
    """The classification cache is keyed by content, so new bytes are looked at."""
    pdf = tmp_path / "doc.pdf"
    image_only_pdf(pdf)
    cache: dict = {}
    assert ocr.classify_cached(pdf, cache) == ("full", 2)
    text_pdf(pdf)
    assert ocr.classify_cached(pdf, cache) == ("none", 0)


# --- a copy must belong to the original on disk ------------------------------
# chunk.py prefers data/OCR/<name> over the original, and stage.py serves what
# chunk.py chunked. Until 2026-10-03 nothing removed a copy, and nothing checked
# that a copy was made from the original now on disk: replace a scanned document
# by a version with a text layer, under the same name, and ocr.py classified it
# "none" and moved on while the OLD copy went on being chunked and served.


def ocr_once(ocr, tmp_path):
    """Run ocr.py over one scanned document with a fake ocrmypdf; return its paths."""
    guidelines = tmp_path / "GUIDELINES"
    guidelines.mkdir()
    image_only_pdf(guidelines / "scanned.pdf")
    manifest = tmp_path / "MANIFEST.tsv"
    manifest.write_text("file\tlanguage\nscanned.pdf\tfr\n", encoding="utf-8")
    out = tmp_path / "OCR"
    ocred = tmp_path / "ocred.pdf"
    text_pdf(ocred)
    args = ["--guidelines", str(guidelines), "--manifest", str(manifest), "--out", str(out),
            "--ocrmypdf-cmd", f'sh -c \'for last; do :; done; cp "{ocred}" "$last"\' --']
    result = CliRunner().invoke(ocr.main, args)
    assert result.exit_code == 0, result.output
    return guidelines, out, args


def test_the_index_records_the_original_a_copy_was_made_from(ocr, tmp_path):
    import hashlib
    guidelines, out, _ = ocr_once(ocr, tmp_path)
    index = json.loads((out / "ocr_index.json").read_text(encoding="utf-8"))
    want = hashlib.sha256((guidelines / "scanned.pdf").read_bytes()).hexdigest()
    assert index["scanned.pdf"]["original_sha256"] == want


def test_a_copy_whose_original_gained_a_text_layer_is_removed(ocr, tmp_path):
    guidelines, out, args = ocr_once(ocr, tmp_path)
    text_pdf(guidelines / "scanned.pdf")  # same name, now born digital
    result = CliRunner().invoke(ocr.main, args)
    assert result.exit_code == 0, result.output
    assert not (out / "scanned.pdf").exists()
    index = json.loads((out / "ocr_index.json").read_text(encoding="utf-8"))
    assert "scanned.pdf" not in index


def test_an_index_from_before_the_field_is_backfilled_without_ocr(ocr, tmp_path):
    guidelines, out, args = ocr_once(ocr, tmp_path)
    path = out / "ocr_index.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    del index["scanned.pdf"]["original_sha256"]
    path.write_text(json.dumps(index), encoding="utf-8")
    # ocrmypdf fails if called, so the backfill provably did not re-OCR.
    result = CliRunner().invoke(ocr.main, [*args[:-1], "sh -c 'exit 9' --"])
    assert result.exit_code == 0, result.output
    assert "original_sha256" in json.loads(path.read_text(encoding="utf-8"))["scanned.pdf"]


def test_chunk_refuses_a_copy_not_made_from_the_original_on_disk(ocr, chunk, tmp_path):
    import click
    guidelines, out, _ = ocr_once(ocr, tmp_path)
    original, copy = guidelines / "scanned.pdf", out / "scanned.pdf"
    chunk.vouched_for(copy, original, chunk.read_ocr_index(out))  # made from it: accepted
    text_pdf(original)  # replaced, and ocr.py not re-run yet
    with pytest.raises(click.ClickException, match="was not made from"):
        chunk.vouched_for(copy, original, chunk.read_ocr_index(out))
