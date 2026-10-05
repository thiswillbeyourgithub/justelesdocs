# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "loguru", "pymupdf==1.28.2"]
# ///
"""Add a text layer to the corpus documents that lack one.

Some documents in the corpus are scans. Without a text layer they contribute
nothing to the index, and pdf.js cannot highlight a passage on a page that has
no text to highlight, so the search feature degrades from "wrong answer" to
"no answer at all" for that document.

Two rules shape this script:

`data/GUIDELINES/` is never modified. Its contents are the sorted corpus whose
identity is recorded in `data/SORTING_LOG.tsv`; rewriting a file there would
break that record and destroy the original. OCR output therefore goes to a
parallel directory, under the same filename.

**The OCR'd derivative must be the file that gets served.** Highlight boxes
are computed from the text layer of whatever file was chunked. If the site
served the original scan while the index described the derivative, every
highlight on those documents would be drawn at coordinates that mean nothing
on the file in the reader's browser. `chunk.py` reads this directory for the
same reason, and `deploy.sh` must ship these files in place of the originals.

Two kinds of document need work, and they need different flags:

fully scanned
    No text anywhere. Plain OCR.
partially scanned
    A text layer for the prose but one or more pages that are a single image,
    typically a decision tree or an infographic. These get `--redo-ocr`, which
    finds image regions and OCRs them while preserving the existing text.

    `--skip-text` is the obvious-looking flag here and it is wrong. It skips
    any page that contains *any* text, and the pages worth OCRing almost
    always carry a title or a caption above the image ("Arbre décisionnel
    interventions thérapeutiques", 46 characters). So it skipped precisely the
    pages that needed work: on the first run, 9 of 12 partially scanned
    documents gained nothing at all while the script reported success. The
    verification at the end of this script exists because of that.

Written with Claude Code.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

import click
import pymupdf
from loguru import logger

from lib import atomic, corpus_config, hashing
from lib import manifest_io
from lib.corpus import in_corpus

# A page with fewer than this many characters of extracted text is treated as
# having no text layer. Not zero, because a scanned page often carries a few
# characters of stray metadata or a single caption, and a title alone ("Arbre
# décisionnel interventions thérapeutiques") is exactly the case worth OCRing.
MIN_PAGE_CHARS = 50

# A text-less page is only worth OCRing if it actually carries content. A page
# whose largest image covers this fraction of the sheet, or which holds this
# many vector drawing operations, has something on it: a decision tree, an
# infographic, a scanned figure. A text-less page with neither is a blank
# verso or a section divider, and OCRing it yields nothing. Requiring content
# as well as absent text is what took the candidate list from 23 documents to
# 14: the 504-page addictions report alone has 32 genuinely blank pages.
MIN_IMAGE_AREA_SHARE = 0.10
MIN_VECTOR_DRAWINGS = 25

# Tesseract language packs, keyed by the manifest's two-letter language code, and
# the argument for a document whose language was never determined (every pack:
# less accurate than a single one but better than guessing wrong). Both are the
# corpus's own, from corpus.toml's [languages.<code>] tesseract and [ocr] default.
TESSERACT_LANGS: dict[str, str] = {
    code: language["tesseract"] for code, language in corpus_config.section("languages").items()}
DEFAULT_LANG: str = corpus_config.section("ocr")["default"]


def page_text_lengths(pdf: Path) -> list[int]:
    """Characters of extractable text per page.

    Parameters
    ----------
    pdf : Path
        The PDF to inspect.

    Returns
    -------
    list of int
        One entry per page, in order.
    """
    with pymupdf.open(pdf) as doc:
        return [len(page.get_text().strip()) for page in doc]


def page_has_content(page: pymupdf.Page) -> bool:
    """Whether a page carries visible content beyond its text layer.

    Used to separate a page that needs OCR (an image-only decision tree) from
    one that does not (a blank verso). See MIN_IMAGE_AREA_SHARE.

    Parameters
    ----------
    page : pymupdf.Page
        The page to inspect.

    Returns
    -------
    bool
        True when the page holds a large image or substantial vector drawing.
    """
    area = abs(page.rect.get_area()) or 1.0
    for info in page.get_image_info():
        if abs(pymupdf.Rect(info["bbox"]).get_area()) / area > MIN_IMAGE_AREA_SHARE:
            return True
    return len(page.get_drawings()) > MIN_VECTOR_DRAWINGS


def classify(pdf: Path) -> tuple[str, int]:
    """Decide what OCR treatment a document needs.

    Parameters
    ----------
    pdf : Path
        The PDF to inspect.

    Returns
    -------
    tuple of (str, int)
        The treatment ("full" when no page has text, "partial" when some
        content-bearing page lacks text, "none" otherwise) and the number of
        pages that would gain a text layer.
    """
    with pymupdf.open(pdf) as doc:
        if not doc.page_count:
            return "none", 0
        textless = [p for p in doc if len(p.get_text().strip()) < MIN_PAGE_CHARS]
        if len(textless) == doc.page_count:
            return "full", doc.page_count
        # Blank pages are text-less too, and OCRing them recovers nothing, so
        # only content-bearing pages count towards needing work.
        needing = sum(1 for page in textless if page_has_content(page))
    return ("partial", needing) if needing else ("none", 0)


# Everything `classify` decides with, so a changed threshold re-classifies the whole
# corpus rather than trusting verdicts reached under the old one. Bump the trailing
# version when the LOGIC changes without a threshold changing.
CLASSIFY_KEY = f"{MIN_PAGE_CHARS}|{MIN_IMAGE_AREA_SHARE}|{MIN_VECTOR_DRAWINGS}|1"


def classify_cached(pdf: Path, cache: dict[str, dict]) -> tuple[str, int]:
    """`classify`, remembered per file content.

    `classify` walks every page of every document, text extraction plus image and
    drawing inspection, and that was the whole cost of a deploy's OCR step: about
    340 s of CPU over 534 PDFs to conclude, every time, that the same 49 documents
    were already done. Hashing the bytes is about a second for the corpus, so an
    unchanged file is never opened again.

    Keyed by content, not by mtime: the corpus lives under Syncthing, which can
    rewrite an mtime without touching a byte, and a new file under an old name
    must still be looked at.

    Parameters
    ----------
    pdf : Path
        The PDF to classify.
    cache : dict
        Filename to `{"sha256", "key", "mode", "pages"}`. Updated in place.

    Returns
    -------
    tuple of (str, int)
        As `classify`.
    """
    sha = hashing.file_sha256(pdf)
    entry = cache.get(pdf.name)
    if entry and entry.get("sha256") == sha and entry.get("key") == CLASSIFY_KEY:
        return entry["mode"], entry["pages"]
    mode, pages = classify(pdf)
    cache[pdf.name] = {"sha256": sha, "key": CLASSIFY_KEY, "mode": mode, "pages": pages}
    return mode, pages


def read_languages(manifest: Path) -> dict[str, str]:
    """Map filename to tesseract language string, from the manifest.

    Parameters
    ----------
    manifest : Path
        The metadata manifest.

    Returns
    -------
    dict
        Filename to tesseract language argument. Missing or unknown languages
        are omitted so the caller can apply its own default.
    """
    if not manifest.exists():
        logger.warning(f"no manifest at {manifest}, defaulting every language to {DEFAULT_LANG}")
        return {}
    return {
        row["file"]: TESSERACT_LANGS[row["language"]]
        for row in manifest_io.read_rows(manifest)
        if row.get("language") in TESSERACT_LANGS
    }


def ocr_flags(mode: str) -> list[str]:
    """ocrmypdf flags for a treatment mode.

    Parameters
    ----------
    mode : str
        "full" or "partial".

    Returns
    -------
    list of str
        Flags to pass. --output-type pdf keeps ocrmypdf from routing the file
        through Ghostscript for PDF/A conversion, which re-encodes every page;
        for a partially scanned document that would rewrite the existing, good
        text layer whose coordinates the search highlights depend on.
    """
    flags = ["--output-type", "pdf"]
    if mode == "partial":
        flags.append("--redo-ocr")
    return flags


# Documents ocrmypdf cannot process, with what it said. Named one by one rather than
# caught by a broad "on failure, carry on": an OCR failure normally means the build is
# about to ship a document nobody can search, which is worth stopping for. These are the
# exceptions where stopping is wrong, because the pages at stake are few and the
# original text layer is fine everywhere else.
UNOCRABLE: dict[str, str] = {
    "Child and Adolescent Clinical Psychopharmacology - John Preston.pdf":
        "ocrmypdf exit 4, 'the generated PDF is INVALID': the source has broken "
        "indirect references qpdf cannot repair. 3 scanned pages out of 137 stay "
        "without a text layer, and the other 134 are unaffected.",
    "Difficult to Treat Depression - A Carlat Guide - Chris Aiken.pdf":
        "ocrmypdf exit 4, same broken indirect references. 3 scanned pages out of 367.",
}


def source_digest(pdf: Path, language: str, mode: str) -> str:
    """Hash gating a cached OCR result.

    Covers the source bytes plus the arguments that change the output, so
    switching a document's language re-runs OCR while re-running the script
    unchanged does not. OCR costs seconds per page, so this gate is what keeps
    the build incremental.

    Parameters
    ----------
    pdf : Path
        Source PDF.
    language : str
        Tesseract language argument used.
    mode : str
        "full" or "partial".

    Returns
    -------
    str
        Hex digest.
    """
    digest = hashlib.sha256()
    digest.update(pdf.read_bytes())
    # The flags are part of the key: changing how a mode is OCR'd must
    # invalidate its cached output, which is what makes fixing a wrong flag
    # actually re-run rather than silently keep the bad derivative.
    digest.update(f"{language}|{mode}|{' '.join(ocr_flags(mode))}".encode())
    return digest.hexdigest()


def original_sha256(pdf: Path) -> str:
    """The sha256 of an ORIGINAL's bytes, as recorded beside its OCR copy.

    Parameters
    ----------
    pdf : Path
        A document under `data/GUIDELINES/`.

    Returns
    -------
    str
        Hex digest.

    Notes
    -----
    `chunk.py` trusts an OCR copy only when the index records this digest and it
    still matches the original on disk. `src_hash` cannot serve that purpose: it
    also folds in the language and the flags, which `chunk.py` has no business
    re-deriving. Without the check, a scanned original replaced under the same
    name by a version with a text layer is classified "none" here, skipped, and
    its OLD copy keeps being chunked and served in place of the new document.
    """
    return hashing.file_sha256(pdf)


def write_index(path: Path, index: dict[str, dict[str, str]]) -> None:
    """Record which derivative came from which source bytes under which flags.

    Parameters
    ----------
    path
        `<out>/ocr_index.json`.
    index
        Filename to its cache entry, as the run has it now. Entries are added only
        after a document's OCR has been verified, so writing a partial index
        records exactly the work that is safely on disk.

    Notes
    -----
    Separate from `main` so it can be called from a `finally`: this file is what
    makes the run incremental, and the alternative (writing it once the loop has
    finished) discards every completed document as soon as one fails.
    """
    atomic.write_text_atomic(path, json.dumps(index, indent=2, sort_keys=True))


@click.command()
@click.option("--guidelines", type=click.Path(exists=True, file_okay=False, path_type=Path),
              **in_corpus("data/GUIDELINES"),
              help="Corpus root. Read only: this script never writes here.")
@click.option("--out", type=click.Path(path_type=Path), **in_corpus("data/OCR"),
              help="Where derivatives are written, under the original filename.")
@click.option("--manifest", type=click.Path(path_type=Path), **in_corpus("data/MANIFEST.tsv"), help="Source of each document's language.")
@click.option("--jobs", type=int, default=4, show_default=True,
              help="Passed to ocrmypdf --jobs.")
@click.option("--ocrmypdf-cmd", default="uvx ocrmypdf", show_default=True,
              help="How to invoke ocrmypdf. With the uvx default, UV_TOOL_DIR "
                   "must be writable.")
@click.option("--force", is_flag=True, help="Re-run OCR even when the hash matches.")
@click.option("--dry-run", is_flag=True, help="Report what would be done and stop.")
def main(guidelines: Path, out: Path, manifest: Path, jobs: int, ocrmypdf_cmd: str,
         force: bool, dry_run: bool) -> None:
    """OCR the corpus documents that have missing or incomplete text layers."""
    languages = read_languages(manifest)
    index_path = out / "ocr_index.json"
    index: dict[str, dict[str, str]] = {}
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))

    cache_path = out / "classify_cache.json"
    cache: dict[str, dict] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    scanned = 0
    todo: list[tuple[Path, str, str, int, str]] = []
    # Every original that still wants an OCR copy. A copy whose original is not in
    # this set (deleted, renamed, replaced by a version with a text layer, or now
    # listed in UNOCRABLE) is stale and is removed below: chunk.py would otherwise
    # keep preferring it, and stage.py keep serving it, over the current original.
    wanted: set[str] = set()
    index_changed = False
    for pdf in sorted(guidelines.glob("*.pdf")):
        mode, pages = classify_cached(pdf, cache)
        if mode == "none":
            continue
        if pdf.name in UNOCRABLE:
            logger.warning(f"skipping {pdf.name}: {UNOCRABLE[pdf.name]}")
            continue
        scanned += 1
        wanted.add(pdf.name)
        language = languages.get(pdf.name, DEFAULT_LANG)
        digest = source_digest(pdf, language, mode)
        # Decided HERE rather than inside the OCR loop, so the list printed below is
        # the work that will actually run. It used to list every scanned document and
        # skip them all afterwards, which read on every deploy as 49 documents being
        # OCR'd again.
        current = (out / pdf.name).exists() and index.get(pdf.name, {}).get("src_hash") == digest
        if force or not current:
            todo.append((pdf, mode, language, pages, digest))
        elif "original_sha256" not in index[pdf.name]:
            # Entries written before 2026-10-03 lack the field chunk.py now checks.
            # `current` means `src_hash` matched, and `src_hash` covers these exact
            # bytes, so recording their digest now vouches for nothing new.
            index[pdf.name]["original_sha256"] = original_sha256(pdf)
            index_changed = True
    # The verdicts are kept even on a dry run: they are facts about the files, not
    # work done, and a dry run is often the first run on a fresh checkout.
    out.mkdir(parents=True, exist_ok=True)
    cache = {name: entry for name, entry in cache.items() if (guidelines / name).exists()}
    atomic.write_text_atomic(cache_path, json.dumps(cache, indent=2, sort_keys=True))

    stale = sorted(path for path in out.glob("*.pdf") if path.name not in wanted)
    for path in stale:
        if dry_run:
            logger.info(f"would remove stale OCR copy {path.name}: its original no longer needs one")
            continue
        logger.warning(f"removing stale OCR copy {path.name}: its original no longer needs one")
        path.unlink()
        index.pop(path.name, None)
        index_changed = True
    if index_changed and not dry_run:
        write_index(index_path, index)

    if not scanned:
        logger.success("every document already has a complete text layer")
        return
    if not todo:
        logger.success(f"all {scanned} documents with scanned pages already have a current OCR text layer")
        return
    for pdf, mode, language, pages, _ in todo:
        logger.info(f"{mode:<7} -l {language:<8} {pages:>3} page(s)  {pdf.name}")
    total = sum(pages for *_, pages, _ in todo)
    logger.info(f"{len(todo)} of {scanned} documents with scanned pages need OCR, {total} pages")
    if dry_run:
        return

    done = 0
    failed: list[str] = []
    # Written whatever happens, because the loop below exits the process on a
    # failed or an empty OCR. The index is the only record of which derivative
    # came from which source bytes under which flags, so losing it after ten
    # documents means re-OCRing the nine that succeeded, whose derivatives are
    # sitting on disk. OCR costs seconds per page; that is the expensive kind of
    # work to throw away. Enumerating the exits instead would rot the first time
    # somebody adds a third one.
    try:
        for pdf, mode, language, pages, digest in todo:
            destination = out / pdf.name
            before = page_text_lengths(pdf)
            command = [*shlex.split(ocrmypdf_cmd), "-l", language, "--jobs", str(jobs),
                       *ocr_flags(mode), str(pdf), str(destination)]

            logger.info(f"running: {shlex.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"ocrmypdf failed on {pdf.name} (exit {result.returncode})")
                logger.error(result.stderr.strip()[-1500:])
                # ocrmypdf writes the file before deciding it is invalid, and what it leaves
                # behind is preferred over the original by chunk.py and served by stage.py.
                # A failed run must not leave a broken derivative where a good original was.
                destination.unlink(missing_ok=True)
                sys.exit(1)

            after = page_text_lengths(destination)
            # OCR that produced no text is a silent failure: the file exists, the
            # build looks clean, and the document is still invisible to search.
            if sum(after) == 0:
                logger.error(f"{pdf.name}: OCR produced a file with no text at all")
                sys.exit(1)
            # The check that matters. A total character count always goes up after
            # OCR, so it cannot tell you whether the PAGES you targeted gained
            # anything. Count pages that crossed the threshold instead, and refuse
            # to record success when none did: that is exactly the failure the
            # --skip-text flag caused, invisibly, on the first run.
            gained = sum(1 for b, a in zip(before, after) if b < MIN_PAGE_CHARS <= a)
            # A fully scanned document that gains nothing is unsearchable, full
            # stop, and that is a build failure. A partially scanned one that gains
            # nothing is usually correct: the page really has no words on it. The
            # four such pages in this corpus are a stylised cover title, a sideways
            # reference string, a pure illustration and a back-cover logo, and
            # tesseract run directly on them returns 0 to 7 junk words. The image
            # test in `page_has_content` cannot tell a decorative image from a
            # decision tree, so the shortfall is recorded rather than treated as
            # an error, and stays visible in the index for auditing.
            if gained == 0 and mode == "full":
                logger.error(f"{pdf.name}: fully scanned and OCR gained no text at all. "
                             f"This document cannot be searched.")
                failed.append(pdf.name)
                continue
            if gained < pages:
                logger.warning(f"{pdf.name}: {gained} of {pages} targeted pages gained "
                               f"text; the others carry images without readable words")
            index[pdf.name] = {"src_hash": digest, "mode": mode, "language": language,
                               "pages_gained": gained, "pages_targeted": pages,
                               "original_sha256": original_sha256(pdf)}
            logger.success(f"{pdf.name}: {gained}/{pages} pages gained text, "
                           f"{sum(after)} chars total")
            done += 1
    finally:
        write_index(index_path, index)

    logger.success(f"{done} documents OCR'd, {scanned - len(todo)} already current")
    if failed:
        logger.error(f"{len(failed)} fully scanned documents gained no text: "
                     f"{', '.join(failed)}")
        sys.exit(1)
    logger.warning("These derivatives, not the originals, must be the files served "
                   "and the files chunked. chunk.py reads them automatically; "
                   "deploy.sh must ship them in place of the originals.")


if __name__ == "__main__":
    main()
