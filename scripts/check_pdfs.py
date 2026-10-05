# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "loguru", "pymupdf==1.28.2"]
# ///
"""Refuse a corpus holding a file the browser will not open.

The site renders every result in pdf.js, so a file that pdf.js rejects is a
document a reader can find, click and then be told "Impossible d'afficher ce
document : Invalid PDF structure." about. That happened to
a file in the first corpus, which is not a PDF at
all: it is an agency website's error page, saved with a .pdf extension. Nothing
caught it, and everything downstream went along with it, because pymupdf opens
HTML and reported four pages of website furniture ("Recherche Menu principal
Contenu principal Professionnels...") which were duly chunked, embedded and made
searchable.

So the test here is not "does something open it" but "is it the format it claims
to be": pymupdf's own `is_pdf`, plus the magic bytes, plus the things pdf.js
refuses that pymupdf tolerates (no pages, a password).

Why a separate gate rather than a check inside chunk.py: chunk.py caches per
document and skips what has not changed, so a bad file that is already cached is
never looked at again. This walks the corpus every time, in a second.

    uv run scripts/check_pdfs.py

It also reports, without failing, the documents that appear to be filed twice:
same page count and the same four opening pages. Three pairs are in the corpus
today, and each is the same guidance reachable under two names, which means a
reader gets one answer twice and half the copies of it are not the one the
manifest curated a title for.

Exits non-zero, listing what is wrong with which file, so it can sit in the build
chain next to verify_chunks.py. It never deletes anything: which document to drop
or re-download is a decision about the corpus, not about the build.

Written by Claude Code.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path

import click
import pymupdf
from loguru import logger
from lib.corpus import in_corpus

# How much of the file to read when looking for the header. The marker is meant to
# be at byte 0, but a stray BOM or a few bytes of junk before it are common and
# tolerated by every reader, so a window rather than a prefix comparison.
MAGIC_WINDOW = 1024


# How much of a document's opening text identifies it. Four pages because a cover
# page alone is often just a logo and a year, and 4000 characters because two
# printings of one document can differ in a footer or a revision date further in.
FINGERPRINT_PAGES = 4
FINGERPRINT_CHARS = 4000

# Below this many characters there is nothing to compare: an image-only scan
# extracts to the empty string, and every image-only scan in the corpus would
# otherwise be reported as a copy of every other one.
FINGERPRINT_MINIMUM = 200


def opening_fingerprint(path: Path) -> str | None:
    """Hash a document's opening text, or None when it has too little to hash.

    Parameters
    ----------
    path : Path
        A PDF in the corpus.

    Returns
    -------
    str or None
        A hex digest of the normalised first `FINGERPRINT_CHARS` characters of the
        first `FINGERPRINT_PAGES` pages. None for a file that cannot be opened or
        that carries no text layer yet.
    """
    try:
        with pymupdf.open(path) as doc:
            text = " ".join(doc[i].get_text() for i in range(min(FINGERPRINT_PAGES, doc.page_count)))
    except Exception:
        return None
    text = re.sub(r"\s+", " ", text).strip().lower()
    if len(text) < FINGERPRINT_MINIMUM:
        return None
    return hashlib.sha256(text[:FINGERPRINT_CHARS].encode("utf-8")).hexdigest()


def same_document(paths: list[Path]) -> list[list[str]]:
    """Group files that look like the same document filed twice.

    Parameters
    ----------
    paths : list of Path
        The corpus, one path per file.

    Returns
    -------
    list of list of str
        One group of two or more file names per apparent duplicate, sorted.

    Notes
    -----
    Page count is the pre-filter: two printings of one document always share it,
    and comparing only within a page count turns a 535-document comparison into a
    handful of small ones. The opening text then decides, because two documents
    can share a page count by coincidence and no two share four pages of prose.

    Reported, never acted on. Which copy of a document to drop is a decision about
    the corpus (the filenames are what `data/SORTING_LOG.tsv`, the chunk cache and
    `dist/pdf/` join on, so dropping one is not a delete), and this script exists to
    say what is there.
    """
    by_pages: dict[int, list[Path]] = defaultdict(list)
    for path in paths:
        try:
            with pymupdf.open(path) as doc:
                by_pages[doc.page_count].append(path)
        except Exception:
            continue
    groups: list[list[str]] = []
    for _pages, candidates in by_pages.items():
        if len(candidates) < 2:
            continue
        seen: dict[str, list[str]] = defaultdict(list)
        for path in candidates:
            digest = opening_fingerprint(path)
            if digest:
                seen[digest].append(path.name)
        groups.extend(sorted(names) for names in seen.values() if len(names) > 1)
    return sorted(groups)


def why_unusable(path: Path) -> str | None:
    """Say why a file cannot be served as a PDF, or None when it can.

    Parameters
    ----------
    path : Path
        The file to inspect. Read only.

    Returns
    -------
    str or None
        A one-line reason, phrased for a build log, or None when the file is a
        PDF the viewer can open.

    Notes
    -----
    The checks are ordered by how much they read: the magic bytes settle the
    HTML-saved-as-PDF case without parsing anything, and only then is the file
    handed to pymupdf.
    """
    try:
        with path.open("rb") as handle:
            head = handle.read(MAGIC_WINDOW)
    except OSError as exc:
        # strerror, not str(exc), because the message is a build log line and the
        # path is already in it. It is None for some OSErrors, which printed
        # "cannot be read (None)" and said nothing at all.
        return f"cannot be read ({exc.strerror or exc})"
    if b"%PDF-" not in head:
        # The likeliest thing a .pdf file is when it is not a PDF: a download that
        # returned a web page. Naming it makes the fix obvious (fetch it again).
        looks_like_html = b"<html" in head.lower() or b"<!doctype html" in head.lower()
        return "an HTML page saved as .pdf" if looks_like_html else "no %PDF- header"
    try:
        with pymupdf.open(path) as doc:
            # pymupdf opens HTML, EPUB, XPS and images too, and reports pages for
            # all of them. is_pdf is the only thing that says what was actually
            # parsed, and it is what separates a real PDF from a rendered one.
            if not doc.is_pdf:
                return "opens as something other than a PDF"
            if doc.page_count == 0:
                return "has no pages"
            if doc.needs_pass:
                return "is password protected"
    except Exception as exc:                      # noqa: BLE001 - any parse failure
        return f"cannot be parsed ({type(exc).__name__}: {exc})"
    return None


@click.command()
@click.option("--guidelines", type=click.Path(exists=True, file_okay=False, path_type=Path),
              **in_corpus("data/GUIDELINES"),
              help="Root of the servable corpus. Never widen this to data/.")
@click.option("--ocr", "ocr_dir", type=click.Path(file_okay=False, path_type=Path),
              **in_corpus("data/OCR"),
              help="OCR'd derivatives, which the site serves in place of their "
                   "originals and which therefore have to open too.")
def main(guidelines: Path, ocr_dir: Path) -> None:
    """Check that every file the site would serve is a PDF a browser can open."""
    roots = [guidelines] + ([ocr_dir] if ocr_dir.is_dir() else [])
    failures: list[tuple[Path, str]] = []
    checked = 0
    for root in roots:
        # Explicitly the corpus root's own files: nothing here may walk data/.
        for path in sorted(root.glob("*.pdf")):
            checked += 1
            reason = why_unusable(path)
            if reason:
                failures.append((path, reason))
        strays = sorted(p.name for p in root.iterdir()
                        if p.is_file() and p.suffix.lower() != ".pdf")
        if strays:
            logger.warning(f"{root}: {len(strays)} file(s) with another extension, "
                           f"not served: {', '.join(strays[:5])}")

    for path, reason in failures:
        logger.error(f"{path.name}: {reason}")
    if failures:
        logger.error(f"{len(failures)} of {checked} files cannot be served. Re-download "
                     "or remove them; the site would offer a document it cannot show.")
        raise SystemExit(1)
    duplicates = same_document(sorted(guidelines.glob("*.pdf")))
    if duplicates:
        logger.warning(f"{len(duplicates)} document(s) appear to be filed twice, same "
                       "page count and same opening text:")
        for group in duplicates:
            logger.warning("    " + " = ".join(group))
        logger.warning("Keeping both means the reader gets the same answer twice under "
                       "two names. Dropping one is a decision about the corpus, so this "
                       "is a warning and not a failure.")
    logger.success(f"{checked} files check out as openable PDFs")


if __name__ == "__main__":
    main()
