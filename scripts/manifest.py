# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "loguru", "pymupdf==1.28.2"]
# ///
"""Build and refresh the guideline metadata manifest.

The manifest (``data/MANIFEST.tsv``) carries the per-document metadata the web UI
exposes as search filters: issuer, country, year, language, doc_type, topic and the
``guideline`` tier the site's level control reads. It is a curated file: this script
only ever *fills in blanks*, so a human value is never overwritten by a heuristic
guess. Re-running it after dropping new PDFs into
``data/GUIDELINES/`` adds their rows and leaves existing curation untouched.

Two columns exist to solve a problem specific to this corpus, documented in
``DESIGN.md``: ``family`` and ``rendition``. French health authorities publish the same
guidance several times over at different lengths (a 4-page synthese, a 71-page
recommendation, a 234-page argumentaire). Those are near-duplicates that will retrieve
against each other. Grouping them under a shared ``family`` id lets the UI collapse them
and prefer the short rendition for display while still searching all of them.

Only ``data/GUIDELINES/`` is ever read. The sibling folders hold commercial books and
material of unconfirmed licence and must never reach the index; see ``CLAUDE.md``.
"""

from __future__ import annotations

import csv
import datetime
import re
import unicodedata
from pathlib import Path

import click
import pymupdf
from loguru import logger

from lib import atomic
from lib import corpus_config, manifest_io, vocabulary

# The facet policy (vocabularies, tiers, rendition classes) lives in its own
# stdlib-only module so that build_index.py and review_manifest.py can import it
# without this file's pymupdf. Re-exported here: the rest of this file and the
# tests read it as manifest.X.
from lib.manifest_policy import (  # noqa: F401
    ACCESS, CLOSED_VOCABULARIES, COMPANION_RENDITIONS, CORPUS_COLUMNS, EDITION_RENDITIONS,
    REDUNDANT_RENDITIONS, TIER_BY_VALUE, TIER_COLUMN, TIER_SOURCE, TIERS, VALUE_SEPARATOR,
    VOCABULARIES, tier_from_cell,
)

# Column order of data/MANIFEST.tsv. Adding a column here is backward compatible:
# existing rows gain an empty cell on the next run.
COLUMNS: list[str] = [
    "file",  # joins to SORTING_LOG.tsv and names the PDF in data/GUIDELINES/
    "title",  # human-readable, shown in the UI (filenames are not presentable)
    "issuer",  # the issuing body, from corpus.toml's [manifest] issuers or by hand
    "country",  # the issuer's country code
    "year",
    "language",  # a [languages] code from corpus.toml
    # The corpus's own columns (corpus.toml, [manifest] corpus_columns): its closed
    # vocabularies, multi-valued and separated by VALUE_SEPARATOR, and its tier
    # column. The tier is single valued and ORDERED: the UI's level control keeps
    # every tier down to the one chosen. Derived from TIER_SOURCE where that is
    # curated, hand-written where it is not, which is the whole reason it is its own
    # column rather than a lookup in build_index.py: a document added to the corpus
    # has no TIER_SOURCE until someone reviews it, and its tier has to be writable
    # meanwhile.
    *CORPUS_COLUMNS,
    "family",  # groups renditions of the SAME guidance; see module docstring
    "rendition",  # synthese, recommendation, argumentaire, lap, full
    "pages",
    "source_url",  # where the document was obtained; blank until curated by hand
    # The two identifiers a document prints about ITSELF, read off the PDF by the
    # block near find_doi below and shown as a clickable link on the result card and
    # in the viewer. A journal article carries a DOI, a book or an agency report an
    # ISBN, almost nothing carries both. NO_IDENTIFIER in either cell means the
    # extractor looked and found nothing, which is what keeps it from looking again
    # on every run; a blank means nobody has looked yet.
    "doi",
    "isbn",
    # Blank means the document is served like any other: openable and downloadable in
    # full. "restricted" means the site may show it but not hand it over: the viewer
    # greys out the download and the full-PDF link and only lets the reader step one
    # page either side of the passage. It is a hand-curated cell precisely because no
    # heuristic can tell whether a file may be redistributed.
    "access",
    "notes",
]

# Columns this script may guess. Everything else is left for human curation, because a
# wrong topic silently mis-filters search results, which is worse than a blank the UI
# can ignore. `doc_type` and `topic` are deliberately absent: no heuristic can tell a
# recommandation from its own argumentaire, or "cites WHO" from "published by WHO",
# so the TSV is their only home and no guess can ever clobber a curated cell.
# `pages` is not in it, and that is not an oversight: it is copied from the sorting
# log, not read off the document, so a row missing only its page count would
# otherwise extract the head of a PDF to learn nothing.
# `doi` and `isbn` are guessed too, but they are not in here: this set gates a scan of
# the first three pages, and the identifiers need nine (see extract_identifier_pages).
# They carry their own gate, NO_IDENTIFIER, for the same reason this one exists.
AUTOFILLABLE: set[str] = {"title", "issuer", "country", "year", "language"}

# Written into `doi` or `isbn` when the extractor read the document and found none.
# A single character, and one no identifier can start with, so a human reading the TSV
# sees "nothing there" rather than a value they have to parse. The alternative was a
# blank cell, which costs a nine-page scan of ~460 documents on every run of a script
# that otherwise takes thirty seconds; `--rescan-identifiers` is how to look again.
NO_IDENTIFIER = "-"

from lib.corpus import corpus_dir, in_corpus

# The two hand-curated tables below are DATA, one row per document, so they live
# as tracked TSVs next to the manifest rather than as dict literals in this file:
# a curator edits a table, and a reviewer reads a one-line diff per document. Each
# carries a `note` column holding why the row exists, which the literals used to
# keep as comments. They are corpus data, so they live under $CORPUS_DIR/data. Both
# are optional: a corpus that never needed a correction has no table, and the test
# suite imports this module against an empty corpus. A missing table is empty.
CORPUS = corpus_dir(required=False)


def curated_table(name: str) -> Path | None:
    """`$CORPUS_DIR/data/<name>` when there is a corpus and it holds that table."""
    path = CORPUS / "data" / name if CORPUS else None
    return path if path and path.is_file() else None


def read_curated(path: Path, *, columns: tuple[str, str]) -> dict[str, tuple[str, str]]:
    """Read a curated table: filename -> the two values it pins.

    Parameters
    ----------
    path
        `data/CURATED_ISSUERS.tsv` or `data/CURATED_FAMILIES.tsv`.
    columns
        The two value columns, in the order the tuple holds them.

    Returns
    -------
    dict[str, tuple[str, str]]
        Keyed by the document's filename. Blank cells stay blank: `("", "")` in
        the issuer table is a decision ("no issuing body"), not a gap.

    Raises
    ------
    ValueError
        On a duplicated filename, which would make the table's answer depend on
        row order.
    """
    table: dict[str, tuple[str, str]] = {}
    for row in manifest_io.read_rows(path):
        if row["file"] in table:
            raise ValueError(f"{path}: {row['file']!r} appears twice")
        table[row["file"]] = (row[columns[0]], row[columns[1]])
    return table


# Issuer detection, read from the corpus's corpus.toml ([manifest] issuers): which
# bodies to recognise and the regex for each, searched case-insensitively in the
# file name plus the first pages. FIRST match wins. The file explains why an
# acronym is wrapped in ``(?-i: )``: a measured failure where ``\bhas\b`` matched
# the English verb and stamped an agency onto journal articles it never issued.
# A wrong issuer is worse than a blank one, because the facet then silently
# excludes documents it should include. `_check_issuer_patterns` below is the gate
# that keeps this from regressing.
_MANIFEST_CONFIG = corpus_config.section("manifest")
ISSUER_PATTERNS: list[tuple[str, str, str]] = [
    (entry["name"], entry["country"], entry["pattern"]) for entry in _MANIFEST_CONFIG["issuers"]]

# Ordinary prose in the documents' languages naming none of the issuers: the
# sentence `_check_issuer_patterns` runs every pattern against.
INNOCENT_PROSE: str = _MANIFEST_CONFIG["issuer_innocent_prose"]


def _check_issuer_patterns() -> None:
    """Fail at import if an issuer pattern fires on ordinary prose.

    A pattern that matches an English or French sentence stamps an issuing body
    onto documents it never issued, and `read_existing` then protects that wrong
    value from ever being refilled. The failure is silent in every other way: the
    manifest looks curated and the facet looks populated.

    Raises
    ------
    AssertionError
        If any pattern matches `INNOCENT_PROSE`.
    """
    guilty = [issuer for issuer, _, pattern in ISSUER_PATTERNS
              if re.search(pattern, INNOCENT_PROSE, re.I)]
    assert not guilty, f"issuer patterns match ordinary prose: {guilty}"


_check_issuer_patterns()

# The languages the documents are written in, each with the function words that
# identify it (corpus.toml, [languages.<code>]). Deliberately crude: these are
# high-frequency words, so a few hundred characters of text is enough to decide.
DOCUMENT_LANGUAGES: dict[str, dict] = corpus_config.section("languages")
LANGUAGE_MARKERS: dict[str, re.Pattern[str]] = {
    code: re.compile(rf"\b({language['markers']})\b", re.I)
    for code, language in DOCUMENT_LANGUAGES.items()}


def count_pages(pdf_path: Path) -> str:
    """The PDF's page count as a manifest cell, or "" when the file cannot be opened.

    Blank rather than an exception, like every other autofill: check_pdfs.py is the
    gate that refuses an unreadable file, and it runs first.
    """
    try:
        with pymupdf.open(pdf_path) as doc:
            return str(doc.page_count)
    except Exception as error:  # pymupdf raises its own types plus RuntimeError
        logger.warning(f"{pdf_path.name}: cannot count pages ({error})")
        return ""


def extract_head_text(pdf_path: Path, *, pages: int = 3, max_chars: int = 4000) -> str:
    """Return the text of a PDF's first pages.

    Parameters
    ----------
    pdf_path : Path
        The PDF to read.
    pages : int, default 3
        How many leading pages to concatenate.
    max_chars : int, default 4000
        Truncate the result to this many characters. The callers only pattern-match,
        so more text costs time without improving the guess.

    Returns
    -------
    str
        Extracted text, empty if the PDF has no text layer or cannot be opened.
    """
    try:
        with pymupdf.open(pdf_path) as doc:
            chunks = [doc[i].get_text() for i in range(min(pages, doc.page_count))]
    except Exception as exc:  # a corrupt PDF must not abort the whole run
        logger.warning(f"could not read {pdf_path.name}: {exc}")
        return ""
    return " ".join(chunks)[:max_chars]


# --- DOI and ISBN ---------------------------------------------------------------
#
# Two identifiers, two completely different failure modes, which is why they are
# extracted by different code over different page windows rather than by one
# "find an identifier" function.
#
# A DOI is printed on a paper's FIRST page, and also in every entry of its
# reference list, thirty pages later. Pulling "the first DOI in the document" would
# therefore label an agency's recommendation with the DOI of the first study it cites, and
# a reader following that link would land on someone else's paper while the site
# said it was this one. So the window is the first two pages, a marker is required
# (`doi:`, `https://doi.org/`, `dx.doi.org`), and a page carrying a crowd of them is
# read as a reference page and skipped entirely.
#
# An ISBN is printed on the copyright page, which is page 2 to 5 of a book, and
# sometimes only on the back cover. It is also checksummed, which a DOI is not, so a
# false positive can be rejected arithmetically instead of by context.
#
# NEITHER is ever inferred from a number that merely looks right: `strip_identifiers`
# above exists because this corpus already had one year read out of an ISBN, and the
# cost of a wrong identifier here is worse than a blank, since it is rendered as a
# link that leaves the site.
BARE_DOI = re.compile(r"""10\.\d{4,9}/[^\s"'<>]+""")
DOI_MARKER = re.compile(
    r"""(?:doi\s*:?\s*|https?://(?:dx\.)?doi\.org/)(10\.\d{4,9}/[^\s"'<>]+)""",
    re.IGNORECASE,
)
# Trailing punctuation belongs to the sentence, not to the identifier. A DOI may end
# in almost anything, so this trims only what a line of prose would have put there,
# and only from the end.
DOI_TRAILING = re.compile(r"""[.,;:)\]}>'"]+$""")
# How many DISTINCT DOIs on one page make it a page of citations rather than a
# title page. Two, which is strict, and the corpus is why: an agency fact sheet
# in the first corpus prints two references
# with their DOIs in its footnotes on page 1 and none of its own, and at a threshold
# of three it was labelled with `10.1016/j.seizure.2016.08.005`, a paper it merely
# cites. A document printing its own DOI prints it once, so the strict rule costs
# almost nothing and the loose one produces a link that lies about what it opens.
DOI_CROWD = 2
# PDF typography turns a hyphen into U+2010/U+2011/U+2013 often enough to matter:
# `10.1136/bmj-2024-082507` came out of a BMJ paper as `10.1136/bmj\u20112024-082507`,
# which is not a DOI and resolves to nothing. Mapped back before anything else looks
# at the string.
DASHES = str.maketrans({c: "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"})
# The marker is required for the same reason find_doi requires one, and here it also
# tells a 13-digit ISBN apart from the other long numbers an agency report prints (a
# depot legal number, a phone number, a reference). The separator class is a plain
# `[- ]` because find_isbn maps every unicode dash back to a hyphen before matching:
# a PDF text layer turns the hyphens of an ISBN into U+2010 or U+2011 as readily as
# it does the hyphen of a DOI.
ISBN_MARKER = re.compile(
    r"ISBN(?:[- ]?1[03])?\s*:?\s*((?:97[89][- ]?)?(?:\d[- ]?){9}[\dXx])",
    re.IGNORECASE,
)


def valid_isbn(digits: str) -> bool:
    """Does a bare ISBN string pass its own check digit?

    Parameters
    ----------
    digits : str
        Ten or thirteen characters, hyphens already removed, an ISBN-10's last
        character possibly ``X``.

    Returns
    -------
    bool
        True when the check digit matches. This is the whole reason the ISBN is
        trusted without a marker-context test the way the DOI needs one: a random
        run of ten digits passes one time in eleven, and a run of thirteen one time
        in ten, so the checksum turns "looks like an ISBN" into "is one".
    """
    body = digits.upper()
    if len(body) == 10 and re.fullmatch(r"\d{9}[\dX]", body):
        total = sum((10 - i) * (10 if c == "X" else int(c)) for i, c in enumerate(body))
        return total % 11 == 0
    if len(body) == 13 and body.isdigit():
        total = sum((1 if i % 2 == 0 else 3) * int(c) for i, c in enumerate(body))
        return total % 10 == 0
    return False


def normalise_isbn(raw: str) -> str:
    """Strip an ISBN down to digits and an optional trailing X, or "" if invalid."""
    body = re.sub(r"[^0-9Xx]", "", raw).upper()
    return body if valid_isbn(body) else ""


def clean_doi(raw: str) -> str:
    """Turn a DOI as a PDF printed it into the DOI the registry knows.

    Three things are wrong with a DOI read off a page, and all three have been seen
    in this corpus:

    - a typographic dash instead of a hyphen (`10.1136/bmj\u20112024-082507`),
    - a control character glued to the end by the text layer (the same BMJ paper
      came out with a `\x08` after the last digit, which made the URL unusable),
    - the punctuation of the sentence it sat in (`(doi:10.x/y).`).

    Parameters
    ----------
    raw : str
        The matched text, with its `doi:` prefix already removed.

    Returns
    -------
    str
        A lower-cased DOI, or "" when nothing printable is left. Lower-cased
        because a DOI is case-insensitive by specification and the registry
        publishes them that way, so two copies of one document cannot end up
        carrying two different-looking identifiers.
    """
    # Cut at the first character that is not printable ASCII rather than deleting
    # it: a control character in the middle means the text layer broke the string,
    # and everything after it is as untrustworthy as the character itself.
    printable = re.split(r"[^\x21-\x7e]", raw.translate(DASHES))[0]
    return DOI_TRAILING.sub("", printable).lower()


def doi_from_metadata(pdf_path: Path) -> str:
    """The DOI a publisher wrote into the PDF's own metadata, or "".

    Read FIRST, before any page text, because it cannot be confused with a citation:
    a reference list lives in the page content and never in the XMP packet. 30 of the
    first 534 documents carry one, all of them journal articles, and every one of those is
    a document whose page text also holds the DOIs of everything it cites.

    Parameters
    ----------
    pdf_path : Path
        The PDF to read.

    Returns
    -------
    str
        The first DOI found in the XMP packet or the info dictionary, lower-cased,
        or "" when there is none or the file cannot be read.
    """
    try:
        with pymupdf.open(pdf_path) as doc:
            xmp = doc.xref_xml_metadata()
            packet = xmp if isinstance(xmp, str) else ""
            info = " ".join(str(v) for v in (doc.metadata or {}).values())
    except Exception as exc:  # a corrupt PDF must not abort the whole run
        logger.warning(f"could not read {pdf_path.name}: {exc}")
        return ""
    found = [clean_doi(d) for d in BARE_DOI.findall(f"{packet} {info}")]
    found = [d for d in found if len(d) > 7]
    return found[0] if found else ""


def find_doi(pages: list[str]) -> str:
    """The document's OWN DOI, read off its first pages, or "".

    Parameters
    ----------
    pages : list[str]
        The text of the leading pages, in order. Only the first two are read: a DOI
        further in is somebody else's, quoted in a reference list.

    Returns
    -------
    str
        The DOI without its `doi:` or `https://doi.org/` prefix, lower-cased
        (a DOI is case-insensitive and the registry publishes them lower-cased), or
        an empty string when the document prints none, or when the page printing
        them prints so many that it is a bibliography rather than a title page.
    """
    for page in pages[:2]:
        found = [clean_doi(m.group(1)) for m in DOI_MARKER.finditer(page)]
        found = [d for d in found if len(d) > 7]
        if not found:
            continue
        if len(set(found)) >= DOI_CROWD:
            # A first page that is already a reference list: seen on journal
            # supplements whose cover page is the table of contents. Refusing is the
            # right answer, because the one this document is ABOUT is not knowable
            # from a crowd of equals.
            return ""
        return found[0]
    return ""


def find_isbn(pages: list[str]) -> str:
    """The document's ISBN, read off its copyright page or its cover, or "".

    Parameters
    ----------
    pages : list[str]
        Text of the pages to search, in the order they should be searched. The
        caller passes the leading pages followed by the trailing ones, because a
        French agency report prints its ISBN on the back cover as often as on the
        copyright page.

    Returns
    -------
    str
        The ISBN as digits with no hyphens, or "" when nothing on those pages
        carries an ISBN marker followed by a number that passes its check digit.
        The FIRST valid one wins, and a book that prints several (paper, PDF, EPUB)
        keeps whichever it printed first; they identify the same work, which is
        what this column is for.
    """
    for page in pages:
        for match in ISBN_MARKER.finditer(page.translate(DASHES)):
            isbn = normalise_isbn(match.group(1))
            if isbn:
                return isbn
    return ""


def extract_identifier_pages(pdf_path: Path, *, head: int = 6, tail: int = 3,
                             max_chars: int = 6000) -> list[str]:
    """Text of a PDF's leading pages, then of its trailing ones.

    Returns
    -------
    list[str]
        One string per page, head pages first and tail pages after them, so a
        caller scanning in order sees the copyright page before the back cover.
        Empty when the PDF cannot be read: a corrupt file must not abort a run over
        the whole corpus.
    """
    try:
        with pymupdf.open(pdf_path) as doc:
            count = doc.page_count
            wanted = list(range(min(head, count)))
            wanted += [i for i in range(max(0, count - tail), count) if i >= head]
            return [doc[i].get_text()[:max_chars] for i in wanted]
    except Exception as exc:  # a corrupt PDF must not abort the whole run
        logger.warning(f"could not read {pdf_path.name}: {exc}")
        return []


def guess_language(text: str) -> str:
    """Guess the document's language code by counting function words.

    Returns an empty string when there is too little text to be confident, or
    when two languages tie, so the cell stays blank for a human rather than being
    filled with a coin flip.
    """
    counts = {code: len(marker.findall(text)) for code, marker in LANGUAGE_MARKERS.items()}
    if sum(counts.values()) < 10:
        return ""
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return ""
    return ranked[0][0]


def guess_issuer(haystack: str) -> tuple[str, str]:
    """Return ``(issuer, country)`` from the first matching pattern, else two blanks."""
    for issuer, country, pattern in ISSUER_PATTERNS:
        if re.search(pattern, haystack, re.I):
            return issuer, country
    return "", ""


# --------------------------------------------------------------------------
# Curated issuers
# --------------------------------------------------------------------------
# Pattern matching cannot tell "this document was published by X" from "this
# document cites X", and for a journal article the second is the common case: a
# journal review quotes WHO on its first page, another article carries an
# author's national-institute affiliation. Every
# entry below is a document where the guess was checked against the title page
# and found wrong.
#
# ("", "") means "checked, and this document has no issuing body": a journal
# article is published by a journal, not by an agency. That is the entry that
# needs the table to exist at all, because a blank cell is indistinguishable
# from an uncurated one and would be refilled with the same wrong guess on the
# next run.
#
# This table beats the guess but not the manifest: a non-empty cell already in
# the TSV always wins, so a hand edit is never reverted by a re-run.
CURATED_ISSUERS: dict[str, tuple[str, str]] = read_curated(
    curated_table("CURATED_ISSUERS.tsv"), columns=("issuer", "country")
) if curated_table("CURATED_ISSUERS.tsv") else {}


def _check_curated_issuers() -> None:
    """Fail at import if a curated issuer is not one the rest of the site knows.

    The issuer column feeds a filter dropdown built from whatever values the
    manifest happens to contain, so a typo here does not raise anywhere: it
    quietly adds a facet entry that matches one document and splits a publisher
    in two. Checking the country too catches the copy-paste error of keeping the
    previous row's country after changing the issuer.

    Raises
    ------
    AssertionError
        If an issuer is unknown, or its country disagrees with ISSUER_PATTERNS.
    """
    countries = {issuer: country for issuer, country, _ in ISSUER_PATTERNS}
    wrong = {
        name: (issuer, country)
        for name, (issuer, country) in CURATED_ISSUERS.items()
        if issuer and countries.get(issuer) != country
    }
    assert not wrong, f"curated issuers unknown or with the wrong country: {wrong}"


_check_curated_issuers()


# --------------------------------------------------------------------------
# Curated document families: the rendition classes are in lib/manifest_policy.py.
# --------------------------------------------------------------------------
# filename -> (family slug, rendition). family is blank for a document with no
# sibling in the corpus: the UI only needs to collapse when a family is shared,
# so a slug of one is noise. rendition is still worth recording on its own for
# filtering ("show me recommendations, not evidence reviews").
CURATED_FAMILIES: dict[str, tuple[str, str]] = read_curated(
    curated_table("CURATED_FAMILIES.tsv"), columns=("family", "rendition")
) if curated_table("CURATED_FAMILIES.tsv") else {}

def _check_rendition_vocabulary() -> None:
    """Fail at import if a curated rendition is in neither redundancy set.

    A rendition that appears in `CURATED_FAMILIES` but in neither
    `REDUNDANT_RENDITIONS` nor `COMPANION_RENDITIONS` would be classified as
    neither collapsible nor protected, which the search would resolve by
    guessing. Catching it here makes adding a new rendition name a deliberate
    act rather than a silent behaviour change.

    Raises
    ------
    AssertionError
        If any curated rendition is unclassified.
    """
    known = REDUNDANT_RENDITIONS | COMPANION_RENDITIONS | EDITION_RENDITIONS | {"pnds"}
    unknown = {rendition for _, rendition in CURATED_FAMILIES.values()} - known - {""}
    assert not unknown, f"unclassified renditions: {sorted(unknown)}"


_check_rendition_vocabulary()

# Publication years live in a narrow band, but so do plenty of four-digit runs
# that are not years at all. Shared by the filename and body-text passes.
YEAR = re.compile(r"\b(19[4-9]\d|20[0-4]\d)\b")

# Four-digit groups inside a book or serial identifier are not years. A
# national-institute report is the case that motivated this: its
# ISBN is 978-2-7598-1956-0, and the bare year regex happily returned 1956,
# which then sorted the document sixty years before it was written. Phone
# numbers and long reference numbers fail the same way, so this strips any
# four-digit group that is glued to another digit group by a hyphen, and any
# run of five or more digits, before the year search runs.
# Four or more hyphen-separated digit groups is the discriminating feature:
# an ISBN-13 (978-2-7598-1956-0) has five and an ISBN-10 (2-7598-1956-0) four,
# while a US date stamp (09-15-2015) has three and must NOT be stripped, since
# for one regulator's safety communication that stamp IS the publication date. An
# earlier, greedier version of this pattern ate it and moved that document to
# 2011, which is what the group count is guarding against.
IDENTIFIER = re.compile(
    r"\b\d{1,4}(?:-\d{1,5}){3,}\b"  # ISBN-10 / ISBN-13, with or without the 978 prefix
    r"|\b\d{4}-\d{4}\b"            # ISSN, and reporting spans such as "2010-2014"
    r"|\b\d{5,}\b"                  # long reference and CIP numbers
)


def strip_identifiers(text: str) -> str:
    """Blank out digit runs that look like identifiers rather than years.

    Parameters
    ----------
    text : str
        Leading document text to be searched for a publication year.

    Returns
    -------
    str
        The same text with identifier-shaped digit runs replaced by spaces,
        so a year regex cannot match inside one. Replaced rather than deleted
        so surrounding offsets stay meaningful when debugging.
    """
    return IDENTIFIER.sub(lambda m: " " * len(m.group(0)), text)


def guess_year(filename: str, text: str, pdf_path: Path) -> str:
    """Guess the publication year.

    Tries the filename first (most reliable: the user names files with their year),
    then a year mentioned in the leading text, then the PDF's creation date (least
    reliable: it reflects when the file was produced, which for a rescan can be
    decades after publication).
    """
    current = datetime.date.today().year
    plausible = lambda y: 1940 <= int(y) <= current  # noqa: E731

    for candidate in re.findall(YEAR, filename):
        if plausible(candidate):
            return candidate
    # Measured, not assumed: preferring a spelled-out "septembre 2016" over the
    # first bare year was tried and reverted. It fixed nothing the identifier
    # stripping had not already fixed and broke three documents, because a
    # spelled-out month near the top of a regulatory document is usually a
    # cited earlier decision (an FDA communication from February 2011, a
    # marketing authorisation dated janvier 2001) rather than this document's
    # own date, which is stamped numerically as 09-15-2015 or 29/03/2006.
    body = strip_identifiers(text[:1500])
    for match in YEAR.finditer(body):
        if plausible(match.group(1)):
            return match.group(1)
    try:
        with pymupdf.open(pdf_path) as doc:
            raw = (doc.metadata or {}).get("creationDate", "") or ""
        if match := re.search(r"(19[4-9]\d|20[0-4]\d)", raw):
            return match.group(1)
    except Exception:
        pass
    return ""


# Lines that an agency cover sets in the same size as the title, or bigger, but that
# are the name of the COLLECTION rather than the name of the document: "FICHE
# MEMO", "RAPPORT D'ELABORATION", "Recommandation de bonne pratique". A pure
# font-size heuristic returns one of these for a large minority of the corpus,
# which is worse than no title at all because every document in the collection
# then carries the same one. Matched against an accent-folded, case-insensitive
# copy of the line, so "RAPPORT D'ELABORATION" and "Rapport d'élaboration" are
# one pattern. Anchored at the start so a title that merely CONTAINS the word
# ("Annexe à la recommandation de bonne pratique ...") survives. The alternatives
# are the corpus's own ([manifest] title_labels in corpus.toml).
TITLE_LABELS = re.compile("|".join(_MANIFEST_CONFIG["title_labels"]), re.I)


# A title that ends on one of these is a fragment: the cover set the rest of it
# smaller, or in another block, and the run above stopped too early. "Bon usage
# des", "Quelle place pour les" and "Situation particuliere de" all came out of a
# first pass over an agency dump, and each is worse than the filename, because it
# reads like a title and names nothing. Same for a title STARTING lowercase,
# which is the same cut seen from the other end ("medecin traitant", "de TND de
# TND Interventions"). The words are per document language, in corpus.toml.
def _dangling(key: str) -> list[str]:
    """Every document language's `key` word list, merged, order kept, no repeats."""
    return list(dict.fromkeys(word for language in DOCUMENT_LANGUAGES.values()
                              for word in language[key]))


DANGLING_START = re.compile(rf"({'|'.join(_dangling('dangling_start'))})(?=[\s'\u2019])", re.I)
DANGLING_END = re.compile(rf"\b({'|'.join(_dangling('dangling_end'))})$", re.I)


def fold_accents(text: str) -> str:
    """Return ``text`` without its diacritics.

    Parameters
    ----------
    text : str
        Any string.

    Returns
    -------
    str
        The same string with combining marks removed, so that a pattern written
        in plain ASCII can match a line typed with accents.
    """
    return "".join(
        char for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )


def title_from_first_page(pdf_path: Path) -> str:
    """Read a document's title off its own title page.

    The filename is a poor title source for an agency dump: one agency names its files
    ``2017-11-08_qualite_air_interieur_rapport.pdf``, which is provenance
    rather than a title, and with ``embed.py --variant meta`` the title is also
    the prefix every chunk of that document is embedded with. The cover page
    carries the real title, set in the biggest font on the page.

    The heuristic is: drop the bottom of the page (logo and publication month),
    drop the collection labels above, take the biggest remaining line, then grow
    outwards while the neighbouring lines are close enough vertically and not
    much smaller. Growing is what catches a title that wraps over three cover
    lines, sometimes with the continuation set smaller ("TRAITEMENT NON
    PHARMACOLOGIQUE / de la maladie d'Alzheimer / et des pathologies
    apparentées").

    Parameters
    ----------
    pdf_path : Path
        The document to read.

    Returns
    -------
    str
        The title, or ``""`` when the page yields nothing plausible: no text
        layer, everything filtered out as a label, or a result too short or too
        long to be a title. The caller falls back to `clean_title`.
    """
    try:
        with pymupdf.open(pdf_path) as doc:
            if doc.page_count == 0:
                return ""
            page = doc[0]
            height = page.rect.height
            lines = []
            for block in page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    text = "".join(span["text"] for span in line["spans"]).strip()
                    if not text:
                        continue
                    lines.append({
                        "size": round(max(span["size"] for span in line["spans"]), 1),
                        "top": line["bbox"][1],
                        "bottom": line["bbox"][3],
                        "text": text,
                    })
    except Exception:  # a broken or image-only PDF is the caller's fallback case
        return ""

    lines = sorted((line for line in lines if line["top"] < height * 0.78),
                   key=lambda line: line["top"])
    is_label = lambda line: bool(TITLE_LABELS.match(fold_accents(line["text"]).strip(" .:-«»")))  # noqa: E731
    if not any(not is_label(line) for line in lines):
        return ""

    # Biggest font wins; ties go to the line highest on the page.
    anchor = max(
        range(len(lines)),
        key=lambda i: (lines[i]["size"], -lines[i]["top"]) if not is_label(lines[i]) else (-1.0, 0.0),
    )
    best = lines[anchor]["size"]
    run = [anchor]
    for step in (-1, 1):
        index = anchor + step
        while 0 <= index < len(lines):
            line, previous = lines[index], lines[index - step]
            gap = (line["top"] - previous["bottom"]) if step == 1 else (previous["top"] - line["bottom"])
            if is_label(line) or line["size"] < best * 0.6 or gap > best * 1.6:
                break
            run.append(index)
            index += step

    title = " ".join(lines[i]["text"] for i in sorted(run))
    title = re.sub(r"\s+", " ", title).strip(" -:")
    title = re.sub(r"\s*:\s*$", "", title)
    if not 8 <= len(title) <= 180:
        return ""
    # Starting on a connector is the same cut seen from the other end: either the
    # cover set the first half of the sentence bigger, or the run began inside a
    # label that wraps ("RAPPORT D'ELABORATION DE REFERENTIEL" / "D'EVALUATION DES
    # PRATIQUES PROFESSIONNELLES"). Case-folded, because a cover shouts.
    if (DANGLING_END.search(title.rstrip("."))
            or title[:1].islower()
            or DANGLING_START.match(fold_accents(title))):
        return ""
    return title


def clean_title(filename: str) -> str:
    """Derive a readable title from a filename.

    The fallback for `title_from_first_page`, and a starting point for curation
    rather than a finished title: it strips the extension and a leading
    publication date, normalises separators and collapses whitespace, but cannot
    rescue a filename like ``1679624144518001752.pdf``.
    """
    stem = Path(filename).stem
    # A leading ISO date is provenance, not a title: an agency dump names
    # every file after its publication date, and `guess_year` reads that date from
    # the filename anyway, so carrying it into the title only makes it unreadable.
    stem = re.sub(r"^\d{4}-\d{2}-\d{2}[_-]", "", stem)
    stem = re.sub(r"[_]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip(" -")
    return stem


def check_vocabulary(rows: list[dict[str, str]]) -> None:
    """Fail before writing if a curated cell names a slug outside its vocabulary.

    Parameters
    ----------
    rows
        The rows about to be written, each a column-to-value mapping.

    Raises
    ------
    click.ClickException
        If any `doc_type` or `topic` cell holds an unknown slug.

    Notes
    -----
    This is the gate that makes the vocabularies above real, and this file is where
    it belongs because this file owns them. `lib.vocabulary` does the checking, and
    `build_index.py` gates a second time against the same module with the same
    vocabularies read out of this file's source.
    """
    try:
        vocabulary.check(rows, VOCABULARIES, VALUE_SEPARATOR)
    except vocabulary.Unknown as exc:
        raise click.ClickException(str(exc)) from exc


#: Documents that share a title on purpose, or at least tolerably. It is empty today
#: and that is the intended resting state: the one pair it held, a guide that had
#: arrived twice under two file names, was resolved on 2026-09-19 by moving one copy
#: to data/DISCARDED/DUPLICATES. The hatch stays because a collision can be
#: legitimate (a guideline reissued unchanged under a second reference), and refusing
#: the manifest over it would block every run until someone edits a title by hand.
#: Anything not listed here that collides is a mistake worth stopping for, because a
#: result list showing one title twice gives the reader no way to tell which of the
#: two to open.
KNOWN_DUPLICATE_TITLES: frozenset[frozenset[str]] = frozenset(
    frozenset(group) for group in _MANIFEST_CONFIG.get("known_duplicate_titles", ()))


def title_key(title: str) -> str:
    """Reduce a title to what a reader would see as the same title.

    Accents, case and punctuation are folded away, because "Episode depressif" and
    "Épisode dépressif" collide on screen even though they differ as strings, and a
    check that compared raw strings would miss the pair a curator most wants told
    about: the same document titled twice by two different hands.

    Parameters
    ----------
    title : str
        A manifest title cell.

    Returns
    -------
    str
        Lower case, unaccented, every run of non-alphanumerics turned into one
        space, stripped.
    """
    return re.sub(r"[^a-z0-9]+", " ", fold_accents(title).lower()).strip()


def check_titles(rows: list[dict[str, str]]) -> None:
    """Fail before writing if two documents would ship under one title.

    Parameters
    ----------
    rows
        The rows about to be written, each a column-to-value mapping.

    Raises
    ------
    click.ClickException
        If two files share a title, unless the pair is in `KNOWN_DUPLICATE_TITLES`.

    Notes
    -----
    The title is the only thing a result card gives the reader to choose by, so two
    documents under one title is not a cosmetic problem: it is two results that
    cannot be told apart, and usually it means either the same document filed twice
    or a curated title that was copied onto a sibling and never specialised. Both
    are worth hearing about at the moment the manifest is written, which is the one
    moment someone is looking at titles.

    Editions of one book are exempt without needing an entry here: they carry the
    edition in their titles, so they do not collide.
    """
    seen: dict[str, list[str]] = {}
    for row in rows:
        title = (row.get("title") or "").strip()
        if title:
            seen.setdefault(title_key(title), []).append(row["file"])
    clashes = [files for files in seen.values()
               if len(files) > 1 and frozenset(files) not in KNOWN_DUPLICATE_TITLES]
    if clashes:
        detail = "; ".join(" + ".join(sorted(files)) for files in sorted(clashes))
        raise click.ClickException(
            f"{len(clashes)} title(s) shared by more than one document ({detail}). "
            "Give each its own title, or add the pair to KNOWN_DUPLICATE_TITLES with "
            "the reason."
        )


def read_existing(manifest_path: Path) -> dict[str, dict[str, str]]:
    """Load the current manifest keyed by filename, or an empty dict if absent."""
    if not manifest_path.is_file():
        return {}
    return manifest_io.read_by_file(manifest_path)


@click.command()
@click.option(
    "--guidelines",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    **in_corpus("data/GUIDELINES"),
    help="Corpus directory. Only this directory is ever read.",
)
@click.option(
    "--manifest",
    type=click.Path(dir_okay=False, path_type=Path),
    **in_corpus("data/MANIFEST.tsv"),
    help="Manifest to create or refresh in place.",
)
@click.option(
    "--sorting-log",
    type=click.Path(dir_okay=False, path_type=Path),
    **in_corpus("data/SORTING_LOG.tsv"),
    help="Source of the authoritative page counts. Optional: a document it does "
         "not list has its pages counted from the PDF.",
)
@click.option(
    "--dry-run", is_flag=True, help="Report what would change without writing."
)
@click.option(
    "--rescan-identifiers", is_flag=True,
    help="Read the PDFs again for documents whose doi and isbn cells hold "
         f"'{NO_IDENTIFIER}' (the extractor looked and found none). Say this after "
         "changing the extractor. A cell holding a real identifier is never re-read.",
)
@click.option(
    "--drop-orphans", is_flag=True,
    help="Write anyway when a curated row has no PDF, dropping that row. Say this "
         "only when the document really has left the corpus: the row is the only "
         "copy of its hand-written title, source_url, doc_type and topic.",
)
def main(guidelines: Path, manifest: Path, sorting_log: Path, dry_run: bool,
         rescan_identifiers: bool, drop_orphans: bool) -> None:
    """Create or refresh the guideline manifest, filling only empty cells."""
    # The curated tables were read at import, empty without a corpus; with explicit
    # paths and no CORPUS_DIR a run would refill every curated cell with a guess.
    corpus_dir()
    pdfs = sorted(p for p in guidelines.iterdir() if p.suffix.lower() == ".pdf")
    logger.info(f"{len(pdfs)} PDFs in {guidelines}")

    # Page counts come from the sorting log rather than being recomputed: it is the
    # authoritative record and disagreeing with it would be a bug worth noticing.
    pages_by_file: dict[str, str] = {}
    if sorting_log.is_file():
        with sorting_log.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                pages_by_file[row["file"]] = row.get("pages", "")

    existing = read_existing(manifest)
    rows: list[dict[str, str]] = []
    added = filled = scanned = 0

    for pdf in pdfs:
        row = {column: "" for column in COLUMNS}
        row.update(existing.get(pdf.name, {}))
        row["file"] = pdf.name
        if pdf.name not in existing:
            added += 1

        # Copied from the sorting log, so it costs nothing and is filled outside the
        # block below. Inside it, a row missing only its page count paid for the
        # text of three pages to learn something already in hand.
        if not row.get("pages") and pages_by_file.get(pdf.name):
            row["pages"] = pages_by_file[pdf.name]
            filled += 1
        # A corpus with no sorting log (or a document added after it) still gets a
        # page count: opening a PDF for its page tree reads no page content, so this
        # costs milliseconds where the extraction below costs seconds.
        if not row.get("pages"):
            row["pages"] = count_pages(pdf)
            filled += bool(row["pages"])

        # Only compute when something is actually missing: extraction dominates runtime.
        if any(not row.get(column) for column in AUTOFILLABLE):
            text = extract_head_text(pdf)
            haystack = f"{pdf.name} {text}"
            # The curated table wins over the guess, and ("", "") in it means
            # "no issuing body", which the blank-only fill below then leaves alone.
            issuer, country = CURATED_ISSUERS.get(pdf.name) or guess_issuer(haystack)
            guesses = {
                "title": title_from_first_page(pdf) or clean_title(pdf.name),
                "issuer": issuer,
                "country": country,
                "year": guess_year(pdf.name, text, pdf),
                "language": guess_language(text),
            }
            for column, value in guesses.items():
                if not row.get(column) and value:
                    row[column] = value
                    filled += 1

        # The identifiers, in a pass of their own because it opens nine pages where
        # the block above opens three, and because what it reads is not in the
        # filename and not in the sorting log. NO_IDENTIFIER is the gate: a blank
        # cell means nobody has looked, "-" means the extractor looked and found
        # nothing, and only the first is worth the scan. Without it, the ~460
        # documents that carry neither identifier would be re-read on every run,
        # doubling the runtime of the whole script to learn what it already knew.
        wanted = {column for column in ("doi", "isbn")
                  if not row.get(column)
                  or (rescan_identifiers and row[column] == NO_IDENTIFIER)}
        if wanted:
            found = {"doi": "", "isbn": ""}
            # The XMP packet first: it cannot be confused with a citation, because a
            # reference list lives in the page text and never in the metadata.
            if "doi" in wanted:
                found["doi"] = doi_from_metadata(pdf)
            # The page scan is the expensive half, so it is skipped when the metadata
            # already answered and no ISBN is being looked for.
            if "isbn" in wanted or not found["doi"]:
                pages = extract_identifier_pages(pdf)
                if "doi" in wanted and not found["doi"]:
                    found["doi"] = find_doi(pages)
                if "isbn" in wanted:
                    found["isbn"] = find_isbn(pages)
            scanned += 1
            for column in sorted(wanted):
                row[column] = found[column] or NO_IDENTIFIER
                if found[column]:
                    filled += 1

        # Curated, not guessed. Applied on every run (not only when something
        # is missing) so that adding a family to the table above propagates
        # without needing the manifest to be deleted first. Still fills only
        # empty cells, so a hand edit in the TSV always wins.
        family, rendition = CURATED_FAMILIES.get(pdf.name, ("", ""))
        for column, value in (("family", family), ("rendition", rendition)):
            if not row.get(column) and value:
                row[column] = value
                filled += 1

        # Derived, not guessed: TIER_SOURCE is itself hand-curated, so the tier that
        # follows from it is as trustworthy as the cell it reads. Applied on every
        # run so that curating the source also fills the tier, and blank-only so
        # that a curator who disagrees with the mapping writes their answer into the
        # TSV and keeps it.
        #
        # A row whose source is blank gets a blank tier rather than the widest one:
        # see TIERS for why the two are not the same thing.
        if not row.get(TIER_COLUMN):
            tier = tier_from_cell(row.get(TIER_SOURCE, ""))
            if tier:
                row[TIER_COLUMN] = tier
                filled += 1

        # Written out rather than left blank, because "blank" would have to mean "may
        # be redistributed" and a rights cell should never say that by omission. The
        # default is still open: everything in the corpus was a freely published
        # guideline until a document was deliberately marked "restricted" by hand.
        # Two spelled-out values are also what gives the reader a filter: the UI builds
        # a facet from a column with more than one value and nothing from a column of
        # blanks.
        if not row.get("access"):
            row["access"] = "open"
        rows.append(row)

    # A row whose PDF is not on disk is not written out, because `rows` is built
    # from the corpus. That is the right shape for the file and the wrong way to
    # arrive at it silently: the manifest is the ONLY home of the hand-written
    # title, source_url, doc_type, topic, family and rendition, days of work that a
    # document moved out of GUIDELINES/ for a moment would take with it. Git has
    # the row, but only for whoever reads the diff, and this script is run for its
    # side effect on a file nobody diffs.
    orphans = sorted(set(existing) - {p.name for p in pdfs})
    if orphans and not drop_orphans:
        raise click.ClickException(
            f"{len(orphans)} manifest row(s) have no PDF in {guidelines}, and each "
            f"holds curation that exists nowhere else: {orphans[:5]}"
            f"{' ...' if len(orphans) > 5 else ''}. Put the document back, or pass "
            "--drop-orphans to write without those rows."
        )
    if orphans:
        logger.warning(f"dropping {len(orphans)} row(s) with no PDF: {orphans[:5]}")

    # Checked on a dry run too: the point is to hear about a bad slug before the
    # build chain does, and a dry run is what a curator reaches for after editing.
    check_vocabulary(rows)
    check_titles(rows)

    if dry_run:
        logger.info(f"dry run: would add {added} rows and fill {filled} cells"
                    f"{f', after reading {scanned} PDFs for identifiers' if scanned else ''}")
        return

    manifest.parent.mkdir(parents=True, exist_ok=True)
    # Atomic because this file is hand-curated and is also this script's input:
    # a crash mid-write would truncate the curation (lib/atomic.py).
    with atomic.open_atomic(manifest, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=COLUMNS, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.success(f"wrote {manifest}: {len(rows)} rows, +{added} new, {filled} cells filled"
                   f"{f', {scanned} PDFs read for identifiers' if scanned else ''}")


if __name__ == "__main__":
    main()
