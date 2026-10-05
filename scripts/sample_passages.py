# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "loguru", "pymupdf==1.28.2"]
# ///
"""Sample evaluation passages from the guideline corpus.

This is step one of building the retrieval eval set. It does NOT write
queries: it draws a reproducible, corpus-spanning sample of real passages
that a human (or an LLM) then writes questions about. Splitting it this way
keeps the sampling deterministic and reviewable, and keeps the expensive
question-writing step from being silently re-run.

Sampling design
---------------
One page from each of N distinct documents, drawn uniformly over documents
and then uniformly over that document's pages. Uniform-over-documents is
the whole point: the corpus is 7910 pages of which the ten longest
documents hold over half, so sampling uniformly over PAGES would put most
of the eval set inside a handful of long documents and measure almost
nothing about the other hundred. Sampling documents WITHOUT replacement
pushes coverage further, giving at most one passage per document.

Why a page and not a chunk
--------------------------
The chunking strategy is not decided yet (see DESIGN.md). If ground truth
were a chunk id, every change to the chunker would invalidate the eval set.
Ground truth here is therefore (file, page), which survives any chunking
scheme: a retrieval hit counts if the returned chunk comes from that file
and overlaps that page. The passage text is recorded only so a question can
be written about it, not as an identifier.

Page quality filter
-------------------
Random pages are frequently useless to write a question about: covers,
tables of contents, bibliographies, pages that are a single figure. The
filter below rejects the cheap-to-detect cases so the sample is mostly
prose. It is deliberately crude; the reviewer is the real filter.

Only open documents are sampled
-------------------------------
The output is a tracked file, and a tracked file is a published one. A passage
may therefore only be drawn from a document the manifest marks `access=open`;
the books held under `access=restricted` are served but never quoted here, since
copying a page of one into a TSV would publish the content the repository is
careful never to commit. See `shareable`.
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys
from pathlib import Path

import click
import pymupdf as fitz
from loguru import logger

from lib import manifest_io
from lib.corpus import in_corpus

# A page needs this much extracted text before it is worth writing a
# question about. Cover pages and figure-only pages fall well under it.
MIN_PAGE_CHARS = 700

# Proportion of characters that must be letters. Filters out pages that are
# mostly tables of figures, dosage grids, or reference lists full of digits
# and punctuation, which produce questions with no natural language answer.
MIN_ALPHA_RATIO = 0.65

# Pages whose text is dominated by these markers are structural rather than
# substantive. Matched case-insensitively against the start of the page.
STRUCTURAL_MARKERS = re.compile(
    r"(table\s+des\s+mati|sommaire|table\s+of\s+contents|"
    r"r[ée]f[ée]rences?\s+bibliographiques?|bibliograph|"
    r"liste\s+des\s+abr[ée]viations|abbreviations|"
    r"remerciements|acknowledg|index\b|annexe\s+\d)",
    re.IGNORECASE,
)

# A citation-dense page is a reference list even without a heading. Counts
# "(Author, 1999)" and "[12]" style markers per thousand characters.
CITATION_MARKER = re.compile(r"\[\d{1,3}\]|\(\d{4}[a-z]?\)|\b\d{4};\s*\d+")
MAX_CITATIONS_PER_KCHAR = 12

# Longest passage handed to the question writer. Long enough to contain a
# self-contained clinical statement, short enough that the question is
# about a specific claim rather than a whole page of loosely related ones.
MAX_PASSAGE_CHARS = 1400

# The caption chunk.py writes above a serialised table, in either language. A page
# carrying one is a page whose answer lives in a grid rather than in a sentence,
# which is the case the prose filter above deliberately throws away and the one the
# `table` query kind exists to measure.
TABLE_CAPTION = re.compile(r"^(?:Tableau de \d+ lignes et \d+ colonnes"
                           r"|Table of \d+ rows and \d+ columns)")

# A passage that states a dose. The reader's most specific question ("how much,
# and how often") is also the one a chunker can break most cheaply, by
# putting the number and what it measures in different chunks, and nothing in the prose
# sample targets it: a uniformly drawn page names a dose perhaps one time in ten.
DOSE = re.compile(r"\b\d+(?:[.,]\d+)?\s?(?:mg|g|µg|mcg|mmol|UI|IU)\b"
                  r"|\b\d+(?:[.,]\d+)?\s?mg\s?/\s?(?:j|jour|day|kg)")

# How many pages to try inside one document before giving up on it. The
# three PDFs with no text layer fail every attempt, so this bounds the work
# spent discovering that.
MAX_PAGE_ATTEMPTS = 25


def normalise(text: str) -> str:
    """Collapse PDF whitespace without destroying paragraph boundaries.

    Parameters
    ----------
    text : str
        Raw text as returned by pymupdf.

    Returns
    -------
    str
        Text with runs of spaces collapsed, soft line wraps joined, and
        blank-line paragraph breaks preserved as a single newline.
    """
    text = text.replace("­", "")  # soft hyphens survive extraction
    text = re.sub(r"-\n(?=[a-zàâçéèêëîïôûùüÿñæœ])", "", text)  # dehyphenate
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"(?<![.!?:;])\n(?=[a-zàâçéèêëîïôûùüÿñæœ(])", " ", text)
    return text.strip()


def alpha_ratio(text: str) -> float:
    """Fraction of non-space characters that are letters.

    Parameters
    ----------
    text : str
        Normalised page text.

    Returns
    -------
    float
        Ratio in [0, 1]; 0.0 for empty input.
    """
    dense = [c for c in text if not c.isspace()]
    if not dense:
        return 0.0
    return sum(c.isalpha() for c in dense) / len(dense)


def is_usable_page(text: str) -> tuple[bool, str]:
    """Decide whether a page can carry an eval question.

    Parameters
    ----------
    text : str
        Normalised page text.

    Returns
    -------
    tuple of (bool, str)
        Whether the page is usable, and the rejection reason when it is
        not. The reason is logged so a corpus-wide problem (for instance a
        whole document rejected as scanned) is visible rather than silent.
    """
    if len(text) < MIN_PAGE_CHARS:
        return False, f"too short ({len(text)} chars)"
    if alpha_ratio(text) < MIN_ALPHA_RATIO:
        return False, f"not prose (alpha {alpha_ratio(text):.2f})"
    if STRUCTURAL_MARKERS.search(text[:200]):
        return False, "structural page"
    citations = len(CITATION_MARKER.findall(text))
    if citations / (len(text) / 1000) > MAX_CITATIONS_PER_KCHAR:
        return False, f"citation dense ({citations} markers)"
    return True, ""


def extract_passage(text: str) -> str:
    """Cut a self-contained passage out of a page.

    Starts at the first sentence boundary so the passage does not open
    mid-clause (page text almost always begins in the middle of a sentence
    carried over from the previous page), and ends at the last sentence
    boundary before the length cap.

    Parameters
    ----------
    text : str
        Normalised page text.

    Returns
    -------
    str
        The passage, or the whole text when no sentence boundary is found.
    """
    first = re.search(r"(?<=[.!?])\s+(?=[A-ZÀÂÇÉÈÊËÎÏÔÛÙÜŸÆŒ])", text[:400])
    body = text[first.end():] if first else text
    if len(body) <= MAX_PASSAGE_CHARS:
        return body.strip()
    window = body[:MAX_PASSAGE_CHARS]
    last = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    return (window[: last + 1] if last > MAX_PASSAGE_CHARS // 2 else window).strip()


def read_manifest(manifest_path: Path) -> dict[str, dict[str, str]]:
    """Load the metadata manifest keyed by filename.

    Parameters
    ----------
    manifest_path : Path
        Path to the tab-separated manifest produced by manifest.py.

    Returns
    -------
    dict
        Mapping of filename to its manifest row. Empty when the manifest
        does not exist, in which case language is left blank downstream.
    """
    if not manifest_path.exists():
        logger.warning(f"no manifest at {manifest_path}, language will be blank")
        return {}
    return manifest_io.read_by_file(manifest_path)


def write_rows(out: Path, rows: list[dict[str, object]]) -> None:
    """Write sampled passages as a TSV, creating the directory if needed.

    Both sampling kinds land here so the two never drift into two column orders,
    which would silently break whatever reads the file back.

    Parameters
    ----------
    out
        Destination path.
    rows
        Sampled passages. The first row's keys fix the column order.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def shareable(name: str, meta: dict[str, dict[str, str]]) -> bool:
    """Whether a document's text may be copied into the sample.

    data/EVAL_PASSAGES.tsv is a tracked file, so every passage written into it is
    published on the repository host. That is fine for a freely published guideline
    and not fine for a book the corpus serves under `access=restricted`: the PDF is
    never committed, and a page of its text landing in a TSV would be the same
    content arriving by another route. The check sits in the sampler rather than in
    the writer so that a refused document is replaced by another one and the sample
    still holds the `n` passages that were asked for.

    A document the manifest does not know is refused too, because "unknown" is not
    "open".

    Parameters
    ----------
    name
        PDF file name, as the manifest keys it.
    meta
        The manifest, keyed by file name.

    Returns
    -------
    bool
        True only when the document is marked `access=open`.
    """
    return meta.get(name, {}).get("access") == "open"


def chunk_rows(chunks_dir: Path, *, meta: dict[str, dict[str, str]], n: int,
               rng: random.Random, match, prefix: str) -> list[dict[str, object]]:
    """Sample pages whose chunk text passes a filter.

    Read from the chunk files rather than from the PDFs, for the table filter
    especially: the table detector lives in chunk.py, and a second detector here
    would drift from it and start offering pages the shipped chunker never treats as
    tables. It also means the question writer sees the text the encoder sees.

    One passage per document, as in the prose sample and for the same reason: a
    handful of documents hold most of the corpus's tables (the burnout report alone
    holds fifteen), and sampling over chunks would put the whole kind inside them.

    Parameters
    ----------
    chunks_dir
        data/chunks, as written by chunk.py.
    meta
        The manifest, keyed by file name.
    n
        How many passages to return.
    rng
        Seeded, so the sample is reproducible.
    match
        Called with a chunk's text; truthy means the chunk is a candidate.
    prefix
        One letter starting each passage id, so the ids of two kinds never collide.

    Returns
    -------
    list of dict
        Rows in the same shape as the prose sample, `passage` holding the chunk
        text as the encoder sees it.
    """
    found: list[tuple[str, int, int, str]] = []
    for path in sorted(chunks_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not shareable(payload["file"], meta):
            continue
        hits = [chunk for chunk in payload["chunks"] if match(chunk["text"])]
        if hits:
            pick = rng.choice(hits)
            pages = pick.get("pages") or [pick.get("page", 0)]
            found.append((payload["file"], pages[0], payload.get("pages_total", 0),
                          pick["text"]))
    rng.shuffle(found)
    rows: list[dict[str, object]] = []
    for name, page, pages_total, text in found[:n]:
        rows.append({
            "passage_id": f"{prefix}{len(rows) + 1:03d}",
            "file": name,
            "page": page,
            "pages_total": pages_total,
            "language": meta.get(name, {}).get("language", ""),
            "issuer": meta.get(name, {}).get("issuer", ""),
            "n_chars": len(text),
            "passage": text[:MAX_PASSAGE_CHARS].replace("\t", " ").replace("\n", " "),
        })
    return rows


@click.command()
@click.option("--guidelines", type=click.Path(exists=True, file_okay=False, path_type=Path),
              **in_corpus("data/GUIDELINES"),
              help="Root of the servable corpus. Never widen this to data/.")
@click.option("--manifest", type=click.Path(path_type=Path),
              **in_corpus("data/MANIFEST.tsv"),
              help="Metadata manifest, used to label each passage's language.")
@click.option("--out", type=click.Path(path_type=Path),
              **in_corpus("data/EVAL_PASSAGES.tsv"),
              help="Where to write the sampled passages.")
@click.option("--n", type=int, default=50, show_default=True,
              help="Number of passages, one per distinct document.")
@click.option("--seed", type=int, default=20260910, show_default=True,
              help="RNG seed. Changing it draws a different sample.")
@click.option("--kind", type=click.Choice(["prose", "table", "dose"]), default="prose",
              show_default=True,
              help="prose samples running text, which is what the filters above are "
                   "for. table samples pages whose answer chunk.py read out as a "
                   "grid: those pages fail the prose filters by design, so without "
                   "this the eval set has no query whose answer is a table cell. "
                   "dose samples passages that state a quantity, for the drug "
                   "questions a reader actually types.")
@click.option("--chunks", "chunks_dir", type=click.Path(path_type=Path),
              **in_corpus("data/chunks"),
              help="Chunk files, read by --kind table only.")
def main(guidelines: Path, manifest: Path, out: Path, n: int, seed: int,
         kind: str, chunks_dir: Path) -> None:
    """Draw a reproducible sample of guideline passages for the eval set."""
    meta = read_manifest(manifest)
    if not meta:
        logger.error(f"no usable manifest at {manifest}; without it there is no way to "
                     "tell an open guideline from a restricted book, and the sample is "
                     "committed")
        sys.exit(1)
    if kind in {"table", "dose"}:
        if not chunks_dir.is_dir():
            logger.error(f"no chunk files at {chunks_dir}; run scripts/chunk.py first")
            sys.exit(1)
        if kind == "table":
            match, prefix = (lambda text: TABLE_CAPTION.match(text.lstrip())), "t"
        else:
            # Two quantities, so that a passing mention of "5 mg" in a sentence about
            # something else is not offered as a dosage passage.
            match, prefix = (lambda text: len(DOSE.findall(text)) >= 2), "d"
        rows = chunk_rows(chunks_dir, meta=meta, n=n, rng=random.Random(seed),
                          match=match, prefix=prefix)
        if not rows:
            logger.error(f"nothing matched in {chunks_dir} for --kind {kind}")
            sys.exit(1)
        write_rows(out, rows)
        logger.success(f"wrote {len(rows)} {kind} passages to {out}")
        return
    pdfs = [pdf for pdf in sorted(guidelines.glob("*.pdf")) if shareable(pdf.name, meta)]
    if not pdfs:
        logger.error(f"no PDFs under {guidelines} that the manifest marks access=open")
        sys.exit(1)
    logger.info(f"{len(pdfs)} open documents available, drawing {n} with seed {seed}")

    rng = random.Random(seed)
    rng.shuffle(pdfs)

    rows: list[dict[str, object]] = []
    rejected: dict[str, int] = {}
    skipped_docs: list[str] = []

    for pdf in pdfs:
        if len(rows) >= n:
            break
        try:
            doc = fitz.open(pdf)
        except Exception as exc:  # a corrupt PDF must not abort the sample
            logger.warning(f"cannot open {pdf.name}: {exc}")
            skipped_docs.append(pdf.name)
            continue
        try:
            pages = list(range(doc.page_count))
            rng.shuffle(pages)
            for page_no in pages[:MAX_PAGE_ATTEMPTS]:
                text = normalise(doc[page_no].get_text())
                usable, reason = is_usable_page(text)
                if not usable:
                    rejected[reason.split(" (")[0]] = rejected.get(reason.split(" (")[0], 0) + 1
                    continue
                rows.append({
                    "passage_id": f"p{len(rows) + 1:03d}",
                    "file": pdf.name,
                    # 1-based to match what the PDF viewer shows the user.
                    "page": page_no + 1,
                    "pages_total": doc.page_count,
                    "language": meta.get(pdf.name, {}).get("language", ""),
                    "issuer": meta.get(pdf.name, {}).get("issuer", ""),
                    "n_chars": len(text),
                    "passage": extract_passage(text).replace("\t", " ").replace("\n", " "),
                })
                break
            else:
                skipped_docs.append(pdf.name)
        finally:
            doc.close()

    write_rows(out, rows)

    logger.success(f"wrote {len(rows)} passages to {out}")
    if len(rows) < n:
        logger.warning(f"only {len(rows)}/{n} passages: the corpus ran out of usable documents")
    for reason, count in sorted(rejected.items(), key=lambda kv: -kv[1]):
        logger.info(f"rejected pages, {reason}: {count}")
    if skipped_docs:
        logger.warning(f"{len(skipped_docs)} documents yielded no usable page "
                       f"(likely no text layer): {', '.join(skipped_docs[:5])}"
                       f"{' ...' if len(skipped_docs) > 5 else ''}")


if __name__ == "__main__":
    main()
