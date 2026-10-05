# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "loguru"]
# ///
"""Check the invariants the brief requires of a chunk build.

These are build failures, not warnings. A corpus with a hole in it produces
search results that are wrong by omission, and omission is invisible: no user
can tell that the guideline answering their question was silently skipped.
So every check here exits non-zero rather than logging and continuing.

Invariants
----------
1. Every PDF under the guidelines root has a chunk file.
2. No chunk file is for a PDF that is no longer in the corpus.
3. No document has zero chunks.
4. No chunk has empty text.
5. Every chunk carries at least one page and at least one box, because a hit
   that cannot be highlighted cannot be shown.
6. Every box lies inside its page and is non-degenerate.
7. Every page listed by a chunk has boxes, and vice versa.
8. Nothing outside the guidelines root was indexed.

It also reports the distributions worth eyeballing (tokens per chunk, chunks
per document) and, when the eval set is present, what share of the eval gold
pages is actually reachable: a gold page with no chunk is a query that can
never be answered, which would silently cap every retrieval metric.

Written with Claude Code.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

import click
from loguru import logger
from lib.corpus import in_corpus


#: How far outside its page a highlight box may reach before it counts as a failure,
#: in PDF points. Not slack for bad boxes: boxes are rounded to whole points when they
#: are written (`boxes_by_page` in chunk.py) and page sizes are rounded the same way
#: here, so a line flush against the page edge can land a point outside it with nothing
#: wrong. Two points is that rounding, doubled. A box further out than this describes
#: text drawn off the page, which the reader cannot see and the search counted.
BOX_MARGIN = 2


def box_outside_page(box: list[float], page: str, sizes: list[list[int]]) -> str | None:
    """Say how a box fails to lie on its page, or None if it does.

    Parameters
    ----------
    box
        `[x0, y0, x1, y1]` in PDF points, origin at the top left of the page.
    page
        The 1-based page number, as the chunk file spells it (a string, because
        JSON object keys are strings).
    sizes
        `[width, height]` per page, in document order, as `chunk.py` records them.
        Empty when the chunk file predates format version 10, in which case nothing
        can be said and the caller has already reported that.

    Returns
    -------
    str or None
        A message naming the page and the offending edge, or None if the box fits.

    Notes
    -----
    This is the check invariant 6 has always claimed and never made: before the
    chunk file recorded page sizes, "inside its page" was not a question it could
    answer. A box outside its page is a highlight pdf.js draws where nobody looks,
    on a result the ranking counted as a hit, so the reader sees a page that
    apparently does not contain what they searched for.
    """
    if not sizes:
        return None
    index = int(page) - 1
    if not 0 <= index < len(sizes):
        return f"box on page {page}, but the document has {len(sizes)} pages"
    width, height = sizes[index]
    x0, y0, x1, y1 = box
    if (x0 < -BOX_MARGIN or y0 < -BOX_MARGIN
            or x1 > width + BOX_MARGIN or y1 > height + BOX_MARGIN):
        return (f"box {box} lies outside page {page}, which is {width}x{height} "
                f"points")
    return None


def load_chunk_files(chunks_dir: Path) -> dict[str, dict]:
    """Load every chunk file, keyed by the PDF filename it records.

    Parameters
    ----------
    chunks_dir : Path
        Directory of per-document JSON files.

    Returns
    -------
    dict
        Mapping of source PDF filename to its parsed chunk document.
    """
    loaded: dict[str, dict] = {}
    for path in sorted(chunks_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        loaded[data["file"]] = data
    return loaded


@click.command()
@click.option("--guidelines", type=click.Path(exists=True, file_okay=False, path_type=Path),
              **in_corpus("data/GUIDELINES"))
@click.option("--chunks", "chunks_dir", type=click.Path(exists=True, file_okay=False, path_type=Path),
              **in_corpus("data/chunks"))
@click.option("--queries", type=click.Path(path_type=Path),
              **in_corpus("data/EVAL_QUERIES.tsv"),
              help="Eval queries, used to check gold pages are reachable.")
@click.option("--allow-empty", multiple=True, metavar="FILENAME",
              help="PDFs known to have no text layer, pending OCR. Repeatable.")
def main(guidelines: Path, chunks_dir: Path, queries: Path, allow_empty: tuple[str, ...]) -> None:
    """Verify a chunk build and report its shape."""
    pdfs = {p.name for p in guidelines.glob("*.pdf")}
    docs = load_chunk_files(chunks_dir)
    failures: list[str] = []

    for name in sorted(pdfs - docs.keys()):
        failures.append(f"no chunk file for {name}")
    for name in sorted(docs.keys() - pdfs):
        failures.append(f"chunk file for {name}, which is not in {guidelines}")

    tokens: list[int] = []
    per_doc: list[int] = []
    for name, data in sorted(docs.items()):
        chunks = data["chunks"]
        per_doc.append(len(chunks))
        if not chunks:
            if name in allow_empty:
                logger.warning(f"{name}: zero chunks, allowed pending OCR")
            else:
                failures.append(f"{name}: zero chunks")
            continue
        # Checked once per document, because every box below is checked against it.
        # A file written before format version 10 carries no page sizes; that is a
        # stale file rather than a corrupt one, and re-running chunk.py is the fix.
        sizes = data.get("page_sizes") or []
        if not sizes:
            failures.append(f"{name}: no page_sizes, so no box can be checked against "
                            "the page it is drawn on. Re-run chunk.py.")
        elif len(sizes) != data["stats"]["pages"]:
            failures.append(f"{name}: {len(sizes)} page sizes for "
                            f"{data['stats']['pages']} pages")
        for chunk in chunks:
            where = f"{name}#{chunk['i']}"
            if not chunk["text"].strip():
                failures.append(f"{where}: empty text")
            if not chunk["pages"]:
                failures.append(f"{where}: no page recorded")
            if not chunk["boxes"]:
                failures.append(f"{where}: no boxes, cannot be highlighted")
            # Pages and boxes must describe the same set, or the viewer will
            # open a page it has nothing to draw on.
            if {str(p) for p in chunk["pages"]} != set(chunk["boxes"]):
                failures.append(f"{where}: pages {chunk['pages']} disagree with "
                                f"box pages {sorted(chunk['boxes'])}")
            for page, boxes in chunk["boxes"].items():
                if not boxes:
                    failures.append(f"{where}: page {page} has an empty box list")
                for box in boxes:
                    # Reported, not raised. A malformed box is exactly the kind of
                    # corruption this script exists to name, and unpacking it into
                    # four variables ended the run with a traceback naming neither
                    # the document nor the chunk.
                    if len(box) != 4 or not all(isinstance(v, (int, float)) for v in box):
                        failures.append(f"{where}: box {box!r} on page {page} is not "
                                        "four numbers")
                        continue
                    x0, y0, x1, y1 = box
                    if x1 <= x0 or y1 <= y0:
                        failures.append(f"{where}: degenerate box {box} on page {page}")
                    outside = box_outside_page(box, page, sizes)
                    if outside:
                        failures.append(f"{where}: {outside}")
            tokens.append(chunk["n_tokens"])

    # A gold page with no chunk is a query that cannot be answered by any
    # retriever, which would cap the eval metrics for a reason unrelated to
    # retrieval quality. Worth knowing before measuring anything.
    if queries.exists():
        reachable = {
            (name, page)
            for name, data in docs.items()
            for chunk in data["chunks"]
            for page in chunk["pages"]
        }
        with queries.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        missing = {
            (r["gold_file"], int(r["gold_page"]))
            for r in rows
            if (r["gold_file"], int(r["gold_page"])) not in reachable
        }
        if missing:
            failures.append(f"{len(missing)} eval gold pages have no chunk: "
                            f"{sorted(missing)[:5]}")
        else:
            logger.success(f"all {len(rows)} eval queries have a reachable gold page")

    logger.info(f"{len(docs)} documents, {len(tokens)} chunks")
    if tokens:
        logger.info(f"tokens per chunk: min={min(tokens)} "
                    f"median={int(statistics.median(tokens))} "
                    f"mean={int(statistics.mean(tokens))} max={max(tokens)}")
        over = sum(1 for t in tokens if t > 512)
        logger.info(f"chunks over 512 tokens: {over} ({over / len(tokens):.2%})")
        logger.info(f"chunks per document: min={min(per_doc)} "
                    f"median={int(statistics.median(per_doc))} max={max(per_doc)}")

    if failures:
        for failure in failures[:25]:
            logger.error(failure)
        if len(failures) > 25:
            logger.error(f"... and {len(failures) - 25} more")
        sys.exit(1)
    logger.success("all invariants hold")


if __name__ == "__main__":
    main()
