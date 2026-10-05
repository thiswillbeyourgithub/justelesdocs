# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "loguru"]
# ///
"""Show the chunks a page produced, the way a reader would have to read them.

Chunking is judged by looking at its output, and the output is 66469 JSON records
nobody can read. This prints the chunks covering one page of one document, in
order, with the two things that decide whether a chunk is usable: where it starts
and where it stops.

    uv run scripts/inspect_chunks.py --doc "TOC_résistants" --page 39
    uv run scripts/inspect_chunks.py --doc canmat --page 1 --full

`--doc` is a case-insensitive substring of the file name, and matching several is
an error rather than a guess. Reads data/chunks only; it never opens a PDF, so it
shows what the encoder was given rather than what the page looks like.

The flags in the margin are the machine-checkable half, and `chunk_quality.py`
counts the same ones over the whole corpus:

    ^ starts mid-sentence (first character is lowercase, or the text opens on a
      closing bracket)
    $ ends mid-sentence (no terminal punctuation, and not a heading or list item)
    = the page's text also appears in a neighbouring chunk, i.e. overlap

Written by Claude Code (Opus 5).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import click
from loguru import logger
from lib.corpus import in_corpus

# A chunk that ends without one of these is mid-sentence, unless it is short enough
# to be a heading or a list item (which legitimately have no full stop).
TERMINAL = tuple(".!?:;…»\")]")
# Text opening on a lowercase letter, or on the closing half of a pair, continues
# something the reader has not been given.
STARTS_MID = re.compile(r"^[a-zà-öø-ÿ),;:.…]")
HEADING_LIKE = re.compile(r"^(\d+[\d.]*\s|•|-\s|–\s|[A-Z][A-Z\s]{3,}$)")


def load(chunks_dir: Path, needle: str) -> tuple[Path, dict]:
    """Find the one chunk file whose document name contains `needle`."""
    hits = [p for p in sorted(chunks_dir.glob("*.json"))
            if needle.lower() in json.loads(p.read_text(encoding="utf-8"))["file"].lower()]
    if not hits:
        raise click.ClickException(f"no document under {chunks_dir} matches {needle!r}")
    if len(hits) > 1:
        names = ", ".join(json.loads(p.read_text(encoding="utf-8"))["file"] for p in hits[:6])
        raise click.ClickException(f"{len(hits)} documents match {needle!r}: {names}")
    return hits[0], json.loads(hits[0].read_text(encoding="utf-8"))


def flags(chunk: dict, text: str) -> str:
    """The three-character margin described in the module docstring."""
    head = "^" if STARTS_MID.match(text) else " "
    tail = " "
    stripped = text.rstrip()
    if stripped and not stripped.endswith(TERMINAL):
        # A short last line that looks like a heading or a bullet is allowed to have
        # no full stop; a 300-token chunk ending on "the operator should be" is not.
        last = stripped.rsplit("\n", 1)[-1]
        tail = " " if (len(last) < 60 and HEADING_LIKE.match(last)) else "$"
    return head + tail


@click.command()
@click.option("--chunks", "chunks_dir", type=click.Path(exists=True, file_okay=False,
                                                        path_type=Path),
              **in_corpus("data/chunks"))
@click.option("--doc", required=True, help="Substring of the document's file name.")
@click.option("--page", type=int, default=0,
              help="Page number as the reader sees it (1-based). 0 means every page.")
@click.option("--full", is_flag=True, help="Print whole chunks instead of their ends.")
@click.option("--edge", type=int, default=240, show_default=True,
              help="Characters of the start and of the end to show when not --full.")
def main(chunks_dir: Path, doc: str, page: int, full: bool, edge: int) -> None:
    """Print the chunks covering one page, with what is wrong at their edges."""
    path, payload = load(chunks_dir, doc)
    wanted = [c for c in payload["chunks"] if not page or page in c["pages"]]
    logger.info(f"{payload['file']} ({path.name}): {len(payload['chunks'])} chunks, "
                f"{len(wanted)} on {'page ' + str(page) if page else 'every page'}")
    for chunk in wanted:
        text = chunk["text"]
        mark = flags(chunk, text)
        pages = ",".join(str(p) for p in chunk["pages"])
        click.echo(f"\n{mark} [{chunk['i']}] {chunk['n_tokens']} tokens, p{pages}, "
                   f"{len(chunk['boxes'])} boxes")
        if full or len(text) <= 2 * edge:
            click.echo(text)
        else:
            click.echo(text[:edge].rstrip())
            click.echo(f"      ... {len(text) - 2 * edge} characters ...")
            click.echo(text[-edge:].lstrip())


if __name__ == "__main__":
    main()
