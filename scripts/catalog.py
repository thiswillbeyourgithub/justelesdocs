# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "loguru"]
# ///
"""Write `CATALOG.md`: every document this site serves, grouped by who published it.

The corpus is gitignored (the PDFs are hundreds of megabytes and some are
copyrighted), so a reader of the repository cannot see what is actually shipped.
`data/SORTING_LOG.tsv` records why each document sits where it does and
`data/MANIFEST.tsv` records its metadata, but neither is something a person reads.
This is: one markdown file, a table of contents by issuing body, and one row per
document with everything the manifest knows plus what the served file weighs.

**It is generated, and the pre-push hook re-generates it and refuses a push when
the result differs.** That is the whole point: a catalog nobody regenerates is a
catalog that describes last month's corpus, and this one is the only public
account of what the site serves.

Two things it deliberately does not do. It does not recompute page counts, which
come from `data/MANIFEST.tsv` (itself filled from `data/SORTING_LOG.tsv`, the
authoritative record). And it does not decide which file is served: `chunk.py`
writes `"source": "ocr"` or `"source": "original"` per document and `stage.py`
reads it, so this imports `stage.py` and asks it, rather than re-deriving a rule
that would then drift. The size in the table is the size of the file a visitor
downloads, which for the OCR'd documents is the derivative, not the original.

Written by Claude Code (Opus 5).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import click
from loguru import logger

from lib import manifest_io, site_config
from lib.corpus import corpus_path, in_corpus

# Documents with no issuing body are not an omission: a journal article is
# published by a journal, not by an agency, and `data/CURATED_ISSUERS.tsv` (read by manifest.py)
# records that verdict for each one. They are grouped under this heading rather
# than under an invented issuer.
NO_ISSUER = "Journal articles and other documents with no issuing body"

HEADER = """<!-- GENERATED FILE. Do not edit by hand.

     Written by `uv run scripts/catalog.py` from data/MANIFEST.tsv and the files
     in data/GUIDELINES/ and data/OCR/. `.githooks/pre-push` regenerates it and
     refuses the push if the result differs, so this always describes the corpus
     that is actually shipped. -->

# Catalog

Every document served by [{name}](README.md), grouped by the body that published it.

The corpus itself is not in this repository: the PDFs are large, and most of them belong to their publishers. This file is the account of what is there.

"""


def load_stage_module() -> object:
    """Import `scripts/stage.py` by path, for its served-file rule.

    Returns
    -------
    object
        The `stage` module, with `read_chunk_index` and `SOURCE_DIRS`.

    Notes
    -----
    By path rather than as a package because `scripts/` is not one: these are
    standalone PEP 723 files. `stage.py` imports `changelog.py` the same way, and
    for the same reason, which is that one rule must have one home. Duplicating
    "prefer data/OCR/" here would work until someone adds a derivative, and then
    the catalog would report the size of a file nobody downloads.
    """
    path = Path(__file__).resolve().parent / "stage.py"
    spec = importlib.util.spec_from_file_location("stage", path)
    if spec is None or spec.loader is None:
        raise click.ClickException(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage"] = module
    spec.loader.exec_module(module)
    return module


def human_size(size: int) -> str:
    """Format a byte count for a table cell.

    Parameters
    ----------
    size
        Bytes.

    Returns
    -------
    str
        e.g. ``"1.2 MB"``. Decimal megabytes, not mebibytes: the number is there
        to answer "is this going to be a long download", and a reader reads it as
        what their browser will say.
    """
    if size >= 1_000_000:
        return f"{size / 1_000_000:.1f} MB"
    return f"{size / 1_000:.0f} kB"


def cell(value: str) -> str:
    """Escape a manifest value for a markdown table cell."""
    return (value or "").replace("|", r"\|").replace("\n", " ").strip() or "-"


def multi(value: str, separator: str = ";") -> str:
    """Render a multi-valued manifest cell (`doc_type`, `topic`) as a list."""
    parts = [piece.strip() for piece in (value or "").split(separator) if piece.strip()]
    return cell(", ".join(parts))


def anchor(heading: str) -> str:
    """GitHub's slug for a heading, so the table of contents resolves.

    Parameters
    ----------
    heading
        The heading text, exactly as written.

    Returns
    -------
    str
        Lowercased, spaces to hyphens, everything but letters, digits, hyphen and
        underscore dropped. Accented letters are kept, which is what GitHub does.
    """
    slug = heading.strip().lower().replace(" ", "-")
    return "".join(character for character in slug
                   if character.isalnum() or character in "-_")


def build(manifest: Path, chunks: Path) -> str:
    """Render the whole catalog.

    Parameters
    ----------
    manifest
        `data/MANIFEST.tsv`.
    chunks
        `data/chunks/`, which says which file each document is served from.

    Returns
    -------
    str
        The complete markdown document, ending in a newline.

    Raises
    ------
    click.ClickException
        If a document in the manifest has no file on disk, which means the
        catalog would describe something nobody can download.
    """
    stage = load_stage_module()
    served = stage.read_chunk_index(chunks)
    # The access tier, from stage.py rather than re-read here: stage.py is what acts
    # on it (a restricted document is staged outside the web root and its text is
    # stripped), and a catalog that decided the tier for itself could describe a
    # document as downloadable while the staging step held it back.
    tiers = stage.read_access(manifest)

    rows = manifest_io.read_rows(manifest)

    groups: dict[str, list[dict[str, str]]] = {}
    total = 0
    for row in rows:
        name = row["file"]
        source = served.get(name, "original")
        path = corpus_path(stage.SOURCE_DIRS[source]) / name
        if not path.is_file():
            raise click.ClickException(
                f"{name}: no file at {path}, so the catalog cannot say what is served"
            )
        row["_size"] = path.stat().st_size
        row["_served"] = source
        row["_access"] = tiers[name]
        total += row["_size"]
        groups.setdefault(row["issuer"].strip() or NO_ISSUER, []).append(row)

    # Issuers alphabetically, with the no-issuer group last: it is a residue, not a
    # publisher, and sorting it into the Is would read as one.
    names = sorted((name for name in groups if name != NO_ISSUER), key=str.casefold)
    if NO_ISSUER in groups:
        names.append(NO_ISSUER)

    out = [HEADER.replace("{name}", site_config.load_site()["name"])]
    restricted = sum(1 for row in rows if row["_access"] == "restricted")
    out.append(f"{len(rows)} documents, {human_size(total)} in total, "
               f"{len(names)} publishers.\n")
    if restricted:
        # Said here rather than only in the rows, because the total above is the
        # first number a reader takes from this file and it would otherwise read as
        # "this much is downloadable", which for these documents it is not.
        out.append(
            f"{restricted} of them are marked **read only** below: they are served a "
            "page at a time, cut out of the file as each page is asked for, and the "
            "whole document is never handed over. The size in their row is the size "
            "of the file on the server, not of anything a visitor receives.\n")
    out.append("## Contents\n")
    for name in names:
        out.append(f"- [{name}](#{anchor(name)}) ({len(groups[name])})")
    out.append("")

    for name in names:
        out.append(f"## {name}\n")
        out.append("| Document | Year | Lang. | Type | Topics | Pages | Size | Source |")
        out.append("| --- | ---: | --- | --- | --- | ---: | ---: | --- |")
        for row in sorted(groups[name], key=lambda r: r["title"].casefold()):
            # The filename is worth showing: it is the identity every other record
            # in the repository joins on, and it is what the reader downloads.
            title = f"**{cell(row['title'])}**<br>`{cell(row['file'])}`"
            if row["_served"] == "ocr":
                title += "<br>*served OCR'd*"
            if row["_access"] == "restricted":
                title += "<br>**read only: one page at a time, no download**"
            link = f"[publisher]({row['source_url']})" if row["source_url"].strip() else "-"
            out.append(
                f"| {title} | {cell(row['year'])} | {cell(row['language'])} | "
                f"{multi(row['doc_type'])} | {multi(row['topic'])} | "
                f"{cell(row['pages'])} | {human_size(row['_size'])} | {link} |"
            )
        out.append("")

    # The AI caveat travels with the links, exactly as it does in the interface: a
    # reader who meets these URLs here meets them without the "Source (IA)" label.
    out.append("---\n")
    out.append("Publisher links were found by a language model and then checked "
               "automatically, one by one, against the document's own text, its DOI or "
               "an archived capture of the page (`data/SOURCE_URL_LOG.tsv` records what "
               "confirmed each one). They can still be wrong. Documents whose publisher "
               "has withdrawn them carry no link, rather than a link to a different "
               "edition.")
    return "\n".join(out) + "\n"


@click.command()
@click.option("--manifest", type=click.Path(dir_okay=False, path_type=Path),
              **in_corpus("data/MANIFEST.tsv"),
              help="Document metadata. The catalog reports it, never recomputes it.")
@click.option("--chunks", type=click.Path(file_okay=False, path_type=Path),
              **in_corpus("data/chunks"),
              help="Chunk files from chunk.py, which say which file is served.")
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path),
              **in_corpus("CATALOG.md"),
              help="Markdown file to write.")
@click.option("--check", is_flag=True,
              help="Write nothing; fail if the file on disk is not what this would "
                   "write. What .githooks/pre-push runs.")
def main(manifest: Path, chunks: Path, out: Path, check: bool) -> None:
    """Write the corpus catalog, or refuse because it is out of date."""
    if not manifest.is_file() or not chunks.is_dir():
        # The corpus is gitignored, so a clone without it can neither generate nor
        # check. Saying so and stopping is right; failing would refuse every push
        # from a machine that has the code and not the documents.
        logger.warning(f"{manifest} or {chunks} missing: nothing to catalog here")
        return
    text = build(manifest, chunks)
    if check:
        current = out.read_text(encoding="utf-8") if out.is_file() else ""
        if current != text:
            raise click.ClickException(
                f"{out} is out of date: run `uv run scripts/catalog.py`. It is "
                "generated from the corpus, and a stale one describes documents the "
                "site no longer serves."
            )
        logger.success(f"{out} matches the corpus")
        return
    out.write_text(text, encoding="utf-8")
    logger.success(f"{out} written ({len(text) / 1e3:.1f} KB)")


if __name__ == "__main__":
    main()
