#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "loguru"]
# ///
"""Hard gate: nothing a restricted document owns may be reachable under the web root.

This gate exists because the invariant it checks broke silently and stayed broken.
A pass on 2026-09-19 moved 53 documents into the corpus out of `DISCARDED` and
`UNSURE` on the stated condition that they be "served without download", and
recorded that condition in the `access` column of `data/MANIFEST.tsv`. The front
end implemented it: `src/viewer.js` hides the download and limits reading to the
passage's page plus one either side. Nothing below the front end knew the column
existed, so `stage.py` symlinked all 534 PDFs into the web root and linked
`index/doc/` whole, and the result was 495 MiB of commercial books at their own
URLs plus 58 MiB of their complete extracted text, one request per title.

Nothing had been deployed, so nothing was exposed. What that episode showed is
that a rule living only in a viewer is not a rule, so this script makes it one:
it reads the tiers from the manifest and refuses if the served tree contradicts
them. It needs no index, reads the corpus only to hash it (a second or two), and
it belongs after `stage.py` in every build.

Three things are checked, because each one alone would have missed the failure:

1. No restricted document is a file under `dist/www/`, by name, at any depth.
2. No published `index/doc/<id>.json` for a restricted document carries chunk
   text. The viewer needs the geometry to place a highlight and never reads the
   words, so the text is what gets stripped.
3. The `.gz` beside each of those says the same thing as the plain file. Caddy
   serves the pre-compressed copy whenever the client accepts it, so a stale one
   hands back exactly what the plain file no longer carries.

Which documents are restricted comes from TWO places, and they must agree: the
manifest's `access` column, which is what `stage.py` obeyed, and the sha256 of
each original in `data/CORPUS_ADDITIONS.tsv`, which no script rewrites. Checking
the tree against the manifest alone would only prove the stager obeyed its own
input (see `restricted_digests`). A corpus that keeps no such record (one that
never took a document in on condition) is checked against the manifest alone,
and the run says so; a corpus that has the record is always checked against it.

Written by Claude Code (Opus 5).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import click
from loguru import logger

from lib import hashing, manifest_io
from lib.corpus import in_corpus


def read_tiers(manifest: Path) -> dict[str, str]:
    """Filename to `access` tier, as `stage.py` read it.

    Parameters
    ----------
    manifest
        `data/MANIFEST.tsv`.

    Returns
    -------
    dict[str, str]
        Filename to `"open"` or `"restricted"`.

    Raises
    ------
    click.ClickException
        On a missing column or a blank or unknown tier (`manifest_io.read_access`,
        the same reader `stage.py` uses, so the two cannot disagree on a cell).
    """
    try:
        return manifest_io.read_access(manifest)
    except ValueError as error:
        raise click.ClickException(str(error)) from error


def restricted_digests(additions: Path) -> set[str]:
    """The sha256 of every document that entered the corpus as `restricted`.

    Parameters
    ----------
    additions
        `data/CORPUS_ADDITIONS.tsv`, one row per document moved into the corpus
        after the first sort, with its sha256 and its `access` tier.

    Returns
    -------
    set[str]
        Hex digests.

    Notes
    -----
    This is the gate's anchor INDEPENDENT of the manifest. The manifest is what
    `stage.py` read, so checking the staged tree against it alone only proves the
    stager obeyed its own input: a restricted PDF renamed on disk gets a fresh
    manifest row, `manifest.py` writes `access=open` on it, `stage.py` publishes
    it and a manifest-only gate agrees. The file's CONTENT does not change with
    its name, and this record is not rewritten by any script.
    """
    rows = manifest_io.read_rows(additions)
    if rows and "access" not in rows[0]:
        raise click.ClickException(f"{additions} has no 'access' column")
    return {row["sha256"] for row in rows if row["access"].strip() == "restricted"}


def restricted_by_content(corpus: Path, digests: set[str]) -> set[str]:
    """Names of the corpus files whose bytes are a restricted document.

    Parameters
    ----------
    corpus
        `data/GUIDELINES/`.
    digests
        Output of :func:`restricted_digests`.

    Returns
    -------
    set[str]
        Filenames, as they are named on disk today.

    Notes
    -----
    The ORIGINAL is hashed, not the staged file: 22 restricted documents are
    served from an OCR copy whose bytes differ, but which carries the original's
    name, so a name found here also catches that copy by name in the web root.
    """
    found = set()
    for path in sorted(corpus.glob("*.pdf")):
        digest = hashing.file_sha256(path)
        if digest in digests:
            found.add(path.name)
    return found


def offending_files(www: Path, restricted: set[str]) -> list[Path]:
    """Every path under `www` whose name is a restricted document.

    Symlinks count: what matters is what a GET returns, and Caddy follows them.
    """
    return sorted(p for p in www.rglob("*") if p.name in restricted)


def offending_text(www: Path, meta: Path, restricted: set[str]) -> list[tuple[Path, int]]:
    """Published per-document text files that still carry a restricted document's words.

    Returns
    -------
    list[tuple[Path, int]]
        The file and how many characters of chunk text it holds.
    """
    if not meta.exists():
        return []
    documents = json.loads(meta.read_text(encoding="utf-8")).get("documents", [])
    found: list[tuple[Path, int]] = []
    for document in documents:
        if document.get("file") not in restricted:
            continue
        for path in (www / "index" / "doc" / f"{document['id']}.json",
                     www / "index" / "doc" / f"{document['id']}.json.gz"):
            if not path.exists():
                continue
            raw = (gzip.decompress(path.read_bytes()) if path.suffix == ".gz"
                   else path.read_bytes())
            payload = json.loads(raw.decode("utf-8"))
            size = sum(len(c.get("text", "")) for c in payload.get("chunks", []))
            if size:
                found.append((path, size))
    return found


@click.command()
@click.option("--www", type=click.Path(file_okay=False, path_type=Path),
              **in_corpus("dist/www"),
              help="The served tree, as stage.py built it.")
@click.option("--manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              **in_corpus("data/MANIFEST.tsv"),
              help="Where the access tiers come from.")
@click.option("--index-dir", type=click.Path(file_okay=False, path_type=Path),
              **in_corpus("dist/index"),
              help="The built index, read for its document ids.")
@click.option("--additions", type=click.Path(dir_okay=False, path_type=Path),
              **in_corpus("data/CORPUS_ADDITIONS.tsv"),
              help="The record of which documents entered as restricted, by sha256. "
                   "Optional: without it only the manifest says what is restricted.")
@click.option("--corpus", type=click.Path(exists=True, file_okay=False, path_type=Path),
              **in_corpus("data/GUIDELINES"),
              help="The originals, hashed to find the restricted ones by content.")
def main(www: Path, manifest: Path, index_dir: Path, additions: Path, corpus: Path) -> None:
    """Refuse a served tree that exposes a restricted document."""
    if not www.exists():
        raise click.ClickException(f"no served tree at {www}; run stage.py first.")
    tiers = read_tiers(manifest)
    restricted = {name for name, tier in tiers.items() if tier == "restricted"}

    # The manifest says what stage.py did; the content says what it should have.
    # A restricted file the manifest calls anything else is refused outright, even
    # before looking at the tree, because the next stage would publish it.
    if additions.is_file():
        by_content = restricted_by_content(corpus, restricted_digests(additions))
    else:
        logger.info(f"no {additions}: the manifest's access column is the only record "
                    "of which documents are restricted")
        by_content = set()
    mislabelled = sorted(by_content - restricted)
    for name in mislabelled:
        logger.error(f"{name} is a restricted document by its content "
                     f"(data/CORPUS_ADDITIONS.tsv) but the manifest says "
                     f"access={tiers.get(name, '<no row>')!r}")
    if mislabelled:
        raise click.ClickException(
            f"{len(mislabelled)} restricted document(s) are not marked restricted in "
            f"{manifest}. Set their 'access' cell to 'restricted', then restage.")
    restricted |= by_content
    if not restricted:
        logger.success("no restricted documents in the manifest; nothing to check")
        return

    files = offending_files(www, restricted)
    texts = offending_text(www, index_dir / "meta.json", restricted)

    for path in files:
        logger.error(f"{path} is a restricted document inside the web root")
    for path, size in texts:
        logger.error(f"{path} carries {size:,} characters of a restricted document's text")
    if files or texts:
        raise click.ClickException(
            f"{len(files)} restricted file(s) and {len(texts)} text file(s) are "
            "reachable under the web root. These documents entered the corpus on "
            "the condition that they are served without download (see "
            "data/CORPUS_ADDITIONS.tsv). Re-run stage.py, which splits them out.")

    logger.success(f"{len(restricted)} restricted documents, none reachable under {www}: "
                   "no file by name, no published text")


if __name__ == "__main__":
    main()
