"""The figure descriptions `figures.py` records, read back as chunks.

`figures.py` crops every figure it finds and has a vision model describe it
(the reviewers' brief belongs to the corpus, e.g. `$CORPUS_DIR/docs/figure_review.md`),
because a flowchart's text layer comes out of the chunker as scattered box labels
and a raster figure has no text layer at all. This module is the other end: `chunk.py` asks it which figures of a document are described, folds
them into that document's content hash, and appends one chunk per figure whose text
is the description and whose single box is the figure's crop, so a hit opens the
page with the figure itself highlighted.

Three rules, each one a way the record could otherwise lie:

- The LAST description of a key wins (`descriptions.tsv` is append-only, and a later
  Opus pass over the same crops is meant to supersede the Sonnet one).
- A figure whose key no longer matches its PDF is dropped: the key is a hash of the
  PDF's bytes, the page and the crop box, so a replaced PDF leaves its old
  descriptions behind instead of pinning them to whatever the new page holds.
- The model travels with the chunk, so the reader is told the text is a model's
  description and not the document's words.

Everything it reads is under `data/private/figures/`, untracked: descriptions
paraphrase copyrighted pages. Needs pymupdf only to turn a crop box, drawn in the
page's displayed orientation, into the unrotated space every other chunk box is in.
Imported, never run. Written by Claude Code (Opus 5.5).
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from lib import hashing
from lib.corpus import corpus_path

WORK = Path("data/private/figures")


@dataclass(frozen=True)
class Figure:
    key: str
    page: int                               # 1-based
    bbox: tuple[float, float, float, float]  # page.rect space, as figures.py cropped it
    description: str
    kind: str
    model: str
    language: str = ""                      # fr or en, the figure's own text


def read_tsv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def figure_key(pdf_sha: str, page: int, bbox: str) -> str:
    """The crop's key: which PDF bytes, which page, which box."""
    return hashlib.sha256(f"{pdf_sha}:{page}:{bbox}".encode()).hexdigest()[:16]


def described(corpus: Path, work: Path = WORK) -> dict[str, list[Figure]]:
    """Every document's described figures, by PDF filename, in page order.

    Only `is_figure = yes` rows, only the latest description per key, and only keys
    that still match the PDF on disk under `corpus`.
    """
    # Corpus-relative until here (WORK), so importing this module needs no corpus.
    work = corpus_path(work)
    latest: dict[str, dict] = {}
    for row in read_tsv(work / "descriptions.tsv"):
        latest[row["key"]] = row
    by_file: dict[str, list[Figure]] = {}
    shas: dict[str, str] = {}
    for row in read_tsv(work / "figures.tsv"):
        said = latest.get(row["key"])
        if not said or said["is_figure"] != "yes":
            continue
        pdf = corpus / row["file"]
        if not pdf.exists():
            continue
        if row["file"] not in shas:
            shas[row["file"]] = hashing.file_sha256(pdf)
        if figure_key(shas[row["file"]], int(row["page"]), row["bbox"]) != row["key"]:
            continue
        box = tuple(float(v) for v in row["bbox"].split(","))
        by_file.setdefault(row["file"], []).append(Figure(
            key=row["key"], page=int(row["page"]), bbox=box,
            description=said["description"].strip(), kind=said["kind"], model=said["model"],
            language=said.get("language", "")))
    for figures in by_file.values():
        figures.sort(key=lambda f: (f.page, f.key))
    return by_file


def fold(digest: str, figures: list[Figure]) -> str:
    """A document's content hash with its figure descriptions folded in.

    Unchanged when there are none, so a document with no described figure keeps the
    hash (and the cached chunks and vectors) it had before this module existed.
    """
    if not figures:
        return digest
    payload = json.dumps([[f.key, f.description, f.kind, f.model, f.language] for f in figures])
    return hashlib.sha256(f"{digest}:{payload}".encode()).hexdigest()


def chunk_entries(pdf: Path, figures: list[Figure], tokenizer, start: int) -> list[dict]:
    """One chunk entry per figure, numbered from `start`, in chunk.py's JSON shape.

    The box is the crop, derotated and clipped to the unrotated page, rounded to whole
    points like every other box.
    """
    if not figures:
        return []
    import pymupdf

    entries = []
    with pymupdf.open(pdf) as doc:
        for f in figures:
            page = doc[f.page - 1]
            upright = pymupdf.Rect(0, 0, page.cropbox.width, page.cropbox.height)
            box = (pymupdf.Rect(f.bbox) * page.derotation_matrix).normalize() & upright
            entries.append({
                "i": start + len(entries), "text": f.description,
                "n_tokens": len(tokenizer.encode(f.description).ids),
                "pages": [f.page],
                "boxes": {str(f.page): [[int(round(v)) for v in box]]},
                "figure": {"model": f.model, "kind": f.kind, "language": f.language},
            })
    return entries
