# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "pymupdf==1.28.2"]
# ///
"""Find the figures in the corpus, crop them, and keep a record of what a model read in them.

A flowchart is the most useful page of a guideline and the worst-indexed one. A
raster figure has no text layer at all, so nothing of it is searchable; a vector
flowchart (a decision tree in a clinical or engineering guide) has its words in the text layer, but
the chunker reads its boxes in layout order, so "if no early improvement after
2-4 weeks, switch or add" comes out as a scatter of box labels with the arrows
gone. The fix is a written description of each figure, made once by a vision
model and kept, so that adding a document means describing ITS figures and
nothing else.

Four commands, in the order they are used:

    uv run scripts/figures.py census                 # data/FIGURE_CENSUS.tsv + counts
    uv run scripts/figures.py extract --sample 20    # crops + data/private/figures/figures.tsv
    uv run scripts/figures.py tables                 # one crop per table chunk.py read out
    uv run scripts/figures.py emit --batch 30          # batch-NNN.tsv: the next undescribed figures
    uv run scripts/figures.py apply <sheet.tsv> --model claude-sonnet-5

`census` scores every page on three cheap signals (below) and is the estimate the
pass was sized from. `extract` renders one crop per selected page, keyed by the
PDF's sha256, the page and the crop box, so re-running it after adding PDFs only
crops the new pages. It also renders the WHOLE page beside the crop, and the
describing model gets both plus the document's title: a figure is often
meaningless on its own (a curve with no drug named, "Groupe 1 / Groupe 2"),
and the page around it says which population, which treatment and which
question it answers. The crop stays, for the detail a page render at a
readable size loses, and because its box is what a highlight will be drawn on. `apply` validates a description sheet and appends it to
`descriptions.tsv` with the MODEL that wrote each row, because the first pass is
Sonnet and an Opus pass over the same keys is expected later: a later row for a
key supersedes an earlier one, and both stay in the file.

`tables` adds every table `chunk.py` serialised (it records their boxes in
each chunk file's `table_boxes`), with stratum `table`. The cell-by-cell text
stays indexed as it is; the description is a second chunk for the same table,
which is what finds a table the serialiser read wrong without anyone knowing.
They are queued by corpus.toml [figures] `table_first` (filename patterns, the
first match wins: the reference works worth describing first), then everything
else (open documents before restricted ones), then `table_last` (older editions,
which mostly repeat the newest one).

`chunk.py` reads the result back through `lib/figure_record.py`: every described
figure becomes one chunk, boxed on its crop, so the next `./deploy.sh` indexes
whatever has been described by then and a new sheet only rechunks its documents.

Everything under `data/private/figures/` is untracked on purpose: the crops are
pieces of copyrighted documents and the descriptions paraphrase them.

The census signals, per page:

- `img_big`: raster images covering 5 to 85% of the page. A full-page image is
  a scan, which `ocr.py` already handles, so it is counted apart (`img_full`).
- `vec_share`: the largest cluster of vector drawings, as a share of the page.
  Tables and coloured boxes land here too, so on its own it is mostly noise.
- `captions`: text blocks opening with "Figure 2", "Algorithme 1", "Schéma 3"...

Calibrated on 2026-09-27 by looking at 20 sampled pages per stratum (one page
per document): a caption plus a raster or a >= 15% vector cluster is a real
figure about 4 times in 5; an uncaptioned raster past page 1 about 1 time in 4
(covers, logos, photos); an uncaptioned vector cluster almost never (tables).
So `extract` selects the first two strata and lets the model's `is_figure`
verdict throw out the rest.

Written by Claude Code (Opus 5.5).
"""

from __future__ import annotations

import collections
import csv
import datetime
import json
import random
import re
from multiprocessing import Pool
from pathlib import Path

import click
import pymupdf

from lib import atomic, hashing
from lib import corpus_config, manifest_io, manifest_policy
from lib.corpus import corpus_path
from lib.figure_record import WORK, figure_key, read_tsv

CORPUS = Path("data/GUIDELINES")
MANIFEST = Path("data/MANIFEST.tsv")
CENSUS = Path("data/FIGURE_CENSUS.tsv")
CAPTION = re.compile(
    r"^\s*(fig(ure|\.)?|algorithm[e]?|sch[ée]ma|arbre (d[ée]cisionnel)?|flow ?chart|diagram(me)?|infographi[ce])\s*[0-9IVX]",
    re.I)
CENSUS_FIELDS = ["file", "page", "img_big", "img_full", "n_drawings", "vec_share", "captions"]
FIGURE_FIELDS = ["key", "file", "page", "bbox", "stratum", "title", "crop", "page_png"]
KINDS = {"flowchart", "chart", "diagram", "infographic", "table", "photo", "decorative", "other"}
SHEET_FIELDS = ["key", "is_figure", "kind", "language", "description"]
DESCRIPTION_FIELDS = SHEET_FIELDS + ["model", "date"]
IMG_MIN, IMG_MAX, VEC_MIN = 0.05, 0.85, 0.15
# The crop is for reading the figure, the page render for knowing what it is about
# (its heading, the paragraph that cites it). 100 dpi keeps body text legible.
DPI, PAGE_DPI, PAD = 150, 100, 20
CHUNKS = Path("data/chunks")


def figure_rects(page: pymupdf.Page) -> tuple[list, list, list, list, int]:
    """(mid-size rasters, full-page rasters, drawing clusters, caption blocks, drawing count) of one page."""
    area = page.rect.get_area() or 1
    images = [r for r in ((pymupdf.Rect(i["bbox"]) & page.rect) for i in page.get_image_info())
              if IMG_MIN <= r.get_area() / area < IMG_MAX]
    full = [r for r in ((pymupdf.Rect(i["bbox"]) & page.rect) for i in page.get_image_info())
            if r.get_area() / area >= IMG_MAX]
    drawings = page.get_drawings()
    clusters = page.cluster_drawings(drawings=drawings) if len(drawings) >= 10 else []
    captions = [pymupdf.Rect(b[:4]) for b in page.get_text("blocks") if CAPTION.match(b[4])]
    return images, full, clusters, captions, len(drawings)


def scan(path: Path) -> list[dict]:
    rows = []
    for page in pymupdf.open(path):
        area = page.rect.get_area() or 1
        images, full, clusters, captions, n_drawings = figure_rects(page)
        rows.append({
            "file": path.name, "page": page.number + 1,
            "img_big": len(images), "img_full": len(full), "n_drawings": n_drawings,
            "vec_share": round(max((r.get_area() / area for r in clusters), default=0), 3),
            "captions": len(captions),
        })
    return rows


def stratum(r: dict) -> str | None:
    """Which calibrated stratum a census row falls in, or None when it is not selected."""
    if int(r["captions"]) > 0 and (int(r["img_big"]) > 0 or float(r["vec_share"]) >= VEC_MIN):
        return "captioned"
    if int(r["img_big"]) > 0 and int(r["page"]) > 1:
        return "raster"
    return None


def append_tsv(path: Path, fields: list[str], rows: list[dict]) -> None:
    """Append rows by writing a new copy and renaming it over the old one.

    A deploy can run `chunk.py` while a batch is applied, and an in-place append
    read halfway would index a truncated description. A rename is atomic, so a
    reader sees the file before this batch or after it, never in between.
    """
    with atomic.open_atomic(path, encoding="utf-8", newline="") as f:
        if path.exists():
            f.write(path.read_text(encoding="utf-8"))
        w = csv.DictWriter(f, fields, delimiter="\t", extrasaction="ignore")
        if not path.exists():
            w.writeheader()
        w.writerows(rows)


@click.group()
def cli() -> None:
    # The paths above are corpus-relative so that importing this module needs no
    # corpus; every subcommand runs after this, so resolving them once here is
    # enough (lib/corpus.py).
    global CORPUS, MANIFEST, CENSUS, CHUNKS, WORK
    CORPUS, MANIFEST, CENSUS, CHUNKS, WORK = (
        corpus_path(p) for p in (CORPUS, MANIFEST, CENSUS, CHUNKS, WORK))


@cli.command()
@click.option("--jobs", default=12, show_default=True)
@click.option("--reuse/--rescan", default=True, help="Only print the summary when the census exists.")
def census(jobs: int, reuse: bool) -> None:
    """Score every page, write data/FIGURE_CENSUS.tsv, print counts per tier."""
    if not (reuse and CENSUS.exists()):
        with Pool(jobs) as pool:
            rows = [r for doc in pool.imap(scan, sorted(CORPUS.glob("*.pdf")), chunksize=2) for r in doc]
        with CENSUS.open("w", newline="") as f:
            w = csv.DictWriter(f, CENSUS_FIELDS, delimiter="\t")
            w.writeheader()
            w.writerows(rows)
    rows = read_tsv(CENSUS)
    manifest = manifest_io.read_by_file(MANIFEST)
    click.echo(f"{len(rows)} pages, {len({r['file'] for r in rows})} documents")
    pages, docs = collections.Counter(), collections.defaultdict(set)
    for r in rows:
        if s := stratum(r):
            m = manifest[r["file"]]
            key = (s, m["access"], m.get(manifest_policy.TIER_COLUMN, ""))
            pages[key] += 1
            docs[key].add(r["file"])
    for key in sorted(pages):
        click.echo(f"  {key[0]:10s} {key[1]:10s} {key[2]:7s} {pages[key]:5d} pages {len(docs[key]):4d} docs")


def table_queue() -> tuple[list[re.Pattern], list[re.Pattern]]:
    """The `tables` queue order, from corpus.toml [figures] (both keys optional).

    Returns
    -------
    tuple of (list of Pattern, list of Pattern)
        `table_first` and `table_last`, case-insensitive filename regexes. The first
        pattern a filename matches wins; unmatched files go between the two groups.
    """
    config = corpus_config.load().get("figures", {})
    def compiled(key: str) -> list[re.Pattern]:
        return [re.compile(pattern, re.I) for pattern in config.get(key, [])]
    return compiled("table_first"), compiled("table_last")


def table_rank(name: str, access: str, queue: tuple[list[re.Pattern], list[re.Pattern]]
               ) -> tuple[int, int]:
    """Where a document's tables go in the queue; see `table_queue`."""
    first, last = queue
    for rank, pattern in enumerate(first):
        if pattern.search(name):
            return rank, 0
    if any(pattern.search(name) for pattern in last):
        return len(first) + 1, 0
    return len(first), access == "restricted"


def record_crop(page: pymupdf.Page, box: pymupdf.Rect, *, name: str, sha: str, why: str,
                title: str, known: set[str]) -> dict | None:
    """Render one crop and its page, or None when that key is already recorded."""
    bbox = ",".join(f"{v:.0f}" for v in box)
    key = figure_key(sha, page.number + 1, bbox)
    if key in known:
        return None
    crops = WORK / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    out = crops / f"{key}.png"
    page.get_pixmap(dpi=DPI, clip=box).save(out)
    whole = crops / f"{key}-page.png"
    page.get_pixmap(dpi=PAGE_DPI).save(whole)
    known.add(key)
    return {"key": key, "file": name, "page": page.number + 1, "bbox": bbox, "stratum": why,
            "title": title, "crop": str(out), "page_png": str(whole)}


def crop_box(page: pymupdf.Page) -> pymupdf.Rect:
    """The union of the page's figure rects and captions, padded; the whole page when that is most of it."""
    images, _, clusters, captions, _ = figure_rects(page)
    area = page.rect.get_area() or 1
    parts = images + [c for c in clusters if c.get_area() / area >= VEC_MIN] + captions
    if not parts:
        return page.rect
    box = pymupdf.Rect(parts[0])
    for r in parts[1:]:
        box |= r
    box = (box + (-PAD, -PAD, PAD, PAD)) & page.rect
    return page.rect if box.get_area() / area >= IMG_MAX else box


@cli.command()
@click.option("--sample", type=int, default=0,
              help="Take N selected pages, at most one per document, instead of all of them.")
@click.option("--seed", default=0, show_default=True)
@click.option("--page", "forced", multiple=True, metavar="FILE:PAGE",
              help="Always include this page, whatever the census says. Repeatable.")
def extract(sample: int, seed: int, forced: tuple[str, ...]) -> None:
    """Crop the selected pages into data/private/figures/crops/ and record them."""
    rows = read_tsv(CENSUS)
    if not rows:
        raise click.ClickException(f"{CENSUS} is missing: run `census` first")
    picked = [(r["file"], int(r["page"]), s) for r in rows if (s := stratum(r))]
    if sample:
        rng = random.Random(seed)
        one = {}
        for item in rng.sample(picked, len(picked)):
            one.setdefault(item[0], item)
        picked = rng.sample(sorted(one.values()), min(sample, len(one)))
    for spec in forced:
        name, _, page = spec.rpartition(":")
        if not (CORPUS / name).exists():
            raise click.ClickException(f"not in {CORPUS}: {name}")
        picked.insert(0, (name, int(page), "forced"))

    manifest = manifest_io.read_by_file(MANIFEST)
    # Open documents first: `emit` hands out batches in this order, so a pass cut
    # short by a usage limit has covered what most readers search.
    picked.sort(key=lambda item: manifest[item[0]]["access"] == "restricted")
    index = WORK / "figures.tsv"
    known = {r["key"] for r in read_tsv(index)}
    shas: dict[str, str] = {}
    new = []
    for name, number, why in picked:
        path = CORPUS / name
        if name not in shas:
            shas[name] = hashing.file_sha256(path)
        page = pymupdf.open(path)[number - 1]
        row = record_crop(page, crop_box(page), name=name, sha=shas[name], why=why,
                          title=manifest[name]["title"], known=known)
        if row:
            new.append(row)
    append_tsv(index, FIGURE_FIELDS, new)
    click.echo(f"{len(new)} new crops, {len(known)} in {index}")


@cli.command()
@click.option("--limit", type=int, default=0, help="Stop after N new crops (a pilot).")
def tables(limit: int) -> None:
    """Crop every table chunk.py read out (its `table_boxes`) and record it, stratum `table`."""
    manifest = manifest_io.read_by_file(MANIFEST)
    found = []
    for path in sorted(CHUNKS.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if "table_boxes" not in doc:
            raise click.ClickException(f"{path} has no table_boxes: run `chunk.py --force` once")
        # An OCR text layer (source "ocr") keeps the original's page geometry, so its
        # boxes crop the original PDF just as well.
        if doc["table_boxes"]:
            found.append((doc["file"], doc["table_boxes"]))
    queue = table_queue()
    found.sort(key=lambda item: table_rank(item[0], manifest[item[0]]["access"], queue))
    index = WORK / "figures.tsv"
    known = {r["key"] for r in read_tsv(index)}
    new = []
    for name, boxes in found:
        path = CORPUS / name
        sha = hashing.file_sha256(path)
        with pymupdf.open(path) as doc:
            for number, box in boxes:
                page = doc[number - 1]
                # The chunk file's boxes are upright, a crop is drawn on the page as shown.
                shown = ((pymupdf.Rect(box) * page.rotation_matrix).normalize()
                         + (-PAD, -PAD, PAD, PAD)) & page.rect
                row = record_crop(page, shown, name=name, sha=sha, why="table",
                                  title=manifest[name]["title"], known=known)
                if row:
                    new.append(row)
                if limit and len(new) >= limit:
                    break
        if limit and len(new) >= limit:
            break
    append_tsv(index, FIGURE_FIELDS, new)
    click.echo(f"{len(new)} new table crops, {len(known)} in {index}")


@cli.command()
@click.option("--batch", "size", default=30, show_default=True)
def emit(size: int) -> None:
    """Write the next SIZE undescribed figures to batch-NNN.tsv, for one describing model to take."""
    done = {r["key"] for r in read_tsv(WORK / "descriptions.tsv")}
    todo = [r for r in read_tsv(WORK / "figures.tsv") if r["key"] not in done]
    if not todo:
        click.echo("nothing left to describe")
        return
    number = 1 + max((int(p.stem.split("-")[1]) for p in WORK.glob("batch-[0-9][0-9][0-9].tsv")), default=0)
    out = WORK / f"batch-{number:03d}.tsv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, FIGURE_FIELDS, delimiter="\t")
        w.writeheader()
        w.writerows(todo[:size])
    click.echo(f"{out}: {min(size, len(todo))} figures, {len(todo) - min(size, len(todo))} left after it")


@cli.command()
@click.argument("sheet", type=click.Path(exists=True, path_type=Path))
@click.option("--model", required=True, help="The model that wrote the sheet, e.g. claude-sonnet-5.")
def apply(sheet: Path, model: str) -> None:
    """Validate a description sheet and append it to descriptions.tsv under MODEL."""
    known = {r["key"] for r in read_tsv(WORK / "figures.tsv")}
    rows, errors = read_tsv(sheet), []
    for i, r in enumerate(rows, 2):
        if missing := [f for f in SHEET_FIELDS if f not in r]:
            raise click.ClickException(f"{sheet} lacks the columns {missing}")
        if r["key"] not in known:
            errors.append(f"line {i}: unknown key {r['key']}")
        if r["is_figure"] not in {"yes", "no"}:
            errors.append(f"line {i}: is_figure must be yes or no, not {r['is_figure']!r}")
        if r["kind"] not in KINDS:
            errors.append(f"line {i}: kind {r['kind']!r} is not one of {sorted(KINDS)}")
        if r["language"] not in {"fr", "en"}:
            errors.append(f"line {i}: language must be fr or en")
        if r["is_figure"] == "yes" and len(r["description"]) < 40:
            errors.append(f"line {i}: a figure needs a real description")
        r.update(model=model, date=datetime.date.today().isoformat())
    if errors:
        raise click.ClickException("\n".join(errors))
    append_tsv(WORK / "descriptions.tsv", DESCRIPTION_FIELDS, rows)
    click.echo(f"{len(rows)} descriptions by {model}, {sum(r['is_figure'] == 'yes' for r in rows)} of them figures")


if __name__ == "__main__":
    cli()
