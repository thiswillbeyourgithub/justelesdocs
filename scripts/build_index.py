# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "loguru", "numpy"]
# ///
"""Turn the baked vectors into the files the browser downloads.

`embed.py` writes one `.npz` per document at the model's full 1024 dimensions.
This script narrows them to the width the query side uses, quantises them the way
the query side quantises, and splits the result into an EAGER part the page
fetches once and a LAZY part it fetches only for results the reader actually
sees.

Layout under `<out>/index/`:

``meta.json``
    Format version, vector geometry, and one entry per document carrying its
    `MANIFEST.tsv` metadata (for the filters) plus where its chunks live in the
    vector file. Around 40 KB.
``vectors-<hash>.b1`` (or ``...i8`` with ``--quant int8``)
    `n_chunks` x `dims` passage vectors, row-major, in the same order as
    `meta.json`'s documents and each document's own chunk order. One BIT per
    dimension by default, packed eight to a byte, most significant bit first:
    3.5 MB at 27k chunks and 1024 dims. This is the one big eager download.
    `meta.json` names the file, so the client never hardcodes it.
``chunks-<hash>.u16``
    Four little-endian uint16 per chunk: the document's index in `meta.json`,
    the chunk's primary page, the row that page occupies in the page vector
    file, and the row its section occupies in the section vector file (0xFFFF
    for either when it has none). The first two are enough to render a ranked
    list of "document, page" without fetching anything else; the last two are
    what let the client blend in the page and section scores without a second
    table. 8 bytes a chunk, so ~1 MB.
``pages-<hash>.b1`` (or ``...i8``)
    `n_pages` x `dims` vectors of WHOLE PAGES, packed exactly like the passage
    file. About 7,900 rows against 26,700 passages, so it costs 28% more eager
    download (around 1 MB at 1024 dims) and buys the gain below.
``sections-<hash>.b1`` (or ``...i8``)
    `n_sections` x `dims` vectors of whole SECTIONS (a heading run and what sits
    under it, from `chunk.py --section-chunks`), packed the same way. Present
    only when `--section-weight` is above zero. A chunk is joined to its
    section by the boxes the two share, see `section_rows`, so neither chunk
    file has to record row indices.

**The two big files carry a content hash in their name, and that is a caching
decision.** `meta.json` is the only name the client knows, it is small, and
`docker/Caddyfile` serves it `no-cache` so every visit revalidates it. Everything
it points at is content-addressed, so those get `immutable` and a one-year
max-age: a browser that already holds `vectors-3f2a....b1` reuses it without a
request, and a rebuilt index cannot be misread from cache because it has a
different name. The alternative, a fixed name plus revalidation, costs a
round trip per visit for a 3.5 MB file and still has a window where a stale
vector file meets a fresh meta.json. Note that the stale window is not a
performance question: `meta.json` carries the document table and the chunk
offsets that index INTO the vector file, so an old pairing does not degrade
ranking, it reads the wrong rows.
``doc/<id>.json``
    Per document, every chunk's text, the pages it spans and its highlight boxes.
    Fetched only for documents that appear in the results. This is what keeps the
    eager download at 7 MB instead of 40: the chunk text alone is 33 MB of UTF-8
    across the corpus, and a reader looks at ten chunks, not 27,309.

**Given a byte budget, buy DIMENSIONS, not precision.** That is the measurement
this script's defaults are built on (DESIGN.md, and `data/EVAL_RESULTS_quant.tsv`
from `evaluate.py --quant`). Across 117 queries, 1024 dims at one bit per
dimension scores page@1 0.4188 in 3.3 MB, while the 256-dim int8 index that
shipped first scores 0.3590 in 6.7 MB: better retrieval at half the size. It is
also indistinguishable from 1024-dim int8, which costs 26.7 MB. The ordering
reverses at narrow widths (binary is the worst scheme at 256 dims and the best at
1024), so this is not "binary is good", it is "spend the bits on width".

**A passage is scored partly by the page it sits on.** The client ranks on
`0.85 * cosine(query, chunk) + 0.15 * cosine(query, page)`, and the 15% is
measured rather than picked (DESIGN.md, `data/EVAL_RESULTS_pagefirst*.tsv`).
Across 117 queries on the shipped binary format it moves page@5 from 0.624 to
0.709 and MRR from 0.511 to 0.544, at the cost of one more eager file.

The reason it works is quantisation, not semantics, which is worth knowing
because it predicts when to drop it: at int8 the same blend is worth almost
nothing (MRR 0.548 to 0.554). A page vector aggregates about four times as much
text as a passage vector, so the SIGN of each of its components is far more
stable, and a one-bit index keeps nothing but signs. The page term is therefore a
cheap partial repair of what binarisation threw away. Build with
`--page-weight 0` to ship a pure passage index; the client reads the weight from
`meta.json` rather than holding an opinion.

A hard two-stage filter (rank pages, then only rank passages on the best K) was
measured too and is worse at every K: a correct page outside the top K can never
be recovered, and at K=3 that costs twelve queries at @10.

**Scoring stays ASYMMETRIC: only the passage side is binarised.** The query keeps
its full precision, which costs nothing because it is encoded per request anyway,
and it is worth a lot: a sign-vs-sign comparison throws away the query's
magnitudes as well as the passage's. The client turns the int8 query into a
lookup table over byte values once per query, then each passage costs `dims / 8`
table lookups instead of `dims` multiplications, so the wider index is also the
faster one to scan.

**With ``--quant int8``, the scale is fixed and that is not a free choice.**
justelesRCP's embed service returns query vectors as base64 int8 using a FIXED
scale (``q = round(v * 127)`` clamped to +/-127, read back as ``q / 127``; see
``build.quantize_int8`` there). The passage side must then use the same fixed
scale, because ranking compares one query against every passage: a per-vector
scale would make each passage's dot product carry its own factor and reorder the
list. With a fixed scale the client can rank on raw int8 dot products and never
dequantise at all. Note that the QUERY is int8 either way; only the stored side
changes, so this wire format is untouched by the default.

Note that this is NOT the quantisation `evaluate.py` measured: that one uses a
per-vector scale. The difference looks alarming and is not. A unit vector's
components are around 1/sqrt(dims), so at 256 dims the fixed scale produces a
median |q| of 6 and a maximum of 28, using under six of the eight available bits.
Measured anyway, the cosine between a float32 vector and its fixed-scale int8
form is **0.99934**: the per-component error averages out across 256 dimensions,
and nothing that agrees to four decimal places reorders a result list.

So do not "fix" the unused range. Recovering it would mean a per-vector scale, the
client would have to dequantise every passage before comparing, and the query side
cannot change anyway: justelesRCP's index is already baked against this wire
format. The waste is real and it buys simplicity for free.

Written by Claude Code.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click
import numpy as np
from loguru import logger

from lib import corpus_config, manifest_io, vocabulary
from lib import manifest_policy
from lib.corpus import in_corpus

# Bump when the on-disk layout changes in a way a deployed client cannot read.
# The client refuses an index whose version it does not know, rather than
# misreading it: a silently misread vector file ranks nonsense convincingly.
# Version 2 packs passage vectors to one bit per dimension, and names both big
# files in meta.json (with a content hash in each name) instead of fixing them as
# vectors.i8 and chunks.u16.
# Version 3 adds the page vector file, widens each chunk row from two uint16 to
# three (the third being the page's row in that file), and carries the blend
# weight. A version 2 client reading it would stride the chunk rows wrongly and
# put every result on the wrong page, which is exactly the kind of confident
# nonsense the version check exists to prevent.
# Version 4 widens each chunk row to four uint16 (the fourth being the row of
# the chunk's section in the section vector file), adds that file, and carries
# two more blend weights: the section's and the previous page's. The previous
# page needs no file of its own, the client finds its row from the locations.
INDEX_FORMAT_VERSION = 5

# The chunk row value meaning "this chunk's page has no vector", which happens
# when a page carries boxes but no packable text. uint16 max, so it cannot
# collide with a real row: the page file would have to hold 65,535 pages, eight
# times the whole corpus.
NO_PAGE_ROW = 0xFFFF
# The same sentinel in the fourth slot: "this chunk's section has no vector". It
# happens when a chunk's boxes match no section's, which a mismatch between the
# two chunkings would cause, so the writer counts these and refuses past a share.
NO_SECTION_ROW = 0xFFFF
# uint16 values per chunk row. src/search.js holds the same number as LOC_STRIDE.
LOC_STRIDE = 4
# Share of chunks allowed to find no section. Above it, the section bake was
# almost certainly made from different text than the passage bake.
ORPHAN_SHARE = 0.02

# The scale the shared encoder uses for query vectors. Not a tuning knob: see the
# module docstring. Changing it here without changing it there breaks ranking.
INT8_SCALE = 127

# The manifest columns `meta.json` carries, which every visitor downloads. An
# ALLOW-list rather than "everything but notes", because meta.json is public and
# the manifest is a working file: a column added to it for curation (a reviewer's
# remark, a licence query, a contact) would otherwise be published by the next
# build without anyone deciding it should be. Every column has to be in exactly one
# of the two sets, and `public_meta` refuses a manifest with one in neither, so a
# new column fails the build until someone says which side it is on.
PUBLIC_COLUMNS = frozenset({
    "title", "issuer", "country", "year", "language", "family", "rendition", "pages",
    "source_url", "doi", "isbn", "access",
    # The corpus's own columns (its closed vocabularies and its tier): what the
    # facets filter on, so public by construction.
    *manifest_policy.CORPUS_COLUMNS,
})
# Curation notes: free text about sourcing and licence doubts, for the people
# maintaining the corpus. Nothing in src/ reads it.
PRIVATE_COLUMNS = frozenset({"notes"})


def check_vocabulary(metadata: dict[str, dict[str, str]], policy: dict[str, Any]) -> None:
    """Refuse to build an index whose facet cells name unknown slugs.

    Parameters
    ----------
    metadata
        The manifest rows, keyed by filename.
    policy
        The result of `lib.manifest_policy.policy()`.

    Raises
    ------
    click.ClickException
        If any `doc_type` or `topic` cell holds a slug outside its vocabulary.

    Notes
    -----
    `manifest.py` already gates this when it writes the TSV, through the same
    `lib.vocabulary.check`, so this is the second line of one defence, and it earns
    its place: `--manifest` accepts any TSV, and the manifest is edited by hand far
    more often than it is regenerated. A bad slug that reaches the index is
    invisible, because the UI's label lookup falls back to the slug itself and the
    bogus option looks deliberate.
    """
    try:
        vocabulary.check(metadata.values(), policy["vocabularies"], policy["separator"])
    except vocabulary.Unknown as exc:
        raise click.ClickException(str(exc)) from exc


@dataclass
class Document:
    """One document's place in the index.

    Attributes
    ----------
    file
        PDF filename, which is also its name under `dist/pdf/`.
    meta
        Its `MANIFEST.tsv` row, minus the filename, for the UI's filters.
    chunk_offset
        Row in the vector file where this document's first chunk lives.
    chunk_count
        How many consecutive rows belong to it.
    figure_count
        How many of those rows, at the END of the range, are described figures
        rather than the document's own text. chunk.py appends them after the
        text's chunks, so a range is enough and `rank()` needs no per-chunk flag
        to include, exclude or keep only them.
    """

    file: str
    meta: dict[str, str]
    chunk_offset: int
    chunk_count: int
    figure_count: int = 0


def trailing_figures(chunks: list[dict], name: str) -> int:
    """Count the figure chunks, refusing a document where one precedes text.

    Parameters
    ----------
    chunks
        One document's chunks, in index order.
    name
        Its filename, for the error.

    Returns
    -------
    int
        How many chunks at the end of `chunks` are figures.
    """
    flags = [bool(c.get("figure")) for c in chunks]
    count = sum(flags)
    if any(flags[:len(flags) - count]):
        raise click.ClickException(
            f"{name}: a figure chunk before a text chunk. The figures filter in "
            "src/search.js reads them as the tail of the document's range.")
    return count


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    """Load `MANIFEST.tsv` keyed by filename.

    Parameters
    ----------
    path
        The TSV written by `manifest.py`.

    Returns
    -------
    dict[str, dict[str, str]]
        Filename to its column values, with the filename column removed.

    Raises
    ------
    click.ClickException
        If the file is empty, or has no `file` column.

    Notes
    -----
    The filename is dropped from the row because it is already the key, and
    because every remaining column is copied verbatim into `meta.json` as the
    document's metadata: leaving `file` in would ship it twice under two names.

    This used to split each line on tabs, which is not what `manifest.py` writes.
    `csv.DictWriter` quotes any cell holding a `"`, a tab or a newline, and a
    curated title is exactly where one turns up; the split read it back mangled,
    silently, all the way onto the result card. See lib/manifest_io.
    """
    try:
        rows = manifest_io.read_by_file(path)
    except KeyError as exc:
        raise click.ClickException(
            f"{path} has no 'file' column, so it is not a manifest. Run manifest.py."
        ) from exc
    if not rows:
        raise click.ClickException(f"{path} is empty; run manifest.py.")
    return {name: {column: value for column, value in row.items() if column != "file"}
            for name, row in rows.items()}


def primary_page(chunk: dict[str, Any]) -> int:
    """The page a result list should name for this chunk.

    Parameters
    ----------
    chunk
        One entry of a chunk file's `chunks` list.

    Returns
    -------
    int
        The page holding the most of the chunk's highlight boxes, falling back to
        the first page it spans.

    Notes
    -----
    A chunk can straddle a page break (`pages` is a list), so "the page it is on"
    needs a rule. Most boxes is the right one: a chunk whose last line spills onto
    the next page should open on the page carrying the sentence, not on the page
    carrying its final three words. The viewer still gets every page and every box
    from the lazy per-document file, so this only decides where it opens.
    """
    boxes = chunk.get("boxes") or {}
    if boxes:
        return int(max(boxes, key=lambda page: len(boxes[page])))
    return int(chunk["pages"][0])


def page_rows(payload: dict[str, Any], name: str) -> dict[int, int]:
    """Map each page number in a page bake to its row in that bake.

    Parameters
    ----------
    payload
        A chunk file written by `chunk.py --page-chunks`, whose chunks are one
        per page in page order.
    name
        The document file name, for error messages.

    Returns
    -------
    dict[int, int]
        `{page number: row index within this document's page vectors}`.

    Raises
    ------
    click.ClickException
        If two chunks claim the same page, or one spans several. Either would
        mean the file was not produced by `--page-chunks`, and pairing vectors
        with pages by position would then attach whole-page scores to the wrong
        pages, silently.
    """
    rows: dict[int, int] = {}
    for row, chunk in enumerate(payload["chunks"]):
        pages = chunk["pages"]
        if len(pages) != 1:
            raise click.ClickException(
                f"{name}: page chunk {row} spans {pages}. --page-vectors needs a "
                "bake of --page-chunks, where every chunk is exactly one page."
            )
        page = int(pages[0])
        if page in rows:
            raise click.ClickException(
                f"{name}: page {page} appears twice in the page chunks."
            )
        rows[page] = row
    return rows


def section_rows(payload: dict[str, Any], name: str) -> dict[tuple[int, tuple[int, ...]], int]:
    """Map every box of a section bake to the section's row in that bake.

    Parameters
    ----------
    payload
        A chunk file written by `chunk.py --section-chunks`, one chunk per section
        in document order.
    name
        The document file name, for error messages.

    Returns
    -------
    dict
        `{(page, box): row index within this document's section vectors}`, with
        the box as a tuple of four whole points, exactly as both chunkers write it.

    Raises
    ------
    click.ClickException
        If the file was not written by `--section-chunks`, because pairing by
        position would then attach every section score to the wrong rows.

    Notes
    -----
    Chunks and sections are two packings of the same rows, and neither file
    records row indices; what both record is each row's box, rounded the same
    way by `Chunk.boxes_by_page`. A box therefore identifies a row, and the
    section holding most of a chunk's boxes is the section the chunk is in,
    which is the same majority rule `primary_page` uses for pages. A row split at
    a sentence end yields two rows with one box, and both sit in one section, so
    the collision is harmless.
    """
    if not payload.get("params", {}).get("section_chunks"):
        raise click.ClickException(
            f"{name}: {payload.get('file')} was not chunked with --section-chunks. "
            "--section-vectors needs a bake of that mode."
        )
    rows: dict[tuple[int, tuple[int, ...]], int] = {}
    for row, chunk in enumerate(payload["chunks"]):
        for page, boxes in chunk["boxes"].items():
            for box in boxes:
                rows[(int(page), tuple(box))] = row
    return rows


def section_of(chunk: dict[str, Any], rows: dict[tuple[int, tuple[int, ...]], int]) -> int | None:
    """The section row holding most of this chunk's boxes, or None for no match.

    Parameters
    ----------
    chunk
        One entry of a chunk file's `chunks` list.
    rows
        What `section_rows` returned for the same document.

    Returns
    -------
    int or None
        A row index into the document's section bake. None when not one box is
        known to any section, which is the orphan case the writer counts.

    Notes
    -----
    Majority rather than first box, because the overlap carry starts a chunk
    with the tail of the previous one, and that tail can belong to the previous
    section: the first box then votes for the wrong section, the other twenty
    for the right one.
    """
    votes: dict[int, int] = {}
    for page, boxes in (chunk.get("boxes") or {}).items():
        for box in boxes:
            row = rows.get((int(page), tuple(box)))
            if row is not None:
                votes[row] = votes.get(row, 0) + 1
    if not votes:
        return None
    return max(votes, key=lambda row: (votes[row], -row))


def check_weights(page: float, section: float, prev_page: float) -> None:
    """Refuse a blend whose context terms leave nothing for the passage itself.

    Parameters
    ----------
    page, section, prev_page
        The three context weights. The chunk's own vector takes the remainder.

    Raises
    ------
    click.ClickException
        If any is negative or they do not sum to strictly less than 1.
    """
    if min(page, section, prev_page) < 0 or page + section + prev_page >= 1:
        raise click.ClickException(
            f"--page-weight {page} + --section-weight {section} + --prev-page-weight "
            f"{prev_page} must be non-negative and sum below 1: the passage's own "
            "vector takes the rest, and it has to keep some."
        )


# How dense the citation markers have to be before a chunk counts as a reference
# list, per thousand characters. A guideline's prose cites too, so a threshold of 1
# would flag half the corpus; measured over this corpus 8 separates a bibliography
# page from a paragraph that carries two or three citations. The markers are counted
# rather than matched as a whole, because a reference list has no fixed shape: some
# are numbered, some are alphabetical, some carry DOIs and some do not.
REF_MARKERS_PER_KCHAR = 8.0

# The same idea from the other end: a chunk that OPENS with the word is one whatever
# its density says, which catches the first chunk of a bibliography where the heading
# takes up most of the text.
# The headings are per document language (corpus.toml, [languages.<code>]
# reference_headings), merged into one pattern.
REF_HEADING = re.compile(r"^\s*(?:" + "|".join(
    heading for language in corpus_config.section("languages").values()
    for heading in language.get("reference_headings", ())) + ")", re.IGNORECASE)

# "et al", a DOI, a PubMed id or a bare URL: the things that appear in a citation and
# almost nowhere else.
REF_MARKER = re.compile(r"\bet\s+al\b|\bdoi\b|\bPMID\b|https?://", re.IGNORECASE)

# A year in the shape a citation writes it, "(2011)" or "2011;" or "2011.", rather
# than any four digits: a dosage table is full of numbers and none of them end a
# citation. The trailing punctuation is what separates the two.
REF_YEAR = re.compile(r"\(?(?:19|20)\d{2}[a-z]?[);.,]")

# At least this many chunks must look like references before an axis is built from
# them. An axis averaged over three chunks is those three chunks, not a direction,
# and subtracting it would punish whatever they happen to be about.
REF_MIN_EXEMPLARS = 50


def looks_like_references(text: str) -> bool:
    """Whether one chunk reads as a citation list rather than as prose.

    Used only to pick the exemplars the reference axis is averaged from, never to
    hide anything: a false positive here shifts the axis slightly, and a false
    negative costs one exemplar out of thousands. That is why the test is cheap and
    errs towards precision, and why the axis needs `REF_MIN_EXEMPLARS` of them.

    Parameters
    ----------
    text
        The chunk's text.

    Returns
    -------
    bool
        True when the chunk opens as a bibliography or carries citation markers at
        `REF_MARKERS_PER_KCHAR` per thousand characters.
    """
    if not text:
        return False
    if REF_HEADING.match(text):
        return True
    markers = len(REF_MARKER.findall(text)) + len(REF_YEAR.findall(text))
    return markers / (len(text) / 1000.0) >= REF_MARKERS_PER_KCHAR


def reference_axis(floats: np.ndarray, flags: np.ndarray) -> np.ndarray:
    """One unit vector pointing at "this text is a citation list".

    Mined from the corpus rather than written by hand, because a DESCRIPTION of a
    reference list does not embed anywhere near one. Measured with the live encoder
    on 2026-09-21: "Bibliographic references in documentation" retrieves
    methodology sections ("Recherche documentaire") at cosine 0.36 to 0.40, while an
    actual citation string retrieves actual bibliographies at 0.40 to 0.52. The
    model has no direction for the idea of a reference, only for the look of one, so
    the axis has to be built from examples of the thing itself.

    Averaging is what turns examples into a direction: what the exemplars share is
    the citation shape, and what they differ in (the topic each one cites about)
    averages towards nothing.

    Parameters
    ----------
    floats
        Narrowed, unit-norm chunk vectors, one row per chunk.
    flags
        Boolean mask of the chunks that `looks_like_references` accepted.

    Returns
    -------
    np.ndarray
        A unit vector of `floats.shape[1]` dimensions.

    Raises
    ------
    click.ClickException
        If too few exemplars were found, or if they average to nothing.
    """
    if int(flags.sum()) < REF_MIN_EXEMPLARS:
        raise click.ClickException(
            f"only {int(flags.sum())} chunks look like reference lists, fewer than "
            f"{REF_MIN_EXEMPLARS}. An axis averaged over that few is those chunks "
            "rather than a direction; pass --no-reference-axis to ship without one."
        )
    # The CONTRAST, not the average. Averaging the exemplars alone gives a vector
    # that still points mostly at the corpus centroid, because every chunk here is
    # one domain written in French or English and that shared direction does not
    # cancel among the exemplars. Measured on 2026-09-21: the plain average scored
    # 31,089 of 44,850 chunks above half of its own maximum, which is not a
    # discriminator, it is a description of the corpus.
    #
    # Subtracting the mean of EVERY chunk removes what the exemplars have in common
    # with everything else and leaves what makes them different, which is the
    # citation shape. This is a contrast direction, and it is the same construction
    # DESIGN.md rejected for the index itself ("centring the corpus fills every bit
    # and buys nothing"): centring a whole index is a change of coordinates that
    # moves every distance equally, while centring to build ONE axis is the whole
    # point of the axis.
    axis = floats[flags].mean(axis=0) - floats.mean(axis=0)
    norm = float(np.linalg.norm(axis))
    if norm <= 0:
        raise click.ClickException(
            "the reference exemplars have the same mean as the corpus, so there is "
            "no direction that separates them."
        )
    return axis / norm


def narrow_and_quantise(vectors: np.ndarray, dims: int) -> np.ndarray:
    """Truncate to `dims`, restore unit norm, quantise at the fixed wire scale.

    Parameters
    ----------
    vectors
        Row-wise float vectors as stored by `embed.py`, unit-norm at full width.
    dims
        Target width. Must not exceed the stored width.

    Returns
    -------
    numpy.ndarray
        int8, same row count, `dims` columns.

    Raises
    ------
    click.ClickException
        If `dims` exceeds what was baked.

    Notes
    -----
    Truncate-then-renormalise is exactly equivalent to normalising the truncated
    raw vector, because the stored normalisation is multiplication by a positive
    scalar. That identity is why one full-width bake can serve any width.
    """
    if dims > vectors.shape[1]:
        raise click.ClickException(
            f"asked for {dims} dims but the bake stored {vectors.shape[1]}. "
            "Re-bake with embed.py, or lower --dims."
        )
    cut = vectors[:, :dims].astype(np.float32)
    norms = np.linalg.norm(cut, axis=1, keepdims=True)
    # A zero-norm row cannot happen (chunk.py rejects empty chunks and every
    # vector is unit-norm at full width), but dividing by zero here would poison
    # the whole file silently, so refuse rather than emit NaNs.
    if not np.all(norms > 0):
        raise click.ClickException("a stored vector has zero norm; the bake is corrupt.")
    cut /= norms
    return np.clip(np.rint(cut * INT8_SCALE), -INT8_SCALE, INT8_SCALE).astype(np.int8)


def narrow_and_binarise(vectors: np.ndarray, dims: int) -> np.ndarray:
    """Truncate to `dims` and keep one BIT per dimension, packed eight to a byte.

    Parameters
    ----------
    vectors
        Row-wise float vectors as stored by `embed.py`, unit-norm at full width.
    dims
        Target width. Must not exceed the stored width, and must be a multiple of
        8 so a vector occupies a whole number of bytes.

    Returns
    -------
    numpy.ndarray
        uint8, same row count, `dims / 8` columns. Bit set means the component was
        positive. Most significant bit first, so byte 0's high bit is dimension 0,
        which is `numpy.packbits`'s default and what the client's table assumes.

    Raises
    ------
    click.ClickException
        If `dims` exceeds what was baked, or is not a multiple of 8.

    Notes
    -----
    No renormalisation and no threshold tuning. The sign of each component is the
    whole representation: the implied vector is +/-1 in every dimension, with norm
    sqrt(dims), and the client divides by that to recover a cosine. A per-vector
    threshold (the component median, say) would store a second number per vector
    and break the one property that makes this cheap, which is that every passage
    shares one implied norm.

    Truncate-then-take-signs is exactly truncate-then-normalise-then-take-signs,
    since the normalisation is multiplication by a positive scalar. Which is why
    this can skip the renormalisation that the int8 path needs.

    A component of exactly 0.0 is stored as negative. It does not happen with real
    embeddings, and the alternative (a third state) does not exist at one bit.
    """
    if dims > vectors.shape[1]:
        raise click.ClickException(
            f"asked for {dims} dims but the bake stored {vectors.shape[1]}. "
            "Re-bake with embed.py, or lower --dims."
        )
    if dims % 8:
        raise click.ClickException(
            f"--dims {dims} is not a multiple of 8, so a packed vector would "
            "straddle a byte boundary. Use --quant int8 for such a width."
        )
    cut = vectors[:, :dims]
    return np.packbits(cut > 0, axis=1)


def write_hashed(directory: Path, stem: str, suffix: str, payload: bytes) -> str:
    """Write `payload` under a content-addressed name and remove older siblings.

    Parameters
    ----------
    directory
        The index directory.
    stem
        The name before the hash, e.g. ``"vectors"``.
    suffix
        Extension including the dot, e.g. ``".b1"``. It encodes the LAYOUT, so a
        file of another layout is never a sibling of this one.
    payload
        The bytes to write.

    Returns
    -------
    str
        The file name written, for `meta.json` to point at.

    Notes
    -----
    16 hex characters of blake2b. Not a security boundary: this only has to make
    an accidental collision impossible in practice, and 64 bits does that for one
    file per build.

    Pruning matters more than it looks. `deploy.sh` rsyncs with `--delete`, so the
    VPS never accumulates, but the local `dist/` would, and every stale 3.5 MB
    vector file would be shipped and then deleted on the next deploy. The prune is
    deliberately narrow (same stem, same suffix, hashed name) so it can never
    touch anything else that lives in the index directory.

    Unhashed legacy names (`vectors.i8`, `chunks.u16`, from format 1) are removed
    too: a client still asking for one of those is reading an index this build did
    not write.
    """
    digest = hashlib.blake2b(payload, digest_size=8).hexdigest()
    name = f"{stem}-{digest}{suffix}"
    (directory / name).write_bytes(payload)
    pattern = re.compile(rf"^{re.escape(stem)}-[0-9a-f]{{16}}{re.escape(suffix)}$")
    for path in directory.iterdir():
        if path.name == name or not path.is_file():
            continue
        if pattern.match(path.name) or path.name == f"{stem}{suffix}":
            path.unlink()
            logger.info(f"pruned {path.name}, superseded by {name}")
    return name


def prune_doc_dir(doc_dir: Path, written: int) -> int:
    """Delete the per-document files this build did not write.

    Parameters
    ----------
    doc_dir
        `<out>/index/doc/`, holding one `<id>.json` and one `<id>.json.gz` per
        document.
    written
        How many documents this build wrote, so the ids it owns are `0` to
        `written - 1`.

    Returns
    -------
    int
        How many files were removed.

    Notes
    -----
    Document ids are POSITIONAL: they are the index into `meta.json`'s document
    list, so a build over fewer chunk files than the last one leaves the tail
    behind. Nothing else removes them. `write_hashed` prunes the superseded vector
    files and `stage.py` prunes `dist/pdf/`, which is the same job on the other two
    halves of the tree; this is the missing third.

    It matters beyond disk. `meta.json` stops naming those ids, so nothing in the
    UI can reach them, but Caddy still serves them at a guessable path and
    `deploy.sh` still ships them. A document removed from `data/GUIDELINES/` would
    therefore keep its full chunk text online after it was withdrawn, which is the
    failure `stage.py`'s own prune-first comment exists to prevent. `--limit 60`
    against a 535-document corpus leaves 475 of them.
    """
    removed = 0
    keep = {f"{i}.json" for i in range(written)}
    for stale in sorted(doc_dir.iterdir()):
        # Both the plain file and the .gz sibling `write_gzipped` writes beside it.
        if stale.name.removesuffix(".gz") in keep:
            continue
        stale.unlink()
        removed += 1
    if removed:
        logger.info(f"pruned {removed} file(s) from {doc_dir}: documents the index "
                    "no longer has, left by a build over more chunk files than this one")
    return removed


def write_gzipped(path: Path, payload: str) -> None:
    """Write a text file plus the `.gz` sibling Caddy serves precompressed.

    Parameters
    ----------
    path
        Destination for the plain text.
    payload
        The text to write.

    Notes
    -----
    `docker/Caddyfile` enables `precompressed br gzip` on the file server, so a
    `.gz` sibling is served with zero runtime CPU to any client that asks for it.
    Only gzip is written: brotli would be another 15% or so but needs a non-stdlib
    dependency, and Caddy falls back to gzip cleanly. mtime is zeroed so rebuilding
    an unchanged file produces identical bytes and rsync skips it.
    """
    path.write_text(payload, encoding="utf-8")
    raw = payload.encode("utf-8")
    with gzip.GzipFile(path.with_suffix(path.suffix + ".gz"), "wb", mtime=0) as fh:
        fh.write(raw)


def check_bake_source(bake: Any, payload: dict, npz_path: Path) -> None:
    """Refuse a bake computed from other chunks than the ones it is paired with.

    Parameters
    ----------
    bake
        The loaded `.npz` of one document.
    payload
        The chunk file it is about to be paired with, position by position.
    npz_path
        Where `bake` came from, for the message.

    Raises
    ------
    click.ClickException
        When the bake records no chunk hash, or a different one. No hash means a
        file written before `embed.py` recorded it (2026-10-03): it cannot be
        checked, and `embed.py` calls it stale for that reason, so an ordinary
        bake (deploy.sh's included) replaces it. When the recorded hash differs. Counting rows is not enough: re-chunking
        with a fix that keeps the chunk count (a dehyphenation, a table read
        differently) leaves every count equal, and the index would ship new text
        beside vectors of the old text with nothing to say so.
    """
    if "chunks_hash" not in bake.files:
        raise click.ClickException(
            f"{payload['file']}: {npz_path} records no chunks_hash, so nothing proves "
            "which chunks its vectors were computed from. It predates 2026-10-03: "
            "re-run embed.py on this bake, which treats such a file as stale.")
    if str(bake["chunks_hash"]) != payload["src_hash"]:
        raise click.ClickException(
            f"{payload['file']}: {npz_path} was baked from other chunks than the ones "
            "on disk now. Re-run embed.py on this bake (embed.py --check lists every "
            "stale document) before building the index.")


def public_meta(row: dict[str, str], name: str) -> dict[str, str]:
    """The part of one manifest row `meta.json` may publish.

    Parameters
    ----------
    row
        The document's `MANIFEST.tsv` row, filename column removed.
    name
        The filename, for the message.

    Returns
    -------
    dict[str, str]
        The `PUBLIC_COLUMNS` cells, in the manifest's own column order.

    Raises
    ------
    click.ClickException
        When the row has a column that is in neither `PUBLIC_COLUMNS` nor
        `PRIVATE_COLUMNS`. See `PUBLIC_COLUMNS` for why that is a refusal.
    """
    unknown = row.keys() - PUBLIC_COLUMNS - PRIVATE_COLUMNS
    if unknown:
        raise click.ClickException(
            f"{name}: manifest column(s) {sorted(unknown)} are neither public nor "
            "private. Add each to PUBLIC_COLUMNS or PRIVATE_COLUMNS in build_index.py: "
            "meta.json is downloaded by every visitor.")
    return {column: value for column, value in row.items() if column in PUBLIC_COLUMNS}


def quantise(vectors: np.ndarray, dims: int, quant: str) -> np.ndarray:
    """One bake's vectors in the index's storage format, at its width.

    Every half (passages, pages, sections) goes through this one function, because
    the client scores all of them with one table: a half stored differently would
    need a second scoring path.
    """
    return (narrow_and_binarise(vectors, dims) if quant == "binary"
            else narrow_and_quantise(vectors, dims))


def read_bake(chunks_path: Path, npz_path: Path, *, name: str,
              kind: str, flag: str) -> tuple[dict[str, Any], Any]:
    """Load one document's chunk file and vectors for a page or section half.

    Parameters
    ----------
    chunks_path, npz_path
        The half's chunk file and its `.npz`.
    name
        The PDF filename, for messages.
    kind
        "page" or "section", for messages.
    flag
        The option that turns this half off, so the message offers the way out.

    Returns
    -------
    tuple
        The parsed chunk file and the loaded `.npz`.
    """
    if not (chunks_path.exists() and npz_path.exists()):
        raise click.ClickException(
            f"{name}: no {kind} bake ({chunks_path} or {npz_path} missing). Rebuild "
            f"the {kind} chunks and their vectors, or pass {flag} 0.")
    return json.loads(chunks_path.read_text(encoding="utf-8")), np.load(npz_path)


@dataclass(frozen=True)
class Inputs:
    """What every document is read against: the bakes, and how to store them."""

    vectors: Path
    page_chunks: Path
    page_vectors: Path
    section_chunks: Path
    section_vectors: Path
    with_pages: bool
    with_sections: bool
    with_references: bool
    quant: str
    dims: int


@dataclass
class Corpus:
    """Everything `load_document` accumulates, in chunk order, for the writers.

    Attributes
    ----------
    documents
        One entry per document, in index order.
    blocks, page_blocks, section_blocks
        Quantised vectors per document, concatenated by `write_matrices`.
    ref_floats, ref_flags
        Narrowed float vectors and the reference flags mined alongside them. Empty
        and unused when --no-reference-axis.
    locations
        One `(doc, page, page row, section row)` per chunk.
    variants
        Every bake variant met; more than one is refused.
    page_rows, section_rows
        Rows written so far in the page and section halves, which is the offset the
        next document's rows start at.
    orphans
        Chunks whose boxes match no section.
    """

    documents: list[Document] = field(default_factory=list)
    blocks: list[np.ndarray] = field(default_factory=list)
    page_blocks: list[np.ndarray] = field(default_factory=list)
    section_blocks: list[np.ndarray] = field(default_factory=list)
    ref_floats: list[np.ndarray] = field(default_factory=list)
    ref_flags: list[bool] = field(default_factory=list)
    locations: list[tuple[int, int, int, int]] = field(default_factory=list)
    variants: set[str] = field(default_factory=set)
    page_rows: int = 0
    section_rows: int = 0
    orphans: int = 0

    @property
    def n_chunks(self) -> int:
        """Chunks loaded so far, which is the next document's chunk offset."""
        return len(self.locations)


def load_document(doc_id: int, path: Path, *, inputs: Inputs,
                  metadata: dict[str, dict[str, str]], manifest: Path,
                  doc_dir: Path, corpus: Corpus) -> None:
    """Add one document's vectors and locations to `corpus`, and write its doc file.

    Parameters
    ----------
    doc_id
        The document's position in the index.
    path
        Its passage chunk file.
    inputs
        The bakes and the storage format.
    metadata
        The manifest, keyed by filename.
    manifest
        Where `metadata` came from, for the message.
    doc_dir
        Where `<doc_id>.json` goes.
    corpus
        The accumulator, mutated.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    name = payload["file"]
    npz_path = inputs.vectors / f"{path.stem}.npz"
    if not npz_path.exists():
        raise click.ClickException(
            f"{name} has chunks but no vectors ({npz_path} is missing). An index "
            "with a hole in it makes that document unfindable while the site looks "
            "healthy. Finish the bake, or point --vectors at a complete one."
        )
    bake = np.load(npz_path)
    doc_chunks = payload["chunks"]
    if bake["vectors"].shape[0] != len(doc_chunks):
        raise click.ClickException(
            f"{name}: {bake['vectors'].shape[0]} vectors against "
            f"{len(doc_chunks)} chunks. One of the two is stale, and pairing them "
            "by position would attach every vector to the wrong page."
        )
    check_bake_source(bake, payload, npz_path)
    corpus.variants.add(str(bake["variant"]))
    corpus.blocks.append(quantise(bake["vectors"], inputs.dims, inputs.quant))

    # The reference axis is measured on the FLOAT vectors, before quantisation.
    # At one bit per dimension a cosine survives only to about two decimals, and
    # the axis is an average over thousands of rows where that error does not
    # cancel: it is the difference between a usable score and a noisy one. The
    # cost is holding the narrowed floats for the whole corpus, 44,850 x 1024 at
    # four bytes is 184 MB, which is a fraction of what the bake this reads was
    # written with.
    if inputs.with_references:
        cut = bake["vectors"][:, :inputs.dims].astype(np.float32)
        cut /= np.linalg.norm(cut, axis=1, keepdims=True)
        corpus.ref_floats.append(cut)
        corpus.ref_flags.extend(looks_like_references(c.get("text", "")) for c in doc_chunks)

    # The page half, quantised exactly like the passage half: the client
    # scores both with one table, so a page vector in another format would
    # need a second scoring path for a 15% term.
    rows: dict[int, int] = {}
    if inputs.with_pages:
        page_npz = inputs.page_vectors / f"{path.stem}.npz"
        page_payload, page_bake = read_bake(
            inputs.page_chunks / f"{path.stem}.json", page_npz,
            name=name, kind="page", flag="--page-weight")
        rows = page_rows(page_payload, name)
        if page_bake["vectors"].shape[0] != len(rows):
            raise click.ClickException(
                f"{name}: {page_bake['vectors'].shape[0]} page vectors against "
                f"{len(rows)} pages. One of the two is stale."
            )
        check_bake_source(page_bake, page_payload, page_npz)
        corpus.variants.add(str(page_bake["variant"]))
        corpus.page_blocks.append(quantise(page_bake["vectors"], inputs.dims, inputs.quant))

    # The section half, same treatment: one file, one format, one table.
    boxes_to_section: dict[tuple[int, tuple[int, ...]], int] = {}
    n_sections = 0
    if inputs.with_sections:
        section_npz = inputs.section_vectors / f"{path.stem}.npz"
        section_payload, section_bake = read_bake(
            inputs.section_chunks / f"{path.stem}.json", section_npz,
            name=name, kind="section", flag="--section-weight")
        n_sections = len(section_payload["chunks"])
        if section_bake["vectors"].shape[0] != n_sections:
            raise click.ClickException(
                f"{name}: {section_bake['vectors'].shape[0]} section vectors "
                f"against {n_sections} sections. One of the two is stale."
            )
        check_bake_source(section_bake, section_payload, section_npz)
        corpus.variants.add(str(section_bake["variant"]))
        boxes_to_section = section_rows(section_payload, name)
        corpus.section_blocks.append(
            quantise(section_bake["vectors"], inputs.dims, inputs.quant))

    if name not in metadata:
        raise click.ClickException(
            f"{name} has no row in {manifest}. Run manifest.py so the UI's "
            "filters have something to filter on."
        )
    corpus.documents.append(Document(
        file=name,
        meta=public_meta(metadata[name], name),
        chunk_offset=corpus.n_chunks,
        chunk_count=len(doc_chunks),
        figure_count=trailing_figures(doc_chunks, name),
    ))

    for chunk in doc_chunks:
        page = primary_page(chunk)
        # A page with boxes but no packable text (a full-page figure, a scan
        # the OCR gave nothing for) has no page vector. Those chunks are
        # scored on their own vector alone rather than dropped.
        local = rows.get(page)
        section = section_of(chunk, boxes_to_section) if inputs.with_sections else None
        if inputs.with_sections and section is None:
            corpus.orphans += 1
        corpus.locations.append((doc_id, page,
                                 NO_PAGE_ROW if local is None
                                 else corpus.page_rows + local,
                                 NO_SECTION_ROW if section is None
                                 else corpus.section_rows + section))
    corpus.page_rows += len(rows)
    corpus.section_rows += n_sections

    # The lazy half. Only what the viewer cannot derive: the text to show as a
    # snippet, every page the chunk touches and every box to draw on them.
    write_gzipped(doc_dir / f"{doc_id}.json", json.dumps({
        "file": name,
        "chunks": [
            {"i": c["i"], "pages": c["pages"], "boxes": c["boxes"], "text": c["text"],
             # A model's description of a figure, not the document's words: the
             # page says so beside it (lib/figure_record.py).
             **({"figure": c["figure"]} if c.get("figure") else {})}
            for c in doc_chunks
        ],
    }, ensure_ascii=False, separators=(",", ":")))


def write_matrix(index_dir: Path, stem: str, blocks: list[np.ndarray], *,
                 quant: str, max_rows: int | None = None) -> tuple[str, np.ndarray | None]:
    """Write one vector half under a content-addressed name, or none at all.

    Parameters
    ----------
    index_dir
        The index directory.
    stem
        "vectors", "pages" or "sections".
    blocks
        Per-document quantised vectors. Empty means this half is off.
    quant
        Picks the suffix, which carries the layout (see `write_hashed`).
    max_rows
        The largest row count a chunk's uint16 slot can point at, for the halves
        a chunk row indexes into.

    Returns
    -------
    tuple[str, numpy.ndarray | None]
        The file name for `meta.json` ("" when off) and the matrix written.

    Notes
    -----
    Every `<stem>-*` file other than the one written is removed. That covers the
    other --quant's suffix, which `write_hashed` deliberately leaves alone, and every
    file left when a half is turned off: an orphan 1 MB file would otherwise ship
    forever, since meta.json stops naming it.
    """
    name, matrix = "", None
    if blocks:
        matrix = np.concatenate(blocks)
        if max_rows is not None and matrix.shape[0] > max_rows:
            raise click.ClickException(
                f"{matrix.shape[0]} {stem} rows do not fit a uint16 row index. The "
                "chunk row layout would have to widen, which is a format version.")
        name = write_hashed(index_dir, stem, ".b1" if quant == "binary" else ".i8",
                            matrix.tobytes())
    for stale in index_dir.glob(f"{stem}-*"):
        if stale.name != name:
            stale.unlink()
            logger.info(f"pruned {stale.name}, superseded")
    return name, matrix


def write_reference_scores(index_dir: Path, corpus: Corpus) -> tuple[str, float, int]:
    """Write the per-chunk reference score, one byte per chunk.

    Written as a separate file rather than into meta.json, because meta.json is
    fetched by every visitor including browse.html and 45 KB of scores is 45 KB
    nobody listing documents needs.

    Returns
    -------
    tuple[str, float, int]
        The file name, the scale a byte is divided by, and the exemplar count.
    """
    ref_matrix = np.concatenate(corpus.ref_floats)
    flags = np.array(corpus.ref_flags, dtype=bool)
    if ref_matrix.shape[0] != flags.shape[0]:
        raise click.ClickException(
            f"{ref_matrix.shape[0]} vectors against {flags.shape[0]} reference "
            "flags. The two are built in the same loop, so this is a bug here "
            "rather than a stale input."
        )
    axis = reference_axis(ref_matrix, flags)
    n_ref_exemplars = int(flags.sum())
    # Cosine against the axis, one per chunk. Negative values are clipped to
    # zero: a chunk pointing AWAY from the citation direction is not more prose
    # than prose, and letting it score below zero would turn the penalty into a
    # bonus for whatever happens to be most unlike a bibliography.
    scores = np.clip(ref_matrix @ axis, 0.0, 1.0)
    # uint8 over the range actually used. A cosine here runs to about 0.55, so
    # spreading 0..1 over 256 levels would waste half of them; the scale is
    # stored in meta.json so the reader divides by the same number rather than
    # assuming one, which is what lets this range change without a format bump.
    ref_scale = float(scores.max()) or 1.0
    quantised = np.rint(scores / ref_scale * 255.0).astype(np.uint8)
    refs_file = write_hashed(index_dir, "refs", ".u8", quantised.tobytes())
    logger.info(
        f"reference axis from {n_ref_exemplars} exemplar chunks of "
        f"{ref_matrix.shape[0]}; scores up to {ref_scale:.3f}, "
        f"{int((scores > ref_scale * 0.5).sum())} chunks above half scale"
    )
    corpus.ref_floats.clear()
    return refs_file, ref_scale, n_ref_exemplars


@click.command()
@click.option("--chunks", type=click.Path(exists=True, file_okay=False, path_type=Path),
              **in_corpus("data/chunks"),
              help="Chunk files from chunk.py. Must be the SAME set the vectors "
                   "were baked from, which is checked per document.")
@click.option("--vectors", type=click.Path(exists=True, file_okay=False, path_type=Path),
              **in_corpus("data/vectors/meta-gpu"),
              help="A per-variant bake directory from embed.py.")
@click.option("--manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              **in_corpus("data/MANIFEST.tsv"),
              help="Document metadata, embedded into meta.json for the filters.")
@click.option("--out", type=click.Path(path_type=Path), **in_corpus("dist"), help="Tree deploy.sh ships. Writes <out>/index/.")
@click.option("--dims", default=1024, show_default=True,
              help="Width to truncate to. MUST equal docker/.env's SEARCH_DIM: a "
                   "mismatch does not degrade ranking, it destroys it. It no "
                   "longer has to equal the shared encoder's EMBED_OUT_DIM, since "
                   "src/search.js asks for this width per request.")
@click.option("--quant", type=click.Choice(["binary", "int8"]), default="binary",
              show_default=True,
              help="Stored precision. binary is one bit per dimension and is what "
                   "makes 1024 dims affordable; int8 is eight times larger for no "
                   "measured gain at this width. See the module docstring.")
@click.option("--page-chunks", type=click.Path(file_okay=False, path_type=Path),
              **in_corpus("data/chunks-page"),
              help="Chunk files from `chunk.py --page-chunks --out <dir>`, one "
                   "chunk per page. Paired with --page-vectors.")
@click.option("--page-vectors", type=click.Path(file_okay=False, path_type=Path),
              **in_corpus("data/vectors/page-meta-gpu"),
              help="A bake of --page-chunks, from `embed.py --chunks <that dir> "
                   "--out-name <this dir's name>`.")
@click.option("--page-weight", default=0.15, show_default=True,
              help="Share of a passage's score that comes from its whole page. "
                   "Measured, not chosen: see the module docstring. 0 ships no "
                   "page file at all, which is the pre-0.1.0 index minus 1 MB.")
@click.option("--section-chunks", type=click.Path(file_okay=False, path_type=Path),
              **in_corpus("data/chunks-section"),
              help="Chunk files from `chunk.py --section-chunks --out <dir>`, one "
                   "chunk per section. Paired with --section-vectors.")
@click.option("--section-vectors", type=click.Path(file_okay=False, path_type=Path),
              **in_corpus("data/vectors/section-meta-gpu"),
              help="A bake of --section-chunks, from `embed.py --chunks <that dir> "
                   "--out-name <this dir's name>`.")
@click.option("--section-weight", default=0.0, show_default=True,
              help="Share of a passage's score that comes from the section it sits "
                   "in. 0 ships no section file. Measured in DESIGN.md.")
@click.option("--prev-page-weight", default=0.0, show_default=True,
              help="Share of a passage's score that comes from the page BEFORE its "
                   "own. Needs the page bake and no file of its own: the client "
                   "finds the previous page's row from the locations. Measured in "
                   "DESIGN.md.")
@click.option("--reference-axis/--no-reference-axis", "with_references",
              default=True, show_default=True,
              help="Score every chunk for how much it reads as a list of "
                   "bibliographic references, so the search can offer to push "
                   "citation pages down. The axis is mined from the corpus rather "
                   "than written by hand: see reference_axis. One byte per chunk, "
                   "in its own file, and the weight is chosen per query.")
@click.option("--limit", default=0, show_default=True,
              help="Only the first N documents. For checking the writer cheaply; "
                   "the result is a deliberately incomplete index and says so.")
def main(chunks: Path, vectors: Path, manifest: Path, out: Path, dims: int,
         quant: str, page_chunks: Path, page_vectors: Path, page_weight: float,
         section_chunks: Path, section_vectors: Path, section_weight: float,
         prev_page_weight: float, with_references: bool,
         limit: int) -> None:
    """Build `<out>/index/` from a bake."""
    check_weights(page_weight, section_weight, prev_page_weight)
    metadata = read_manifest(manifest)
    policy = manifest_policy.policy()
    check_vocabulary(metadata, policy)
    chunk_files = sorted(chunks.glob("*.json"))
    if limit:
        chunk_files = chunk_files[:limit]
    if not chunk_files:
        raise click.ClickException(f"no chunk files in {chunks}; run chunk.py first.")

    # Refuse up front rather than per document: a missing page bake would
    # otherwise be discovered 60 documents in, after the writer has already
    # replaced half of dist/index/.
    with_pages = page_weight > 0
    if with_pages and not (page_chunks.is_dir() and page_vectors.is_dir()):
        raise click.ClickException(
            f"--page-weight {page_weight} needs {page_chunks} and {page_vectors}. "
            f"Build them with `uv run scripts/chunk.py --page-chunks --out "
            f"{page_chunks}` then `uv run scripts/embed.py --chunks {page_chunks} "
            f"--variant title --out-name {page_vectors.name} --weights "
            "model.onnx --gpu`, or pass --page-weight 0 to ship without them."
        )

    with_sections = section_weight > 0
    if with_sections and not (section_chunks.is_dir() and section_vectors.is_dir()):
        raise click.ClickException(
            f"--section-weight {section_weight} needs {section_chunks} and "
            f"{section_vectors}. Build them with `uv run scripts/chunk.py "
            f"--section-chunks --out {section_chunks}` then `uv run scripts/embed.py "
            f"--chunks {section_chunks} --out-name {section_vectors.name} --weights "
            "model.onnx --gpu`, or pass --section-weight 0 to ship without them."
        )

    index_dir = out / "index"
    doc_dir = index_dir / "doc"
    doc_dir.mkdir(parents=True, exist_ok=True)

    inputs = Inputs(vectors=vectors, page_chunks=page_chunks, page_vectors=page_vectors,
                    section_chunks=section_chunks, section_vectors=section_vectors,
                    with_pages=with_pages, with_sections=with_sections,
                    with_references=with_references, quant=quant, dims=dims)
    corpus = Corpus()
    for doc_id, path in enumerate(chunk_files):
        load_document(doc_id, path, inputs=inputs, metadata=metadata,
                      manifest=manifest, doc_dir=doc_dir, corpus=corpus)

    if len(corpus.variants) != 1:
        raise click.ClickException(
            f"the bake mixes variants {sorted(corpus.variants)}. Passages embedded from "
            "different text cannot share one index: re-bake with a single --variant."
        )

    prune_doc_dir(doc_dir, len(corpus.documents))

    if with_sections and corpus.orphans > corpus.n_chunks * ORPHAN_SHARE:
        raise click.ClickException(
            f"{corpus.orphans} of {corpus.n_chunks} chunks match no section by their "
            "boxes. The section bake was chunked from different text than the passage "
            "bake: re-run chunk.py for both from the same corpus and code."
        )

    refs_file, ref_scale, n_ref_exemplars = "", 1.0, 0
    if with_references:
        refs_file, ref_scale, n_ref_exemplars = write_reference_scores(index_dir, corpus)
    # Same rule as write_matrix: a build without the axis leaves no refs file behind
    # for meta.json to stop naming while deploy.sh keeps shipping it.
    for stale in index_dir.glob("refs-*"):
        if stale.name != refs_file:
            stale.unlink()
            logger.info(f"pruned {stale.name}, superseded")

    # The suffix carries the layout, so a client that fetched one can never be
    # handed the other by a half-finished deploy: the two files have the same
    # row count and different strides, and misreading one as the other ranks
    # nonsense rather than failing.
    vectors_file, matrix = write_matrix(index_dir, "vectors", corpus.blocks, quant=quant)
    # Little-endian explicitly: the client reads this with a DataView, and the
    # default on the writing machine is not a contract.
    chunks_file = write_hashed(index_dir, "chunks", ".u16",
                               np.asarray(corpus.locations, dtype="<u2").tobytes())
    pages_file, page_matrix = write_matrix(index_dir, "pages", corpus.page_blocks,
                                           quant=quant, max_rows=NO_PAGE_ROW)
    sections_file, section_matrix = write_matrix(index_dir, "sections",
                                                 corpus.section_blocks, quant=quant,
                                                 max_rows=NO_SECTION_ROW)
    n_pages = 0 if page_matrix is None else int(page_matrix.shape[0])
    n_section_rows = 0 if section_matrix is None else int(section_matrix.shape[0])

    write_gzipped(index_dir / "meta.json", json.dumps({
        "format_version": INDEX_FORMAT_VERSION,
        "dims": dims,
        "quant": "binary" if quant == "binary" else f"int8/fixed{INT8_SCALE}",
        # Named here rather than hardcoded client-side, which is what lets the
        # layout change, and the content hash change on every rebuild, without a
        # matching edit in src/search.js. meta.json is the ONLY name the client
        # knows, and the only one docker/Caddyfile makes it revalidate.
        "vectors_file": vectors_file,
        # Empty when the index ships no page half. The client reads the weight
        # rather than assuming one, so an index built with --page-weight 0 and a
        # client that knows about pages agree without a second format version.
        "pages_file": pages_file,
        "n_pages": n_pages,
        "page_weight": page_weight if with_pages else 0.0,
        # The two terms added by format 4, read the same way. The previous-page
        # term has no file: it scores the page bake at the row of (doc, page - 1),
        # which the client finds from the locations, so it is only meaningful
        # when the index ships pages.
        "sections_file": sections_file,
        "n_sections": n_section_rows,
        "section_weight": section_weight if with_sections else 0.0,
        "prev_page_weight": prev_page_weight if with_pages else 0.0,
        "chunks_file": chunks_file,
        # Format 5. Empty when the index ships no reference axis, and the client
        # reads the name rather than assuming one, exactly as the page and section
        # halves do: an index built with --no-reference-axis and a client that
        # knows about the term agree without a second format version.
        "refs_file": refs_file,
        "ref_scale": ref_scale,
        "n_ref_exemplars": n_ref_exemplars,
        "bytes_per_vector": int(matrix.shape[1] * matrix.dtype.itemsize),
        "variant": next(iter(corpus.variants)),
        "bake": vectors.name,
        "page_bake": page_vectors.name if with_pages else "",
        "section_bake": section_vectors.name if with_sections else "",
        "n_chunks": int(matrix.shape[0]),
        "n_documents": len(corpus.documents),
        # Marked when --limit was used, so a partial index cannot be deployed by
        # accident and then puzzled over.
        "partial": bool(limit),
        # The collapse policy, carried from manifest.py so the client never holds a
        # second copy of it. See lib/manifest_policy.py.
        "renditions": policy["renditions"],
        # What separates the slugs inside a doc_type or topic cell. Carried for the
        # same reason: src/search.js splits on this rather than on a literal of its
        # own, so changing it in manifest.py changes it everywhere.
        "separator": policy["separator"],
        # The guideline tiers, NARROWEST FIRST. The level control in src/app.js
        # offers these and src/search.js keeps every document at the chosen tier or
        # narrower, both by position in this list. An index built before this key
        # existed simply has no list, and the control then filters nothing rather
        # than hiding a corpus it cannot rank.
        "guideline_tiers": policy["guideline_tiers"],
        # Which manifest column holds that tier, which columns the page offers as
        # filters (in panel order) and which of them as a numeric range. The page and
        # the search service build their filters from these three rather than from
        # lists of their own, so a corpus declares its facets once, in corpus.toml.
        # An index built before they existed offers no facets, and its tier is read
        # from `guideline`, the one column that held it then.
        "tier_field": policy["tier_field"],
        "facet_fields": policy["facet_fields"],
        "range_fields": policy["range_fields"],
        "documents": [
            {"id": i, "file": d.file, "chunk_offset": d.chunk_offset,
             "chunk_count": d.chunk_count,
             **({"figure_count": d.figure_count} if d.figure_count else {}), **d.meta}
            for i, d in enumerate(corpus.documents)
        ],
    }, ensure_ascii=False, separators=(",", ":")))

    vector_bytes = sum(m.nbytes for m in (matrix, page_matrix, section_matrix)
                       if m is not None)
    if with_sections:
        logger.info(f"{n_section_rows} sections, {corpus.orphans} chunks matching none "
                    f"({100 * corpus.orphans / max(corpus.n_chunks, 1):.2f}%)")
    logger.success(
        f"{len(corpus.documents)} documents, {matrix.shape[0]} chunks at {dims} dims "
        f"({quant}) -> {index_dir}"
    )
    logger.info(
        f"eager download: {vector_bytes / 1e6:.1f} MB vectors "
        f"({n_pages} of those rows whole pages, {n_section_rows} whole sections) + "
        f"{corpus.n_chunks * 2 * LOC_STRIDE / 1e3:.0f} KB locations + "
        f"{(index_dir / 'meta.json').stat().st_size / 1e3:.0f} KB meta"
    )
    if limit:
        logger.warning(f"--limit {limit}: this index is INCOMPLETE and marked partial")

if __name__ == "__main__":
    main()
