# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "loguru", "pymupdf==1.28.2", "pymupdf-layout", "tokenizers"]
# ///
"""Split the guideline corpus into embeddable chunks with page and bbox provenance.

Every chunk records exactly where on which page its text came from, because
the search UI renders the source PDF page with the matching passage
highlighted. A chunk without boxes cannot be highlighted, so it is a build
failure rather than a degraded result.

Pipeline
--------
1. Line-level extraction. pymupdf's "dict" output gives one bbox per line,
   which is the natural highlight unit: it is what a text selection looks
   like. Block-level boxes would over-highlight whole paragraphs and
   character-level boxes would multiply the payload for no visible gain.
2. Column detection and row merging. A page's lines are first split into the
   flows they belong to: one flow on an ordinary page, and one per column plus
   the full-width lines between them on a two-column page. Within a flow,
   lines overlapping vertically are merged into one row, which is how bulleted
   text arrives (a glyph at x=57 and the sentence at x=75). Across flows
   nothing merges, so the end of a left-column line is never welded to the
   middle of a right-column one.
3. Boilerplate removal. Agency documents repeat a running header on nearly every
   page ("Qualité de l'eau potable : recommandations ... / Décembre 2017")
   plus a bare page number. Left in, that text is embedded hundreds of times
   and matches every query about the document's title. Lines are treated as
   boilerplate when their digit-normalised form recurs on a large share of
   pages AND they sit in the top or bottom margin.
4. Packing. Rows are accumulated in reading order up to a token budget,
   measured with the ACTUAL tokenizer of the embedding model rather than a
   characters-per-token guess. Rows are never split, so every chunk boundary
   falls on a line boundary and every box stays exact. Once the budget is
   reached the chunk does not end there: it rewinds to the last structural
   break it passed (paragraph gap, bullet, heading, page change, end of
   sentence) as long as that leaves it at least two thirds full. This is the
   --boundaries default, and it is measured rather than assumed: DESIGN.md
   records what it is worth.
5. Overlap. Trailing rows are carried into the next chunk so that a fact
   straddling a boundary is retrievable from both sides.

Chunks are written one JSON file per document, gated on a content hash of the
PDF plus the chunking parameters plus a format version, so re-running after
adding one document re-chunks only that document.

Where the code lives
--------------------
This file is the entry point: the CLI, the content hash and its parameters
(`ChunkSettings`, `chunk_params`, `source_hash`), `chunk_document`, which runs
the pipeline above on one PDF, and the worker pool. The stages themselves are
in `lib/`, each importing only the ones below it:

- `lib/chunk_row.py`: `Row`, the unit every stage passes along, and `median`.
- `lib/chunk_text.py`: text repair (ligatures, false spaces, hyphen breaks),
  language and sentence detection.
- `lib/chunk_glyphs.py`: characters to words and lines, sideways pages.
- `lib/chunk_tables.py`: table detection and serialisation, the layout cache.
- `lib/chunk_extract.py`: steps 1 to 3, a page to rows in reading order.
- `lib/chunk_pack.py`: steps 4 and 5, rows to chunks.

The names tests know as `chunk.X` are re-exported here.

Written with Claude Code.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import click
import pymupdf
from loguru import logger
from tokenizers import Tokenizer

from lib import atomic, hashing
from lib import figure_record
from lib.chunk_row import Row
from lib.chunk_text import dehyphenate, resolve_glue, resolve_ligatures, split_sentences
from lib.chunk_glyphs import upright_rect
from lib.chunk_tables import LAYOUT_CACHE, LayoutGrids, layout_version
from lib.chunk_extract import boilerplate_keys, extract_rows, normalise_key
from lib.chunk_pack import (
    Chunk, HEADING_MAX_TOKENS, line_break_flags, pack, pack_pages, pack_sections,
)
# Re-exported, not used here: tests and other callers know these names as chunk.X
# (tests/conftest.py loads this file by path), and they kept that spelling when the
# code moved to lib/chunk_*.py.
from lib.chunk_text import (  # noqa: F401
    BULLET_START, _join_inside, clean_text, ends_sentence, glue_for, looks_french,
)
from lib.chunk_glyphs import (  # noqa: F401
    continues_word, line_text, reread_symbols, split_words, symbol_glyph, word_text,
)
from lib.chunk_tables import (  # noqa: F401
    Detected, TableCarry, contents_listing, grid_cells, is_tabular, prose_columns,
    resolve_headers, runs_across_rows, serialise_table, table_names,
)
from lib.chunk_extract import (  # noqa: F401
    _merge, _same_row, clip_to_page, flow_margins, margin_for, reading_groups,
)
from lib.chunk_pack import (  # noqa: F401
    _carry_over, break_strengths, heading_context, heading_flags, list_cuts,
)
from lib.corpus import corpus_path, in_corpus


# Bumping this invalidates every cached chunk file. Change it whenever the
# meaning of the output changes, not merely when this file is edited.
# 3: words split by a line break are rejoined, and a chunk that cannot end at a
#    block boundary ends at a sentence end rather than wherever the budget ran out.
# 4: layout is read before text. A two-column page is read one column at a time
#    and nothing merges across the gutter; sideways text is merged across the
#    direction it runs in rather than down the page, so a rotated annex table
#    stops collapsing into one row; and a running header or footer is measured
#    against its own page's height, so a landscape page in a portrait document
#    has margins at all.
# 5: a row is split where a sentence ends inside it, so a sentence end is always
#    available at a row boundary; a combining accent stranded after a space is
#    reattached to its letter; and a chunk no longer stores the same box twice.
# 6: a cut has to leave a FINISHED sentence behind. The layout signals (a bullet,
#    a wider line gap, a heading, a page change) say where a block starts, not
#    whether the line before it ended anything, and cutting on them was what left
#    chunks ending in "les resultats sont". Cuts are now ranked: a block boundary
#    with a finished sentence behind it, then any finished sentence, then reading
#    forward to the next one, then a short but finished chunk, and only then a cut
#    that truncates. A combining mark opening a line is a Symbol-font bullet and is
#    written as one, and a dotless i or j carrying a mark gets its dot back so the
#    accent composes.
# 7: a detected table is read out cell by cell, with its column names, instead of
#    being flattened line by line across its columns; and a word broken by a line
#    break INSIDE one row (a serialised cell, or two lines merged into one row) is
#    rejoined like one broken between two rows.
# 9: a table continued at the top of the next page inherits the column names of the
#    fragment it continues. The detector sees two tables, and the second has no
#    header because the names were printed once, a page earlier; serialised alone it
#    came out as "Escitalopram ; 10 ; 5-20." with nothing saying which number was the
#    starting dose. The same change stops the continuation's first row being eaten as
#    a header by the first-row guess in table_names. See apply_tables.
# 10: the chunk file records each page's size. verify_chunks.py has always claimed
#    to check that every highlight box lies inside its page, and could not: the file
#    said nothing about how big a page is. A box outside the page is a highlight the
#    reader never sees, on a hit the search counted.
# 11: packing prefers a section boundary over a paragraph one, and the target
#    grew to 1024 tokens with 256 of overlap. Both change where every cut falls.
# 12: a word broken by a line break is rejoined with the WHOLE token that
#    finishes it, not just the punctuation glued to it, so "adher-" + "ence.92"
#    and "atten-" + "tion-deficit" stop reaching the reader in halves.
# 13: chunk text keeps its paragraph breaks. A row that starts a block is joined
#    with a newline instead of a space, so a list of recommendations still reads
#    as a list on the result page.
# 14: chunk text also keeps the line breaks the AUTHOR made. A line that stops
#    short of its column's right margin, with room for the next line's first word,
#    ended on purpose, so it is joined with a newline too: a table of drug names
#    reached the reader as one long line of drug names (`line_break_flags`).
# 15: tables the ruled-line detector cannot see are read too, from the grid the
#    layout model (pymupdf-layout) predicts, and a header may leave its stub column
#    blank. A guideline's dosing tables, ruled only under their headers, came out as nine
#    columns read in a zigzag: one page had its doses, frequencies and comments
#    interleaved line by line. See LAYOUT_CACHE.
# 16: a page whose text runs up or down it is read in that direction's order
#    (`reading_blocks`), so a sideways table's title precedes the table and its
#    footnotes come in order, where pymupdf's unrotated sort put "TABLE 8" after it.
# 17: a soft hyphen followed by a space (a table cell's own line break) is a
#    broken word too, and is rejoined: "infrastruc ture" became "infrastructure".
# 18: text is read glyph by glyph (rawdict), and a space the page shows no gap for
#    is nominated as false and decided by the document's vocabulary: "ar e
#    needed", "proﬁ les", and two touching lines "reinit" + "iate" read as the
#    words they are, "of Action" stays two. See GLUE.
# 19: a table's column names are said once, in its caption, instead of before
#    every cell. See serialise_table.
# 20: a layout-model grid is refused when a paragraph runs on across the rows where
#    its first column starts new entries: a prose column beside the table, or rows
#    cut through the table's own cells. See runs_across_rows.
# 21: a spacing accent drawn on a letter becomes that letter's mark ("Clı´nica" reads
#    "Clínica", see attach_accents), a control code is a ligature inside a word and
#    a bullet outside one (see CONTROL), and zero-width spaces and U+FFFD are
#    dropped (see INVISIBLE).
# 22: a glyph a dingbat font drew is a bullet whatever character code it came
#    with, so Wingdings3's arrow stops reading as an acute accent. See symbol_glyph.
#
# Note that entry 11 mixed a format change with a parameter one, which was a
# mistake worth not repeating: the target size and the overlap are hashed into
# every document's `src_hash` by `source_hash`, so changing them invalidates the
# cache on their own and needs no version here. The target moved to 256 tokens
# with 26 of overlap on 2026-09-26 (DESIGN.md, "the chunk is 256 tokens, not
# 1024") and this number did not move with it, on purpose.
# 23: the fill window is 170 to 512 tokens rather than 171 to 307, with no read
#    past the target (DESIGN.md, "Decided: a 170 to 512 token window"). The
#    floors are fractions of the target and NOT hashed, so moving them has to
#    move this number or the cache keeps the old chunks.
# 24: a bulleted list is split only when no cut outside it is in reach, and an
#    item only when no cut between items is. See list_cuts.
CHUNK_FORMAT_VERSION = 24


def load_tokenizer(path: Path) -> Tokenizer:
    """Load the embedding model's tokenizer with padding disabled.

    Disabling padding is not cosmetic. `encode_batch` pads every sequence to
    the longest one in the batch, so `len(encoding.ids)` reports the padded
    length and every row in a batch measures the same. Budgeting a chunk with
    those numbers packs by "rows until the longest row's length is exceeded"
    instead of by real tokens: the first version of this script did exactly
    that and produced 161377 chunks from 7910 pages, roughly twelve times too
    many, with every chunk reporting an identical token count. That identical
    count is the symptom to watch for if this ever regresses.

    Parameters
    ----------
    path : Path
        Path to the model's tokenizer.json.

    Returns
    -------
    Tokenizer
        Tokenizer that reports true, unpadded token counts.
    """
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_padding()
    tokenizer.no_truncation()  # a long row must report its real length, not a cap
    return tokenizer


@dataclass(frozen=True)
class ChunkSettings:
    """The chunking policy, as one value instead of eight keyword arguments.

    It travels from `main` through `Job` and `run_job` into `chunk_document`, and
    `params` turns it into what the content hash is taken over. Frozen and made of
    plain fields because a `Job` carrying it is pickled into a worker process.

    Attributes
    ----------
    target, overlap : int
        Packing budget and overlap, in tokens. Ignored when `page_chunks` or
        `section_chunks` is set, since neither packs to a budget.
    boundaries : bool
        End chunks at detected paragraph, list, heading and page breaks.
    sections : bool
        With `boundaries`, prefer a section boundary (a heading or a page change)
        over a paragraph one, rewinding further to reach it. False is the
        paragraph-preferred policy; see `pack`.
    headings : int
        Longest heading run, in tokens, to prefix to a chunk that continues a
        section. 0 carries nothing, which is what `--no-headings` gives.
    page_chunks : bool
        One chunk per page instead of packing by budget.
    section_chunks : bool
        One chunk per section, split only at the encoder's window.
    tables : bool
        Read detected tables out cell by cell. See `apply_tables`.
    overlap_boundaries : bool
        Carry the overlap across boundary cuts too. See `pack`.
    """

    target: int
    overlap: int
    boundaries: bool = False
    sections: bool = True
    headings: int = 0
    page_chunks: bool = False
    section_chunks: bool = False
    tables: bool = False
    overlap_boundaries: bool = False

    def params(self, *, tokenizer_name: str, table_layout: str | None) -> dict[str, object]:
        """The parameters a document's content hash is taken over.

        Parameters
        ----------
        tokenizer_name : str
            Directory name of the tokenizer, standing for the model whose tokens
            the budget is counted in.
        table_layout : str or None
            The pymupdf-layout version reading the tables, or None.

        Returns
        -------
        dict
            Exactly what `chunk_params` returns for the same policy, because it
            is `chunk_params`: there is one implementation.

        Examples
        --------
        >>> ChunkSettings(target=256, overlap=26, boundaries=True).params(
        ...     tokenizer_name="jina", table_layout=None)
        {'tokenizer': 'jina', 'target_tokens': 256, 'overlap_tokens': 26, 'boundaries': True}
        """
        return chunk_params(
            target_tokens=self.target, overlap_tokens=self.overlap,
            tokenizer_name=tokenizer_name, boundaries=self.boundaries,
            sections=self.sections, heading_tokens=self.headings,
            page_chunks=self.page_chunks, section_chunks=self.section_chunks,
            tables=self.tables, table_layout=table_layout,
            overlap_boundaries=self.overlap_boundaries)


def chunk_params(
    *, target_tokens: int, overlap_tokens: int, tokenizer_name: str,
    boundaries: bool, sections: bool, heading_tokens: int,
    page_chunks: bool, section_chunks: bool, tables: bool,
    table_layout: str | None = None, overlap_boundaries: bool = False
) -> dict[str, object]:
    """The parameters a document's content hash is taken over.

    Everything that changes a chunk's TEXT belongs here, and nothing else does.
    A parameter left out is one whose change silently reuses the previous run's
    chunks; a parameter wrongly included throws away a bake that is still valid.

    Two conventions hold the second half of that up. A flag is recorded only
    when it is on, which is why flipping `--boundaries`' default did not
    invalidate the measurement bakes: the boundary chunks written before the
    flip already carry ``{"boundaries": true}`` and still match, and
    `--no-boundaries` still hashes exactly as the greedy chunks that shipped
    first did. And a parameter the chosen packer does not READ is left out
    entirely: `pack_pages` and `pack_sections` take neither the target nor the
    overlap, so recording them there would tie 21,268 page vectors to a
    passage-chunking decision that cannot reach them. That is not hypothetical:
    the 2026-09-26 move from 1024 tokens to 256 invalidated the whole page bake
    before this function existed, for a number no page chunk consulted.

    Parameters
    ----------
    target_tokens, overlap_tokens : int
        The packing budget and the overlap. Omitted from the result under
        `page_chunks` or `section_chunks`, which do not pack to a budget.
    tokenizer_name : str
        Directory name of the tokenizer, standing for the model whose tokens
        the budget is counted in.
    boundaries, sections, heading_tokens, page_chunks, section_chunks, tables
        The chunking policy, as `chunk_document` takes it.
    table_layout : str, optional
        The pymupdf-layout version reading the tables, because a new model moves
        grids and so changes text without changing anything else here.

    Returns
    -------
    dict
        JSON-serialisable, and hashed with its keys sorted by `source_hash`.

    Examples
    --------
    >>> passage = chunk_params(
    ...     target_tokens=256, overlap_tokens=26, tokenizer_name="jina",
    ...     boundaries=True, sections=True, heading_tokens=0,
    ...     page_chunks=False, section_chunks=False, tables=True)
    >>> sorted(passage)
    ['boundaries', 'overlap_tokens', 'tables', 'target_tokens', 'tokenizer']
    >>> pages = chunk_params(
    ...     target_tokens=256, overlap_tokens=26, tokenizer_name="jina",
    ...     boundaries=True, sections=True, heading_tokens=0,
    ...     page_chunks=True, section_chunks=False, tables=True)
    >>> sorted(pages)
    ['boundaries', 'page_chunks', 'tables', 'tokenizer']
    """
    params: dict[str, object] = {"tokenizer": tokenizer_name}
    if not (page_chunks or section_chunks):
        params["target_tokens"] = target_tokens
        params["overlap_tokens"] = overlap_tokens
    if boundaries:
        params["boundaries"] = True
    if boundaries and not sections and not (page_chunks or section_chunks):
        # Recorded only when OFF, the mirror of --boundaries just above: the default
        # has to hash exactly as every chunk written before this flag existed, or the
        # shipped bake goes stale for a policy change that did not happen. Gated on
        # `boundaries` because with the ladder gone the rung order means nothing, and
        # two runs that differ only in a parameter neither reads must share a cache,
        # which is the same reason it is gated on the packer being the one that packs
        # to a budget: the rung order is `pack`'s, and the other two never see it.
        params["section_first"] = False
    if overlap_boundaries and overlap_tokens and not (page_chunks or section_chunks):
        # Recorded only when ON, so the default hashes exactly as every chunk written
        # before the flag existed; gated on a non-zero overlap and on the budget
        # packer for the same reason as section_first: with nothing to carry, or a
        # packer that never reads it, the text is the same and so must the cache be.
        params["overlap_boundaries"] = True
    if heading_tokens:
        # The VALUE, not a flag: the cap changes the text, so two runs with different
        # caps must not share a cache entry.
        params["headings"] = heading_tokens
    if page_chunks:
        params["page_chunks"] = True
    if section_chunks:
        params["section_chunks"] = True
    if tables:
        params["tables"] = True
    if tables and table_layout:
        params["table_layout"] = table_layout
    return params


def source_hash(pdf: Path, params: dict[str, object]) -> str:
    """Content hash gating a document's cached chunks.

    Covers the PDF bytes, the chunking parameters and the format version, so
    that changing the target size or the tokenizer invalidates the cache just
    as surely as editing the PDF does.

    Parameters
    ----------
    pdf : Path
        The source PDF.
    params : dict
        Chunking parameters to bind into the hash.

    Returns
    -------
    str
        Hex digest.
    """
    digest = hashlib.sha256()
    digest.update(pdf.read_bytes())
    digest.update(json.dumps(params, sort_keys=True).encode())
    digest.update(str(CHUNK_FORMAT_VERSION).encode())
    return digest.hexdigest()


def chunk_document(
    pdf: Path, tokenizer: Tokenizer, settings: ChunkSettings
) -> tuple[list[Chunk], dict[str, int]]:
    """Chunk one PDF.

    Parameters
    ----------
    pdf : Path
        Source PDF.
    tokenizer : Tokenizer
        Embedding model tokenizer.
    settings : ChunkSettings
        The chunking policy. `target` and `overlap` are ignored when
        `page_chunks` is set, since a page is whatever size it is.
        `page_chunks` is a variant for measuring page-level retrieval, not a
        shipping configuration. `section_chunks` is one chunk per section (a
        heading run and what sits under it), split only at the encoder's
        window: the bake behind the section term of the blend, and the literal
        "a chunk is a section" experiment; see `pack_sections`. `tables` reads
        detected tables out cell by cell instead of leaving their lines to be
        extracted left to right. See `apply_tables`. Includes the tables the
        layout model finds, whose grids are cached under LAYOUT_CACHE.

    Returns
    -------
    tuple
        The chunks, and per-document statistics for the build report.
    """
    layout = LayoutGrids(pdf, corpus_path(LAYOUT_CACHE)) if settings.tables else None
    with pymupdf.open(pdf) as doc:
        try:
            table_boxes: list = []
            pages, n_tables, n_clipped, n_off_page = extract_rows(
                doc, tables=settings.tables, layout=layout, table_boxes=table_boxes)
        finally:
            # In `finally` so a document that fails halfway still keeps the grids it
            # paid for: they are the slow part, and the retry reuses them.
            if layout is not None:
                layout.close()
        # Upright, because every box these are compared against is: see
        # `upright_rect`. The recorded sizes are what verify_chunks.py checks each
        # box against and what src/viewer.js draws inside.
        rects = [upright_rect(page) for page in doc]
        heights = [rect.height for rect in rects]
        sizes = [[int(round(rect.width)), int(round(rect.height))] for rect in rects]
    dropped = boilerplate_keys(pages, heights)
    kept: list[Row] = []
    n_dropped = 0
    for rows in pages:
        for row in rows:
            if normalise_key(row.text) in dropped:
                n_dropped += 1
                continue
            kept.append(row)
    # Before packing, so that both bakes (chunk and page) read the same repaired
    # text, and before the tokenizer ever sees a row.
    # First, so a word the producer split with a false space is whole before
    # dehyphenate looks at its ends.
    resolve_ligatures(kept)
    resolve_glue(kept)
    n_joined = dehyphenate(kept)
    # After dehyphenation, so a word rejoined across two lines is whole before any
    # sentence end is looked for, and before the tokenizer sees a row.
    kept, n_split = split_sentences(kept)
    # Here rather than in `pack`, because all three packers put rows in a chunk and
    # all three want a table to read as a table. It is geometry, not packing: unlike
    # `Row.starts_block` it says nothing about where a chunk may END, so it is not
    # tied to --boundaries.
    for row, starts in zip(kept, line_break_flags(kept)):
        row.starts_line = starts
    if settings.page_chunks:
        chunks = pack_pages(kept, tokenizer)
    elif settings.section_chunks:
        chunks = pack_sections(kept, tokenizer)
    else:
        chunks = pack(kept, tokenizer, target=settings.target, overlap=settings.overlap,
                      boundaries=settings.boundaries, sections=settings.sections,
                      headings=settings.headings,
                      overlap_boundaries=settings.overlap_boundaries)
    stats = {
        # Not a statistic: carried out through this dict because that is what
        # `chunk_document` returns, and lifted to the top level by `run_job`.
        # One [width, height] per page, in whole points like every box, so that
        # `verify_chunks.py` can tell a box that lies outside its page.
        "page_sizes": sizes,
        # Not a statistic either: where each serialised table sits, lifted to the top
        # level by `run_job` for `figures.py tables` to crop.
        "table_boxes": table_boxes,
        "pages": len(pages),
        "rows": len(kept),
        "boilerplate_rows_dropped": n_dropped,
        "words_rejoined": n_joined,
        "rows_split_at_sentences": n_split,
        "tables_serialised": n_tables,
        "boxes_clipped": n_clipped,
        "rows_off_page": n_off_page,
        "chunks": len(chunks),
    }
    return chunks, stats


# One tokenizer per worker process, loaded by the pool initializer rather than sent
# with every job: a Tokenizer is a Rust handle, and shipping it through the queue for
# each of 484 documents would cost more than the chunking does. A module global is how
# a pool initializer is allowed to leave something behind for the jobs that follow.
_WORKER_TOKENIZER: Tokenizer | None = None


def _init_worker(tokenizer_path: Path) -> None:
    """Load this worker process's tokenizer once, at startup.

    Parameters
    ----------
    tokenizer_path : Path
        tokenizer.json of the embedding model.
    """
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = load_tokenizer(tokenizer_path)


@dataclass
class Job:
    """One document to chunk, and everything needed to write its file.

    Carries paths and parameters rather than objects, because it is pickled into a
    worker process. The chunks themselves are never sent back: a worker writes its
    own JSON and returns counts, so a 2530-chunk document does not travel through a
    queue for the parent to write out again.
    """

    pdf_name: str
    served: Path
    destination: Path
    digest: str
    params: dict
    settings: ChunkSettings
    from_ocr: bool
    # Described figures, appended after the text's chunks (lib/figure_record.py).
    figures: tuple = ()


def run_job(job: Job, tokenizer: Tokenizer | None = None) -> dict:
    """Chunk one document and write its JSON file.

    Parameters
    ----------
    job : Job
        What to chunk and where to put it.
    tokenizer : Tokenizer, optional
        The tokenizer to use. Defaults to this worker process's own, which is what
        the pool path passes nothing for; the serial path passes the parent's.

    Returns
    -------
    dict
        `file`, `empty` and the document's `stats`, which is all the parent needs
        to keep its counters and its warnings.
    """
    encoder = tokenizer if tokenizer is not None else _WORKER_TOKENIZER
    assert encoder is not None, "run_job called without a tokenizer"
    chunks, stats = chunk_document(job.served, encoder, job.settings)
    page_sizes = stats.pop("page_sizes")
    table_boxes = stats.pop("table_boxes")
    stats["figures"] = len(job.figures)
    # Atomic: the next run trusts any chunk file whose src_hash matches, so a
    # half-written one must never sit at this path (lib/atomic.py).
    atomic.write_text_atomic(job.destination, json.dumps({
        "file": job.pdf_name,
        # Which file the boxes actually describe, and therefore which file
        # must be served. "ocr" means data/OCR/<file> is authoritative.
        "source": "ocr" if job.from_ocr else "original",
        "src_hash": job.digest,
        "format_version": CHUNK_FORMAT_VERSION,
        "params": job.params,
        # One [width, height] per page of the file the boxes describe, in the same
        # whole points. Lifted out of `stats` because it is a property of the
        # document, not a count of what chunking did to it.
        "page_sizes": page_sizes,
        "table_boxes": table_boxes,
        "stats": stats,
        "chunks": [
            {"i": i, "text": c.text, "n_tokens": c.n_tokens,
             "pages": sorted({r.page + 1 for r in c.rows}),
             "boxes": c.boxes_by_page()}
            for i, c in enumerate(chunks)
        ] + figure_record.chunk_entries(job.served, list(job.figures), encoder,
                                        start=len(chunks)),
    }, ensure_ascii=False), encoding="utf-8")
    return {"file": job.pdf_name, "empty": not chunks, "stats": stats}


def worker_count(requested: int) -> int:
    """How many processes to chunk with.

    Parameters
    ----------
    requested : int
        What the user asked for, or 0 for the automatic choice.

    Returns
    -------
    int
        At least 1.

    Notes
    -----
    Measured on a 12-core machine: extraction and dehyphenation are 87% of the time
    and both hold the GIL, so threads gave 1.01x and processes 1.9x. The automatic
    value leaves two cores alone, because this runs on a workstation that is doing
    other things, and stops at 8 because the corpus has a handful of documents so
    much larger than the rest that they, not the worker count, set the wall clock.
    """
    if requested > 0:
        return requested
    return max(1, min(8, (os.cpu_count() or 2) - 2))


def read_name_list(path: Path) -> set[str]:
    """Read a list of PDF file names, one per line.

    Blank lines and lines whose first non-space character is `#` are dropped, so
    a generated list can carry a header saying where it came from and still be
    read back without a parser.

    Parameters
    ----------
    path
        A UTF-8 text file. Each remaining line is one file name as it appears in
        the guidelines directory, extension included, no directory part.

    Returns
    -------
    set of str
        The names, stripped of surrounding whitespace.
    """
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.add(line)
    return names


def read_ocr_index(ocr_dir: Path) -> dict[str, dict]:
    """`ocr.py`'s record of which original each OCR copy was made from.

    Parameters
    ----------
    ocr_dir
        `data/OCR/`.

    Returns
    -------
    dict
        Filename to index entry; empty when there is no index (and so, unless
        someone copied PDFs in by hand, no copies either).
    """
    path = ocr_dir / "ocr_index.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def vouched_for(copy: Path, original: Path, ocr_index: dict[str, dict]) -> None:
    """Refuse an OCR copy that was not made from the original now on disk.

    Parameters
    ----------
    copy
        `data/OCR/<name>`, which would be chunked and served instead of the original.
    original
        `data/GUIDELINES/<name>`.
    ocr_index
        Output of :func:`read_ocr_index`.

    Raises
    ------
    click.ClickException
        When the index does not record the original's sha256, or records another.
        Chunking the copy anyway is how a replaced document stays invisible: the
        copy's own hash has not moved, so the chunk cache is "current", and the
        OLD document is chunked, embedded and served. Running `ocr.py` brings the
        copies back in line (it re-OCRs a changed original and removes a copy
        whose original no longer needs one).
    """
    recorded = ocr_index.get(copy.name, {}).get("original_sha256")
    if recorded is None or recorded != hashing.file_sha256(original):
        raise click.ClickException(
            f"{copy} was not made from the {original} now on disk (data/OCR/ocr_index.json "
            f"{'records another original' if recorded else 'has no original_sha256 for it'}). "
            "Run scripts/ocr.py, which re-OCRs or removes it.")


@click.command()
@click.option("--guidelines", type=click.Path(exists=True, file_okay=False, path_type=Path),
              **in_corpus("data/GUIDELINES"),
              help="Root of the servable corpus. Never widen this to data/.")
@click.option("--out", type=click.Path(path_type=Path), **in_corpus("data/chunks"), help="Directory for per-document chunk files.")
@click.option("--ocr", "ocr_dir", type=click.Path(path_type=Path), **in_corpus("data/OCR"),
              help="Directory of OCR'd derivatives, produced by ocr.py. When a "
                   "derivative exists for a document it is chunked INSTEAD of the "
                   "original, because highlight boxes must describe the file the "
                   "site serves.")
@click.option("--tokenizer", "tokenizer_path", type=click.Path(path_type=Path),
              default=Path("../justelesRCP/models/jinaai/jina-embeddings-v5-text-small-retrieval/tokenizer.json"),
              show_default=True,
              help="tokenizer.json of the embedding model, so budgets are exact.")
@click.option("--target-tokens", type=int, default=512, show_default=True,
              help="Maximum tokens per chunk. 512 since 2026-09-30, with a 170-token "
                   "floor, so a section up to 512 stays whole. The chunking sweep "
                   "preferred 256; this is a deliberate choice against it "
                   "(DESIGN.md, 'a 170 to 512 token window').")
@click.option("--overlap-tokens", type=int, default=0, show_default=True,
              help="Tokens of trailing context repeated into the next chunk. 0 since "
                   "2026-09-30: only a cut the budget FORCES ever carried any (a cut "
                   "at a boundary never does), and the page vector covers what such "
                   "a cut splits.")
@click.option("--overlap-at-boundaries", "overlap_boundaries", is_flag=True,
              default=False,
              help="Carry --overlap-tokens across a cut at a boundary too, so the "
                   "chunk after a section break opens on the end of the one before. "
                   "Off by default: it exists for the grid's overlap rows, which "
                   "measure whether that helps (DESIGN.md).")
@click.option("--boundaries/--no-boundaries", default=True, show_default=True,
              help="End chunks at paragraph, list, heading and page breaks once they "
                   "are two thirds full, instead of wherever the token budget runs "
                   "out. On by default because it is measured better (DESIGN.md); "
                   "--no-boundaries reproduces the pre-0.1.0 chunking, and because it "
                   "changes every chunk's text it needs its own --out directory and a "
                   "fresh bake.")
@click.option("--section-first/--no-section-first", "sections", default=True,
              show_default=True,
              help="With --boundaries, rewind past a paragraph boundary to reach a "
                   "SECTION one (a heading or a page change) when the chunk is full. "
                   "--no-section-first ends the chunk at the last block boundary of "
                   "any kind instead, which is the paragraph-preferred policy: "
                   "chunks land closer to the target and a long section is cut at "
                   "its own paragraphs. It changes every chunk's text, so it needs "
                   "its own --out directory and a fresh bake.")
@click.option("--headings/--no-headings", default=False, show_default=True,
              help="Prefix a chunk that continues a section with that section's "
                   "heading, so the second and later chunks of a long section still "
                   "carry the words the section is about. Text only: the heading is "
                   "never added to the chunk's pages or boxes. Off by default until "
                   "it is measured (DESIGN.md), and because it changes every chunk's "
                   "text it needs its own --out directory and a fresh bake.")
@click.option("--heading-tokens", type=int, default=HEADING_MAX_TOKENS,
              show_default=True,
              help="With --headings, the longest heading run to carry, in tokens. A "
                   "run longer than this is dropped whole: past some length the size "
                   "test has caught a paragraph rather than a heading.")
@click.option("--page-chunks", is_flag=True, default=False,
              help="One chunk per page, ignoring --target-tokens. For measuring "
                   "page-level retrieval (DESIGN.md); not a shipping mode.")
@click.option("--section-chunks", is_flag=True, default=False,
              help="One chunk per section, a heading run and everything under it "
                   "up to the next one, ignoring --target-tokens and split only "
                   "at the encoder's window. The bake behind the section term of "
                   "the blend (DESIGN.md); not a shipping mode. Exclusive with "
                   "--page-chunks.")
@click.option("--tables/--no-tables", default=True, show_default=True,
              help="Detect tables and read their cells out with the column names, "
                   "instead of letting the extractor flatten them line by line. On by "
                   "default because it is measured better (DESIGN.md); it costs a "
                   "table scan on every page that draws at least two ruled lines, "
                   "which is most of the build's time, and --no-tables reproduces the "
                   "format 6 text.")
@click.option("--figures/--no-figures", default=True, show_default=True,
              help="Append one chunk per figure described in data/private/figures/ "
                   "(scripts/figures.py), boxed on the figure itself. Passage chunking "
                   "only: page and section bakes never carry them.")
@click.option("--workers", type=int, default=0, show_default=True,
              help="Documents to chunk in parallel, as separate processes. 0 picks "
                   "a value from the core count. 1 keeps everything in this process, "
                   "which is what a profiler or a debugger wants.")
@click.option("--only", "only_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None,
              help="A file listing the PDF names to chunk, one per line, blank "
                   "lines and # comments ignored. Chunking a subset is what makes "
                   "a parameter sweep affordable: scripts/bench.py builds such a "
                   "list from the eval set and runs the whole chain over it. The "
                   "name is NOT part of the cache key, because which documents a "
                   "run covers does not change any one document's chunks.")
@click.option("--force", is_flag=True, help="Re-chunk even when the content hash matches.")
def main(guidelines: Path, out: Path, ocr_dir: Path, tokenizer_path: Path,
         target_tokens: int, overlap_tokens: int, overlap_boundaries: bool,
         boundaries: bool, sections: bool, headings: bool, heading_tokens: int, page_chunks: bool,
         section_chunks: bool, tables: bool, figures: bool, workers: int, force: bool,
         only_path: Path | None) -> None:
    """Chunk every PDF under the guidelines root, with provenance."""
    if page_chunks and section_chunks:
        raise click.ClickException("--page-chunks and --section-chunks are two "
                                   "different bakes; ask for one per --out.")
    if not tokenizer_path.exists():
        raise click.ClickException(
            f"tokenizer not found at {tokenizer_path}. It ships with the embedding "
            "model in the sibling justelesRCP checkout; pass --tokenizer to point elsewhere."
        )
    tokenizer = load_tokenizer(tokenizer_path)
    settings = ChunkSettings(
        target=target_tokens, overlap=overlap_tokens, boundaries=boundaries,
        sections=sections, headings=heading_tokens if headings else 0,
        page_chunks=page_chunks, section_chunks=section_chunks, tables=tables,
        overlap_boundaries=overlap_boundaries,
    )
    params = settings.params(tokenizer_name=tokenizer_path.parent.name,
                             table_layout=layout_version() if tables else None)
    out.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(guidelines.glob("*.pdf"))
    if only_path is not None:
        # A named subset has to fail loudly on a name that is not in the corpus:
        # silently chunking 40 documents when the list asked for 41 would make a
        # sweep compare two runs over different corpora and call the difference a
        # result.
        wanted = read_name_list(only_path)
        missing = sorted(wanted - {pdf.name for pdf in pdfs})
        if missing:
            raise click.ClickException(
                f"{only_path} names {len(missing)} file(s) absent from {guidelines}: "
                + ", ".join(missing[:5]) + ("..." if len(missing) > 5 else "")
            )
        pdfs = [pdf for pdf in pdfs if pdf.name in wanted]
        logger.info(f"restricted to the {len(pdfs)} documents named in {only_path}")
    how = ("one chunk per page" if page_chunks else
           "one chunk per section" if section_chunks else
           f"target {target_tokens} tokens with {overlap_tokens} overlap"
           + ((", ending at structure"
               + ("" if sections else ", paragraph-preferred")) if boundaries else ", greedy")
           + (f", headings up to {heading_tokens} tokens" if headings else ""))
    logger.info(f"{len(pdfs)} PDFs in {guidelines}, {how}")

    cached = ocred = 0
    ocr_index = read_ocr_index(ocr_dir)
    described = (figure_record.described(guidelines)
                 if figures and not (page_chunks or section_chunks) else {})
    totals = {"chunks": 0, "figures": 0, "pages": 0, "boilerplate_rows_dropped": 0,
              "tables_serialised": 0, "boxes_clipped": 0, "rows_off_page": 0}
    empty: list[str] = []
    # The cache check stays here, in one process, and decides the worklist before any
    # worker starts. It is a hash and a small read per document, and keeping it in the
    # parent is what preserves the cheap incremental run: adding one PDF to the corpus
    # sends one job to the pool rather than 484.
    todo: list[Job] = []
    for pdf in pdfs:
        # An OCR'd derivative replaces the original for every purpose here.
        # pdf.js can only highlight where a text layer exists, and the boxes
        # stored below come from whichever file was parsed, so chunking the
        # original while serving the derivative (or the reverse) would draw
        # every highlight at coordinates that mean nothing in the reader's
        # browser. deploy.sh must ship the same file this chunked.
        served = ocr_dir / pdf.name
        if served.exists():
            vouched_for(served, pdf, ocr_index)
            ocred += 1
        else:
            served = pdf

        # Stem rather than name so the JSON does not end in ".pdf.json"; the
        # PDF's real filename is stored inside the file as the identity.
        destination = out / f"{pdf.stem}.json"
        own = tuple(described.get(pdf.name, ()))
        digest = figure_record.fold(source_hash(served, params), list(own))
        if destination.exists() and not force:
            try:
                if json.loads(destination.read_text(encoding="utf-8"))["src_hash"] == digest:
                    cached += 1
                    continue
            except (json.JSONDecodeError, KeyError):
                logger.warning(f"unreadable cache for {pdf.name}, rebuilding")

        todo.append(Job(pdf_name=pdf.name, served=served, destination=destination,
                        digest=digest, params=params, settings=settings,
                        from_ocr=served != pdf, figures=own))

    # A pool costs a fork and a tokenizer load per worker, which is not worth paying
    # to chunk one document: the incremental run after adding a single PDF stays in
    # this process.
    parallel = worker_count(workers)
    if len(todo) > 1 and parallel > 1:
        logger.info(f"{len(todo)} documents to chunk, {min(parallel, len(todo))} at a time")
        with ProcessPoolExecutor(max_workers=min(parallel, len(todo)),
                                 initializer=_init_worker,
                                 initargs=(tokenizer_path,)) as pool:
            # map rather than as_completed: results arrive in corpus order, so the
            # progress line below means what it says and the log of a parallel run
            # reads the same as the log of a serial one.
            results = list(pool.map(run_job, todo))
    else:
        results = [run_job(job, tokenizer=tokenizer) for job in todo]

    for result in results:
        if result["empty"]:
            # A PDF with no text layer produces nothing. The brief requires
            # this to be loud: a silently empty document is a hole in the
            # corpus that no search result can ever reveal.
            empty.append(result["file"])
        for key in totals:
            totals[key] += result["stats"].get(key, 0)
    built = len(results)

    logger.success(f"{built} documents chunked, {cached} unchanged")
    if ocred:
        logger.info(f"{ocred} documents read from {ocr_dir} instead of the original; "
                    f"deploy.sh must serve those same files")
    if built:
        logger.info(f"{totals['chunks']} chunks over {totals['pages']} pages, "
                    f"{totals['boilerplate_rows_dropped']} boilerplate rows dropped"
                    + (f", {totals['tables_serialised']} tables read out cell by cell"
                       if tables else "")
                    + (f", {totals['figures']} described figures" if totals["figures"] else ""))
        # Reported every run, not only when it is large: this is the number that
        # would have gone from 277 to 14165 the day a box stopped being in the
        # page's own coordinates, and the gate cannot see it because clipping is
        # what makes the gate pass. See clip_to_page.
        if totals["boxes_clipped"] or totals["rows_off_page"]:
            logger.info(f"{totals['boxes_clipped']} boxes cut down to their page, "
                        f"{totals['rows_off_page']} rows dropped as drawn entirely "
                        "off it")
    if empty:
        logger.error(f"{len(empty)} documents produced ZERO chunks (no text layer): "
                     f"{', '.join(empty)}")
        logger.error("These need OCR before they can be searched. See DESIGN.md.")
        sys.exit(1)


if __name__ == "__main__":
    main()
