"""Packing rows into chunks.

Given a document's rows in reading order, `pack` fills chunks to a token
budget and ends each at the best structural break in reach (section, paragraph,
list item, sentence), `pack_pages` and `pack_sections` make the page and
section bakes, and `attach_heading` carries a section's heading into a chunk
that continues it. The structure signals they read (`heading_flags`,
`break_strengths`, `line_break_flags`, `list_cuts`) are here too, because they
are only ever computed for packing. Nothing here opens a PDF.

Written by Claude Code (Opus 5.5), split out of chunk.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tokenizers import Tokenizer

from lib.encoder import MAX_PASSAGE_TOKENS
from lib.chunk_row import Row, median
from lib.chunk_text import BULLET_START, SENTENCE_END, SENTENCE_FLOOR, ends_sentence


# --- Structure detection, used only when packing with --boundaries ----------------
#
# A PDF has no paragraphs. It has glyphs at coordinates, and what a reader calls a
# paragraph break is a vertical gap larger than the one between lines of the same
# paragraph. So every threshold below is RELATIVE to the document's own body text:
# an absolute number in points would classify a 9-point annex and a 14-point public
# leaflet by their type size rather than by their structure.

# A gap between consecutive baselines wider than this many times the document's
# median baseline pitch starts a new paragraph. 1.35 rather than something tighter
# because line pitch varies within a paragraph (a line containing a superscript or a
# formula is taller), and a false break costs more than a missed one: it cuts a
# sentence in half, which is exactly what this feature exists to stop.
PARAGRAPH_PITCH = 1.35

# How much room a line must leave at the right margin before that room is read as a
# deliberate line ending rather than as a word that did not fit: the next row's first
# word, its space, and this much again as slack. 1.25 because the width of that word
# is estimated from an AVERAGE character, and a capital W is not an average character.
# A false break here costs a newline inside a wrapped sentence, which reads as a
# ragged passage; a missed one costs a table read as one long line, which is what
# `line_break_flags` exists to stop.
LINE_BREAK_MARGIN = 1.25

# A row whose font is this much larger than the document's body size is treated as a
# heading, and a heading belongs with the text UNDER it, so the break goes before it.
# 1.15 is low enough to catch the common 11pt-body/13pt-heading layout and high
# enough to ignore the half-point differences that font substitution introduces.
HEADING_SIZE = 1.15

# A heading carried into a chunk that does not contain it is context, and context has
# to stay small next to the passage it describes. Anything longer than this is not a
# heading run but a title page, a pull quote, or a paragraph of large type that the
# size test mistook for one, and prefixing it would drown the passage it was meant to
# situate. The run is dropped whole rather than truncated: half a heading is worse
# than none. Swept with --heading-tokens; DESIGN.md records what the sweep found.
HEADING_MAX_TOKENS = 60

# A chunk may close early at a structural boundary once it holds this share of the
# target. Without a floor, a document of short paragraphs produces a chunk per
# paragraph: 40-token vectors that carry no context, which the 200-token sweep
# already showed is the direction that hurts rank 1 (DESIGN.md, "chunk size barely
# matters"). With it, the packer fills to two thirds of the budget and then takes the
# first paragraph break it finds, so chunks end where the text ends rather than
# mid-sentence, while staying the size the measurements like.
#
# 2026-09-30: 0.332 of the 512-token target, so 170 tokens. The window used to be
# 171 to 307 (0.66 and an OVERSHOOT of 0.2 at 256); it is now 170 to 512, so a
# section up to 512 tokens stays whole and a chunk still never drops under 170
# except at the last-resort `SENTENCE_FLOOR` or a document's own tail. Chosen by
# the user against the grid search, which preferred 256 (DESIGN.md).
FILL_FLOOR = 0.332

# How far past the budget a chunk may run to finish the sentence it is in, as a
# share of the target. Rewinding alone leaves a third of all chunks ending in the
# middle of a clause: the last third of a chunk often holds no line that ENDS a
# sentence, because a sentence usually ends in the middle of a line and rows are
# never split. Reading a little further almost always reaches one. 0.2 of a
# 300-token target is 60 tokens, two or three lines, and the ceiling matters more
# than the exact value: the model reads 8192 tokens (config.json says 8194 positions,
# and encoding the same 3208-token passage at 512, 1024, 2048 and 4096 gives four
# different vectors, so nothing truncates silently), which puts 360 far inside what
# it will read, and nothing is stranded the way a rewind can strand it. What limits
# the chunk here is the size sweeps in DESIGN.md, not the encoder.
#
# 2026-09-30: 0. The target is now the MAXIMUM, 512 tokens, and the 342 tokens
# between the floor and it almost always hold a sentence end to rewind to, so
# reading past it would only break the ceiling. Rows are never split, so a single
# row longer than the target (a table row, a reference) still makes a longer
# chunk: the ceiling is on packing, not on one row.
OVERSHOOT = 0.0


@dataclass
class Chunk:
    """A packed run of rows, ready to embed.

    `token_counts` is parallel to `rows`. Keeping it on the chunk rather than
    indexing back into a document-wide list is what makes the overlap carry
    correct: a chunk's rows start at an arbitrary offset in the document, so
    a positional slice of a global list describes different rows entirely.
    """

    rows: list[Row] = field(default_factory=list)
    token_counts: list[int] = field(default_factory=list)
    # The headings this chunk sits under, when they are not among its own rows. A
    # section longer than the budget becomes several chunks, and only the first of
    # them starts at the heading; the rest used to read as anonymous prose, which is
    # exactly what the reader typed the heading's words to find. These rows are
    # prefixed to `text` and counted in `n_tokens`, and they are deliberately NOT
    # added to `rows`: a heading can be pages back, and putting it in `rows` would
    # add its page to the chunk's provenance, move the page the site opens on, and
    # draw a highlight on a page the passage is not on. Context for the encoder,
    # nothing for the viewer.
    heading: list[Row] = field(default_factory=list)
    heading_tokens: list[int] = field(default_factory=list)

    @property
    def n_tokens(self) -> int:
        """Total tokens across the chunk's rows, carried heading included."""
        return sum(self.token_counts) + sum(self.heading_tokens)

    def add(self, row: Row, n_tokens: int) -> None:
        """Append one row and its token count."""
        self.rows.append(row)
        self.token_counts.append(n_tokens)

    def split(self, at: int) -> tuple[Chunk, Chunk]:
        """Cut in two before row `at`, keeping both halves whole.

        Used by boundary packing to hand the rows that belong to the next block
        back to the next chunk, instead of leaving them at the tail of this one.

        Parameters
        ----------
        at : int
            Index of the first row of the second half.

        Returns
        -------
        head, tail : Chunk
            The rows before `at` and the rows from `at` on. No row is dropped or
            duplicated, so every box still quotes the page exactly once.
        """
        head, tail = Chunk(), Chunk()
        head.rows, head.token_counts = self.rows[:at], self.token_counts[:at]
        tail.rows, tail.token_counts = self.rows[at:], self.token_counts[at:]
        return head, tail

    @property
    def text(self) -> str:
        """Chunk text: rows joined, under any carried heading.

        A newline goes in front of a row that STARTS a block (`Row.starts_block`)
        or a LINE the author ended (`Row.starts_line`), and a space in front of
        every other one, so a paragraph is reflowed into one line and a list stays
        a list. Joining everything with spaces lost the shape of a table of
        recommendations, and joining everything with newlines would reproduce the
        PDF's own line wrapping, which is narrower than any screen the passage is
        read on. The two flags are how the difference is told: one asks whether a
        reader would call this a new block, the other whether the previous line
        stopped before the margin with room to spare (`line_break_flags`).

        The heading, when one is carried, always gets a line of its own.
        """
        pieces: list[str] = []
        for row in self.heading:
            pieces.append(row.text if not pieces else " " + row.text)
        for index, row in enumerate(self.rows):
            if not pieces:
                pieces.append(row.text)
                continue
            pieces.append(("\n" if row.starts_block or row.starts_line or index == 0
                           else " ") + row.text)
        return "".join(pieces)

    def boxes_by_page(self) -> dict[str, list[list[int]]]:
        """Group row boxes by 1-based page number.

        Returns
        -------
        dict
            Page number as a string (JSON object keys must be strings) mapped
            to the list of boxes on that page, rounded to whole points.
            Sub-point precision is invisible at any zoom level and doubles the
            payload size.
        """
        grouped: dict[str, list[list[int]]] = {}
        for row in self.rows:
            box = [int(round(v)) for v in row.bbox]
            # Several rows share one box when a line held several sentences (see
            # split_sentences), and drawing the same rectangle twice is at best
            # wasted payload and at worst a double-darkened highlight.
            page = grouped.setdefault(str(row.page + 1), [])
            if box not in page:
                page.append(box)
        return grouped


def heading_flags(rows: list[Row]) -> list[bool]:
    """Say which rows are headings, by type size alone.

    The single definition of "heading" in this file. `break_strengths` uses it to
    decide where a section starts, and `heading_context` uses it to decide what a
    chunk sits under; if the two ever disagreed, a chunk could be cut at a heading
    that it then refused to carry.

    Parameters
    ----------
    rows : list of Row
        All rows of one document, in reading order.

    Returns
    -------
    list of bool
        One flag per row. All False when no row carries a size, which is what a
        PDF with no font information yields.

    Notes
    -----
    The median row size is the BODY size: headings are by definition the minority,
    so the middle of the distribution is the text they head.
    """
    sizes = [row.size for row in rows if row.size > 0]
    body = median(sizes) if sizes else 0.0
    return [bool(body) and row.size > body * HEADING_SIZE for row in rows]


def heading_context(rows: list[Row]) -> list[list[int]]:
    """For each row, the run of heading rows it sits under.

    "Run" rather than "row" because one heading is regularly several rows: a
    numbered heading whose text wraps, or "5. Prise en charge" immediately followed
    by "5.1 Chez l'adulte". Taking the contiguous run gives the whole of it, and
    gives the sub-heading its parent for free when the two are adjacent.

    A heading row's own context is the run it belongs to, so a chunk that starts at
    a heading is recognised as already carrying it and does not repeat it.

    Parameters
    ----------
    rows : list of Row
        All rows of one document, in reading order.

    Returns
    -------
    list of list of int
        One list of row indices per row, empty for rows before the document's
        first heading.
    """
    flags = heading_flags(rows)
    context: list[list[int]] = []
    run: list[int] = []      # The heading run being read right now.
    active: list[int] = []   # The last completed run, in effect for the body below it.
    for index, is_heading in enumerate(flags):
        if is_heading:
            run.append(index)
            # The same list object, so the rows of a run that is still being read
            # end up seeing the whole of it once it closes.
            context.append(run)
        else:
            if run:
                active, run = run, []
            context.append(active)
    return [list(entry) for entry in context]


def line_break_flags(rows: list[Row]) -> list[bool]:
    """Which rows begin a line the author ended, rather than one the margin did.

    A reflowed paragraph reaches the right margin on every line but its last, so a
    line that stops well short of the margin stopped for a reason: it is a table
    row, a list item without a bullet, an address, a heading, the end of a
    paragraph. Joining those with a space is what turned a page of one drug per
    line into one long line of drug names.

    The test is not "is this line short" but "would the next word have fitted on
    it". That is the same question the typesetter answered, and it is what keeps
    ragged-right prose out of this: a ragged line is short exactly because the
    next word did not fit, so it flags nothing, while a table row is short with
    room for the whole of the next row's first word and then some.

    Parameters
    ----------
    rows : list of Row
        All rows of one document, in reading order.

    Returns
    -------
    list of bool
        One flag per row, parallel to `rows`. The first row is always False: what
        it says about the line before it is nothing.

    Notes
    -----
    Rows sharing a bbox are the pieces of one visual line (`split_sentences`), and
    a line cannot have been ended twice: only the last piece of a line is asked
    about. A page change, which is not a line break either, is left to
    `break_strengths`, which scores it 3.
    """
    flags = [False] * len(rows)
    for index in range(1, len(rows)):
        previous, row = rows[index - 1], rows[index]
        if row.page != previous.page or row.vertical or previous.vertical:
            continue
        if row.bbox == previous.bbox:
            continue        # two sentences of one line, not two lines
        room = previous.right_edge - previous.bbox[2]
        if previous.right_edge <= 0 or room <= 0 or not row.text:
            continue
        # The next row's own average character, which is the best estimate of how
        # wide its first word would have been on the line before: the two lines are
        # neighbours in one flow, so they are set in the same type.
        width = (row.bbox[2] - row.bbox[0]) / len(row.text)
        if width <= 0:
            continue
        first = row.text.split(" ", 1)[0]
        # The word plus the space in front of it, and then a margin, because the
        # estimate is an average over proportional type: a line that ends one
        # narrow word short of the margin is a wrap, not a decision.
        needed = (len(first) + 1) * width * LINE_BREAK_MARGIN
        flags[index] = room >= needed
    return flags


def break_strengths(rows: list[Row]) -> list[int]:
    """Score how strongly a new block of text starts at each row.

    The PDF says nothing about paragraphs, so structure is inferred from the
    geometry the rows already carry: where the page changes, where the vertical
    gap widens, where the type gets bigger, and where a list item begins.

    Scores are ordinal, not additive: 3 is "a reader would call this a new
    section", 2 "a new paragraph or list item", 1 "a sentence ended here", 0
    "the same sentence continues". `pack` treats 2 and above as a place a chunk
    may end.

    Parameters
    ----------
    rows : list of Row
        All rows of one document, in reading order.

    Returns
    -------
    list of int
        One score per row, parallel to `rows`. The first row scores 0: nothing
        precedes it, so it cannot be a boundary between two things.

    Notes
    -----
    The median pitch is taken over consecutive rows ON THE SAME PAGE, because a
    page break produces a meaningless negative or huge delta. A document whose
    rows are all on different pages (a slide deck) yields no pitch at all, in
    which case gap detection is skipped and only the page, heading and bullet
    signals remain.
    """
    if not rows:
        return []
    pitches = [
        b.bbox[1] - a.bbox[1]
        for a, b in zip(rows, rows[1:])
        if a.page == b.page and b.bbox[1] - a.bbox[1] > 0
    ]
    pitch = median(pitches) if pitches else 0.0
    headings = heading_flags(rows)

    scores = [0]
    for index, (previous, row) in enumerate(zip(rows, rows[1:]), start=1):
        score = 0
        if SENTENCE_END.search(previous.text):
            score = 1
        if BULLET_START.match(row.text):
            score = 2
        if pitch and row.page == previous.page and row.bbox[1] - previous.bbox[1] > pitch * PARAGRAPH_PITCH:
            score = 2
        if headings[index]:
            score = 3
        if row.page != previous.page:
            score = 3
        scores.append(score)
    return scores


def list_cuts(rows: list[Row], breaks: list[int]) -> list[int]:
    """Where cutting before each row would break up a bulleted list.

    A list is read as one thing: "the criteria are:" followed by five items, of
    which a chunk holding three answers a question about the criteria wrongly. So
    `pack` ranks a cut INSIDE a list below every cut outside one, and a cut inside
    a single item below a cut between two items.

    A list starts at a row `BULLET_START` matches and runs on through that item's
    continuation lines and through every further item, until a row that is not an
    item starts a block of its own (a paragraph gap, a heading, a page change):
    that row is the text after the list, and cutting before it is an ordinary
    boundary. A page change INSIDE a list therefore only counts as one when the
    next page opens on a new item, where it ranks as a cut between items rather
    than as the section boundary a page change otherwise is. A page that opens on
    a row that is NOT an item is read as the end of the list: nothing in the
    geometry tells an item continued overleaf from the paragraph after the list,
    and the unfinished sentence of a continued item already stops a cut there.

    Parameters
    ----------
    rows : list of Row
        All rows of one document, in reading order.
    breaks : list of int
        `break_strengths(rows)`.

    Returns
    -------
    list of int
        Parallel to `rows`: 0 when cutting before the row leaves any list whole,
        1 when it falls between two items of one list, 2 when it falls inside an
        item.
    """
    kinds = [0] * len(rows)
    in_list = False
    for index, row in enumerate(rows):
        item = bool(BULLET_START.match(row.text))
        if in_list:
            if item:
                kinds[index] = 1
            elif breaks[index] < 2:
                kinds[index] = 2
        if item:
            in_list = True
        elif breaks[index] >= 2:
            in_list = False
    return kinds


def pack_pages(rows: list[Row], tokenizer: Tokenizer) -> list[Chunk]:
    """Make one chunk per page, whatever the page holds.

    The page is the unit the reader is shown: a hit opens a page with its passage
    highlighted, so a vector per page asks the index exactly the question the
    interface answers. It is also the coarsest useful unit, which is why it is a
    variant rather than the default: a whole dense page can run past 1000 tokens,
    and averaging a page's meaning into one vector is what makes "which document
    is this about" easy and "where does it say that" impossible.

    Parameters
    ----------
    rows : list of Row
        All rows of one document, in reading order.
    tokenizer : Tokenizer
        Tokenizer of the embedding model, used only to record token counts, since
        no budget is enforced here.

    Returns
    -------
    list of Chunk
        One chunk per page that has any text, in page order. Pages whose rows
        were all boilerplate simply do not appear.
    """
    if not rows:
        return []
    lengths = [len(enc.ids) for enc in tokenizer.encode_batch([r.text for r in rows])]
    by_page: dict[int, Chunk] = {}
    for row, length in zip(rows, lengths):
        by_page.setdefault(row.page, Chunk()).add(row, length)
    return [by_page[page] for page in sorted(by_page)]


def pack_sections(rows: list[Row], tokenizer: Tokenizer,
                  limit: int = MAX_PASSAGE_TOKENS) -> list[Chunk]:
    """Make one chunk per section, whatever size the section is.

    A section is a heading run and everything under it up to the next heading
    run, as `heading_context` already defines it: consecutive rows that sit under
    the same run are one section, and the rows before the document's first
    heading are one section too. There is no hierarchy here on purpose. A
    sub-heading opens a section of its own, because the alternative, closing a
    section only at a heading of the same or larger type, makes a top-level
    section swallow every sub-section under it and run past the encoder's window
    on most guidelines; the lowest-level section is the one a reader would call
    "the section this passage is in".

    The budget is not a target but a ceiling: a section longer than `limit`
    tokens is cut into consecutive pieces at row boundaries, with no overlap,
    because `embed.py` would otherwise truncate it and embed its first half only.
    A document with no font sizes has no headings and so is one section, split
    at the ceiling, which is the right degenerate case: its section vector is
    then a document vector.

    Parameters
    ----------
    rows : list of Row
        All rows of one document, in reading order.
    tokenizer : Tokenizer
        Tokenizer of the embedding model, so the ceiling is exact.
    limit : int, optional
        The encoder's window, in tokens. The default is the shared constant.

    Returns
    -------
    list of Chunk
        Sections in document order. Every row appears in exactly one of them.
    """
    if not rows:
        return []
    lengths = [len(enc.ids) for enc in tokenizer.encode_batch([r.text for r in rows])]
    context = heading_context(rows)
    chunks: list[Chunk] = []
    current = Chunk()
    key: list[int] | None = None
    for row, length, under in zip(rows, lengths, context):
        opens = under != key
        if current.rows and (opens or current.n_tokens + length > limit):
            chunks.append(current)
            current = Chunk()
        key = under
        current.add(row, length)
    if current.rows:
        chunks.append(current)
    return chunks


def attach_heading(
    chunk: Chunk,
    *,
    start: int,
    rows: list[Row],
    lengths: list[int],
    context: list[list[int]],
    limit: int = HEADING_MAX_TOKENS,
) -> Chunk:
    """Give a chunk the heading it sits under, unless it already holds it.

    Parameters
    ----------
    chunk : Chunk
        The chunk about to be emitted. Modified in place and returned.
    start : int
        Index, in `rows`, of the chunk's first row. That is what decides which
        heading applies: the one in force where the chunk BEGINS, not where it ends,
        so a chunk that runs on into the next section is still filed under the
        section it opened in.
    rows, lengths : list
        The document's rows and their token counts, parallel.
    context : list of list of int
        Output of `heading_context`, parallel to `rows`. Empty to carry nothing,
        which is how `pack` turns the whole feature off.
    limit : int, optional
        Longest run to carry, in tokens.

    Returns
    -------
    Chunk
        The same chunk.

    Notes
    -----
    Rows the chunk already holds are dropped from the carry, which is what stops the
    first chunk of a section from printing its heading twice. An over-long run is
    dropped whole rather than truncated (HEADING_MAX_TOKENS), since half a heading
    situates nothing.
    """
    if not context or not chunk.rows:
        return chunk
    held = set(range(start, start + len(chunk.rows)))
    carried = [index for index in context[start] if index not in held]
    if not carried:
        return chunk
    tokens = [lengths[index] for index in carried]
    if sum(tokens) > limit:
        return chunk
    chunk.heading = [rows[index] for index in carried]
    chunk.heading_tokens = tokens
    return chunk


def pack(
    rows: list[Row], tokenizer: Tokenizer, *, target: int, overlap: int,
    boundaries: bool = False, sections: bool = True, headings: int = 0,
    overlap_boundaries: bool = False
) -> list[Chunk]:
    """Pack rows into overlapping chunks under a token budget.

    Rows are never split, so a chunk boundary always falls on a line boundary
    and every stored box remains an exact quotation of the page. A single row
    longer than the budget becomes its own oversized chunk rather than being
    cut: that happens on dense table rows, where splitting would produce text
    that is meaningless anyway, and the model accepts far longer inputs than
    the target.

    With `headings`, a chunk that continues a section is prefixed with that
    section's heading. A section longer than the budget becomes several chunks, and
    only the first of them holds the heading; the others read as anonymous prose,
    although the words a reader would search for ("prise en charge", "posologie")
    are often in the heading alone. The prefix is text only, never provenance: see
    `Chunk.heading`.

    With `boundaries`, the budget stops being the only thing that ends a chunk.
    The packer fills to the budget as usual and then rewinds to the last
    structural break it passed (see `break_strengths`), provided that leaves at
    least `FILL_FLOOR` of the target behind, so a chunk ends where a paragraph, a
    list item, a heading or a page ends rather than at whatever row happened to
    exhaust the budget. The rows after that break open the next chunk instead of
    being stranded at the end of this one.

    A SECTION boundary is preferred over a paragraph one. `break_strengths` scores
    a heading or a page change 3 where a paragraph or a list item scores 2, and the
    packer rewinds to the last 3 in the window before it considers any 2. A chunk
    that ends where a section ends is one whole piece of the document's own
    structure rather than an arbitrary window over it, which is the point of the
    1024-token target: a section of a guideline rarely fits in 300 tokens, so at the
    old budget the section rung would almost never have been reachable. Sections
    shorter than `FILL_FLOOR` are passed over, so the floor still bounds how small
    this can make a chunk.

    Where no block boundary is far enough along, it rewinds to the last SENTENCE
    end instead, under the same floor. A page of unbroken prose has no paragraph
    to stop at for hundreds of tokens, and the budget then cut mid-clause: two
    thirds of all chunks used to end in the middle of a sentence, and half of one
    is what the reader is shown in the result row and what the model embeds.
    A sentence end is a weaker boundary than a paragraph, which is why it is only
    ever the second choice.

    Where there is neither behind it, the packer reads FORWARD instead, up to
    `OVERSHOOT` of the target, to finish the sentence it is in: rows are never
    split and a sentence rarely ends at a line end, so the last third of a chunk
    often holds no boundary to rewind to at all. Only when even that is out of
    reach (a dense table, a page of figures) does the budget cut where it lands,
    carrying the overlap forward as it always did.

    Rewinding rather than closing at the first break past the floor: both consider
    the same breaks, but the first-break rule lands chunks near the floor, and the
    200-token row of the size sweep in DESIGN.md is the measurement that says
    smaller chunks lose rank 1. When there is no break in the window at all (a
    dense table, a page of running prose), the budget cuts as it always did.

    Parameters
    ----------
    rows : list of Row
        All rows of one document, in reading order.
    tokenizer : Tokenizer
        Tokenizer of the embedding model, so the budget is exact.
    target : int
        Soft maximum tokens per chunk.
    overlap : int
        Tokens of trailing context carried into the next chunk.
    boundaries : bool, optional
        Whether to end chunks at detected structure. Default False, which is the
        purely greedy packing every shipped index so far was built with.
    headings : int, optional
        Longest heading run to carry into a chunk that continues a section, in
        tokens. 0, the default, carries nothing. See `attach_heading`.
    overlap_boundaries : bool, optional
        Carry the overlap across a boundary cut too, not only across a cut the
        budget forced. Off by default, which is what ships; the grid's overlap rows
        turn it on to measure the Notes below rather than assume them.

    Returns
    -------
    list of Chunk
        Chunks in document order. Empty only when `rows` is empty.

    Notes
    -----
    A chunk closed AT a boundary carries no overlap into the next one, while a
    chunk closed by the budget still does. The overlap exists so that a fact
    split across a chunk boundary stays retrievable from both sides; when the
    boundary is one the author put there, nothing is split, and carrying the tail
    forward would only re-open the paragraph that just ended, which is precisely
    the boundary this mode was asked to respect.

    That reasoning is an argument, not a measurement, and `overlap_boundaries` is
    the switch that tests it: with it on, the chunk after a boundary cut opens on
    the tail of the one before, so a section's first chunk starts a few lines
    before the section does. The case for it is a question that spans the end of
    one section and the start of the next ("contre-indications" listed just before
    the "posologie" heading); the case against is that the page vector already
    covers the neighbourhood and the overlap dilutes a chunk that was one whole
    section.
    """
    if not rows:
        return []
    # One batch encode rather than one call per row: tokenizers releases the
    # GIL and batches internally, and this is the hot loop of the whole build.
    lengths = [len(enc.ids) for enc in tokenizer.encode_batch([r.text for r in rows])]
    breaks = break_strengths(rows) if boundaries else [0] * len(rows)
    # Recorded on the rows themselves, because `Chunk.text` joins rows and has no
    # other way to know which of them a reader would see as starting a paragraph.
    # Rows are shared between overlapping chunks, so this is written once per row
    # and read by every chunk that carries it.
    for row, strength in zip(rows, breaks):
        row.starts_block = strength >= 2
    context = heading_context(rows) if headings else []
    floor = int(target * FILL_FLOOR)
    relaxed = int(target * SENTENCE_FLOOR)
    slack = int(target * OVERSHOOT)
    # Whether cutting BEFORE row i leaves a finished sentence behind, which is a
    # property of row i-1 rather than of row i. `break_strengths` cannot answer this:
    # its score is overwritten by the bullet, pitch and heading signals, so a row that
    # opens a paragraph after a line ending in "les résultats sont" scores 2 there and
    # used to look like a clean place to cut. Every cut below is checked against this
    # list instead, and that is what stopped chunks ending in the middle of a claim.
    clean = [False] + [ends_sentence(row.text) for row in rows[:-1]]
    # And whether it would break up a bulleted list, which ranks below every cut
    # that does not (see `list_cuts`). All zeros without `boundaries`, like `breaks`.
    in_list = list_cuts(rows, breaks) if boundaries else [0] * len(rows)

    chunks: list[Chunk] = []
    current = Chunk()
    # Index, within `current`, of the last row that STARTED a block on a finished
    # sentence, and of the last row that merely started a sentence. Not "the last
    # boundary seen", which would point into the previous chunk once one is emitted.
    last_break = -1
    last_sentence = -1
    # And the last row that started a SECTION on a finished sentence: a heading or a
    # page change, which `break_strengths` scores 3 where a paragraph or a list item
    # scores 2. Tracked apart from `last_break` so the ladder can prefer it: a chunk
    # that ends where a section ends is one whole piece of the document's own
    # structure, which is what the 1024-token target is for.
    last_section = -1
    # And the last row that started a block WITHOUT a finished sentence behind it: a
    # heading under a truncated line, a paragraph after a caption. Kept apart because
    # it is the rung of the ladder below that cuts mid-sentence, and it is only taken
    # when nothing else is left.
    last_rough = -1
    # The last finished-sentence cut BETWEEN two items of a list, and the last one
    # INSIDE an item. Kept apart from the rungs above so a list is only split when
    # nothing outside it is in reach, and an item only when nothing between items is.
    last_item = -1
    last_inner = -1
    # Index, in `rows`, of `current`'s first row. Needed only to look that chunk's
    # heading up, but it has to be maintained at each emission point rather than
    # recomputed, because the overlap carry makes a chunk start behind the row being
    # read.
    start = 0
    # Rows the look-ahead below has already put in a chunk. The loop walks past them
    # without reading them again.
    absorbed = 0
    carry = dict(rows=rows, lengths=lengths, context=context, limit=headings)
    for position, (row, length, strength) in enumerate(zip(rows, lengths, breaks)):
        if position < absorbed:
            continue
        if current.rows and current.n_tokens + length > target:
            # Fill to the budget, then rewind to the last structural boundary, rather
            # than closing at the FIRST boundary past the floor. Both cut in the same
            # places; rewinding picks the last candidate instead of the first, so
            # chunks land near the target instead of near the floor, and the 200-token
            # sweep in DESIGN.md is the measurement that says near the target is where
            # rank 1 lives.
            # A block boundary first, then a sentence end. Two thirds of the budget
            # has to sit behind either of them, so that neither shrinks the chunk back
            # towards the 200-token size the sweep in DESIGN.md rejected.
            # The ladder, best rung first: a SECTION boundary (a heading or a page
            # change, strength 3) with a finished sentence behind it, then any block
            # boundary, then any finished sentence, then reading forward to the next
            # one, then a short but finished chunk, and only then a cut that truncates.
            # The section rung is why the target is 1024 rather than 300: a chunk that
            # ends where a section ends is one whole piece of the document's own
            # structure, and a section rarely fits in 300 tokens.
            # Without `sections` the top rung is dropped: the last block boundary
            # wins whatever kind it is, which is the paragraph-preferred policy.
            # `last_break` already covers strength 3, so a chunk can still end at a
            # heading; what stops is the packer rewinding FURTHER back to reach one,
            # so chunks land closer to the target and a long section is cut at its
            # own paragraphs instead of being held whole.
            # Below all of them, a cut between two list items, then one inside an
            # item: a bulleted list is split only when nothing outside it is in reach.
            ladder = ((last_section, last_break, last_sentence, last_item, last_inner)
                      if sections else (last_break, last_sentence, last_item, last_inner))
            cut = _first_fit(current, ladder, floor=floor)
            end = None
            if cut < 0 and boundaries:
                end = _reach_boundary(position, breaks, clean, lengths, slack=slack)
            if cut < 0 and end is None:
                cut = _first_fit(current, ladder, floor=relaxed)
                if cut < 0:
                    cut = _first_fit(current, (last_rough,), floor=floor)
            if cut > 0:
                head, current = current.split(cut)
                chunks.append(attach_heading(head, start=start, **carry))
                start += cut
                # Rows prepended to the tail by the boundary overlap. Every index
                # below shifts by that many, and none of them may point INTO the
                # carry: a cut there would emit a chunk made mostly of rows the
                # previous one already holds.
                shift = 0
                if overlap_boundaries and overlap:
                    seed = _carry_over(head, overlap=overlap)
                    shift = len(seed.rows)
                    current.rows[:0] = seed.rows
                    current.token_counts[:0] = seed.token_counts
                    start -= shift
                # The tail already carries the start of the new block or sentence, and
                # that boundary is one the author put there, so nothing is straddling
                # it and there is nothing to repeat: no overlap is carried across a
                # boundary, only across a break the budget forced.
                # A boundary further along than the cut survives it, renumbered onto
                # the rows that are still here. Those used to be forgotten, which left
                # the next chunk blind to every break it had already read past.
                last_section = last_section - cut + shift if last_section > cut else -1
                last_break = last_break - cut + shift if last_break > cut else -1
                last_sentence = last_sentence - cut + shift if last_sentence > cut else -1
                last_rough = last_rough - cut + shift if last_rough > cut else -1
                last_item = last_item - cut + shift if last_item > cut else -1
                last_inner = last_inner - cut + shift if last_inner > cut else -1
            elif end is not None:
                # Nothing behind to rewind to. Read forward instead, as far as the
                # end of the sentence in progress.
                for index in range(position, end + 1):
                    current.add(rows[index], lengths[index])
                chunks.append(attach_heading(current, start=start, **carry))
                current = Chunk()
                start = end + 1
                last_section = last_break = last_sentence = last_rough = last_item = last_inner = -1
                if end >= position:
                    # Those rows are in the chunk that just closed, so the loop
                    # must not add them again on its way past.
                    absorbed = end + 1
                    continue
                # end is one row behind: the row being read starts a block, so
                # the chunk closed on the row before it and this one opens the
                # next chunk, added by the statement after this branch.
            else:
                # Nothing whole anywhere in reach: a dense table, a reference list, a
                # page of figure labels. The budget cuts where it lands, as it always
                # did, and the overlap carries the tail forward so the sentence is
                # still readable from the next chunk.
                chunks.append(attach_heading(current, start=start, **carry))
                current = _carry_over(current, overlap=overlap)
                # The carried rows are the ones just read, so the new chunk starts
                # that many rows behind the row about to be added.
                start = position - len(current.rows)
                last_section = last_break = last_sentence = last_rough = last_item = last_inner = -1
        current.add(row, length)
        if clean[position] and in_list[position] == 1:
            last_item = len(current.rows) - 1
        elif clean[position] and in_list[position] == 2:
            last_inner = len(current.rows) - 1
        elif clean[position]:
            last_sentence = len(current.rows) - 1
            if strength >= 2:
                last_break = len(current.rows) - 1
            if strength >= 3:
                last_section = len(current.rows) - 1
        elif strength >= 2:
            last_rough = len(current.rows) - 1
    if current.rows:
        chunks.append(attach_heading(current, start=start, **carry))
    return chunks


def _first_fit(current: Chunk, candidates: tuple[int, ...], *, floor: int) -> int:
    """First candidate cut that leaves at least `floor` tokens behind, or -1.

    Parameters
    ----------
    current : Chunk
        The chunk being filled.
    candidates : tuple of int
        Row indices within `current`, in order of preference.
    floor : int
        Minimum tokens the chunk must keep.

    Returns
    -------
    int
        The chosen index, or -1 when no candidate is far enough along.
    """
    for candidate in candidates:
        if candidate > 0 and sum(current.token_counts[:candidate]) >= floor:
            return candidate
    return -1


def _reach_boundary(position: int, breaks: list[int], clean: list[bool],
                    lengths: list[int], *, slack: int) -> int | None:
    """Last row to absorb so that the chunk ends on a boundary, or None.

    Called when the budget is spent and there is nothing behind worth rewinding
    to. Rather than cutting mid-clause, the packer reads a little further: the
    first row that ENDS something (the next row starts a sentence, a paragraph, a
    heading or a page) closes the chunk, provided it is within `slack` tokens.

    Returns None when the next boundary is further than the slack allows, which is
    what happens inside a dense table: there the budget still cuts, as it always
    did, because a table has no sentence to finish.

    Parameters
    ----------
    position : int
        Index of the row the budget just refused.
    breaks : list of int
        Break strength per row, as `break_strengths` scores them.
    clean : list of bool
        Whether cutting before each row leaves a finished sentence behind, which
        is what makes a boundary worth stopping on. A row that opens a paragraph
        after a truncated line is a boundary by layout and not by meaning, and
        stopping there is the cut this reading forward exists to avoid.
    lengths : list of int
        Token count per row, parallel to `breaks`.
    slack : int
        Tokens the chunk may run past its budget.

    Returns
    -------
    int or None
        Index of the last row to absorb, inclusive.
    """
    spent = 0
    for index in range(position, len(breaks)):
        # A row that STARTS a block is a heading or a new paragraph, and a heading
        # belongs with the text under it: ending on it orphans it from its own
        # section and gives the next chunk anonymous prose. The row before it is a
        # boundary too, and a better one. That can be the row already in the chunk,
        # which is why this may point one row behind where the reading started.
        if breaks[index] >= 2 and clean[index]:
            return index - 1
        spent += lengths[index]
        if spent > slack:
            return None
        # The document's last row ends everything there is to end.
        if index + 1 >= len(breaks) or clean[index + 1]:
            return index
    return None


def _carry_over(previous: Chunk, *, overlap: int) -> Chunk:
    """Open a new chunk seeded with the tail of the finished one.

    Walks the finished chunk backwards collecting trailing rows until the
    overlap budget is met, so the next chunk reopens on the sentence the
    previous one closed with and a fact straddling the boundary is reachable
    from both sides.

    The carry is capped at half the finished chunk's rows. Without that cap a
    chunk made of many short rows carries nearly all of itself forward, the
    next chunk starts almost full, and the document yields several times more
    chunks than it has text: the first version of this function had exactly
    that failure and produced 161k chunks from 7910 pages.

    Parameters
    ----------
    previous : Chunk
        The chunk just completed.
    overlap : int
        Token budget for the carried tail.

    Returns
    -------
    Chunk
        A new chunk holding only the carried rows.
    """
    carried = Chunk()
    limit = len(previous.rows) // 2
    for row, length in zip(reversed(previous.rows), reversed(previous.token_counts)):
        if len(carried.rows) >= limit or carried.n_tokens + length > overlap:
            break
        carried.rows.insert(0, row)
        carried.token_counts.insert(0, length)
    # The budget decided where the carry starts, which is to say it started in the
    # middle of a sentence: the next chunk then OPENED on half a clause, the mirror
    # of the problem the sentence rewind fixes at the other end. Leading rows are
    # dropped until one that follows a full stop, and never all of them, because an
    # empty carry is no overlap at all. Dropping loses nothing: those rows are in the
    # chunk that just closed.
    for index in range(1, len(carried.rows)):
        # The same test the packer cuts on, so that the two ends of a chunk agree:
        # a carry that starts after "Les critères sont :" opens on a fragment just
        # as surely as one that starts mid-clause.
        if ends_sentence(carried.rows[index - 1].text):
            del carried.rows[:index]
            del carried.token_counts[:index]
            break
    return carried
