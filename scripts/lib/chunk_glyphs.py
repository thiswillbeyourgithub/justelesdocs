"""Glyph and word level reading of a PDF page.

pymupdf's rawdict gives one box per character. The functions here turn those
into words and lines: deciding which spaces are real (`glued`, `letter_gap`),
reattaching accents drawn as separate glyphs, reading Symbol-font bullets and
arrows, and handling pages whose text runs sideways (`reading_blocks`,
`upright_rect`). They sit below both row extraction and table reading,
because both read words off a page and neither may import the other.

Written by Claude Code (Opus 5.5), split out of chunk.py.
"""

from __future__ import annotations

import pymupdf

from lib.chunk_row import median
from lib.chunk_text import (
    DINGBAT_FONT, GLUE, GLUED_MIN_PAIRS, GLUED_OVERLAP_EM, GLUED_SPACE_EM, SPACING_ACCENTS,
    SYMBOL_GLYPHS,
)


# Vertical overlap, as a fraction of the shorter line's height, above which
# two lines are considered to sit on the same visual row and get merged.
ROW_MERGE_OVERLAP = 0.5

# A page is read in its sideways frame when more than this share of its characters
# run up or down it. pymupdf reports text in the UNROTATED page's coordinates and
# sorts blocks top to bottom there, so on a sideways table (a /Rotate 90 page, or
# an upright page printed sideways) it orders them by where they sit ACROSS the
# reader's view: one guideline's "TABLE 8" title, the leftmost line on paper and the
# first line to the reader, came out after the table under it.
SIDEWAYS_SHARE = 0.5


def overlap_share(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    """How much of the smaller of two boxes the other one covers, from 0 to 1."""
    smaller = min(a.get_area(), b.get_area())
    return (a & b).get_area() / smaller if smaller > 0 else 0.0


def upright_rect(page: pymupdf.Page) -> pymupdf.Rect:
    """The page as the text coordinates see it: unrotated, origin at the top left.

    Parameters
    ----------
    page : pymupdf.Page
        The page.

    Returns
    -------
    pymupdf.Rect
        `(0, 0, width, height)` of the page's crop box, before any /Rotate.

    Notes
    -----
    NOT `page.rect`, which carries the rotation: on a landscape scan of a portrait
    page it is 1224x792 where every line bbox on it runs to 792x1224. Line
    coordinates come out of pymupdf unrotated, `src/viewer.js` converts them
    assuming exactly that (the viewport is what applies /Rotate), and this is the
    rectangle they live in. Any page dimension compared against a row's bbox has to
    come from here, or the comparison is right on 34 of 36448 pages and wrong on
    the rest by a quarter turn.
    """
    return pymupdf.Rect(0, 0, page.cropbox.width, page.cropbox.height)


def reading_blocks(page: pymupdf.Page) -> list[dict]:
    """A page's text blocks in reading order, sideways pages included.

    Parameters
    ----------
    page : pymupdf.Page
        The page to read.

    Returns
    -------
    list of dict
        The blocks of ``page.get_text("rawdict", sort=True)``, whose lines carry
        each glyph's box for `line_text` to read. On a page whose text
        mostly runs up or down it, they are re-sorted by pymupdf's own key, bottom
        edge then left edge, but measured with the page turned so that text reads
        left to right. On one guideline's rotated page, where the text runs upward, the title
        line sits at x = 64 and the table at x = 84 to 518, which turned by 90
        degrees become y = 64 and y = 84 onward: title first. Blocks running any
        other way keep pymupdf's order, after those.
    """
    blocks = page.get_text("rawdict", sort=True)["blocks"]
    sides: list[int] = []
    weight = {-1: 0, 0: 0, 1: 0}  # characters running up the page, across it, down it
    for block in blocks:
        tally = {-1: 0, 0: 0, 1: 0}
        for line in block.get("lines", []):
            across, down = line["dir"]
            side = 0 if abs(across) >= abs(down) else (1 if down > 0 else -1)
            tally[side] += sum(1 for span in line["spans"] for char in span["chars"]
                               if not char["c"].isspace())
        sides.append(max(tally, key=tally.__getitem__))
        for side, count in tally.items():
            weight[side] += count
    side = max((-1, 1), key=weight.__getitem__)
    if weight[side] <= SIDEWAYS_SHARE * sum(weight.values()):
        return blocks
    # Matrix(90) turns text running up the page, direction (0, -1), into (1, 0).
    turn = pymupdf.Matrix(90 if side < 0 else -90)

    def key(block: dict) -> tuple[float, float]:
        box = pymupdf.Rect(block["bbox"]) * turn
        return box.y1, box.x0

    # Blocks running the other way (the running footer and page number the journal
    # prints upright on its sideways page) follow, in the order they always had:
    # sorted in with the table they would land among its footnotes, and apart from
    # each other, where the boilerplate filter no longer recognises the pair.
    along = [block for block, block_side in zip(blocks, sides) if block_side == side]
    other = [block for block, block_side in zip(blocks, sides) if block_side != side]
    return sorted(along, key=key) + other


def glyph_extent(char: dict, direction: tuple[float, float]) -> tuple[float, float]:
    """Where a glyph starts and ends along the direction its line reads.

    Parameters
    ----------
    char : dict
        One of a rawdict span's ``chars``.
    direction : tuple of float
        The line's ``dir``: (1, 0) across the page, (0, -1) up it, (0, 1) down it.

    Returns
    -------
    tuple of (float, float)
        Start and end, growing in reading order whatever the direction, so that
        ``next_start - previous_end`` is the gap between two glyphs on any line.
    """
    x0, y0, x1, y1 = char["bbox"]
    across, down = direction
    if abs(across) >= abs(down):
        return (x0, x1) if across > 0 else (-x1, -x0)
    return (y0, y1) if down > 0 else (-y1, -y0)


Glyph = tuple[dict, dict]  # a rawdict char and the span it belongs to


def letter_gap(glyphs: list[Glyph], direction: tuple[float, float],
               size: float) -> float | None:
    """The gap a line leaves between two letters of one word.

    Parameters
    ----------
    glyphs : list of (dict, dict)
        The line's chars, each with its span, in stream order.
    direction : tuple of float
        The line's ``dir``.
    size : float
        The line's font size, which the thresholds are fractions of.

    Returns
    -------
    float or None
        The median gap between consecutive letters, often negative because glyph
        boxes are wider than the glyphs. None when the boxes cannot say: fewer than
        GLUED_MIN_PAIRS letter pairs, boxes of no width (one leaflet's are all zero),
        or a stream running backwards (right-to-left script).
    """
    gaps: list[float] = []
    widths: list[float] = []
    for (before, _), (after, _) in zip(glyphs, glyphs[1:]):
        if before["c"].isalpha() and after["c"].isalpha():
            start, end = glyph_extent(before, direction)
            gaps.append(glyph_extent(after, direction)[0] - end)
            widths.append(end - start)
    if len(gaps) < GLUED_MIN_PAIRS:
        return None
    gap, width = median(gaps), median(widths)
    if width < 0.2 * size or gap + width <= 0:
        return None
    return gap


def glued(before: Glyph, after: Glyph, *, gap: float, direction: tuple[float, float],
          size: float) -> bool:
    """Whether two letters with only a space or a line break between them are one word.

    Parameters
    ----------
    before, after : tuple of (dict, dict)
        The two letters, each with its span.
    gap : float
        The line's own letter gap, from `letter_gap`.
    direction : tuple of float
        The line's ``dir``.
    size : float
        The line's font size.

    Returns
    -------
    bool
        True when they are the same font at the same size and sit no further apart
        than two letters of a word do. The font test is what keeps "*Insight* et"
        apart: an italic's overhang closes the gap in front of the roman word.
    """
    (first, first_span), (second, second_span) = before, after
    if not (first["c"].isalpha() and second["c"].isalpha()):
        return False
    if (first_span["font"], first_span["size"]) != (second_span["font"], second_span["size"]):
        return False
    extra = glyph_extent(second, direction)[0] - glyph_extent(first, direction)[1] - gap
    return -GLUED_OVERLAP_EM * size < extra < GLUED_SPACE_EM * size


def line_glyphs(line: dict) -> list[Glyph]:
    """A rawdict line's chars, each with its span, in stream order.

    A spacing accent drawn on top of a letter is made the combining mark it
    stands for, placed after that letter (`attach_accents`).
    """
    glyphs = [(symbol_glyph(char, span["font"]), span) for span in line["spans"]
              for char in span["chars"]]
    return attach_accents(glyphs, line.get("dir", (1.0, 0.0)))


def symbol_glyph(char: dict, font: str) -> dict:
    """A rawdict char, with a dingbat font's glyph read as the mark it draws.

    Parameters
    ----------
    char : dict
        One of a rawdict span's ``chars``.
    font : str
        Its span's font name.

    Returns
    -------
    dict
        ``char`` itself, or a copy whose text is "\u2022" when a DINGBAT_FONT drew
        it, or the SYMBOL_GLYPHS reading of a Symbol-font code pymupdf mis-maps.

    Notes
    -----
    A dingbat font's codes are the font's own glyph numbers, and pymupdf reports
    them as whatever Latin-1 character shares the number: Wingdings3's arrow bullet
    extracts as an acute accent (one reference document's 214 bullets, six
    agency fact sheets), Wingdings' check box as a diaeresis ("\u00a81 ; \u00a82" in
    a 2002 guideline). NFKC then made each a floating combining mark.
    Whitespace is left alone, and so is the private-use area, which `clean_text`
    drops on purpose.
    """
    text = char["c"]
    if text.isspace() or 0xE000 <= ord(text[0]) <= 0xF8FF:
        return char
    if DINGBAT_FONT.search(font):
        return {**char, "c": "\u2022"}
    reading = SYMBOL_GLYPHS.get(text) if "symbol" in font.lower() else None
    return {**char, "c": reading} if reading else char


def symbol_readings(page: pymupdf.Page) -> list[tuple[pymupdf.Point, str, str]]:
    """Every glyph on the page that `symbol_glyph` reads as something else.

    Returns
    -------
    list of tuple
        ``(centre, extracted, reading)`` for each such glyph, so `reread_symbols`
        can apply the same reading to the text `find_tables` extracts, which
        never passes through `line_glyphs`.
    """
    found = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", ()):
            for span in line["spans"]:
                for char in span["chars"]:
                    reading = symbol_glyph(char, span["font"])["c"]
                    if reading != char["c"]:
                        box = pymupdf.Rect(char["bbox"])
                        found.append(((box.tl + box.br) / 2, char["c"], reading))
    return found


def reread_symbols(table, data: list[list], symbols: list, upright) -> list[list]:
    """``table.extract()``'s cells with each dingbat or Symbol glyph read as `symbol_glyph` does.

    A glyph belongs to the cell whose box holds its centre, and replaces one
    occurrence of the character it extracted as in that cell. Without this the
    référentiel tabac's tables kept 80 Wingdings bullets as acute accents and
    one guideline's dosing table read "May \u00af AMP" for "May \u2193 AMP". ``upright``
    brings a cell box from find_tables' rotated frame into the glyphs' one.
    """
    if not symbols:
        return data
    for row, cells in zip(table.rows, data):
        for column, (box, text) in enumerate(zip(row.cells, cells)):
            if box is None or not text:
                continue
            box = upright(box)
            for centre, extracted, reading in symbols:
                if centre in box and extracted in text:
                    text = text.replace(extracted, reading, 1)
            cells[column] = text
    return data


def attach_accents(glyphs: list[Glyph], direction: tuple[float, float]) -> list[Glyph]:
    """Turn every spacing accent that sits on a letter into that letter's mark.

    Parameters
    ----------
    glyphs : list of (dict, dict)
        A line's chars, each with its span, in stream order.
    direction : tuple of float
        The line's ``dir``.

    Returns
    -------
    list of (dict, dict)
        The same glyphs, each accent in SPACING_ACCENTS whose middle lies inside
        the letter before or after it replaced by its combining mark and moved
        behind that letter, so NFKC composes the pair. An acute BETWEEN two
        letters it does not sit on is an apostrophe and becomes one. Every other
        glyph is returned as it came.

    Notes
    -----
    TeX-set PDFs draw an accent as its own glyph at its letter's position, and
    the text layer then reads "Clı´nica", "Hoˆpital", "Salpeˆtrie`re": 1,500
    such accents in 31 documents, which NFKC turned into a space and a floating
    mark ("Clı ́nica") that the tokenizer spends two junk tokens on. Only the
    geometry can tell that accent from the same character used as an
    apostrophe: one classification's "individual´s" draws it after the "l", not on it.
    """
    out = list(glyphs)
    for index, (char, span) in enumerate(glyphs):
        mark = SPACING_ACCENTS.get(char["c"])
        if mark is None:
            continue
        start, end = glyph_extent(char, direction)
        middle = (start + end) / 2
        for base in (index - 1, index + 1):
            if not 0 <= base < len(glyphs) or not glyphs[base][0]["c"].isalpha():
                continue
            low, high = glyph_extent(glyphs[base][0], direction)
            if low <= middle <= high:
                out[index] = ({**char, "c": mark}, span)
                if base > index:  # drawn before its letter: the mark goes after it
                    out[index], out[base] = out[base], out[index]
                break
        else:
            if (char["c"] == "´" and 0 < index < len(glyphs) - 1
                    and glyphs[index - 1][0]["c"].isalpha() and glyphs[index + 1][0]["c"].isalpha()):
                out[index] = ({**char, "c": "’"}, span)
    return out


def line_size(line: dict) -> float:
    """The largest font size on a line, 0 when it has no span."""
    return max((span["size"] for span in line["spans"]), default=0.0)


def split_words(line: dict) -> list[list[Glyph]]:
    """A rawdict line's words, split at every space but the ones inside a word.

    Parameters
    ----------
    line : dict
        One line of ``page.get_text("rawdict")``.

    Returns
    -------
    list of list of (dict, dict)
        Each word's glyphs. A space `glued` says sits inside a word splits nothing
        and becomes GLUE: one guideline's "ar e needed" is two words, "ar⁠e" and
        "needed", and one handbook's "proﬁ les" one, for `resolve_glue` to decide.
    """
    glyphs = line_glyphs(line)
    size = line_size(line)
    gap = letter_gap(glyphs, line["dir"], size) if size else None
    words: list[list[Glyph]] = []
    current: list[Glyph] = []
    for index, glyph in enumerate(glyphs):
        char = glyph[0]["c"]
        if not char.isspace():
            current.append(glyph)
            continue
        if (gap is not None and char == " " and current and index < len(glyphs) - 1
                and glued(glyphs[index - 1], glyphs[index + 1], gap=gap,
                          direction=line["dir"], size=size)):
            current.append(({"c": GLUE, "bbox": glyph[0]["bbox"]}, glyph[1]))
            continue
        if current:
            words.append(current)
            current = []
    if current:
        words.append(current)
    return words


def word_text(word: list[Glyph]) -> str:
    """The characters of a word from `split_words`."""
    return "".join(char["c"] for char, _ in word)


def line_text(line: dict) -> str:
    """A rawdict line's text, GLUE in place of the spaces `glued` doubts.

    The line's other whitespace becomes single spaces, which `clean_text` would
    have made of it anyway.
    """
    return " ".join(word_text(word) for word in split_words(line))


def page_words(page: pymupdf.Page) -> list[tuple]:
    """The page's words as ``get_text("words")`` gives them, GLUE for doubted spaces.

    Parameters
    ----------
    page : pymupdf.Page
        The page to read.

    Returns
    -------
    list of tuple
        ``(x0, y0, x1, y1, text, block, line, word)`` in the unrotated page's
        coordinates. Words are split by `split_words`, and a line that
        `continues_word` its previous one hands its first word to that line's last
        and takes its line number, so "reinit" and "iate", two lines on paper, are
        one word "reinit⁠iate".
    """
    words: list[tuple] = []
    for block in page.get_text("rawdict")["blocks"]:
        if block["type"] != 0:
            continue
        previous: dict | None = None
        number = -1
        count = 0  # words so far on line `number`
        for line in block["lines"]:
            found = split_words(line)
            if not found:
                continue
            if previous is not None and words and continues_word(previous, line):
                head, *found = found
                x0, y0, x1, y1, text, *place = words[-1]
                box = pymupdf.Rect(x0, y0, x1, y1)
                for char, _ in head:
                    box |= char["bbox"]
                words[-1] = (*box, text + GLUE + word_text(head), *place)
            else:
                number += 1
                count = 0
            previous = line
            for word in found:
                box = pymupdf.Rect(word[0][0]["bbox"])
                for char, _ in word[1:]:
                    box |= char["bbox"]
                words.append((*box, word_text(word), block["number"], number, count))
                count += 1
    return words


def continues_word(previous: dict, line: dict) -> bool:
    """Whether a block's line carries on the word its previous line stopped in.

    Parameters
    ----------
    previous, line : dict
        Two consecutive rawdict lines of one block, blank lines between them aside.

    Returns
    -------
    bool
        True when they share a direction and a visual line, and their facing letters
        pass `glued`. One guideline's "reinit" and "iate" are two lines that touch (one
        ends at 270.3, the other starts there), which used to reach the reader as
        "reinit iate".
    """
    if previous["dir"] != line["dir"]:
        return False
    before = [glyph for glyph in line_glyphs(previous) if not glyph[0]["c"].isspace()]
    after = [glyph for glyph in line_glyphs(line) if not glyph[0]["c"].isspace()]
    if not (before and after):
        return False
    across = 1 if abs(line["dir"][0]) >= abs(line["dir"][1]) else 0
    first, second = previous["bbox"], line["bbox"]
    overlap = min(first[across + 2], second[across + 2]) - max(first[across], second[across])
    height = min(first[across + 2] - first[across], second[across + 2] - second[across])
    if height <= 0 or overlap < ROW_MERGE_OVERLAP * height:
        return False
    size = max(line_size(previous), line_size(line))
    gap = letter_gap(before + after, line["dir"], size) if size else None
    return gap is not None and glued(before[-1], after[0], gap=gap, direction=line["dir"],
                                     size=size)
