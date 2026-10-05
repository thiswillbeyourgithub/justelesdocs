"""Text repair and language helpers for chunking.

What a PDF hands back as text is not what the page shows: ligatures arrive as
control codes, accents as separate glyphs, words cut in half at a line break,
false spaces inside words. The constants and functions here turn a row's raw
text into the words a reader sees (`clean_text`, `resolve_ligatures`,
`resolve_glue`, `dehyphenate`), tell French from English (`looks_french`) and
find where sentences end (`split_sentences`, `ends_sentence`). None of it reads
geometry, which is why it is separate from extraction: it can be tested on
strings, and the sentence and bullet patterns are shared with packing.

Written by Claude Code (Opus 5.5), split out of chunk.py.
"""

from __future__ import annotations

import re
import unicodedata

from lib.chunk_row import Row


# Some producers put a space glyph INSIDE a word: a ligature followed by one
# ("proﬁ les", 3,794 times in one handbook), or a letter drawn out of order
# with the space before it (one guideline's "ar e needed", which the page shows as "are
# needed"). The page itself says which spaces are real: across a real one the gap
# between the two letters is a space wider than the gap between two letters of a
# word on the same line (+0.20 to +0.32 em measured), across a false one it is no
# wider. So a space is dropped when it adds less than GLUED_SPACE_EM to the line's
# own letter gap, and not when the letters overlap by more than GLUED_OVERLAP_EM,
# which is text drawn twice rather than a word. The gap is measured against the
# line and never in absolute terms, because glyph boxes lie: in some documents every
# letter overlaps its neighbour by 0.36 em, in one handbook by 0.74. It needs
# GLUED_MIN_PAIRS letter pairs to know the line's gap at all.
#
# Geometry only NOMINATES a join, because it cannot tell a false space from a real
# one a kern has closed up: Ansari's "of Action" adds 0.066 em to its line's gap,
# one guideline's false "ar e" nothing, and a threshold between them fails one book or the
# other. A nominated space becomes GLUE, and `resolve_glue` lets the document's own
# vocabulary decide once the whole document is read, as `dehyphenate` does for a
# hyphen break. GLUE is WORD JOINER, invisible and never whitespace, so nothing
# between extraction and that decision splits a word on it.
GLUE = "\u2060"
GLUED_SPACE_EM = 0.08
GLUED_OVERLAP_EM = 0.3
GLUED_MIN_PAIRS = 3

# Stopwords that decide which language a page is written in, which decides which
# language the table is described in. Counting the commonest words of each is enough
# here: the corpus is French and English only, the two lists share no member, and the
# text being judged is a whole page.
FRENCH_STOPWORDS = frozenset("de la le les des et du une un est pour dans par au aux "
                             "sur avec ou ce cette sont plus".split())
ENGLISH_STOPWORDS = frozenset("the of and to in for is with that on are as at by from "
                              "this be or an was".split())

# A row starting with one of these is a list item, which is a structural boundary in
# the same way a paragraph is: a bullet begins a new statement, and the previous one
# is complete. Covers the usual glyphs plus "1." / "1)" / "(1)" / "a)" numbering.
BULLET_START = re.compile(r"^(?:[-–\u2014•▪‣·*o►▶➢]\s|\(?\d{1,2}[.)]\s|[a-z][.)]\s)")

# Sentence-final punctuation, the weakest signal of the three and never enough on its
# own to close a chunk: "Dr." and "n°3." end this way and end nothing.
SENTENCE_END = re.compile(r"[.!?;:»”\"']\s*$")

# The same idea, minus the two marks that end a clause but not a statement. A chunk
# closing on ":" strands the list that was being introduced, and one closing on ";"
# hands the reader an item whose sentence started in the previous chunk, so neither
# counts as a place the packer may end on. This is the set scripts/chunk_quality.py
# measures against, deliberately: the packer and the metric have to agree on what a
# finished chunk looks like or the number it reports means nothing.
SENTENCE_FINAL = re.compile(r"[.!?…»”\"')\]]\s*$")

# Floor for a cut that ends a sentence, as a share of the target, used only when the
# ordinary `FILL_FLOOR` finds nothing and reading forward finds nothing either. A
# 120-token chunk that says something whole beats a 300-token one that stops at "les
# résultats sont", and the size sweep in DESIGN.md measured a small loss for short
# chunks against a large one for truncated ones.
#
# 2026-09-30: 0.2 of 512, about 102 tokens, below `FILL_FLOOR` as it has to be for
# this rung to mean anything. At 0.4 of the new target it would sit ABOVE the
# floor and never be reached.
SENTENCE_FLOOR = 0.2

# A combining mark separated from its base letter by whitespace. The capture keeps
# the letter and the mark and drops what is between them.
ORPHANED_MARK = re.compile(r"(\w)\s+([\u0300-\u036f])")

# The same mark with no base letter anywhere before it: a combining acute opening a
# line, which is what a Symbol-font bullet extracts as in several agency documents
# ("\u0301 L'eau potable est..."). There is nothing to reattach it to, so it is
# what it looks like on the page, a list marker, and writing it as one also lets
# `BULLET_START` see the list item it introduces.
LEADING_MARK = re.compile(r"^\s*[\u0300-\u036f]+\s*(?=[^\W\d_])")

# A dotless i or j carrying a mark, which is how "\u00ef" comes out of a producer that
# draws the diaeresis separately ("valpro\u0131\u0308que"). NFKC will not compose those
# because the base letter is the wrong one; putting the dot back lets it.
DOTLESS_BASE = re.compile(r"[\u0131\u0237]([\u0300-\u036f])")

# Fonts whose every glyph is a picture rather than a letter. See `symbol_glyph`.
# "Sorts" is Monotype's name for its dingbats (a 1998 recommendation's
# bullets extract as a macron in MonotypeSorts).
DINGBAT_FONT = re.compile(r"(?i)wingding|webding|dingbat|sorts")

# Symbol-font codes pymupdf maps to the Latin-1 character sharing the number
# instead of the symbol drawn: 0xAF is Symbol's down arrow ("May \u00af AMP
# absorption" in one 2018 guideline). Symbol's Greek letters map correctly, which is why
# this is a table and not the whole font.
SYMBOL_GLYPHS = {"\u00af": "\u2193"}

# Spacing accents and the combining marks they stand for when drawn on a letter.
# See `attach_accents`. "^" and "~" are left out: they are operators far more
# often than accents, and none of the 31 TeX-set documents draws one on a letter.
SPACING_ACCENTS = {
    "\u00b4": "\u0301", "\u02ca": "\u0301", "`": "\u0300", "\u02cb": "\u0300",
    "\u02c6": "\u0302", "\u02dc": "\u0303", "\u00a8": "\u0308", "\u00b8": "\u0327",
    "\u02c7": "\u030c", "\u02d8": "\u0306", "\u00af": "\u0304", "\u02d9": "\u0307",
    "\u02da": "\u030a", "\u02dd": "\u030b", "\u02db": "\u0328",
}

# Characters that carry no text, and what each becomes. A zero-width space or
# byte-order mark splits a word the page shows whole ("ACKNO\u200bWLE\u200bDGE"),
# and U+FFFD is a glyph pymupdf could not decode, 415 of them in one document. Two
# producers repurpose a letter: Calibri's "ti" ligature extracts as U+019F
# ("solu\u019fons"), and one agency layout draws its dashes as U+0154 ("Service
# Documentation \u0154 Information"). GLUE (U+2060) is NOT in this table: it is
# invisible too, but `resolve_glue` still has to read it.
INVISIBLE = {0x200B: None, 0xFEFF: None, 0xFFFD: None, 0x019F: "ti", 0x0154: "\u2013"}

# The C0 control codes, 13,788 of them in 146 documents, are a subset font's own
# glyph numbers and mean what that font says, so two readings are needed. Between
# two letters one is a LIGATURE: one guideline's flowchart reads "selec\x02ng" and
# "a\x05er" for "selecting" and "after", and `resolve_ligatures` asks the document
# which ligature it is. Anywhere else it is a MARK and becomes a bullet: a
# Wingdings list bullet opening a line ("\x01 Quels sont..."), the symbols in the
# same guideline's interaction table that say which pollutant a filter removes
# ("CYP2C19 \x02 \x02 \x02"), InDesign's indent-to-here in front of a heading
# ("1. \x07FORMALISER"). Spacing them all, the first version of format 21, emptied
# that table (the CYP450 question fell out of the top 50) and ran lists together;
# a bullet is one token to the encoder, as the raw code was, and keeps both. 0x0b,
# 0x0c and 0x1c to 0x1f are whitespace to `str.isspace`, so they are left to it.
CONTROL = "\x00-\x08\x0e-\x1b"
CONTROL_MARK = re.compile(rf"(?<![^\W\d_])[{CONTROL}]+|[{CONTROL}]+(?![^\W\d_])")
LIGATURES = ("ti", "ft", "fi", "fl", "ff", "tt", "ffi", "ffl", "th")
LIGATURED_WORD = re.compile(rf"\w+(?:[{CONTROL}]\w+)+")

# Where a sentence ends inside a row. A row is a visual LINE, and a sentence almost
# never ends at one: that is why "rewind to the last sentence end" could so rarely
# find one, and why 46% of chunks used to stop mid-sentence. Splitting rows here
# gives the packer a boundary at every sentence instead of at every line.
SENTENCE_SPLIT = re.compile(r"(?<=[.!?\u2026])\s+(?=[\u00abA-Z\u00c0-\u00d6\u00d8-\u00de\"\u201c(\[])")
# What looks like a sentence end and is not. Abbreviations first, in both languages
# of the corpus, then an initial ("S. H. Kennedy"), then a numbered heading or list
# item ("3.3.3.", "2."), which is the commonest false positive in a guideline.
ABBREVIATIONS = ("cf", "al", "etc", "ex", "p", "pp", "fig", "tab", "vol", "no", "nb",
                 "dr", "pr", "mme", "mlle", "mm", "m", "vs", "ca", "env", "art", "ed",
                 "\u00e9d", "ref", "r\u00e9f", "chap", "min", "max", "moy", "i.e", "e.g",
                 "approx", "st", "ste", "inc", "ltd", "co", "univ", "dept")
ABBREVIATION_TAIL = re.compile(
    r"(?:^|[\s(\[])(?:" + "|".join(ABBREVIATIONS) + r")\.$", re.I)
INITIAL_TAIL = re.compile(r"(?:^|[\s(\[])[A-Z\u00c0-\u00de]\.$")
# A numbered heading ("3.3.3.", "2.", "VII.2.3.") is not a sentence end. A YEAR is:
# "publié en 2016. La suite" has to split, so the test asks for an internal dot, or
# for the whole piece to be nothing but the number. A dot is allowed before the digits
# so that a roman-numbered section ("VII.2.3.") is caught too: some agency documents number
# their parts that way and a chunk was ending on the number with its title in the next
# one.
NUMBER_TAIL = re.compile(r"(?:^|[\s(\[.])\d+(?:\.\d+)+\.$|^\s*\d+\.$"
                         r"|^\s*[IVXLCDM]+(?:\.\d+)*\.$")

# A word cut in half by a line break: the left row ends with the hyphen, the right row
# opens with the rest. Only a lowercase continuation counts, because an uppercase one
# is a proper noun or the start of something else ("Alzheimer- Type", "Score- 3") and
# joining those invents a word.
HYPHEN_TAIL = re.compile(r"(\w{2,})[-\u2010\u2011]$")
# The same break, but inside one row's text rather than between two rows.
INSIDE_BREAK = re.compile(r"(\w{2,})[-\u2010\u2011]\s+([a-zà-öø-ÿ]\w+)")
HYPHEN_HEAD = re.compile(r"^([a-zà-öø-ÿ]\w+)(\S*)(\s.*|)$", re.DOTALL)
# Three groups, not two: the word that finishes the broken one, whatever is stuck
# to it up to the first space, and the rest of the row. The second group travels
# with the word ("condi-" + "tions," has to end up as "conditions,") while the
# attestation count below must not see it. Requiring whitespace directly after
# the word, which is what this rule used to do, refused every break whose second
# half happens to end a clause, and that is where the line breaks most often.
# Taking the WHOLE token rather than only the punctuation after it matters as
# much: a superscript citation sticks to the word with nothing between them
# ("adher-" + "ence.92"), and so does a second hyphen inside a compound
# ("atten-" + "tion-deficit/hyperactivity"). Stopping at the punctuation left
# both of those broken in the chunk the reader is shown.


def clean_text(raw: str) -> str:
    """Normalise a line of extracted PDF text.

    Removes private-use-area characters (Symbol and Wingdings bullets, which
    extract as meaningless glyphs like U+F0B7), normalises Unicode to NFKC so
    ligatures and non-breaking spaces do not survive into the embedding, and
    collapses whitespace.

    Parameters
    ----------
    raw : str
        Text as concatenated from a line's spans.

    Returns
    -------
    str
        Cleaned text, possibly empty.
    """
    text = "".join(ch for ch in raw if not (0xE000 <= ord(ch) <= 0xF8FF)).translate(INVISIBLE)
    # A control code between two letters is left for `resolve_ligatures`.
    text = CONTROL_MARK.sub(" \u2022 ", text)
    # Some producers emit an accent as its own glyph, and the extractor then puts a
    # space in front of it: "Mu \u0308ller", "Vale \u0301rie" (this is real, from a
    # guideline's title page). NFKC cannot compose those, because composition needs
    # the mark to follow its base letter directly, so the space goes first and the
    # normalisation below then produces "Müller". Left alone it survives into the
    # embedding as two junk tokens and into the result snippet as visible mojibake.
    text = LEADING_MARK.sub("- ", text)
    text = DOTLESS_BASE.sub(lambda m: ("i" if m.group(0)[0] == "\u0131" else "j") + m.group(1), text)
    text = ORPHANED_MARK.sub(r"\1\2", text)
    text = unicodedata.normalize("NFKC", text)
    # A soft hyphen inside a line is invisible and means nothing to a reader, so it
    # goes and the word closes up. One at the END of a line is the typesetter
    # breaking a word across two lines, which is the same thing as a visible hyphen
    # there and is what `dehyphenate` exists to repair; dropping it silently left
    # "hy" and "pomanic" as two words in the text that gets embedded. It is promoted
    # to a real hyphen here rather than handled there, so the repair has one rule.
    # A table cell's own line breaks reach this function as spaces, so a soft hyphen
    # followed by one is a line end too: one guideline's cells came out "infrastruc ture".
    text = re.sub("\u00ad+(?=\\s|$)", "-", text.rstrip())
    text = text.replace("\u00ad", "")
    text = re.sub(r"\s+", " ", text).strip()
    # A row with no letter or digit carries no meaning for a search: it is a
    # table rule, a row of dots in a table of contents, or an OCR artefact.
    # Dropping it here keeps both the embedding and the highlight boxes clean.
    return text if any(ch.isalnum() for ch in text) else ""


def glue_for(first: str, second: str, *, attested: str,
             decided: dict[tuple[str, str], str]) -> str:
    """Whether a word broken across a line break closes up or keeps its hyphen.

    Parameters
    ----------
    first, second
        The two halves, as they are printed.
    attested
        The document's own lowercased text, snapshotted before any joining.
    decided
        The answers already worked out for this document, extended in place.

    Returns
    -------
    str
        `"-"` or `""`, to put between the halves.

    Notes
    -----
    The document is asked rather than a dictionary: whichever form it uses
    elsewhere in its own text wins. See `dehyphenate` for what that answers.

    The cache is the whole point of the signature. Each answer costs two scans of
    the document's entire text, `dehyphenate` asks 46685 times over the corpus,
    and the pairs repeat heavily inside one document ("théra-peutique" is broken
    at the same place on every second page). The answer cannot change within a
    document, because `attested` is a snapshot.
    """
    key = (first.lower(), second.lower())
    glue = decided.get(key)
    if glue is None:
        glue = "-" if attested.count(f"{key[0]}-{key[1]}") > \
            attested.count(f"{key[0]}{key[1]}") else ""
        decided[key] = glue
    return glue


GLUED_WORD = re.compile(rf"\w+(?:{GLUE}\w+)+")


def resolve_ligatures(rows: list[Row]) -> int:
    """Read every control code left between two letters as the ligature it draws.

    Each code is decided once for the whole document, by vote: every word it sits
    in tries each of LIGATURES, and a reading that makes a word the document uses
    elsewhere scores that word. "selec\x02ng", "an\x02depressant" and
    "medica\x02on" all vote "ti" for 0x02, so "ini\x02ate" becomes "initiate"
    even if "initiate" is written nowhere else. A code nothing votes for becomes
    a space, which is what the page shows where the reader cannot see the glyph
    either.

    Parameters
    ----------
    rows : list of Row
        All kept rows of one document. Mutated in place: no control code
        survives outside whitespace.

    Returns
    -------
    int
        How many codes were read as a ligature.
    """
    text = " ".join(row.text for row in rows)
    if not re.search(f"[{CONTROL}]", text):
        return 0
    attested: dict[str, int] = {}
    for word in re.findall(r"\w+", LIGATURED_WORD.sub(" ", text).lower()):
        attested[word] = attested.get(word, 0) + 1
    votes: dict[str, dict[str, int]] = {}
    for word in LIGATURED_WORD.findall(text.lower()):
        for code in set(re.findall(f"[{CONTROL}]", word)):
            for ligature in LIGATURES:
                reading = re.sub(f"[{CONTROL}]", lambda m: ligature if m.group(0) == code else "", word)
                if reading in attested:
                    tally = votes.setdefault(code, {})
                    tally[ligature] = tally.get(ligature, 0) + 1
    chosen = {code: max(tally, key=tally.get) for code, tally in votes.items()}
    read = 0

    def replace(match: re.Match) -> str:
        nonlocal read
        ligature = chosen.get(match.group(0))
        read += ligature is not None
        return ligature if ligature is not None else " "

    for row in rows:
        row.text = re.sub(f"[{CONTROL}]", replace, row.text)
    return read


def resolve_glue(rows: list[Row]) -> int:
    """Decide every space `glued` doubted: one word, or two after all.

    A pair is joined when the document uses the joined word elsewhere ("ar⁠e" and
    "are"), kept apart when both halves are words it uses and the joined form is
    not ("of⁠Action": "of", "action", never "ofaction"), and joined otherwise
    ("reinit⁠iate", "benefi⁠ts": at least one half is no word of this document).
    Only words touching no GLUE count as the document's usage. Otherwise the fault
    vouches for itself: every "fi" in one handbook is followed by a false space,
    so "fi" and "nal" would each look like a word.

    Parameters
    ----------
    rows : list of Row
        All kept rows of one document. Mutated in place: no GLUE survives.

    Returns
    -------
    int
        How many doubted spaces were closed up.
    """
    usage = GLUED_WORD.sub(" ", " ".join(row.text for row in rows).lower())
    attested: dict[str, int] = {}
    for word in re.findall(r"\w+", usage):
        attested[word] = attested.get(word, 0) + 1
    joined = 0

    def decide(match: re.Match) -> str:
        nonlocal joined
        first, *rest = match.group(0).split(GLUE)
        text = first
        for part in rest:
            left = re.search(r"\w+$", text).group(0).lower()
            whole = left + part.lower()
            if whole in attested or not (left in attested and part.lower() in attested):
                text += part
                joined += 1
            else:
                text += " " + part
        return text

    for row in rows:
        if GLUE in row.text:
            row.text = GLUED_WORD.sub(decide, row.text).replace(GLUE, " ")
    return joined


def dehyphenate(rows: list[Row]) -> int:
    """Rejoin words that justification split across two rows.

    A third of the corpus's chunks carry at least one of these, and they reach both
    the embedding and the reader: "prise en charge théra- peutique" is what the
    result snippet says today, and the tokenizer sees "théra", "-", "peutique"
    instead of one word.

    Whether the two halves close up ("per- sonnes" -> "personnes") or keep the
    hyphen ("médico- sociaux" -> "médico-sociaux") is not decidable from the pair
    alone, so the document is asked: whichever form it uses elsewhere in its own
    text wins. Measured over the 483-document corpus, that answers 78% of the
    46,685 breaks outright (69% closed, 9% hyphenated); the rest are hyphenated
    nowhere and closed nowhere, and are overwhelmingly ordinary soft breaks
    ("neurobeha- vioral", "Uni- versity", "frus- trating"), so the default is to
    close them up.

    The attestation text is snapshotted before any joining, so the result does not
    depend on the order the pairs are visited.

    Rows are joined only within a page: across a page break the two halves are not
    neighbours on paper, whatever the reading order says, and a dropped header may
    have stood between them.

    The right row keeps its box and loses only its first word, so provenance is
    untouched: a highlight still covers both lines, because both rows are still in
    the chunk.

    Parameters
    ----------
    rows : list of Row
        All kept rows of one document, in reading order. Mutated in place.

    Returns
    -------
    int
        How many words were rejoined.
    """
    attested = " ".join(row.text for row in rows).lower()
    decided: dict[tuple[str, str], str] = {}
    joined = 0
    # A break can also sit INSIDE a row, and two things put it there: a table cell
    # serialised out of several visual lines ("se rendre a un rendez- vous"), and two
    # lines merged into one row by _same_row. Same question, same answer, so the same
    # attestation text decides it before the row-to-row pass runs.
    for row in rows:
        row.text, inside = _join_inside(row.text, attested=attested, decided=decided)
        joined += inside
    for left, right in zip(rows, rows[1:]):
        if left.page != right.page:
            continue
        tail = HYPHEN_TAIL.search(left.text)
        head = HYPHEN_HEAD.match(right.text)
        if not (tail and head):
            continue
        rest = head.group(3).lstrip()
        # A row that is nothing but the second half of a word would be left with no
        # text at all, and an empty row is a box with nothing to highlight. Rare
        # enough (a one-word last line) that leaving the pair alone costs nothing.
        # Punctuation alone is text, though, and refusing to join over it is what
        # left "activités signifi- catives ;" in the opioid argumentaire: the right
        # row keeps its ";" and its box, and the word is whole again.
        if not rest:
            continue
        first, second = tail.group(1), head.group(1)
        glue = glue_for(first, second, attested=attested, decided=decided)
        left.text = left.text[: tail.start(1)] + first + glue + second + head.group(2)
        right.text = rest
        joined += 1
    return joined


def looks_french(text: str) -> bool:
    """Whether a page reads as French rather than English.

    Parameters
    ----------
    text : str
        The page's text.

    Returns
    -------
    bool
        True for French, which is also the answer when neither language shows,
        because the corpus is mostly French and a table of bare numbers has to be
        described in something.
    """
    words = re.findall(r"[a-zà-öø-ÿ']+", text.lower())
    french = sum(1 for word in words if word in FRENCH_STOPWORDS)
    english = sum(1 for word in words if word in ENGLISH_STOPWORDS)
    return french >= english


def split_sentences(rows: list[Row]) -> tuple[list[Row], int]:
    """Cut every row at the sentence ends inside it.

    Parameters
    ----------
    rows : list of Row
        All rows of one document, in reading order.

    Returns
    -------
    (list of Row, int)
        The rows, with each multi-sentence row replaced by one row per sentence,
        and how many extra rows that produced.

    Notes
    -----
    A row is a visual LINE. A sentence ends where the typesetter's line happens to
    run out only by accident, so the packer's "rewind to the last sentence end"
    almost never had one to rewind to and the token budget cut wherever it landed:
    46% of chunks ended mid-sentence, including "Les psychotropes sont" and "peuvent
    être". Making the row the unit of a SENTENCE rather than of a line gives every
    downstream rule (break strengths, the rewind, the overlap carry) a boundary a
    reader would recognise, without changing any of them.

    The pieces share the original row's bbox, page, size and direction, which is the
    one thing given up here: a chunk that ends mid-line is highlighted to the end of
    that line in the viewer. `Chunk.boxes_by_page` de-duplicates, so the cost is a
    slightly generous highlight and never a duplicated or missing box.

    What is NOT a sentence end: an abbreviation, an initial in a name, and a
    numbered heading ("3.3.3."), which is the commonest false positive in a
    guideline and would otherwise turn every section number into its own row.
    """
    out: list[Row] = []
    added = 0
    for row in rows:
        pieces = _sentence_pieces(row.text)
        if len(pieces) == 1:
            out.append(row)
            continue
        added += len(pieces) - 1
        for piece in pieces:
            out.append(Row(page=row.page, bbox=row.bbox, text=piece, size=row.size,
                           vertical=row.vertical, right_edge=row.right_edge))
    return out, added


def _sentence_pieces(text: str) -> list[str]:
    """Split one row's text into sentences, keeping the punctuation with its own."""
    pieces: list[str] = []
    start = 0
    for match in SENTENCE_SPLIT.finditer(text):
        head = text[start:match.start()]
        if (ABBREVIATION_TAIL.search(head) or INITIAL_TAIL.search(head)
                or NUMBER_TAIL.search(head)):
            continue
        pieces.append(head)
        start = match.end()
    pieces.append(text[start:])
    return [piece for piece in (p.strip() for p in pieces) if piece]


def ends_sentence(text: str) -> bool:
    """Say whether a row's text finishes a sentence.

    Parameters
    ----------
    text : str
        One row's text, as extracted.

    Returns
    -------
    bool
        True when the text ends on sentence-final punctuation that is not an
        abbreviation, an initial or a numbered heading.

    Notes
    -----
    The three exclusions are the same ones `split_sentences` applies, and for the
    same reason: "cf.", "J." and "VII.2.3." all end on a full stop and none of them
    ends a sentence. Sharing the patterns keeps the packer from cutting where the
    splitter refused to split, which is how a chunk used to end on a section number
    with its title in the next one.
    """
    stripped = text.rstrip()
    if not SENTENCE_FINAL.search(stripped):
        return False
    return not (ABBREVIATION_TAIL.search(stripped) or INITIAL_TAIL.search(stripped)
                or NUMBER_TAIL.search(stripped))


def _join_inside(text: str, *, attested: str,
                 decided: dict[tuple[str, str], str]) -> tuple[str, int]:
    """Rejoin words broken across a line break that ended up inside one row.

    Parameters
    ----------
    text : str
        The row's text.
    attested : str
        The document's own lowercased text, used to decide whether the pair keeps
        its hyphen, exactly as `dehyphenate` decides it between rows.
    decided : dict
        The document's answers so far, shared with `dehyphenate` so that a pair
        broken both inside a row and between two rows is counted once.

    Returns
    -------
    tuple of (str, int)
        The repaired text and how many words were rejoined.

    Notes
    -----
    The head must start with a lowercase letter, which is what keeps "Alzheimer-
    Type" and "5- 10" out: a capital after the break is another word, and a digit is
    a range.
    """
    joined = 0

    def repair(match: re.Match) -> str:
        nonlocal joined
        joined += 1
        first, second = match.group(1), match.group(2)
        return first + glue_for(first, second, attested=attested,
                                decided=decided) + second

    return INSIDE_BREAK.sub(repair, text), joined
