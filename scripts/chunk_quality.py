# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "loguru"]
# ///
"""Count what is wrong with a set of chunks, so a change to chunk.py can be judged.

Chunking has no right answer to assert, which is why it kept being tuned by reading
a few pages and hoping. This gives it a number instead: a handful of defects that
are checkable without knowing what the page said, counted over the whole corpus, so
two chunkings can be compared in one line each.

    uv run scripts/chunk_quality.py                       # data/chunks
    uv run scripts/chunk_quality.py --chunks data/chunks-b --baseline data/chunks

The defects, all of them things a reader would notice:

`cut_mid_sentence`   the chunk stops with no terminal punctuation, and its last
                     line is neither a heading (short, or all caps, or ending in a
                     colon) nor a marked list item (a number, a bullet or a dash,
                     up to one printed line). The encoder gets half a claim and the
                     reader gets half a sentence.
`starts_mid_sentence`the chunk opens on a lowercase word or a closing bracket, so
                     it begins in the middle of something it does not contain.
`split_word`         a word is cut across a hyphen inside the text ("thera- peutique"),
                     which is the extractor's line break surviving into the chunk.
`mangled_accent`     a combining accent stands alone after a space ("Mu ̈ller"), which
                     is what some PDF producers emit and what tokenises as garbage.
`control_space`      a non-breaking space, a narrow space or a tab: invisible to the
                     eye, a different token to the model.
`front_matter`       the chunk holds a journal title page's furniture (a DOI, a
                     "Reprints and permission" line, a corresponding-author email)
                     mixed with more than a little running text.
`tiny`               under 40 tokens and not the last chunk of its document: usually
                     a fragment orphaned by a bad break.

Each is a count of chunks, plus a percentage. Exit status is always 0: this is a
measurement, not a gate, because none of these can be driven to zero on a real
corpus and a gate that fails on the 400th document teaches nothing.

Written by Claude Code (Opus 5).
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import click
from loguru import logger
from lib.corpus import in_corpus

# Sentence-final punctuation. Note what is NOT here: ";" and "," end clauses, not
# sentences, and a chunk stopping on one is exactly the defect being counted.
TERMINAL = tuple(".!?…»”\")]")
# A chunk opening on a lowercase letter or on the closing half of a pair continues
# something the reader has not been given. Accented lowercase included, since half
# the corpus is French.
STARTS_MID = re.compile(r"^[a-zà-öø-ÿ),;:.…]")
# Headings and list items legitimately end without punctuation. They are split into
# two patterns because they are not the same length class, which the single 60-char
# cap got wrong: a heading is short by nature, while a MARKED list item is as long as
# whatever it lists. The case that exposed it, on 2026-09-26 when the target moved to
# 256 tokens and a journal cover started splitting four ways: "7 Department of
# Statistics, University of Calgary, Calgary, Alberta" is an affiliation list item,
# 67 characters, and was being counted as a sentence cut in half.
MARKED_ITEM = re.compile(r"^(?:\d+[\d.]*\s|[•▪‣·*o►▶➢]\s|[-–—]\s)")
HEADING_LIKE = re.compile(r"^(?:[A-Z][A-Z\s]{3,}$|.{0,40}:$)")
HEADING_MAX_CHARS = 60
# One printed line of a two-column journal page. Past this a "list item" is a
# paragraph that happens to start with a number, and a cut in it is a real one.
MARKED_ITEM_MAX_CHARS = 120
# "thera- peutique": a hyphen between two lowercase runs, inside the text. The
# conjunctions are excluded because "first- and second-line", "short- and
# longer-term", "court- et moyen-terme" are written that way on purpose: the hyphen
# hangs waiting for the second half of a compound, and nothing is broken.
SPLIT_WORD = re.compile(r"\w[-‐‑]\s+(?!(?:and|or|et|ou|to|nor)\b)[a-zà-öø-ÿ]")
# A combining mark that has lost the letter it belongs to: "Mu ̈ller", "Vale ́rie".
MANGLED_ACCENT = re.compile(r"\s[̀-ͯ]")
CONTROL_SPACE = re.compile(r"[    \t​]")
# Journal front matter. Each alone is weak, so `front_matter` needs two of them.
FRONT_MATTER = (
    re.compile(r"\bdoi\s*[:.]?\s*10\.\d{4}", re.I),
    re.compile(r"reprints and permission", re.I),
    re.compile(r"corresponding author", re.I),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    re.compile(r"\bISSN\b|\bsagepub\b|journals\.sagepub", re.I),
    re.compile(r"©|\bThe Author\(s\)\b", re.I),
)
FRONT_MATTER_MIN_HITS = 2
TINY_TOKENS = 40


def defects(chunk: dict, *, is_last: bool) -> set[str]:
    """Name every defect one chunk carries.

    Parameters
    ----------
    chunk : dict
        A record from a chunk file: needs `text` and `n_tokens`.
    is_last : bool
        Whether this is its document's last chunk, which is allowed to be short and
        to stop wherever the document stopped.

    Returns
    -------
    set of str
        Defect names, from the list in the module docstring. Empty for a clean chunk.
    """
    text = chunk["text"]
    found: set[str] = set()
    stripped = text.rstrip()
    if stripped and not stripped.endswith(TERMINAL) and not is_last:
        last = stripped.rsplit("\n", 1)[-1].strip()
        if MARKED_ITEM.match(last):
            forgiven = len(last) <= MARKED_ITEM_MAX_CHARS
        else:
            forgiven = len(last) <= HEADING_MAX_CHARS and bool(HEADING_LIKE.match(last))
        if not forgiven:
            found.add("cut_mid_sentence")
    if STARTS_MID.match(text):
        found.add("starts_mid_sentence")
    if SPLIT_WORD.search(text):
        found.add("split_word")
    if MANGLED_ACCENT.search(text):
        found.add("mangled_accent")
    if CONTROL_SPACE.search(text):
        found.add("control_space")
    if sum(1 for pattern in FRONT_MATTER if pattern.search(text)) >= FRONT_MATTER_MIN_HITS:
        found.add("front_matter")
    if chunk["n_tokens"] < TINY_TOKENS and not is_last:
        found.add("tiny")
    return found


def survey(chunks_dir: Path) -> dict[str, object]:
    """Count defects and sizes over every chunk file in a directory."""
    counts: dict[str, int] = {}
    total = 0
    tokens: list[int] = []
    documents = 0
    for path in sorted(chunks_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        documents += 1
        chunks = payload["chunks"]
        for position, chunk in enumerate(chunks):
            total += 1
            tokens.append(chunk["n_tokens"])
            for name in defects(chunk, is_last=position == len(chunks) - 1):
                counts[name] = counts.get(name, 0) + 1
    tokens.sort()
    return {"documents": documents, "chunks": total, "counts": counts,
            "median_tokens": tokens[len(tokens) // 2] if tokens else 0,
            "p05_tokens": tokens[len(tokens) // 20] if tokens else 0,
            "p95_tokens": tokens[len(tokens) * 19 // 20] if tokens else 0}


def report(name: str, result: dict[str, object], baseline: dict[str, object] | None) -> None:
    """Print one survey, with the change against a baseline when there is one."""
    total = result["chunks"] or 1
    logger.info(f"{name}: {result['documents']} documents, {result['chunks']} chunks, "
                f"tokens p05/median/p95 {result['p05_tokens']}/{result['median_tokens']}"
                f"/{result['p95_tokens']}")
    names = sorted(set(result["counts"]) | set((baseline or {}).get("counts", {})))
    for defect in names:
        count = result["counts"].get(defect, 0)
        line = f"  {defect:20s} {count:6d}  {100 * count / total:5.2f}%"
        if baseline:
            was = baseline["counts"].get(defect, 0)
            line += f"   (was {was}, {count - was:+d})"
        click.echo(line)


@click.command()
@click.option("--chunks", "chunks_dir", type=click.Path(exists=True, file_okay=False,
                                                        path_type=Path),
              **in_corpus("data/chunks"))
@click.option("--baseline", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=None, help="Another chunk directory to compare against.")
def main(chunks_dir: Path, baseline: Path | None) -> None:
    """Count the defects in a chunking, and compare two chunkings."""
    before = survey(baseline) if baseline else None
    if before:
        report(str(baseline), before, None)
    report(str(chunks_dir), survey(chunks_dir), before)


if __name__ == "__main__":
    main()
