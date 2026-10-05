"""The judge's verdict cache: how a passage is named, and how the file is read.

`judge_rerank.py` writes `data/grid/judge-scores.jsonl`, one verdict per line
`{"q": <query hash>, "d": <passage hash>, "n": <token budget>, "s": <score>}`,
and `grid_report.py` reads it back. The two used to carry their own `sha` and
their own reader, which is how a cache key drifts: a report that hashes a
passage one way finds none of the verdicts written the other way, and prints an
empty column rather than an error. `scripts/dump_candidates.mjs` computes the
same hash in JavaScript (`text_sha`), and `tests/test_grid.py` pins the two
together.

A JSONL rather than one JSON object because it is appended to while a run is in
progress: a process killed mid-write leaves one unparseable last line, which is
recoverable, where a truncated JSON object is the whole cache lost. A later line
wins, so re-judging a pair is a matter of appending rather than of editing.

Module imported, not run. Standard library only.

Written by Claude Code (Opus 5.5).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha(text: str) -> str:
    """The 16 hex digits the dumps and the cache identify a query or passage by.

    Parameters
    ----------
    text
        The query or the passage, verbatim.

    Returns
    -------
    str
        The first 16 hex digits of its UTF-8 sha256.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def read_cache(path: Path) -> tuple[dict[tuple[str, str, int], float], int]:
    """Every verdict on disk, keyed by (query hash, passage hash, token budget).

    Parameters
    ----------
    path
        The JSONL cache. A missing file is an empty cache.

    Returns
    -------
    scores : dict
        (q, d, n) to score, the last line for a key winning.
    skipped : int
        Unparseable lines (an interrupted write), for the caller to report.
    """
    scores: dict[tuple[str, str, int], float] = {}
    skipped = 0
    if not path.exists():
        return scores, skipped
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        scores[(row["q"], row["d"], row["n"])] = row["s"]
    return scores, skipped
