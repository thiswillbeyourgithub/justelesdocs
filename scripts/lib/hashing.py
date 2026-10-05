"""The sha256 of a file's bytes, computed once per process.

A document's sha256 is the identity every cache here keys on: chunk.py's layout
cache and its `src_hash`, ocr.py's record of which original an OCR copy came from,
figures.py's description sheets, check_served.py's restricted-file list. Each used
to spell `hashlib.sha256(path.read_bytes()).hexdigest()` itself, nine times over,
and chunk.py hashed the same 30 MB PDF up to three times per document.

One helper, streamed (`hashlib.file_digest`) rather than read whole, and memoised
on `(path, size, mtime_ns)` so a file hashed twice in one run is read once while a
file rewritten mid-run is hashed again. The memo is in-process only: a stat-based
cache that outlived the process would trust a file replaced with the same size
and mtime, and the hash is the thing every other cache trusts.

Module imported, not run. Standard library only.

Written by Claude Code (Opus 5.5).
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=4096)
def _digest(path: str, size: int, mtime_ns: int) -> str:
    """Hash one file. `size` and `mtime_ns` are only part of the memo key."""
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def file_sha256(path: Path) -> str:
    """Hex sha256 of a file's bytes.

    Parameters
    ----------
    path
        Any readable file.

    Returns
    -------
    str
        64 hex characters, the same as `hashlib.sha256(path.read_bytes()).hexdigest()`.
    """
    stat = Path(path).stat()
    return _digest(str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns)
