"""Write a file so that a reader sees the old version or the new one, never half.

Every build artefact here is trusted by the next step on sight: `chunk.py`'s
cache believes a chunk file whose `src_hash` matches, `embed.py`'s believes an
`.npz`, `manifest.py` re-reads the hand-curated `MANIFEST.tsv` as its own input.
Writing any of them in place means a Ctrl-C, a full disk or a crash mid-write
leaves a truncated file where a good one was. The docs invite exactly that
("Ctrl-C at any point, re-run to continue"), and a truncated `.npz` used to
crash every later run with `BadZipFile` instead of rebuilding one document.

So every writer goes through one helper: write a sibling temporary file, flush
it to disk, then `os.replace` it over the target, which is atomic on POSIX. The
temporary name starts with a dot and ends in `.tmp`, so none of the globs the
scripts read their inputs with (`*.json`, `*.npz`, `*.pdf`) can pick one up,
and it carries the pid so two processes writing the same target cannot clobber
each other's half-written copy.

Module imported, not run. Standard library only, so any script can use it.

Written by Claude Code (Opus 5.5).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any


@contextmanager
def open_atomic(path: Path, mode: str = "w", **kwargs: Any) -> Iterator[IO[Any]]:
    """Open a temporary sibling of `path` and move it over `path` on success.

    Parameters
    ----------
    path
        The file to (re)write.
    mode
        `"w"` (text, the default) or `"wb"`.
    **kwargs
        Passed to `open`, e.g. `encoding="utf-8", newline=""`.

    Yields
    ------
    IO
        The open temporary file. Write to it as to `path`.

    Notes
    -----
    On an exception the temporary file is removed and `path` is left exactly as
    it was, which is the whole point: the previous good version survives.
    """
    if mode not in ("w", "wb"):
        raise ValueError(f"open_atomic writes a whole file, mode {mode!r} is not one")
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open(mode, **kwargs) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_text_atomic(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """`Path.write_text`, atomically. See :func:`open_atomic`.

    Parameters
    ----------
    path
        The file to (re)write.
    text
        Its whole new content.
    encoding
        Text encoding.
    """
    with open_atomic(path, "w", encoding=encoding) as handle:
        handle.write(text)
