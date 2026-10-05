"""`scripts/lib/atomic.py`: a reader sees the old file or the new one, never half.

Every artefact here is trusted on sight by the next run (a chunk file whose
`src_hash` matches, an `.npz`, the manifest that is also `manifest.py`'s input),
so a write interrupted halfway must leave the previous good file in place.

Written by Claude Code (Opus 5.5).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lib import atomic  # noqa: E402


def test_a_write_replaces_the_file_and_leaves_no_temporary(tmp_path):
    target = tmp_path / "a.json"
    target.write_text("old", encoding="utf-8")
    atomic.write_text_atomic(target, "new")
    assert target.read_text(encoding="utf-8") == "new"
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]


def test_an_interrupted_write_keeps_the_previous_file(tmp_path):
    target = tmp_path / "MANIFEST.tsv"
    target.write_text("curated", encoding="utf-8")
    with pytest.raises(KeyboardInterrupt):
        with atomic.open_atomic(target, encoding="utf-8") as handle:
            handle.write("half of the")
            raise KeyboardInterrupt
    assert target.read_text(encoding="utf-8") == "curated"
    assert [p.name for p in tmp_path.iterdir()] == ["MANIFEST.tsv"]


def test_the_temporary_name_escapes_the_input_globs(tmp_path):
    """chunk.py globs *.json, embed.py *.npz, ocr.py *.pdf: none may see a temporary."""
    seen = []
    with atomic.open_atomic(tmp_path / "doc.npz", "wb") as handle:
        handle.write(b"x")
        seen = [p.name for p in tmp_path.glob("*.npz")]
    assert seen == []


def test_a_truncated_bake_is_rebuilt_not_fatal(embed, tmp_path):
    """A bake killed mid-write before 2026-10-03 left a truncated archive, and
    `cached_digest` let its BadZipFile escape, crashing every later run."""
    import numpy as np
    whole = tmp_path / "whole.npz"
    np.savez_compressed(whole, vectors=np.zeros((50, 64)), src_hash=np.array("abc"),
                        chunks_hash=np.array("c"))
    cut = tmp_path / "cut.npz"
    cut.write_bytes(whole.read_bytes()[: whole.stat().st_size // 2])
    assert embed.cached_digest(cut) is None
