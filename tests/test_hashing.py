"""scripts/lib/hashing.py: the one file sha256 every cache keys on.

Written by Claude Code (Opus 5.5).
"""

import hashlib
import os

from conftest import load_lib


def test_it_is_the_plain_sha256_of_the_bytes(tmp_path):
    """Every existing cache key was made with sha256(read_bytes()); the helper must
    produce the same hex, or switching to it would invalidate all of them."""
    hashing = load_lib("hashing")
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF-1.7\n" + bytes(range(256)) * 100)
    assert hashing.file_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_rewritten_file_is_hashed_again(tmp_path):
    """The memo is keyed on size and mtime, so a file replaced mid-run is re-read."""
    hashing = load_lib("hashing")
    path = tmp_path / "a.pdf"
    path.write_bytes(b"one")
    first = hashing.file_sha256(path)
    path.write_bytes(b"two")
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert hashing.file_sha256(path) != first
    assert hashing.file_sha256(path) == hashlib.sha256(b"two").hexdigest()
