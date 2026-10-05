"""The page service: it must cut one page, and it must not hang.

`server/pages.py` is the only way a reader reaches a restricted document, since
`stage.py` keeps those files out of the web root entirely. Two of its failures
are invisible by reading it, and both happened:

  - it deadlocked on the first request, because `cut_page` held a non-reentrant
    lock and called `open_document`, which took the same lock. The symptom is
    not an error: the thread waits forever and the reader watches a page that
    never arrives. `test_a_cut_does_not_deadlock` is that bug, and it is written
    with a timeout rather than a plain call so a regression fails the suite
    instead of hanging it.
  - it refused every document, because the containment check resolved the path
    before testing it and `dist/restricted/` is staged as symlinks into
    `data/GUIDELINES/`. That one would have passed on the VPS, where deploy.sh
    rsyncs with `-L` and the entries are real files, and failed on the machine
    it was written on, which is the worst way round for a check nobody rereads.

Everything here builds its own two-page PDF in a tmp_path, so the suite still
needs neither the corpus nor a built index.

Written by Claude Code (Opus 5).
"""

from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path

import pymupdf
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def pages(tmp_path, monkeypatch):
    """server/pages.py, pointed at a two-page document staged as a symlink."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    source = corpus / "book.pdf"
    doc = pymupdf.open()
    for text in ("PAGE ONE", "PAGE TWO"):
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(source)
    doc.close()

    # Staged the way stage.py stages it: a symlink pointing OUT of the mount.
    mount = tmp_path / "restricted"
    mount.mkdir()
    (mount / "book.pdf").symlink_to(source)

    spec = importlib.util.spec_from_file_location("pages_service", ROOT / "server" / "pages.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["pages_service"] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "PAGES_DIR", mount)
    module._open_docs.clear()
    return module


def test_a_cut_returns_exactly_one_page(pages):
    """One page in, one page out, with that page's own text and no other."""
    body = pages.cut_page("book.pdf", 2)
    cut = pymupdf.open(stream=body, filetype="pdf")
    assert cut.page_count == 1
    assert "PAGE TWO" in cut[0].get_text()
    assert "PAGE ONE" not in cut[0].get_text()


def test_a_cut_does_not_deadlock(pages):
    """The first request must return. It once waited forever, holding the socket.

    Run in a thread with a join timeout: a reentrancy regression would otherwise
    hang the whole suite instead of failing this one test.
    """
    done: list[bytes] = []
    worker = threading.Thread(target=lambda: done.append(pages.cut_page("book.pdf", 1)),
                              daemon=True)
    worker.start()
    worker.join(timeout=20)
    assert not worker.is_alive(), "cut_page deadlocked: _lock must be reentrant"
    assert done and done[0].startswith(b"%PDF")


def test_a_symlinked_document_is_readable(pages):
    """dist/restricted/ is symlinks on the dev machine and real files on the VPS."""
    assert (pages.PAGES_DIR / "book.pdf").is_symlink()
    assert pages.open_document("book.pdf").page_count == 2


@pytest.mark.parametrize("name", ["../secret.pdf", "sub/book.pdf", "..", "/etc/passwd"])
def test_a_name_that_escapes_the_mount_is_refused(pages, name):
    """The one place a name becomes a path."""
    with pytest.raises(FileNotFoundError):
        pages.open_document(name)


def test_the_cache_is_bounded_and_closes_what_it_evicts(pages, tmp_path, monkeypatch):
    """Each entry holds a file handle and mupdf's structures, on a box with little RAM."""
    monkeypatch.setattr(pages, "CACHE_SIZE", 2)
    for i in range(4):
        doc = pymupdf.open()
        doc.new_page()
        doc.save(tmp_path / "corpus" / f"d{i}.pdf")
        doc.close()
        (pages.PAGES_DIR / f"d{i}.pdf").symlink_to(tmp_path / "corpus" / f"d{i}.pdf")
        pages.open_document(f"d{i}.pdf")
    assert len(pages._open_docs) == 2
    assert list(pages._open_docs) == ["d2.pdf", "d3.pdf"], "least recently used goes first"


def test_the_catalog_holds_restricted_documents_only(pages, tmp_path):
    """An open document must be fetched from /pdf/; this service speaks about one tier."""
    index = tmp_path / "index"
    index.mkdir()
    (index / "meta.json").write_text(
        '{"documents": ['
        '{"id": 1, "file": "book.pdf", "access": "restricted", "pages": 2},'
        '{"id": 2, "file": "open.pdf", "access": "open", "pages": 9},'
        '{"id": 3, "file": "blank.pdf", "pages": 4}]}', encoding="utf-8")
    catalog = pages.load_catalog(index)
    assert set(catalog) == {1}
    assert catalog[1] == {"file": "book.pdf", "pages": 2}
