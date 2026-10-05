#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pymupdf==1.28.2"]
# ///
"""The page service: one page of a restricted document, cut out per request.

Why this exists. 53 documents entered the corpus on the condition that they are
served without download (`data/CORPUS_ADDITIONS.tsv`, the `access` column of
`data/MANIFEST.tsv`). The reader may see the page a search hit landed on, and
one either side; nobody may fetch the document. `stage.py` keeps those files out
of the web root entirely, in `dist/restricted/`, which ONLY this service mounts.
So there is no URL that returns a document, and the only way to see a page is to
ask for that page and get a one-page PDF back.

The cut is done per request rather than at build time because the alternative
costs disk this VPS does not have: per-page PDFs re-embed each page's fonts, and
pre-rendered images of all 15,703 restricted pages run to about a gigabyte.
mupdf opens a document lazily, so pulling page 500 out of a 1,275-page reference book
costs milliseconds and a bounded amount of memory, and nothing is stored.

pymupdf rather than a Node library, though the search service next door is Node:
it is the same library that produced the chunk geometry in the first place
(`scripts/chunk.py`), so the page numbering here and the highlight boxes in the
index cannot drift apart, and it keeps the search image free of node_modules,
which is a property that image deliberately has.

What this service will NOT do:

  - serve an open document: those are ordinary files under `/pdf/`, and having
    exactly one route to each document is what keeps the split checkable;
  - serve a page of anything it cannot find in the index's own document list;
  - return more than one page per request, whatever is asked for.

It does not enforce the reader's three-page window. That was a deliberate
decision (2026-09-24): enforcing it means signing the allowed range into the
search response, and the window is a courtesy to the reader rather than a
boundary. What this design buys is that the document is never addressable as a
whole; reassembling one means fetching its pages one at a time through the
edge's per-IP rate limit.

Environment:

  PAGES_PORT        port to listen on (default 8651)
  PAGES_DIR         the restricted documents (default /restricted)
  INDEX_DIR         the built index, read for its document list (default /index)
  PAGES_CONCURRENCY how many cuts may run at once (default 2)
  PAGES_CACHE       how many opened documents to keep (default 4)

Written by Claude Code (Opus 5).
"""

from __future__ import annotations

import http.server
import json
import os
import socketserver
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pymupdf

PORT = int(os.environ.get("PAGES_PORT", "8651"))
PAGES_DIR = Path(os.environ.get("PAGES_DIR", "/restricted"))
INDEX_DIR = Path(os.environ.get("INDEX_DIR", "/index"))
# Two at a time: a cut is short and CPU-bound, and this box runs two sites and a
# shared encoder. The queue behind it is the socket backlog, as for the ranker.
CONCURRENCY = int(os.environ.get("PAGES_CONCURRENCY", "2"))
CACHE_SIZE = int(os.environ.get("PAGES_CACHE", "4"))

_gate = threading.BoundedSemaphore(CONCURRENCY)
# REENTRANT on purpose: `cut_page` holds this lock for the whole cut and calls
# `open_document`, which takes it again to touch the cache. With a plain Lock
# that is a deadlock on the very first request, and a silent one: the thread
# waits forever, the socket stays open and the reader sees a page that never
# arrives. It held for 170 seconds in testing before the process was killed.
#
# One lock for every document, rather than one per document, so cuts are
# serialised: mupdf documents are not thread-safe, this box has two cores, and a
# cut is milliseconds. CONCURRENCY is therefore not parallelism, it is the depth
# of the queue allowed to form in front of the lock before requests are refused.
_lock = threading.RLock()
_open_docs: OrderedDict[str, pymupdf.Document] = OrderedDict()


def load_catalog(index_dir: Path) -> dict[int, dict]:
    """The documents this service may cut, by index id.

    Parameters
    ----------
    index_dir
        The built index, whose `meta.json` carries every document's id, filename,
        page count and access tier.

    Returns
    -------
    dict[int, dict]
        Id to `{"file": str, "pages": int}`, restricted documents only.

    Raises
    ------
    SystemExit
        If `meta.json` is absent. Starting without it would answer every request
        with a 404 that looks like a routing fault rather than a missing mount.
    """
    meta_path = index_dir / "meta.json"
    if not meta_path.exists():
        sys.exit(f"no index at {meta_path}: bind-mount the built index at {index_dir}")
    documents = json.loads(meta_path.read_text(encoding="utf-8")).get("documents", [])
    catalog: dict[int, dict] = {}
    for document in documents:
        if document.get("access") != "restricted":
            continue
        pages = document.get("pages")
        catalog[int(document["id"])] = {
            "file": document["file"],
            "pages": int(pages) if pages else 0,
        }
    return catalog


def open_document(name: str) -> pymupdf.Document:
    """Open a restricted document, keeping the last few open.

    Re-opening a 1,275-page reference book on every request is wasteful but not slow;
    what the cache actually protects is the page tree mupdf builds on first
    access, which is per document and worth keeping while a reader pages through
    one. The cache is bounded because each entry holds a file handle and mupdf's
    own structures, on a box with little RAM.

    Parameters
    ----------
    name
        Filename, already checked against the catalog.

    Returns
    -------
    pymupdf.Document
        An open document. Callers must hold `_lock` for the whole time they use
        it: mupdf documents are not thread-safe. The lock is reentrant, so
        calling this from a caller that already holds it is correct.

    Raises
    ------
    FileNotFoundError
        If the name does not resolve to a file inside the mount.
    """
    with _lock:
        if name in _open_docs:
            _open_docs.move_to_end(name)
            return _open_docs[name]
        # The containment check is on the NAME, not on where the name leads. It
        # rejects anything with a separator or a `..` in it, which is the only
        # way a corpus filename out of the index could address another
        # directory. Checking the resolved path instead would be wrong here:
        # stage.py stages this tree as symlinks into data/GUIDELINES, so every
        # entry resolves outside the mount on the development machine and the
        # service would refuse all 53. (deploy.sh rsyncs with -L, so on the VPS
        # they are real files, which is exactly the difference that would have
        # made this pass here and fail there, or the reverse.)
        if Path(name).name != name:
            raise FileNotFoundError(name)
        path = PAGES_DIR / name
        if not path.is_file():
            raise FileNotFoundError(name)
        document = pymupdf.open(path)
        _open_docs[name] = document
        while len(_open_docs) > CACHE_SIZE:
            _, evicted = _open_docs.popitem(last=False)
            evicted.close()
        return document


def cut_page(name: str, page: int) -> bytes:
    """Return a one-page PDF holding `page` of `name`.

    Parameters
    ----------
    name
        Filename of a restricted document.
    page
        1-based page number, as the index and the reader count them.

    Returns
    -------
    bytes
        A PDF with exactly one page.

    Notes
    -----
    `select` keeps the page and drops every other page's objects; `garbage=3`
    then collects what nothing references any more, so the result carries this
    page's content and its fonts rather than the whole document's. That is the
    difference between handing over a page and handing over a book with 1,274
    pages hidden in it.
    """
    # Held across the whole cut: `insert_pdf` reads the source document, so it
    # may not run while another thread is using or evicting the same handle.
    with _lock:
        document = open_document(name)
        cut = pymupdf.open()
        cut.insert_pdf(document, from_page=page - 1, to_page=page - 1)
        try:
            return cut.tobytes(garbage=3, deflate=True)
        finally:
            cut.close()


class Handler(http.server.BaseHTTPRequestHandler):
    """`GET /api/page?doc=<id>&p=<n>` and `GET /api/page/health`."""

    protocol_version = "HTTP/1.1"
    catalog: dict[int, dict] = {}

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # A page is a fragment of a document the site may not redistribute, so it
        # is never cached by a shared cache and never stored on disk by the
        # browser beyond the session.
        self.send_header("Cache-Control", "private, no-store")
        self.end_headers()
        self.wfile.write(body)

    def _fail(self, status: int, reason: str) -> None:
        self._send(status, json.dumps({"error": reason}).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802 (http.server's naming)
        """Answer a page request, or refuse it with a reason."""
        parsed = urlparse(self.path)
        if parsed.path == "/api/page/health":
            self._send(200, json.dumps({"ok": True, "documents": len(self.catalog)})
                       .encode("utf-8"), "application/json; charset=utf-8")
            return
        if parsed.path != "/api/page":
            self._fail(404, "not found")
            return

        query = parse_qs(parsed.query)
        try:
            doc_id = int(query.get("doc", [""])[0])
            page = int(query.get("p", [""])[0])
        except (ValueError, IndexError):
            self._fail(400, "doc and p must be integers")
            return

        entry = self.catalog.get(doc_id)
        if entry is None:
            # Deliberately the same answer for "no such document" and "that one
            # is open, fetch it from /pdf/": this service speaks only about the
            # restricted tier.
            self._fail(404, "no such restricted document")
            return
        if page < 1 or (entry["pages"] and page > entry["pages"]):
            self._fail(404, f"page {page} is outside this document")
            return

        if not _gate.acquire(timeout=10):
            self._fail(503, "busy")
            return
        try:
            body = cut_page(entry["file"], page)
        except FileNotFoundError:
            self._fail(404, "document not mounted")
            return
        except Exception as exc:  # pymupdf raises its own types for a bad page
            self._fail(500, f"could not cut that page: {exc.__class__.__name__}")
            return
        finally:
            _gate.release()
        self._send(200, body, "application/pdf")

    def log_message(self, fmt: str, *args) -> None:
        """One line per request on stdout, the container's only log."""
        sys.stderr.write(f"pages {self.address_string()} {fmt % args}\n")


class Server(socketserver.ThreadingTCPServer):
    """Threaded, so a slow cut does not block the health check."""

    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    """Load the catalog and serve until killed."""
    Handler.catalog = load_catalog(INDEX_DIR)
    if not Handler.catalog:
        print(f"no restricted documents in {INDEX_DIR}/meta.json; serving nothing",
              file=sys.stderr)
    print(f"pages: {len(Handler.catalog)} restricted documents from {PAGES_DIR}, "
          f"listening on :{PORT}, {CONCURRENCY} concurrent cuts", file=sys.stderr)
    with Server(("0.0.0.0", PORT), Handler) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    main()
