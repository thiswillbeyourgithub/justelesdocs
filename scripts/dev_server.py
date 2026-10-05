# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "loguru"]
# ///
"""Serve `dist/www/` locally, with the one request the site cannot answer by itself.

The site is static apart from a single call: `POST /api/search`, which takes the
reader's question and returns ranked passages. In production Caddy proxies exactly
that path (and `/api/search/health`) to the search container, and refuses every
other `/api/*`. Opening `dist/www/index.html` from the filesystem therefore gives a
page that loads and cannot search, and a plain `python -m http.server` gives the
same page with a 404 on every query.

This script is that proxy, and nothing more. It re-implements no part of the
ranking: the upstream is the real service, run locally against the real index as

    cd ../justelesRCP && uv run src/embed-service.py --port 8461 --no-backlog
    INDEX_DIR=$PWD/dist/index EMBED_URL=http://127.0.0.1:8461 node server/service.mjs

so what the browser shows comes from the same ranker, the same index and the same
encoder as on the VPS. The moment this file starts ranking or embedding itself, it
becomes a second definition of the contract and will drift from the one that
matters.

**What it mimics of production is read out of `docker/Caddyfile`**, rather than
restated here: which `/api/` paths are proxied and to which upstream, which files
the file server hides, and the response headers (the Content-Security-Policy among
them). A local server without the CSP hides a whole class of production-only
breakage: `style-src 'self'` blocks a `style=""` attribute parsed from HTML (though
not `el.style.x = y` from script), and a blocked attribute fails quietly, leaving a
page that renders slightly wrong and a console full of violations. Six of those
shipped in the markup before this server started sending the header. The routes
are read for the same reason: a hand-kept allow-list here once 404ed a path Caddy
served, and the page read the 404 as a feature switched off.

Two deliberate differences from production remain, because copying them here would
only hide mistakes:

* No rate limiting. That is Caddy's, and is verified against the real Caddyfile.
* `/app-config.js` is served from `dist/` (the local-development fallback copied
  from `src/`) rather than rendered from the environment. Empty means "off" for
  every key in it, so the local site tracks nothing and shows no banner.

Usage: uv run scripts/dev_server.py [--port 8649] [--upstream http://127.0.0.1:8650]

Written by Claude Code.
"""

from __future__ import annotations

import functools
import http.server
import json
import mimetypes
import re
import socketserver
import urllib.error
import urllib.request
from pathlib import Path

import click
from loguru import logger
from lib.corpus import in_corpus

# .mjs is absent from the system mime table on many distributions, and a module
# served as application/octet-stream is refused by the browser. pdf.js ships as
# pdf.mjs plus pdf.worker.mjs, so without this the viewer fails and the search
# page works, which is a confusing way to discover a MIME problem.
mimetypes.add_type("text/javascript", ".mjs")

# Everything this server mimics of production is READ from docker/Caddyfile rather
# than copied: which /api/ paths are proxied and to which upstream, which files the
# file server hides, and the response headers. A second copy would drift from what
# Caddy enforces, and the drift is invisible locally: everything keeps working here
# and the difference only shows on the deployed site. That has happened once
# already, with a stale allow-list here 404ing a path Caddy served.
HANDLE = re.compile(r"^\s*handle\s+(/api/[^\s{*]+)\s*\{", re.M)
UPSTREAM = re.compile(r"reverse_proxy\s+\{\$([A-Z]+)_UPSTREAM:")
HIDE = re.compile(r"^\s*hide\s+(.+)$", re.M)
HEADER_BLOCK = re.compile(r"^\theader\s*\{\n(.*?)^\t\}", re.M | re.S)
HEADER_LINE = re.compile(r'^\s*([A-Za-z-]+)\s+"([^"]*)"\s*$', re.M)
PLACEHOLDER = re.compile(r"\{\$[A-Z_]+:([^}]*)\}")


def strip_comments(text: str) -> str:
    """Drop Caddyfile `#` comments, so a path named in a comment is not a route."""
    return re.sub(r"(?m)^\s*#.*$", "", text)


def read_routes(caddyfile: Path) -> dict[str, str]:
    """The `/api/` paths Caddy proxies, each with the upstream it goes to.

    Parameters
    ----------
    caddyfile
        Path to `docker/Caddyfile`.

    Returns
    -------
    dict[str, str]
        Path -> upstream name, lower-cased from the `{$NAME_UPSTREAM:...}`
        placeholder of the `reverse_proxy` inside that `handle`:
        `{"/api/search": "search", "/api/page": "pages", ...}`. A `handle` with no
        `reverse_proxy` (the `/api/*` catch-all that answers 404) is not a route.

    Raises
    ------
    click.ClickException
        If the file cannot be read or proxies nothing: serving without the API
        would make every search look broken for a reason that is not the code.
    """
    try:
        text = strip_comments(caddyfile.read_text(encoding="utf-8"))
    except OSError as exc:
        raise click.ClickException(f"cannot read {caddyfile}: {exc}") from exc
    starts = list(HANDLE.finditer(text))
    routes: dict[str, str] = {}
    for n, match in enumerate(starts):
        # The handle's body runs to the next handle; the first reverse_proxy in it
        # is this handle's. Brace counting would be exact but the Caddyfile nests
        # no handle inside another, which the test pins.
        end = starts[n + 1].start() if n + 1 < len(starts) else len(text)
        upstream = UPSTREAM.search(text, match.end(), end)
        if upstream:
            routes[match.group(1)] = upstream.group(1).lower()
    if not routes:
        raise click.ClickException(f"{caddyfile} proxies no /api/ path; is it the right file?")
    return routes


def read_hidden(caddyfile: Path) -> set[str]:
    """File names Caddy's file_server refuses to serve (`hide`), as basenames."""
    try:
        text = strip_comments(caddyfile.read_text(encoding="utf-8"))
    except OSError:
        return set()
    return {name for line in HIDE.findall(text) for name in line.split()}


def read_headers(caddyfile: Path) -> dict[str, str]:
    """The response headers Caddy sets on every answer, for local parity.

    Parameters
    ----------
    caddyfile
        Path to `docker/Caddyfile`.

    Returns
    -------
    dict[str, str]
        Header name -> value, with Caddy's `{$VAR:default}` placeholders resolved
        to their defaults (all empty today: locally there is no analytics origin to
        allow). Removals (`-Server`) are skipped. Empty when the file cannot be read
        or has no top-level `header` block, in which case the caller serves without
        them and says so.
    """
    try:
        block = HEADER_BLOCK.search(strip_comments(caddyfile.read_text(encoding="utf-8")))
    except OSError:
        return {}
    if not block:
        return {}
    return {name: re.sub(r"\s+", " ", PLACEHOLDER.sub(r"\1", value)).strip()
            for name, value in HEADER_LINE.findall(block.group(1))}


class Handler(http.server.SimpleHTTPRequestHandler):
    """Static files from `dist/www`, plus a forward for the routes Caddy proxies."""

    # Upstream name (from the Caddyfile) -> base URL given on the command line.
    upstreams: dict[str, str] = {}
    routes: dict[str, str] = {}
    hidden: set[str] = set()
    headers_out: dict[str, str] = {}
    timeout: float = 20.0

    def _route(self) -> str | None:
        """The upstream base URL for this request's path, or None.

        Compared on the path alone: /api/page carries its document and page in the
        query string, which is forwarded verbatim.
        """
        name = self.routes.get(self.path.split("?", 1)[0])
        return self.upstreams.get(name) if name else None

    def do_POST(self) -> None:  # noqa: N802 (http.server's naming)
        """Forward a POST to a proxied path, verbatim."""
        base = self._route()
        if base is None:
            # 404, not 405: the production Caddyfile answers every other /api/*
            # with 404, and a local server that answers differently would make a
            # client-side mistake look like a routing mistake.
            self.send_error(404, "not proxied here")
            return
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self._forward(urllib.request.Request(
            f"{base}{self.path}", data=body, method="POST",
            headers={"Content-Type": self.headers.get("Content-Type", "application/json")}))

    def do_GET(self) -> None:  # noqa: N802
        """Serve a file, or forward a proxied path; 404 for any other /api/ or hidden file."""
        base = self._route()
        if base is not None:
            self._forward(urllib.request.Request(f"{base}{self.path}", method="GET"))
            return
        path = self.path.split("?", 1)[0]
        if path.startswith("/api/") or path.rsplit("/", 1)[-1] in self.hidden:
            self.send_error(404)
            return
        super().do_GET()

    def _forward(self, request: urllib.request.Request) -> None:
        """Pass one request upstream and copy the answer back.

        An upstream that is down is reported as 502 with a readable body, which is
        the same status Caddy produces and the status `search.js` turns into the
        "search unavailable" message. Getting that path exercised locally is half
        the point of running this at all.
        """
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as answer:
                payload, status = answer.read(), answer.status
                content_type = answer.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as exc:
            payload, status, content_type = exc.read(), exc.code, "application/json"
        except OSError as exc:
            # The URL actually tried, not one fixed upstream: there are two, and a
            # log naming the search service while the page service is down sends
            # the reader to the wrong process.
            logger.error(f"upstream {request.full_url} unreachable: {exc}")
            payload = json.dumps({"error": "unavailable"}).encode()
            status, content_type = 502, "application/json"
        self.send_response(status)
        # Whatever the upstream said, not always JSON: /api/page answers with
        # application/pdf, and pdf.js refuses a PDF served as anything else.
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def end_headers(self) -> None:
        # dist/ is symlinks into a working tree being edited, so caching anything
        # locally means editing a file and reloading the old one.
        self.send_header("Cache-Control", "no-store")
        for name, value in self.headers_out.items():
            self.send_header(name, value)
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        logger.debug(fmt % args)


class Server(socketserver.ThreadingTCPServer):
    """Threaded because one query blocks on the upstream while the page keeps
    fetching per-document text; single-threaded, that serialises into a stall that
    looks like a slow site."""

    daemon_threads = True
    allow_reuse_address = True


@click.command()
@click.option("--root", type=click.Path(exists=True, file_okay=False, path_type=Path),
              **in_corpus("dist/www"), help="Tree to serve, as shipped.")
@click.option("--port", type=int, default=8649, show_default=True,
              help="Local port. Not 8648, so this cannot be confused with the container.")
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind address.")
@click.option("--upstream", default="http://127.0.0.1:8650", show_default=True,
              help="The real search service: server/service.mjs, run locally against dist/index.")
@click.option("--pages-upstream", default="http://127.0.0.1:8651", show_default=True,
              help="The page service: server/pages.py, run locally against dist/restricted.")
@click.option("--caddyfile", type=click.Path(path_type=Path), default=Path("docker/Caddyfile"),
              show_default=True, help="Where to read the routes, hidden files and headers from, for parity with production.")
def main(root: Path, port: int, host: str, upstream: str, pages_upstream: str,
         caddyfile: Path) -> None:
    """Serve the staged site with the embedding call proxied to a real encoder."""
    if not (root / "index" / "meta.json").exists():
        logger.warning(f"no index in {root / 'index'}: the page will load but not "
                       "list the corpus. Build one with scripts/build_index.py, then stage.")
    Handler.upstreams = {"search": upstream.rstrip("/"), "pages": pages_upstream.rstrip("/")}
    Handler.routes = read_routes(caddyfile)
    unknown = set(Handler.routes.values()) - Handler.upstreams.keys()
    if unknown:
        raise click.ClickException(f"{caddyfile} proxies to {sorted(unknown)}, which this "
                                   "server has no --*-upstream option for.")
    Handler.hidden = read_hidden(caddyfile)
    Handler.headers_out = read_headers(caddyfile)
    logger.info(f"routes from {caddyfile}: " + ", ".join(f"{p} -> {u}" for p, u in sorted(Handler.routes.items())))
    if "Content-Security-Policy" in Handler.headers_out:
        logger.info(f"sending {len(Handler.headers_out)} headers from {caddyfile}, the CSP included")
    else:
        logger.warning(f"no Content-Security-Policy found in {caddyfile}; serving without one. "
                       "A style attribute blocked in production will work here.")
    handler = functools.partial(Handler, directory=str(root.resolve()))
    with Server((host, port), handler) as server:
        logger.success(f"serving {root} on http://{host}:{port} "
                       f"(search -> {Handler.upstreams['search']}, pages -> {Handler.upstreams['pages']})")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("stopped")


if __name__ == "__main__":
    main()
