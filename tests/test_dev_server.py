"""scripts/dev_server.py reads its routes, hidden files and headers from the Caddyfile.

The point of reading rather than restating is that the local server cannot drift
from production; these pin that the reader sees what the real Caddyfile says.

Written by Claude Code (Opus 5.5).
"""

from pathlib import Path

from conftest import ROOT, load

CADDYFILE = ROOT / "docker" / "Caddyfile"


def test_the_routes_are_the_caddyfile_s_proxied_paths():
    dev = load("dev_server")
    assert dev.read_routes(CADDYFILE) == {
        "/api/search": "search", "/api/search/health": "search",
        "/api/page": "pages", "/api/page/health": "pages",
    }


def test_a_catch_all_and_a_commented_route_are_not_routes(tmp_path):
    """`handle /api/* { respond 404 }` proxies nothing, and a comment is not config."""
    dev = load("dev_server")
    caddy = tmp_path / "Caddyfile"
    caddy.write_text(":80 {\n"
                     "\t# handle /api/old { reverse_proxy {$SEARCH_UPSTREAM:x} }\n"
                     "\thandle /api/search {\n\t\treverse_proxy {$SEARCH_UPSTREAM:search:8650}\n\t}\n"
                     "\thandle /api/* {\n\t\trespond 404\n\t}\n}\n", encoding="utf-8")
    assert dev.read_routes(caddy) == {"/api/search": "search"}


def test_hidden_files_and_headers_come_from_the_caddyfile():
    dev = load("dev_server")
    assert ".stage-manifest.json" in dev.read_hidden(CADDYFILE)
    headers = dev.read_headers(CADDYFILE)
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Referrer-Policy"] == "no-referrer"
    # Placeholders resolve to their (empty) defaults, leaving no "{$" behind.
    assert "default-src 'self'" in headers["Content-Security-Policy"]
    assert "{$" not in headers["Content-Security-Policy"]
    assert "Server" not in headers   # "-Server" is a removal, not a header to send
