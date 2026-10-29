"""scripts/lib/llm_pages.py: the site as plain text for AI assistants.

The one rule that matters is the restricted tier's: such a document gets a line
in the listing and nothing else. The rest pins the URL scheme the search service
(src/paths.js: apiDocHref) links to, so a search hit never points at a 404.

Written by Claude Code (Opus 5.5).
"""

from __future__ import annotations

import json

import pytest

from conftest import load_lib


@pytest.fixture
def llm_pages():
    return load_lib("llm_pages")


def write_index(root, documents, chunks_by_id):
    index_dir = root / "index"
    (index_dir / "doc").mkdir(parents=True)
    (index_dir / "meta.json").write_text(json.dumps({
        "documents": documents, "facet_fields": ["issuer", "year", "topic", "access"],
        "range_fields": ["year"], "separator": ";", "guideline_tiers": ["strict", "wide", "no"]}),
        encoding="utf-8")
    for doc_id, chunks in chunks_by_id.items():
        (index_dir / "doc" / f"{doc_id}.json").write_text(json.dumps({"chunks": chunks}), encoding="utf-8")
    return index_dir


@pytest.fixture
def built(tmp_path, llm_pages):
    documents = [
        {"id": 0, "file": "Open doc.pdf", "title": "Open", "issuer": "HAS", "year": "2020",
         "topic": "a;b", "pages": "3"},
        {"id": 1, "file": "book.pdf", "title": "Book", "issuer": "", "year": "2019", "pages": "2"},
    ]
    index_dir = write_index(tmp_path, documents, {
        0: [{"i": 0, "pages": [1], "text": "first page"},
            {"i": 1, "pages": [1, 2], "text": "spans two"},
            {"i": 2, "pages": [3], "text": "a chart", "figure": True}],
        1: [{"i": 0, "pages": [1], "text": "copyrighted words"}],
    })
    www = tmp_path / "www"
    tiers = {"Open doc.pdf": "open", "book.pdf": "restricted"}
    llm_pages.stage_api_docs(index_dir, www, tiers, "https://x.org")
    return index_dir, www, tiers


def test_the_url_scheme_matches_src_paths(llm_pages):
    """tests/markdown.test.mjs pins the same example for apiDocHref."""
    assert llm_pages.doc_href("Open doc.pdf") == "api/doc/Open%20doc"
    assert llm_pages.doc_href("Open doc.pdf", 3) == "api/doc/Open%20doc/p/3"


def test_an_open_document_gets_its_pages(built):
    _, www, _ = built
    folder = www / "api" / "doc" / "Open doc"
    assert sorted(p.name for p in (folder / "p").iterdir()) == ["1.md", "2.md", "3.md"]
    two = (folder / "p" / "2.md").read_text(encoding="utf-8")
    assert "(continued from p. 1)" in two and "spans two" in two
    assert "previous: https://x.org/api/doc/Open%20doc/p/1" in two
    assert "AI model" in (folder / "p" / "3.md").read_text(encoding="utf-8")
    assert "- issuer: HAS" in (folder / "index.md").read_text(encoding="utf-8")


def test_a_restricted_document_gets_a_line_and_nothing_else(built):
    _, www, _ = built
    assert not (www / "api" / "doc" / "book").exists()
    listing = (www / "api" / "doc" / "index.md").read_text(encoding="utf-8")
    assert "Book" in listing and "restricted" in listing
    assert not any("copyrighted" in p.read_text(encoding="utf-8") for p in www.rglob("*.md"))


def test_a_document_that_becomes_restricted_loses_its_folder(built, llm_pages):
    index_dir, www, tiers = built
    llm_pages.stage_api_docs(index_dir, www, {**tiers, "Open doc.pdf": "restricted"}, "")
    assert not (www / "api" / "doc" / "Open doc").exists()


def test_every_page_starts_with_the_byline_when_the_index_has_one(built, llm_pages):
    index_dir, www, tiers = built
    meta_path = index_dir / "meta.json"
    meta = {**json.loads(meta_path.read_text(encoding="utf-8")), "byline": "Docs, par Jane Doe"}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    llm_pages.stage_api_docs(index_dir, www, tiers, "")
    pages = list((www / "api" / "doc").rglob("*.md"))
    assert {p.name for p in pages} >= {"index.md", "not_found.md", "1.md"}
    assert all(p.read_text(encoding="utf-8").startswith("Docs, par Jane Doe\n\n# ") for p in pages)
    site = {"name": "Docs", "description": "", "url": "", "repo_url": "https://r"}
    assert llm_pages.render_llms_txt(site, meta, tiers).startswith("# Docs, par Jane Doe\n")


def test_the_byline_is_in_the_site_s_language_and_absent_without_an_author():
    site_config = load_lib("site_config")
    site = {"name": "Docs", "author": "Jane Doe", "default_language": "fr"}
    assert site_config.byline(site) == "Docs, par Jane Doe"
    assert site_config.byline({**site, "default_language": "en"}) == "Docs, by Jane Doe"
    assert site_config.byline({**site, "author": ""}) == ""


def test_llms_txt_lists_the_filters_and_the_restricted_tier(built, llm_pages):
    index_dir, _, tiers = built
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    site = {"name": "Docs", "description": "A corpus.", "url": "https://x.org", "repo_url": "https://r"}
    text = llm_pages.render_llms_txt(site, meta, tiers)
    assert text.startswith("# Docs\n\n> A corpus.")
    assert "https://x.org/api/search.md?q=" in text
    assert "`topic`" in text and "`a` (1)" in text
    assert "from 2019 to 2020" in text
    assert "`strict`, `wide` narrow" in text
    assert "1 documents may only be read by a person" in text


def test_the_gate_refuses_a_page_quoting_a_restricted_line_found_nowhere_open(built, check_served):
    index_dir, www, tiers = built
    restricted = {"book.pdf"}
    long_line = "a restricted sentence long enough to be its own line, not a shared heading"
    doc = json.loads((index_dir / "doc" / "1.json").read_text(encoding="utf-8"))
    doc["chunks"][0]["text"] = long_line
    (index_dir / "doc" / "1.json").write_text(json.dumps(doc), encoding="utf-8")
    assert check_served.offending_markdown(www, index_dir, restricted) == []
    page = www / "api" / "doc" / "Open doc" / "p" / "1.md"
    page.write_text(page.read_text(encoding="utf-8") + f"\n> {long_line}\n", encoding="utf-8")
    assert [p for p, _ in check_served.offending_markdown(www, index_dir, restricted)] == [page]
    # The same line in an open document is a quotation the open PDF already serves.
    opened = json.loads((index_dir / "doc" / "0.json").read_text(encoding="utf-8"))
    opened["chunks"][0]["text"] = long_line
    (index_dir / "doc" / "0.json").write_text(json.dumps(opened), encoding="utf-8")
    assert check_served.offending_markdown(www, index_dir, restricted) == []


def test_the_gate_refuses_a_restricted_document_s_folder(built, check_served):
    index_dir, www, _ = built
    (www / "api" / "doc" / "book").mkdir()
    assert [p.name for p, _ in check_served.offending_markdown(www, index_dir, {"book.pdf"})] == ["book"]


def test_robots_lets_assistants_into_api_and_keeps_crawlers_out(llm_pages):
    """A person's assistant must be able to read what llms.txt points it at."""
    from urllib.robotparser import RobotFileParser

    robots = RobotFileParser()
    robots.parse(llm_pages.ROBOTS_TXT.splitlines())
    for agent in llm_pages.ASSISTANT_AGENTS:
        assert robots.can_fetch(agent, "/api/search.md?q=x"), agent
        assert robots.can_fetch(agent, "/api/doc/some-doc/p/3"), agent
    assert not robots.can_fetch("Googlebot", "/api/search.md?q=x")
    assert not robots.can_fetch("ClaudeBot", "/api/doc/")
    assert robots.can_fetch("Googlebot", "/llms.txt")
