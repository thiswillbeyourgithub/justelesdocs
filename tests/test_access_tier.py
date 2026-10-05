"""The `access: restricted` tier, and the two places it has to hold.

53 documents entered the corpus on the condition that they are served without
download (`data/CORPUS_ADDITIONS.tsv`, the `access` column of the manifest). That
condition was implemented in `src/viewer.js` alone for five days, during which
`stage.py` put all 534 PDFs in the web root and linked `index/doc/` whole, so
every one of those documents was a plain file at its own URL and its complete
extracted text was one request away. Nothing was deployed, so nothing leaked.

These tests cover the two halves of the fix, because each half failed
independently and neither is visible by reading the site:

  - `stage_pdfs` sends a restricted document OUTSIDE the web root, where only
    the page service can reach it, and prunes it out of the web root if it was
    there before (the tier can change; the staged tree must follow).
  - `stage_doc_text` publishes geometry without words, which is exactly what
    `src/viewer.js` needs to draw a highlight and no more.

and the gate that refuses a tree where either half went wrong.

Everything runs on a tiny synthetic corpus in a tmp_path, so the suite still
needs neither the real documents nor a built index.

Written by Claude Code (Opus 5).
"""

from __future__ import annotations

import gzip
import json

import pytest


def write_manifest(path, rows):
    """A manifest with just the columns these functions read."""
    lines = ["file\taccess"]
    lines += [f"{name}\t{tier}" for name, tier in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def corpus(tmp_path, monkeypatch, stage):
    """Two documents, one open and one restricted, with a chunk index and a manifest."""
    guidelines = tmp_path / "GUIDELINES"
    guidelines.mkdir()
    for name in ("open.pdf", "book.pdf"):
        (guidelines / name).write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setitem(stage.SOURCE_DIRS, "original", guidelines)
    manifest = tmp_path / "MANIFEST.tsv"
    write_manifest(manifest, [("open.pdf", "open"), ("book.pdf", "restricted")])
    return tmp_path, manifest, {"open.pdf": "original", "book.pdf": "original"}


def test_a_restricted_pdf_is_staged_outside_the_web_root(corpus, stage):
    """The whole mechanism: no URL may return the document."""
    root, manifest, index = corpus
    www, held = root / "www", root / "restricted"
    total, _, restricted = stage.stage_pdfs(index, www / "pdf",
                                            stage.read_access(manifest), held)
    assert (total, restricted) == (1, 1)
    assert (www / "pdf" / "open.pdf").exists()
    assert not (www / "pdf" / "book.pdf").exists()
    assert (held / "book.pdf").exists()
    # The sibling-not-child relationship is what makes it unreachable however
    # the site's routes are rewritten later, so it is asserted rather than assumed.
    assert held.resolve() not in www.resolve().parents
    assert www.resolve() not in held.resolve().parents


def test_a_document_that_becomes_restricted_is_pruned_from_the_web_root(corpus, stage):
    """The tier can change, and a stale symlink keeps serving the old answer."""
    root, manifest, index = corpus
    www, held = root / "www", root / "restricted"
    open_everything = {"open.pdf": "open", "book.pdf": "open"}
    stage.stage_pdfs(index, www / "pdf", open_everything, held)
    assert (www / "pdf" / "book.pdf").exists()

    stage.stage_pdfs(index, www / "pdf", stage.read_access(manifest), held)
    assert not (www / "pdf" / "book.pdf").exists(), "restricting a document must unstage it"
    assert (held / "book.pdf").exists()


def test_an_unknown_access_value_refuses_to_stage(corpus, stage):
    """A typo must not be read as permission to publish."""
    import click
    root, manifest, _ = corpus
    write_manifest(manifest, [("open.pdf", "open"), ("book.pdf", "restrcited")])
    with pytest.raises(click.ClickException, match="neither 'open' nor 'restricted'"):
        stage.read_access(manifest)


def test_a_blank_access_cell_refuses_to_stage(corpus, stage, check_served):
    """A blank used to read as 'open', in both the stager and the gate, so a cell
    cleared by hand published a book and the gate agreed with it."""
    import click
    root, manifest, _ = corpus
    write_manifest(manifest, [("open.pdf", "open"), ("book.pdf", "")])
    with pytest.raises(click.ClickException, match="neither 'open' nor 'restricted'"):
        stage.read_access(manifest)
    with pytest.raises(click.ClickException, match="neither 'open' nor 'restricted'"):
        check_served.read_tiers(manifest)


def test_a_document_missing_from_the_manifest_refuses_to_stage(corpus, stage):
    """A rename orphans the row; the new name must not default to 'open'."""
    import click
    root, manifest, index = corpus
    write_manifest(manifest, [("open.pdf", "open")])
    with pytest.raises(click.ClickException, match="book.pdf has no row"):
        stage.stage_pdfs(index, root / "www" / "pdf", stage.read_access(manifest),
                         root / "restricted")


def test_the_gate_finds_a_renamed_restricted_document_by_its_content(tmp_path, check_served):
    """The manifest is what the stager obeyed, so the gate needs a second anchor:
    the sha256 recorded when the document entered the corpus."""
    import hashlib
    corpus = tmp_path / "GUIDELINES"
    corpus.mkdir()
    (corpus / "renamed book.pdf").write_bytes(b"%PDF a book")
    (corpus / "guide.pdf").write_bytes(b"%PDF a guideline")
    additions = tmp_path / "CORPUS_ADDITIONS.tsv"
    book = hashlib.sha256(b"%PDF a book").hexdigest()
    guide = hashlib.sha256(b"%PDF a guideline").hexdigest()
    additions.write_text("file\tsha256\taccess\n"
                         f"book.pdf\t{book}\trestricted\n"
                         f"guide.pdf\t{guide}\topen\n", encoding="utf-8")
    digests = check_served.restricted_digests(additions)
    assert check_served.restricted_by_content(corpus, digests) == {"renamed book.pdf"}


def test_the_gate_refuses_a_restricted_document_the_manifest_calls_open(tmp_path, check_served):
    """End to end through the CLI: the rename scenario, before anything is staged."""
    import hashlib
    from click.testing import CliRunner
    corpus = tmp_path / "GUIDELINES"
    corpus.mkdir()
    (corpus / "renamed.pdf").write_bytes(b"%PDF a book")
    additions = tmp_path / "CORPUS_ADDITIONS.tsv"
    additions.write_text("file\tsha256\taccess\n"
                         f"book.pdf\t{hashlib.sha256(b'%PDF a book').hexdigest()}\trestricted\n",
                         encoding="utf-8")
    manifest = tmp_path / "MANIFEST.tsv"
    write_manifest(manifest, [("renamed.pdf", "open")])
    www = tmp_path / "www"
    www.mkdir()
    result = CliRunner().invoke(check_served.main, [
        "--www", str(www), "--manifest", str(manifest), "--index-dir", str(tmp_path / "none"),
        "--additions", str(additions), "--corpus", str(corpus)])
    assert result.exit_code != 0
    assert "not marked restricted" in result.output


def test_the_gate_runs_on_the_manifest_alone_when_there_is_no_additions_record(
        tmp_path, check_served):
    """A corpus that never took a document in on condition keeps no such record;
    the gate still refuses a restricted file in the tree, by the manifest."""
    from click.testing import CliRunner
    corpus = tmp_path / "GUIDELINES"
    corpus.mkdir()
    (corpus / "book.pdf").write_bytes(b"%PDF a book")
    manifest = tmp_path / "MANIFEST.tsv"
    write_manifest(manifest, [("book.pdf", "restricted")])
    www = tmp_path / "www"
    (www / "pdf").mkdir(parents=True)
    args = ["--www", str(www), "--manifest", str(manifest), "--index-dir", str(tmp_path / "none"),
            "--additions", str(tmp_path / "absent.tsv"), "--corpus", str(corpus)]
    assert CliRunner().invoke(check_served.main, args).exit_code == 0
    (www / "pdf" / "book.pdf").write_bytes(b"%PDF a book")
    result = CliRunner().invoke(check_served.main, args)
    assert result.exit_code != 0
    assert "reachable under the web root" in result.output


def test_published_text_keeps_geometry_and_drops_words(tmp_path, stage):
    """What `src/viewer.js` reads is `boxes` and `pages`; what it never reads is `text`."""
    index_dir = tmp_path / "index"
    (index_dir / "doc").mkdir(parents=True)
    (index_dir / "meta.json").write_text(json.dumps({"documents": [
        {"id": 0, "file": "open.pdf"}, {"id": 1, "file": "book.pdf"}]}), encoding="utf-8")
    for doc_id in (0, 1):
        (index_dir / "doc" / f"{doc_id}.json").write_text(json.dumps({
            "file": f"{doc_id}", "chunks": [
                {"i": 0, "pages": [1], "boxes": {"1": [[0, 0, 10, 10]]}, "text": "copyrighted"}]}),
            encoding="utf-8")

    destination = tmp_path / "www" / "index" / "doc"
    linked, stripped = stage.stage_doc_text(index_dir, destination,
                                            {"open.pdf": "open", "book.pdf": "restricted"})
    assert (linked, stripped) == (1, 1)

    published = json.loads((destination / "1.json").read_text(encoding="utf-8"))
    chunk = published["chunks"][0]
    assert "text" not in chunk
    assert chunk["boxes"] == {"1": [[0, 0, 10, 10]]}, "the highlight still has to land"
    assert chunk["pages"] == [1]
    # The open one is a link to the real file, text and all.
    assert (destination / "0.json").is_symlink()
    assert "copyrighted" in (destination / "0.json").read_text(encoding="utf-8")


def test_the_gzip_beside_a_stripped_file_says_the_same_thing(tmp_path, stage):
    """Caddy serves the .gz whenever the client accepts it, so a stale one leaks."""
    index_dir = tmp_path / "index"
    (index_dir / "doc").mkdir(parents=True)
    (index_dir / "meta.json").write_text(json.dumps({"documents": [{"id": 1, "file": "book.pdf"}]}),
                                         encoding="utf-8")
    body = json.dumps({"file": "b", "chunks": [{"i": 0, "pages": [1], "boxes": {},
                                                "text": "copyrighted"}]})
    (index_dir / "doc" / "1.json").write_text(body, encoding="utf-8")
    (index_dir / "doc" / "1.json.gz").write_bytes(gzip.compress(body.encode("utf-8")))

    destination = tmp_path / "www" / "index" / "doc"
    stage.stage_doc_text(index_dir, destination, {"book.pdf": "restricted"})
    packed = json.loads(gzip.decompress((destination / "1.json.gz").read_bytes()).decode("utf-8"))
    assert "text" not in packed["chunks"][0]


def test_the_gate_refuses_a_restricted_pdf_in_the_web_root(tmp_path, check_served):
    """The failure that actually happened, caught by name at any depth."""
    www = tmp_path / "www" / "pdf"
    www.mkdir(parents=True)
    (www / "book.pdf").write_bytes(b"%PDF")
    assert check_served.offending_files(tmp_path / "www", {"book.pdf"}) == [www / "book.pdf"]
    assert check_served.offending_files(tmp_path / "www", {"other.pdf"}) == []


def test_the_gate_refuses_published_text_including_the_gzip(tmp_path, check_served):
    """Both spellings of the same leak, since Caddy will serve either."""
    www = tmp_path / "www"
    (www / "index" / "doc").mkdir(parents=True)
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"documents": [{"id": 7, "file": "book.pdf"}]}), encoding="utf-8")
    body = json.dumps({"chunks": [{"text": "copyrighted"}]})
    (www / "index" / "doc" / "7.json").write_text(body, encoding="utf-8")
    (www / "index" / "doc" / "7.json.gz").write_bytes(gzip.compress(body.encode("utf-8")))

    found = check_served.offending_text(www, meta, {"book.pdf"})
    assert {p.name for p, _ in found} == {"7.json", "7.json.gz"}
    assert all(size == len("copyrighted") for _, size in found)

    (www / "index" / "doc" / "7.json").write_text(json.dumps({"chunks": [{"boxes": {}}]}),
                                                  encoding="utf-8")
    (www / "index" / "doc" / "7.json.gz").unlink()
    assert check_served.offending_text(www, meta, {"book.pdf"}) == []
