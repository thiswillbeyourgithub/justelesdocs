"""Staging's destructive half: what `remove` deletes, and what it refuses to follow.

Everything `stage.py` puts in `dist/` is a symlink, and every staging run prunes
before it links. That combination is where the sharp edges are: `Path.is_dir()`
follows a symlink, so the obvious deletion spelling reaches THROUGH a link in
`dist/` at the corpus behind it. These tests pin the two behaviours that matter,
neither of which the build chain would report if it regressed: `dist/` would be
left dirty, or `data/` would be left short.

Written by Claude Code (Opus 5).
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_remove_unlinks_a_plain_file(stage, tmp_path):
    victim = tmp_path / "app.js"
    victim.write_text("x", encoding="utf-8")
    stage.remove(victim)
    assert not victim.exists()


def test_remove_deletes_a_real_directory_whole(stage, tmp_path):
    tree = tmp_path / "vendor"
    (tree / "pdfjs").mkdir(parents=True)
    (tree / "pdfjs" / "build.js").write_text("x", encoding="utf-8")
    stage.remove(tree)
    assert not tree.exists()


def test_remove_unlinks_a_symlink_to_a_directory_without_following_it(stage, tmp_path):
    """The case four different spellings disagreed about.

    `shutil.rmtree` refuses a symlink outright ("Cannot call rmtree on a symbolic
    link"), so the spelling that tested `is_dir()` first raised OSError here rather
    than pruning. Worse would have been a spelling that followed the link: `dist/`
    is symlinks all the way down, and the target is the corpus.
    """
    target = tmp_path / "GUIDELINES"
    target.mkdir()
    (target / "guideline.pdf").write_text("corpus", encoding="utf-8")
    link = tmp_path / "vendor"
    link.symlink_to(target)

    stage.remove(link)

    assert not link.exists() and not link.is_symlink()
    assert (target / "guideline.pdf").read_text(encoding="utf-8") == "corpus", \
        "remove followed the symlink and deleted the corpus behind it"


def test_remove_unlinks_a_symlink_to_a_file(stage, tmp_path):
    target = tmp_path / "real.pdf"
    target.write_text("pdf", encoding="utf-8")
    link = tmp_path / "staged.pdf"
    link.symlink_to(target)
    stage.remove(link)
    assert not link.is_symlink()
    assert target.exists()


def test_remove_unlinks_a_broken_symlink(stage, tmp_path):
    """A document removed from the corpus leaves exactly this behind in dist/pdf."""
    link = tmp_path / "withdrawn.pdf"
    link.symlink_to(tmp_path / "gone.pdf")
    assert not link.exists() and link.is_symlink()
    stage.remove(link)
    assert not link.is_symlink()


def test_remove_reports_a_path_that_is_not_there(stage, tmp_path):
    """Not silently tolerated: the callers only ever pass something they just saw."""
    with pytest.raises(FileNotFoundError):
        stage.remove(tmp_path / "never-existed")


def _index(tmp_path):
    """A built index: meta.json, a vector file and the per-document text."""
    index_dir = tmp_path / "dist" / "index"
    (index_dir / "doc").mkdir(parents=True)
    # One open document, so stage_doc_text has something to publish. The tier
    # split itself is covered in test_access_tier.py.
    (index_dir / "meta.json").write_text(
        '{"documents": [{"id": 0, "file": "open.pdf"}]}', encoding="utf-8")
    (index_dir / "meta.json.gz").write_bytes(b"\x1f\x8b")
    (index_dir / "vectors-abc.b1").write_bytes(b"\x00")
    (index_dir / "doc" / "0.json").write_text("{}", encoding="utf-8")
    return index_dir


def test_stage_index_links_meta_only_and_never_the_doc_directory(stage, tmp_path):
    """meta.json by relative link, no vectors, and `doc/` left to stage_doc_text.

    Linking `doc/` as a directory is what published the full text of every
    restricted document, so the assertion that it is absent here is the fix, not
    a detail: whatever lands in `doc/` now goes through the per-document path
    that knows about the access tier.
    """
    index_dir = _index(tmp_path)
    www = tmp_path / "dist" / "www" / "index"

    assert stage.stage_index(index_dir, www) is True

    assert sorted(p.name for p in www.iterdir()) == ["meta.json", "meta.json.gz"]
    for name in ("meta.json", "meta.json.gz"):
        link = www / name
        assert link.is_symlink() and not link.readlink().is_absolute(), \
            "links must be relative so dist/ can be rsynced as a whole"
        assert link.resolve() == (index_dir / name).resolve()
    assert not (www / "doc").exists(), "stage_index must not link the doc directory"
    assert not (www / "vectors-abc.b1").exists()


def test_stage_index_repoints_and_survives_a_moved_tree(stage, tmp_path):
    """Re-running after a rebuild re-points; moving dist/ keeps the links valid."""
    index_dir = _index(tmp_path)
    www = tmp_path / "dist" / "www" / "index"
    www.mkdir(parents=True)
    (www / "meta.json").symlink_to(tmp_path / "elsewhere.json")

    stage.stage_index(index_dir, www)
    stage.stage_doc_text(index_dir, www / "doc", {"open.pdf": "open"})
    (tmp_path / "dist").rename(tmp_path / "moved")

    moved = tmp_path / "moved" / "www" / "index"
    assert (moved / "meta.json").resolve() == (tmp_path / "moved" / "index" / "meta.json")
    # The per-document links have to survive the move for the same reason, since
    # deploy.sh rsyncs dist/ as a whole.
    assert (moved / "doc" / "0.json").exists()


def test_stage_index_without_an_index_leaves_no_dangling_link(stage, tmp_path):
    """No index yet: report it, and remove any link from an earlier run."""
    index_dir = tmp_path / "dist" / "index"
    www = tmp_path / "dist" / "www" / "index"
    www.mkdir(parents=True)
    (www / "meta.json").symlink_to(index_dir / "meta.json")

    assert stage.stage_index(index_dir, www) is False

    assert list(www.iterdir()) == []


def test_a_corpus_changelog_replaces_the_software_s(stage, tmp_path):
    """A corpus's own changelog/ and VERSION win; without them the software's are used."""
    import os
    corpus = Path(os.environ["CORPUS_DIR"])
    software = tmp_path / "docs-changelog"
    assert stage.corpus_or_software("changelog", software) == software
    (corpus / "changelog").mkdir()
    assert stage.corpus_or_software("changelog", software) == corpus / "changelog"
