"""scripts/lib/site_config.py and stage.py's brand_site: a corpus's branding on the pages.

Written by Claude Code (Opus 5.5).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from conftest import load_lib


@pytest.fixture
def site_config():
    return load_lib("site_config")


def write_site(table: str) -> None:
    """Replace the test corpus's [site] table, and drop the cached configuration."""
    path = Path(os.environ["CORPUS_DIR"]) / "corpus.toml"
    text = path.read_text(encoding="utf-8")
    path.write_text(text[:text.index("[site]")] + table, encoding="utf-8")
    load_lib("corpus_config").load.cache_clear()


@pytest.fixture(autouse=True)
def fresh_config():
    """Every test reads its own corpus.toml, not the one a previous test cached."""
    load_lib("corpus_config").load.cache_clear()
    yield
    load_lib("corpus_config").load.cache_clear()


def test_the_example_site_loads_with_its_defaults(site_config):
    site = site_config.load_site()
    assert site["id"] == "example"
    assert site["brand"] == ["juste les ", "docs"]
    assert site["default_language"] == "fr"
    assert site["author"] == "" and site["logo"] is None
    assert site["legacy_keys"] == {}
    # No repo_url: the software's own, written in one place only.
    assert site["repo_url"] == site_config.SOFTWARE_REPO_URL


def test_a_one_part_brand_has_an_empty_emphasis(site_config):
    write_site('[site]\nid = "x"\nname = "Plain"\n')
    site = site_config.load_site()
    assert site["brand"] == ["Plain", ""]
    assert site["languages"] == ["fr", "en"]


@pytest.mark.parametrize("table, word", [
    ('[site]\nid = "Has Space"\nname = "n"\n', "id"),
    ('[site]\nid = "x"\nname = ""\n', "name"),
    ('[site]\nid = "x"\nname = "n"\nlanguages = ["fr"]\ndefault_language = "en"\n', "default_language"),
    ('[site]\nid = "x"\nname = "n"\nbrand = ["a", "b", "c"]\n', "brand"),
    ('[site]\nid = "x"\nname = "n"\nlogo = "missing.svg"\n', "logo"),
    ('[site]\nid = "x"\nname = "n"\nlegacy_keys = { old = 3 }\n', "legacy_keys"),
])
def test_a_malformed_site_table_is_refused_by_key(site_config, table, word):
    write_site(table)
    with pytest.raises(ValueError, match=word):
        site_config.load_site()


def test_legacy_keys_reach_the_page(site_config):
    """A corpus's pre-prefix localStorage names are handed to store.js, not hardcoded there."""
    write_site('[site]\nid = "x"\nname = "n"\nlegacy_keys = { old_seen = "changelogSeen" }\n')
    script = site_config.site_script(site_config.load_site(), {})
    payload = json.loads(script[script.index("{"):script.rindex("}") + 1])
    assert payload["legacy_keys"] == {"old_seen": "changelogSeen"}


def test_placeholders_are_filled_and_escaped(site_config):
    page = '<title>{{name}}</title><meta content="{{description}}">'
    out = site_config.render_page(page, {"name": "A & B", "description": 'say "hi"'})
    assert out == '<title>A &amp; B</title><meta content="say &quot;hi&quot;">'


def test_an_unknown_placeholder_is_refused_rather_than_shipped(site_config):
    with pytest.raises(KeyError, match="nope"):
        site_config.render_page("{{nope}}", {"name": "x"})


def test_an_overlay_must_be_flat_strings(site_config, tmp_path):
    (tmp_path / "fr.json").write_text(json.dumps({"a": {"b": "c"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="flat"):
        site_config.load_strings(["fr"], tmp_path)


def test_a_missing_overlay_is_a_language_without_one(site_config, tmp_path):
    (tmp_path / "en.json").write_text('{"tagline": "t"}', encoding="utf-8")
    assert site_config.load_strings(["fr", "en"], tmp_path) == {"en": {"tagline": "t"}}


def test_brand_site_writes_pages_without_touching_the_templates(stage, tmp_path):
    """The staged page is a symlink INTO src/; writing through it would overwrite
    the template with its own rendering, and the next run would find no
    placeholder left to fill."""
    src = tmp_path / "src"
    src.mkdir()
    template = '<html lang="{{lang}}"><title>{{name}}</title></html>'
    (src / "index.html").write_text(template, encoding="utf-8")
    (src / "logo.svg").write_text("<svg/>", encoding="utf-8")
    out = tmp_path / "www"
    out.mkdir()
    stage.stage_assets([src], out)
    stage.brand_site(out, [src])
    assert (src / "index.html").read_text(encoding="utf-8") == template
    assert not (out / "index.html").is_symlink()
    assert (out / "index.html").read_text(encoding="utf-8") == \
        '<html lang="fr"><title>justelesdocs</title></html>'
    assert 'window.__SITE__' in (out / "site-config.js").read_text(encoding="utf-8")
    # No corpus logo declared: the software's stays linked.
    assert (out / "logo.svg").is_symlink()
