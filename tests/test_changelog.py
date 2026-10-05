"""The release-notes parser, which is a gate as much as a compiler.

What makes these worth writing: `stage.py` refuses to stage when a version has no
notes or a bullet has no French line, and the failure it prevents is silent (a
French reader shown an English bullet, or the previous release's notes under a new
version's name). Every rejection below is one `parse_changelog` is expected to
make, so a lenient rewrite of it fails here rather than in production.

Written by Claude Code (Opus 5).
"""

from __future__ import annotations

import pytest

GOOD = """# 0.2.0 - 2026-01-31

## New features

- A new thing [abc1234]
fr: Une nouveauté

## Bug fixes

- A fixed thing [abc1234, def5678]
fr: Une correction
"""


def test_version_key_is_numeric(changelog):
    """0.10.0 comes after 0.9.0, which string comparison gets backwards."""
    assert changelog.version_key("0.10.0") > changelog.version_key("0.9.0")
    assert sorted(["0.10.0", "0.9.0", "0.2.1"], key=changelog.version_key) == [
        "0.2.1", "0.9.0", "0.10.0",
    ]


def test_parses_a_release(changelog):
    release = changelog.parse_changelog(GOOD, "0.2.0")
    assert release["date"] == "2026-01-31"
    assert [section["key"] for section in release["sections"]] == ["features", "fixes"]
    item = release["sections"][0]["items"][0]
    assert item == {"en": "A new thing", "fr": "Une nouveauté", "commits": ["abc1234"]}
    assert release["sections"][1]["items"][0]["commits"] == ["abc1234", "def5678"]


def test_sections_come_out_in_display_order(changelog):
    """The file may write the categories in any order; the popup's order is fixed."""
    text = GOOD.replace("## New features", "## ZZZ").replace("## Bug fixes", "## New features")
    text = text.replace("## ZZZ", "## Bug fixes")
    release = changelog.parse_changelog(text, "0.2.0")
    assert [section["key"] for section in release["sections"]] == ["features", "fixes"]


@pytest.mark.parametrize("mangle, expected", [
    (lambda t: t.replace("fr: Une nouveauté\n", ""), "no 'fr:' line"),
    (lambda t: t.replace("## New features", "## Nouveautés"), "unknown category"),
    (lambda t: t.replace("# 0.2.0 - 2026-01-31", "# 0.3.0 - 2026-01-31"),
     "does not match the directory"),
    (lambda t: t.replace("# 0.2.0 - 2026-01-31", "# version 0.2.0"),
     "the title must read"),
    (lambda t: t + "\n## New features\n\n- again\nfr: encore\n", "appears twice"),
    (lambda t: "- orphan\nfr: orpheline\n" + t, "outside any"),
    (lambda t: t.replace("- A new thing [abc1234]", "- [abc1234]"), "an empty bullet"),
    (lambda t: t.replace("fr: Une nouveauté", "fr:"), "an empty 'fr:' line"),
    (lambda t: t.replace("- A new thing [abc1234]", "A new thing"), "not a title"),
])
def test_rejects(changelog, mangle, expected):
    """Each of these is a way to ship the wrong text to a reader."""
    with pytest.raises(ValueError) as caught:
        changelog.parse_changelog(mangle(GOOD), "0.2.0")
    assert expected in str(caught.value)


def test_error_messages_name_the_file_and_line(changelog):
    with pytest.raises(ValueError) as caught:
        changelog.parse_changelog(GOOD.replace("## New features", "## Nope"),
                                  "0.2.0", where="docs/changelog/0.2.0/changelog.md")
    assert "docs/changelog/0.2.0/changelog.md:3:" in str(caught.value)


def test_refuses_a_version_with_no_notes(changelog, tmp_path):
    """The exact failure stage.py exists to prevent."""
    import click

    notes = tmp_path / "changelog"
    (notes / "0.2.0").mkdir(parents=True)
    (notes / "0.2.0" / "changelog.md").write_text(GOOD, encoding="utf-8")
    assert changelog.load_changelog(notes, "0.2.0")["current"] == "0.2.0"
    with pytest.raises(click.ClickException) as caught:
        changelog.load_changelog(notes, "0.3.0")
    assert "no release notes for version 0.3.0" in str(caught.value)


def test_releases_come_out_newest_first(changelog, tmp_path):
    notes = tmp_path / "changelog"
    for version in ("0.2.0", "0.10.0", "0.9.0"):
        (notes / version).mkdir(parents=True)
        (notes / version / "changelog.md").write_text(
            GOOD.replace("0.2.0", version), encoding="utf-8")
    payload = changelog.load_changelog(notes, "0.10.0")
    assert [r["version"] for r in payload["releases"]] == ["0.10.0", "0.9.0", "0.2.0"]


def test_the_shipped_notes_parse(changelog, root):
    """The repository's own release notes, which is what actually gets served."""
    version = changelog.read_version(root / "VERSION")
    payload = changelog.load_changelog(root / "docs" / "changelog", version)
    assert payload["releases"], "no release has notes"
    for release in payload["releases"]:
        for section in release["sections"]:
            for item in section["items"]:
                assert item["fr"] and item["en"], f"{release['version']}: a half-translated bullet"
