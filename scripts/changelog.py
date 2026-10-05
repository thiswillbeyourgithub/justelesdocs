# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "loguru"]
# ///
"""Compile `docs/changelog/<version>/changelog.md` into `dist/changelog.json`.

Release notes are AUTHORED per version, one directory per release, so a version's
notes land in the same commit as the change they describe. They are COMPILED here
into one small JSON file that `src/changelog.js` fetches for the "Quoi de neuf ?"
popup. Notes are written for a casual reader (whoever the corpus serves), not for a
developer: short bullets under four fixed headings, each carrying the commit
sha(s) it came from so the popup can link to the code for anyone who wants it.

**This is a gate as much as a compiler.** `stage.py` imports it and refuses to
stage when `VERSION` names a release with no notes, when a file does not parse, or
when a bullet is missing one of its two languages. The failure it prevents is
quiet: a deploy that ships the previous release's notes under the new version's
name, or a popup that shows a French reader an English bullet.

The format is documented in `docs/changelog/README.md` and is deliberately
identical to `../justelesRCP`'s, which is where this mechanism comes from. **One
thing differs, and it is not cosmetic: this site is bilingual, so both languages
are served.** The sibling compiles the French half only and leaves the English
line in the markdown as authoring material, because that site is French. Here a
reader can be reading in either language when the popup opens, so a bullet ships
as `{"fr": ..., "en": ...}` and `changelog.js` picks by locale.

Written by Claude Code.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import click
from loguru import logger
from lib.corpus import in_corpus

# Where a commit sha in a bullet points. The compiled JSON carries it so
# changelog.js hardcodes no repository, exactly as meta.json names the index files
# so search.js hardcodes no filename. stage.py passes the site's repository
# (corpus.toml [site] repo_url, else the software's own, both resolved in
# lib/site_config.py). Run on its own, this script links no sha: changelog.js
# shows a bare sha when the prefix is not an http(s) URL.
COMMIT_URL = ""

# The only four categories a release note may use, in display order:
# (key, the H2 the markdown must use, French label, English label). Both labels
# ride in the JSON so the client keeps no copy of this table; the English H2 is
# the authoring grammar and is not shown as-is, which is why "New features" and
# its label are separate strings even when they are equal.
CATEGORIES = (
    ("features", "New features", "Nouveautés", "New features"),
    ("improvements", "Improvements", "Améliorations", "Improvements"),
    ("fixes", "Bug fixes", "Corrections", "Bug fixes"),
    ("docs", "Documentation", "Documentation", "Documentation"),
)

_H1 = re.compile(r"^#\s+v?(\d+\.\d+\.\d+)\s+[-–]\s+(\d{4}-\d{2}-\d{2})\s*$")
_SHAS = re.compile(r"\[([0-9a-f]{7,40}(?:\s*,\s*[0-9a-f]{7,40})*)\]$")
_VERSION = re.compile(r"^\d+\.\d+\.\d+$")


def version_key(version: str) -> tuple[int, int, int]:
    """Sort key for a `major.minor.patch` string.

    Parameters
    ----------
    version
        A version like ``"0.10.0"``.

    Returns
    -------
    tuple[int, int, int]
        Component-wise, so that 0.10.0 sorts after 0.9.0 rather than before it,
        which is what string comparison would do.
    """
    return tuple(int(part) for part in version.split("."))  # type: ignore[return-value]


def parse_changelog(text: str, version: str, where: str = "changelog") -> dict:
    """Parse ONE release-notes file.

    Parameters
    ----------
    text
        The file's contents.
    version
        The version its directory is named after. The title must agree with it.
    where
        A path used in error messages, so a malformed file names itself.

    Returns
    -------
    dict
        ``{"version": str, "date": "YYYY-MM-DD",
           "sections": [{"key": str, "items": [{"en", "fr", "commits"}]}]}``,
        sections in `CATEGORIES` order rather than in the order the file wrote
        them, so the popup's reading order is a property of the format.

    Raises
    ------
    ValueError
        On anything the format does not allow: a missing or disagreeing title, an
        unknown or repeated category, a bullet outside a category, a bullet with
        no French line, an empty section, or a line that is none of those things.
        Every message carries `where:line`.

    Notes
    -----
    The parser is deliberately strict and line-based rather than a markdown
    library. These files are written by hand a few times a year, and the failure
    that matters (a bullet whose `fr:` line was forgotten, shipping English text
    to a French reader) is one a lenient parser would carry all the way to the
    popup.
    """
    labels = {en: key for key, en, _, _ in CATEGORIES}
    date = ""
    buckets: dict[str, list[dict]] = {}
    items: list[dict] | None = None      # the section being filled
    item: dict | None = None             # the bullet a "fr:" line would belong to

    for num, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        pos = f"{where}:{num}"
        if not line:
            continue
        if line.startswith("# "):
            match = _H1.match(line)
            if not match:
                raise ValueError(f"{pos}: the title must read '# <version> - <YYYY-MM-DD>'")
            if match.group(1) != version:
                raise ValueError(
                    f"{pos}: title version {match.group(1)} does not match the "
                    f"directory, {version}"
                )
            date = match.group(2)
        elif line.startswith("## "):
            title = line[3:].strip()
            if title not in labels:
                raise ValueError(
                    f"{pos}: unknown category {title!r}; use one of "
                    + ", ".join(repr(en) for _, en, _, _ in CATEGORIES)
                )
            if labels[title] in buckets:
                raise ValueError(f"{pos}: category {title!r} appears twice")
            items = buckets[labels[title]] = []
            item = None
        elif line.startswith("- "):
            if items is None:
                raise ValueError(f"{pos}: a bullet outside any '## <category>' section")
            body = line[2:].strip()
            commits: list[str] = []
            match = _SHAS.search(body)
            if match:
                commits = [sha.strip() for sha in match.group(1).split(",")]
                body = body[: match.start()].strip()
            if not body:
                raise ValueError(f"{pos}: an empty bullet")
            item = {"en": body, "fr": "", "commits": commits}
            items.append(item)
        elif line.startswith("fr:"):
            if item is None:
                raise ValueError(f"{pos}: a 'fr:' line before any bullet")
            if item["fr"]:
                raise ValueError(f"{pos}: this bullet already has a 'fr:' line")
            item["fr"] = line[3:].strip()
            if not item["fr"]:
                raise ValueError(f"{pos}: an empty 'fr:' line")
        else:
            raise ValueError(f"{pos}: not a title, category, bullet or 'fr:' line: {line[:60]!r}")

    if not date:
        raise ValueError(f"{where}: no title line '# <version> - <YYYY-MM-DD>'")
    sections = []
    for key, english, _, _ in CATEGORIES:
        bucket = buckets.get(key)
        if bucket is None:
            continue
        if not bucket:
            raise ValueError(f"{where}: category {english!r} has no bullets")
        for entry in bucket:
            if not entry["fr"]:
                raise ValueError(
                    f"{where}: no 'fr:' line under {entry['en'][:40]!r}; both "
                    "languages are required because the site shows both"
                )
        sections.append({"key": key, "items": bucket})
    if not sections:
        raise ValueError(f"{where}: no categories, so nothing to show")
    return {"version": version, "date": date, "sections": sections}


def load_changelog(directory: Path, current: str, commit_url: str = COMMIT_URL) -> dict:
    """Compile every release-notes file into the payload served as changelog.json.

    Parameters
    ----------
    directory
        `docs/changelog/`, holding one directory per released version.
    current
        The version in `VERSION`, which must have notes.
    commit_url
        Prefix a commit sha is appended to, ending in `/commit/`.

    Returns
    -------
    dict
        ``{"current", "commit_url", "categories", "releases"}``, releases sorted
        newest first, which is the order the popup renders them in.

    Raises
    ------
    click.ClickException
        If the current version has no notes, or a directory that holds a
        changelog.md is not named after a version, or any file fails to parse.
    """
    releases = []
    if directory.is_dir():
        for entry in sorted(directory.iterdir()):
            note = entry / "changelog.md"
            if not note.is_file():
                continue
            if not _VERSION.fullmatch(entry.name):
                raise click.ClickException(
                    f"{note}: {entry.name!r} is not a major.minor.patch directory name"
                )
            try:
                releases.append(parse_changelog(note.read_text(encoding="utf-8"),
                                                entry.name, str(note)))
            except ValueError as error:
                raise click.ClickException(str(error)) from error

    if not any(release["version"] == current for release in releases):
        raise click.ClickException(
            f"no release notes for version {current}: write "
            f"{directory / current / 'changelog.md'} (docs/changelog/README.md has "
            "the format, and an existing one has an example). This refuses rather "
            "than shipping the previous release's notes under a new version's name."
        )
    releases.sort(key=lambda release: version_key(release["version"]), reverse=True)
    return {
        "current": current,
        "commit_url": commit_url,
        "categories": {key: {"fr": fr, "en": en} for key, _, fr, en in CATEGORIES},
        "releases": releases,
    }


def write_changelog(payload: dict, out: Path) -> Path:
    """Write `<out>/changelog.json` and the `.gz` sibling Caddy serves.

    Parameters
    ----------
    payload
        What `load_changelog` returned.
    out
        `dist/`.

    Returns
    -------
    Path
        The JSON file written.

    Notes
    -----
    Same treatment as the index files: `docker/Caddyfile` enables `precompressed
    gzip`, so the sibling is served at no runtime CPU. mtime is zeroed so an
    unchanged compile produces identical bytes and rsync skips it.
    """
    path = out / "changelog.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    encoding="utf-8")
    with gzip.GzipFile(path.with_suffix(".json.gz"), "wb", mtime=0) as handle:
        handle.write(path.read_bytes())
    return path


def read_version(path: Path) -> str:
    """Read the one line of `VERSION`.

    Parameters
    ----------
    path
        The `VERSION` file at the repo root.

    Returns
    -------
    str
        The version, e.g. ``"0.1.0"``.

    Raises
    ------
    click.ClickException
        If the file is missing or does not hold a `major.minor.patch` string.

    Notes
    -----
    One file, one line, read by everything that needs the number: this script, the
    `app-version.js` that `stage.py` generates, and whatever stamps a release
    later. The alternative, a constant in a script plus a directory name plus an
    environment variable, is three places to disagree.
    """
    if not path.exists():
        raise click.ClickException(f"{path} does not exist; it holds the site's version")
    version = path.read_text(encoding="utf-8").strip()
    if not _VERSION.fullmatch(version):
        raise click.ClickException(f"{path} holds {version!r}, not a major.minor.patch version")
    return version


@click.command()
@click.option("--notes", type=click.Path(file_okay=False, path_type=Path),
              default=Path("docs/changelog"), show_default=True,
              help="One directory per released version, each with a changelog.md.")
@click.option("--version-file", type=click.Path(dir_okay=False, path_type=Path),
              default=Path("VERSION"), show_default=True,
              help="The version that must have notes.")
@click.option("--out", type=click.Path(path_type=Path), **in_corpus("dist"), help="Tree deploy.sh ships. Writes <out>/changelog.json.")
@click.option("--check", is_flag=True,
              help="Parse and report, write nothing. What a hook or a review runs.")
def main(notes: Path, version_file: Path, out: Path, check: bool) -> None:
    """Compile the release notes, or refuse to."""
    current = read_version(version_file)
    payload = load_changelog(notes, current)
    bullets = sum(len(section["items"])
                  for release in payload["releases"] for section in release["sections"])
    logger.info(f"version {current}: {len(payload['releases'])} documented releases, "
                f"{bullets} bullets, both languages present")
    if check:
        logger.success("release notes parse")
        return
    path = write_changelog(payload, out)
    logger.success(f"{path} written ({path.stat().st_size / 1e3:.1f} KB)")


if __name__ == "__main__":
    main()
