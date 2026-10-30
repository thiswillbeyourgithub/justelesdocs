"""What a corpus says about the site that serves it: name, branding, languages, strings.

The software's pages (`src/*.html`) and scripts name no corpus. Everything a reader
sees that belongs to ONE corpus comes from two places in `$CORPUS_DIR`:

- `corpus.toml`, table `[site]`: the site id (the localStorage prefix), its name and
  two-part brand, the meta descriptions, the languages offered and the default,
  the repository the footer credits, the author it names, an optional logo, and
  `legacy_keys`, the localStorage keys a site used before it had a prefix (old
  name -> its name under the prefix), which `store.js` moves once.
- `strings/<lang>.json`, one flat object per language: an OVERLAY on the
  software's own string table in `src/i18n.js`. Any key it carries replaces the
  software's; keys the software does not have (facet value labels, tier labels)
  are simply added. A missing file is a language with no overlay, not an error.

`scripts/stage.py` turns both into the served tree: it renders the `{{name}}`
placeholders in the HTML pages (so the title and brand are in the markup, readable
without JavaScript) and writes `site-config.js`, which sets `window.__SITE__` for
`store.js`, `i18n.js` and `site.js`.

Module imported, not run. Standard library only, plus lib.corpus_config.

Written by Claude Code (Opus 5.5).
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

from lib import corpus_config
from lib.corpus import corpus_path

# An id becomes a localStorage key prefix and nothing else, so it is kept to
# characters that read unambiguously in a key and in a log line.
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
# A `{{key}}` in a page template. Lowercase and underscores only, so a stray pair
# of braces in a comment is not mistaken for one.
PLACEHOLDER = re.compile(r"\{\{([a-z_]+)\}\}")


# The software's own repository: what the footer credits and where the changelog's
# commit links point when a corpus names no repository of its own ([site] repo_url).
# Held HERE only: stage.py writes it into site-config.js and passes it to
# changelog.py, so site.js and changelog.py carry no copy that could drift.
SOFTWARE_REPO_URL = "https://github.com/thiswillbeyourgithub/justelesdocs"

def load_site() -> dict[str, Any]:
    """The `[site]` table, validated and with every default filled in.

    Returns
    -------
    dict
        Keys: id, name, brand (three strings: plain part, emphasised part, plain tail),
        description, browse_description, languages, default_language, repo_url,
        url (the public origin, or ""), author, logo (a Path, or None).

    Raises
    ------
    ValueError
        On a malformed value, naming the key and the file.
    """
    raw = corpus_config.section("site")
    where = corpus_config.path_of()

    def text(key: str, default: str | None = None) -> str:
        value = raw.get(key, default)
        if not isinstance(value, str) or (default is None and not value.strip()):
            raise ValueError(f"{where}: [site] {key} must be a non-empty string")
        return value

    site_id = text("id")
    if not ID_PATTERN.match(site_id):
        raise ValueError(f"{where}: [site] id {site_id!r} must match {ID_PATTERN.pattern}")
    name = text("name")
    brand = raw.get("brand", [name])
    if (not isinstance(brand, list) or not 1 <= len(brand) <= 3
            or not all(isinstance(part, str) for part in brand)):
        raise ValueError(f"{where}: [site] brand must be one to three strings")
    languages = raw.get("languages", ["fr", "en"])
    if (not isinstance(languages, list) or not languages
            or not all(isinstance(code, str) and code for code in languages)):
        raise ValueError(f"{where}: [site] languages must be a non-empty list of codes")
    default_language = text("default_language", languages[0])
    if default_language not in languages:
        raise ValueError(f"{where}: [site] default_language {default_language!r} "
                         f"is not one of {languages}")
    description = text("description", "")
    # Where the site is served, for the links in llms.txt and the /api/doc/ pages,
    # which an assistant may quote out of context. Optional: without it those links
    # are root-relative, which resolve against wherever the file was fetched from.
    url = text("url", "").rstrip("/")
    if url and not url.startswith(("https://", "http://")):
        raise ValueError(f"{where}: [site] url must start with https:// or http://, got {url!r}")
    logo = raw.get("logo")
    if logo is not None:
        logo = corpus_path(text("logo"))
        if logo.suffix != ".svg" or not logo.is_file():
            raise ValueError(f"{where}: [site] logo must be an existing .svg file, got {logo}")
    legacy_keys = raw.get("legacy_keys", {})
    if (not isinstance(legacy_keys, dict)
            or not all(isinstance(k, str) and isinstance(v, str) and k and v
                       for k, v in legacy_keys.items())):
        raise ValueError(f"{where}: [site] legacy_keys must map old key names to new ones")
    return {
        "id": site_id,
        "name": name,
        "brand": brand + [""] * (3 - len(brand)),
        "description": description,
        "browse_description": text("browse_description", description),
        "languages": languages,
        "default_language": default_language,
        "repo_url": text("repo_url", SOFTWARE_REPO_URL),
        "url": url,
        "author": text("author", ""),
        "logo": logo,
        "legacy_keys": legacy_keys,
    }


# The word before the author's name in `byline`, per default language; "by" otherwise.
BYLINE_WORD = {"fr": "par"}


def byline(site: dict[str, Any]) -> str:
    """The line every Markdown page for AI assistants starts with, or "" without an author.

    For example "psychiathèque, par Jane Doe (https://docs.example.org)", the URL only
    when [site] url is set: a page an assistant reads, or that ends up in a training
    set, then names the site, who made it and where it lives.
    """
    if not site["author"]:
        return ""
    where = f" ({site['url']})" if site["url"] else ""
    return f"{site['name']}, {BYLINE_WORD.get(site['default_language'], 'by')} {site['author']}{where}"


def load_strings(languages: list[str], directory: Path) -> dict[str, dict[str, str]]:
    """Every language's overlay from `directory/<lang>.json`, skipping absent files.

    Parameters
    ----------
    languages : list of str
        The site's languages; only their files are read.
    directory : Path
        Usually `$CORPUS_DIR/strings`.

    Returns
    -------
    dict
        Language code to a flat {key: string} table.

    Raises
    ------
    ValueError
        When a file is not one flat object of strings: a nested value or a number
        would reach the page as "[object Object]" or crash `replaceAll`.
    """
    overlay: dict[str, dict[str, str]] = {}
    for code in languages:
        path = directory / f"{code}.json"
        if not path.is_file():
            continue
        table = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(table, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in table.items()):
            raise ValueError(f"{path} must be one flat JSON object of strings")
        overlay[code] = table
    return overlay


def page_values(site: dict[str, Any]) -> dict[str, str]:
    """The placeholders a page template may use, unescaped."""
    return {
        "name": site["name"],
        "brand": site["brand"][0],
        "brand_em": site["brand"][1],
        "brand_after": site["brand"][2],
        "description": site["description"],
        "browse_description": site["browse_description"],
        "lang": site["default_language"],
    }


def render_page(template: str, values: dict[str, str]) -> str:
    """`template` with every `{{key}}` replaced by its HTML-escaped value.

    Raises
    ------
    KeyError
        On a placeholder with no value, rather than shipping the braces: a page
        titled "{{name}}" looks like a typo, not like a missing config key.
    """
    def fill(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise KeyError(f"page placeholder {{{{{key}}}}} has no value; "
                           f"known: {', '.join(sorted(values))}")
        return html.escape(values[key], quote=True)
    return PLACEHOLDER.sub(fill, template)


# The PNG icons `web_manifest` lists, file name to edge in pixels, which stage.py
# rasterises from the site's logo. 192 and 512 are the two sizes Chrome requires
# before it offers to install; 180 is the one iOS reads from <link rel="apple-touch-icon">,
# which ignores the manifest's icons and SVG alike.
ICONS = {"icon-192.png": 192, "icon-512.png": 512, "apple-touch-icon.png": 180}
# The same two sizes again for Android's adaptive icons (`"purpose": "maskable"`), which
# crop the icon to a circle, squircle or the like. Only the central circle of 80% of the
# edge is sure to show, so the logo is drawn at MASKABLE_SCALE of the edge, centred on a
# full-bleed backdrop: without these, Android shrinks the plain icon into a white disc.
MASKABLE_ICONS = {"icon-maskable-192.png": 192, "icon-maskable-512.png": 512}
MASKABLE_SCALE = 0.7
# The icons' backdrop and the installed window's colours: the light theme's --bg in
# src/app.css. A transparent icon would be laid on black by iOS.
ICON_BACKGROUND = "#ffffff"


def web_manifest(site: dict[str, Any]) -> str:
    """The body of `manifest.webmanifest`, which makes the site installable as an app.

    Every URL in it is relative, and a manifest resolves them against its own URL,
    so the site installs the same whether it is served at a domain's root or under
    a path. There is no service worker: no current browser requires one to install,
    and a cached copy of the index or the PDFs is exactly what the Cache-Control
    note in docker/Caddyfile rules out.
    """
    payload = {
        "name": site["name"],
        "short_name": site["name"],
        "description": site["description"],
        "lang": site["default_language"],
        "start_url": "./",
        "scope": "./",
        "display": "standalone",
        "background_color": ICON_BACKGROUND,
        "theme_color": ICON_BACKGROUND,
        "icons": [{"src": name, "sizes": f"{size}x{size}", "type": "image/png"}
                  for name, size in ICONS.items() if name.startswith("icon-")]
                 + [{"src": name, "sizes": f"{size}x{size}", "type": "image/png",
                     "purpose": "maskable"} for name, size in MASKABLE_ICONS.items()],
    }
    return json.dumps(payload, ensure_ascii=False, indent=1) + "\n"


def site_script(site: dict[str, Any], strings: dict[str, dict[str, str]]) -> str:
    """The body of `site-config.js`: `window.__SITE__`, everything the pages read."""
    payload = {key: site[key] for key in
               ("id", "name", "languages", "default_language", "repo_url", "author",
                "legacy_keys")}
    payload["strings"] = strings
    return ("/* Generated by scripts/stage.py from corpus.toml [site] and strings/*.json. */\n"
            f"window.__SITE__ = {json.dumps(payload, ensure_ascii=False, indent=1)};\n")
