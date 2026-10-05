# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "loguru"]
# ///
"""Assemble the tree that gets shipped, and nothing else.

`dist/` is what `deploy.sh` rsyncs to the VPS. Building it as an explicit step,
rather than letting the deploy script pick files out of `data/` with rsync
include rules, exists for one reason: **the file we serve must be the exact
file the highlight boxes were computed from.**

Fourteen documents in the corpus are served from `data/OCR/` instead of
`data/GUIDELINES/` because they had scanned pages with no text layer. pdf.js can
only highlight where a text layer exists, so serving the original while having
chunked the derivative would place every highlight at coordinates that mean
nothing in the reader's browser: the search would look like it works and quietly
point at the wrong part of the page.

That decision is NOT re-derived here. `chunk.py` writes `"source": "ocr"` or
`"source": "original"` into each chunk file, and this script reads it. One rule,
one place. If the rule is ever re-implemented in rsync flags or a shell loop, the
two copies will diverge on the day someone adds or removes an OCR derivative, and
the failure is invisible: the site still works, the highlights are just wrong.

Files are staged as **symlinks**, not copies, so `dist/` costs nothing on the dev
machine and never holds a stale duplicate of a PDF. `deploy.sh` rsyncs with `-L`
so the VPS receives real files.

`dist/` holds two trees, and the split is the whole point of the layout:

- `dist/www/` is what Caddy serves, assembled here.
- `dist/index/` is the search index, written by `build_index.py` and read by the
  search service (`server/service.mjs`) only. The vectors are never served: the
  browser asks the service for ranked results and downloads no matrix.

The page still needs two pieces of the index, `meta.json` (the document table
and facets) and `doc/` (per-document text and highlight boxes, one file at a
time, for the viewer), so `stage_index` links exactly those two into
`dist/www/index/`. Everything else in `dist/index/` stays outside the served
root, which is what makes it unreachable: a Caddy rule hiding the vector files
would fail open the day the index grows a fourth file.

Two files are GENERATED here rather than linked, and both are about the version:
`changelog.json`, compiled from `docs/changelog/` by `changelog.py` (which refuses
to compile when `VERSION` names a release with no notes, so a site cannot be
staged with somebody else's release notes in it), and `app-version.js`, the one
line that tells the page which version it is. Generating them at staging time is
what keeps the shipped notes and the shipped site the same age.

The corpus's branding is generated too (lib/site_config.py): the HTML pages are
written with their `{{name}}` placeholders filled from corpus.toml's `[site]`
table rather than linked, `site-config.js` carries the site id, languages and
string overlay (`$CORPUS_DIR/strings/<lang>.json`) to the scripts, and a corpus
logo, when it declares one, replaces the software's `logo.svg`.

Written by Claude Code.
"""

from __future__ import annotations

import importlib.util
import gzip
import json
import os
import shutil
import sys
from pathlib import Path

import click
from loguru import logger

from lib import manifest_io, site_config
from lib.corpus import corpus_path, in_corpus

# Where the two candidate sources live, relative to the repo root. Only these
# two directories may ever contribute a served PDF: GUIDELINES is the legally
# servable corpus (see CLAUDE.md) and OCR holds derivatives of its members.
SOURCE_DIRS = {"original": Path("data/GUIDELINES"), "ocr": Path("data/OCR")}


def corpus_or_software(name: str, software: Path) -> Path:
    """`$CORPUS_DIR/<name>` when it exists, else the software's own `software` path."""
    candidate = corpus_path(name)
    return candidate if candidate.exists() else software


def load_changelog_module() -> object:
    """Import `scripts/changelog.py` by path, so the release notes parse once.

    Returns
    -------
    module
        `changelog.py`, whose `read_version`, `load_changelog` and
        `write_changelog` this script calls.

    Notes
    -----
    By path rather than as a package because every script here is a standalone
    PEP 723 file with no package around it, and `changelog.py` declares the same
    two dependencies this one does, so it imports cleanly under `uv run`. The
    alternative, compiling the notes in a separate chain step, would let a site be
    staged with nobody's notes in it: staging is what writes dist/, so staging is
    where the notes have to be written, and where the gate has to refuse.
    """
    path = Path(__file__).resolve().parent / "changelog.py"
    spec = importlib.util.spec_from_file_location("changelog", path)
    if spec is None or spec.loader is None:
        raise click.ClickException(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["changelog"] = module
    spec.loader.exec_module(module)
    return module


def remove(path: Path) -> None:
    """Delete `path` whatever it is: a symlink, a file, or a directory.

    Parameters
    ----------
    path
        What to remove. A symlink is unlinked, never followed.

    Notes
    -----
    One spelling, because there were four and they disagreed. `Path.is_dir()`
    FOLLOWS a symlink, so the obvious `unlink() if not is_dir() else rmtree()` sends
    a symlink pointing at a directory to `shutil.rmtree`, which refuses it with
    "Cannot call rmtree on a symbolic link". Everything this script stages is a
    symlink, so that was the spelling most likely to meet one.

    Testing the symlink FIRST is what makes it right, and it is also what keeps a
    prune inside `dist/` from following a link out of `dist/` and deleting the
    corpus it points at.
    """
    if path.is_symlink() or not path.is_dir():
        path.unlink()
    else:
        shutil.rmtree(path)


def read_chunk_index(chunks: Path) -> dict[str, str]:
    """Map each document's filename to the source it must be served from.

    Parameters
    ----------
    chunks
        Directory of per-document chunk files written by `chunk.py`.

    Returns
    -------
    dict[str, str]
        Filename (with the `.pdf` extension) to `"original"` or `"ocr"`.

    Raises
    ------
    click.ClickException
        If a chunk file is unreadable, or declares a source that is not one of
        the two known directories. Both mean the build is inconsistent, and a
        deploy that guesses here ships wrong highlights.
    """
    index: dict[str, str] = {}
    for path in sorted(chunks.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            name, source = payload["file"], payload["source"]
        except (json.JSONDecodeError, KeyError) as exc:
            raise click.ClickException(
                f"{path} is not a usable chunk file ({exc}). Re-run chunk.py."
            ) from exc
        if source not in SOURCE_DIRS:
            raise click.ClickException(
                f"{path} declares source {source!r}, which is neither 'original' "
                "nor 'ocr'. chunk.py and stage.py have drifted apart."
            )
        index[name] = source
    return index


def read_access(manifest: Path) -> dict[str, str]:
    """Map each document's filename to its `access` tier.

    Parameters
    ----------
    manifest
        `data/MANIFEST.tsv`, whose `access` column carries the tier.

    Returns
    -------
    dict[str, str]
        Filename to `"open"` or `"restricted"`.

    Raises
    ------
    click.ClickException
        When the column is missing, or a row's tier is blank or neither value:
        the script cannot tell a book it may not redistribute from a guideline
        (`data/CORPUS_ADDITIONS.tsv` records which is which). The reading itself
        is `manifest_io.read_access`, shared with `check_served.py` so the stager
        and the gate that checks it cannot disagree about a cell.
    """
    try:
        return manifest_io.read_access(manifest)
    except ValueError as error:
        raise click.ClickException(str(error)) from error


def tier_of(tiers: dict[str, str], name: str) -> str:
    """The tier of one document, refusing a document the manifest does not list.

    Parameters
    ----------
    tiers
        Output of :func:`read_access`.
    name
        A document's filename.

    Returns
    -------
    str
        `"open"` or `"restricted"`.

    Raises
    ------
    click.ClickException
        When `name` has no manifest row. Defaulting it to `"open"` is how a
        restricted PDF renamed on disk would come back as a public file: the
        rename orphans its row, and the new name has none yet.
    """
    if name not in tiers:
        raise click.ClickException(
            f"{name} has no row in the manifest, so its access tier is unknown. "
            "Run manifest.py, then check its 'access' cell before staging.")
    return tiers[name]


def stage_pdfs(index: dict[str, str], destination: Path, tiers: dict[str, str],
               restricted_destination: Path) -> tuple[int, int, int]:
    """Symlink each PDF into the tree its `access` tier allows, pruning the rest.

    Open documents go to `destination`, inside the web root, and are served as
    plain files. Restricted ones go to `restricted_destination`, which is
    OUTSIDE the web root and is bind-mounted into the page service alone, so no
    URL returns the whole document: the reader gets the pages a hit allows, cut
    out of it per request. That split is the whole mechanism, and it lives here
    because staging is the last place that still knows which file is which.

    Parameters
    ----------
    index
        Output of :func:`read_chunk_index`.
    destination
        Directory to fill with the open documents, typically `dist/www/pdf`.
    tiers
        Output of :func:`read_access`.
    restricted_destination
        Directory to fill with the restricted documents, typically
        `dist/restricted`. Created if absent, and never under `dist/www`.

    Returns
    -------
    tuple[int, int, int]
        Documents staged as files, how many of those came from OCR, and how many
        were held back as restricted.

    Raises
    ------
    click.ClickException
        If a declared source file is missing or empty. Staging a corpus with a
        hole in it is the failure mode the brief explicitly forbids, and it is
        invisible at runtime: the document simply never appears in a result.
    """
    destination.mkdir(parents=True, exist_ok=True)
    restricted_destination.mkdir(parents=True, exist_ok=True)

    served = {n: s for n, s in index.items() if tier_of(tiers, n) == "open"}
    held = {n: s for n, s in index.items() if tier_of(tiers, n) == "restricted"}

    # Prune first, so a document removed from the corpus cannot survive in dist/
    # and keep being served after it was withdrawn for a legal reason. That is
    # the one staging bug with consequences beyond a broken page. A document that
    # was open and is now restricted prunes out of the web root by the same rule,
    # since it is no longer in `served`.
    for stale in destination.iterdir():
        if stale.name not in served:
            logger.warning(f"pruning {stale} (not in the chunk index, or now restricted)")
            remove(stale)
    for stale in restricted_destination.iterdir():
        if stale.name not in held:
            logger.warning(f"pruning {stale} (not in the chunk index, or now open)")
            remove(stale)

    ocred = 0
    for name, source in list(served.items()) + list(held.items()):
        target_dir = destination if name in served else restricted_destination
        origin = (corpus_path(SOURCE_DIRS[source]) / name).resolve()
        if not origin.exists() or origin.stat().st_size == 0:
            raise click.ClickException(
                f"{name} is declared as {source!r} but {origin} is missing or empty. "
                "Run ocr.py and chunk.py before staging."
            )
        link = target_dir / name
        # Re-point unconditionally: a document that switched from original to
        # OCR since the last stage must not keep the old target.
        if link.is_symlink() or link.exists():
            remove(link)
        link.symlink_to(origin)
        ocred += source == "ocr" and name in served
    return len(served), ocred, len(held)


def stage_assets(sources: list[Path], destination: Path) -> list[str]:
    """Symlink the site's own files into `destination`, pruning what left.

    Parameters
    ----------
    sources
        Directories whose contents become the site. `src/` is staged flat (so
        `src/index.html` is served at `/index.html`), while any other directory is
        staged as a directory of the same name (so `vendor/` is served at
        `/vendor/`). That asymmetry is deliberate: the pages have to sit at the
        site root for their relative URLs to work, and the vendored trees have to
        keep their own paths because pdf.js resolves its worker and font data
        relative to itself.
    destination
        `dist/`.

    Returns
    -------
    list[str]
        The names staged at the top level of `destination`, recorded in
        `.stage-manifest.json` so the next run can prune what it no longer stages.

    Notes
    -----
    Pruning is driven by that record rather than by guessing which top-level
    entries belong to the site. `dist/` also holds `pdf/`, `index/`, `MANIFEST.tsv`
    and the record itself, none of which this function owns, and a rule like
    "delete every symlink" would eventually delete one of them. Renaming a script
    without pruning leaves the old file served forever, which is how a stale
    `app.js` outlives the HTML that stopped loading it.
    """
    staged: list[str] = []
    for source in sources:
        if not source.exists():
            raise click.ClickException(
                f"{source} does not exist, so the site cannot be staged. For the "
                "vendored trees, run: uv run scripts/vendor.py"
            )
        entries = sorted(source.iterdir()) if source.name == "src" else [source]
        for entry in entries:
            if entry.name.startswith("."):
                continue
            link = destination / entry.name
            if link.is_symlink() or link.exists():
                # Re-point unconditionally, as for the PDFs: a name that changed
                # from a file to a directory upstream must not keep the old target.
                remove(link)
            link.symlink_to(entry.resolve())
            staged.append(entry.name)
    return staged


# `doc/` is deliberately NOT here: it is staged per document by
# `stage_doc_text`, because a restricted document's entry may not be published
# whole. Linking the directory, as this did until 2026-09-24, served the
# complete extracted text of every book in the corpus as one JSON per document.
PUBLIC_INDEX = ("meta.json", "meta.json.gz")


def stage_doc_text(index_dir: Path, destination: Path,
                   tiers: dict[str, str]) -> tuple[int, int]:
    """Publish per-document text, with the restricted documents stripped of it.

    `src/viewer.js` fetches `index/doc/<id>.json` to draw its highlights, so the
    file has to be public for the viewer to work at all. It reads only `boxes`
    and `pages` from each chunk, never `text`, so a restricted document can keep
    its geometry and lose its words: the highlight still lands on the right line,
    and the book's text stops being one request away. The search service reads
    the FULL files from its own mount of `dist/index`, not from here, so
    ranking and excerpts are unaffected.

    Parameters
    ----------
    index_dir
        The built index, typically `dist/index`.
    destination
        Directory to fill, typically `dist/www/index/doc`.
    tiers
        Output of :func:`read_access`.

    Returns
    -------
    tuple[int, int]
        Documents linked whole, and documents published stripped.

    Notes
    -----
    Stripped entries are written as real files rather than symlinks, so nothing
    under the web root points at a file that still holds the text. The `.gz`
    beside each one is rewritten to match: serving a stale pre-compressed copy
    would hand back exactly what the plain file no longer carries, since Caddy
    picks the `.gz` whenever the client accepts it.
    """
    destination.mkdir(parents=True, exist_ok=True)
    meta_path = index_dir / "meta.json"
    if not meta_path.exists():
        return 0, 0
    documents = json.loads(meta_path.read_text(encoding="utf-8")).get("documents", [])
    restricted_ids = {d["id"] for d in documents
                      if tier_of(tiers, d.get("file", "")) == "restricted"}
    known = {str(d["id"]) for d in documents}

    for stale in destination.iterdir():
        if stale.name.split(".")[0] not in known:
            remove(stale)

    linked = stripped = 0
    for document in documents:
        doc_id = document["id"]
        source = index_dir / "doc" / f"{doc_id}.json"
        if not source.exists():
            continue
        plain, packed = destination / f"{doc_id}.json", destination / f"{doc_id}.json.gz"
        for victim in (plain, packed):
            if victim.is_symlink() or victim.exists():
                remove(victim)
        if doc_id in restricted_ids:
            payload = json.loads(source.read_text(encoding="utf-8"))
            for chunk in payload.get("chunks", []):
                chunk.pop("text", None)
            # `separators` keeps the file as tight as the generated one, which
            # matters: this is per-document and the viewer fetches it on open.
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            plain.write_text(body, encoding="utf-8")
            packed.write_bytes(gzip.compress(body.encode("utf-8"), 9))
            stripped += 1
        else:
            for link, origin in ((plain, source), (packed, source.with_suffix(".json.gz"))):
                if origin.exists():
                    link.symlink_to(Path(os.path.relpath(origin.resolve(),
                                                         destination.resolve())))
            linked += 1
    return linked, stripped


def stage_index(index_dir: Path, destination: Path) -> bool:
    """Link the index's public half, `meta.json` and `doc/`, into the served tree.

    `meta.json.gz`, the precompressed sibling `build_index.py` writes for Caddy's
    `precompressed` file_server, is linked too when it is there: meta.json is the
    page's first request and the one file every visit downloads.

    Parameters
    ----------
    index_dir
        The index from `build_index.py`, normally `dist/index/`. The search
        service reads the whole of it; the browser only ever gets the two
        entries in `PUBLIC_INDEX`.
    destination
        `dist/www/index/`. Created if absent.

    Returns
    -------
    bool
        True when an index was there to link. False, with nothing linked and any
        earlier links removed, when `index_dir` holds no `meta.json`: a dangling
        link would make the page's first request a 404 that looks like a proxy
        fault rather than a missing build.

    Notes
    -----
    The links are RELATIVE (`../../index/meta.json`), so `dist/` can be moved or
    rsynced as a whole and they still resolve. `deploy.sh` rsyncs with `-L`, so
    the VPS gets real files; `doc/` is about 120 MB and arrives twice there, once
    under `index/` for the service and once under `www/index/` for the page. That
    duplication is the price of a served root that physically cannot reach the
    vectors, and it was preferred over a Caddy rule that hides them, which fails
    open the day the index grows a file the rule does not name.

    Only these two names are linked, never the directory: linking `dist/index`
    itself would serve every file in it, which is exactly what this split exists
    to prevent.
    """
    destination.mkdir(parents=True, exist_ok=True)
    present = (index_dir / "meta.json").exists()
    for name in PUBLIC_INDEX:
        link = destination / name
        if link.is_symlink() or link.exists():
            remove(link)
        if present and (index_dir / name).exists():
            # Relative to the link's own directory, computed from the resolved
            # paths so it is right whatever the caller's working directory.
            target = Path(os.path.relpath(index_dir.resolve() / name, destination.resolve()))
            link.symlink_to(target)
    if present:
        logger.success(f"index linked into {destination}: {', '.join(PUBLIC_INDEX)} "
                       f"(the vectors stay in {index_dir}, unserved)")
    return present


def brand_site(out: Path, assets: list[Path]) -> None:
    """Write the corpus's branding over the software's neutral pages.

    Runs AFTER `stage_assets`, which linked every `src/` file: each page and the
    logo are replaced here by a generated file or a link to the corpus's own, under
    the same name, so the pruning record still lists them and nothing is left
    behind when a corpus stops declaring a logo (the next run links the
    software's again first).

    Parameters
    ----------
    out : Path
        The served tree.
    assets : list of Path
        The asset directories `stage_assets` staged; the pages are the `.html`
        files of the one named `src`.
    """
    site = site_config.load_site()
    strings = site_config.load_strings(site["languages"], corpus_path("strings"))
    values = site_config.page_values(site)
    for source in assets:
        if source.name != "src":
            continue
        for page in sorted(source.glob("*.html")):
            target = out / page.name
            # The staged entry is a symlink INTO src/: writing through it would
            # overwrite the template itself, so it goes first.
            remove(target)
            target.write_text(site_config.render_page(page.read_text(encoding="utf-8"), values),
                              encoding="utf-8")
    (out / "site-config.js").write_text(site_config.site_script(site, strings), encoding="utf-8")
    if site["logo"] is not None:
        target = out / "logo.svg"
        if target.is_symlink() or target.exists():
            remove(target)
        target.symlink_to(site["logo"].resolve())
    logger.success(f"site {site['id']!r} ({site['name']}): languages "
                   f"{', '.join(site['languages'])}, string overlays for "
                   f"{', '.join(strings) or 'none'}"
                   f"{', corpus logo' if site['logo'] else ''}")


@click.command()
@click.option("--chunks", type=click.Path(exists=True, file_okay=False, path_type=Path),
              **in_corpus("data/chunks"),
              help="Chunk files from chunk.py. They record which file to serve.")
@click.option("--manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              **in_corpus("data/MANIFEST.tsv"),
              help="Document metadata, copied into dist for the UI's filters.")
@click.option("--out", type=click.Path(path_type=Path), **in_corpus("dist/www"), help="The served tree. deploy.sh ships its parent, dist/.")
@click.option("--index", "index_dir", type=click.Path(path_type=Path),
              **in_corpus("dist/index"),
              help="The search index from build_index.py. Only meta.json and doc/ "
                   "are linked into the served tree; the vectors never are.")
@click.option("--assets", multiple=True, type=click.Path(path_type=Path),
              default=(Path("src"), Path("vendor")), show_default=True,
              help="Directories that make up the site itself. src/ is staged "
                   "flat at the root, everything else keeps its directory name.")
@click.option("--notes", type=click.Path(file_okay=False, path_type=Path), default=None,
              help="Release notes, compiled into dist/changelog.json (see changelog.py). "
                   "Default: $CORPUS_DIR/changelog when the corpus has one, else the "
                   "software's docs/changelog.")
@click.option("--version-file", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="The site's version. Must have release notes, or staging refuses. "
                   "Default: $CORPUS_DIR/VERSION when the corpus has one, else VERSION.")
def main(chunks: Path, manifest: Path, out: Path, index_dir: Path, assets: tuple[Path, ...],
         notes: Path, version_file: Path) -> None:
    """Build the served tree from the chunk index, which decides what gets served."""
    # The release notes a reader sees are the SITE's, and a site is a corpus served
    # by this software: what changed for its readers is as much the corpus's story
    # (documents added, filters curated) as the software's. So a corpus with its own
    # changelog/ and VERSION gets those, and one without gets the software's. The
    # two travel together: notes from one place checked against the other's version
    # would refuse, or worse, pass with the wrong release's notes.
    notes = notes or corpus_or_software("changelog", Path("docs/changelog"))
    version_file = version_file or corpus_or_software("VERSION", Path("VERSION"))
    index = read_chunk_index(chunks)
    if not index:
        raise click.ClickException(
            f"no chunk files in {chunks}; run chunk.py first. Staging an empty "
            "corpus would deploy a site that finds nothing."
        )

    tiers = read_access(manifest)
    # `out` is dist/www; the restricted tree is its SIBLING, never a child, so no
    # Caddy root can reach it however the site's routes are rewritten later.
    restricted_dir = out.parent / "restricted"
    total, ocred, held = stage_pdfs(index, out / "pdf", tiers, restricted_dir)
    logger.success(f"{total} PDFs staged in {out / 'pdf'}, {ocred} of them from data/OCR")
    if held:
        logger.success(f"{held} restricted PDFs staged in {restricted_dir}, outside the "
                       "web root: the page service cuts pages out of them per request")

    # The manifest is copied rather than symlinked: it is 30 KB, it is tracked in
    # git, and the UI reads it directly, so a real file keeps dist/ meaningful
    # even when inspected without the repo around it.
    shutil.copy2(manifest, out / "MANIFEST.tsv")
    logger.info(f"copied {manifest} to {out / 'MANIFEST.tsv'}")

    record_path = out / ".stage-manifest.json"
    previous: list[str] = []
    if record_path.exists():
        try:
            previous = json.loads(record_path.read_text(encoding="utf-8")).get("assets", [])
        except (json.JSONDecodeError, AttributeError):
            # An unreadable record means nothing can be pruned safely, which is
            # better than deleting on a guess. Staging still proceeds.
            logger.warning(f"{record_path} is unreadable; skipping asset pruning")

    staged = stage_assets(list(assets), out)
    for stale in set(previous) - set(staged):
        victim = out / stale
        if victim.is_symlink() or victim.exists():
            logger.warning(f"pruning {victim} (no longer part of the site)")
            remove(victim)
    logger.success(f"{len(staged)} site entries staged in {out}: {', '.join(staged)}")

    brand_site(out, [Path(a) for a in assets])

    # The release notes, compiled into the tree they ship in. Done AFTER the asset
    # pruning above, because both write into dist/ at the top level and the pruner
    # only spares what it knows about: generated files are not in `staged`, so they
    # are written on the far side of it rather than deleted by it.
    changelog = load_changelog_module()
    version = changelog.read_version(version_file)
    repo_url = site_config.load_site()["repo_url"].rstrip("/")
    payload = changelog.load_changelog(notes, version, f"{repo_url}/commit/")
    path = changelog.write_changelog(payload, out)
    logger.success(f"{path.name}: version {version}, {len(payload['releases'])} "
                   f"documented releases ({path.stat().st_size / 1e3:.1f} KB)")

    # The version, as a script the pages load before their modules. It is generated
    # rather than checked in for the same reason /gen/app-config.js is: a value that
    # belongs to the build has no business being edited in two places, and
    # src/changelog.js is written to stay silent when it is absent, which is what a
    # plain file server over src/ (no dist/, no staging) gets.
    (out / "app-version.js").write_text(
        "/* Generated by scripts/stage.py from VERSION. */\n"
        f'window.__APP_VERSION__ = "{version}";\n', encoding="utf-8")

    record_path.write_text(json.dumps({
        "version": version,
        "documents": total,
        "from_ocr": ocred,
        "assets": staged,
        "sources": index,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    # The search index is built separately, by scripts/build_index.py, because it
    # depends on a bake (data/vectors/<variant>/) rather than on the chunk files
    # this script reads, and because rebuilding it is minutes of work while
    # restaging the site is milliseconds. Staging is therefore cheap to re-run
    # after an edit to src/ without touching the index. What IS done here is
    # linking the index's public half into the served tree.
    linked, stripped = stage_doc_text(index_dir, out / "index" / "doc", tiers)
    if linked or stripped:
        logger.success(f"per-document text: {linked} published whole, {stripped} published "
                       "without their text (access: restricted)")
    if not stage_index(index_dir, out / "index"):
        logger.warning(f"no search index in {index_dir}: the site will load but "
                       "cannot search. Build one with:\n"
                       "    uv run scripts/build_index.py --dims 1024")


if __name__ == "__main__":
    main()
