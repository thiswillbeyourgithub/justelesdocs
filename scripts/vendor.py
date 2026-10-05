# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "loguru"]
# ///
"""Fetch and verify the vendored front-end dependencies.

Dependencies are vendored rather than loaded from a CDN. That is a hard
requirement, not a preference: `docker/Caddyfile` serves a `default-src 'self'`
Content-Security-Policy with no external origins, so a CDN script would simply be
blocked. It also means a compromised CDN cannot reach this site's readers.

The cost of vendoring is that upgrades rot silently, so this script exists to make
them a single command with a checked result. It is not a package manager: it
downloads one pinned archive, verifies its SHA-256 against the constant below,
and extracts an explicit ALLOW-LIST of paths. Anything not on that list stays out,
which is how a 17 MB release archive becomes a 3.6 MB vendored tree.

To upgrade: change `PDFJS` (version, url, sha256), run this with `--force`, check
what changed with `git diff --stat`, and read the release notes for anything that
touches the worker contract or the CSP.

Written by Claude Code.
"""

from __future__ import annotations

import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import click
from loguru import logger

from lib import hashing


@dataclass(frozen=True)
class Vendored:
    """One pinned archive and the subset of it that gets vendored.

    Attributes
    ----------
    name
        Directory under `vendor/` to populate.
    version
        Upstream release, recorded so the tree's provenance is readable.
    url
        Exact archive to download. Pinned to a release asset, never a branch.
    sha256
        Digest of the archive. A mismatch aborts: this is the whole point of
        recording it, and "the upstream re-tagged a release" is indistinguishable
        from "someone replaced the asset".
    members
        Allow-list of paths inside the archive. A trailing `/` takes the whole
        subtree. Everything else is discarded.
    strip
        Leading path components to drop when writing (so `build/pdf.mjs` lands at
        `vendor/pdfjs/pdf.mjs`). Applied per member prefix, see `_destination`.
    """

    name: str
    version: str
    url: str
    sha256: str
    members: tuple[str, ...]
    strip: tuple[str, ...] = field(default=())


PDFJS = Vendored(
    name="pdfjs",
    version="4.10.38",
    url="https://github.com/mozilla/pdf.js/releases/download/v4.10.38/pdfjs-4.10.38-dist.zip",
    sha256="32bdd9c5198b77dbaa8f02de81f34476888b3abdd64b9fb5f607f81f01487e6a",
    members=(
        # The API and its worker. Both are ES modules; app code sets
        # GlobalWorkerOptions.workerSrc to the second one, served same-origin so
        # the CSP's `worker-src 'self'` covers it.
        "build/pdf.mjs",
        "build/pdf.worker.mjs",
        # Liberation fonts, used when a PDF references one of the standard 14
        # faces without embedding it. Several of the older French documents do.
        # Without these, such pages render with wrong metrics or not at all.
        "web/standard_fonts/",
        # Apache 2.0 requires the licence to travel with the code.
        "LICENSE",
    ),
    strip=("build/", "web/"),
)

# Deliberately NOT vendored, recorded so the omissions read as decisions:
#   build/*.map          5+ MB of source maps, for debugging pdf.js itself.
#   build/pdf.sandbox.*  Executes JavaScript embedded in PDFs. The corpus has no
#                        interactive forms, and the CSP would block it anyway.
#   web/cmaps/           1.1 MB of CJK character maps. The corpus is French and
#                        English. A CJK document would render with missing glyphs;
#                        add this subtree if one ever arrives.
#   web/viewer.*         Mozilla's own viewer UI. This site draws its own, because
#                        it has to overlay chunk highlight boxes and cannot use a
#                        viewer that owns the whole page.
TARGETS = (PDFJS,)


def _destination(member: str, target: Vendored) -> str:
    """Strip the configured leading components from an archive path."""
    for prefix in target.strip:
        if member.startswith(prefix):
            return member[len(prefix):]
    return member


def fetch(target: Vendored, cache: Path) -> Path:
    """Download the archive unless a verified copy is already cached.

    Parameters
    ----------
    target
        What to fetch.
    cache
        Directory to keep the archive in, so a re-run does not re-download.

    Returns
    -------
    pathlib.Path
        The verified archive.

    Raises
    ------
    click.ClickException
        If the download's digest does not match `target.sha256`.
    """
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / f"{target.name}-{target.version}.zip"
    if not archive.exists():
        logger.info(f"downloading {target.url}")
        with urllib.request.urlopen(target.url) as response:
            archive.write_bytes(response.read())
    digest = hashing.file_sha256(archive)
    if digest != target.sha256:
        archive.unlink()
        raise click.ClickException(
            f"{target.name} {target.version}: SHA-256 is {digest}, expected "
            f"{target.sha256}. The archive was replaced upstream or the download "
            "was tampered with. Do NOT vendor it; verify by hand first."
        )
    logger.success(f"{target.name} {target.version}: digest verified")
    return archive


def extract(target: Vendored, archive: Path, root: Path) -> int:
    """Replace `root/<name>/` with the allow-listed members of `archive`.

    Parameters
    ----------
    target
        What is being vendored.
    archive
        Verified zip from :func:`fetch`.
    root
        The `vendor/` directory.

    Returns
    -------
    int
        Bytes written.

    Notes
    -----
    The destination is removed first. Extracting over an existing tree would leave
    files that a new upstream release dropped, and a stale `pdf.worker.mjs` beside
    a new `pdf.mjs` is a version skew that fails at runtime with an unhelpful
    message.
    """
    destination = root / target.name
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    written = 0
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if not any(info.filename == m or info.filename.startswith(m)
                       for m in target.members if m.endswith("/") or m == info.filename):
                # Not on the allow-list. Checked as "exact match, or inside an
                # allow-listed subtree", so a member like "build/pdf.mjs" cannot
                # accidentally admit "build/pdf.mjs.map".
                continue
            relative = _destination(info.filename, target)
            out = destination / relative
            # An allow-listed subtree admits any name under it, "build/../../x"
            # included. The archive's sha256 is pinned, so this would take a
            # malicious upstream release that we then pinned, but the check is one
            # line and the alternative is writing outside vendor/ on a re-vendor.
            if not out.resolve().is_relative_to(destination.resolve()):
                raise click.ClickException(
                    f"{info.filename!r} in {archive.name} would extract outside "
                    f"{destination}; refusing the whole archive.")
            out.parent.mkdir(parents=True, exist_ok=True)
            data = zf.read(info)
            out.write_bytes(data)
            written += len(data)
    (destination / "VERSION").write_text(
        f"{target.name} {target.version}\n{target.url}\nsha256 {target.sha256}\n",
        encoding="utf-8",
    )
    return written


@click.command()
@click.option("--vendor", "vendor_dir", type=click.Path(path_type=Path),
              default=Path("vendor"), show_default=True,
              help="Where the vendored trees live. Tracked in git on purpose.")
@click.option("--cache", type=click.Path(path_type=Path),
              default=Path(tempfile.gettempdir()) / "justelesdocs-vendor-cache",
              show_default=True,
              help="Download cache. Outside the repo on purpose, so it needs no "
                   ".gitignore entry. Safe to delete; a re-run re-downloads.")
@click.option("--force", is_flag=True,
              help="Re-extract even when the tree already carries the right VERSION.")
def main(vendor_dir: Path, cache: Path, force: bool) -> None:
    """Vendor the pinned front-end dependencies."""
    for target in TARGETS:
        stamp = vendor_dir / target.name / "VERSION"
        if stamp.exists() and target.version in stamp.read_text(encoding="utf-8") and not force:
            logger.info(f"{target.name} {target.version} already vendored; --force to redo")
            continue
        archive = fetch(target, cache)
        written = extract(target, archive, vendor_dir)
        logger.success(f"{target.name} {target.version}: {written / 1e6:.1f} MB "
                       f"in {vendor_dir / target.name}")


if __name__ == "__main__":
    main()
