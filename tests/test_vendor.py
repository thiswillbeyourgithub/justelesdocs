"""scripts/vendor.py extracts only allow-listed members, and only inside vendor/.

Written by Claude Code (Opus 5.5).
"""

import zipfile

import click
import pytest

from conftest import load


def target(vendor):
    return vendor.Vendored(name="lib", version="1", url="https://example.invalid/a.zip",
                           sha256="0" * 64, members=("build/",), strip=("build/",))


def archive(tmp_path, members):
    path = tmp_path / "a.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def test_allow_listed_members_land_under_vendor_and_the_rest_stays_out(tmp_path):
    vendor = load("vendor")
    zipped = archive(tmp_path, {"build/pdf.mjs": "x", "web/viewer.html": "y"})
    vendor.extract(target(vendor), zipped, tmp_path / "vendor")
    assert (tmp_path / "vendor" / "lib" / "pdf.mjs").read_text() == "x"
    assert not list((tmp_path / "vendor").rglob("viewer.html"))


def test_a_member_climbing_out_of_vendor_refuses_the_archive(tmp_path):
    """"build/../../escape" passes the allow-list's prefix test; it must not be written."""
    vendor = load("vendor")
    zipped = archive(tmp_path, {"build/../../escape.txt": "pwned"})
    with pytest.raises(click.ClickException, match="outside"):
        vendor.extract(target(vendor), zipped, tmp_path / "vendor")
    assert not (tmp_path / "escape.txt").exists()
