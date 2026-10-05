"""One pymupdf version everywhere: build scripts, tests, and the page container.

pymupdf extracts the page geometry the highlights are drawn from (chunk.py) and
cuts the restricted pages that are served (server/pages.py). A different version
in each place means the suite tests a library production does not run, which is
what happened while docker/pages.Dockerfile pinned 1.24.14 and every script
floated to the latest. Each pin is written where its runner reads it (a PEP 723
header, a Dockerfile ARG), so this test is what makes them one value.

Written by Claude Code (Opus 5.5).
"""

import re

from conftest import ROOT

HEADER_PIN = re.compile(r'^# dependencies = \[.*?"pymupdf(?P<spec>[^"]*)"', re.M)
DOCKER_PIN = re.compile(r"^ARG PYMUPDF_VERSION=(?P<version>\S+)$", re.M)


def test_every_pymupdf_pin_is_the_same_exact_version():
    pins = {}
    for path in sorted([*ROOT.glob("scripts/*.py"), *ROOT.glob("server/*.py"), ROOT / "tests" / "run.py"]):
        match = HEADER_PIN.search(path.read_text(encoding="utf-8"))
        if match:
            pins[path.relative_to(ROOT).as_posix()] = match.group("spec")
    docker = DOCKER_PIN.search((ROOT / "docker" / "pages.Dockerfile").read_text(encoding="utf-8"))
    assert docker, "docker/pages.Dockerfile no longer declares ARG PYMUPDF_VERSION"
    pins["docker/pages.Dockerfile"] = f"=={docker.group('version')}"
    assert "server/pages.py" in pins and "scripts/chunk.py" in pins and "tests/run.py" in pins
    assert all(spec.startswith("==") for spec in pins.values()), pins
    assert len(set(pins.values())) == 1, pins
