"""Every tmpfs in docker/docker-compose.yml carries an explicit `mode=`.

All the services run as the host uid, not root, so a tmpfs is only writable to
them if it is world-writable. Docker makes a bare `- /gen` mode 1777 when it
creates the container, but remounts it root-owned 0755 when the restart policy
restarts the container (docker-ce 29.9.3, 2026-11-04: the daemon restart during
an unattended upgrade left the web container restart-looping on "can't create
/gen/app-config.js: Permission denied"). A written mode survives the restart.

Written by Claude Code (Opus 5.5).
"""

import re

from conftest import ROOT

COMPOSE = ROOT / "docker" / "docker-compose.yml"


def tmpfs_entries(text: str) -> list[str]:
    """The list items of every `tmpfs:` block, comments skipped."""
    entries, inside = [], False
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^\s+tmpfs:\s*$", line):
            inside = True
            continue
        if not inside or not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- "):
            entries.append(stripped[2:].strip())
        else:
            inside = False
    return entries


def test_compose_has_tmpfs_mounts():
    assert tmpfs_entries(COMPOSE.read_text()), "parser found no tmpfs block"


def test_every_tmpfs_sets_its_mode():
    bare = [e for e in tmpfs_entries(COMPOSE.read_text()) if "mode=" not in e]
    assert not bare, f"tmpfs without an explicit mode= (0755 after a restart): {bare}"


def test_parser_sees_a_bare_tmpfs():
    sample = "    tmpfs:\n      # why\n      - /tmp:mode=1777\n      - /gen\n    cap_drop:\n      - ALL\n"
    assert tmpfs_entries(sample) == ["/tmp:mode=1777", "/gen"]
