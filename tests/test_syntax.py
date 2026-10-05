"""Every Python and shell file outside the site, parsed.

Most of `scripts/` is reached by some test through `conftest.load`, but not all of
it: the evaluation scripts, the grid tools and the page service's entry points are
run by hand, so a syntax error in one passes the whole suite and surfaces only when
someone needs that script, possibly weeks after the commit that broke it. This
compiles each file without importing it, so no dependency has to be installed and
no module-level code runs. tests/syntax.test.mjs is the JavaScript half.

Written by Claude Code (Opus 5.5).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

PYTHON = sorted(
    path
    for folder in ("scripts", "scripts/lib", "server", "server/lib", "tests")
    for path in (ROOT / folder).glob("*.py")
)
# The shell gates and the deploy hook. deploy.sh is gitignored and environment
# specific, so it is not listed: a clone without it must still pass.
SHELL = sorted([*ROOT.glob("scripts/*.sh"), *ROOT.glob("*.sh"), ROOT / ".githooks" / "pre-push"])
SHELL = [path for path in SHELL if path.exists() and path.name != "deploy.sh"]


@pytest.mark.parametrize("path", PYTHON, ids=lambda p: str(p.relative_to(ROOT)))
def test_python_compiles(path: Path) -> None:
    """`compile` parses the file the way the interpreter would, without running it."""
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


@pytest.mark.parametrize("path", SHELL, ids=lambda p: str(p.relative_to(ROOT)))
def test_shell_parses(path: Path) -> None:
    """`-n` reads the script without executing it, with the shell its shebang names."""
    shebang = path.read_text(encoding="utf-8").splitlines()[0]
    shell = "bash" if "bash" in shebang else "sh"
    if shutil.which(shell) is None:
        pytest.skip(f"{shell} is not installed")
    result = subprocess.run([shell, "-n", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
