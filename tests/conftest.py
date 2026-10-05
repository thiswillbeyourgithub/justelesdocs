"""Make the PEP 723 scripts in `scripts/` importable from the tests.

They are standalone files run through `uv run`, not a package, so there is no
import path to them. Loading each by path is what `stage.py` already does to reach
`changelog.py`, and doing the same here means the tests exercise the file that
actually runs rather than a copy.

Written by Claude Code (Opus 5).
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# The caller's corpus, captured at import, before `empty_corpus` repoints CORPUS_DIR
# at an empty directory for every test. None when the suite runs with no corpus
# (a fresh clone of the software, CI), and the tests that read it then skip.
_corpus = os.environ.get("CORPUS_DIR", "").strip()
REAL_CORPUS = Path(_corpus).resolve() if _corpus else None

# The configuration every test's corpus carries: the software's own example, so the
# suite needs no corpus of its own. Modules that read corpus.toml at import
# (lib/manifest_policy.py) may be imported by a session fixture before any
# per-test fixture has run, so CORPUS_DIR points at a corpus holding it from the
# moment this file is imported, and `empty_corpus` gives each test a fresh one.
EXAMPLE_CONFIG = ROOT / "corpus.example" / "corpus.toml"


def example_corpus(directory: Path) -> Path:
    """`directory` made into a corpus with only the example configuration in it."""
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(EXAMPLE_CONFIG, directory / "corpus.toml")
    return directory


os.environ["CORPUS_DIR"] = str(example_corpus(Path(tempfile.mkdtemp(prefix="corpus-"))))


def load(name: str):
    """Import `scripts/<name>.py` and return the module.

    Parameters
    ----------
    name
        The script's basename, without `.py`.

    Returns
    -------
    module
        The imported module, also registered in `sys.modules` under `name` so a
        second load is cheap and identities compare equal.
    """
    if name in sys.modules:
        return sys.modules[name]
    # scripts/ goes on the path because a couple of the scripts import their shared
    # helpers as `lib.evalbake`, which is what uv gives them at run time (the script's
    # own directory is sys.path[0]) and what importlib alone would not.
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, f"cannot import {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_lib(name: str):
    """Import `scripts/lib/<name>.py`, one of the modules the scripts share.

    A real package import rather than `load`\'s by-path one, because that is how
    the scripts reach it: uv puts a script\'s own directory on sys.path, so
    `from lib.x import y` resolves from `scripts/`.
    """
    import importlib
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    return importlib.import_module(f"lib.{name}")


def cover(tmp_path, lines, *, name="cover.pdf"):
    """Write a one-page PDF whose first page holds `lines`.

    Shared by test_manifest.py's title-heuristic tests and by a corpus's own policy tests.

    Parameters
    ----------
    tmp_path : Path
        pytest's temporary directory.
    lines : list[tuple[float, float, str]]
        One ``(y, fontsize, text)`` per line, in any order.
    name : str
        The filename to write.

    Returns
    -------
    Path
        The written PDF.
    """
    import fitz

    path = tmp_path / name
    doc = fitz.open()
    page = doc.new_page()
    for top, size, text in lines:
        page.insert_text((60, top), text, fontsize=size)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture(autouse=True)
def empty_corpus(tmp_path_factory, monkeypatch):
    """Point CORPUS_DIR at an empty corpus (the example corpus.toml alone) for every test.

    Click resolves every option's default when a command runs, used or not, and
    every corpus default needs CORPUS_DIR (scripts/lib/corpus.py). An empty corpus
    rather than the real one, so a test that forgot to pass a path fails on a
    missing file instead of quietly reading the private corpus. A test about the
    unset variable deletes it with monkeypatch.
    """
    monkeypatch.setenv("CORPUS_DIR", str(example_corpus(tmp_path_factory.mktemp("corpus"))))


@pytest.fixture(scope="session")
def changelog():
    return load("changelog")


@pytest.fixture(scope="session")
def manifest():
    return load("manifest")


@pytest.fixture(scope="session")
def catalog():
    return load("catalog")


@pytest.fixture(scope="session")
def chunk():
    return load("chunk")


@pytest.fixture(scope="session")
def figures():
    return load("figures")


@pytest.fixture(scope="session")
def check_pdfs():
    return load("check_pdfs")


@pytest.fixture(scope="session")
def sample_passages():
    return load("sample_passages")


@pytest.fixture(scope="session")
def build_index():
    return load("build_index")


@pytest.fixture(scope="session")
def stage():
    return load("stage")


@pytest.fixture(scope="session")
def verify_chunks():
    return load("verify_chunks")


@pytest.fixture(scope="session")
def ocr():
    return load("ocr")


@pytest.fixture(scope="session")
def bench():
    return load("bench")


@pytest.fixture(scope="session")
def check_served():
    """scripts/check_served.py, the gate on what the web root may hold."""
    return load("check_served")


@pytest.fixture(scope="session")
def review_manifest():
    """scripts/review_manifest.py, the second pass over the shallow classifications."""
    return load("review_manifest")


@pytest.fixture(scope="session")
def grid_report():
    """scripts/grid_report.py, the chunking sweep's arithmetic."""
    return load("grid_report")


@pytest.fixture(scope="session")
def eval_queries():
    return load("eval_queries")


@pytest.fixture(scope="session")
def language_axis():
    return load("eval_language_axis")


@pytest.fixture(scope="session")
def evalbake():
    """scripts/lib/evalbake.py, the helpers the two bake experiments share."""
    return load_lib("evalbake")


@pytest.fixture(scope="session")
def manifest_io():
    """scripts/lib/manifest_io.py, the one reader of data/MANIFEST.tsv."""
    return load_lib("manifest_io")


@pytest.fixture(scope="session")
def vocabulary():
    """scripts/lib/vocabulary.py, the closed-vocabulary gate both scripts run."""
    return load_lib("vocabulary")


@pytest.fixture(scope="session")
def embed():
    """embed.py, imported with a stub in place of onnxruntime.

    The script imports onnxruntime at module level but only calls it inside
    functions, and the real package is a ~300 MB download that would be installed
    for every run of the suite. The stub keeps `uv run tests/run.py` cheap on a
    machine that will never bake anything.
    """
    sys.modules.setdefault("onnxruntime", types.ModuleType("onnxruntime"))
    return load("embed")


@pytest.fixture(scope="session")
def root() -> Path:
    """The repository root, for the tests that read a real tracked file."""
    return ROOT


@pytest.fixture(scope="session")
def real_corpus() -> Path:
    """The real corpus named by the caller's CORPUS_DIR, for the tests that check it.

    Skips when CORPUS_DIR was unset: those tests are about the corpus, not the
    software, and a checkout without one has nothing for them to read.
    """
    if REAL_CORPUS is None:
        pytest.skip("CORPUS_DIR not set: no corpus to check")
    return REAL_CORPUS


@pytest.fixture(scope="session")
def figure_record():
    """scripts/lib/figure_record.py, the figure descriptions read back as chunks."""
    return load_lib("figure_record")


# --- chunking helpers ---------------------------------------------------------
# Shared by the test_chunk_*.py files, one per lib/chunk_*.py stage, and by the
# golden test. They take the `chunk` module as an argument rather than importing
# it, so every test still reaches the code through the one fixture.


class FakeEncoding:
    """What `tokenizers` returns, reduced to the one attribute chunk.py reads."""

    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class WordTokenizer:
    """One token per word, which makes every budget in these tests countable."""

    def encode_batch(self, texts: list[str]) -> list[FakeEncoding]:
        return [FakeEncoding(list(range(len(text.split())))) for text in texts]


def flow_row(chunk, page: int, y: float, x0: float, x1: float, text: str, edge: float):
    """One row of a flow whose right margin is `edge`, for the line-break rule."""
    return chunk.Row(page=page, bbox=(x0, y, x1, y + 12.0), text=text, size=10.0,
                     right_edge=edge)


def line(chunk, index: int, text: str, page: int = 0) -> object:
    """One body row on `page`, stacked under the previous one."""
    return chunk.Row(page=page, bbox=(72.0, 100.0 + 12 * index, 500.0, 112.0 + 12 * index),
                     text=text, size=10.0)
