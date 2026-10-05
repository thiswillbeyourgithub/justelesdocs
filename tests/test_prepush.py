"""The one check in `.githooks/pre-push` that inspects the tree being pushed.

That check is the reason the hook exists: a pushed PDF cannot be unpushed, and 44
documents in this corpus carry explicit no-reposting licences. It is also the check
nothing else can cover, because the hook only runs on a push and a push that is
refused for the wrong reason is indistinguishable, from the user's side, from one
refused for the right one. It was refused for the wrong reason once: the pattern
matched a top-level table such as `data/SORTING_LOG.tsv`, a tracked record of the
sorting, because the folder name was not followed by a separator.

The pattern is read out of the hook and run through the real `grep -Ei`, so what is
tested is the line that runs rather than a copy of it that can drift.

Written by Claude Code (Opus 5).
"""

from __future__ import annotations

import os
import re
import subprocess

import pytest

# Paths the hook must refuse: the corpus itself, its derivatives, and any PDF
# anywhere, whatever folder someone drops it in.
MUST_REFUSE = [
    "data/GUIDELINES/annual_report_2024.pdf",
    "data/UNSURE/PAYWALLED/a paywalled standard.pdf",
    "data/UNSURE/TEACHING/slides 21.03.2023.pptx",
    "data/DISCARDED/LEGAL_ISSUE/a book chapter.pdf",
    "data/OCR/scanned_report.pdf",
    "data/chunks/10803_2017_Article_3166.json",
    "data/chunks-page/10803_2017_Article_3166.json",
    "dist/index/meta.json",
    "dist/pdf/anything",
    "models/Snowflake/model.onnx",
    "docs/a stray copy.PDF",
    "data/private/notes.tsv",
    "local/psydocs/corpus.toml",
]

# Paths the hook must let through: the tracked tables at the top level of data/,
# which carry filenames and page counts but no document content, and which
# data/README.md declares as the authoritative record of the sorting.
MUST_ALLOW = [
    "data/CORPUS_ADDITIONS.tsv",
    "data/SORTING_LOG.tsv",
    "data/SORTING_POLICY.md",
    "data/MANIFEST.tsv",
    "data/HAS_DUMP_2025-07.tsv",
    "data/README.md",
    "data/EVAL_QUERIES.tsv",
    "scripts/chunk.py",
    "CATALOG.md",
]


@pytest.fixture(scope="module")
def pattern(root) -> str:
    """The extended regex the corpus refusal greps the pushed paths with."""
    script = (root / ".githooks" / "refuse_corpus.sh").read_text()
    found = re.search(r"^pattern='([^']+)'$", script, flags=re.M)
    assert found, "the corpus refusal no longer sets `pattern='...'`"
    return found.group(1)


def refused(pattern: str, paths: list[str]) -> set[str]:
    """Run the hook's own grep over `paths` and return the ones it flags."""
    out = subprocess.run(["grep", "-Ei", pattern], input="\n".join(paths),
                         capture_output=True, text=True)
    return {line for line in out.stdout.splitlines() if line}


def test_the_corpus_and_its_derivatives_are_refused(pattern):
    assert refused(pattern, MUST_REFUSE) == set(MUST_REFUSE)


def test_the_tracked_sorting_records_are_not_refused(pattern):
    assert refused(pattern, MUST_ALLOW) == set()


def test_the_repository_as_it_stands_would_be_pushed(root, pattern):
    """The regression itself: HEAD must be pushable."""
    tracked = subprocess.run(["git", "ls-tree", "-r", "--name-only", "HEAD"],
                             cwd=root, capture_output=True, text=True, check=True)
    assert refused(pattern, tracked.stdout.splitlines()) == set()


# --- the curation nag ------------------------------------------------------------
# The hook counts documents with no `guideline` tier and asks before pushing. The
# counter is read out of the hook and run for real, the same way the refusal pattern
# above is: a copy of the awk in this file would pass while the hook's own drifted.


@pytest.fixture(scope="module")
def tier_counter(root) -> str:
    """The awk program the hook uses to count untiered documents."""
    hook = (root / ".githooks" / "pre-push").read_text()
    found = re.search(r"untiered=\"\$\(awk -v tier=\"\$tier\" -F'\\t' '(.*?)' \"\$corpus/data/MANIFEST\.tsv\"\)\"",
                      hook, re.S)
    assert found, "the untiered-document counter is no longer where the test reads it"
    return found.group(1)


def count_untiered(program: str, tsv: str, tier: str = "guideline") -> int:
    """Run the hook's own awk over a manifest given as text, `tier` the column name."""
    out = subprocess.run(["awk", "-v", f"tier={tier}", "-F", "\t", program, "-"],
                         input=tsv, capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


def test_blank_tiers_are_counted(tier_counter):
    """A blank cell is the state the site never hides, so it is the one to report."""
    tsv = ("file\ttitle\tguideline\n"
           "a.pdf\tA\tstrict\n"
           "b.pdf\tB\t\n"
           "c.pdf\tC\tno\n"
           "d.pdf\tD\t\n")
    assert count_untiered(tier_counter, tsv) == 2


def test_a_fully_curated_manifest_asks_nothing(tier_counter):
    """`no` is a curated answer, not a blank: it must not be counted as backlog."""
    tsv = "file\ttitle\tguideline\na.pdf\tA\tstrict\nb.pdf\tB\tno\n"
    assert count_untiered(tier_counter, tsv) == 0


def test_a_manifest_without_the_column_asks_nothing(tier_counter):
    """An old TSV, or one from another branch, must not make the hook talk.

    The column is found by NAME in the header rather than by position, which is also
    what lets `guideline` sit next to `doc_type` instead of at the end of the row.
    """
    tsv = "file\ttitle\tdoc_type\na.pdf\tA\trecommandation\n"
    assert count_untiered(tier_counter, tsv) == 0


def test_the_tier_column_is_the_corpus_s_own(tier_counter):
    """The column comes from corpus.toml [tiers] column, not a hardcoded name; a
    corpus with no tiers (an empty name) has nothing to count."""
    tsv = "file\tlevel\tguideline\na.pdf\t\tstrict\nb.pdf\t\t\n"
    assert count_untiered(tier_counter, tsv, tier="level") == 2
    assert count_untiered(tier_counter, tsv, tier="") == 0


# --- the refusal run end to end ----------------------------------------------
# The pattern tests above prove WHICH paths are refused. These prove WHERE the
# script looks: a PDF that one commit adds and a later commit deletes is gone from
# the tip tree but still travels with the push, and inspecting only the tip tree
# let exactly that through until 2026-10-03.

ZERO = "0" * 40


def git(repo, *args: str) -> str:
    """Run git in `repo` with a fixed identity and return stdout."""
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
           "PATH": os.environ["PATH"], "HOME": str(repo)}
    return subprocess.run(["git", *args], cwd=repo, env=env, check=True,
                          capture_output=True, text=True).stdout.strip()


def commit(repo, message: str, add: dict[str, str] = {}, remove: tuple[str, ...] = ()) -> str:
    """Write `add`, delete `remove`, commit everything, return the new sha."""
    for name, text in add.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        git(repo, "add", name)
    for name in remove:
        git(repo, "rm", "-q", name)
    git(repo, "commit", "-q", "--allow-empty", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def run_refusal(root, repo, local: str, remote: str) -> subprocess.CompletedProcess:
    """Feed the script one pushed ref, as git feeds a pre-push hook."""
    return subprocess.run([str(root / ".githooks" / "refuse_corpus.sh")], cwd=repo,
                          input=f"refs/heads/main {local} refs/heads/main {remote}\n",
                          capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q", "-b", "main")
    return tmp_path


def test_a_pdf_added_then_deleted_in_the_range_is_refused(root, repo):
    base = commit(repo, "base", add={"README.md": "x"})
    commit(repo, "oops", add={"data/GUIDELINES/x.pdf": "%PDF"})
    tip = commit(repo, "undo", remove=("data/GUIDELINES/x.pdf",))
    out = run_refusal(root, repo, tip, base)
    assert out.returncode == 1
    assert "data/GUIDELINES/x.pdf" in out.stderr


def test_a_new_branch_is_checked_against_all_its_history(root, repo):
    commit(repo, "oops", add={"a.pdf": "%PDF"})
    tip = commit(repo, "undo", remove=("a.pdf",))
    out = run_refusal(root, repo, tip, ZERO)
    assert out.returncode == 1 and "a.pdf" in out.stderr


def test_a_clean_range_passes(root, repo):
    base = commit(repo, "base", add={"README.md": "x"})
    tip = commit(repo, "docs", add={"data/MANIFEST.tsv": "file\n"})
    assert run_refusal(root, repo, tip, base).returncode == 0


def test_a_pdf_pushed_before_does_not_block_later_pushes(root, repo):
    """Only the pushed range is inspected, not history the remote already has:
    otherwise a single past mistake would refuse every push forever."""
    commit(repo, "old", add={"a.pdf": "%PDF"})
    base = commit(repo, "removed", remove=("a.pdf",))
    tip = commit(repo, "later", add={"README.md": "x"})
    assert run_refusal(root, repo, tip, base).returncode == 0


def run_branch(root, repo, refs: str) -> subprocess.CompletedProcess:
    """Feed .githooks/refuse_branch.sh the ref lines git would give the hook."""
    return subprocess.run([str(root / ".githooks" / "refuse_branch.sh")], cwd=repo,
                          input=refs, capture_output=True, text=True)


def test_main_to_main_is_the_one_push_allowed(root, repo):
    tip = commit(repo, "base")
    assert run_branch(root, repo, f"refs/heads/main {tip} refs/heads/main {ZERO}\n").returncode == 0


@pytest.mark.parametrize("line", [
    "refs/heads/old {tip} refs/heads/old {zero}",     # another branch
    "refs/heads/main {tip} refs/heads/other {zero}",  # main, to another name
    "refs/heads/old {tip} refs/heads/main {zero}",    # another branch, onto main
    "refs/tags/v1 {tip} refs/tags/v1 {zero}",         # a tag
    "(delete) {zero} refs/heads/main {tip}",          # deleting the remote main
])
def test_any_other_ref_is_refused(root, repo, line):
    tip = commit(repo, "base")
    out = run_branch(root, repo, line.format(tip=tip, zero=ZERO) + "\n")
    assert out.returncode == 1 and "Only main" in out.stderr


def test_a_push_from_another_checked_out_branch_is_refused(root, repo):
    tip = commit(repo, "base")
    git(repo, "checkout", "-q", "-b", "other")
    out = run_branch(root, repo, f"refs/heads/main {tip} refs/heads/main {ZERO}\n")
    assert out.returncode == 1 and "not main" in out.stderr
