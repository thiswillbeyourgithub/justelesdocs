"""The chunking sweep's three halves agree with each other.

The sweep is a shell driver, a JavaScript dumper and two Python scripts, and the
places they can silently disagree are worth a test each:

- the passage hash. The dumper writes it, the judge caches verdicts under it and the
  report reads them back. A drift there does not fail: it produces an empty cache and
  a table of zeros, after four GPU hours.
- the strategy names. `grid_search.sh` defines them, `grid_grid_report.py` names one of
  them as the baseline every other row is compared against. A rename leaves the
  win/loss columns quietly empty.

The rest is the report's arithmetic, which decides which chunking ships and is pure
enough to check directly.

Written by Claude Code (Opus 5).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DRIVER = ROOT / "grid_search.sh"
DUMPER = ROOT / "scripts" / "dump_candidates.mjs"


def strategies() -> list[tuple[str, str, str, str]]:
    """The driver's strategy table, read out of the heredoc that defines it."""
    lines = DRIVER.read_text(encoding="utf-8").splitlines()
    start = lines.index("  cat <<'EOF'") + 1
    end = lines.index("EOF", start)
    return [tuple(line.split("|")) for line in lines[start:end]]


def test_the_strategy_table_is_a_set_of_distinct_names():
    names = [row[0] for row in strategies()]
    assert len(names) == len(set(names)), "a duplicate name would overwrite a dump"
    assert all(row[0] and row[1] for row in strategies()), "a row with no chunker args"


def test_a_reusing_strategy_names_one_that_exists():
    """`reuse` shares another row's chunks and bake, so it has to point at one.

    A typo here does not fail loudly: build_one skips the row with a warning and the
    sweep finishes one strategy short.
    """
    names = {row[0] for row in strategies()}
    for row in strategies():
        if row[3]:
            assert row[3] in names, f"{row[0]} reuses '{row[3]}', which is not a strategy"


def test_the_report_compares_against_a_strategy_the_sweep_builds(grid_report):
    """The default baseline exists, and is the chunking that ships today."""
    default = next(p for p in grid_report.main.params if p.name == "baseline").default
    assert default in {row[0] for row in strategies()}
    assert default == "t170-512-sec", (
        "the baseline should stay the shipped chunking, so every row reads as a "
        "change to what is deployed")
    # Its row passes nothing but chunk.py's own defaults, so it is the shipped
    # chunking by construction rather than by a name that could drift from it.
    row = next(row for row in strategies() if row[0] == default)
    assert row[1].split() == ["--target-tokens", "512", "--overlap-tokens", "0"]
    assert row[2] == "" and row[3] == ""


@pytest.mark.skipif(not shutil.which("node"), reason="node is not installed")
def test_the_dumper_and_the_judge_hash_a_passage_identically(grid_report):
    """The same bytes, the same 16 hex digits, in JavaScript and in Python."""
    passage = "Chez l'habitant, la consommation initiale est de 12,5 m3 par jour.\n"
    in_node = subprocess.run(
        ["node", "-e",
         'const {createHash}=require("node:crypto");'
         'process.stdout.write(createHash("sha256").update(process.argv[1])'
         '.digest("hex").slice(0,16));', passage],
        capture_output=True, text=True, check=True).stdout
    assert in_node == grid_report.sha(passage)
    assert in_node == hashlib.sha256(passage.encode("utf-8")).hexdigest()[:16]
    # And the dumper really does hash the passage text, not the id or the file name.
    source = DUMPER.read_text(encoding="utf-8")
    assert 'createHash("sha256").update(c.text || "")' in source


def dump(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    """A candidates.json with one query per row: {gold_rank, texts}."""
    queries = []
    for i, row in enumerate(rows):
        queries.append({
            "query_id": f"q{i:03d}", "query": row.get("query", f"question {i}"),
            "type": row.get("type", "precise"), "query_lang": "fr",
            "gold_rank": row["gold_rank"],
            "candidates": [
                {"rank": j + 1, "doc": 0, "file": "a.pdf", "chunk": j, "pages": [1],
                 "score": 0.5, "text_sha": hashlib.sha256(t.encode()).hexdigest()[:16],
                 "text": t}
                for j, t in enumerate(row["texts"])],
        })
    path = tmp_path / name / "candidates.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "strategy": name, "index_dir": f"data/grid/{name}/index",
        "meta": {"n_chunks": 100, "dims": 1024, "quant": "int8"},
        "topk": len(rows[0]["texts"]), "queries": queries}), encoding="utf-8")
    return path


def verdicts(tmp_path: Path, scored: dict[tuple[str, str], float], budget: int = 1024) -> Path:
    """A judge cache, keyed the way judge_rerank.py writes it."""
    path = tmp_path / "judge-scores.jsonl"
    path.write_text("".join(
        json.dumps({"q": hashlib.sha256(q.encode()).hexdigest()[:16],
                    "d": hashlib.sha256(d.encode()).hexdigest()[:16],
                    "n": budget, "s": s}) + "\n"
        for (q, d), s in scored.items()), encoding="utf-8")
    return path


def test_an_unjudged_candidate_is_counted_rather_than_scored_as_zero(tmp_path, grid_report):
    """A half-finished judge run must not make a strategy look bad.

    This is the failure that would be hardest to notice from the table alone: the
    means are over what WAS judged, and the count of what was not is printed next to
    them so a table read too early announces itself.
    """
    dump(tmp_path, "s1", [{"query": "q", "gold_rank": 1, "texts": ["a", "b"]}])
    cache = verdicts(tmp_path, {("q", "a"): 0.9})
    payload = json.loads((tmp_path / "s1" / "candidates.json").read_text())
    rows = grid_report.per_query(payload, grid_report.read_scores(cache, 1024), 0.5)
    assert rows[0]["missing"] == 1
    assert rows[0]["top1"] == pytest.approx(0.9)
    # The mean over the top 10 is 0.9, not 0.45: "b" has no verdict, it does not
    # have a verdict of zero.
    assert rows[0]["top10"] == pytest.approx(0.9)


def test_the_label_metrics_come_from_the_rank_the_shipped_ranker_gave(tmp_path, grid_report):
    """page@k and MRR are read off gold_rank, which lib/eval.mjs computed."""
    dump(tmp_path, "s1", [
        {"query": "q1", "gold_rank": 1, "texts": ["a"]},
        {"query": "q2", "gold_rank": 4, "texts": ["b"]},
        {"query": "q3", "gold_rank": 0, "texts": ["c"]},
    ])
    payload = json.loads((tmp_path / "s1" / "candidates.json").read_text())
    rows = grid_report.per_query(payload, {}, 0.5)
    summary = grid_report.summarise(rows, None, 0.02)
    assert summary["page_1"] == pytest.approx(1 / 3)
    assert summary["page_5"] == pytest.approx(2 / 3)
    # 0 means "not in the list at all" and contributes nothing, rather than 1/0.
    assert summary["mrr"] == pytest.approx((1 + 0.25) / 3)


def test_a_win_is_per_query_and_ties_inside_the_dead_band_are_neither(tmp_path, grid_report):
    """The column that says whether a change is broad or is one lucky query."""
    dump(tmp_path, "base", [{"query": f"q{i}", "gold_rank": 1, "texts": [f"b{i}"]}
                            for i in range(3)])
    dump(tmp_path, "other", [{"query": f"q{i}", "gold_rank": 1, "texts": [f"o{i}"]}
                             for i in range(3)])
    cache = verdicts(tmp_path, {
        ("q0", "b0"): 0.50, ("q0", "o0"): 0.80,   # a win
        ("q1", "b1"): 0.50, ("q1", "o1"): 0.20,   # a loss
        ("q2", "b2"): 0.50, ("q2", "o2"): 0.51,   # inside the dead band
    })
    scores = grid_report.read_scores(cache, 1024)
    base = grid_report.per_query(json.loads((tmp_path / "base" / "candidates.json").read_text()),
                            scores, 0.5)
    other = grid_report.per_query(json.loads((tmp_path / "other" / "candidates.json").read_text()),
                             scores, 0.5)
    summary = grid_report.summarise(other, base, 0.02)
    assert (summary["win"], summary["loss"]) == (1, 1)


def test_verdicts_at_another_token_budget_are_a_different_measurement(tmp_path, grid_report):
    """--doc-tokens is in the cache key: the length-bias control is not a mixture."""
    cache = verdicts(tmp_path, {("q", "a"): 0.9}, budget=256)
    assert grid_report.read_scores(cache, 1024) == {}
    assert len(grid_report.read_scores(cache, 256)) == 1


def test_the_judge_and_the_report_read_one_cache_one_way(tmp_path, grid_report):
    """Both go through lib/judge_cache.py: a torn last line is skipped and counted,
    and a later verdict for the same pair replaces the earlier one."""
    from conftest import load_lib
    judge_cache = load_lib("judge_cache")
    cache = tmp_path / "judge-scores.jsonl"
    cache.write_text('{"q": "a", "d": "b", "n": 1024, "s": 0.1}\n'
                     '{"q": "a", "d": "b", "n": 1024, "s": 0.7}\n'
                     '{"q": "a", "d"', encoding="utf-8")
    scores, skipped = judge_cache.read_cache(cache)
    assert (scores, skipped) == ({("a", "b", 1024): 0.7}, 1)
    assert grid_report.read_scores(cache, 1024) == {("a", "b"): 0.7}
    assert grid_report.sha is judge_cache.sha
    # judge_rerank.py imports torch, so it is checked by its source: no second copy.
    judge = (Path(__file__).resolve().parents[1] / "scripts" / "judge_rerank.py").read_text(encoding="utf-8")
    assert "def sha(" not in judge and "judge_cache.read_cache(" in judge
