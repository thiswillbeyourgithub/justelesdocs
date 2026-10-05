"""The document subset a sweep runs over.

A sweep is only worth reading if two of its points covered the same documents, so
these tests are about one thing: the subset is complete (every gold document is in
it) and reproducible (the same seed picks the same distractors).

Written by Claude Code (Opus 5).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture()
def corpus(tmp_path: Path) -> tuple[Path, Path]:
    """A manifest of 20 documents and an eval set asking about three of them."""
    manifest = tmp_path / "MANIFEST.tsv"
    write_tsv(manifest, ["file", "title"],
              [{"file": f"doc{i:02d}.pdf", "title": f"Document {i}"} for i in range(20)])
    queries = tmp_path / "EVAL_QUERIES.tsv"
    write_tsv(queries, ["query_id", "type", "query", "gold_file", "gold_page"],
              [{"query_id": f"q{i}", "type": "precise", "query": "?",
                "gold_file": name, "gold_page": "1"}
               for i, name in enumerate(["doc03.pdf", "doc11.pdf", "doc17.pdf"])])
    return queries, manifest


def test_every_gold_document_is_in_the_subset(bench, corpus):
    queries, manifest = corpus
    names = bench.bench_corpus(queries=queries, manifest=manifest, distractors=2, seed=1)
    assert {"doc03.pdf", "doc11.pdf", "doc17.pdf"} <= set(names)
    assert len(names) == 5


def test_an_alternative_gold_document_is_in_the_subset(bench, corpus, tmp_path):
    """Leaving it out would score as a miss an answer the full index gives."""
    _queries, manifest = corpus
    queries = tmp_path / "alt.tsv"
    write_tsv(queries, ["query_id", "type", "query", "gold_file", "gold_page", "gold_alt"],
              [{"query_id": "q1", "type": "precise", "query": "?", "gold_file": "doc03.pdf",
                "gold_page": "1", "gold_alt": "doc11.pdf#7;doc17.pdf#2"}])
    names = bench.bench_corpus(queries=queries, manifest=manifest, distractors=0, seed=1)
    assert names == ["doc03.pdf", "doc11.pdf", "doc17.pdf"]


def test_the_same_seed_draws_the_same_distractors(bench, corpus):
    queries, manifest = corpus
    first = bench.bench_corpus(queries=queries, manifest=manifest, distractors=6, seed=7)
    again = bench.bench_corpus(queries=queries, manifest=manifest, distractors=6, seed=7)
    other = bench.bench_corpus(queries=queries, manifest=manifest, distractors=6, seed=8)
    assert first == again
    assert first != other, "two seeds drawing the same sample would make --seed a lie"


def test_no_distractors_means_the_gold_documents_alone(bench, corpus):
    queries, manifest = corpus
    names = bench.bench_corpus(queries=queries, manifest=manifest, distractors=0, seed=1)
    assert names == ["doc03.pdf", "doc11.pdf", "doc17.pdf"]


def test_a_gold_document_missing_from_the_manifest_stops_the_run(bench, corpus, tmp_path):
    queries, manifest = corpus
    thin = tmp_path / "thin.tsv"
    write_tsv(thin, ["file", "title"], [{"file": "doc03.pdf", "title": "Document 3"}])
    # Silently dropping the query would leave the sweep measuring 2 questions out of
    # 3 and reporting the average as if nothing had happened.
    with pytest.raises(Exception) as caught:
        bench.bench_corpus(queries=queries, manifest=thin, distractors=1, seed=1)
    assert "doc11.pdf" in str(caught.value)


def test_a_point_is_named_after_what_changes_its_chunks(bench):
    name = bench.point_slug({"target_tokens": 450, "overlap_tokens": 0,
                             "boundaries": False, "tables": True})
    assert name == "t450-o0-greedy-tab"
    other = bench.point_slug({"target_tokens": 450, "overlap_tokens": 0,
                              "boundaries": False, "tables": False})
    assert name != other, "two points sharing a directory would overwrite each other"


def test_a_sweep_axis_is_read_as_a_list(bench):
    assert bench.numbers("200,300,450") == [200, 300, 450]
    assert bench.numbers("300") == [300]


def test_the_section_point_is_named_without_a_token_target(bench):
    # Sections have no size and cut at headings, so a slug carrying t/o/bound
    # would name parameters the point does not use.
    name = bench.point_slug({"target_tokens": "section", "overlap_tokens": 0,
                             "boundaries": True, "tables": True})
    assert name == "section-tab"
    assert name != bench.point_slug({"target_tokens": "section", "overlap_tokens": 0,
                                     "boundaries": True, "tables": False})
