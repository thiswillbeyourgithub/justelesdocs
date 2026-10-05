"""The arithmetic the two bake experiments share.

Neither experiment can be run in a test (both want a 50,000-vector bake and a
running encoder), so what is checked here is the part that would silently produce a
plausible wrong answer: the ranking helper's definition of a hit, the per-kind
grouping, and the projection the language-axis experiment rests on.

Written by Claude Code (Opus 5).
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture()
def bake():
    """Three chunks over two documents, with the pages each covers."""
    owner = np.array(["a.pdf", "a.pdf", "b.pdf"])
    pages = [{1, 2}, {3}, {7}]
    return owner, pages


def test_a_hit_is_the_gold_page_in_the_gold_document(evalbake, bake):
    owner, pages = bake
    queries = [{"gold_file": "a.pdf", "gold_page": "3", "type": "precise"}]
    # Chunk 1 scores highest but sits on pages 1-2; the answer is chunk 2.
    scores = np.array([[0.9, 0.8, 0.7]])
    assert evalbake.gold_ranks(scores=scores, queries=queries, owner=owner,
                               pages=pages, depth=3) == [2]


def test_the_right_page_in_the_wrong_document_is_not_a_hit(evalbake, bake):
    owner, pages = bake
    queries = [{"gold_file": "b.pdf", "gold_page": "3", "type": "precise"}]
    scores = np.array([[0.1, 0.9, 0.5]])
    assert evalbake.gold_ranks(scores=scores, queries=queries, owner=owner,
                               pages=pages, depth=3) == [0]


def test_a_gold_page_below_the_shortlist_counts_as_a_miss(evalbake, bake):
    """The shipped client never looks past its candidate list, so neither does this."""
    owner, pages = bake
    queries = [{"gold_file": "b.pdf", "gold_page": "7", "type": "vague"}]
    scores = np.array([[0.9, 0.8, 0.1]])
    assert evalbake.gold_ranks(scores=scores, queries=queries, owner=owner,
                               pages=pages, depth=2) == [0]


def test_a_second_acceptable_document_also_counts_as_a_hit(evalbake, bake):
    """Four editions of one book: answering from the wrong one is still answering."""
    owner, pages = bake
    queries = [{"gold_file": "a.pdf", "gold_page": "1", "type": "precise",
                "gold_alt": "b.pdf#7"}]
    # b.pdf's chunk wins; the labelled page in a.pdf is ranked below it.
    scores = np.array([[0.2, 0.1, 0.9]])
    assert evalbake.gold_ranks(scores=scores, queries=queries, owner=owner,
                               pages=pages, depth=3) == [1]


def test_an_alternative_still_has_to_be_the_right_page(evalbake, bake):
    owner, pages = bake
    queries = [{"gold_file": "a.pdf", "gold_page": "1", "type": "precise",
                "gold_alt": "b.pdf#99"}]
    # b.pdf ranks first but on page 7, so the hit is still the labelled page, at 2.
    scores = np.array([[0.2, 0.1, 0.9]])
    assert evalbake.gold_ranks(scores=scores, queries=queries, owner=owner,
                               pages=pages, depth=3) == [2]


def test_a_query_with_no_alternatives_is_unchanged(evalbake):
    """The column is empty for almost every query, and absent in older files."""
    assert evalbake.acceptable({"gold_file": "a.pdf", "gold_page": "4"}) == [("a.pdf", 4)]
    assert evalbake.acceptable({"gold_file": "a.pdf", "gold_page": "4",
                                "gold_alt": ""}) == [("a.pdf", 4)]


def test_a_file_name_may_hold_anything_but_the_separators(evalbake):
    """Names in this corpus carry spaces, commas, apostrophes and degree signs."""
    pairs = evalbake.acceptable({
        "gold_file": "a.pdf", "gold_page": "1",
        "gold_alt": "Annual Housing Survey Tables, N\u00b014.pdf#30"})
    assert pairs[1] == ("Annual Housing Survey Tables, N\u00b014.pdf", 30)


def test_metrics_are_grouped_by_kind_and_over_everything(evalbake):
    queries = [{"type": "precise"}, {"type": "precise"}, {"type": "crosslingual"}]
    grouped = evalbake.by_kind([1, 0, 4], queries)
    assert grouped["all"]["n"] == 3
    assert grouped["precise"]["page_1"] == 0.5
    assert grouped["crosslingual"]["page_1"] == 0.0
    assert grouped["crosslingual"]["page_5"] == 1.0
    assert grouped["all"]["mrr"] == pytest.approx((1 + 0 + 0.25) / 3)


def test_removing_the_axis_leaves_rows_orthogonal_to_it(language_axis):
    # No row lies along the axis here; that case has its own test below.
    rows = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 1.0], [2.0, -1.0, 0.5]], dtype=np.float32)
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    cleaned = language_axis.without(rows, axis)
    assert np.allclose(cleaned @ axis, 0, atol=1e-6)
    assert np.allclose(np.linalg.norm(cleaned, axis=1), 1, atol=1e-6)


def test_a_row_lying_along_the_axis_cannot_be_cleaned(language_axis):
    """It would divide by zero; the experiment must not silently emit NaNs."""
    rows = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        cleaned = language_axis.without(rows, axis)
    assert not np.isfinite(cleaned).all(), (
        "a row parallel to the axis has nothing left after the projection; if this "
        "ever starts returning finite numbers, something normalised it away silently"
    )


def test_binarise_stores_a_zero_as_the_shipped_index_does(evalbake):
    """build_index.py packs `cut > 0`, so a component of exactly 0 is a 0 bit (-1).
    evaluate.py used `>= 0` and measured a slightly different index."""
    signs = evalbake.binarise(np.array([[0.0, 0.5, -0.5, 0.0]], dtype=np.float32))
    assert (np.sign(signs) == [[-1, 1, -1, -1]]).all()
    packed = np.unpackbits(np.packbits(np.array([[0.0, 0.5, -0.5, 0.0]]) > 0, axis=1), axis=1)[:, :4]
    assert ((signs > 0) == packed.astype(bool)).all()


def test_a_bake_that_disagrees_with_its_chunks_is_refused(evalbake, tmp_path):
    import json
    chunks, vectors = tmp_path / "chunks", tmp_path / "vectors"
    chunks.mkdir(), vectors.mkdir()
    (chunks / "a.json").write_text(json.dumps(
        {"file": "a.pdf", "chunks": [{"pages": [1]}, {"pages": [2]}]}), encoding="utf-8")
    np.savez(vectors / "a.npz", vectors=np.ones((3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="2 chunks but 3 vectors"):
        evalbake.load_bake(chunks=chunks, vectors=vectors)


def test_evaluate_scores_an_alternative_edition_as_a_hit(evalbake, bake):
    """evaluate.py used to check `gold_file` only, so the same retrieval was a hit
    in gold_ranks and a miss in EVAL_RESULTS.tsv."""
    from conftest import load
    evaluate = load("evaluate")
    owner, pages = bake
    corpus = evaluate.Corpus(vectors=np.eye(3, dtype=np.float32), files=owner.tolist(),
                             pages=[frozenset(p) for p in pages])
    queries = [{"gold_file": "a.pdf", "gold_page": "1", "gold_alt": "b.pdf#7", "type": "precise"}]
    metrics = evaluate.score(corpus, np.array([[0.1, 0.0, 0.9]], dtype=np.float32), queries, top_k=3)
    assert metrics["all"]["page@1"] == 1.0
    assert metrics["all"]["doc@1"] == 1.0
    assert metrics["precise"]["mrr"] == 1.0
