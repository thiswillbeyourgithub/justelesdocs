"""The index writer's housekeeping: what it removes as well as what it writes.

`build_index.py` is mostly numpy and JSON, which the gates and `check_search.mjs`
cover end to end. What they cannot see is the directory left BEHIND a build: a
file nothing names any more is invisible to every check that starts from
`meta.json`, and is still served and still deployed.

Written by Claude Code (Opus 5).
"""

from __future__ import annotations

import pytest


def populate(doc_dir, ids):
    """Write one `<id>.json` and one `<id>.json.gz` per id, as a build would."""
    doc_dir.mkdir(parents=True, exist_ok=True)
    for i in ids:
        (doc_dir / f"{i}.json").write_text("{}", encoding="utf-8")
        (doc_dir / f"{i}.json.gz").write_bytes(b"")


def test_prune_removes_documents_the_index_no_longer_has(build_index, tmp_path):
    """A build over fewer documents than the last one takes its tail with it."""
    doc_dir = tmp_path / "doc"
    populate(doc_dir, range(537))

    removed = build_index.prune_doc_dir(doc_dir, 534)

    # Three documents, each with its .gz sibling.
    assert removed == 6
    assert not (doc_dir / "534.json").exists()
    assert not (doc_dir / "534.json.gz").exists()
    assert not (doc_dir / "536.json").exists()
    # And nothing the index still names was touched.
    assert (doc_dir / "533.json").exists()
    assert (doc_dir / "533.json.gz").exists()
    assert (doc_dir / "0.json").exists()


def test_prune_is_a_no_op_on_a_matching_build(build_index, tmp_path):
    doc_dir = tmp_path / "doc"
    populate(doc_dir, range(12))
    assert build_index.prune_doc_dir(doc_dir, 12) == 0
    assert len(list(doc_dir.iterdir())) == 24


def test_prune_after_limit_leaves_only_what_was_built(build_index, tmp_path):
    """`--limit 3` against a full corpus is the case that leaves hundreds behind."""
    doc_dir = tmp_path / "doc"
    populate(doc_dir, range(200))
    assert build_index.prune_doc_dir(doc_dir, 3) == 394
    assert sorted(p.name for p in doc_dir.iterdir()) == [
        "0.json", "0.json.gz", "1.json", "1.json.gz", "2.json", "2.json.gz",
    ]


@pytest.mark.parametrize("stray", ["notes.txt", "12.json.bak", "doc.json"])
def test_prune_removes_anything_not_written_by_this_build(build_index, tmp_path, stray):
    """The directory belongs to the writer: an id-shaped name is not the test.

    Pruning by "what this build wrote" rather than by "what looks like an id" is
    what keeps a renamed or half-written file from surviving forever.
    """
    doc_dir = tmp_path / "doc"
    populate(doc_dir, range(3))
    (doc_dir / stray).write_text("x", encoding="utf-8")
    build_index.prune_doc_dir(doc_dir, 3)
    assert not (doc_dir / stray).exists()
    assert (doc_dir / "2.json").exists()


# --- sections ---------------------------------------------------------------
# Format 4 joins a chunk to its section by the boxes the two chunkings share.
# What that join promises: a section's boxes name its row, a chunk goes to the
# section holding most of its boxes (the overlap carry puts its first box in the
# previous one), and a chunk no section knows is an orphan rather than a guess.


def section_file(sections):
    """A `--section-chunks` payload from `{page: [boxes]}` dicts, one per section."""
    return {"file": "x.pdf", "params": {"section_chunks": True},
            "chunks": [{"i": i, "boxes": boxes} for i, boxes in enumerate(sections)]}


def test_section_rows_name_every_box(build_index):
    rows = build_index.section_rows(section_file([
        {"1": [[10, 10, 100, 20], [10, 22, 100, 32]]},
        {"1": [[10, 40, 100, 50]], "2": [[10, 10, 100, 20]]},
    ]), "x.pdf")
    assert rows[(1, (10, 10, 100, 20))] == 0
    assert rows[(1, (10, 40, 100, 50))] == 1
    assert rows[(2, (10, 10, 100, 20))] == 1


def test_a_chunk_goes_to_the_section_holding_most_of_its_boxes(build_index):
    rows = build_index.section_rows(section_file([
        {"1": [[0, 0, 1, 1]]},
        {"1": [[0, 2, 1, 3], [0, 4, 1, 5], [0, 6, 1, 7]]},
    ]), "x.pdf")
    # First box carried over from the previous section, three from its own.
    chunk = {"boxes": {"1": [[0, 0, 1, 1], [0, 2, 1, 3], [0, 4, 1, 5], [0, 6, 1, 7]]}}
    assert build_index.section_of(chunk, rows) == 1
    # A tie goes to the earlier section, so the answer never depends on dict order.
    tied = {"boxes": {"1": [[0, 0, 1, 1], [0, 2, 1, 3]]}}
    assert build_index.section_of(tied, rows) == 0


def test_a_chunk_no_section_knows_is_an_orphan(build_index):
    rows = build_index.section_rows(section_file([{"1": [[0, 0, 1, 1]]}]), "x.pdf")
    assert build_index.section_of({"boxes": {"3": [[5, 5, 6, 6]]}}, rows) is None
    assert build_index.section_of({"boxes": {}}, rows) is None


def test_a_bake_of_the_wrong_mode_is_refused(build_index):
    import click
    with pytest.raises(click.ClickException):
        build_index.section_rows({"file": "x.pdf", "params": {"page_chunks": True},
                                  "chunks": []}, "x.pdf")


@pytest.mark.parametrize("weights", [(0.15, 0.0, 0.0), (0.05, 0.10, 0.025), (0.0, 0.0, 0.0)])
def test_context_weights_that_leave_the_passage_a_share_pass(build_index, weights):
    build_index.check_weights(*weights)


@pytest.mark.parametrize("weights", [(0.5, 0.5, 0.0), (-0.1, 0.0, 0.0), (0.2, 0.2, 0.7)])
def test_context_weights_that_leave_nothing_or_go_negative_are_refused(build_index, weights):
    import click
    with pytest.raises(click.ClickException):
        build_index.check_weights(*weights)


# --- the reference axis -------------------------------------------------------
# The exemplar test and the direction built from it. What cannot be covered here
# is whether the axis separates anything on the real corpus, which is a property
# of the encoder rather than of this code; DESIGN.md records that measurement.


REAL_BIBLIOGRAPHY = (
    "Martin JP, Novak RW, Lindqvist RS, et al. National Survey of Household Energy "
    "Use: 2016 methodological report. J Off Stat. 2016;61(9):540-560. "
    "doi:10.1177/0706743716659417. Bertrand A, Okafor TA, Salmon G, et al. "
    "Comparative efficiency of 21 heat pumps. Energy Policy. 2018;391(10128):1357-1366. "
    "Hallberg F, Eriksen JF, Nilsson S, et al. Consistent performance. "
    "Build Environ. 2016;21(4):523-530. PMID: 26033244."
)
PLAIN_PROSE = (
    "La rénovation énergétique d'un logement collectif d'ancienneté modérée "
    "repose en première intention sur une isolation structurée, un remplacement "
    "des menuiseries, ou l'association des deux. Le choix se fait avec l'occupant, "
    "en tenant compte de ses préférences, des travaux antérieurs et des contraintes "
    "techniques. La consommation est réévaluée à quatre saisons."
)


def test_a_citation_list_is_taken_as_an_exemplar(build_index):
    assert build_index.looks_like_references(REAL_BIBLIOGRAPHY)
    # A heading is enough on its own, because the chunk that opens a bibliography
    # carries the word and then the citations.
    assert build_index.looks_like_references("Références\n1. Martin JP et al.")
    assert build_index.looks_like_references("Bibliographie")
    assert build_index.looks_like_references("REFERENCES")


def test_plain_prose_is_not_taken_as_an_exemplar(build_index):
    assert not build_index.looks_like_references(PLAIN_PROSE)
    assert not build_index.looks_like_references("")
    # One citation in a paragraph of prose is not a citation list. This is the
    # false positive that matters: it is the shape of most document text.
    assert not build_index.looks_like_references(
        PLAIN_PROSE + " (Bertrand et al., 2018)."
    )


def test_the_axis_is_the_contrast_and_not_the_average(build_index):
    import numpy as np
    # Fifty exemplars and fifty others, all sharing a strong first component: the
    # corpus centroid. The exemplars alone differ on the second. A plain average
    # would point mostly at the shared component, which is why the axis subtracts
    # the mean of everything.
    shared = np.array([1.0, 0.0, 0.0])
    apart = np.array([0.0, 1.0, 0.0])
    rows = [shared * 0.99 + apart * 0.14 for _ in range(50)]
    rows += [shared * 0.99 - apart * 0.14 for _ in range(50)]
    floats = np.array(rows, dtype=np.float32)
    floats /= np.linalg.norm(floats, axis=1, keepdims=True)
    flags = np.array([True] * 50 + [False] * 50)
    axis = build_index.reference_axis(floats, flags)
    assert abs(float(np.linalg.norm(axis)) - 1.0) < 1e-5
    # It points at what tells the two apart, not at what they share.
    assert float(axis @ apart) > 0.99
    assert abs(float(axis @ shared)) < 0.05
    # And it scores the exemplars above the rest, which is the whole job.
    scores = floats @ axis
    assert scores[:50].min() > scores[50:].max()


def test_too_few_exemplars_are_refused_rather_than_averaged(build_index):
    import click
    import numpy as np
    floats = np.eye(3, dtype=np.float32)[[0, 1, 2] * 20]
    flags = np.array([True] * 10 + [False] * 50)
    with pytest.raises(click.ClickException, match="fewer than"):
        build_index.reference_axis(floats, flags)


def test_exemplars_that_are_the_corpus_leave_no_direction(build_index):
    import click
    import numpy as np
    floats = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (60, 1))
    flags = np.array([True] * 60)
    with pytest.raises(click.ClickException, match="no direction"):
        build_index.reference_axis(floats, flags)


# --- the manifest policy shipped in meta.json ------------------------------------


def test_the_guideline_tiers_ship_in_manifest_order():
    """Every other list in the policy is sorted; this one must not be.

    `src/search.js` keeps "every tier at or before the chosen one", by position in
    this list. Sorting it would turn (core, related, background, no) into (background,
    core, no, related) and hand a reader asking for core documents the background ones.
    """
    from conftest import load_lib

    policy = load_lib("manifest_policy").policy()
    assert policy["guideline_tiers"] == ["core", "related", "background", "no"]
    assert policy["guideline_tiers"] != sorted(policy["guideline_tiers"]), \
        "the test would pass by accident if the order happened to be alphabetical"


def test_the_guideline_column_is_gated_like_the_other_vocabularies(build_index):
    """`--manifest` takes any TSV, and a bad tier is invisible once it ships."""
    import pytest

    from conftest import load_lib

    policy = load_lib("manifest_policy").policy()
    rows = {"a.pdf": {"file": "a.pdf", "level": "kore"}}
    with pytest.raises(Exception) as caught:
        build_index.check_vocabulary(rows, policy)
    assert "kore" in str(caught.value)


def test_figures_are_counted_from_the_tail_of_a_document(build_index):
    """The figures filter in src/search.js cuts the range at `figure_count`."""
    chunks = [{"text": "a"}, {"text": "b"}, {"text": "c", "figure": {"n": 1}},
              {"text": "d", "figure": {"n": 2}}]
    assert build_index.trailing_figures(chunks, "x.pdf") == 2
    assert build_index.trailing_figures(chunks[:2], "x.pdf") == 0


def test_a_figure_before_the_text_is_refused(build_index):
    """Counting from the tail would then exclude text and keep a figure."""
    import click
    chunks = [{"text": "a", "figure": {"n": 1}}, {"text": "b"}]
    with pytest.raises(click.ClickException, match="figure chunk before a text chunk"):
        build_index.trailing_figures(chunks, "x.pdf")


# --- vectors are paired with the chunks they were computed from ----------------
# Pairing was checked by row count alone, so re-chunking with a fix that keeps
# the count (a dehyphenation, a table read differently) shipped new text beside
# vectors of the old text. embed.py now records the chunk file's src_hash.


def bake_file(tmp_path, **fields):
    import numpy as np
    path = tmp_path / "a.npz"
    np.savez_compressed(path, vectors=np.zeros((2, 4)), **fields)
    return path, np.load(path)


def test_a_bake_of_other_chunks_is_refused(build_index, tmp_path):
    import click
    import numpy as np
    path, bake = bake_file(tmp_path, chunks_hash=np.array("old"))
    with pytest.raises(click.ClickException, match="other chunks"):
        build_index.check_bake_source(bake, {"file": "a.pdf", "src_hash": "new"}, path)


def test_a_bake_of_these_chunks_is_accepted(build_index, tmp_path):
    import numpy as np
    path, bake = bake_file(tmp_path, chunks_hash=np.array("same"))
    build_index.check_bake_source(bake, {"file": "a.pdf", "src_hash": "same"}, path)


def test_a_bake_from_before_the_field_is_refused(build_index, tmp_path):
    """It used to be counted and warned about, until both bakes were rebaked."""
    import click
    path, bake = bake_file(tmp_path)
    with pytest.raises(click.ClickException, match="records no chunks_hash"):
        build_index.check_bake_source(bake, {"file": "a.pdf", "src_hash": "x"}, path)


# --- what meta.json publishes ------------------------------------------------------
# meta.json is downloaded by every visitor, and the manifest is a working file, so a
# column reaches it only by being named public. These pin that it is an allow-list.

def test_notes_never_reach_meta_json(build_index):
    row = {"title": "T", "year": "2020", "notes": "licence unclear, ask the publisher"}
    assert build_index.public_meta(row, "x.pdf") == {"title": "T", "year": "2020"}


def test_a_column_nobody_classified_is_refused(build_index):
    import click
    with pytest.raises(click.ClickException, match="reviewer_remark"):
        build_index.public_meta({"title": "T", "reviewer_remark": "x"}, "x.pdf")


def test_every_manifest_column_is_classified(build_index):
    """The manifest's own column list and the two sets agree, so a real build passes."""
    from conftest import load
    columns = set(load("manifest").COLUMNS) - {"file"}
    classified = build_index.PUBLIC_COLUMNS | build_index.PRIVATE_COLUMNS
    assert columns == classified
    assert not build_index.PUBLIC_COLUMNS & build_index.PRIVATE_COLUMNS
