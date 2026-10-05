"""The closed-vocabulary gate, and the two scripts that run it.

`doc_type` and `topic` are the only manifest columns the UI turns into filters, so
a slug outside its vocabulary is not a cosmetic mistake: it becomes a filter option
that matches one document, labelled with the raw slug because the bilingual lookup
has nothing for it. The check used to exist twice, once per script; these tests are
against the one copy, plus one test per script that it is still wired in.

Written by Claude Code (Opus 5).
"""

from __future__ import annotations

import pytest

VOCABULARIES = {"doc_type": {"standard", "summary"}, "topic": {"housing", "water"}}


def test_split_values_strips_and_drops_blanks(vocabulary):
    """Hand-edited cells: "housing; water" and "housing;water" must mean the same thing."""
    assert vocabulary.split_values("a;b ; c", ";") == ["a", "b", "c"]
    assert vocabulary.split_values("", ";") == []
    assert vocabulary.split_values(" ; ", ";") == []


def test_a_known_slug_passes(vocabulary):
    vocabulary.check([{"doc_type": "summary", "topic": "housing;water"}], VOCABULARIES, ";")


def test_a_blank_cell_passes(vocabulary):
    """A blank is "not curated yet", which the UI can ignore; a typo it cannot."""
    vocabulary.check([{"doc_type": "", "topic": ""}], VOCABULARIES, ";")
    vocabulary.check([{}], VOCABULARIES, ";")


def test_a_typo_is_refused_and_named(vocabulary):
    with pytest.raises(vocabulary.Unknown) as caught:
        vocabulary.check([{"doc_type": "recommandaton"}], VOCABULARIES, ";")
    assert "recommandaton" in str(caught.value)
    assert "doc_type" in str(caught.value)


def test_every_bad_slug_is_reported_in_one_run(vocabulary):
    """A 500-row manifest must not need 500 runs to be fixed."""
    rows = [{"doc_type": "typo-one", "topic": "typo-two"},
            {"doc_type": "typo-three", "topic": "housing"}]
    with pytest.raises(vocabulary.Unknown) as caught:
        vocabulary.check(rows, VOCABULARIES, ";")
    message = str(caught.value)
    assert all(bad in message for bad in ("typo-one", "typo-two", "typo-three"))


def test_a_slug_is_only_checked_against_its_own_column(vocabulary):
    """A topic in the doc_type column is exactly the mistake worth catching."""
    with pytest.raises(vocabulary.Unknown):
        vocabulary.check([{"doc_type": "housing"}], VOCABULARIES, ";")


def test_the_separator_is_the_caller_s(vocabulary):
    """lib/manifest_policy.py owns VALUE_SEPARATOR; nothing here may restate it."""
    vocabulary.check([{"topic": "housing|water"}], VOCABULARIES, "|")
    with pytest.raises(vocabulary.Unknown):
        vocabulary.check([{"topic": "housing|water"}], VOCABULARIES, ";")


def test_manifest_py_still_gates_and_translates(manifest):
    """The write-time gate, raising the exception click prints without a traceback."""
    import click

    manifest.check_vocabulary([{"file": "x.pdf", "doc_type": "standard",
                                "topic": sorted(manifest.CLOSED_VOCABULARIES["topic"])[0]}])
    with pytest.raises(click.ClickException) as caught:
        manifest.check_vocabulary([{"file": "x.pdf", "doc_type": "standad"}])
    assert "standad" in str(caught.value)


def test_build_index_py_still_gates_and_translates(build_index):
    """The read-time gate: --manifest accepts any TSV, including a hand-edited one."""
    import click

    policy = {"separator": ";", "vocabularies": VOCABULARIES}
    build_index.check_vocabulary({"x.pdf": {"doc_type": "summary"}}, policy)
    with pytest.raises(click.ClickException) as caught:
        build_index.check_vocabulary({"x.pdf": {"topic": "adult"}}, policy)
    assert "adult" in str(caught.value)
