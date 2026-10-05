"""scripts/eval_queries.py joins a corpus's hand-written queries to their gold labels.

Run on a tiny neutral corpus written in tmp_path. The checks that read a real
corpus's queries and manifest belong to that corpus's own tests.

Written by Claude Code (Opus 5); rewritten for the queries file by Claude Code (Opus 5.5).
"""

from __future__ import annotations

import csv
from pathlib import Path

import click
import pytest

HEADER = "set\tpassage_id\ttype\tquery_lang\tquery\tgold_file\tgold_page\talso_acceptable\n"


def _tsv(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


@pytest.fixture
def corpus(tmp_path: Path) -> dict[str, Path]:
    """Two documents, one French and one English, and one sampled passage each."""
    return {
        "passages": _tsv(tmp_path / "EVAL_PASSAGES.tsv",
                         "passage_id\tfile\tpage\tlanguage\tissuer\n"
                         "p001\tair_quality_report.pdf\t4\ten\tWHO\n"
                         "p002\tlogement_enquete.pdf\t2\tfr\tINSEE\n"),
        "manifest": _tsv(tmp_path / "MANIFEST.tsv",
                         "file\tlanguage\tissuer\tpages\n"
                         "air_quality_report.pdf\ten\tWHO\t40\n"
                         "logement_enquete.pdf\tfr\tINSEE\t12\n"
                         "air_quality_report_2019.pdf\ten\tWHO\t38\n"),
        "unusable": _tsv(tmp_path / "unusable.tsv", "passage_id\treason\n"),
        "dir": tmp_path,
    }


def _run(eval_queries, corpus, queries: str):
    from click.testing import CliRunner

    path = _tsv(corpus["dir"] / "queries.tsv", HEADER + queries)
    out, cross = corpus["dir"] / "out.tsv", corpus["dir"] / "cross.tsv"
    result = CliRunner().invoke(eval_queries.main, [
        "--queries", str(path), "--unusable", str(corpus["unusable"]),
        "--passages", str(corpus["passages"]), "--manifest", str(corpus["manifest"]),
        "--out", str(out), "--out-crosslingual", str(cross),
    ])
    return result, out, cross


def test_the_three_sets_land_where_they_belong(eval_queries, corpus):
    result, out, cross = _run(eval_queries, corpus, (
        "# a note, ignored\n"
        "sampled\tp001\tprecise\ten\tWhat is the annual PM2.5 guideline value?\t\t\t"
        "air_quality_report_2019.pdf#5\n"
        "\n"
        "sampled\tp002\tcrosslingual\ten\tHow many households rent their home?\t\t\t\n"
        "labelled\t\ttable\tfr\tQuel est le taux de propriétaires en 2020 ?\tlogement_enquete.pdf\t7\t\n"
        "mirror\t\tcrosslingual\tfr\tQuelle est la valeur guide annuelle des PM2,5 ?\tair_quality_report.pdf\t4\t\n"
    ))
    assert result.exit_code == 0, result.output
    rows = _read(out)
    assert [r["query_id"] for r in rows] == ["q001", "q002", "q003"]
    assert rows[0]["gold_file"] == "air_quality_report.pdf" and rows[0]["gold_page"] == "4"
    assert rows[0]["gold_alt"] == "air_quality_report_2019.pdf#5"
    assert (rows[2]["gold_file"], rows[2]["doc_lang"], rows[2]["issuer"]) == ("logement_enquete.pdf", "fr", "INSEE")
    crossing = _read(cross)
    # The crossing main row, then the mirror, which inherits its twin's alternatives.
    assert [r["query_id"] for r in crossing] == ["q002", "x001"]
    assert crossing[1]["gold_alt"] == "air_quality_report_2019.pdf#5"
    assert crossing[1]["doc_lang"] == "en"


def test_a_mirror_must_mirror_an_existing_gold(eval_queries, corpus):
    result, _out, _cross = _run(eval_queries, corpus, (
        "sampled\tp001\tprecise\ten\tWhat is the annual PM2.5 guideline value?\t\t\t\n"
        "mirror\t\tcrosslingual\tfr\tQuelle est la valeur guide ?\tair_quality_report.pdf\t9\t\n"
    ))
    assert result.exit_code != 0
    assert "not the gold of any query" in result.output


def test_a_passage_missing_from_the_sample_fails_loudly(eval_queries, corpus):
    result, _out, _cross = _run(eval_queries, corpus,
                                "sampled\tp099\tprecise\ten\tWhere is this?\t\t\t\n")
    assert result.exit_code != 0
    assert "p099" in result.output


def test_an_alternative_outside_the_document_is_refused(eval_queries, corpus):
    result, _out, _cross = _run(eval_queries, corpus, (
        "sampled\tp001\tprecise\ten\tWhat is the annual PM2.5 guideline value?\t\t\t"
        "air_quality_report_2019.pdf#99\n"
    ))
    assert result.exit_code != 0
    assert "has 38 pages" in result.output


def test_a_labelled_query_needs_its_gold(eval_queries, tmp_path):
    path = _tsv(tmp_path / "q.tsv", HEADER + "labelled\t\ttable\ten\tWhich year?\t\t\t\n")
    with pytest.raises(click.ClickException, match="needs a gold_file"):
        eval_queries.read_queries(path)


def test_an_english_query_with_an_accent_is_refused(eval_queries, tmp_path):
    path = _tsv(tmp_path / "q.tsv", HEADER + "sampled\tp001\tvague\ten\tHousing dépenses by region\t\t\t\n")
    with pytest.raises(click.ClickException, match="accents"):
        eval_queries.check_orthography(eval_queries.read_queries(path))


def test_a_ragged_row_is_refused(eval_queries, tmp_path):
    path = _tsv(tmp_path / "q.tsv", HEADER + "sampled\tp001\tvague\n")
    with pytest.raises(click.ClickException, match="cells"):
        eval_queries.read_queries(path)
