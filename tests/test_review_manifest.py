"""The second-pass review tool: the validation, and what it writes where.

What is worth testing here is not the classification (a model does that) but the
machinery around it, because every failure of that machinery is silent. A slug
outside the vocabulary would reach `build_index.py`'s gate eventually, but a review
recorded against the wrong file, a blank cell read as "leave it alone", or a
tier left stale after its `doc_type` changed would not: they all produce
a manifest that looks curated and filters wrongly.

Written by Claude Code (Opus 5).
"""

from __future__ import annotations

import csv

from click.testing import CliRunner

VOCAB = {"doc_type": ["standard", "report", "tool"],
         "topic": ["health", "water", "statistics"]}


def read_tsv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def corpus(review_manifest, tmp_path, *, doc_type="report", topic="health"):
    """A manifest of two rows and a review log queueing the first of them."""
    manifest = tmp_path / "MANIFEST.tsv"
    manifest.write_text(
        "file\ttitle\tdoc_type\tlevel\ttopic\tpages\n"
        f"a.pdf\tA report, or is it\t{doc_type}\tno\t{topic}\t30\n"
        "b.pdf\tHand curated, not queued\tstandard\tcore\twater\t12\n",
        encoding="utf-8")
    log = tmp_path / "REVIEW.tsv"
    review_manifest.write_log(log, [{"file": "a.pdf", "before_doc_type": doc_type,
                                     "before_topic": topic}])
    return manifest, log


def apply(review_manifest, manifest, log, rows, *, extra=()):
    """Run `apply` over a decision sheet built from `rows`."""
    sheet = manifest.parent / "decisions.tsv"
    with sheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_manifest.DECISION_COLUMNS),
                                delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return CliRunner().invoke(review_manifest.cli, [
        "apply", str(sheet), "--by", "a test", "--on", "2026-09-27",
        "--manifest", str(manifest), "--log", str(log),
        *extra])


def test_a_blank_cell_is_a_skipped_row_not_a_confirmation(review_manifest):
    problems = review_manifest.check_decision(
        {"doc_type": "", "topic": "health", "confidence": "high", "note": ""},
        VOCAB, ";", {"before_doc_type": "report", "before_topic": "health"})
    assert any("doc_type is blank" in problem for problem in problems)


def test_a_slug_outside_the_vocabulary_is_refused(review_manifest):
    problems = review_manifest.check_decision(
        {"doc_type": "standard;manual", "topic": "health",
         "confidence": "high", "note": ""},
        VOCAB, ";", {"before_doc_type": "standard;manual", "before_topic": "health"})
    assert any("['manual']" in problem for problem in problems)


def test_a_change_without_a_note_is_refused(review_manifest):
    """The note is the only account of WHY a value moved, so a change needs one."""
    before = {"before_doc_type": "report", "before_topic": "health"}
    decision = {"doc_type": "standard", "topic": "health",
                "confidence": "high", "note": ""}
    assert any("note is blank" in problem for problem in
               review_manifest.check_decision(decision, VOCAB, ";", before))
    decision["note"] = "the body is 40 numbered recommendations with grades"
    assert not review_manifest.check_decision(decision, VOCAB, ";", before)


def test_confidence_is_a_closed_vocabulary_too(review_manifest):
    problems = review_manifest.check_decision(
        {"doc_type": "report", "topic": "health", "confidence": "sure", "note": ""},
        VOCAB, ";", {"before_doc_type": "report", "before_topic": "health"})
    assert any("confidence" in problem for problem in problems)


def test_a_new_doc_type_blanks_the_tier_for_manifest_py_to_refill(review_manifest, tmp_path):
    manifest, log = corpus(review_manifest, tmp_path)
    result = apply(review_manifest, manifest, log, [
        {"file": "a.pdf", "doc_type": "standard", "topic": "health;statistics",
         "confidence": "high", "note": "40 numbered recommendations, each graded"}])
    assert result.exit_code == 0, result.output
    rows = {row["file"]: row for row in read_tsv(manifest)}
    assert rows["a.pdf"]["doc_type"] == "standard"
    assert rows["a.pdf"]["topic"] == "health;statistics"
    # Blank, not "core": deriving the tier here would be a second copy of
    # the tier table in corpus.toml ([tiers.by_value]).
    assert rows["a.pdf"]["level"] == ""
    # And the row nobody reviewed is untouched, tier included.
    assert rows["b.pdf"]["level"] == "core"


def test_confirming_a_value_leaves_the_tier_alone(review_manifest, tmp_path):
    """Re-reading a document and agreeing must not blank a tier a curator set."""
    manifest, log = corpus(review_manifest, tmp_path)
    result = apply(review_manifest, manifest, log, [
        {"file": "a.pdf", "doc_type": "report", "topic": "health",
         "confidence": "high", "note": ""}])
    assert result.exit_code == 0, result.output
    rows = {row["file"]: row for row in read_tsv(manifest)}
    assert rows["a.pdf"]["level"] == "no"


def test_the_log_records_the_before_and_the_after(review_manifest, tmp_path):
    manifest, log = corpus(review_manifest, tmp_path)
    apply(review_manifest, manifest, log, [
        {"file": "a.pdf", "doc_type": "tool", "topic": "water",
         "confidence": "medium", "note": "it is a rating scale with scoring rules"}])
    row = read_tsv(log)[0]
    assert (row["before_doc_type"], row["doc_type"]) == ("report", "tool")
    assert (row["before_topic"], row["topic"]) == ("health", "water")
    assert row["reviewed_by"] == "a test" and row["reviewed_on"] == "2026-09-27"


def test_a_document_outside_the_queue_is_refused_and_nothing_is_written(review_manifest, tmp_path):
    """b.pdf is hand-curated. A sheet naming it is a mistake, not an invitation."""
    manifest, log = corpus(review_manifest, tmp_path)
    before = manifest.read_text(encoding="utf-8")
    result = apply(review_manifest, manifest, log, [
        {"file": "a.pdf", "doc_type": "tool", "topic": "water",
         "confidence": "high", "note": "a scale"},
        {"file": "b.pdf", "doc_type": "tool", "topic": "water",
         "confidence": "high", "note": "a scale"}])
    assert result.exit_code != 0
    # The whole sheet is refused, not the offending row: a partly applied sheet
    # would leave the reviewer guessing which half landed.
    assert manifest.read_text(encoding="utf-8") == before
    assert not read_tsv(log)[0]["reviewed_on"]


def test_a_second_review_needs_regrade(review_manifest, tmp_path):
    manifest, log = corpus(review_manifest, tmp_path)
    decision = [{"file": "a.pdf", "doc_type": "tool", "topic": "water",
                 "confidence": "high", "note": "a rating scale"}]
    assert apply(review_manifest, manifest, log, decision).exit_code == 0
    assert apply(review_manifest, manifest, log, decision).exit_code != 0
    again = [{"file": "a.pdf", "doc_type": "standard", "topic": "water",
              "confidence": "high", "note": "second look: it is the guidance itself"}]
    assert apply(review_manifest, manifest, log, again, extra=["--regrade"]).exit_code == 0
    assert read_tsv(manifest)[0]["doc_type"] == "standard"


def test_dry_run_changes_nothing(review_manifest, tmp_path):
    manifest, log = corpus(review_manifest, tmp_path)
    before = manifest.read_text(encoding="utf-8"), log.read_text(encoding="utf-8")
    result = apply(review_manifest, manifest, log, [
        {"file": "a.pdf", "doc_type": "tool", "topic": "water",
         "confidence": "high", "note": "a scale"}], extra=["--dry-run"])
    assert result.exit_code == 0
    assert (manifest.read_text(encoding="utf-8"), log.read_text(encoding="utf-8")) == before


def test_emit_takes_the_pending_documents_and_refuses_to_overwrite(review_manifest, tmp_path):
    manifest, log = corpus(review_manifest, tmp_path)
    # Two queued documents, one per packet, so that a second emit under the same name
    # still has something to write and so reaches the overwrite guard.
    manifest.write_text(manifest.read_text(encoding="utf-8")
                        + "c.pdf\tAlso queued\treport\tno\thealth\t9\n", encoding="utf-8")
    review_manifest.write_log(log, [{"file": "a.pdf", "before_doc_type": "report",
                                     "before_topic": "health"},
                                    {"file": "c.pdf", "before_doc_type": "report",
                                     "before_topic": "health"}])
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    (chunks / "a.json").write_text(
        '{"chunks": [{"pages": [1], "text": "Recommandations de bonne pratique"},'
        ' {"pages": [2], "text": "Sommaire"}]}', encoding="utf-8")
    out = tmp_path / "packets"
    def run():
        return CliRunner().invoke(review_manifest.cli, [
            "emit", "--batch", "1", "--out", str(out), "--manifest", str(manifest),
            "--log", str(log), "--chunks", str(chunks), "--name", "batch-001"])
    assert run().exit_code == 0
    sheet = read_tsv(out / "batch-001-decisions.tsv")
    assert [row["file"] for row in sheet] == ["a.pdf"]
    assert sheet[0]["doc_type"] == "", "the sheet is emitted empty so every row is decided"
    text = (out / "batch-001-evidence.md").read_text(encoding="utf-8")
    assert "Recommandations de bonne pratique" in text
    # The first pass's answer comes after the evidence, not before it.
    assert text.index("First pass said") > text.index("Recommandations de bonne pratique")
    assert run().exit_code != 0, "a second emit would silently replace a filled sheet"


def test_a_document_with_no_chunks_still_gets_a_block(review_manifest, tmp_path):
    manifest, log = corpus(review_manifest, tmp_path)
    row = dict(read_tsv(manifest)[0])
    block = review_manifest.evidence(row, [])
    assert "no chunk file" in block and row["title"] in block


def test_an_untouched_row_keeps_its_bytes(review_manifest, tmp_path):
    """One curated row must not rewrite the other 534.

    `manifest.py` writes MANIFEST.tsv through `csv.DictWriter` with its default CRLF
    line terminator. A writer here that disagreed produced a 535-line diff for a
    one-row change, which hides the curation inside the noise and makes two agent
    sessions collide on every line of the file.
    """
    manifest = tmp_path / "MANIFEST.tsv"
    manifest.write_bytes(b"file\ttitle\tdoc_type\tlevel\ttopic\tpages\r\n"
                         b"a.pdf\tA report, or is it\treport\tno\thealth\t30\r\n"
                         b"b.pdf\tHand curated\tstandard\tcore\twater\t12\r\n")
    log = tmp_path / "REVIEW.tsv"
    review_manifest.write_log(log, [{"file": "a.pdf", "before_doc_type": "report",
                                     "before_topic": "health"}])
    result = apply(review_manifest, manifest, log, [
        {"file": "a.pdf", "doc_type": "tool", "topic": "water",
         "confidence": "high", "note": "a rating scale with scoring rules"}])
    assert result.exit_code == 0, result.output
    lines = manifest.read_bytes().split(b"\r\n")
    assert lines[2] == b"b.pdf\tHand curated\tstandard\tcore\twater\t12"
    assert lines[0].startswith(b"file\t")


def test_a_document_out_with_a_reviewer_is_not_emitted_again(review_manifest, tmp_path):
    """Two packets made before either comes back must not cover the same documents.

    Pending means "not applied", so without this the second emit starts at the same
    row as the first, two reviewers read the same thirty documents, and the second
    sheet is refused as already reviewed after the work is done.
    """
    manifest = tmp_path / "MANIFEST.tsv"
    manifest.write_text("file\ttitle\tdoc_type\tlevel\ttopic\tpages\n"
                        "a.pdf\tFirst\treport\tno\thealth\t30\n"
                        "b.pdf\tSecond\treport\tno\twater\t12\n", encoding="utf-8")
    log = tmp_path / "REVIEW.tsv"
    review_manifest.write_log(log, [{"file": "a.pdf", "before_doc_type": "report",
                                     "before_topic": "health"},
                                    {"file": "b.pdf", "before_doc_type": "report",
                                     "before_topic": "water"}])
    out = tmp_path / "packets"

    def emit(name):
        return CliRunner().invoke(review_manifest.cli, [
            "emit", "--batch", "1", "--out", str(out), "--manifest", str(manifest),
            "--log", str(log), "--chunks", str(tmp_path / "none"), "--name", name])

    assert emit("batch-001").exit_code == 0
    assert emit("batch-002").exit_code == 0
    first = [row["file"] for row in read_tsv(out / "batch-001-decisions.tsv")]
    second = [row["file"] for row in read_tsv(out / "batch-002-decisions.tsv")]
    assert first == ["a.pdf"] and second == ["b.pdf"]
    # And with everything spoken for, a third emit says so rather than writing an
    # empty packet somebody would then wait on.
    assert emit("batch-003").exit_code == 0
    assert not (out / "batch-003-decisions.tsv").exists()
