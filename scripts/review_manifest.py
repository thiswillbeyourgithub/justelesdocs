# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "loguru"]
# ///
"""Re-decide `doc_type` and `topic` on the 418 rows a shallow first pass classified.

The corpus arrived in two halves. The first 114 documents were curated by hand, one
at a time, from the document itself. The 418 that followed (an agency's bulk download and
two later sorting passes) were classified on 2026-09-24 from each document's TITLE
and its first ~700 characters, which is a cover page and not a reading. That is
enough to tell a guideline from a journal article and not enough for much else: a
background report and the recommendation it argues for share a cover, a "rapport" that is
really guidance looks like a report on page one, and a document's THEME is usually
stated in its table of contents rather than on its title page.

Why it matters more here than the word "metadata" suggests: `doc_type` and `topic`
are closed-vocabulary FILTERS. A wrong value does not show up as a wrong label on a
card, it removes the document from the filter it belongs in, and the reader who ticks
a topic is never told that the answer they wanted is one tick away. The
`guideline` tier is worse still, because it is on by default: a recommendation filed
as a "report" is tier "no" and is hidden from every default search.

So this is the machinery for a second pass that reads more of each document, keeps a
record of what it changed, and cannot write a slug that does not exist.

Three commands, in the order they are used:

    uv run scripts/review_manifest.py init          # once: build the queue
    uv run scripts/review_manifest.py emit --batch 20
    uv run scripts/review_manifest.py apply \\
        data/private/review/batch-001-decisions.tsv --by "Claude Opus 5"

`emit` writes the evidence a reviewer reads and an empty decision sheet; `apply`
validates the sheet against the vocabularies, writes the cells into
`data/MANIFEST.tsv` and appends one row per document to `data/MANIFEST_REVIEW.tsv`,
which is both the queue and the record of the pass.

Two things it deliberately does not do:

- It does not extract text from the PDFs. `data/chunks/` already holds every
  document's text with its page numbers, built and content-hash gated by `chunk.py`,
  so the evidence is read from there. A second extractor would be a second answer to
  "what does this document say", and the two would drift.
- It does not compute the `guideline` tier. `manifest.py` derives it from `doc_type`
  and owns that mapping; this blanks the cell on every row whose `doc_type` changed
  and leaves `manifest.py` to fill it in on its next run, which it does for blank
  cells only. `apply` says so when it finishes.

Written by Claude Code (Opus 5).
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import click
from loguru import logger

from lib import atomic
from lib import manifest_io
from lib import manifest_policy
from lib.corpus import corpus_dir, in_corpus

#: The three commits of the first pass. `init` reads the rows they touched, which is
#: the definition of "classified by the shallow pass" and needs no second list to
#: fall out of date: one commit filled `guideline`, one `doc_type`, one `topic`, and
#: all three touched exactly the same 418 rows.
FIRST_PASS = ("5dbeff9", "9ff6487", "645d1c2")

#: The review log, which is the queue as well: a row with no `reviewed_on` is a
#: document nobody has read yet. One file rather than a queue and a record, because
#: two files would need reconciling and the reconciliation is the bug.
LOG_COLUMNS = ("file", "before_doc_type", "before_topic", "doc_type", "topic",
               "confidence", "note", "reviewed_by", "reviewed_on")

#: What the reviewer fills in. `file` identifies the row; the rest is the decision.
DECISION_COLUMNS = ("file", "doc_type", "topic", "confidence", "note")

CONFIDENCE = ("high", "medium", "low")

#: How much of each document goes into the evidence packet. Four windows rather than
#: one long head, because the four answer different questions: the cover says who
#: issued it and what it calls itself, the pages after it usually hold the table of
#: contents (the best single source for `topic`), the middle says what the body of
#: the document actually is (recommendations, evidence tables, a rating scale), and
#: the end carries conclusions or annexes. A single 6000-character head would spend
#: most of its budget on a title page and a list of working-group members.
HEAD_CHARS = 2600
TOC_CHARS = 1800
MIDDLE_CHARS = 1400
TAIL_CHARS = 900


def read_log(path: Path) -> list[dict[str, str]]:
    """Every review-log row, in file order.

    Parameters
    ----------
    path
        `data/MANIFEST_REVIEW.tsv`.

    Returns
    -------
    list of dict
        One dict per row, keyed by `LOG_COLUMNS`.

    Raises
    ------
    click.ClickException
        If the file does not exist (run `init`) or has lost a column.
    """
    if not path.is_file():
        raise click.ClickException(f"{path} does not exist. Run `init` first.")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    missing = set(LOG_COLUMNS) - set(rows[0] if rows else {})
    if missing:
        raise click.ClickException(f"{path} has no {sorted(missing)} column")
    return rows


def write_log(path: Path, rows: list[dict[str, str]]) -> None:
    """Write the review log back, with its columns in `LOG_COLUMNS` order.

    Written through `csv.DictWriter` for the reason `lib/manifest_io` exists: a note
    holding a quotation mark or a tab has to survive the round trip.
    """
    with atomic.open_atomic(path, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LOG_COLUMNS), delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in LOG_COLUMNS})


def first_pass_files(shas: tuple[str, ...] = FIRST_PASS) -> set[str]:
    """The manifest rows the first pass wrote, read from its own commits.

    Parameters
    ----------
    shas
        The commits to read. Each is shown as a diff of `data/MANIFEST.tsv` alone.

    Returns
    -------
    set of str
        The filename in column one of every line those commits ADDED.

    Notes
    -----
    Read from git rather than recomputed from the manifest because there is nothing
    in the manifest that says who filled a cell. This runs once, in `init`; the
    answer is then written into the log and git is never consulted again, so a
    shallow clone or a rewritten history cannot invalidate a review in progress.
    """
    files: set[str] = set()
    for sha in shas:
        diff = subprocess.run(
            ["git", "-C", str(corpus_dir()), "show", "--format=", "-U0", sha, "--",
             "data/MANIFEST.tsv"],
            capture_output=True, text=True, check=True).stdout
        for line in diff.split("\n"):
            if line.startswith("+") and not line.startswith("+++"):
                name = line[1:].split("\t")[0].strip('"')
                if name.lower().endswith(".pdf"):
                    files.add(name)
    return files


def document_text(chunks_dir: Path, pdf_name: str) -> list[tuple[int, str]]:
    """One document's chunks as (first page, text) pairs, in document order.

    Returns an empty list when the document has no chunk file, which happens while a
    document is new and `chunk.py` has not run since. The caller says so in the
    packet rather than failing the batch: a reviewer can still judge a document from
    its manifest row, and a missing chunk file is `verify_chunks.py`'s business.
    """
    path = chunks_dir / f"{Path(pdf_name).stem}.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [(int(chunk["pages"][0]) if chunk.get("pages") else 0, chunk["text"])
            for chunk in payload.get("chunks", [])]


def window(pairs: list[tuple[int, str]], start: int, limit: int) -> str:
    """`limit` characters of text, starting at chunk index `start`.

    Parameters
    ----------
    pairs
        The document, as `document_text` returns it.
    start
        Index of the first chunk to take.
    limit
        How many characters to keep.

    Returns
    -------
    str
        The text, with runs of blank lines collapsed and an ellipsis if it was cut.
    """
    out: list[str] = []
    total = 0
    for _, text in pairs[start:]:
        out.append(text)
        total += len(text)
        if total >= limit:
            break
    joined = "\n".join(out).strip()
    joined = "\n".join(line.rstrip() for line in joined.split("\n") if line.strip())
    return joined[:limit] + ("..." if len(joined) > limit else "")


def evidence(row: dict[str, str], pairs: list[tuple[int, str]]) -> str:
    """One document's block in the evidence packet.

    The manifest facts first (a reviewer needs to know it is a 300-page agency document
    before reading a word of it), then four windows into the text, then the first
    pass's values LAST and labelled as shallow. Last on purpose: the sheet asks for a
    value on every row, so the reviewer decides from the document and then sees what
    the first pass said, rather than being handed an answer to agree with.
    """
    facts = [f"- title: {row.get('title', '') or '(none)'}",
             f"- issuer: {row.get('issuer', '') or '(none)'}"
             f" | country: {row.get('country', '') or '?'}"
             f" | year: {row.get('year', '') or '?'}"
             f" | language: {row.get('language', '') or '?'}"
             f" | pages: {row.get('pages', '') or '?'}"]
    if row.get("family"):
        facts.append(f"- family: {row['family']} | rendition: {row.get('rendition', '')}")
    if row.get("notes"):
        facts.append(f"- notes: {row['notes']}")

    parts = [f"## {row['file']}", "", *facts, ""]
    if not pairs:
        parts += ["*(no chunk file: this document has not been chunked, judge it from"
                  " its title and the facts above)*", ""]
    else:
        middle = len(pairs) // 2
        tail = max(0, len(pairs) - max(1, len(pairs) // 10))
        sections = [("first pages", window(pairs, 0, HEAD_CHARS)),
                    ("what follows the cover (table of contents, scope)",
                     window(pairs, min(2, len(pairs) - 1), TOC_CHARS)),
                    ("middle of the document", window(pairs, middle, MIDDLE_CHARS)),
                    ("end of the document", window(pairs, tail, TAIL_CHARS))]
        for label, text in sections:
            if text:
                parts += [f"### {label}", "", "```", text, "```", ""]
    parts += [f"First pass said (shallow: title + ~700 characters): doc_type="
              f"`{row.get('doc_type', '') or '(blank)'}`, topic="
              f"`{row.get('topic', '') or '(blank)'}`.", ""]
    return "\n".join(parts)


def check_decision(row: dict[str, str], vocab: dict[str, list[str]], separator: str,
                   before: dict[str, str]) -> list[str]:
    """Everything wrong with one decision row, as sentences.

    Parameters
    ----------
    row
        One row of a decisions TSV.
    vocab
        `doc_type` and `topic` vocabularies, from `manifest.py`.
    separator
        The multi-value separator the manifest uses.
    before
        That document's log row, for the checks that need the previous value.

    Returns
    -------
    list of str
        Empty when the row is usable. Collected rather than raised one at a time so
        a reviewer gets every problem in their sheet from one run.

    Notes
    -----
    A blank cell is an error, not a "leave it alone". The sheet is emitted empty, so
    blank means the reviewer skipped the row, and silently keeping a value a shallow
    pass wrote is exactly what this pass exists to stop.
    """
    problems: list[str] = []
    for column in ("doc_type", "topic"):
        cell = (row.get(column) or "").strip()
        if not cell:
            problems.append(f"{column} is blank: every row needs a decision")
            continue
        values = [value.strip() for value in cell.split(separator)]
        if any(not value for value in values):
            problems.append(f"{column}={cell!r} has an empty value (a stray {separator!r})")
        unknown = [value for value in values if value and value not in vocab[column]]
        if unknown:
            problems.append(f"{column}: {unknown} not in the vocabulary")
        if len(set(values)) != len(values):
            problems.append(f"{column}={cell!r} repeats a value")
    confidence = (row.get("confidence") or "").strip()
    if confidence not in CONFIDENCE:
        problems.append(f"confidence={confidence!r} is not one of {list(CONFIDENCE)}")
    note = (row.get("note") or "").strip()
    changed = ((row.get("doc_type") or "").strip() != before.get("before_doc_type", "")
               or (row.get("topic") or "").strip() != before.get("before_topic", ""))
    if changed and not note:
        problems.append("note is blank on a row that changes a value: the note IS the record")
    return problems


@click.group()
def cli() -> None:
    """Second pass over the manifest values a shallow first pass wrote."""


@cli.command()
@click.option("--manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              **in_corpus("data/MANIFEST.tsv"))
@click.option("--log", type=click.Path(dir_okay=False, path_type=Path),
              **in_corpus("data/MANIFEST_REVIEW.tsv"))
@click.option("--force", is_flag=True,
              help="Rebuild the log even though it exists. Discards every review in it.")
def init(manifest: Path, log: Path, force: bool) -> None:
    """Build the review queue from the first pass's own commits."""
    if log.is_file() and not force:
        raise click.ClickException(
            f"{log} already exists. It holds the reviews done so far; --force would "
            "throw them away.")
    rows = manifest_io.read_by_file(manifest)
    queue = sorted(first_pass_files())
    missing = [name for name in queue if name not in rows]
    if missing:
        raise click.ClickException(
            f"{len(missing)} file(s) classified by the first pass are no longer in "
            f"the manifest, e.g. {missing[:3]}. Sort that out before reviewing.")
    write_log(log, [{"file": name,
                     "before_doc_type": rows[name].get("doc_type", ""),
                     "before_topic": rows[name].get("topic", "")}
                    for name in queue])
    logger.success(f"{len(queue)} documents queued in {log}, "
                   f"{len(rows) - len(queue)} hand-curated rows left alone")


@cli.command()
@click.option("--log", type=click.Path(dir_okay=False, path_type=Path),
              **in_corpus("data/MANIFEST_REVIEW.tsv"))
def status(log: Path) -> None:
    """How much of the queue is read, and how much is left."""
    rows = read_log(log)
    done = [row for row in rows if row.get("reviewed_on")]
    changed = [row for row in done
               if row["doc_type"] != row["before_doc_type"]
               or row["topic"] != row["before_topic"]]
    logger.info(f"{len(done)}/{len(rows)} reviewed, {len(rows) - len(done)} pending")
    if done:
        low = [row for row in done if row.get("confidence") == "low"]
        logger.info(f"{len(changed)} changed a value ({100 * len(changed) / len(done):.0f}% "
                    f"of what has been read), {len(low)} marked low confidence")


@cli.command()
@click.option("--batch", default=20, show_default=True, help="Documents in this packet.")
@click.option("--out", type=click.Path(file_okay=False, path_type=Path),
              **in_corpus("data/private/review"),
              help="Where the packets go. Under data/private/, which .gitignore already "
                   "excludes, because a packet quotes several thousand characters of a "
                   "copyrighted document and only the corpus may hold that.")
@click.option("--manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              **in_corpus("data/MANIFEST.tsv"))
@click.option("--log", type=click.Path(dir_okay=False, path_type=Path),
              **in_corpus("data/MANIFEST_REVIEW.tsv"))
@click.option("--chunks", type=click.Path(file_okay=False, path_type=Path),
              **in_corpus("data/chunks"))
@click.option("--name", default="", help="Packet name. Default: the next batch-NNN.")
def emit(batch: int, out: Path, manifest: Path, log: Path, chunks: Path, name: str) -> None:
    """Write the next batch's evidence and its empty decision sheet."""
    rows = read_log(log)
    out.mkdir(parents=True, exist_ok=True)
    # A document stops being pending when its sheet is APPLIED, so several packets
    # emitted before any of them comes back would all start at the same row. That is
    # the whole failure mode of reviewing in parallel: two reviewers read the same
    # thirty documents and the second sheet to be applied is refused as already
    # reviewed, after the work. So a document already sitting in an unapplied sheet
    # here is spoken for.
    spoken_for: set[str] = set()
    for sheet in sorted(out.glob("*-decisions.tsv")):
        with sheet.open(encoding="utf-8", newline="") as handle:
            spoken_for.update((row.get("file") or "").strip()
                              for row in csv.DictReader(handle, delimiter="\t"))
    pending = [row for row in rows
               if not row.get("reviewed_on") and row["file"] not in spoken_for]
    if not pending:
        logger.success(f"nothing to emit: {len(spoken_for)} document(s) are out with a "
                       "reviewer and the rest are done")
        return
    take = pending[:batch]
    if not name:
        name = f"batch-{1 + len(list(out.glob('*-decisions.tsv'))):03d}"
    evidence_path = out / f"{name}-evidence.md"
    decisions_path = out / f"{name}-decisions.tsv"
    for path in (evidence_path, decisions_path):
        if path.exists():
            raise click.ClickException(f"{path} exists. Apply it, or pass another --name.")

    manifest_rows = manifest_io.read_by_file(manifest)
    blocks = [f"# {name}: {len(take)} documents to classify", "",
              "Each block below is one document: its manifest facts, then four windows",
              "into its text. Decide `doc_type` and `topic` for every one of them and",
              f"write them into `{decisions_path}`.", "",
              "The vocabularies, the distinctions that go wrong and what the note is for",
              "are in the corpus's own review brief (`$CORPUS_DIR/docs/manifest_review.md`",
              "for the corpus this tool was written for). Read it first.", ""]
    for row in take:
        merged = dict(manifest_rows[row["file"]])
        merged["file"] = row["file"]
        blocks.append(evidence(merged, document_text(chunks, row["file"])))
    evidence_path.write_text("\n".join(blocks), encoding="utf-8")

    with decisions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(DECISION_COLUMNS), delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        for row in take:
            writer.writerow({"file": row["file"], "doc_type": "", "topic": "",
                             "confidence": "", "note": ""})
    logger.success(f"{len(take)} documents in {evidence_path} "
                   f"({evidence_path.stat().st_size // 1024} kB) and {decisions_path}")


@cli.command()
@click.argument("decisions", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--by", required=True, help="Who reviewed it, e.g. 'Claude Opus 5 (subagent)'.")
@click.option("--on", default="", help="Review date. Default: today.")
@click.option("--manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              **in_corpus("data/MANIFEST.tsv"))
@click.option("--log", type=click.Path(dir_okay=False, path_type=Path),
              **in_corpus("data/MANIFEST_REVIEW.tsv"))
@click.option("--regrade", is_flag=True, help="Allow rows that have already been reviewed.")
@click.option("--dry-run", is_flag=True, help="Check the sheet and change nothing.")
def apply(decisions: Path, by: str, on: str, manifest: Path, log: Path,
          regrade: bool, dry_run: bool) -> None:
    """Validate a decision sheet, write it into the manifest, record it in the log."""
    import datetime

    on = on or datetime.date.today().isoformat()
    policy = manifest_policy.policy()
    vocab, separator = policy["vocabularies"], policy["separator"]
    with decisions.open(encoding="utf-8", newline="") as handle:
        sheet = list(csv.DictReader(handle, delimiter="\t"))
    if not sheet:
        raise click.ClickException(f"{decisions} has no rows")

    log_rows = read_log(log)
    by_file = {row["file"]: row for row in log_rows}
    problems: list[str] = []
    for row in sheet:
        name = (row.get("file") or "").strip()
        if name not in by_file:
            problems.append(f"{name!r}: not in the review queue")
            continue
        if by_file[name].get("reviewed_on") and not regrade:
            problems.append(f"{name}: already reviewed on {by_file[name]['reviewed_on']}"
                            " (--regrade to overrule)")
            continue
        problems += [f"{name}: {problem}"
                     for problem in check_decision(row, vocab, separator, by_file[name])]
    if problems:
        for problem in problems:
            logger.error(problem)
        raise click.ClickException(f"{len(problems)} problem(s) in {decisions}: nothing written")

    manifest_rows = manifest_io.read_rows(manifest)
    index = {row["file"]: row for row in manifest_rows}
    changed_type = changed_topic = 0
    for row in sheet:
        name = row["file"].strip()
        doc_type = row["doc_type"].strip()
        topic = row["topic"].strip()
        target = index[name]
        if target.get("doc_type", "") != doc_type:
            changed_type += 1
            # Blanked, not recomputed: `manifest.py` owns the doc_type -> tier mapping
            # and fills this cell when it is empty. Recomputing it here would be a
            # second copy of that mapping, and the two would disagree the day somebody
            # adds a doc_type.
            target[manifest_policy.TIER_COLUMN] = ""
        if target.get("topic", "") != topic:
            changed_topic += 1
        target["doc_type"] = doc_type
        target["topic"] = topic

        entry = by_file[name]
        entry.update({"doc_type": doc_type, "topic": topic,
                      "confidence": row["confidence"].strip(),
                      "note": (row.get("note") or "").strip(),
                      "reviewed_by": by, "reviewed_on": on})

    if dry_run:
        logger.info(f"{len(sheet)} rows would be written: {changed_type} doc_type, "
                    f"{changed_topic} topic changed. Nothing touched (--dry-run).")
        return

    # No `lineterminator`: `manifest.py` writes this file with csv's default CRLF, and
    # a writer that disagrees rewrites all 535 lines to change one, which buries the
    # curation in a diff nobody can read.
    with atomic.open_atomic(manifest, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(manifest_rows)
    write_log(log, log_rows)
    logger.success(f"{len(sheet)} rows reviewed by {by}: {changed_type} doc_type and "
                   f"{changed_topic} topic changed")
    if changed_type:
        logger.warning(f"{changed_type} row(s) have a blank `{manifest_policy.TIER_COLUMN}` "
                       "waiting for `uv run scripts/manifest.py` to derive the tier from "
                       "the new doc_type")


if __name__ == "__main__":
    cli()
