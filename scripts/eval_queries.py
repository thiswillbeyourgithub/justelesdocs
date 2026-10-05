# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "loguru"]
# ///
"""Join the corpus's hand-written evaluation queries to their gold labels.

The queries belong to the corpus, not to the software: they are written by
someone reading that corpus's documents. They live in `$CORPUS_DIR/eval/`, a
tracked part of the corpus, and this script turns them into
`data/EVAL_QUERIES.tsv` (the main set every evaluator reads) and
`data/EVAL_QUERIES_CROSS.tsv` (the crosslingual set, read through `QUERIES=`).

The input files
---------------
`eval/queries.tsv`, tab separated, `#` lines and blank lines ignored, so a
group of queries can carry the note that explains it. One row per query:

set
    `sampled`: the gold (file, page) is joined from `data/EVAL_PASSAGES.tsv`
    by `passage_id` (see `sample_passages.py`). `labelled`: the gold is
    written on the row, because the page was picked by a sampler that reads the
    chunk files and so would pick another page after a rechunk; a (file, page)
    written down cannot drift. `mirror`: a crosslingual twin, see below.
passage_id
    Sampled rows only.
type
    Free label, reported per group: `precise`, `vague`, `crosslingual`, or any
    other kind a corpus finds worth measuring separately (a table cell, a
    number).
query_lang, query
    The question, in the language a reader would type it.
gold_file, gold_page
    Labelled and mirror rows only.
also_acceptable
    Other pages that answer the question just as well, `file.pdf#12;other.pdf#7`.
    A corpus holding several editions or renditions of one document scores a
    perfect answer from the "wrong" edition as a miss otherwise, which makes every
    configuration look worse and makes a change that shuffles editions look like
    an improvement. Written out as the `gold_alt` column.

`eval/unusable_passages.tsv` (`passage_id`, `reason`): sampled passages that get
no query (a reference list, an affiliation block), recorded rather than silently
dropped so a future sampler improvement can be checked against them.

How the labels work
-------------------
Ground truth is (file, page), NOT a chunk id: a chunk-id label would be
invalidated by every change to the chunker. A retrieval result counts as a hit
when the returned chunk comes from the gold file and overlaps the gold page.
This keeps the eval set usable across chunking experiments.

Query types
-----------
precise
    The answer is literally stated in the sampled passage. Measures whether
    the embedding can find a specific fact. These are the queries where a
    miss is unambiguously a retrieval failure.
vague
    A topic-level question the passage is relevant to but does not uniquely
    answer. Measures behaviour on the realistic case where a reader does not
    know the exact wording. The gold label is weaker here: other passages in
    the corpus may legitimately answer just as well, so treat a miss as a
    signal to inspect rather than as a hard failure.
crosslingual
    Asked in the opposite language to the document. The encoder is
    multilingual, so a reader must be able to reach a document written in the
    other language. Nothing else in the eval measures that.

Mirrors and the crosslingual file
---------------------------------
A crosslingual group of a few dozen queries has a standard error near 0.1, so
nothing short of a doubling is resolvable. A `mirror` row is the cheap way out:
it asks an existing row's question in the OTHER language, against the same
gold (file, page), so it needs no new document read, and a mirror and its twin
differ in exactly one thing, the language of the question. Mirrors carry no
`passage_id`, inherit their twin's `gold_alt`, `doc_lang` and `issuer`, and go
ONLY to the crosslingual file, so adding them does not reweight every number
measured over the main set. The crosslingual file holds every main row whose
query language differs from its document's, plus the mirrors.

Known circularity
-----------------
The precise queries are written while looking at the passage, so they share
vocabulary with it and will score optimistically. They are useful for
COMPARING configurations (quantisation, MRL truncation, chunk size) against
each other, and useless as an absolute quality number. The vague and
crosslingual sets are less circular because they deliberately avoid the
passage's wording.

Written with Claude Code.
"""

from __future__ import annotations

import csv
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import click
from loguru import logger
from lib.corpus import in_corpus

#: The column names of `eval/queries.tsv`, in order.
QUERY_COLUMNS = ("set", "passage_id", "type", "query_lang", "query",
                 "gold_file", "gold_page", "also_acceptable")
#: The values the `set` column may take.
SETS = ("sampled", "labelled", "mirror")

#: What separates one alternative from the next in the `gold_alt` cell, and what
#: separates a file name from its page. `src/search.js` never reads this file, so the
#: only readers are `evaluate_rescore.mjs` and `lib/evalbake.py`, both of which split
#: on these two characters. `#` is safe in a file name column as long as no document
#: has one in its name, which `check_alternatives` refuses.
ALT_SEPARATOR = ";"
ALT_PAGE_MARK = "#"


@dataclass(frozen=True)
class Query:
    """One row of `eval/queries.tsv`.

    Attributes
    ----------
    set : str
        One of `SETS`.
    passage_id : str
        Sampled rows only, else empty.
    type : str
        The group the query is reported under.
    lang : str
        The query's language.
    text : str
        The query itself.
    gold_file : str
        Labelled and mirror rows only, else empty.
    gold_page : str
        As written (a decimal string), labelled and mirror rows only.
    alternatives : tuple of (str, int)
        Other acceptable (file, page) answers.
    """

    set: str
    passage_id: str
    type: str
    lang: str
    text: str
    gold_file: str
    gold_page: str
    alternatives: tuple[tuple[str, int], ...]


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a tab-separated file whose `#` lines and blank lines are notes.

    Parameters
    ----------
    path : Path
        The file. Its first line that is neither blank nor a `#` note is the header.

    Returns
    -------
    list of dict
        One dict per data row, keyed by the header. Cells are taken verbatim:
        no quoting, so a query may hold any character but a tab or a newline.

    Raises
    ------
    click.ClickException
        If a row does not have as many cells as the header.
    """
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        cells = line.split("\t")
        if header is None:
            header = cells
            continue
        if len(cells) != len(header):
            raise click.ClickException(
                f"{path}:{number} has {len(cells)} cells where the header has {len(header)}")
        rows.append(dict(zip(header, cells)))
    return rows


def parse_alternatives(cell: str) -> tuple[tuple[str, int], ...]:
    """Split an `also_acceptable` cell into (file, page) pairs.

    Parameters
    ----------
    cell : str
        `"file.pdf#12;other.pdf#7"`, or empty.

    Returns
    -------
    tuple of (str, int)
        The pairs, in the cell's order.
    """
    pairs = []
    for item in filter(None, cell.split(ALT_SEPARATOR)):
        name, _, page = item.rpartition(ALT_PAGE_MARK)
        if not name or not page.isdigit():
            raise click.ClickException(f"cannot read alternative {item!r}: expected file.pdf{ALT_PAGE_MARK}page")
        pairs.append((name, int(page)))
    return tuple(pairs)


def read_queries(path: Path) -> list[Query]:
    """Read and check `eval/queries.tsv`.

    Parameters
    ----------
    path : Path
        The file.

    Returns
    -------
    list of Query
        In file order, which fixes the query ids.

    Raises
    ------
    click.ClickException
        On a missing column, an unknown `set`, a row missing the field its set
        needs, or a mirror carrying alternatives of its own.
    """
    rows = read_tsv(path)
    if rows and (missing := set(QUERY_COLUMNS) - rows[0].keys()):
        raise click.ClickException(f"{path} lacks the column(s) {sorted(missing)}")
    queries = []
    for row in rows:
        query = Query(set=row["set"], passage_id=row["passage_id"], type=row["type"],
                      lang=row["query_lang"], text=row["query"], gold_file=row["gold_file"],
                      gold_page=row["gold_page"],
                      alternatives=parse_alternatives(row["also_acceptable"]))
        if query.set not in SETS:
            raise click.ClickException(f"unknown set {query.set!r} for {query.text!r}, expected one of {SETS}")
        if query.set == "sampled" and not query.passage_id:
            raise click.ClickException(f"sampled query {query.text!r} has no passage_id")
        if query.set != "sampled" and not (query.gold_file and query.gold_page.isdigit()):
            raise click.ClickException(f"{query.set} query {query.text!r} needs a gold_file and a gold_page")
        if query.set == "mirror" and query.alternatives:
            raise click.ClickException(f"mirror {query.text!r} lists alternatives; it inherits its twin's")
        queries.append(query)
    return queries


def alt_cell(query: Query) -> str:
    """Render a query's extra acceptable answers as one TSV cell.

    Parameters
    ----------
    query : Query
        The query.

    Returns
    -------
    str
        `"file.pdf#12;other.pdf#7"`, or an empty string when the query has only
        the one gold answer, which is the usual case.
    """
    return ALT_SEPARATOR.join(f"{name}{ALT_PAGE_MARK}{page}" for name, page in query.alternatives)


def check_alternatives(queries: list[Query], served: dict[str, dict[str, str]]) -> None:
    """Fail if an alternative label points nowhere, or cannot be written down.

    Parameters
    ----------
    queries : list of Query
        Every query read.
    served : dict
        The manifest keyed by file name. Empty when there is no manifest to read,
        in which case only the syntactic checks run.

    Raises
    ------
    click.ClickException
        If an alternative names a document outside the corpus or a page the
        document does not have, or if a file name contains the separator
        characters the cell is built from.
    """
    problems: list[str] = []
    for query in queries:
        for name, page in query.alternatives:
            if ALT_SEPARATOR in name or ALT_PAGE_MARK in name:
                problems.append(f"{name!r} cannot be written in a gold_alt cell")
                continue
            if not served:
                continue
            if name not in served:
                problems.append(f"{name!r} is not in the corpus ({query.text[:40]}...)")
                continue
            total = served[name].get("pages", "")
            if total.isdigit() and not 1 <= page <= int(total):
                problems.append(f"{name!r} has {total} pages, not {page}")
    if problems:
        raise click.ClickException("; ".join(problems))


def check_orthography(queries: list[Query]) -> None:
    """Fail if an English query carries an accent.

    Parameters
    ----------
    queries : list of Query
        Every query read.

    Raises
    ------
    click.ClickException
        Listing every offending query.

    Notes
    -----
    This guards a bug that happened twice in opposite directions. Queries in an
    accented language were first written WITHOUT accents, which is an accidental
    confound: the corpus is accented, so unaccented queries depress that
    language's scores uniformly and the measurement blames the retriever for a
    typing habit. Fixing that with a word-level accent map then over-reached and
    accented English queries, which is the same confound mirrored onto the
    crosslingual set.

    Only the English direction is checkable mechanically: an English query must
    be unchanged by stripping combining marks. A query in an accented language
    legitimately may or may not contain an accent, so no assertion can cover it.
    """
    offenders = [
        query.text for query in queries
        if query.lang == "en" and unicodedata.normalize(
            "NFC", "".join(c for c in unicodedata.normalize("NFD", query.text)
                           if unicodedata.category(c) != "Mn")) != query.text
    ]
    if offenders:
        raise click.ClickException(
            "English queries carrying accents, which biases the crosslingual "
            f"measurement: {offenders}")


@click.command()
@click.option("--queries", "queries_path", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              **in_corpus("eval/queries.tsv"),
              help="The hand-written queries, see the module docstring for the columns.")
@click.option("--unusable", type=click.Path(dir_okay=False, path_type=Path),
              **in_corpus("eval/unusable_passages.tsv"),
              help="Sampled passages that deliberately get no query (passage_id, reason). "
                   "Optional.")
@click.option("--passages", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              **in_corpus("data/EVAL_PASSAGES.tsv"),
              help="Sampled passages produced by sample_passages.py.")
@click.option("--out", type=click.Path(path_type=Path),
              **in_corpus("data/EVAL_QUERIES.tsv"),
              help="Where to write the labelled query set.")
@click.option("--out-crosslingual", type=click.Path(path_type=Path),
              **in_corpus("data/EVAL_QUERIES_CROSS.tsv"),
              help="Where to write the crosslingual subset: the rows whose query "
                   "language differs from their document's, plus the mirrors. "
                   "Same columns, so an evaluator reads it through QUERIES=.")
@click.option("--manifest", type=click.Path(path_type=Path),
              **in_corpus("data/MANIFEST.tsv"),
              help="Used to label the labelled queries' language and issuer, and to "
                   "catch a gold document that has left the corpus.")
def main(queries_path: Path, unusable: Path, passages: Path, out: Path,
         out_crosslingual: Path, manifest: Path) -> None:
    """Join the hand-written queries to their gold (file, page) labels."""
    queries = read_queries(queries_path)
    check_orthography(queries)
    sampled = [q for q in queries if q.set == "sampled"]
    labelled = [q for q in queries if q.set == "labelled"]
    mirrors = [q for q in queries if q.set == "mirror"]
    unusable_passages = ({row["passage_id"]: row["reason"] for row in read_tsv(unusable)}
                         if unusable.exists() else {})

    with passages.open(newline="", encoding="utf-8") as handle:
        by_id = {row["passage_id"]: row for row in csv.DictReader(handle, delimiter="\t")}

    # A passage id that is missing from the sample means the seed or the page
    # filter changed and the labels no longer point where the query author
    # looked. That silently corrupts every metric, so fail loud.
    unknown = {q.passage_id for q in sampled} - by_id.keys()
    if unknown:
        raise click.ClickException(
            f"queries reference passages absent from {passages}: {sorted(unknown)}. "
            "The passage sample changed; re-review the affected queries."
        )
    uncovered = by_id.keys() - {q.passage_id for q in sampled} - unusable_passages.keys()
    if uncovered:
        logger.warning(f"sampled passages with no query and no exclusion reason: {sorted(uncovered)}")

    rows = []
    for index, query in enumerate(sampled, start=1):
        passage = by_id[query.passage_id]
        rows.append({
            "query_id": f"q{index:03d}",
            "passage_id": query.passage_id,
            "type": query.type,
            "query_lang": query.lang,
            "query": query.text,
            "gold_file": passage["file"],
            "gold_page": passage["page"],
            "gold_alt": alt_cell(query),
            "doc_lang": passage["language"],
            "issuer": passage["issuer"],
        })

    # The labelled queries carry their own gold, so they only need the manifest for
    # the two descriptive columns, plus the check that the document is still served:
    # a gold file that left the corpus turns its query into a permanent miss that
    # looks like a retrieval failure.
    served: dict[str, dict[str, str]] = {}
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as handle:
            served = {row["file"]: row for row in csv.DictReader(handle, delimiter="\t")}
    gone = sorted({q.gold_file for q in labelled} - served.keys()) if served else []
    if gone:
        raise click.ClickException(
            f"labelled queries point at documents absent from {manifest}: {gone}"
        )
    check_alternatives(queries, served)
    for offset, query in enumerate(labelled, start=1):
        rows.append({
            "query_id": f"q{len(sampled) + offset:03d}",
            "passage_id": "",
            "type": query.type,
            "query_lang": query.lang,
            "query": query.text,
            "gold_file": query.gold_file,
            "gold_page": query.gold_page,
            "gold_alt": alt_cell(query),
            "doc_lang": served.get(query.gold_file, {}).get("language", ""),
            "issuer": served.get(query.gold_file, {}).get("issuer", ""),
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    # The crosslingual file. Selection is on the LANGUAGES, not on the `type`
    # column: `type` is one label per query and some crossing queries are typed by
    # what they ask instead, so reading the type would drop them. A row whose
    # document has no detected language ("?") is not evidence either way and stays out.
    crossing = [row for row in rows
                if row["doc_lang"] and row["doc_lang"] != "?"
                and row["query_lang"] != row["doc_lang"]]
    twins = {(row["gold_file"], str(row["gold_page"])): row for row in rows}
    for offset, query in enumerate(mirrors, start=1):
        twin = twins.get((query.gold_file, query.gold_page))
        if twin is None:
            raise click.ClickException(
                f"mirrored query {query.text!r} points at {query.gold_file} page {query.gold_page}, "
                "which is not the gold of any query. A mirror must mirror something."
            )
        if query.lang == twin["doc_lang"]:
            raise click.ClickException(
                f"mirrored query {query.text!r} is in {query.lang} and so is its document: "
                "it does not cross a language and belongs in the labelled set."
            )
        crossing.append({
            "query_id": f"x{offset:03d}",
            "passage_id": "",
            "type": "crosslingual",
            "query_lang": query.lang,
            "query": query.text,
            "gold_file": query.gold_file,
            "gold_page": query.gold_page,
            "gold_alt": twin["gold_alt"],
            "doc_lang": twin["doc_lang"],
            "issuer": twin["issuer"],
        })
    with out_crosslingual.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(crossing)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["type"]] = counts.get(row["type"], 0) + 1
    logger.success(f"wrote {len(rows)} queries over {len(by_id) - len(unusable_passages)} passages to {out}")
    logger.info(" ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if unusable_passages:
        logger.info(f"excluded passages: {', '.join(f'{k} ({v})' for k, v in unusable_passages.items())}")
    directions: dict[str, int] = {}
    for row in crossing:
        key = f"{row['query_lang']}->{row['doc_lang']}"
        directions[key] = directions.get(key, 0) + 1
    logger.success(f"wrote {len(crossing)} crosslingual queries to {out_crosslingual}, "
                   f"{len(mirrors)} of them mirrors")
    logger.info(" ".join(f"{k}={v}" for k, v in sorted(directions.items())))


if __name__ == "__main__":
    main()
