# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "loguru", "numpy", "onnxruntime", "tokenizers"]
# ///
"""Measure whether the page a chunk sits on should have a say in its rank.

The question, from the brief: a hit is shown as a PDF page, so would it help to
embed whole pages as well as chunks, and either rank pages first and look for
chunks inside the winners, or blend the page's score into the chunk's?

Three schemes are measured here, all against the same 117 queries and the same
(file, page) ground truth `evaluate.py` uses, so the numbers are comparable with
the tables in DESIGN.md:

`chunks`
    The shipped scheme. Chunk cosine alone.
`blend`
    ``alpha * chunk + (1 - alpha) * page``, where `page` is the cosine of the
    whole-page vector for the page the chunk sits on. A chunk spanning two pages
    takes the better of the two, because the reader is sent to one page and it
    will be that one.
`two-stage`
    Rank pages, keep the top K, and rank only the chunks living on those pages by
    their own cosine. This is the "find the best pages, then the best chunks in
    them" of the brief, in its strict form: a chunk on a page outside the top K
    cannot be returned at all.

Everything but the scoring is borrowed from `evaluate.py` (corpus loading, query
embedding, Matryoshka narrowing, int8 round-tripping, the metrics). A second
definition of recall@5 would be a second answer to the same question.

Needs two bakes: a chunk index and a page index, the latter from chunks built
with ``chunk.py --page-chunks``. See DESIGN.md, "page-level vectors".

Written with Claude Code.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import click
import numpy as np
from loguru import logger
from lib.corpus import in_corpus

# How deep the ranking is read. Same as evaluate.py's, so that a number here can
# be compared with a number there without a footnote.
TOP_K = 10


def load_module(path: Path, name: str) -> Any:
    """Import a single-file script by path.

    `evaluate.py` is a PEP 723 script rather than a package, so it cannot be
    imported by name. It is still the definition of what the metrics mean, and
    importing it is what keeps that definition single.

    Parameters
    ----------
    path : Path
        The script file.
    name : str
        Module name to register it under.

    Returns
    -------
    Any
        The imported module.

    Raises
    ------
    click.ClickException
        If the file does not exist, since every caller passes a path a user gave.
    """
    if not path.exists():
        raise click.ClickException(f"{name} not found at {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def page_rows(corpus: Any) -> dict[tuple[str, int], int]:
    """Map (file, page) to the row of the page index holding that page.

    Parameters
    ----------
    corpus : evaluate.Corpus
        A corpus loaded from ``chunk.py --page-chunks`` output, where every row
        is exactly one page.

    Returns
    -------
    dict
        (filename, 1-based page) to row index.

    Raises
    ------
    click.ClickException
        If a row spans several pages, which means the index was not built with
        `--page-chunks` and every number below would be meaningless.
    """
    rows: dict[tuple[str, int], int] = {}
    for row, (file, pages) in enumerate(zip(corpus.files, corpus.pages)):
        if len(pages) != 1:
            raise click.ClickException(
                f"{file}: page index row {row} covers {len(pages)} pages. "
                "Build it with chunk.py --page-chunks."
            )
        rows[(file, next(iter(pages)))] = row
    return rows


def chunk_to_pages(chunks: Any, rows: dict[tuple[str, int], int]) -> np.ndarray:
    """For each chunk row, the page-index rows it sits on.

    Parameters
    ----------
    chunks : evaluate.Corpus
        The chunk index.
    rows : dict
        Output of `page_rows`.

    Returns
    -------
    np.ndarray
        (chunks, max_pages) int array, padded with -1. Padded rather than ragged
        because the whole point is to gather page scores with one NumPy call per
        query set rather than a Python loop over 27000 chunks.

    Notes
    -----
    A missing page is padded away rather than raising. Boilerplate stripping can
    empty a page completely, in which case it has no vector, while a chunk
    spanning the page break either side of it still lists it.
    """
    width = max((len(p) for p in chunks.pages), default=1)
    mapped = np.full((len(chunks.files), max(width, 1)), -1, dtype=np.int64)
    for row, (file, pages) in enumerate(zip(chunks.files, chunks.pages)):
        for slot, page in enumerate(sorted(pages)):
            index = rows.get((file, page))
            if index is not None:
                mapped[row, slot] = index
    return mapped


def page_scores_for_chunks(page_similarity: np.ndarray, mapped: np.ndarray) -> np.ndarray:
    """Lift page scores onto chunk rows, taking the best page of each chunk.

    Parameters
    ----------
    page_similarity : np.ndarray
        (queries, pages) cosines.
    mapped : np.ndarray
        (chunks, slots) page rows, -1 where padded.

    Returns
    -------
    np.ndarray
        (queries, chunks) matrix holding, for each chunk, the score of its best
        page. Chunks whose pages are all missing score -1, which is below any
        cosine, so they can only be reached on their own merit.
    """
    padded = np.concatenate(
        [page_similarity, np.full((page_similarity.shape[0], 1), -1.0, dtype=page_similarity.dtype)],
        axis=1,
    )
    gathered = padded[:, mapped]              # (queries, chunks, slots)
    return gathered.max(axis=2)


def previous_page_rows(chunks: Any, rows: dict[tuple[str, int], int]) -> np.ndarray:
    """For each chunk row, the page-index row of the page BEFORE it.

    The question this exists for: a passage often continues an argument that starts
    on the page before it, and a reader asking about that argument names words that
    are on neither page alone. Whether the previous page's vector is context or noise
    is measurable, and this is the mapping that makes it measurable.

    Parameters
    ----------
    chunks : evaluate.Corpus
        The chunk index.
    rows : dict
        Output of `page_rows`.

    Returns
    -------
    np.ndarray
        (chunks, 1) int array holding one page row per chunk, -1 where the chunk
        starts on page 1 or where that page has no vector (boilerplate stripping can
        empty a page completely). Shaped for `page_scores_for_chunks`, which takes
        any number of slots.

    Notes
    -----
    The page before the chunk's FIRST page, not before its last: a chunk straddling
    a page break already holds both, and its context is what came before the pair.
    """
    mapped = np.full((len(chunks.files), 1), -1, dtype=np.int64)
    for row, (file, pages) in enumerate(zip(chunks.files, chunks.pages)):
        if not pages:
            continue
        index = rows.get((file, min(pages) - 1))
        if index is not None:
            mapped[row, 0] = index
    return mapped


def report(name: str, metrics: dict[str, Any], rows: list[dict[str, Any]], **extra: Any) -> None:
    """Log one line of results and keep it for the TSV.

    Parameters
    ----------
    name : str
        Scheme name.
    metrics : dict
        Output of `evaluate.score`, keyed by query bucket.
    rows : list of dict
        Accumulator for the output file.
    **extra : Any
        Scheme parameters (alpha, k) recorded alongside the numbers.
    """
    overall = metrics["all"]
    detail = " ".join(f"{key}={value}" for key, value in extra.items())
    logger.info(
        f"  {name:<10} {detail:<16} @1={overall['page@1']:.3f}  @5={overall['page@5']:.3f}  "
        f"@10={overall['page@10']:.3f}  mrr={overall['mrr']:.3f}  doc@1={overall['doc@1']:.3f}"
    )
    for bucket, values in metrics.items():
        rows.append({"scheme": name, **extra, "queries": bucket,
                     **{k: round(v, 4) for k, v in values.items()}})


@click.command()
@click.option("--chunks", type=click.Path(exists=True, file_okay=False, path_type=Path),
              **in_corpus("data/chunks"),
              help="Chunk files behind the chunk index.")
@click.option("--chunk-vectors", type=click.Path(exists=True, file_okay=False, path_type=Path),
              **in_corpus("data/vectors/title-fp16"),
              help="Vectors for those chunks.")
@click.option("--page-chunks", type=click.Path(exists=True, file_okay=False, path_type=Path),
              **in_corpus("data/chunks-page"),
              help="Chunk files from chunk.py --page-chunks.")
@click.option("--page-vectors", type=click.Path(exists=True, file_okay=False, path_type=Path),
              **in_corpus("data/vectors/page-title-fp16"),
              help="Vectors for the page index.")
@click.option("--dims", type=int, default=1024, show_default=True,
              help="Matryoshka width both indexes are narrowed to.")
@click.option("--quant", type=click.Choice(["binary", "int8", "fp32"]), default="binary",
              show_default=True,
              help="Storage scheme for BOTH indexes. Defaults to what the site "
                   "ships, because a scheme that only wins at higher precision "
                   "wins nowhere a reader can see.")
@click.option("--alphas", default="1.0,0.85,0.7,0.5,0.3,0.0", show_default=True,
              help="Chunk weights to sweep for the blend. 1.0 is the shipped scheme.")
@click.option("--weights", default="0.85/0.10/0.05,0.85/0.05/0.10,0.80/0.10/0.10,0.90/0.05/0.05",
              show_default=True,
              help="chunk/page/previous-page weight triples to sweep, on top of the "
                   "--alphas sweep. The shipped scheme is 0.85/0.15/0 and appears "
                   "there. Pass \"\" to skip this part.")
@click.option("--maxpool/--no-maxpool", default=True, show_default=True,
              help="Also score the page term as max(this page, previous page) at "
                   "every --alphas weight, instead of summing the two separately.")
@click.option("--top-pages", default="3,5,10,20", show_default=True,
              help="Page cut-offs to sweep for the two-stage scheme.")
@click.option("--queries", "queries_path", type=click.Path(exists=True, dir_okay=False,
              path_type=Path), **in_corpus("data/EVAL_QUERIES.tsv"))
@click.option("--evaluate", "evaluate_path", type=click.Path(path_type=Path),
              default=Path("scripts/evaluate.py"), show_default=True,
              help="Where the metrics live. Imported, never copied.")
@click.option("--onnx-embed", "onnx_embed_path", type=click.Path(path_type=Path),
              default=Path("../justelesRCP/src/onnx_embed.py"), show_default=True)
@click.option("--model-dir", type=click.Path(path_type=Path),
              default=Path("../justelesRCP/models/jinaai/jina-embeddings-v5-text-small-retrieval"),
              show_default=True)
@click.option("--out", type=click.Path(path_type=Path),
              **in_corpus("data/EVAL_RESULTS_pagefirst.tsv"))
def main(chunks: Path, chunk_vectors: Path, page_chunks: Path, page_vectors: Path,
         dims: int, quant: str, alphas: str, weights: str, maxpool: bool,
         top_pages: str, queries_path: Path,
         evaluate_path: Path, onnx_embed_path: Path, model_dir: Path, out: Path) -> None:
    """Score chunks with, and against, the pages they sit on."""
    evaluate = load_module(evaluate_path, "evaluate")

    queries = evaluate.read_queries(queries_path)
    chunk_corpus = evaluate.load_corpus(chunks, chunk_vectors)
    page_corpus = evaluate.load_corpus(page_chunks, page_vectors)
    logger.info(f"{len(queries)} queries, {len(chunk_corpus.files)} chunks, "
                f"{len(page_corpus.files)} pages, {dims} dims, {quant}")

    # The same construction evaluate.py uses, for the same reason: queries are
    # embedded at FULL width and narrowed alongside the corpus, so both sides are
    # always truncated identically.
    onnx_embed = evaluate.load_encoder_module(onnx_embed_path)
    encoder = onnx_embed.Encoder(
        model_dir=model_dir, model_name=onnx_embed.RUNTIME_MODEL,
        intra_threads=max(1, (os.cpu_count() or 2) // 2), out_dim=0, query_cache=0,
    )
    query_matrix = encoder.encode([q["query"] for q in queries],
                                  prefix=encoder.query_prefix, batch_size=16)
    query_vectors = evaluate.narrow(np.asarray(query_matrix, dtype=np.float32), dims)

    # Both indexes stored the same way, and by default the way the site stores
    # them. `binarise` returns sign vectors renormalised, so a dot product over
    # them ranks the way the client's bit index does.
    store = {"binary": evaluate.binarise, "int8": evaluate.quantise_int8,
             "fp32": lambda m: m}[quant]
    narrow_chunks = evaluate.Corpus(
        vectors=store(evaluate.narrow(chunk_corpus.vectors, dims)),
        files=chunk_corpus.files, pages=chunk_corpus.pages)
    narrow_pages = evaluate.Corpus(
        vectors=store(evaluate.narrow(page_corpus.vectors, dims)),
        files=page_corpus.files, pages=page_corpus.pages)

    chunk_similarity = query_vectors @ narrow_chunks.vectors.T
    page_similarity = query_vectors @ narrow_pages.vectors.T
    mapped = chunk_to_pages(narrow_chunks, page_rows(narrow_pages))
    lifted = page_scores_for_chunks(page_similarity, mapped)

    previous = page_scores_for_chunks(page_similarity, previous_page_rows(narrow_chunks,
                                                                          page_rows(narrow_pages)))
    # A chunk on page 1 has no previous page and scores -1 there, which would drag its
    # blended score below every other chunk's for a term it cannot have. Falling back
    # to the chunk's own page makes the term read as "the context of this passage",
    # which the first page of a document is its own.
    previous = np.where(previous < 0, lifted, previous)

    rows: list[dict[str, Any]] = []

    logger.info("the page index on its own, ranked as pages")
    report("pages", evaluate.score(narrow_pages, query_vectors, queries, top_k=TOP_K),
           rows, alpha="", k="")

    logger.info("blending the page's score into the chunk's")
    for alpha in [float(a) for a in alphas.split(",")]:
        blended = alpha * chunk_similarity + (1.0 - alpha) * lifted
        report("blend" if alpha < 1.0 else "chunks",
               evaluate.score(narrow_chunks, query_vectors, queries, top_k=TOP_K,
                              similarity=blended),
               rows, alpha=alpha, k="")

    if weights.strip():
        logger.info("adding the previous page as a third term")
        for spec in [w for w in weights.split(",") if w.strip()]:
            chunk_w, page_w, prev_w = (float(part) for part in spec.split("/"))
            blended = chunk_w * chunk_similarity + page_w * lifted + prev_w * previous
            report("prev-page",
                   evaluate.score(narrow_chunks, query_vectors, queries, top_k=TOP_K,
                                  similarity=blended),
                   rows, alpha=spec, k="")

    if maxpool:
        logger.info("the page term as max(this page, the previous one)")
        pooled = np.maximum(lifted, previous)
        for alpha in [float(a) for a in alphas.split(",") if float(a) < 1.0]:
            blended = alpha * chunk_similarity + (1.0 - alpha) * pooled
            report("maxpool",
                   evaluate.score(narrow_chunks, query_vectors, queries, top_k=TOP_K,
                                  similarity=blended),
                   rows, alpha=alpha, k="")

    logger.info("pages first, then the chunks that live on them")
    for k in [int(v) for v in top_pages.split(",")]:
        # The top K pages per query, then every chunk not on one of them removed
        # from the running. -inf rather than a penalty: this is the strict reading
        # of the idea, where the page stage is a filter and not a hint.
        depth = min(k, page_similarity.shape[1])
        keep = np.argpartition(-page_similarity, kth=depth - 1, axis=1)[:, :depth]
        allowed = np.zeros_like(page_similarity, dtype=bool)
        np.put_along_axis(allowed, keep, True, axis=1)
        padded = np.concatenate([allowed, np.zeros((allowed.shape[0], 1), dtype=bool)], axis=1)
        on_a_kept_page = padded[:, mapped].any(axis=2)
        staged = np.where(on_a_kept_page, chunk_similarity, -np.inf)
        report("two-stage", evaluate.score(narrow_chunks, query_vectors, queries,
                                           top_k=TOP_K, similarity=staged),
               rows, alpha="", k=k)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    logger.success(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
