# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "loguru", "numpy", "onnxruntime", "tokenizers"]
# ///
"""Measure retrieval quality against the eval set, exactly, before choosing anything.

Every design question left open about search (how wide a vector to ship, whether
int8 costs accuracy, whether prepending the document title helps, what chunk size
to use) is empirical. This script answers them with one number each, on the same
117 queries, so the answers are comparable.

Ground truth is `(file, page)`, never a chunk id. A retrieved chunk counts as a
hit when it comes from the gold document AND covers the gold page. That is
deliberate: the eval set survives every chunker change, so a chunk-size sweep
compares configurations rather than comparing each configuration against its own
chunk boundaries. It also matches what the reader actually gets, which is a page
with a highlight on it, not a chunk.

Search here is EXACT brute force: 27k vectors by 1024 dims is a 110 MB matmul,
instant in numpy. The index backend is deliberately deferred (see DESIGN.md), and
an exact baseline is what any approximate backend must later be measured against.

Two known biases, restated from eval_queries.py because they bound how these
numbers may be read:

- Precise queries were written while looking at their passage, so they share its
  vocabulary. That inflates absolute recall. It does not invalidate a comparison
  between configurations, which is what this script is for.
- An agency may publish the same guideline as a summary, a full text and a
  background report. A query whose answer sits in all three scores a miss when the top
  hit is the sibling rather than the sampled document. Document-level recall is
  reported separately for that reason: it is the weaker metric, and the gap
  between the two is mostly duplicate renditions.

Written by Claude Code.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import numpy as np
from loguru import logger

# The storage schemes and the hit definition are shared with every other Python
# evaluator; re-exported under this module's name because eval_pagefirst.py reads
# them as `evaluate.narrow`, `evaluate.binarise` and so on.
from lib.evalbake import (acceptable, binarise, by_kind, load_bake, narrow,  # noqa: F401
                          quantise_int4, quantise_int8)
from lib.corpus import in_corpus

# Ranks reported. 10 is the practical ceiling: a reader scanning a result list
# does not go far past the first screen, so recall beyond that is academic.
CUTOFFS = (1, 5, 10)


@dataclass
class Corpus:
    """Flattened chunk index, aligned row-for-row with the vector matrix.

    Attributes
    ----------
    vectors
        float32 matrix, one row per chunk, L2-normalised at full model width.
    files
        Source filename of each row.
    pages
        Pages each row's text came from, 1-based, as stored by chunk.py.
    """

    vectors: np.ndarray
    files: list[str]
    pages: list[frozenset[int]]


def load_encoder_module(path: Path) -> Any:
    """Import the sibling repo's `onnx_embed` module from an explicit path.

    Parameters
    ----------
    path
        Path to `onnx_embed.py` in the justelesRCP checkout.

    Returns
    -------
    Any
        The imported module.

    Raises
    ------
    click.ClickException
        If the file is absent.
    """
    if not path.is_file():
        raise click.ClickException(
            f"{path} not found; pass --onnx-embed to point at the sibling checkout."
        )
    spec = importlib.util.spec_from_file_location("onnx_embed", path)
    if spec is None or spec.loader is None:
        raise click.ClickException(f"cannot import {path} as a module")
    module = importlib.util.module_from_spec(spec)
    sys.modules["onnx_embed"] = module
    spec.loader.exec_module(module)
    return module


def load_corpus(chunks: Path, vectors: Path) -> Corpus:
    """Join the chunk files with their vectors into one flat index.

    Parameters
    ----------
    chunks
        Directory of chunk files from chunk.py.
    vectors
        Directory of `.npz` files from embed.py, for one variant.

    Returns
    -------
    Corpus
        Vectors and the provenance of each row.

    Raises
    ------
    click.ClickException
        If `lib.evalbake.load_bake` refuses the pair: a document with no vectors,
        or with vectors and chunks in different numbers.
    """
    try:
        matrix, owner, pages = load_bake(chunks=chunks, vectors=vectors)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    return Corpus(vectors=matrix, files=owner.tolist(), pages=[frozenset(p) for p in pages])


def read_queries(path: Path) -> list[dict[str, str]]:
    """Read the eval queries TSV.

    Parameters
    ----------
    path
        `data/EVAL_QUERIES.tsv`.

    Returns
    -------
    list[dict[str, str]]
        One row per query, keyed by column name.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def score(corpus: Corpus, query_vectors: np.ndarray, queries: list[dict[str, str]],
          *, top_k: int, rescore_vectors: np.ndarray | None = None,
          shortlist: int = 200, within_document: bool = False,
          similarity: np.ndarray | None = None) -> dict[str, Any]:
    """Rank every query against the whole corpus and summarise the hits.

    Parameters
    ----------
    corpus
        Flat index, already narrowed and optionally quantised.
    query_vectors
        One normalised row per query, same width as the corpus.
    queries
        Query rows, carrying `gold_file`, `gold_page` and `type`.
    top_k
        How deep to look. Must be at least `max(CUTOFFS)`.
    rescore_vectors
        Optional second index, same rows as `corpus.vectors` but at a higher
        precision. When given, `corpus.vectors` only picks a shortlist and these
        decide the final order. This is the cheap-index-plus-rescore pattern: the
        small index is scanned over the whole corpus, and the expensive one is
        touched for `shortlist` rows per query. Bytes on the wire are the sum of
        both, so it only pays when the cheap index is very small.
    shortlist
        How many candidates the cheap index forwards to the rescorer. Ignored
        without `rescore_vectors`.
    similarity
        Pre-computed (queries, chunks) score matrix, replacing the dot product
        this function would otherwise take. It exists so that an experiment
        scoring chunks by something other than their own cosine (see
        `eval_pagefirst.py`, which blends in the score of the page a chunk sits
        on) reuses the ranking and the metrics rather than reimplementing them,
        which is how two harnesses end up disagreeing about what recall@5 means.

    Returns
    -------
    dict[str, Any]
        Overall and per-type metrics: page-level recall at each cutoff, document
        level recall, and MRR over the top `top_k`.
    """
    # Cosine on normalised vectors is a dot product, so one matmul ranks
    # everything. 117 x 27309 floats is 12 MB, no need to chunk the computation.
    if similarity is None:
        similarity = query_vectors @ corpus.vectors.T
    else:
        # Copied because the masks below write into it, and the caller keeps its
        # matrix to try other weightings with.
        similarity = similarity.copy()
    if within_document:
        # The SIBLING's regime, not this site's: justelesRCP searches inside one
        # drug's RCP, never across a corpus. Restricting each query to its gold
        # document turns this harness into "given the right document, does it find
        # the right page", which is the only measurement here that says anything
        # about a format change over there. The candidate pool goes from 27,309
        # chunks to a few dozen, and a scheme can behave differently when the
        # candidates are homogeneous, which is exactly what needs checking rather
        # than assuming.
        by_file: dict[str, np.ndarray] = {}
        for row, query in enumerate(queries):
            gold = query["gold_file"]
            if gold not in by_file:
                by_file[gold] = np.array([f != gold for f in corpus.files])
            similarity[row, by_file[gold]] = -np.inf
    # argpartition then sort only the top slice: sorting 27309 scores per query
    # costs more than it returns when only 10 ranks are read.
    depth = min(shortlist if rescore_vectors is not None else top_k, similarity.shape[1])
    top = np.argpartition(-similarity, kth=depth - 1, axis=1)[:, :depth]
    if rescore_vectors is None:
        order = np.argsort(-np.take_along_axis(similarity, top, 1), axis=1)
    else:
        # Dot each query against only ITS OWN shortlist, which is what a rescoring
        # client would fetch. The gather is (queries, shortlist, dim) and is the
        # reason shortlist is not simply the whole corpus.
        refined = np.einsum("qd,qkd->qk", query_vectors, rescore_vectors[top])
        order = np.argsort(-refined, axis=1)
    ordered = np.take_along_axis(top, order, 1)[:, :top_k]

    # A hit is any (file, page) the query accepts, `gold_alt` included: four
    # editions of one book means the right answer can sit in the edition that was
    # not sampled, and the other evaluators already score that as a hit.
    page_ranks: list[int] = []
    doc_ranks: list[int] = []
    for row, query in enumerate(queries):
        wanted = acceptable(query)
        files = {name for name, _ in wanted}
        page_rank = doc_rank = 0
        for rank, index in enumerate(ordered[row], start=1):
            name = corpus.files[index]
            if not doc_rank and name in files:
                doc_rank = rank
            if not page_rank and any(name == f and p in corpus.pages[index] for f, p in wanted):
                page_rank = rank
            if page_rank and doc_rank:
                break
        page_ranks.append(page_rank)
        doc_ranks.append(doc_rank)

    # `by_kind` is the shared summary; the keys are renamed to this script's
    # historical `page@k` columns so EVAL_RESULTS*.tsv stay comparable over time.
    pages_summary = by_kind(page_ranks, queries)
    docs_summary = by_kind(doc_ranks, queries)
    return {kind: {"n": float(values["n"]),
                   **{f"page@{c}": values[f"page_{c}"] for c in CUTOFFS},
                   f"doc@{CUTOFFS[0]}": docs_summary[kind][f"page_{CUTOFFS[0]}"],
                   "mrr": values["mrr"]}
            for kind, values in pages_summary.items()}


@click.command()
@click.option("--chunks", type=click.Path(exists=True, file_okay=False, path_type=Path),
              **in_corpus("data/chunks"))
@click.option("--vectors", type=click.Path(exists=True, file_okay=False, path_type=Path),
              **in_corpus("data/vectors"),
              help="Root holding one directory per embedding variant.")
@click.option("--variant", "variants", multiple=True, default=("plain",), show_default=True,
              help="Variants to evaluate. Repeat the flag to compare several.")
@click.option("--dims", default="1024,512,256,128", show_default=True,
              help="Comma-separated Matryoshka widths to sweep.")
@click.option("--queries", "queries_path", type=click.Path(exists=True, dir_okay=False,
              path_type=Path), **in_corpus("data/EVAL_QUERIES.tsv"))
@click.option("--onnx-embed", "onnx_embed_path", type=click.Path(path_type=Path),
              default=Path("../justelesRCP/src/onnx_embed.py"), show_default=True)
@click.option("--model-dir", type=click.Path(path_type=Path),
              default=Path("../justelesRCP/models/jinaai/jina-embeddings-v5-text-small-retrieval"),
              show_default=True)
@click.option("--quant", default="fp32,int8", show_default=True,
              help="Comma-separated storage schemes to sweep: fp32, int8, int4, "
                   "binary, binary+int8. The last ships a bit index AND an int8 "
                   "one, using the first to shortlist for the second.")
@click.option("--query-prefix", default="", show_default=True,
              help="Text prepended to every query BEFORE the model's own \"query: \" "
                   "prefix is applied, to test whether telling the encoder what kind "
                   "of question this is helps it. The corpus side is untouched, so a "
                   "sweep needs no re-bake.")
@click.option("--shortlist", type=int, default=200, show_default=True,
              help="Candidates the cheap index forwards to the rescorer. Only "
                   "used by the binary+int8 scheme.")
@click.option("--within-document", is_flag=True, default=False,
              help="Rank only inside each query's gold document, which is the "
                   "sibling justelesRCP's regime (search within one drug's RCP). "
                   "Use it to check whether an index format decided here also "
                   "holds there, where the candidates are few and homogeneous.")
@click.option("--out", type=click.Path(path_type=Path), **in_corpus("data/EVAL_RESULTS.tsv"), help="Where the result table is written.")
def main(chunks: Path, vectors: Path, variants: tuple[str, ...], dims: str,
         queries_path: Path, onnx_embed_path: Path, model_dir: Path, quant: str,
         query_prefix: str,
         shortlist: int, within_document: bool, out: Path) -> None:
    """Sweep vector width and storage scheme, and report retrieval metrics."""
    widths = [int(d) for d in dims.split(",") if d.strip()]
    schemes = [s.strip() for s in quant.split(",") if s.strip()]
    known = {"fp32", "int8", "int4", "binary", "binary+int8"}
    if set(schemes) - known:
        raise click.ClickException(
            f"unknown --quant {sorted(set(schemes) - known)}, pick from {sorted(known)}")
    queries = read_queries(queries_path)
    logger.info(f"{len(queries)} queries, widths {widths}, variants {list(variants)}")

    onnx_embed = load_encoder_module(onnx_embed_path)
    # Queries are embedded at FULL width here and narrowed alongside the corpus,
    # so both sides are always truncated identically. Embedding the query at 256
    # while the corpus sits at 1024 is the mistake this avoids by construction.
    encoder = onnx_embed.Encoder(
        model_dir=model_dir, model_name=onnx_embed.RUNTIME_MODEL,
        # Half the logical CPUs approximates one thread per physical core, which
        # embed.py measured as 3.3x faster than one per hardware thread for this
        # int8 graph (see its physical_cores docstring). Not worth importing the
        # exact version here: this encodes 117 queries, not 27309 passages.
        intra_threads=max(1, (os.cpu_count() or 2) // 2), out_dim=0, query_cache=0,
    )
    # The reader's prefix goes inside the model's: the encoder's own query marker
    # ("Query: " for jina-embeddings-v5, "query: " for arctic before it) has to stay
    # first, and what follows it is what the reader would have typed if the site
    # typed it for them.
    if query_prefix:
        logger.info(f"query prefix: {query_prefix!r}")
    query_matrix = encoder.encode([query_prefix + q["query"] for q in queries],
                                  prefix=encoder.query_prefix, batch_size=16)
    logger.info(f"queries embedded at {query_matrix.shape[1]} dims")

    rows: list[dict[str, Any]] = []
    for variant in variants:
        corpus = load_corpus(chunks, vectors / variant)
        logger.info(f"variant {variant}: {len(corpus.files)} chunks "
                    f"at {corpus.vectors.shape[1]} dims")
        for width in widths:
            narrowed = narrow(corpus.vectors, width)
            narrow_queries = narrow(query_matrix, width)
            for quantisation in schemes:
                # Bytes per stored vector, so the table carries the cost next to
                # the benefit. This is the number that decides the format: the
                # whole index is downloaded by every visitor before the first
                # search, so it is bandwidth, not disk.
                per_vector = {"fp32": width * 4, "int8": width, "int4": width // 2,
                              "binary": width // 8}
                if quantisation == "binary+int8":
                    # Both indexes ship: the bit index is scanned, the int8 one is
                    # read for the shortlist only. Sum, not max.
                    cost = width // 8 + width
                    indexed, rescore = binarise(narrowed), quantise_int8(narrowed)
                else:
                    cost = per_vector[quantisation]
                    rescore = None
                    indexed = {"fp32": lambda m: m, "int8": quantise_int8,
                               "int4": quantise_int4,
                               "binary": binarise}[quantisation](narrowed)
                metrics = score(Corpus(indexed, corpus.files, corpus.pages),
                                narrow_queries, queries, top_k=max(CUTOFFS),
                                rescore_vectors=rescore, shortlist=shortlist,
                                within_document=within_document)
                for bucket, values in metrics.items():
                    rows.append({"variant": variant, "dims": width,
                                 "quant": quantisation, "bytes_per_vec": cost,
                                 "queries": bucket, **values})

    # Print the headline comparison first: the whole query set, one line per
    # configuration, which is what a decision is actually made from.
    logger.info("page-level recall on all 117 queries")
    for row in rows:
        if row["queries"] != "all":
            continue
        logger.info(f"  {row['variant']:>5} {row['dims']:>5}d {row['quant']:>5}  "
                    + "  ".join(f"@{c}={row[f'page@{c}']:.3f}" for c in CUTOFFS)
                    + f"  mrr={row['mrr']:.3f}  doc@1={row[f'doc@{CUTOFFS[0]}']:.3f}")

    fields = ["variant", "dims", "quant", "bytes_per_vec", "queries", "n",
              *[f"page@{c}" for c in CUTOFFS], f"doc@{CUTOFFS[0]}", "mrr"]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) and k != "n"
                                 else int(v) if k == "n" else v)
                             for k, v in row.items()})
    logger.success(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
