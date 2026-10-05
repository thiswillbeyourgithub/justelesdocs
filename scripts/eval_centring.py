# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "httpx", "loguru", "numpy"]  # httpx: used by lib.evalbake
# ///
"""Does subtracting the corpus mean before binarising buy anything? (No.)

The question this answers, asked on 2026-09-17: this corpus is one speciality, so
every vector in it shares a large common component and only a thin shell of the
sphere is ever used. A one-bit index stores the SIGN of each dimension, so a
dimension whose sign is the same across most of the corpus stores almost nothing.
Would centring the vectors (or going further: PCA, whitening) make the index
carry more?

The bit-budget half of that is true and measurable. On the 483-document bake the
mean vector has norm 0.53, two unrelated chunks already score a cosine of 0.27,
49 dimensions of 1024 have the same sign more than 90% of the time, and the
shipped index uses 880 of its 1024 bits. Centring takes every dimension to a
near-perfect 50/50 split and recovers all 1024.

The retrieval half is false. Nothing in the ranking improves, and the cleanest
comparison (float passages, no quantisation at all) says centring is slightly
WORSE. The reason is in the last two rows of the table: binarising the passages
costs about 0.017 MRR against float, and that is the entire prize. Any scheme
that rearranges the bits, centring, a random rotation, ITQ, PCA, is competing for
0.017 MRR on a metric whose standard error over 117 queries is 0.045. The bits
that looked wasted were not: a dimension that is positive 84% of the time still
says something every time it is negative, and after centring its sign sits near
zero where the model's own noise decides it.

This script is the measurement, kept so the answer can be re-checked rather than
remembered. It is not in the build chain and nothing imports it.

Usage, with the shared encoder running (see CLAUDE.md, Commands):

    uv run scripts/eval_centring.py --embed-url http://127.0.0.1:8461

Written by Claude Code (Opus 5).
"""

from __future__ import annotations

import csv
from pathlib import Path

import click
import numpy as np
from loguru import logger

from lib.evalbake import SCALE, embed_queries, load_bake, mean_reciprocal_rank
from lib.corpus import in_corpus


def bit_balance(matrix: np.ndarray) -> tuple[float, int]:
    """How much of a one-bit index's budget a matrix would actually use.

    Parameters
    ----------
    matrix
        Row-wise vectors. Only the sign of each component is looked at, which is
        all a binary index keeps.

    Returns
    -------
    usable_bits : float
        Summed binary entropy of the per-dimension sign, out of `dims`. A
        dimension whose sign never varies contributes 0, a 50/50 one contributes 1.
    lopsided : int
        Dimensions whose rarer sign occurs under 10% of the time.
    """
    positive = (matrix > 0).mean(axis=0)
    safe = np.clip(positive, 1e-12, 1 - 1e-12)
    entropy = -(safe * np.log2(safe) + (1 - safe) * np.log2(1 - safe))
    return float(entropy.sum()), int((np.minimum(positive, 1 - positive) < 0.1).sum())


@click.command()
@click.option("--chunks", type=click.Path(exists=True, file_okay=False, path_type=Path),
              **in_corpus("data/chunks"),
              help="Chunk files, for the page each chunk covers.")
@click.option("--vectors", type=click.Path(exists=True, file_okay=False, path_type=Path),
              **in_corpus("data/vectors/meta-gpu"),
              help="The bake to measure. Defaults to the shipped one.")
@click.option("--queries", "queries_path", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              **in_corpus("data/EVAL_QUERIES.tsv"),
              help="Eval set from `eval_queries.py`.")
@click.option("--embed-url", default="http://127.0.0.1:8461", show_default=True,
              help="The shared encoder. Nothing here works without it: the point "
                   "is to measure the real wire format.")
@click.option("--dims", default=1024, show_default=True,
              help="Index width to measure at.")
@click.option("--depth", default=50, show_default=True,
              help="Shortlist length, matching RESCORE_DEPTH in src/search.js.")
def main(chunks: Path, vectors: Path, queries_path: Path, embed_url: str,
         dims: int, depth: int) -> None:
    """Compare centred against plain, in float and in one bit, on the eval set."""
    matrix, owner, pages = load_bake(chunks=chunks, vectors=vectors)
    matrix = matrix[:, :dims]
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    logger.info(f"{matrix.shape[0]} chunks at {dims} dims from {vectors}")

    centroid = matrix.mean(axis=0)
    centred = matrix - centroid
    centred /= np.linalg.norm(centred, axis=1, keepdims=True)
    logger.info(f"corpus mean vector: norm {float(np.linalg.norm(centroid)):.4f}")
    for label, block in (("plain  ", matrix), ("centred", centred)):
        usable, lopsided = bit_balance(block)
        logger.info(f"{label}: {usable:.0f} of {dims} bits usable, "
                    f"{lopsided} dimensions lopsided past 90/10")

    # The one-bit index's implied passage vector: +/-1 per dimension. Scoring it
    # against the int8 query is exactly the client's asymmetric dot product, up to
    # a positive factor that cannot reorder anything.
    signs = np.where(matrix > 0, 1.0, -1.0).astype(np.float32)
    centred_signs = np.where(centred > 0, 1.0, -1.0).astype(np.float32)

    queries = list(csv.DictReader(queries_path.open(encoding="utf-8"), delimiter="\t"))
    raw = embed_queries(texts=[q["query"] for q in queries], embed_url=embed_url, dims=dims)
    plain_query = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    centred_query = raw - centroid
    centred_query /= np.linalg.norm(centred_query, axis=1, keepdims=True)
    # What the client would really send: the encoder's int8 comes back, the mean is
    # subtracted in the page, and the result has to be re-rounded to int8 to build
    # the lookup table. That second rounding is the price of centring client-side,
    # and it is measured rather than assumed because the encoder is shared with
    # justelesRCP and its wire format cannot move.
    requantised = np.clip(np.round(centred_query * SCALE), -SCALE, SCALE) / SCALE
    requantised /= np.linalg.norm(requantised, axis=1, keepdims=True)

    logger.info(f"{len(queries)} queries embedded through {embed_url}")
    click.echo("\npassages          query              MRR      gold in top "
               f"{depth}")
    for label, query_side, passage_side in (
        ("float             plain            ", plain_query, matrix),
        ("float             centred          ", centred_query, centred),
        ("1-bit             plain            ", plain_query, signs),
        ("1-bit             centred          ", centred_query, centred_signs),
        ("1-bit             centred, re-int8 ", requantised, centred_signs),
    ):
        mrr, found = mean_reciprocal_rank(scores=query_side @ passage_side.T, queries=queries,
                                          owner=owner, pages=pages, depth=depth)
        click.echo(f"{label} {mrr:.4f}   {found}/{len(queries)}")
    click.echo("\nStandard error over 117 queries is about 0.045: read differences, not decimals.")


if __name__ == "__main__":
    main()
