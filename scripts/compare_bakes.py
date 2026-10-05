# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "loguru", "numpy"]
# ///
"""Measure how far two bakes of the same passages disagree.

This exists to answer one specific question, and the answer was not the one that
was expected.

The VPS embeds *queries* with `model_int8.onnx`: that is what the shared `embed`
container runs, and it cannot change, because justelesRCP's index was baked
against it (see DESIGN.md, "fp16 weights bake 50x faster"). The
int8 graph has no CUDA kernels for `MatMulInteger`, so baking passages with it
runs on CPU at 2 chunks/s, 2.5 hours for the corpus. The fp16 graph runs on the
GPU at 140 chunks/s, 3 minutes, which is the difference between a sweep being
affordable and not. The temptation is therefore to bake passages with fp16 and
embed queries with int8, and assume the two weightings are interchangeable.

They are not. On the first 16 documents, the same passage embedded both ways has
a median cosine of 0.966, not the 0.999 that "interchangeable" would mean, and
the top 10 nearest neighbours churn by about 10% when only the corpus side
switches. Self-retrieval survives (top-1 agrees 99.8% of the time) because a
chunk's own vector is far closer than any other, but that is the easy case and
not what a user's query does.

Vector agreement is the sharper instrument here. The retrieval metrics in
`evaluate.py` run over 117 queries, so their standard error is around 0.045:
they cannot resolve a small difference, and a null result there would be
uninformative rather than reassuring. This script can see a difference that the
retrieval metric would miss. Both are reported, because "the vectors moved" and
"the answers got worse" are different claims and only the second one matters for
shipping.

Identity of the compared rows is NOT checked through `src_hash`. `embed.py`
folds the weights filename into the hash on purpose, so that switching weights
invalidates the cache; two bakes with different weights therefore never agree
there, by construction. What makes row `i` the same chunk in both files is that
both bakes read the same `data/chunks` at the same `--variant`, which is checked,
plus equal row counts, which is checked.

Written by Claude Code.
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
from loguru import logger

from lib.evalbake import narrow
from lib.corpus import in_corpus


def load_pair(a_dir: Path, b_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Stack the vectors the two bakes have in common, in the same row order.

    Parameters
    ----------
    a_dir, b_dir
        Per-variant vector directories written by `embed.py`, each holding one
        `.npz` per document.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, list[str]]
        The two stacked matrices and the documents they came from.

    Raises
    ------
    click.ClickException
        If the two bakes share no document, or a shared document was baked at a
        different `--variant`. A variant mismatch means the two files embed
        different *text* (title-prefixed versus plain), so any disagreement
        measured would be that, not the weights.
    """
    shared = sorted({p.name for p in a_dir.glob("*.npz")} & {p.name for p in b_dir.glob("*.npz")})
    if not shared:
        raise click.ClickException(
            f"{a_dir} and {b_dir} have no document in common; nothing to compare."
        )

    left, right, used = [], [], []
    for name in shared:
        a, b = np.load(a_dir / name), np.load(b_dir / name)
        if str(a["variant"]) != str(b["variant"]):
            raise click.ClickException(
                f"{name} was baked as {a['variant']} in {a_dir} but {b['variant']} "
                f"in {b_dir}. Those embed different text, so the comparison would "
                "measure the variant, not the weights."
            )
        va, vb = a["vectors"].astype(np.float32), b["vectors"].astype(np.float32)
        if va.shape != vb.shape:
            # One side was baked against an older data/chunks. Skipping is right:
            # aligning row i to row i would silently compare unrelated passages.
            logger.warning(f"skipping {name}: {va.shape} versus {vb.shape}, one bake is stale")
            continue
        left.append(va)
        right.append(vb)
        used.append(name)
    if not left:
        raise click.ClickException("every shared document had mismatched shapes; re-bake one side.")
    return np.concatenate(left), np.concatenate(right), used


def agreement(a: np.ndarray, b: np.ndarray, dim: int) -> dict[str, float]:
    """Per-chunk cosine between the two bakes at one width.

    Parameters
    ----------
    a, b
        Aligned, unit-norm matrices of the same shape.
    dim
        Width to narrow to before comparing, `0` for full width.

    Returns
    -------
    dict[str, float]
        `min`, `p1`, `median` and `mean` of the per-chunk cosine. The 1st
        percentile is reported alongside the median because the mean hides the
        tail, and it is the tail that reorders a result list.
    """
    x, y = narrow(a, dim), narrow(b, dim)
    cos = np.sum(x * y, axis=1)
    return {
        "min": float(cos.min()),
        "p1": float(np.percentile(cos, 1)),
        "median": float(np.median(cos)),
        "mean": float(cos.mean()),
    }


def neighbour_churn(a: np.ndarray, b: np.ndarray, dim: int, *, probes: int,
                    top_k: int, seed: int) -> dict[str, float]:
    """How much a result list changes when only the corpus side switches weights.

    Parameters
    ----------
    a
        The matched corpus: the same weights the query side uses.
    b
        The mismatched corpus, baked with the other weights.
    dim
        Width to narrow to before searching.
    probes
        How many chunks to use as stand-in queries. Sampled rather than
        exhaustive because the full matrix is 27k x 27k.
    top_k
        Depth of the compared result list.
    seed
        Fixes the sample so a re-run of the same bakes reports the same number.

    Returns
    -------
    dict[str, float]
        `same_top1` and `topk_overlap`, both fractions.

    Notes
    -----
    Chunks are used as queries here, which is a proxy: a real query is shorter
    and carries a different prefix. `evaluate.py` is the real measurement. The
    value of this one is that it has thousands of samples instead of 117, so it
    can detect a difference rather than merely fail to.
    """
    x, y = narrow(a, dim), narrow(b, dim)
    rng = np.random.default_rng(seed)
    probe = rng.choice(x.shape[0], size=min(probes, x.shape[0]), replace=False)
    matched = np.argsort(-(x[probe] @ x.T), axis=1)[:, :top_k]
    crossed = np.argsort(-(x[probe] @ y.T), axis=1)[:, :top_k]
    overlap = [len(set(m) & set(c)) / top_k for m, c in zip(matched, crossed)]
    return {
        "same_top1": float((matched[:, 0] == crossed[:, 0]).mean()),
        f"top{top_k}_overlap": float(np.mean(overlap)),
    }


@click.command()
@click.option("--vectors", type=click.Path(exists=True, file_okay=False, path_type=Path),
              **in_corpus("data/vectors"),
              help="Root holding the per-variant bake directories.")
@click.option("--a", "a_name", default="plain", show_default=True,
              help="Bake treated as MATCHED: same weights the query side uses.")
@click.option("--b", "b_name", default="plain-fp16", show_default=True,
              help="Bake treated as MISMATCHED, the one under suspicion.")
@click.option("--dims", default="1024,256", show_default=True,
              help="Comma-separated widths to compare at. The shipped width is "
                   "what matters; the others say whether truncation hides or "
                   "amplifies the disagreement.")
@click.option("--probes", default=500, show_default=True,
              help="Chunks sampled as stand-in queries for the churn measure.")
@click.option("--top-k", default=10, show_default=True,
              help="Depth of the compared result list.")
@click.option("--seed", default=0, show_default=True, help="Fixes the probe sample.")
def main(vectors: Path, a_name: str, b_name: str, dims: str, probes: int,
         top_k: int, seed: int) -> None:
    """Report how far two bakes of the same passages disagree."""
    a, b, used = load_pair(vectors / a_name, vectors / b_name)
    logger.info(f"{len(used)} documents, {a.shape[0]} chunks, {a.shape[1]} dims: "
                f"{a_name} (matched) versus {b_name} (mismatched)")
    if len(used) < 114:
        logger.warning(f"only {len(used)} documents compared; one bake is incomplete, "
                       "so these numbers are provisional")

    for dim in [int(d) for d in dims.split(",")]:
        stats = agreement(a, b, dim)
        churn = neighbour_churn(a, b, dim, probes=probes, top_k=top_k, seed=seed)
        logger.info(
            f"{dim or a.shape[1]} dims: cosine min={stats['min']:.4f} "
            f"p1={stats['p1']:.4f} median={stats['median']:.4f} mean={stats['mean']:.4f} | "
            f"same top-1 {churn['same_top1']:.3f}, "
            f"top-{top_k} overlap {churn[f'top{top_k}_overlap']:.3f}"
        )


if __name__ == "__main__":
    main()
