"""Read a bake, store it as an index would, embed queries, score a ranking.

This is the ONE definition of what the Python evaluators measure: how a bake is
read (and refused when it disagrees with its chunks), how a width is narrowed,
how a storage scheme is simulated, and what counts as a hit. `evaluate.py`,
`eval_pagefirst.py` (through it), `eval_centring.py`, `eval_language_axis.py`,
`compare_bakes.py` and `bench.py` all read it from here. Each used to carry its
own copy, and the copies drifted: `evaluate.py` ignored `gold_alt`, so a query
answered from another edition of the same book scored a miss there and a hit
everywhere else, and its binariser sent a zero component to +1 where the shipped
index (`build_index.py`, `packbits(cut > 0)`) sends it to 0.

This module is imported, not run. It assumes the importing script declared numpy
in its PEP 723 header, and httpx too if it calls `embed_queries` (imported there,
so a script that never embeds a query need not declare it); uv puts the script's
own directory on sys.path, so `from lib.evalbake import load_bake` works from
scripts/.

Written by Claude Code (Opus 5).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

#: The query wire format's fixed scale, matching build_index.py's INT8_SCALE and
#: src/search.js's SCALE. The encoder returns round(v * 127) clamped to +/-127.
SCALE = 127


def load_bake(*, chunks: Path, vectors: Path) -> tuple[np.ndarray, np.ndarray, list[set[int]]]:
    """Read every document's chunk vectors and the pages each chunk covers.

    Parameters
    ----------
    chunks
        Directory of `chunk.py` output, one JSON per document.
    vectors
        Directory of `embed.py` output, one `.npz` per document, paired by stem.

    Returns
    -------
    matrix : numpy.ndarray
        `n_chunks` x `dims` float32, unit-norm rows, documents in sorted filename
        order and each document's chunks in its own order. That is the same row
        order `build_index.py` writes, so a rank here is comparable to a rank there.
    owner : numpy.ndarray
        `n_chunks` strings: the document filename each row belongs to.
    pages : list of set of int
        Per row, the pages that chunk's text sits on.

    Raises
    ------
    ValueError
        If there are no chunk files, a document has no vectors, or its vectors and
        chunks differ in number. The last means one of the two is stale, and
        scoring would then attribute a vector to the wrong page, which is worse
        than failing.
    """
    blocks: list[np.ndarray] = []
    owner: list[str] = []
    pages: list[set[int]] = []
    for path in sorted(chunks.glob("*.json")):
        vector_path = vectors / f"{path.stem}.npz"
        if not vector_path.exists():
            raise ValueError(f"{path.name} has no vectors at {vector_path}; run embed.py first.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        matrix = np.asarray(np.load(vector_path)["vectors"], dtype=np.float32)
        if len(matrix) != len(payload["chunks"]):
            raise ValueError(
                f"{payload['file']}: {len(payload['chunks'])} chunks but {len(matrix)} "
                "vectors. One of the two is stale; re-run embed.py.")
        blocks.append(matrix)
        for chunk in payload["chunks"]:
            owner.append(payload["file"])
            pages.append(set(chunk.get("pages") or [chunk["page"]]))
    if not blocks:
        raise ValueError(f"no chunk files in {chunks}")
    matrix = np.concatenate(blocks)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix, np.array(owner), pages


def narrow(matrix: np.ndarray, dim: int) -> np.ndarray:
    """Truncate to `dim` dimensions and renormalise, the Matryoshka way.

    Parameters
    ----------
    matrix
        Vectors at full width.
    dim
        Target width, or 0 to keep the full width.

    Returns
    -------
    np.ndarray
        float32 vectors of width `dim`, L2-normalised.

    Notes
    -----
    Truncating a normalised vector then renormalising is identical to truncating
    the raw vector then normalising, because the first normalisation is a positive
    scalar. That identity is what lets embed.py store one full-width matrix and
    the evaluators sweep every width for free.
    """
    cut = matrix if not dim or dim >= matrix.shape[1] else matrix[:, :dim]
    norms = np.clip(np.linalg.norm(cut, axis=-1, keepdims=True), 1e-12, None)
    return (cut / norms).astype(np.float32)


def quantise_int8(matrix: np.ndarray) -> np.ndarray:
    """Round-trip vectors through per-vector symmetric int8, as a shipped index would.

    Parameters
    ----------
    matrix
        L2-normalised float32 vectors.

    Returns
    -------
    np.ndarray
        The same vectors after int8 quantisation and dequantisation, renormalised.

    Notes
    -----
    This measures the ACCURACY cost of int8, not its speed: the point is whether a
    4x smaller index retrieves the same documents. The scale is per vector rather
    than global because a global scale is set by the single largest component in
    the whole corpus and wastes most of the range on every other vector.
    """
    scale = np.clip(np.abs(matrix).max(axis=-1, keepdims=True), 1e-12, None) / 127.0
    codes = np.clip(np.rint(matrix / scale), -127, 127).astype(np.int8)
    return narrow(codes.astype(np.float32) * scale, 0)


def quantise_int4(matrix: np.ndarray) -> np.ndarray:
    """Round-trip vectors through per-vector symmetric int4, two codes per byte.

    Parameters
    ----------
    matrix
        L2-normalised float32 vectors.

    Returns
    -------
    np.ndarray
        The same vectors after int4 quantisation and dequantisation, renormalised.

    Notes
    -----
    Fifteen levels (-7..7) rather than sixteen: a symmetric range with a true zero
    costs one code and keeps the quantiser unbiased, which matters more here than
    the extra level because the error is what the dot product accumulates.

    Half the bytes of int8 at the same width. The interesting comparison is not
    int4 against int8 at one width, it is 1024 dims at int4 against 256 dims at
    int8, which cost the SAME bytes per vector and answer whether width or
    precision is the better place to spend them.
    """
    scale = np.clip(np.abs(matrix).max(axis=-1, keepdims=True), 1e-12, None) / 7.0
    codes = np.clip(np.rint(matrix / scale), -7, 7)
    return narrow(codes.astype(np.float32) * scale, 0)


def binarise(matrix: np.ndarray) -> np.ndarray:
    """Reduce each component to its sign, one bit per dimension, as build_index.py does.

    Parameters
    ----------
    matrix
        L2-normalised float32 vectors.

    Returns
    -------
    np.ndarray
        The same vectors as +/-1, renormalised, so a dot-product scoring path
        measures what the packed bit index retrieves.

    Notes
    -----
    A component of exactly 0 becomes -1, because the shipped index packs
    `cut > 0` and so stores a 0 bit for it. This used to be `>= 0`, which
    measured a slightly different index from the one that ships.

    Scoring stays ASYMMETRIC on purpose. The query keeps its full precision and
    only the stored side is binarised, which is free (the query is encoded per
    request, not stored) and is worth several points over binarising both sides.
    """
    # Every row has the same norm, sqrt(dim), so the renormalisation is a constant
    # rescale. It is applied anyway so the returned matrix obeys the same contract
    # as the other schemes and no caller compares unnormalised scores.
    return narrow(np.where(matrix > 0.0, 1.0, -1.0).astype(np.float32), 0)


def embed_queries(*, texts: list[str], embed_url: str, dims: int) -> np.ndarray:
    """Embed query texts through the REAL encoder service.

    The wire format quantises to int8, and that rounding is part of what is being
    measured: a float query would flatter every row of a comparison table equally
    but would not be what ships.

    Parameters
    ----------
    texts
        Query strings.
    embed_url
        Base URL of `../justelesRCP/src/embed-service.py`.
    dims
        Requested width.

    Returns
    -------
    numpy.ndarray
        `len(texts)` x `dims` float32, dequantised by `SCALE`, NOT normalised.
    """
    out = np.empty((len(texts), dims), dtype=np.float32)
    import httpx

    with httpx.Client(timeout=30) as client:
        for i, text in enumerate(texts):
            response = client.post(f"{embed_url}/api/sem/embed", json={"q": text, "dim": dims})
            response.raise_for_status()
            raw = base64.b64decode(response.json()["q"])
            out[i] = np.frombuffer(raw, dtype=np.int8).astype(np.float32) / SCALE
    return out


def acceptable(query: dict[str, str]) -> list[tuple[str, int]]:
    """Every (file, page) that answers this query, not only the labelled one.

    Parameters
    ----------
    query
        A row of `EVAL_QUERIES.tsv`. `gold_alt` is optional and usually empty.

    Returns
    -------
    list of (str, int)
        The gold pair first, then any alternative the question also has a correct
        answer on. Four editions of one book means a query can be answered from the
        wrong edition and be right, and scoring that as a miss would measure a
        retrieval failure that did not happen.

    Notes
    -----
    The cell is written by `eval_queries.py` as `file.pdf#12;other.pdf#7`. A
    malformed entry is skipped rather than raised on, because a partly readable
    label still measures the primary answer correctly, and the generator already
    refuses to write a bad one.

    `scripts/evaluate_rescore.mjs` carries the same six lines in JavaScript, under
    the same name. That is a duplicate, kept deliberately: the two evaluators run in
    different languages by design (the JS one imports the shipped `src/search.js` so
    that the tokeniser it measures is the one in the page), and the alternative is a
    third file format between the TSV and both readers. If the syntax of the cell
    ever changes, both copies change together.
    """
    pairs = [(query["gold_file"], int(query["gold_page"]))]
    for entry in (query.get("gold_alt") or "").split(";"):
        name, mark, page = entry.rpartition("#")
        if mark and page.strip().isdigit():
            pairs.append((name, int(page)))
    return pairs


def gold_ranks(*, scores: np.ndarray, queries: list[dict[str, str]], owner: np.ndarray,
               pages: list[set[int]], depth: int) -> list[int]:
    """Rank of the first chunk answering each query, or 0 if none does.

    Parameters
    ----------
    scores
        `n_queries` x `n_chunks`. Higher is better; only the order matters.
    queries
        Rows of `EVAL_QUERIES.tsv`, carrying `gold_file`, `gold_page` and optionally
        `gold_alt`, which names other documents answering the same question.
    owner, pages
        From `load_bake`.
    depth
        Shortlist length. A gold page outside it counts as a miss, as it does in the
        shipped client: it never looks past its candidate list.

    Returns
    -------
    list of int
        One 1-based rank per query, 0 where the gold page never appears.
    """
    ranks = []
    for i, query in enumerate(queries):
        row = scores[i]
        # argpartition needs a kth strictly inside the array. A real bake has tens of
        # thousands of chunks so depth is never near it, but a small index (one
        # document, or a test) would otherwise raise instead of ranking.
        kth = min(depth, row.shape[0] - 1)
        shortlist = np.argpartition(-row, kth)[:depth]
        shortlist = shortlist[np.argsort(-row[shortlist])]
        wanted = acceptable(query)
        hit = 0
        for rank, chunk in enumerate(shortlist, start=1):
            if any(owner[chunk] == name and page in pages[chunk] for name, page in wanted):
                hit = rank
                break
        ranks.append(hit)
    return ranks


def mean_reciprocal_rank(*, scores: np.ndarray, queries: list[dict[str, str]],
                         owner: np.ndarray, pages: list[set[int]],
                         depth: int) -> tuple[float, int]:
    """Mean 1/rank over queries, and how many found their gold page at all."""
    ranks = gold_ranks(scores=scores, queries=queries, owner=owner, pages=pages, depth=depth)
    return sum(1 / r for r in ranks if r) / len(ranks), sum(1 for r in ranks if r)


def by_kind(ranks: list[int], queries: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    """Group ranks by the eval set's `type` column.

    An average over every query hides the group most likely to break on its own:
    the crosslingual queries are 23 of 166, and a change that helps the other 143
    while quietly killing them reads as an improvement.

    Parameters
    ----------
    ranks
        As returned by `gold_ranks`, parallel to `queries`.
    queries
        Rows of `EVAL_QUERIES.tsv`.

    Returns
    -------
    dict
        Kind, plus "all", to {n, page_1, page_5, page_10, mrr}.
    """
    groups: dict[str, list[int]] = {"all": list(ranks)}
    for rank, query in zip(ranks, queries):
        groups.setdefault(query.get("type", "?"), []).append(rank)
    out = {}
    for kind, values in groups.items():
        n = len(values)
        out[kind] = {
            "n": n,
            "page_1": sum(1 for r in values if 0 < r <= 1) / n,
            "page_5": sum(1 for r in values if 0 < r <= 5) / n,
            "page_10": sum(1 for r in values if 0 < r <= 10) / n,
            "mrr": sum(1 / r for r in values if r) / n,
        }
    return out
