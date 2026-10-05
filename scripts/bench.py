# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "loguru", "numpy"]  # numpy: imported by lib.evalbake
# ///
"""Run the whole retrieval chain over the eval corpus only, so a parameter can be swept.

Measuring a chunking or baking idea used to mean rebuilding all 534 documents: about
seven minutes of chunking, a full GPU bake, then the evaluation. At that price a
sweep over four chunk sizes is an afternoon, so in practice ideas were argued about
instead of measured.

The eval set only ever asks about 47 documents. Everything else in the corpus is
there to be ranked BELOW them, which is a job a sample does as well as the whole
shelf. So this script chunks, bakes and indexes the gold documents plus a fixed
sample of distractors, and runs the shipped evaluation against that index:

    # one point, the current defaults
    EMBED_URL=http://127.0.0.1:8461 uv run scripts/bench.py

    # a sweep: every combination of the listed values, one row of results each
    EMBED_URL=http://127.0.0.1:8461 uv run scripts/bench.py \
        --target-tokens 200,300,450 --overlap-tokens 0,60

What the numbers mean, and do not mean
--------------------------------------
A 150-document index holds fewer wrong answers than a 537-document one, so page@1
here is HIGHER than the shipped number and is not comparable to it. Only differences
between two points of the same sweep are: the subset is fixed by `--distractors` and
`--seed`, drawn from the tracked data/MANIFEST.tsv, so two runs a week apart cover
the same documents unless the corpus itself changed.

What is reused, and what is not
-------------------------------
chunk.py hashes its parameters into each document's cache entry, so a grid point
whose chunking is unchanged costs nothing the second time; embed.py does the same
against the chunk file's hash. That is what makes a grid affordable after the first
point. The encoder is NOT started here: it is a long-lived service in the sibling
justelesRCP checkout, and bench.py refuses to start rather than spend ten minutes
chunking for an evaluation that cannot run.

    cd ../justelesRCP && uv run src/embed-service.py --port 8461 --no-backlog

Results accumulate in data/bench/RESULTS.tsv, one row per grid point, with the
parameters and the per-kind metrics, so a sweep interrupted halfway is not lost and
two sweeps a month apart can be read side by side.

Written by Claude Code (Opus 5).
"""

from __future__ import annotations

import csv
import itertools
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import click
from loguru import logger

from lib import manifest_io
from lib.evalbake import acceptable
from lib.corpus import in_corpus

# The CUDA wheels onnxruntime-gpu needs on sys.path. uv layers them in for the
# duration of the run; without them --gpu dies in the first inference, which is the
# bug this list exists to keep fixed (see the batching entry in DESIGN.md).
CUDA_EXTRAS = ["nvidia-cudnn-cu12", "nvidia-cublas-cu12", "nvidia-cuda-runtime-cu12",
               "nvidia-cufft-cu12", "nvidia-curand-cu12"]

# Columns of data/bench/RESULTS.tsv. Written once at creation and then only
# appended to, so a column added later would have to be added here AND backfilled;
# keeping the set small is deliberate.
RESULT_FIELDS = ["run_at", "slug", "docs", "chunks", "target_tokens", "overlap_tokens",
                 "boundaries", "tables", "page_weight", "quant", "dims", "weight",
                 "kind", "n", "page_1", "page_5", "page_10", "mrr"]


def gold_documents(queries: Path) -> set[str]:
    """The file names the eval set asks about.

    Parameters
    ----------
    queries
        data/EVAL_QUERIES.tsv, as written by scripts/eval_queries.py.

    Returns
    -------
    set of str
        One name per gold document, including the alternatives in `gold_alt`. A
        query whose gold document is missing from the index cannot be answered at
        all, so these are not optional; leaving out an alternative would be subtler
        and worse, since the sweep would then score as a miss an answer the full
        index gives correctly, and only for the configurations that happen to
        prefer the missing document.
    """
    names: set[str] = set()
    with queries.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            names.update(name for name, _ in acceptable(row))
    return names


def bench_corpus(*, queries: Path, manifest: Path, distractors: int, seed: int) -> list[str]:
    """Pick the documents a sweep runs over: every gold document plus a sample.

    The sample is drawn from the sorted manifest with a seeded shuffle rather than
    from the directory listing, because the manifest is tracked: the same seed picks
    the same documents on another machine and after another clone.

    Parameters
    ----------
    queries
        data/EVAL_QUERIES.tsv.
    manifest
        data/MANIFEST.tsv, the list of served documents.
    distractors
        How many non-gold documents to add. 0 measures an index of gold documents
        only, which flatters every configuration equally and is mostly useful for
        checking that the harness itself works.
    seed
        Fixes the sample.

    Returns
    -------
    list of str
        File names, sorted, gold documents included.
    """
    gold = gold_documents(queries)
    served = sorted(manifest_io.read_by_file(manifest))
    missing = sorted(gold - set(served))
    if missing:
        raise click.ClickException(
            f"{len(missing)} gold document(s) are not in {manifest}: " + ", ".join(missing[:3])
        )
    pool = [name for name in served if name not in gold]
    random.Random(seed).shuffle(pool)
    return sorted(gold | set(pool[:distractors]))


def run(command: list[str], *, log: Path, what: str) -> None:
    """Run one step of the chain, sending its output to a file.

    Parameters
    ----------
    command
        argv. Run from the repository root, which is this file's parent's parent.
    log
        Where stdout and stderr go. A sweep produces a lot of output and none of it
        is interesting until something fails, at which point the tail of this file
        is exactly what is wanted.
    what
        Human name for the step, used in the log line and in the error.

    Raises
    ------
    click.ClickException
        If the step exits non-zero. The tail of the log is included, because the
        alternative is a sweep that says "chunking failed" and nothing else.
    """
    root = Path(__file__).resolve().parent.parent
    started = time.monotonic()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        code = subprocess.call(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT)
    took = time.monotonic() - started
    if code != 0:
        tail = "\n".join(log.read_text(encoding="utf-8").splitlines()[-15:])
        raise click.ClickException(f"{what} failed ({code}), see {log}:\n{tail}")
    logger.info(f"{what} in {took:.0f}s")


def point_slug(params: dict[str, object]) -> str:
    """A short directory name for one grid point.

    The name carries the parameters that change the chunks, so two points never
    share a directory and a stale run cannot be mistaken for a fresh one.

    Parameters
    ----------
    params
        The chunking parameters of the point.

    Returns
    -------
    str
        Something like `t300-o60-bound-tab`.
    """
    if params["target_tokens"] == "section":
        # Sections have no token target and cut at headings, so neither the
        # size nor the boundary mode says anything about them.
        parts = ["section"]
    else:
        parts = [f"t{params['target_tokens']}", f"o{params['overlap_tokens']}"]
        parts.append("bound" if params["boundaries"] else "greedy")
    parts.append("tab" if params["tables"] else "notab")
    return "-".join(parts)


def append_results(path: Path, rows: list[dict[str, object]]) -> None:
    """Append one grid point's rows to the results file, writing a header if new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t",
                                lineterminator="\n", extrasaction="ignore")
        if fresh:
            writer.writeheader()
        writer.writerows(rows)


def numbers(value: str) -> list[int]:
    """Parse a comma-separated list of integers from the command line."""
    return [int(part) for part in str(value).split(",") if part.strip()]


@click.command()
@click.option("--queries", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              **in_corpus("data/EVAL_QUERIES.tsv"),
              help="The eval set. Regenerate it with scripts/eval_queries.py after "
                   "adding queries there.")
@click.option("--manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              **in_corpus("data/MANIFEST.tsv"))
@click.option("--guidelines", type=click.Path(exists=True, file_okay=False, path_type=Path),
              **in_corpus("data/GUIDELINES"),
              help="The only directory that may be indexed. Passed through to chunk.py "
                   "explicitly rather than globbed here.")
@click.option("--work", type=click.Path(path_type=Path), **in_corpus("data/bench"), help="Where each grid point's chunks, vectors and index go.")
@click.option("--distractors", default=100, show_default=True,
              help="Non-gold documents added to the index. More is slower and closer to "
                   "the shipped corpus; 0 indexes the gold documents alone.")
@click.option("--seed", default=20260919, show_default=True,
              help="Fixes which distractors are drawn, so two runs compare like with like.")
@click.option("--target-tokens", default="300", show_default=True,
              help="Comma-separated chunk sizes to sweep.")
@click.option("--overlap-tokens", default="60", show_default=True,
              help="Comma-separated overlaps to sweep.")
@click.option("--boundaries/--no-boundaries", default=True, show_default=True,
              help="Cut at structure, or greedily at the token target.")
@click.option("--tables/--no-tables", default=True, show_default=True,
              help="Read tables out cell by cell. --no-tables reproduces chunk format 6.")
@click.option("--section-chunks/--no-section-chunks", default=False, show_default=True,
              help="One passage per heading run (chunk.py --section-chunks) instead of "
                   "a token target: the token grid is ignored and the point is one.")
@click.option("--page-weight", default=0.15, show_default=True,
              help="Weight of the whole-page index in the blend. 0 skips the page bake "
                   "entirely, which halves the cost of a point.")
@click.option("--quant", default="binary", show_default=True,
              help="Stored precision, comma-separated to sweep: binary, int8. A quant "
                   "sweep is nearly free, since the chunks and the vectors are shared "
                   "and only the index is rebuilt.")
@click.option("--dims", default=1024, show_default=True)
@click.option("--gpu/--no-gpu", default=True, show_default=True,
              help="Bake on the GPU. On CPU a point takes long enough that a sweep is "
                   "back to being an afternoon.")
@click.option("--results", type=click.Path(path_type=Path), **in_corpus("data/bench/RESULTS.tsv"))
def main(queries: Path, manifest: Path, guidelines: Path, work: Path, distractors: int,
         seed: int, target_tokens: str, overlap_tokens: str, boundaries: bool, tables: bool,
         section_chunks: bool, page_weight: float, quant: str, dims: int, gpu: bool,
         results: Path) -> None:
    """Chunk, bake, index and evaluate the eval corpus for every point of a grid."""
    embed_url = os.environ.get("EMBED_URL", "").rstrip("/")
    if not embed_url:
        raise click.ClickException(
            "set EMBED_URL to a running encoder, or the chunking happens and the "
            "evaluation then cannot run:\n"
            "  cd ../justelesRCP && uv run src/embed-service.py --port 8461 --no-backlog"
        )

    names = bench_corpus(queries=queries, manifest=manifest, distractors=distractors, seed=seed)
    work.mkdir(parents=True, exist_ok=True)
    only = work / "docs.txt"
    only.write_text(
        f"# {len(names)} documents: every gold document of {queries}, plus {distractors} "
        f"distractors drawn with seed {seed}\n" + "\n".join(names) + "\n", encoding="utf-8")
    logger.info(f"{len(names)} documents ({len(gold_documents(queries))} gold) -> {only}")

    if section_chunks:
        # One point, recorded in RESULTS.tsv with "section" where a token target
        # would be, so the row reads against the token rows without a new column.
        grid = [dict(target_tokens="section", overlap_tokens=0, boundaries=boundaries,
                     tables=tables)]
    else:
        grid = [dict(target_tokens=t, overlap_tokens=o, boundaries=boundaries, tables=tables)
                for t, o in itertools.product(numbers(target_tokens), numbers(overlap_tokens))]
    logger.info(f"{len(grid)} grid point(s)")

    for position, params in enumerate(grid, start=1):
        slug = point_slug(params)
        cell = work / slug
        logger.info(f"[{position}/{len(grid)}] {slug}")

        chunks = cell / "chunks"
        if params["target_tokens"] == "section":
            how = ["--section-chunks"]
        else:
            how = ["--target-tokens", str(params["target_tokens"]),
                   "--overlap-tokens", str(params["overlap_tokens"]),
                   "--boundaries" if params["boundaries"] else "--no-boundaries"]
        run(["uv", "run", "scripts/chunk.py", "--guidelines", str(guidelines),
             "--out", str(chunks), "--only", str(only), *how,
             "--tables" if params["tables"] else "--no-tables"],
            log=cell / "chunk.log", what=f"{slug}: chunk")

        bake = ["uv", "run"]
        if gpu:
            bake += [arg for extra in CUDA_EXTRAS for arg in ("--with", extra)]
        run(bake + ["scripts/embed.py", "--chunks", str(chunks), "--out", str(cell / "vectors"),
                    "--out-name", "meta-gpu", "--weights", "model.onnx",
                    "--manifest", str(manifest)] + (["--gpu"] if gpu else []),
            log=cell / "embed.log", what=f"{slug}: bake")

        index_args = ["uv", "run", "scripts/build_index.py", "--chunks", str(chunks),
                      "--vectors", str(cell / "vectors" / "meta-gpu"), "--manifest", str(manifest),
                      "--dims", str(dims), "--page-weight", str(page_weight)]
        if page_weight > 0:
            page_chunks = cell / "chunks-page"
            run(["uv", "run", "scripts/chunk.py", "--guidelines", str(guidelines),
                 "--out", str(page_chunks), "--only", str(only), "--page-chunks"],
                log=cell / "chunk_page.log", what=f"{slug}: page chunk")
            run(bake + ["scripts/embed.py", "--chunks", str(page_chunks),
                        "--out", str(cell / "vectors"), "--out-name", "page-meta-gpu",
                        "--weights", "model.onnx", "--manifest", str(manifest)]
                + (["--gpu"] if gpu else []),
                log=cell / "embed_page.log", what=f"{slug}: page bake")
            index_args += ["--page-chunks", str(page_chunks),
                           "--page-vectors", str(cell / "vectors" / "page-meta-gpu")]

        # The quantisation sweep sits inside the point: the chunks and the vectors
        # above are what a precision change does NOT affect, so rebuilding them per
        # value would triple the cost of answering "does int8 buy anything".
        for precision in [part.strip() for part in quant.split(",") if part.strip()]:
            name = slug if len(quant.split(",")) == 1 else f"{slug}-{precision}"
            dist = cell / f"dist-{precision}"
            run(index_args + ["--quant", precision, "--out", str(dist)],
                log=cell / f"index-{precision}.log", what=f"{name}: index")

            report = cell / f"eval-{precision}.json"
            environment = dict(os.environ, INDEX_DIR=str((dist / "index").resolve()),
                               QUERIES=str(queries.resolve()), REPORT_JSON=str(report.resolve()))
            root = Path(__file__).resolve().parent.parent
            code = subprocess.call(["node", "scripts/evaluate_rescore.mjs"], cwd=root,
                                   env=environment)
            if code != 0:
                raise click.ClickException(f"{name}: evaluation failed ({code})")

            # The metrics come out of the evaluation's own JSON rather than being
            # recomputed here: there must stay exactly one definition of a hit, and it
            # is the one in evaluate_rescore.mjs that imports the shipped
            # src/search.js.
            payload = json.loads(report.read_text(encoding="utf-8"))
            rows = [{"run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "slug": name,
                     "docs": len(names), "chunks": payload["n_chunks"],
                     "page_weight": page_weight, "quant": precision, "dims": dims,
                     "weight": payload["shipped_weight"], **params, **group}
                    for group in payload["by_kind"]]
            append_results(results, rows)
            logger.success(f"{name}: " + ", ".join(
                f"{group['kind']} page@1 {group['page_1']:.4f}" for group in payload["by_kind"]))

    logger.success(f"{len(grid)} point(s) in {results}")


if __name__ == "__main__":
    sys.exit(main())
