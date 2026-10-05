# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "httpx", "loguru", "numpy"]
# ///
"""Does removing a French/English direction help a query reach the other language?

The question, asked on 2026-09-18: search visibly favours documents written in the
query's own language, so a French question about housing ranks the French reports
first even when an English report answers it better. If "j'aime les chiens" and "I
like dogs" mean the same thing, the difference between their vectors is mostly
language and not meaning. Average that difference over many translation pairs and
you have a direction; project it out of every vector and, the argument goes,
language stops deciding the ranking.

What this script does not assume: that it works. Two things could go wrong and
both are measurable here. The direction could carry meaning as well as language,
in which case removing it hurts everything. Or the model's language separation
could be spread over many directions rather than one, in which case removing one
does nothing at all. The table below the run tells which.

Three variants are measured against the plain bake, all on the fp16 vectors,
because the shipped one-bit index cannot show a change of a fraction of a bit:

    passages plain,    queries plain          the shipped ranking
    passages cleaned,  queries cleaned        the proposal
    passages cleaned,  queries plain          cleaning at bake time only
    passages plain,    queries cleaned        cleaning in the browser only

Results are broken out by kind of question, since the whole point is the 23
crosslingual queries; a variant that helps them by three and costs the other 143
one each is a trade to look at, and an average hides it in both directions.

Usage, with the shared encoder running (see CLAUDE.md, Commands):

    uv run scripts/eval_language_axis.py --embed-url http://127.0.0.1:8461

Not in the build chain, nothing imports it. Written by Claude Code (Opus 5).
"""

from __future__ import annotations

import csv
from pathlib import Path

import click
import numpy as np
from loguru import logger

from lib.evalbake import by_kind, embed_queries, gold_ranks, load_bake
from lib.corpus import in_corpus

# Translation pairs, French first. Deliberately ordinary sentences about nothing
# in the corpus's domain: a pair of domain terms ("qualité de l'eau" / "water
# quality") would build a direction that also carries the domain, and projecting
# THAT out of the corpus would remove the topic along with the language. The pairs stay short and
# everyday for the same reason the encoder's own training pairs do, and properly
# accented: unaccented French would fold "writes without accents" into the
# direction, and the corpus is accented.
PAIRS: list[tuple[str, str]] = [
    ("J'aime les chiens.", "I like dogs."),
    ("Il pleut depuis ce matin.", "It has been raining since this morning."),
    ("La réunion commence à neuf heures.", "The meeting starts at nine o'clock."),
    ("Elle habite près de la gare.", "She lives near the station."),
    ("Le train est arrivé en retard.", "The train arrived late."),
    ("Nous avons mangé au restaurant hier soir.", "We ate at a restaurant last night."),
    ("Ce livre est très intéressant.", "This book is very interesting."),
    ("Peux-tu fermer la fenêtre ?", "Could you close the window?"),
    ("Les enfants jouent dans le jardin.", "The children are playing in the garden."),
    ("Je ne comprends pas cette phrase.", "I do not understand this sentence."),
    ("Le magasin ferme à dix-huit heures.", "The shop closes at six in the evening."),
    ("Il faut tourner à gauche après le pont.", "You have to turn left after the bridge."),
    ("Mon frère travaille dans une banque.", "My brother works in a bank."),
    ("La soupe est trop chaude.", "The soup is too hot."),
    ("Nous partons en vacances la semaine prochaine.", "We are going on holiday next week."),
    ("Le téléphone a sonné trois fois.", "The telephone rang three times."),
    ("Elle a oublié ses clés à la maison.", "She left her keys at home."),
    ("Ce chemin mène à la rivière.", "This path leads to the river."),
    ("Le concert a duré deux heures.", "The concert lasted two hours."),
    ("Je préfère le thé au café.", "I prefer tea to coffee."),
]


def language_axis(*, embed_url: str, dims: int) -> np.ndarray:
    """A unit vector pointing from English to French, averaged over the pairs.

    Parameters
    ----------
    embed_url
        The encoder service. Queries and pairs go through the same encoder, which
        matters: the passage bake and the query path use different code, and an
        axis built on one side would not sit where the other side's vectors do.
    dims
        Requested width.

    Returns
    -------
    numpy.ndarray
        `dims` float32, unit norm.

    Notes
    -----
    Each pair is normalised before subtracting, so a long sentence does not weigh
    more than a short one. The mean of the differences is the axis; its norm before
    normalising says how consistent the pairs were, and is logged, because an axis
    averaged out of twenty inconsistent differences would be noise dressed as a
    direction.
    """
    french = embed_queries(texts=[fr for fr, _en in PAIRS], embed_url=embed_url, dims=dims)
    english = embed_queries(texts=[en for _fr, en in PAIRS], embed_url=embed_url, dims=dims)
    french /= np.linalg.norm(french, axis=1, keepdims=True)
    english /= np.linalg.norm(english, axis=1, keepdims=True)
    differences = french - english
    axis = differences.mean(axis=0)
    per_pair = np.linalg.norm(differences, axis=1).mean()
    logger.info(f"axis norm {float(np.linalg.norm(axis)):.4f} against a mean pair "
                f"difference of {float(per_pair):.4f}: the closer these are, the more "
                "the pairs agreed on one direction")
    return axis / np.linalg.norm(axis)


def without(matrix: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Project the axis out of every row, then renormalise.

    Parameters
    ----------
    matrix
        Row-wise vectors.
    axis
        Unit vector to remove.

    Returns
    -------
    numpy.ndarray
        Rows orthogonal to `axis`, unit norm. Renormalising matters because the
        ranking is a dot product: a row that lay close to the axis loses more length
        than one that did not, and without renormalising it would be demoted for
        having been French rather than for being a worse answer.
    """
    cleaned = matrix - np.outer(matrix @ axis, axis)
    return cleaned / np.linalg.norm(cleaned, axis=1, keepdims=True)


@click.command()
@click.option("--chunks", type=click.Path(exists=True, file_okay=False, path_type=Path),
              **in_corpus("data/chunks"))
@click.option("--vectors", type=click.Path(exists=True, file_okay=False, path_type=Path),
              **in_corpus("data/vectors/meta-gpu"),
              help="The GPU bake. The one-bit index cannot show this effect.")
@click.option("--queries", "queries_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              **in_corpus("data/EVAL_QUERIES.tsv"))
@click.option("--embed-url", default="http://127.0.0.1:8461", show_default=True,
              help="../justelesRCP/src/embed-service.py, running.")
@click.option("--dims", default=1024, show_default=True)
@click.option("--depth", default=50, show_default=True,
              help="Shortlist length, matching RESCORE_DEPTH in src/search.js.")
def main(chunks: Path, vectors: Path, queries_path: Path, embed_url: str,
         dims: int, depth: int) -> None:
    """Measure what removing a language direction does to retrieval."""
    matrix, owner, pages = load_bake(chunks=chunks, vectors=vectors)
    logger.info(f"{matrix.shape[0]} chunks at {dims} dims from {vectors}")

    axis = language_axis(embed_url=embed_url, dims=dims)
    queries = list(csv.DictReader(queries_path.open(encoding="utf-8"), delimiter="\t"))
    raw = embed_queries(texts=[q["query"] for q in queries], embed_url=embed_url, dims=dims)
    plain_query = raw / np.linalg.norm(raw, axis=1, keepdims=True)

    # How far apart the two halves of the corpus sit along the axis, before anything
    # is removed. If this is near zero the axis is not finding language in the bake
    # and the rest of the table is measuring noise.
    projection = matrix @ axis
    logger.info(f"passage projection on the axis: mean {float(projection.mean()):+.4f}, "
                f"spread {float(projection.std()):.4f}")

    cleaned = without(matrix, axis)
    cleaned_query = without(plain_query, axis)

    kinds = ["all"] + sorted({q["type"] for q in queries if q.get("type")})
    click.echo("\npassages  queries   " + "".join(f"{kind[:9]:>10}" for kind in kinds)
               + "   (page@10, then MRR)")
    for label, query_side, passage_side in (
        ("plain     plain  ", plain_query, matrix),
        ("cleaned   cleaned", cleaned_query, cleaned),
        ("cleaned   plain  ", plain_query, cleaned),
        ("plain     cleaned", cleaned_query, matrix),
    ):
        ranks = gold_ranks(scores=query_side @ passage_side.T, queries=queries,
                           owner=owner, pages=pages, depth=depth)
        grouped = by_kind(ranks, queries)
        click.echo(f"{label}   " + "".join(f"{grouped[k]['page_10']:>10.4f}" for k in kinds))
        click.echo(f"{'':17}  " + "".join(f"{grouped[k]['mrr']:>10.4f}" for k in kinds))

    counts = by_kind([0] * len(queries), queries)
    click.echo("\na group of n queries carries a standard error near 0.5/sqrt(n): "
               + ", ".join(f"{k} {0.5 / np.sqrt(counts[k]['n']):.3f}" for k in kinds))


if __name__ == "__main__":
    main()
