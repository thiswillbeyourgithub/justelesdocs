#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "loguru"]
# ///
"""Read the chunking sweep's dumps and the judge's verdicts, and say which won.

Two metrics, side by side, because they fail in different directions:

- **The judge.** The mean probability Qwen3-Reranker-4B puts on "this passage
  answers this question", over the top 1, 3 and 10 of each strategy's own result
  list. It has no idea which strategy produced what, and it is the only metric here
  that can reward a strategy for returning a BETTER passage than the labelled one.
  Its weakness is that it is a model: it has a length bias (run the judge a second
  time with a smaller --doc-tokens to see it) and it has never been checked against
  a domain expert.
- **The label.** page@1, page@5, page@10 and MRR against the (file, page) ground
  truth in data/EVAL_QUERIES.tsv, computed by the shipped `goldRank` at dump time.
  Its weakness is the whole reason the judge exists, but it is an actual human
  judgement about an actual page, which no model score is.

A strategy that wins on both is a decision. A strategy that wins on one and loses
the other is the interesting row, and the point of printing them together.

`win` is per QUERY rather than on the means: for each query the strategy's mean top-3
judge score is compared with the baseline's, and ties inside --epsilon count as
neither. A strategy can hold a higher mean while losing most queries (one query it
fixes spectacularly), and that is worth seeing before shipping a chunking.

    uv run scripts/grid_report.py                      # table plus data/GRID_RESULTS.tsv
    uv run scripts/grid_report.py --by-kind            # split by question type

Written by Claude Code (Opus 5).
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from loguru import logger

from lib import judge_cache
from lib.corpus import in_corpus


#: Re-exported: the dumps' `text_sha` is checked against it in tests/test_grid.py.
sha = judge_cache.sha


def read_scores(path: Path, budget: int) -> dict[tuple[str, str], float]:
    """Verdicts at one token budget, keyed by (query hash, passage hash).

    The format and the reader are `lib.judge_cache`'s.
    """
    scores, _ = judge_cache.read_cache(path)
    return {(q, d): s for (q, d, n), s in scores.items() if n == budget}


def mean(values: list[float]) -> float:
    """Mean, or 0.0 for an empty list (a strategy nothing was judged for)."""
    return sum(values) / len(values) if values else 0.0


def per_query(dump: dict, scores: dict, threshold: float) -> list[dict]:
    """One row per query: the judge's view and the label's view of that query.

    Returns
    -------
    list of dict
        `top1`, `top3`, `top10` (mean judge score over that prefix), `best` (the
        highest score anywhere in the list), `rel1`/`rel10` (anything over
        `threshold` at rank 1 / in the list), `rank` (the labelled page's rank, 0
        for absent) and `missing` (candidates with no verdict yet).
    """
    rows = []
    for entry in dump["queries"]:
        qhash = judge_cache.sha(entry["query"])
        graded = [scores.get((qhash, c["text_sha"])) for c in entry["candidates"]]
        missing = sum(1 for g in graded if g is None)
        known = [g for g in graded if g is not None]
        head = [g for g in graded[:1] if g is not None]
        rows.append({
            "query_id": entry["query_id"],
            "type": entry.get("type", ""),
            "top1": mean(head),
            "top3": mean([g for g in graded[:3] if g is not None]),
            "top10": mean(known),
            "best": max(known) if known else 0.0,
            "rel1": 1.0 if head and head[0] >= threshold else 0.0,
            "rel10": 1.0 if any(g >= threshold for g in known) else 0.0,
            "rank": int(entry["gold_rank"]),
            "missing": missing,
        })
    return rows


def summarise(rows: list[dict], baseline: list[dict] | None, epsilon: float) -> dict:
    """Averages over queries, plus the win/loss count against a baseline."""
    n = len(rows) or 1
    at = lambda k: sum(1 for r in rows if 0 < r["rank"] <= k) / n  # noqa: E731
    summary = {
        "n": len(rows),
        "judge_1": mean([r["top1"] for r in rows]),
        "judge_3": mean([r["top3"] for r in rows]),
        "judge_10": mean([r["top10"] for r in rows]),
        "judge_best": mean([r["best"] for r in rows]),
        "rel_1": mean([r["rel1"] for r in rows]),
        "rel_10": mean([r["rel10"] for r in rows]),
        "page_1": at(1),
        "page_5": at(5),
        "page_10": at(10),
        "mrr": sum(1 / r["rank"] for r in rows if r["rank"]) / n,
        "missing": sum(r["missing"] for r in rows),
    }
    if baseline is not None:
        against = {r["query_id"]: r for r in baseline}
        wins = losses = 0
        for row in rows:
            other = against.get(row["query_id"])
            if other is None:
                continue
            if row["top3"] > other["top3"] + epsilon:
                wins += 1
            elif row["top3"] < other["top3"] - epsilon:
                losses += 1
        summary["win"] = wins
        summary["loss"] = losses
    return summary


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--grid", type=click.Path(path_type=Path), **in_corpus("data/grid"), help="Where the per-strategy dumps live.")
@click.option("--cache", "cache_path", type=click.Path(path_type=Path),
              **in_corpus("data/grid/judge-scores.jsonl"))
@click.option("--out", type=click.Path(path_type=Path),
              **in_corpus("data/GRID_RESULTS.tsv"))
@click.option("--doc-tokens", type=int, default=1024, show_default=True,
              help="Which judging budget to report. Must match a judge run.")
@click.option("--baseline", default="t170-512-sec", show_default=True,
              help="The strategy win/loss is counted against. The default is the "
                   "chunking that ships today, so every row reads as a change to it. "
                   "A grid judged before that row existed has no such strategy: pass "
                   "--baseline t1024-o256-sec, which shipped until 2026-09-30.")
@click.option("--threshold", type=float, default=0.5, show_default=True,
              help="Judge score at which a passage counts as answering the question.")
@click.option("--epsilon", type=float, default=0.02, show_default=True,
              help="Judge-score difference below which a query is neither a win nor "
                   "a loss. The judge is not precise to the third decimal.")
@click.option("--by-kind", is_flag=True, default=False,
              help="Also print one table per question type (precise, vague, "
                   "crosslingual): an average over all of them hides the thing most "
                   "likely to break on its own.")
def main(grid: Path, cache_path: Path, out: Path, doc_tokens: int, baseline: str,
         threshold: float, epsilon: float, by_kind: bool) -> None:
    """Compile data/GRID_RESULTS.tsv from the dumps and the judge's cache."""
    dumps = sorted(grid.glob("*/candidates.json"))
    if not dumps:
        raise click.ClickException(f"no <strategy>/candidates.json under {grid}")
    scores = read_scores(cache_path, doc_tokens)
    logger.info(f"{len(dumps)} strategies, {len(scores)} verdicts at "
                f"--doc-tokens {doc_tokens}")

    loaded = {}
    for path in dumps:
        payload = json.loads(path.read_text(encoding="utf-8"))
        loaded[payload["strategy"]] = payload
    if baseline not in loaded:
        logger.warning(f"no baseline strategy '{baseline}' in the grid; "
                       "win/loss columns will be empty")

    base_rows = (per_query(loaded[baseline], scores, threshold)
                 if baseline in loaded else None)
    kinds = ["all"]
    if by_kind:
        kinds += sorted({q.get("type", "") for d in loaded.values()
                         for q in d["queries"] if q.get("type")})

    columns = ["strategy", "kind", "chunks", "n", "judge_1", "judge_3", "judge_10",
               "judge_best", "rel_1", "rel_10", "page_1", "page_5", "page_10", "mrr",
               "win", "loss", "missing"]
    table = []
    for kind in kinds:
        keep = lambda rows: [r for r in rows if kind == "all" or r["type"] == kind]  # noqa: E731
        for name, payload in sorted(loaded.items()):
            rows = keep(per_query(payload, scores, threshold))
            summary = summarise(rows, keep(base_rows) if base_rows else None, epsilon)
            table.append({"strategy": name, "kind": kind,
                          "chunks": payload["meta"]["n_chunks"], **summary})

    def cell(row: dict, column: str) -> str:
        value = row.get(column, "")
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(["\t".join(columns)]
                             + ["\t".join(cell(r, c) for c in columns) for r in table])
                   + "\n", encoding="utf-8")

    for kind in kinds:
        print(f"\n{kind}:")
        head = ["strategy", "chunks", "judge_1", "judge_3", "judge_10", "rel_10",
                "page_1", "page_10", "mrr", "win", "loss", "missing"]
        print("  " + "".join(h.rjust(14 if h == "strategy" else 9) for h in head))
        # Best first on the judge's top-3 mean, which is the number the sweep is for.
        for row in sorted((r for r in table if r["kind"] == kind),
                          key=lambda r: -r["judge_3"]):
            mark = " <- baseline" if row["strategy"] == baseline else ""
            print("  " + "".join(cell(row, h).rjust(14 if h == "strategy" else 9)
                                 for h in head) + mark)
    total_missing = sum(r["missing"] for r in table if r["kind"] == "all")
    if total_missing:
        logger.warning(f"{total_missing} (query, passage) pairs have no verdict yet: "
                       "run scripts/judge_rerank.py again before reading this table")
    logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()
