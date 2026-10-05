#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click",
#   "loguru",
#   "tqdm",
#   "torch",
#   "transformers>=4.51",
#   "accelerate",
# ]
# ///
"""Grade retrieved passages with a SOTA cross-encoder, on the machine that has a GPU.

Why this exists
---------------
The chunking sweep builds one index per strategy and has to say which one retrieves
better. The eval set's (file, page) label answers a narrower question: "did the page
I wrote this question from come back". A strategy that returns a BETTER passage, from
the argumentaire instead of the synthèse, or from the 15th edition instead of the
14th, is scored as a miss. With 535 documents and several renditions of the same
guideline that error is not rare and it is not symmetric.

So each index's top K is dumped verbatim (scripts/dump_candidates.mjs) and every
(query, passage) pair is scored here by Qwen3-Reranker-4B, which is far too large to
serve but is not in the serving path: it runs once, offline, on the 3090 Ti, and its
scores are the yardstick the strategies are measured against.

The judge is blind to which strategy produced a passage. It is handed a query and a
passage and nothing else, and the same passage retrieved by three strategies is
scored ONCE and shared, which is both a large saving and the reason no strategy can
be graded on a slightly different reading of the same text.

How the score is computed
-------------------------
Qwen3-Reranker is not a regression head. It is the instruct model asked a yes/no
question in its own chat template, and the score is the probability it puts on "yes"
at the first generated position:

    softmax([logit("no"), logit("yes")])[1]

so every score is in [0, 1] and comparable across queries, which a raw cross-encoder
logit (unbounded, calibrated per model) is not.

Length bias
-----------
A cross-encoder that reads 1000 tokens has more chances to find something relevant
than one that reads 300, and this sweep varies exactly that. `--doc-tokens` is the
control: it truncates every passage to the same budget before judging, so a run at
--doc-tokens 256 asks "which chunking puts the answer in its first 256 tokens" and
the default asks "which chunking retrieves text that answers the question". The
budget is part of the cache key, so both can live in the same cache file.

Resuming
--------
Every verdict is appended to a JSONL cache as soon as its batch finishes, and the
cache is read back at startup. Killing this script (the box is needed for something
else) loses at most one batch, and starting it again picks up where it stopped.
`--calibrate` scores a handful of pairs and prints the projected wall-clock for the
rest instead of running it, which is how to get an honest ETA before committing the
GPU for hours.

    uv run scripts/judge_rerank.py --calibrate 24
    uv run scripts/judge_rerank.py                      # everything not yet judged

Written by Claude Code (Opus 5).
"""

from __future__ import annotations

import json
import signal
import time
from pathlib import Path

import click
import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib import corpus_config, judge_cache
from lib.corpus import in_corpus

MODEL = "Qwen/Qwen3-Reranker-4B"

# The instruction the judge is given, once, for every pair. It is the CORPUS's to
# write (corpus.toml [judge] task), because it should name who asks and what the
# documents are; this is the fallback. Either way "relevant" has to mean "answers
# the question", not "is on the same topic", or every passage from the right
# document scores high and the chunking strategies become indistinguishable.
DEFAULT_TASK = ("Given a reader's question, judge whether the passage from a reference "
                "document contains information that answers it. The question and the "
                "passage may be in different languages, which does not make the passage "
                "irrelevant.")


def judge_task() -> str:
    """The judge's instruction: corpus.toml [judge] task, else DEFAULT_TASK."""
    return corpus_config.load().get("judge", {}).get("task", DEFAULT_TASK)

# Qwen3-Reranker's own template, from the model card. The empty <think></think> block
# is not decoration: the model is a reasoning model and the template puts it in the
# post-thinking state, where the next token is the answer rather than the start of a
# chain of thought. Without it the "yes"/"no" logits are read off a position that was
# never meant to hold the answer.
PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements '
          'based on the Query and the Instruct provided. Note that the answer can '
          'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def read_cache(path: Path) -> dict[tuple[str, str, int], float]:
    """Verdicts already on disk, keyed by (query hash, passage hash, token budget).

    The format and the reader are `lib.judge_cache`'s; this only reports the
    lines an interrupted write left unparseable.
    """
    scores, skipped = judge_cache.read_cache(path)
    if skipped:
        logger.warning(f"ignoring {skipped} unparseable cache line(s) (an interrupted write)")
    return scores


def collect_pairs(dumps: list[Path], budget: int,
                  cached: dict) -> tuple[list[dict], dict[str, str]]:
    """Every (query, passage) in the dumps that has no verdict yet, deduplicated.

    Parameters
    ----------
    dumps : list of Path
        candidates.json files written by scripts/dump_candidates.mjs.
    budget : int
        The `--doc-tokens` this run judges at; part of the cache key.
    cached : dict
        Verdicts already known, so they are not queued again.

    Returns
    -------
    list of dict
        One entry per pair to score, with the query and passage text.
    dict
        Passage hash to the file it came from, used only for the log line.
    """
    todo: dict[tuple[str, str], dict] = {}
    origin: dict[str, str] = {}
    for dump in dumps:
        payload = json.loads(dump.read_text(encoding="utf-8"))
        for entry in payload["queries"]:
            qhash = judge_cache.sha(entry["query"])
            for candidate in entry["candidates"]:
                key = (qhash, candidate["text_sha"])
                origin.setdefault(candidate["text_sha"], candidate["file"])
                if (qhash, candidate["text_sha"], budget) in cached or key in todo:
                    continue
                todo[key] = {"q": qhash, "d": candidate["text_sha"],
                             "query": entry["query"], "text": candidate["text"]}
    return list(todo.values()), origin


class Judge:
    """The model, its template and the two token ids the score is read from."""

    def __init__(self, model_name: str, *, dtype: str, quant: str,
                 max_tokens: int) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        kwargs: dict = {"dtype": getattr(torch, dtype)}
        if quant in {"4bit", "8bit"}:
            # bitsandbytes is an optional dependency on purpose: the default path is
            # bf16 on an empty card, and asking every run to carry a quantisation
            # library for the case where the GPU is busy is the wrong default. It is
            # not in this script's PEP 723 header, so this path wants
            # `uv run --with bitsandbytes`.
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:  # pragma: no cover - depends on the wheel set
                raise click.ClickException(
                    "--quant needs bitsandbytes: rerun as "
                    "`uv run --with bitsandbytes scripts/judge_rerank.py "
                    f"--quant {quant} ...`") from exc
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=quant == "4bit", load_in_8bit=quant == "8bit",
                bnb_4bit_compute_dtype=torch.bfloat16)
            kwargs["device_map"] = "cuda:0"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).eval()
        if quant == "none":
            self.model = self.model.to("cuda:0")
        # Read once. A tokenizer that splits "yes" differently would silently score
        # every pair off an unrelated logit, so this is an assertion, not a lookup.
        self.yes = self.tokenizer.convert_tokens_to_ids("yes")
        self.no = self.tokenizer.convert_tokens_to_ids("no")
        unknown = self.tokenizer.unk_token_id
        assert self.yes not in (None, unknown) and self.no not in (None, unknown), \
            "this tokenizer does not have single 'yes' and 'no' tokens"
        self.prefix_ids = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.task = judge_task()
        self.max_tokens = max_tokens

    def encode(self, pair: dict, budget: int) -> list[int]:
        """One pair as token ids: prefix, the truncated body, suffix.

        The body is truncated rather than the whole prompt, so the template's last
        tokens (where the answer is read) are never cut off, and the QUERY is never
        cut: a truncated question is a different question.
        """
        head = f"<Instruct>: {self.task}\n<Query>: {pair['query']}\n<Document>: "
        head_ids = self.tokenizer.encode(head, add_special_tokens=False)
        room = min(budget, self.max_tokens - len(self.prefix_ids)
                   - len(self.suffix_ids) - len(head_ids))
        body = self.tokenizer.encode(pair["text"], add_special_tokens=False)[:max(room, 16)]
        return self.prefix_ids + head_ids + body + self.suffix_ids

    @torch.inference_mode()
    def score(self, encoded: list[list[int]]) -> list[float]:
        """P(yes) for a padded batch, read at the last real position of each row."""
        width = max(len(ids) for ids in encoded)
        pad = self.tokenizer.pad_token_id or 0
        # Left padding, which is why the answer is always at index -1: the alternative
        # is gathering per-row positions, and getting that wrong is invisible (it
        # reads a logit from inside the passage and still returns plausible numbers).
        input_ids = torch.tensor([[pad] * (width - len(ids)) + ids for ids in encoded],
                                 device=self.model.device)
        mask = torch.tensor([[0] * (width - len(ids)) + [1] * len(ids) for ids in encoded],
                            device=self.model.device)
        # `logits_to_keep=1` is not an optimisation detail, it is most of this
        # batch's memory: the LM head is 151k wide, so the full [rows, width, vocab]
        # tensor is 3.1 GB for 8 rows of 1280 tokens and 10.5 GB for 32 (measured, as
        # a CUDA OOM). Only the last position is ever read here, and computing the
        # head for that one position alone also removes about 15% of the arithmetic.
        # The keyword was renamed in transformers 4.50, hence the fallback: an old
        # wheel should be slow, not broken.
        try:
            out = self.model(input_ids=input_ids, attention_mask=mask, logits_to_keep=1)
        except TypeError:
            out = self.model(input_ids=input_ids, attention_mask=mask)
        logits = out.logits[:, -1, :]
        pair = torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1).float()
        return torch.softmax(pair, dim=1)[:, 1].tolist()


def batches(pairs: list[dict], encoded: list[list[int]], *, rows: int,
            tokens: int) -> list[list[int]]:
    """Indices grouped into batches, longest first, under a row and a token budget.

    Sorted by length so a batch is not padded to twice its own content: at 1000-token
    passages the padding waste of an unsorted batch is most of the batch. Longest
    first so that if the card cannot hold the widest batch, it fails in the first
    seconds rather than an hour in.
    """
    order = sorted(range(len(pairs)), key=lambda i: -len(encoded[i]))
    out: list[list[int]] = []
    current: list[int] = []
    for i in order:
        width = len(encoded[i])
        if current and (len(current) >= rows or (len(current) + 1) * width > tokens):
            out.append(current)
            current = []
        current.append(i)
    if current:
        out.append(current)
    return out


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--grid", type=click.Path(path_type=Path), **in_corpus("data/grid"),
              help="Directory holding one <strategy>/candidates.json per strategy.")
@click.option("--cache", "cache_path", type=click.Path(path_type=Path),
              **in_corpus("data/grid/judge-scores.jsonl"),
              help="Verdicts, appended as they are produced and read back on resume.")
@click.option("--model", default=MODEL, show_default=True,
              help="The judge. Nothing about this file assumes Qwen beyond the "
                   "yes/no template, but the template IS Qwen's.")
@click.option("--doc-tokens", type=int, default=1024, show_default=True,
              help="Passage tokens the judge reads. Part of the cache key: a second "
                   "run at a smaller budget is the length-bias control, not a "
                   "contradiction of the first.")
@click.option("--max-tokens", type=int, default=1536, show_default=True,
              help="Hard ceiling on the whole prompt.")
@click.option("--batch-rows", type=int, default=8, show_default=True,
              help="Passages per forward pass.")
@click.option("--batch-tokens", type=int, default=12288, show_default=True,
              help="Padded tokens per forward pass, the real memory knob.")
@click.option("--dtype", default="bfloat16", show_default=True,
              type=click.Choice(["bfloat16", "float16", "float32"]))
@click.option("--quant", default="none", show_default=True,
              type=click.Choice(["none", "8bit", "4bit"]),
              help="4bit fits the judge beside another job on the card, at some cost "
                   "in fidelity. Scores from different --quant are NOT comparable, "
                   "and this is not in the cache key: pick one per sweep. Needs "
                   "bitsandbytes, which is deliberately not a dependency of this "
                   "script: uv run --with bitsandbytes scripts/judge_rerank.py "
                   "--quant 4bit.")
@click.option("--calibrate", type=int, default=0,
              help="Score this many pairs, print the projected wall-clock for the "
                   "rest, and stop. Nothing is written to the cache.")
@click.option("--limit", type=int, default=0, help="Stop after this many pairs.")
def main(grid: Path, cache_path: Path, model: str, doc_tokens: int, max_tokens: int,
         batch_rows: int, batch_tokens: int, dtype: str, quant: str, calibrate: int,
         limit: int) -> None:
    """Score every unjudged (query, passage) pair in the grid's dumps."""
    dumps = sorted(grid.glob("*/candidates.json"))
    if not dumps:
        raise click.ClickException(
            f"no <strategy>/candidates.json under {grid}. Run ./grid_search.sh first.")
    cached = read_cache(cache_path)
    pairs, origin = collect_pairs(dumps, doc_tokens, cached)
    logger.info(f"{len(dumps)} strategies, {len(cached)} verdicts cached, "
                f"{len(pairs)} pairs to score over "
                f"{len({p['d'] for p in pairs})} distinct passages")
    if not pairs:
        logger.info("nothing to do")
        return
    if limit:
        pairs = pairs[:limit]

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}, "
                    f"{free / 2**30:.1f} of {total / 2**30:.1f} GiB free")
        # 4B parameters at 2 bytes plus activations. Said here rather than discovered
        # as a CUDA OOM twenty minutes in, on a machine whose card is often busy.
        # Measured at the default --batch-rows: peak allocation is 9.5 GiB, which the
        # run prints at the end so this line can be checked rather than believed. It
        # only became true once the LM head stopped being computed for every position
        # (see Judge.score); before that, 8 rows peaked around 12.6 GiB and this guard
        # was admitting cards that would then die mid-sweep.
        need = {"none": 9.5, "8bit": 6.0, "4bit": 4.0}[quant]
        if free / 2**30 < need:
            raise click.ClickException(
                f"{free / 2**30:.1f} GiB free, {model} needs about {need} GiB at "
                f"--quant {quant}. Free the card (nvidia-smi), or rerun as "
                "`uv run --with bitsandbytes scripts/judge_rerank.py --quant 4bit`.")
    else:
        raise click.ClickException("no CUDA device; this judge is not a CPU job")

    started = time.perf_counter()
    judge = Judge(model, dtype=dtype, quant=quant, max_tokens=max_tokens)
    logger.info(f"loaded {model} in {time.perf_counter() - started:.0f}s")

    encoded = [judge.encode(pair, doc_tokens) for pair in pairs]
    groups = batches(pairs, encoded, rows=batch_rows, tokens=batch_tokens)
    total_tokens = sum(len(e) for e in encoded)
    logger.info(f"{len(groups)} batches, {total_tokens / 1e6:.2f}M passage tokens, "
                f"{total_tokens / len(encoded):.0f} tokens per pair on average")

    # Ctrl-C between batches rather than inside one: the cache is flushed after every
    # batch, so the cost of stopping is bounded by one forward pass and the file is
    # never half-written.
    stopping = {"now": False}

    def stop(*_: object) -> None:
        logger.warning("stopping after this batch (the cache is already on disk)")
        stopping["now"] = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    if calibrate:
        # The widest batches, since the list is sorted longest first. A calibration
        # therefore measures the slowest work there is, which makes the projection
        # below a ceiling rather than a hope.
        picked, taken = [], 0
        for group in groups:
            picked.append(group)
            taken += len(group)
            if taken >= calibrate:
                break
        groups = picked

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done = scored_tokens = 0
    clock = time.perf_counter()
    with cache_path.open("a", encoding="utf-8") as sink:
        bar = tqdm(groups, unit="batch", desc="judging",
                   postfix={"pairs": f"0/{len(pairs)}"})
        for group in bar:
            scores = judge.score([encoded[i] for i in group])
            if not calibrate:
                for i, score in zip(group, scores):
                    sink.write(json.dumps({"q": pairs[i]["q"], "d": pairs[i]["d"],
                                           "n": doc_tokens, "s": round(score, 6)}) + "\n")
                sink.flush()
            done += len(group)
            scored_tokens += sum(len(encoded[i]) for i in group)
            bar.set_postfix({"pairs": f"{done}/{len(pairs)}",
                             "tok/s": f"{scored_tokens / (time.perf_counter() - clock):.0f}"})
            if stopping["now"] or (calibrate and done >= calibrate):
                break

    elapsed = time.perf_counter() - clock
    rate = scored_tokens / elapsed if elapsed else 0
    logger.info(f"{done} pairs in {elapsed:.0f}s "
                f"({elapsed / max(done, 1) * 1000:.0f} ms per pair, {rate:.0f} tokens/s)")
    if torch.cuda.is_available():
        # Printed on every run, not just a calibration, because it is what the guard
        # above should be set from: the alternative is guessing the batch's headroom
        # and either refusing on a card that would have fitted or dying halfway.
        logger.info(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB "
                    f"(guard: {need:.1f} GiB free)")
    if calibrate:
        # Projected on TOKENS, not on pairs: the batches are sorted longest first, so
        # the pairs scored during a calibration are the slowest ones and a per-pair
        # projection would be pessimistic by a wide margin.
        logger.info(f"projected for all {len(pairs)} pairs "
                    f"({total_tokens / 1e6:.2f}M tokens): "
                    f"{total_tokens / rate / 3600:.1f} h at this rate")
        logger.info("nothing was written to the cache (--calibrate)")


if __name__ == "__main__":
    main()
