# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "loguru", "numpy", "onnxruntime-gpu", "tokenizers"]
# ///
# onnxruntime-GPU, not the CPU-only onnxruntime the VPS runs, for the reason
# ../justelesRCP/src/embed-rcp.py documents at length: this is the OFFLINE bake and
# the dev machine has a GPU, while the GPU wheel also runs fine CPU-only (the
# CUDA provider simply fails to register). Measured on this corpus, CPU int8 does
# 0.9 chunks/s, which is 40 hours for the 128,841 chunks of this corpus, so the GPU is not a nicety.
# CUDA needs the driver plus the CUDA 12 runtime and cuDNN 9. Without root, supply
# them as wheels for one run; onnx_embed.ensure_cudnn_visible then makes onnxruntime
# find them, re-executing this process with LD_LIBRARY_PATH set if it has to:
#   uv run --with nvidia-cudnn-cu12 --with nvidia-cublas-cu12 \
#          --with nvidia-cuda-runtime-cu12 --with nvidia-cufft-cu12 \
#          --with nvidia-curand-cu12 scripts/embed.py --gpu
# They stay OUT of the dependency list on purpose: a CPU-only run must not pull
# ~1.5 GB of CUDA wheels, and a bad pin must not break the working CPU fallback.
#
# MEASURED, which is why --gpu defaults to OFF and the CUDA plumbing below is
# dormant: on this corpus, with 6 intra-op threads and a batch of 64, the GPU does
# 2.7 chunks/s and the CPU alone does 2.99. onnxruntime says why in one line,
# "336 Memcpy nodes are added to the graph for CUDAExecutionProvider": the int8
# operators have no CUDA kernels, so the graph is cut into CPU and GPU partitions
# and tensors cross the bus 336 times per batch. An int8 ONNX is a CPU artefact,
# and no GPU can rescue it.
#
# A real GPU bake needs float weights, which run in minutes rather than hours:
# `--weights model.onnx --gpu`, the fp32 pair ../justelesRCP/scripts/download-model.sh
# keeps with --keep-fp32 (jina-embeddings-v5 publishes no fp16 graph; arctic did, and
# the shipped index used to be baked from it). Those are NOT the weights the VPS
# embeds queries with, so it trades a measured 2.5 hours for a mismatch between the
# query and passage sides, which DESIGN.md measured on arctic and found to cost
# nothing that 117 queries can resolve.
"""Embed every chunk locally, once, at full model width.

Embeddings are computed on this machine and shipped. The VPS is small and also
serves justelesRCP from the same box, so it never runs the encoder over the
corpus; see CLAUDE.md.

The encoder is NOT reimplemented here. It is loaded from
`../justelesRCP/src/onnx_embed.py`, the same file the VPS embed service runs,
because a query vector and a passage vector have to come from identical weights,
pooling, prefix and Matryoshka width. A second local copy of that logic would
drift silently: nothing crashes when the query side CLS-pools and the passage
side mean-pools, the results simply get worse in a way no test notices.
`chunk.py` already reaches into the sibling checkout for the tokenizer, so the
dependency is established rather than new.

**Vectors are stored at the model's full 1024 dimensions, not truncated to the
256 the site will serve.** Arctic-embed-v2.0 is Matryoshka-trained, so a
narrower vector is a prefix of a wider one: slicing to 256 and renormalising
gives bit-for-bit what encoding with `out_dim=256` would have given, because the
pre-truncation normalisation is a positive scalar. Storing full width therefore
lets the evaluation compare 1024, 512, 256 and 128 dims, and float32 against
int8, from ONE encoding pass instead of four. The truncation and the int8
quantisation belong to the shipping step, not to this one.

Output is one `.npz` per document under `data/vectors/<variant>/`, mirroring
`data/chunks/`, and gated on a content hash so adding a document embeds only
that document. The hash binds the chunk texts, the model identity, the prefix
and the format version: anything that changes what a vector MEANS must
invalidate it, or the index quietly mixes vintages.

Written by Claude Code.
"""

from __future__ import annotations

import hashlib
import importlib.util
import fcntl
import json
import math
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import click
import numpy as np
import onnxruntime as ort
from loguru import logger

from lib import atomic, corpus_config
from lib import manifest_io
from lib.encoder import MAX_PASSAGE_TOKENS
from lib.corpus import in_corpus

# The variants whose label is read from data/MANIFEST.tsv rather than built from the
# filename. Defined once because FOUR separate decisions keyed off it and every one
# of them said `variant == "meta"`: whether to load the manifest at all, whether an
# empty --meta-fields is an error, whether the manifest missing is an error, and
# whether --meta-fields belongs in the bake's cache key. Adding `metaend` and `both`
# to the layouts table on 2026-09-25 left all four behind, so those two bakes died
# one document in with "no row in the manifest" while the row was sitting there: the
# manifest had simply never been read. Keep this the single source of that fact.
CURATED_VARIANTS = frozenset({"meta", "metaend", "both", "metabi",
                             "metaq", "metanl", "metaqnl"})

# The variants that also need data/TITLES_OTHER.tsv, the hand-written translation of
# each curated title into the language the document is not in. Separate from the set
# above because the manifest alone is not enough for them and the error has to say
# which file is missing.
BILINGUAL_VARIANTS = frozenset({"metabi"})

# Bumped when the stored vectors stop being comparable with previously stored
# ones for a reason the content hash cannot see (a change to how text is fed to
# the encoder, for instance). Part of every source hash.
# 3: the tokenizer's own 512-token truncation no longer overrides MAX_PASSAGE_TOKENS.
#    Passage chunks are unaffected (the longest is 360 tokens); the page bake's rows
#    are whole pages and were being cut in half.
# 2: the sibling's batch_feed now takes the attention mask from the tokenizer.
#    tokenizer.json ships with padding enabled, so encode_batch had already padded
#    every row to the batch's longest, and the mask built here from len(ids)
#    declared those pads to be real tokens. Every vector in a version-1 bake
#    therefore depended on which other passages shared its batch of 64: cosine
#    against the same passage embedded alone was 0.987 on average and 0.920 at
#    worst. Version 2 measures 0.99999+ at every batch size from 8 to 512.
EMBED_FORMAT_VERSION = 3

# Longest passage, in tokens, fed to the encoder, and it is passed to the Encoder
# as `max_tokens` so that it is the tokenizer's ceiling too: tokenizer.json ships
# with truncation at 512, which used to win. Chunking targets 300 tokens and the
# longest chunk this corpus produces is 360, so no ordinary chunk is ever cut; the
# number is here for the page bake, whose rows are whole pages, 79% of which are
# longer than 512 and the longest of which is 4481 tokens. 8192 is the model's own
# limit (8194 positions, two of them sentinels), so with it nothing in this corpus
# is truncated at all: at the previous value of 1024 the tail of that 4481-token
# page, and of every other page past 1024, was dropped before the forward pass, and
# a dropped tail is text that is in the PDF, is shown to the reader, and cannot be
# searched for. Raising it re-embeds everything, because it is part of the source
# hash.
#
# It costs nothing on a row shorter than the old ceiling: a forward pass is sized by
# its own batch's padded length, not by this number. It does cost on the rows that
# were being cut, which now carry a whole page through attention. What it used to
# cost on top of that was batch size, because the batch was sized for this worst
# case rather than for its contents; see plan_batches, which sizes each call from
# the rows that are in it.
# The value itself lives in lib/encoder.py, because chunk.py needs the same number
# to split a section that would not fit; see that file.

# Batch sizing, all four numbers measured on this corpus with model_fp16.onnx on an
# RTX 3090 Ti, and the first one is the surprise: throughput does NOT keep climbing
# with the batch. 8 rows gives 151 chunks/s, 32 gives 218, 64 gives 220, and from
# there it falls back (214 at 128, 206 at 512) while the memory grows linearly. Two
# reasons, and neither is occupancy: the graph is already wide enough to fill the
# card at a few dozen rows, and every row in a batch is padded to the longest one in
# it, so a bigger batch buys more wasted padding. So "as large as the card allows" is
# the wrong rule, and BATCH_PLATEAU is where the measurement stops paying. It is a
# throughput ceiling and not a memory one, which is why it stays an upper bound no
# matter how much VRAM happens to be free.
BATCH_PLATEAU = 64
# What a batch costs beyond the session itself, and the reason a batch cannot be
# sized in rows alone. Two terms, both measured on this card with model_fp16.onnx,
# one row length per process so that the arena starts empty every time:
#
#     512 tokens x 64 rows ->  2048 MiB      32 MiB per row
#    1024 tokens x 64 rows ->  7168 MiB     112 MiB per row
#    2048 tokens x 32 rows -> 12288 MiB     384 MiB per row
#    4096 tokens x  8 rows -> 12288 MiB    1536 MiB per row
#    8192 tokens x  2 rows -> 12288 MiB    6144 MiB per row
#
# Doubling the row length roughly quadruples what the row costs, because attention
# materialises a rows x heads x length x length score tensor: the allocation the
# card refused when this was first run was 8589934592 bytes, which is exactly
# 16 rows x 16 heads x 4096 x 4096 x 2 bytes, and also exactly 4 rows at 8192. So
# the quadratic coefficient is not a fit, it is the tensor, and 96 bytes is three
# such buffers live at once (16 heads x 2 bytes x 3), which is what the table above
# measures once arena rounding is included.
#
# The linear term is the old measurement (2350 MiB flat to 64 rows, then 13.7 MiB
# per extra row of 244 tokens) and is deliberately left at its rounded-up value: it
# now over-estimates, since part of what it was measuring was the quadratic term,
# and over-estimating is the safe direction.
SESSION_VRAM_MIB = 2400
VRAM_PER_ROW_PER_TOKEN_MIB = 0.06
ATTENTION_BYTES_PER_ROW_PER_TOKEN2 = 96
# Of the free VRAM, how much this bake will plan to use. The rest is for the display
# server, for whatever else the operator starts while a 6-minute bake runs, and for
# the arena's own fragmentation.
VRAM_BUDGET = 0.70
# What to budget when there is no GPU to ask. A CPU run pays this in ordinary RAM,
# and 4 GiB is roughly what the old rule reserved on a well-provisioned card, which
# leaves the plateau binding for ordinary chunks and single-row calls for a page of
# several thousand tokens.
FALLBACK_BUDGET_MIB = 4096
# The floor under a computed budget. It is not a row: `plan_batches` always emits at
# least one row whatever the budget says, so this exists only so that a machine with
# less free VRAM than the session itself costs plans single-row calls rather than
# dividing by a negative number.
MIN_BUDGET_MIB = 512


def free_vram_mib() -> int | None:
    """Ask nvidia-smi how much video memory is free right now.

    Returns
    -------
    int or None
        Free VRAM in MiB on the first GPU, or None when there is no nvidia-smi to
        ask (no NVIDIA driver, a container without the tool, an AMD card).

    Notes
    -----
    nvidia-smi rather than a Python binding on purpose: this script already depends
    on onnxruntime and nothing else GPU-shaped, and pynvml/torch would be a large
    dependency to answer one integer. The failure mode of the subprocess (missing,
    slow, unparseable) is None, which the caller treats as "no GPU information",
    never as zero.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        try:
            return int(line.strip())
        except ValueError:
            continue
    return None


#: How many texts to measure the length of in one tokenizer call. The tokenizer
#: pads a batch to its longest row, so asking it for a whole 1000-page document at
#: once materialises a thousand lists as long as that document's longest page. The
#: forward pass never sees these rows, they exist only to be counted, so they are
#: taken in slices to keep the transient cost flat.
LENGTH_PROBE_ROWS = 256


def padded_lengths(texts: list[str], *, tokenizer: Any, ceiling: int) -> list[int]:
    """Count the tokens in each text the way the encoder will count them.

    Parameters
    ----------
    texts : list of str
        Passages WITH whatever prefix the encoder prepends. The prefix is a few
        tokens the batch has to carry like any other, and leaving it out here would
        under-measure every row by the same amount.
    tokenizer : tokenizers.Tokenizer
        The encoder's own tokenizer, already configured with truncation at the
        bake's ceiling. Any other one would measure a different number from the one
        the forward pass pays for.
    ceiling : int
        Upper bound applied after the tokenizer, mirroring `batch_feed`'s own
        `ids[:max_len]`. Truncation is normally enabled at this same number, so this
        is belt and braces rather than the load-bearing part.

    Returns
    -------
    list of int
        One length per text, in the same order.

    Notes
    -----
    The length is read off the attention mask rather than off `len(ids)`, because
    `tokenizer.json` ships with padding enabled: `ids` comes back padded to the
    longest row of whatever slice it was measured in, and counting that would make
    every row in a slice look as long as its longest, which is exactly the error the
    planner exists to avoid.

    This is a second tokenisation of every passage, since `encode` tokenises again
    on its own. It is cheap next to the forward pass (the tokenizer is Rust and runs
    over the batch in parallel) and it is what buys the planner real lengths instead
    of a worst case.
    """
    out: list[int] = []
    for start in range(0, len(texts), LENGTH_PROBE_ROWS):
        for encoding in tokenizer.encode_batch(texts[start:start + LENGTH_PROBE_ROWS]):
            out.append(min(sum(encoding.attention_mask), ceiling))
    return out


def row_cost_mib(width: int) -> float:
    """What one row of a batch costs on the card, at a given padded length.

    Parameters
    ----------
    width : int
        The batch's padded length in tokens, which is its longest row: every other
        row in the batch is padded up to it and costs the same.

    Returns
    -------
    float
        MiB per row, over and above the session. Linear plus quadratic, for the
        reason set out above the constants: attention is quadratic in the length,
        and at 8192 tokens the quadratic term is 92% of the answer.
    """
    attention = ATTENTION_BYTES_PER_ROW_PER_TOKEN2 * width * width / (1024 * 1024)
    return VRAM_PER_ROW_PER_TOKEN_MIB * width + attention


def memory_budget(free_mib: int | None) -> tuple[float, str]:
    """How much video memory one encoder call may use, and why that number.

    Parameters
    ----------
    free_mib : int or None
        Free VRAM in MiB, from `free_vram_mib`. None means there is no GPU to ask,
        which is also what a CPU run looks like.

    Returns
    -------
    (float, str)
        A budget in MiB, and a one-line explanation for the log.

    Notes
    -----
    The old rule was in rows: as many as fit a worst-case row, capped at the
    plateau. That left the card three quarters empty on a document of 200-token
    chunks, and it could not have survived the ceiling moving from 1024 to 8192,
    because a worst case eight times longer is a worst case sixty-four times more
    expensive. The budget here is spent by `plan_batches` against the lengths
    actually in hand, so the ceiling costs nothing until a row reaches for it.
    """
    if free_mib is None:
        return float(FALLBACK_BUDGET_MIB), (
            f"no GPU information (nvidia-smi absent), budgeting "
            f"{FALLBACK_BUDGET_MIB} MiB per call")
    budget = free_mib * VRAM_BUDGET - SESSION_VRAM_MIB
    if budget < MIN_BUDGET_MIB:
        return float(MIN_BUDGET_MIB), (
            f"{free_mib} MiB free is tight for this model; budgeting the floor of "
            f"{MIN_BUDGET_MIB} MiB per call and hoping")
    return budget, (f"{free_mib} MiB free allows {budget:.0f} MiB per call: "
                    f"{max(int(budget / row_cost_mib(MAX_PASSAGE_TOKENS)), 1)} rows at the "
                    f"{MAX_PASSAGE_TOKENS}-token ceiling, "
                    f"{min(BATCH_PLATEAU, int(budget / row_cost_mib(256)))} at 256 tokens")


def plan_batches(lengths: list[int], *, budget_mib: float,
                 plateau: int = BATCH_PLATEAU) -> list[int]:
    """Split one document's rows into encoder calls that fit the budget.

    Parameters
    ----------
    lengths : list of int
        Tokens per row, in the order the rows will be fed. Sorting them longest
        first (which the caller does, for padding reasons) makes the groups as tight
        as they can be, but nothing here requires or assumes it.
    budget_mib : float
        Video memory allowed for one call, from `memory_budget`.
    plateau : int, optional
        Hard upper bound on rows per call whatever the budget allows. Past a few
        dozen rows a bigger batch measures slower, so spending the rest of the budget
        on rows would buy memory pressure and lose throughput.

    Returns
    -------
    list of int
        Group sizes, summing to `len(lengths)`. Every entry is at least 1, including
        when one row is longer than the entire budget: there is no smaller call to
        make, and refusing to emit it would mean refusing to embed that page.

    Notes
    -----
    A group costs its row count times `row_cost_mib` of its PADDED length, which is
    the longest row in it, so admitting one long row makes every row already in the
    group more expensive, and quadratically so. That is why the test re-prices the
    whole group at the new width rather than adding a per-row cost to a running
    total: a running total would accept a group that does not fit.
    """
    sizes: list[int] = []
    index = 0
    while index < len(lengths):
        width = max(lengths[index], 1)
        size = 1
        while index + size < len(lengths) and size < plateau:
            grown = max(width, lengths[index + size])
            if (size + 1) * row_cost_mib(grown) > budget_mib:
                break
            width = grown
            size += 1
        sizes.append(size)
        index += size
    return sizes


#: Substrings that mark an allocator refusal in an onnxruntime exception. There is
#: no exception class to catch: the refusal arrives as a generic
#: onnxruntime.capi.onnxruntime_pybind11_state.Fail whose message is the only thing
#: separating it from a graph that cannot run at all, and retrying half a batch of a
#: graph that cannot run would only fail more slowly.
OUT_OF_MEMORY_MARKERS = ("Failed to allocate memory", "out of memory",
                         "CUDA_ERROR_OUT_OF_MEMORY", "cudaErrorMemoryAllocation")


def encode_group(encoder: Any, texts: list[str], *, max_len: int) -> tuple[np.ndarray, int]:
    """Encode one planned group, halving it if the allocator refuses.

    Parameters
    ----------
    encoder : onnx_embed.Encoder
        The loaded model. Its `passage_prefix` is applied here, as in a direct call.
    texts : list of str
        One planned group, all of them in a single encoder call unless that fails.
    max_len : int
        Token ceiling for the call.

    Returns
    -------
    (numpy.ndarray, int)
        The vectors in the order given, and how many times the group had to be
        halved. Zero is the ordinary case; anything else says `memory_budget`'s
        picture of what a row costs was optimistic on these rows, which the caller
        uses to lower the budget for the rest of the bake instead of paying the same
        failure again on the next group.

    Notes
    -----
    Why retry at all when the budget is computed up front: the memory model is a
    fitted curve, the card is shared with a desktop, and a full bake is 40 minutes.
    A model that is 10% optimistic on one unusual document should cost that document
    a couple of seconds rather than the whole run. Halving is safe because the
    vectors do not depend on the batch (see EMBED_FORMAT_VERSION 2), so a split
    group and an intact one produce the same bake.
    """
    size = len(texts)
    splits = 0
    while True:
        try:
            return np.vstack([
                encoder.encode(texts[start:start + size], prefix=encoder.passage_prefix,
                               batch_size=size, max_len=max_len)
                for start in range(0, len(texts), size)
            ]), splits
        except Exception as exc:
            if size <= 1 or not any(mark in str(exc) for mark in OUT_OF_MEMORY_MARKERS):
                raise
            size = max(size // 2, 1)
            splits += 1
            logger.warning(f"the card refused a batch of {len(texts)} rows; retrying at "
                           f"{size} ({exc.__class__.__name__})")


def load_sibling_module(path: Path, name: str) -> Any:
    """Import a module from the justelesRCP checkout by explicit path.

    Parameters
    ----------
    path
        Path to the `.py` file in the sibling checkout.
    name
        Name to register it under in `sys.modules`.

    Returns
    -------
    Any
        The imported module.

    Raises
    ------
    click.ClickException
        If the file is absent, which means the sibling checkout is missing or
        elsewhere; pass the matching option to point at it.

    Notes
    -----
    Only `onnx_embed` is borrowed. It holds the encoder, which must be shared so
    query and passage vectors come from identical weights, and it depends on
    nothing but onnxruntime, tokenizers and numpy.
    """
    if not path.is_file():
        raise click.ClickException(
            f"{path} not found. It lives in the sibling justelesRCP checkout; pass "
            "--onnx-embed to point elsewhere."
        )
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise click.ClickException(f"cannot import {path} as a module")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec_module because a module using dataclass-style machinery
    # looks itself up in sys.modules during class creation; without this the import
    # fails with an opaque AttributeError on None.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --- CPU plumbing -------------------------------------------------------------
# The CUDA half of this section is GONE: preload_cuda_libs, ensure_cudnn_visible and
# select_providers now live in ../justelesRCP/src/onnx_embed.py, the module this script
# already loads the encoder from, so the copies that used to sit here (and the comment
# explaining why they had to be copies) are no longer duplicated. Only the thread count
# stays local, because it is genuinely different work: see the docstring.
def physical_cores() -> int:
    """Count physical cores, not hardware threads.

    Returns
    -------
    int
        Number of distinct physical cores visible to this process, falling back to
        half the logical count, and to 1 if even that is unavailable.

    Notes
    -----
    This is the single biggest knob measured on this corpus, and the intuitive
    setting is the wrong one. On a CPU whose cores carry two hardware threads each, the int8
    encoder does 0.90 chunks/s with 12 intra-op threads and 2.99 with 6: giving
    onnxruntime every hardware thread makes it 3.3x SLOWER, which is 8.5 hours
    against 2.5 for the corpus. Quantised GEMM saturates a core's execution units
    and its L2 on its own, so a second thread on the same core only adds cache
    pressure and synchronisation.

    The sibling's `_resolve_intra_threads(-1)` returns the LOGICAL count, which is
    right for its workload (many small concurrent query encodes) and wrong for this
    one (one large batch at a time), so this script does not reuse it.
    """
    try:
        visible = set(os.sched_getaffinity(0))
    except AttributeError:
        visible = set(range(os.cpu_count() or 1))
    cores: set[tuple[str, str]] = set()
    current: dict[str, str] = {}
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                if int(current.get("processor", -1)) in visible:
                    cores.add((current.get("physical id", "0"), current.get("core id", "0")))
                current = {}
                continue
            key, _, value = line.partition(":")
            current[key.strip()] = value.strip()
    except (OSError, ValueError):
        return max(1, len(visible) // 2)
    return len(cores) or max(1, len(visible) // 2)


def claim_output_dir(destination_dir: Path):
    """Take an exclusive lock on a bake directory, or refuse to start.

    Parameters
    ----------
    destination_dir
        The per-variant directory this run will write `.npz` files into.

    Returns
    -------
    io.TextIOWrapper
        The open lock file. Keep the reference alive for the whole run: the lock
        lives on the open file descriptor, and closing it releases the lock.

    Raises
    ------
    click.ClickException
        If another bake already holds the directory.

    Notes
    -----
    This exists because it actually happened. Two bakes were started against
    `data/vectors/plain/` at once, and the damage was worse than the obvious
    wasted CPU. They halved each other's throughput by contending for the same
    physical cores (1.8 chunks/s each against 2.99 alone), they raced on
    `np.savez_compressed` writes to the same paths, and because the two processes
    had loaded different revisions of this file they computed *different*
    `src_hash` values for identical work, so whichever wrote last decided whether
    the file counted as cached. That last part is the nasty one: the directory
    ends up a mix of valid and stale cache keys with nothing to show it.

    `flock` rather than a pid file: the kernel drops the lock when the process
    dies, however it dies, so there is no stale-lock case to reason about and no
    liveness check to get wrong. The lock file itself is left behind, which is
    harmless; it is the flock that matters, not the file's existence.

    The failure is loud on purpose. The alternative, waiting for the other run,
    would leave an operator staring at a silent process for hours.
    """
    lock_path = destination_dir / ".bake.lock"
    handle = lock_path.open("w")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise click.ClickException(
            f"another bake already holds {destination_dir} (lock: {lock_path}). "
            "Two bakes into one directory halve each other's speed and race on "
            "the same files. Wait for it, or use a different --out-name to bake "
            "alongside it."
        ) from None
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def source_hash(texts: list[str], params: dict[str, Any]) -> str:
    """Hash the chunk texts and everything that changes their meaning.

    Parameters
    ----------
    texts
        Chunk texts of one document, in order.
    params
        Embedding parameters: model name, prefix, max tokens, variant.

    Returns
    -------
    str
        Hex digest gating the cached `.npz` for this document.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(json.dumps(params, sort_keys=True).encode("utf-8"))
    h.update(str(EMBED_FORMAT_VERSION).encode("utf-8"))
    for text in texts:
        h.update(b"\x00")
        h.update(text.encode("utf-8"))
    return h.hexdigest()


def bake_params(onnx_embed: Any, *, variant: str, weights: str,
                fields: tuple[str, ...]) -> dict[str, Any]:
    """Everything besides the text that goes into a document's `src_hash`.

    Parameters
    ----------
    onnx_embed
        The sibling encoder module (see `load_sibling_module`).
    variant, weights, fields
        The `--variant`, `--weights` and resolved `--meta-fields` of the bake.

    Returns
    -------
    dict
        The parameters `source_hash` mixes into the digest.

    Notes
    -----
    Read from the encoder's PROFILE rather than from a built encoder, so that
    `--check` can compute the same digest without loading a gigabyte of weights.
    They are the same values a built encoder reports: `Encoder.__init__` takes
    its pooling and prefixes from the same `_profile(model_name)` lookup, and the
    bake always builds it for `RUNTIME_MODEL`.
    """
    profile = onnx_embed._profile(onnx_embed.RUNTIME_MODEL)
    params = {
        "model": onnx_embed.RUNTIME_MODEL,
        "weights": weights or "profile default",
        "passage_prefix": profile["passage"],
        "pooling": profile["pooling"],
        "max_tokens": MAX_PASSAGE_TOKENS,
        "variant": variant,
    }
    # Part of the hash only where it changes the text, so plain and title bakes
    # written before the option existed still match their cached vectors.
    if variant in CURATED_VARIANTS:
        params["meta_fields"] = ",".join(fields)
    return params


def cached_digest(destination: Path) -> str | None:
    """The `src_hash` a vector file was baked from, or None when there is none to trust.

    An unreadable file counts as missing, with a warning: re-embedding one
    document is cheap, and refusing the whole bake for it is not.

    So does a file with no `chunks_hash`, silently. `build_index.py` refuses to
    pair vectors with chunks it cannot prove they were computed from, so a file
    written before that field existed (before 2026-10-03, or anything under
    data/grid/ baked earlier) is useless to it however current its `src_hash`
    is. Calling it stale HERE is what makes the next ordinary bake, `--check` in
    deploy.sh included, rebake it, instead of the index build refusing with no
    step in the chain that would fix it.
    """
    if not destination.exists():
        return None
    try:
        bake = np.load(destination)
        if "chunks_hash" not in bake.files:
            return None
        return str(bake["src_hash"])
    # BadZipFile is what a truncated archive raises, and it is none of the others.
    # Bakes are written atomically since 2026-10-03, but one killed before that
    # left exactly such a file, which crashed every later run here.
    except (KeyError, ValueError, OSError, zipfile.BadZipFile):
        logger.warning(f"unreadable cache for {destination.name}, treating it as missing")
        return None


def document_digests(files: list[Path], *, variant: str, manifest: dict[str, dict[str, str]],
                     fields: tuple[str, ...], params: dict[str, Any]):
    """Yield `(path, payload, texts, digest)` for each chunk file, in order.

    The one place the bake and `--check` agree on what a document's digest is:
    a second copy of these four lines would be the one that drifts.
    """
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        texts = passage_texts(payload, variant=variant, manifest=manifest, meta_fields=fields)
        yield path, payload, texts, source_hash(texts, params)


def load_manifest(path: Path, titles_other: Path | None = None) -> dict[str, dict[str, str]]:
    """Read `data/MANIFEST.tsv` into rows keyed by filename.

    Only the curated variants need it, so a missing file is not an error here: it is
    an error in `passage_texts`, where it can say which variant asked for it.

    Parameters
    ----------
    path
        The manifest.
    titles_other
        `data/TITLES_OTHER.tsv`, the other-language title of each document. When it
        is given and exists, each manifest row gains `title_other` and `lang_other`
        columns from it. Merging here rather than threading a second dictionary
        through `document_digests` and `passage_texts` keeps one source of "what this
        document is called", which is what the layouts read; it is also what makes an
        edited translation invalidate exactly that document's vectors, since the
        per-document hash covers the text the layout builds.

    Returns
    -------
    dict
        `{filename: {column: value}}`, empty when the file does not exist.
    """
    if not path.exists():
        return {}
    rows = manifest_io.read_by_file(path)
    if titles_other is not None and titles_other.exists():
        for name, row in manifest_io.read_by_file(titles_other).items():
            if name in rows:
                rows[name]["title_other"] = row.get("title", "")
                rows[name]["lang_other"] = row.get("lang", "")
    return rows


# The word a figure chunk opens with, by its `kind` and in its own language (French
# spacing before the colon). A described table reads "Plomb : seuil cible ..." and
# nothing in it says it is a table, while a reader types "tableau des seuils du
# plomb": 1278 of the 5838 table descriptions and 593 of the 1830 figure ones never
# name their kind in their first eight words. Unmeasured: no eval query has a figure
# for its answer, so this is the obvious fix rather than a measured one. The words
# and the separator are the corpus's (corpus.toml, [languages.<code>] figure_words
# and figure_separator; [figures] fallback_language for a blank or unknown one).
# They are NOT in the hashed bake params, so editing them leaves every vector file
# looking current: rebake the figure documents by hand after such an edit.
FIGURE_LANGUAGES: dict[str, dict] = corpus_config.section("languages")
FIGURE_FALLBACK_LANGUAGE: str = corpus_config.section("figures")["fallback_language"]


def chunk_body(chunk: dict[str, Any]) -> str:
    """A chunk's text as the encoder reads it: a figure chunk gains its kind word.

    Only the encoder's input changes. The text shown to the reader, and stored in the
    index, stays the description as written.
    """
    figure = chunk.get("figure")
    if not figure:
        return chunk["text"]
    language = FIGURE_LANGUAGES.get(figure.get("language", ""),
                                    FIGURE_LANGUAGES[FIGURE_FALLBACK_LANGUAGE])
    words = language["figure_words"]
    word = words.get(figure.get("kind", ""), words["other"])
    return f"{word}{language['figure_separator']}{chunk['text']}"


def passage_texts(payload: dict[str, Any], *, variant: str,
                  manifest: dict[str, dict[str, str]] | None = None,
                  meta_fields: tuple[str, ...] = ("title", "issuer", "year")) -> list[str]:
    """Build the strings actually handed to the encoder.

    Parameters
    ----------
    payload
        A parsed chunk file.
    variant
        `"plain"` embeds the chunk text as chunked. `"title"` prepends the
        document's filename stem, which is a rough stand-in for its title.
        `"meta"` prepends the CURATED title with its issuer and year, from
        `data/MANIFEST.tsv`. `"titleend"` and `"metaend"` put those same two
        labels AFTER the text instead, and `"both"` puts the curated label
        first and the filename stem last.

        Position is a real variable rather than a formatting preference, and it
        became one when the encoder changed. Snowflake arctic pooled on CLS, so
        the FIRST tokens dominated the vector and a prefix was the only sensible
        place for a label; every prefix measurement in DESIGN.md was taken that
        way. jina-embeddings-v5 pools on the LAST token, which inverts that: a
        suffix now sits where the pooled state is read. The two labels are not
        interchangeable either, and were measured pulling in opposite
        directions (DESIGN.md, "the curated title is a WORSE prefix than the
        filename"): the filename stem won passage retrieval because it reads
        like the keyword list a reader types, while the curated title, issuer
        and year won document retrieval. So `"both"` is the layout that expects
        the filename stem to want the influential end and the curated metadata
        to be enough help at the quiet one.
    manifest
        Rows from `load_manifest`, required by the `meta` variant only.
    meta_fields
        Which manifest columns the `meta` variant prepends. The first is the head of
        the sentence and the rest go in brackets after it, so ("title",) alone is a
        curated title and nothing else.

    Returns
    -------
    list[str]
        One string per chunk, in chunk order.

    Notes
    -----
    The `title` variant tests contextual retrieval: a chunk on page 60 of a long guide
    never repeats which disease the document is about, so a query naming the
    disease can only match it through the surrounding words. Prepending the title
    fixes that but spends tokens and pulls every chunk of a document toward the
    same point in the space, which can flatten within-document ranking. Whether
    it helps is an empirical question, which is why both variants exist rather
    than one being assumed.

    `meta` asks the same question of better metadata. The filename stem that
    `title` uses is what somebody's download saved ("INSEE logt enq&res2017"),
    while the manifest holds a title written by hand from the document's own title
    page, plus the issuing body and the year. If context is what helps, the curated
    context should help more; if the gain is really just the subject's name appearing
    somewhere, it will not.
    """
    chunks = payload["chunks"]
    if variant == "plain":
        return [chunk_body(c) for c in chunks]

    def filename_label() -> str:
        """The filename stem, punctuation opened up into spaces."""
        return Path(payload["file"]).stem.replace("_", " ").replace("-", " ")

    def manifest_row() -> dict[str, str]:
        """This document's curated row, or the error that names what is missing."""
        row = (manifest or {}).get(payload["file"])
        if row is None:
            raise click.ClickException(
                f"{payload['file']} has no row in the manifest, which the "
                f"{variant} variant needs. Run: uv run scripts/manifest.py"
            )
        return row

    def bracketed(row: dict[str, str], head: str) -> str:
        """`head` followed by the remaining meta fields in brackets.

        Blank cells print nothing rather than "(, 2017)": 42 rows have no issuer
        because a journal article has no issuing body, and that is a fact about the
        corpus, not a gap to paper over.
        """
        parts = [bit for bit in (row.get(field, "") for field in meta_fields[1:]) if bit]
        return f"{head} ({', '.join(parts)})" if parts else head

    def curated_label() -> str:
        """The curated title, with issuer and year in brackets after it."""
        row = manifest_row()
        return bracketed(row, row.get(meta_fields[0], "") or Path(payload["file"]).stem)

    def bilingual_label() -> str:
        """The curated title, then its translation, then issuer and year in brackets.

        "Titre francais / English title (INSEE, 2017)", so an English question meets
        English words at the head of every chunk of a French document and vice versa.
        The document's OWN title stays first, which keeps this string a superset of
        what the shipped `meta` bake feeds the encoder rather than a rewrite of it.
        """
        row = manifest_row()
        other = row.get("title_other", "")
        if not other:
            raise click.ClickException(
                f"{payload['file']} has no translated title, which the {variant} "
                "variant needs. Add a row to data/TITLES_OTHER.tsv."
            )
        head = row.get(meta_fields[0], "") or Path(payload["file"]).stem
        # A title with nothing to translate (35 of the 535 are a filename read off
        # the cover, "10803 2017 Article 3166") carries itself as its translation, and
        # printing it twice would spend tokens on nothing and tell the encoder the
        # words are twice as important as they are.
        return bracketed(row, head if other == head else f"{head} / {other}")

    def quoted_curated_label() -> str:
        """The curated title inside quotation marks, issuer and year after it.

        Straight double quotes, not typographic ones and not guillemets: of the 534
        curated titles, one already carries a typographic pair ("Putting on My Best
        Normal") and four carry guillemets, while none carries a straight quote, so
        this is the one pair that cannot be read as a title ending early.
        """
        row = manifest_row()
        head = row.get(meta_fields[0], "") or Path(payload["file"]).stem
        return bracketed(row, f'"{head}"')

    # One row per variant: what goes before the chunk, what goes after it, and what
    # separates a LEADING label from the chunk's first word.
    # Written as a table rather than a chain of branches so that adding a layout
    # cannot accidentally change an existing one, and so the shipped `meta` bake
    # keeps producing byte-identical strings (its src_hash must not move).
    #
    # The separator was swept rather than chosen. For the first year it was ". "
    # everywhere, which makes the label read as the passage's own opening sentence:
    # nothing told the encoder where the metadata stopped. `metanl`, `metaq` and
    # `metaqnl` are the other three cells of that 2x2 (quote the title or not, crossed
    # with ". " or a newline), and they measured as NO EFFECT, which is why ". " is
    # still what ships: the delimiter is one token out of a 249-token sequence.
    # DESIGN.md, "the label's delimiter", has the numbers. The cells stay here so a
    # re-measure is one command each if the label ever changes shape.
    layouts = {
        "title": (filename_label, None, ". "),
        "meta": (curated_label, None, ". "),
        "titleend": (None, filename_label, ". "),
        "metaend": (None, curated_label, ". "),
        "both": (curated_label, filename_label, ". "),
        "metabi": (bilingual_label, None, ". "),
        "metaq": (quoted_curated_label, None, ". "),
        "metanl": (curated_label, None, "\n"),
        "metaqnl": (quoted_curated_label, None, "\n"),
    }
    before, after, separator = layouts[variant]
    head = f"{before()}{separator}" if before else ""
    # A leading space before the trailing label, and a full stop after it, so the
    # label reads as its own sentence at either end and the chunk's own final
    # punctuation is never merged into it.
    tail = f" {after()}." if after else ""
    return [f"{head}{chunk_body(c)}{tail}" for c in chunks]


def meta_field_names(meta_fields: str, variant: str) -> tuple[str, ...]:
    """The manifest columns the meta variant prepends, as names.

    Parameters
    ----------
    meta_fields
        The raw `--meta-fields` value, a comma-separated list.
    variant
        The bake variant. Only `meta` reads these.

    Returns
    -------
    tuple of str
        The names, stripped, blanks dropped.

    Raises
    ------
    click.ClickException
        If the meta variant is asked for with nothing to prepend.

    Notes
    -----
    `--meta-fields ""` used to survive this far and fail inside the first document,
    as `meta_fields[0]` raising IndexError: a traceback out of the encoding loop,
    after the model had loaded, naming neither the option nor the mistake.
    """
    fields = tuple(f.strip() for f in meta_fields.split(",") if f.strip())
    if variant in CURATED_VARIANTS and not fields:
        raise click.ClickException(
            f"--meta-fields is empty, so the {variant} variant has nothing to add. "
            "Name at least one manifest column, or bake --variant plain."
        )
    return fields


@click.command()
@click.option("--chunks", type=click.Path(exists=True, file_okay=False, path_type=Path),
              **in_corpus("data/chunks"),
              help="Chunk files from chunk.py.")
@click.option("--out", type=click.Path(path_type=Path), **in_corpus("data/vectors"), help="Root for per-variant vector directories.")
@click.option("--onnx-embed", "onnx_embed_path", type=click.Path(path_type=Path),
              default=Path("../justelesRCP/src/onnx_embed.py"), show_default=True,
              help="The encoder the VPS runs. Shared on purpose, never copied.")
@click.option("--model-dir", type=click.Path(path_type=Path),
              default=Path("../justelesRCP/models/jinaai/jina-embeddings-v5-text-small-retrieval"),
              show_default=True, help="int8 ONNX weights plus tokenizer.json.")
@click.option("--manifest", "manifest_path", type=click.Path(path_type=Path),
              **in_corpus("data/MANIFEST.tsv"),
              help="Per-document metadata, read by the meta variant only.")
@click.option("--titles-other", "titles_other_path", type=click.Path(path_type=Path),
              **in_corpus("data/TITLES_OTHER.tsv"),
              help="Each document's title in the language it is not written in, read "
                   "by the metabi variant only.")
@click.option("--meta-fields", default="title,issuer,year", show_default=True,
              help="With --variant meta, the manifest columns to prepend, in order. "
                   "The first heads the sentence and the rest follow it in brackets.")
@click.option("--variant", type=click.Choice(
                  ["plain", "title", "meta", "titleend", "metaend", "both", "metabi",
                   "metaq", "metanl", "metaqnl"]),
              default="meta",
              show_default=True,
              help="What text gets embedded. See passage_texts for the tradeoff. The "
                   "default prepends the curated title, issuer and year, which is a "
                   "product decision rather than a measured one: DESIGN.md records "
                   "that the filename stem retrieves better on this eval set.")
@click.option("--weights", default="", show_default=False,
              help="ONNX file under <model-dir>/onnx/ to encode with. Empty means "
                   "the model profile's default, model_int8.onnx, which is what the "
                   "VPS embeds queries with. Pass model.onnx to bake with the fp32 "
                   "weights instead, which is what a GPU wants: the int8 graph has no "
                   "CUDA kernels for its quantised operators, so onnxruntime splits "
                   "the graph around them and the card buys almost nothing. "
                   "jina-embeddings-v5 publishes no fp16 graph, so fp32 is the GPU "
                   "artefact here; download-model.sh --keep-fp32 is what leaves it "
                   "on disk. Either way the bake is no longer byte-identical to the "
                   "query side, which is measured and fine (DESIGN.md).")
@click.option("--out-name", default="", show_default=False,
              help="Subdirectory of --out to write into. Defaults to --variant. Give "
                   "a distinct name (e.g. plain-fp16) when baking alternative "
                   "weights, so two bakes can be compared instead of overwriting.")
@click.option("--gpu/--no-gpu", default=False, show_default=True,
              help="Prefer a GPU provider when the installed onnxruntime exposes "
                   "one. OFF by default because it measured no faster: see the "
                   "note at the top of this file.")
@click.option("--batch-size", default="auto", show_default=True,
              help="Rows per encoder call, or 'auto' to read the free VRAM from "
                   "nvidia-smi and plan each call from the rows in it. Bigger is "
                   "not faster past a few dozen rows: see BATCH_PLATEAU. A number "
                   "here pins every call to that many rows, which is what comparing "
                   "a timing against an older bake wants.")
@click.option("--threads", type=int, default=0,
              help="onnxruntime intra-op threads. The default, 0, means one per "
                   "PHYSICAL core, which measured 3.3x faster than one per "
                   "hardware thread; see physical_cores.")
@click.option("--limit", type=int, default=0,
              help="Embed only the first N documents. For timing a change quickly.")
@click.option("--force", is_flag=True, help="Re-embed even when the content hash matches.")
@click.option("--check", is_flag=True,
              help="Report which documents the bake would re-embed, without loading the "
                   "model, and exit 1 when there are any. deploy.sh asks this before "
                   "baking.")
def main(chunks: Path, out: Path, onnx_embed_path: Path, model_dir: Path,
         manifest_path: Path, titles_other_path: Path, meta_fields: str, variant: str,
         weights: str, out_name: str, gpu: bool, batch_size: str, threads: int,
         limit: int, force: bool, check: bool) -> None:
    """Embed every chunk of every document at full model width."""
    # Resolved before anything expensive: an unparseable --batch-size should fail
    # now, not after the model has loaded. The batch size is deliberately NOT part
    # of the source hash: since EMBED_FORMAT_VERSION 2 the vectors no longer depend
    # on it (0.99999+ cosine from batch 8 to batch 512), so two bakes at different
    # sizes are interchangeable and re-running with 'auto' must not invalidate a
    # cache baked at 64.
    if batch_size.strip().lower() == "auto":
        budget_mib, why = memory_budget(free_vram_mib() if gpu else None)
        plateau = BATCH_PLATEAU
        logger.info(f"batch budget: {why}")
    else:
        try:
            plateau = int(batch_size)
        except ValueError:
            raise click.BadParameter("--batch-size takes an integer or 'auto'")
        if plateau < 1:
            raise click.BadParameter("--batch-size must be at least 1")
        # A number means exactly that number of rows per call, whatever is in them,
        # which is what timing one bake against an older one needs. An unbounded
        # budget hands plan_batches back the old fixed-size behaviour without a
        # second code path; the retry below still catches a card that says no.
        budget_mib = math.inf
        logger.info(f"batch size fixed at {plateau} rows per call")
    if threads <= 0:
        threads = physical_cores()
        logger.info(f"using {threads} intra-op threads (one per physical core)")
    onnx_embed = load_sibling_module(onnx_embed_path, "onnx_embed")
    destination_dir = out / (out_name or variant)
    fields = meta_field_names(meta_fields, variant)
    params = bake_params(onnx_embed, variant=variant, weights=weights, fields=fields)
    # Read once, before the loop, and only where it is used: the per-document hash
    # covers the text this produces, so a curated title edited in MANIFEST.tsv
    # invalidates that document's vectors and nothing else.
    manifest = (load_manifest(manifest_path, titles_other_path)
                if variant in CURATED_VARIANTS else {})
    if variant in CURATED_VARIANTS and not manifest:
        raise click.ClickException(
            f"the {variant} variant needs {manifest_path}, which does not exist. "
            "Run: uv run scripts/manifest.py"
        )
    if variant in BILINGUAL_VARIANTS and not titles_other_path.exists():
        raise click.ClickException(
            f"the {variant} variant needs {titles_other_path}, which does not exist."
        )

    files = sorted(chunks.glob("*.json"))
    if limit:
        files = files[:limit]

    if check:
        # deploy.sh's question: would the bake below re-embed anything? Answered
        # from the chunk text and the cached digests alone, so it costs seconds
        # rather than the model load, and it can never disagree with the bake
        # because it is the bake's own gate run without the encoder.
        stale = [path.name for path, _, _, digest
                 in document_digests(files, variant=variant, manifest=manifest,
                                     fields=fields, params=params)
                 if force or cached_digest(destination_dir / f"{path.stem}.npz") != digest]
        for name in stale[:10]:
            logger.info(f"needs baking: {name}")
        if len(stale) > 10:
            logger.info(f"... and {len(stale) - 10} more")
        if stale:
            logger.warning(f"{len(stale)} of {len(files)} documents need baking into {destination_dir}")
            sys.exit(1)
        logger.success(f"{destination_dir} is current: all {len(files)} documents baked")
        return

    destination_dir.mkdir(parents=True, exist_ok=True)
    # Claimed BEFORE the encoder is built: loading the model costs several seconds
    # and about a gigabyte of RAM, and a run that is about to be refused should
    # not pay for either. Held for the whole run, so the handle is bound to a name
    # rather than discarded: garbage collecting it would close the file and
    # release the lock. See claim_output_dir for what a second concurrent bake
    # into one directory actually costs.
    lock = claim_output_dir(destination_dir)

    if gpu:
        # Both must run BEFORE onnxruntime probes CUDA, which the provider list
        # below does, so the CUDA and cuDNN wheels become loadable in time.
        # Both may be no-ops; ensure_cudnn_visible may RE-EXECUTE this process with
        # LD_LIBRARY_PATH set, which is why it runs before the encoder is built.
        onnx_embed.ensure_cudnn_visible(log=logger.info)
        onnx_embed.preload_cuda_libs(log=logger.debug)
    providers, available = onnx_embed.select_providers(gpu)
    if gpu and providers[0] == "CPUExecutionProvider":
        logger.warning(f"no GPU provider registered (available: {available}). Falling "
                       "back to CPU, which measures 0.9 chunks/s on this corpus, about "
                       "8 hours for the whole thing. See the note at the top of this "
                       "file for supplying CUDA 12 + cuDNN 9 as wheels.")
    # out_dim=0 means "keep the model's native width". The site will serve 256;
    # storing 1024 is what makes the dimension sweep free, see the module
    # docstring.
    # `weights` names another ONNX file under <model-dir>/onnx/, leaving the rest of the
    # model profile (pooling, prefixes) alone. This used to monkeypatch the sibling's
    # private _profile, because the VPS only ever runs int8 and a second profile entry
    # would be a footgun there; Encoder now takes the override directly, so the hack is
    # gone. The filename goes into the source hash below either way, so switching weights
    # cannot silently reuse the old vectors.
    # The one combination that quietly wastes an afternoon. It is a warning and not
    # a resolved default, unlike embed-rcp.py: `weights` goes into the source hash,
    # so a value that depends on what hardware registered would make `--check` and
    # the bake disagree and call every document stale. deploy.sh passes it to both.
    if gpu and not weights:
        logger.warning("--gpu with the profile's int8 weights: its quantised operators "
                       "have no CUDA kernels, so the card buys almost nothing. Add "
                       "`--weights model.onnx` (../justelesRCP/scripts/download-model.sh "
                       "--keep-fp32 leaves that graph on disk).")

    encoder = onnx_embed.Encoder(
        model_dir=model_dir,
        model_name=onnx_embed.RUNTIME_MODEL,
        intra_threads=threads,
        providers=providers,
        # Only reaches encode_passages, which this script does not call: it plans
        # and slices its own batches below. Passed anyway so an encoder built here
        # and used another way does not silently fall back to the library default.
        passage_batch_size=plateau,
        out_dim=0,
        weights=weights or None,
        # Without this the tokenizer keeps tokenizer.json's own 512-token ceiling and
        # MAX_PASSAGE_TOKENS below is a ceiling that never applies. It changes nothing
        # for ordinary chunks (the longest is 360 tokens) and everything for the
        # whole-page rows of the page bake, 79% of which are longer than 512.
        max_tokens=MAX_PASSAGE_TOKENS,
    )
    logger.info(f"encoder ready: {encoder.model_name} at {encoder.dim} dims, "
                f"pooling={encoder.pooling}, variant={variant}, "
                f"providers={encoder.session.get_providers()}, "
                f"batch<={plateau} rows within {budget_mib:.0f} MiB")


    built = cached = 0
    vectors_written = 0
    # Reported once at the end rather than per document: what the planner does is
    # only legible in aggregate, and a line per document would bury the bake's own.
    groups = at_ceiling = widest_group = retried = 0
    narrowest_group = plateau
    started = time.perf_counter()
    digests = document_digests(files, variant=variant, manifest=manifest,
                               fields=fields, params=params)
    for position, (path, payload, texts, digest) in enumerate(digests, start=1):
        destination = destination_dir / f"{path.stem}.npz"
        if not force and cached_digest(destination) == digest:
            cached += 1
            continue

        if not texts:
            # chunk.py already fails the build on a document with no chunks, so
            # this is defensive: an empty array keeps the file set complete.
            matrix = np.zeros((0, encoder.dim), dtype=np.float32)
        else:
            # Longest first, then put the rows back in chunk order. Every row in a
            # batch is padded to the longest one in it, so feeding a document in its
            # own order pads short chunks up to whatever long chunk lands beside
            # them; sorting puts similar lengths together and measured 177 -> 204
            # chunks/s on 512 corpus passages at batch 64. It is only safe to
            # reorder because the vectors no longer depend on the batch (see
            # EMBED_FORMAT_VERSION 2): the same 512 passages come back with cosine
            # 0.999997 against themselves in either order.
            #
            # Sorted on the TOKEN length rather than on len(text), now that the
            # lengths are measured anyway for the batch planner: characters per
            # token vary by a factor of two or so between a table of dosages and a
            # paragraph of running French, so the character count was only ever a
            # proxy for the thing the batch actually pads and pays for.
            lengths = padded_lengths([encoder.passage_prefix + text for text in texts],
                                     tokenizer=encoder.tokenizer,
                                     ceiling=MAX_PASSAGE_TOKENS)
            order = sorted(range(len(texts)), key=lambda i: -lengths[i])
            blocks = []
            offset = 0
            for size in plan_batches([lengths[i] for i in order],
                                     budget_mib=budget_mib, plateau=plateau):
                block, splits = encode_group(
                    encoder, [texts[i] for i in order[offset:offset + size]],
                    max_len=MAX_PASSAGE_TOKENS)
                if splits:
                    # The budget was wrong about these rows, and the next group is
                    # drawn from the same document: carry the correction forward
                    # rather than rediscovering it group by group.
                    budget_mib = max(budget_mib / 2 ** splits, MIN_BUDGET_MIB)
                    retried += 1
                    logger.warning(f"batch budget lowered to {budget_mib:.0f} MiB for "
                                   "the rest of this bake")
                blocks.append(block)
                groups += 1
                widest_group = max(widest_group, size)
                narrowest_group = min(narrowest_group, size)
                offset += size
            at_ceiling += sum(1 for n in lengths if n >= MAX_PASSAGE_TOKENS)
            grouped = np.vstack(blocks)
            matrix = np.empty_like(grouped)
            matrix[order] = grouped
        # Through a temporary file (lib/atomic.py): a bake killed mid-write must
        # leave the previous good file or none, never a truncated archive.
        with atomic.open_atomic(destination, "wb") as handle:
            np.savez_compressed(
                handle,
                vectors=matrix,
                src_hash=np.array(digest),
                # The chunk file's own `src_hash`, so build_index.py can refuse to pair
                # these vectors with chunks other than the ones they were computed
                # from. `src_hash` above cannot serve: it is THIS bake's cache key and
                # mixes in the model and the label, which build_index.py cannot
                # recompute without the encoder's profile.
                chunks_hash=np.array(payload["src_hash"]),
                # Named "document", not "file": savez_compressed's own first
                # parameter is called `file`, so a `file=` keyword here is a
                # TypeError rather than an array in the archive.
                document=np.array(payload["file"]),
                model=np.array(encoder.model_name),
                variant=np.array(variant),
                weights=np.array(weights or "profile default"),
            )
        built += 1
        vectors_written += len(matrix)
        elapsed = time.perf_counter() - started
        logger.info(f"[{position}/{len(files)}] {payload['file']}: {len(matrix)} vectors "
                    f"({vectors_written / max(elapsed, 1e-9):.1f} chunks/s overall)")

    logger.success(f"{built} documents embedded, {cached} unchanged, "
                   f"{vectors_written} vectors written to {destination_dir}")
    if built:
        logger.info(f"took {time.perf_counter() - started:.0f}s")
    if groups:
        logger.info(f"{groups} planned batches, {vectors_written / groups:.1f} rows each "
                    f"on average, between {narrowest_group} and {widest_group}; "
                    f"{at_ceiling} rows reached the {MAX_PASSAGE_TOKENS}-token ceiling "
                    f"and were cut there; {retried} batches had to be split for memory")


if __name__ == "__main__":
    main()
