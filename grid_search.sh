#!/usr/bin/env bash
# Run the whole chunking sweep, unattended, and be safe to stop at any moment.
#
# The question: which chunking retrieves best for THIS corpus. The sweep answers it
# by building one index per strategy, dumping what a reader would actually be shown
# for each of the eval queries, and grading every retrieved passage with a
# cross-encoder far too big to serve but perfectly usable offline on the 3090 Ti
# (scripts/judge_rerank.py). Nothing here touches dist/ or the VPS: the whole run
# lives under data/grid/ and the only shipped artefacts it reads are the page bake
# and the manifest.
#
# Four stages per strategy, then two global ones:
#
#   chunk   scripts/chunk.py into data/grid/<name>/chunks
#   embed   scripts/embed.py on the GPU, fp32 weights, into <name>/vectors/bake
#   index   scripts/build_index.py into <name>/index, int8 at 1024 dims
#   dump    scripts/dump_candidates.mjs: the shipped ranker's top K, with text
#   judge   scripts/judge_rerank.py over every dump at once, deduplicated
#   report  scripts/grid_report.py: data/GRID_RESULTS.tsv
#
# STOPPING AND RESUMING. Ctrl-C is safe at any point, and so is killing the box's
# power. Every stage is either content-hash gated by the script itself (chunk,
# embed) or skipped when its output file exists (index, dump), and the judge appends
# each verdict to data/grid/judge-scores.jsonl as it is produced and reads it back on
# the next run. So re-running this script after an interruption picks up where it
# stopped, and re-running it after everything is done costs about a minute of hash
# checks. Nothing is ever recomputed to be sure.
#
# WHAT IT NEEDS BEFORE IT STARTS:
#   - the corpus in data/GUIDELINES/ and data/OCR/ (as for any build)
#   - the page bake, which every index's 15% page term reads and which no strategy
#     changes: data/chunks-page and data/vectors/page-meta-gpu
#   - the fp32 GPU weights, ../justelesRCP/models/.../onnx/model.onnx
#   - an encoder answering at $EMBED_URL, for the FIRST strategy only: query vectors
#     are cached on disk afterwards (data/grid/query-vectors.json)
#   - a free GPU for the judge (about 9.5 GiB at bf16, or 4 with --quant 4bit)
#
# Usage:
#   ./grid_search.sh                 # everything, resuming whatever is already there
#   ./grid_search.sh --list          # the strategy table, and what is already built
#   ./grid_search.sh --only t256-o26-sec
#   ./grid_search.sh --stop-after index     # build everything, judge later
#   ./grid_search.sh --stage judge   # just the judge, then the report
#   ./grid_search.sh --prune         # delete a strategy's vectors once it is dumped
#   ./grid_search.sh --docs /tmp/ten.txt --only t1024-o256-sec   # smoke the chain
#
# Written by Claude Code (Opus 5).
set -euo pipefail
# Everything the sweep reads and writes is corpus data under $CORPUS_DIR
# (scripts/lib/corpus.py). Made absolute BEFORE the cd below, so a relative
# CORPUS_DIR means what it meant where the script was started, and exported so
# every script it runs resolves its own defaults against the same corpus.
: "${CORPUS_DIR:?CORPUS_DIR is not set: it names the corpus directory, e.g. local/psydocs}"
CORPUS_DIR=$(realpath "$CORPUS_DIR")
export CORPUS_DIR
cd "$(dirname "$0")"

GRID=$CORPUS_DIR/data/grid
QUERIES=${QUERIES:-$CORPUS_DIR/data/EVAL_QUERIES.tsv}
EMBED_URL=${EMBED_URL:-http://127.0.0.1:8461}
TOPK=${TOPK:-10}
DIMS=1024
QUANT=int8
# The page term every index blends in. Chunking-independent (one vector per PDF page)
# and therefore built once, outside this sweep, which is also what keeps the strategies
# comparable: only the passage half of the score changes between them.
PAGE_CHUNKS=$CORPUS_DIR/data/chunks-page
PAGE_VECTORS=$CORPUS_DIR/data/vectors/page-meta-gpu
WEIGHTS=model.onnx
VARIANT=meta

ONLY=""
STOP_AFTER=""
STAGE=""
FORCE=0
PRUNE=0
JUDGE_ARGS=""
# A file of PDF names, passed straight to chunk.py --only. The whole chain over ten
# documents runs in a couple of minutes and proves the wiring before a night is
# committed to it; the numbers it produces mean nothing, which is the point.
DOCS=""

usage() { sed -n '2,47p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    --docs) DOCS="$2"; shift 2 ;;
    --stop-after) STOP_AFTER="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --prune) PRUNE=1; shift ;;
    --topk) TOPK="$2"; shift 2 ;;
    --judge-args) JUDGE_ARGS="$2"; shift 2 ;;
    --list) LIST=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage 1 ;;
  esac
done

# The strategy table: name | chunk.py arguments | build_index.py arguments | reuse.
#
# `reuse` names another strategy whose chunks AND bake this one shares, for rows that
# only change how the index is built: those cost two minutes instead of half an hour.
#
# t170-512-sec is the chunking that ships today (chunk.py's defaults since
# 2026-09-30: a cut at a section boundary once 170 tokens are behind it, 512 a
# ceiling, no overlap) and the baseline every other row is read against. The four
# targets and two overlaps after it are the grid the brief asks for; t1024-o256-sec
# shipped before 2026-09-30, which is why the single-variable rows after it are
# still phrased as changes to it rather than to the baseline: one
# chunk per page, one per section, the heading prefix that DESIGN.md left unmeasured,
# the pre-0.1.0 greedy packing, table detection off, and the same bake stored as one
# bit per dimension instead of int8.
#
# The two t170-512-o51 rows ask whether overlap helps the shipped window. The
# first carries 51 tokens only across a cut the budget forced (what
# --overlap-tokens has always done), the second across every cut, so a section's
# first chunk opens on the end of the previous one (--overlap-at-boundaries).
# Read against t170-512-sec, the first says how much the rare forced cut costs,
# the second whether bleeding across section boundaries pays for the dilution
# when the page vector already covers the neighbourhood.
strategies() {
  cat <<'EOF'
t170-512-sec|--target-tokens 512 --overlap-tokens 0||
t170-512-o51-sec|--target-tokens 512 --overlap-tokens 51||
t170-512-o51-all-sec|--target-tokens 512 --overlap-tokens 51 --overlap-at-boundaries||
t256-o26-sec|--target-tokens 256 --overlap-tokens 26||
t256-o51-sec|--target-tokens 256 --overlap-tokens 51||
t256-o26-para|--target-tokens 256 --overlap-tokens 26 --no-section-first||
t256-o51-para|--target-tokens 256 --overlap-tokens 51 --no-section-first||
t450-o45-sec|--target-tokens 450 --overlap-tokens 45||
t450-o90-sec|--target-tokens 450 --overlap-tokens 90||
t450-o45-para|--target-tokens 450 --overlap-tokens 45 --no-section-first||
t450-o90-para|--target-tokens 450 --overlap-tokens 90 --no-section-first||
t768-o77-sec|--target-tokens 768 --overlap-tokens 77||
t768-o154-sec|--target-tokens 768 --overlap-tokens 154||
t768-o77-para|--target-tokens 768 --overlap-tokens 77 --no-section-first||
t768-o154-para|--target-tokens 768 --overlap-tokens 154 --no-section-first||
t1024-o102-sec|--target-tokens 1024 --overlap-tokens 102||
t1024-o205-sec|--target-tokens 1024 --overlap-tokens 205||
t1024-o102-para|--target-tokens 1024 --overlap-tokens 102 --no-section-first||
t1024-o205-para|--target-tokens 1024 --overlap-tokens 205 --no-section-first||
t1024-o256-sec|--target-tokens 1024 --overlap-tokens 256||
page|--page-chunks|--page-weight 0|
section|--section-chunks||
t1024-o256-headings|--target-tokens 1024 --overlap-tokens 256 --headings||
t1024-o256-greedy|--target-tokens 1024 --overlap-tokens 256 --no-boundaries||
t1024-o256-notables|--target-tokens 1024 --overlap-tokens 256 --no-tables||
t1024-o256-sec-b1|--target-tokens 1024 --overlap-tokens 256|--quant binary|t1024-o256-sec
EOF
}

field() { echo "$1" | cut -d'|' -f"$2"; }
names() { strategies | cut -d'|' -f1; }

if [ "${LIST:-0}" -eq 1 ]; then
  printf '%-22s %-9s %-9s %-9s %s\n' strategy chunks vectors index candidates
  while IFS= read -r line; do
    name="$(field "$line" 1)"
    d="$GRID/$name"
    printf '%-22s %-9s %-9s %-9s %s\n' "$name" \
      "$([ -d "$d/chunks" ] && ls "$d/chunks" | wc -l || echo -)" \
      "$([ -d "$d/vectors/bake" ] && echo yes || echo -)" \
      "$([ -f "$d/index/meta.json" ] && echo yes || echo -)" \
      "$([ -f "$d/candidates.json" ] && echo yes || echo -)"
  done < <(strategies)
  exit 0
fi

mkdir -p "$GRID"
# One sweep at a time. Two would fight over the GPU and, worse, over embed.py's own
# flock, which would fail the second run mid-bake rather than at the start.
exec 9>"$GRID/.lock"
if ! flock -n 9; then
  echo "another ./grid_search.sh is running ($GRID/.lock)" >&2
  exit 1
fi

TIMINGS="$GRID/timings.tsv"
[ -f "$TIMINGS" ] || printf 'when\tstrategy\tstage\tseconds\n' > "$TIMINGS"

# Elapsed time per stage, appended, so the second run of this sweep can be planned
# from the first one's numbers rather than from a guess.
timed() {
  local name="$1" stage="$2"; shift 2
  local started; started=$(date +%s)
  "$@"
  printf '%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "$name" "$stage" "$(( $(date +%s) - started ))" >> "$TIMINGS"
}

TOTAL=$(names | wc -l)
[ -n "$ONLY" ] && TOTAL=1
DONE=0

# A bar rather than a percentage: this runs for hours in a terminal somebody glances
# at, and "where am I" has to be readable without reading.
bar() {
  local done=$1 total=$2 width=30 filled
  filled=$(( done * width / total ))
  printf '\n[%s%s] %d/%d  %s\n' \
    "$(printf '#%.0s' $(seq 1 $filled) 2>/dev/null || true)" \
    "$(printf '.%.0s' $(seq 1 $(( width - filled )) ) 2>/dev/null || true)" \
    "$done" "$total" "$3"
}

stage_wanted() { [ -z "$STAGE" ] || [ "$STAGE" = "$1" ]; }
past_stop() {
  case "$STOP_AFTER" in
    "") return 1 ;;
    chunk) [ "$1" != chunk ] ;;
    embed) [ "$1" != chunk ] && [ "$1" != embed ] ;;
    index) [ "$1" = dump ] || [ "$1" = judge ] || [ "$1" = report ] ;;
    dump) [ "$1" = judge ] || [ "$1" = report ] ;;
    *) return 1 ;;
  esac
}

build_one() {
  local line="$1"
  local name chunk_args index_args reuse dir chunks vectors
  name="$(field "$line" 1)"
  chunk_args="$(field "$line" 2)"
  index_args="$(field "$line" 3)"
  reuse="$(field "$line" 4)"
  dir="$GRID/$name"
  chunks="$dir/chunks"
  vectors="$dir/vectors/bake"
  # A reusing row shares the chunks and the bake of the row it names, so only its
  # index differs. Nothing is copied: both are read where they already are.
  if [ -n "$reuse" ]; then
    chunks="$GRID/$reuse/chunks"
    vectors="$GRID/$reuse/vectors/bake"
    if [ ! -d "$vectors" ]; then
      echo "$name reuses $reuse, which is not built yet; skipping" >&2
      return 0
    fi
  fi
  mkdir -p "$dir"

  bar "$DONE" "$TOTAL" "$name"

  if [ -z "$reuse" ] && stage_wanted chunk && ! past_stop chunk; then
    echo "-- chunk"
    # Hash gated inside chunk.py: a second run over unchanged PDFs and unchanged
    # parameters reads 483 hashes and writes nothing.
    # shellcheck disable=SC2086
    timed "$name" chunk uv run scripts/chunk.py --out "$chunks" $chunk_args \
      $([ -n "$DOCS" ] && echo --only "$DOCS") \
      $([ "$FORCE" -eq 1 ] && echo --force)
  fi

  if [ -z "$reuse" ] && stage_wanted embed && ! past_stop embed; then
    echo "-- embed"
    # shellcheck disable=SC2086
    timed "$name" embed uv run scripts/embed.py --chunks "$chunks" \
      --out "$dir/vectors" --out-name bake --variant "$VARIANT" \
      --weights "$WEIGHTS" --gpu $([ "$FORCE" -eq 1 ] && echo --force)
  fi

  if stage_wanted index && ! past_stop index; then
    if [ "$FORCE" -eq 1 ] || [ ! -f "$dir/index/meta.json" ]; then
      # --no-reference-axis on a subset run: the axis is an average over the corpus's
      # reference lists and ten documents do not have enough of them to average, so
      # build_index refuses. A full run builds it as usual.
      echo "-- index"
      # shellcheck disable=SC2086
      timed "$name" index uv run scripts/build_index.py --chunks "$chunks" \
        --vectors "$vectors" --out "$dir" --dims "$DIMS" --quant "$QUANT" \
        --page-chunks "$PAGE_CHUNKS" --page-vectors "$PAGE_VECTORS" $index_args \
        $([ -n "$DOCS" ] && echo --no-reference-axis)
    else
      echo "-- index: already built"
    fi
  fi

  if stage_wanted dump && ! past_stop dump; then
    if [ "$FORCE" -eq 1 ] || [ ! -f "$dir/candidates.json" ]; then
      echo "-- dump"
      timed "$name" dump env EMBED_URL="$EMBED_URL" INDEX_DIR="$dir/index" \
        OUT="$dir/candidates.json" QVEC_CACHE="$GRID/query-vectors.json" \
        QUERIES="$(realpath "$QUERIES")" TOPK="$TOPK" STRATEGY="$name" \
        node scripts/dump_candidates.mjs
    else
      echo "-- dump: already there"
    fi
  fi

  # The bake is the big artefact (a gigabyte at the small targets) and nothing reads
  # it once the index is built. Off by default because rebuilding one costs GPU
  # minutes and disk is cheaper than that, but a sweep of 24 strategies is ~10 GB.
  if [ "$PRUNE" -eq 1 ] && [ -f "$dir/candidates.json" ] && [ -z "$reuse" ]; then
    rm -rf "$dir/vectors"
    echo "-- pruned $dir/vectors"
  fi
  DONE=$(( DONE + 1 ))
}

trap 'printf "\n\nstopped. Re-run ./grid_search.sh to resume where it left off.\n"; exit 130' INT TERM

case "${STAGE:-all}" in
  judge|report) ;;
  *)
    while IFS= read -r line; do
      name="$(field "$line" 1)"
      [ -n "$ONLY" ] && [ "$name" != "$ONLY" ] && continue
      build_one "$line"
    done < <(strategies)
    bar "$DONE" "$TOTAL" "built"
    ;;
esac

if stage_wanted judge && ! past_stop judge && [ -z "$ONLY" ]; then
  printf '\n=== judging every dump at once (deduplicated across strategies)\n'
  # shellcheck disable=SC2086
  timed all judge uv run scripts/judge_rerank.py --grid "$GRID" $JUDGE_ARGS
fi

if stage_wanted report && ! past_stop report; then
  printf '\n=== report\n'
  timed all report uv run scripts/grid_report.py --grid "$GRID"
fi
