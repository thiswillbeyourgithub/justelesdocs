# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**justelesdocs** serves a curated corpus of PDFs as a fast, ad-free static site with multilingual semantic search over the whole corpus. It is generic software: the documents, their metadata and everything specific to one deployment live in a separate *corpus* directory named by `CORPUS_DIR`.

It is the sister project of **`../justelesRCP`** and deliberately reuses its architecture, its theme, and (on a server) its embedding model container. Read `../justelesRCP/CLAUDE.md` before designing anything here: most questions about hardening, Caddy, compose layout, deploy flow and the `.vec.json` content-hash gate are already answered there.

It is **not RAG**. There is no LLM in the serving path, only embeddings and nearest-neighbour search.

What it does that justelesRCP does not:

- Search runs across **all chunks of all documents at once**.
- A hit is presented as **the PDF page it came from, with the matching chunk highlighted**. Several chunks on one page get several highlights plus an indicator of which ranks best.
- The reader can go from "one highlighted page" to "the whole PDF" easily and **download the original PDF**, except for documents in the `access: restricted` tier, which are served one page at a time and never handed over whole.
- Documents carry **metadata** used as **search filters**, from closed vocabularies the corpus declares. Multi-valued fields match on any value; year is a range slider.
- The UI is **bilingual French and English**, and a corpus can overlay the strings.

`ARCHITECTURE.md` is the map; `DESIGN.md` holds the decisions. **Read `DESIGN.md` before changing anything about chunking, embedding or the index**: most of it is measured rather than assumed, and the measurements are the reason the defaults are what they are.

## The corpus directory

`CORPUS_DIR` is REQUIRED and has no default (`scripts/lib/corpus.py`, and `server/lib/corpus.mjs` for the Node scripts). Every `data/...` and `dist/...` path below is resolved inside it. A corpus holds:

- `corpus.toml`: `[site]`, `[facets]`, `[vocabularies.*]`, `[tiers]`, `[renditions]`, `[languages.*]`, `[figures]`, `[ocr]`, `[manifest]`, `[judge]`. `corpus.example/corpus.toml` is the neutral, commented example, and the test suite runs against it.
- `data/GUIDELINES/`: the only servable documents. `data/MANIFEST.tsv` and the optional curated tables beside it.
- optionally `strings/<lang>.json` (overlays of `src/i18n.js`), `scenarios.json` (what the gates expect of this corpus, read by `scripts/lib/scenarios.mjs`, with fallbacks derived from `meta.json`), `changelog/` and `VERSION` (which replace the software's in `stage.py`), `tests/` (run after the software's), and its own deploy script and server env file.

Corpora conventionally live in `local/<name>/`, which is gitignored here, as their own repositories. A corpus's own `CLAUDE.md` holds what is specific to it. Never put a corpus's names, counts or documents in this repository's code or docs.

**Only `$CORPUS_DIR/data/GUIDELINES/` may ever be served, indexed or shipped.** A corpus directory may hold other documents (rejected, unsure, not redistributable), and none of them may leave the build machine. Two consequences for the code:

1. Any indexer must take `data/GUIDELINES/` as its explicit root. Never glob `data/**` for PDFs.
2. A corpus may hold several renditions of the same document (a summary, the full text, a long supporting report). They retrieve against each other, and the long ones dominate by volume. `family` and `rendition` in the manifest, with the rendition policy in `corpus.toml`, are how they are collapsed.

## Hard constraints

**The server is small: no GPU, little RAM, little disk, and it must serve many users without being upgraded.** This is the dominant constraint. It rules out running heavy inference per request and rules out storing an unquantised float32 index. A serving reranker was tried and crashed a server by exhausting its RAM; a reranker exists only as the offline judge.

**One embedding model on the server, shared with justelesRCP.** The model must be **multilingual**, so that a query in one language can match a passage in another. justelesRCP runs one int8 ONNX encoder behind an `embed` container and this site borrows it (currently `jinaai/jina-embeddings-v5-text-small-retrieval`, last-token pooling, 1024 dims). The optional compose `embed` profile builds the same encoder from a justelesRCP checkout for a host without the sibling.

**Embeddings are computed locally and shipped, not computed on the server.** The build is content-hash gated so nothing is recomputed without reason: hash of chunk text, plus embedding model identity, plus chunking parameters, plus an index format version.

**Static-first.** Plain HTML is preferred over JavaScript wherever it can do the job. Dependencies are vendored to avoid supply-chain exposure, with tooling to make periodic upgrades manageable. Minimal attack surface, hardened containers, strict CSP.

**Correctness guarantees**: no PDF silently skipped, no PDF ending up with zero chunks, no empty chunks. These belong in gates and tests, and the build fails loudly rather than ship a corpus with a hole in it.

## Conventions

Consistent across `../justelesRCP`, `../parakeet_web` and `../WebSend_git`:

- Python run through `uv run`, single-file scripts using PEP 723, in `scripts/` (build-time tools, not the served application).
- `docker/` holding `docker-compose.yml`, one Dockerfile per service, `Caddyfile`, `entrypoint.sh`, and a committed `env.example` whose real counterpart `docker/.env` is gitignored. The software ships no embedder of its own; the `embed` profile reuses the sibling's Dockerfile. DESIGN.md, "Embedding service", explains why.
- A root `.dockerignore`, written as a deny-all allow-list, because the search image builds from the repo root and an exclude list would fail open.
- Deployment is per corpus: the software ships no deploy script. A corpus's `deploy.sh` transfers an allow-list (`dist/`, `docker/`, `server/`, the `src/` modules the search image copies, the root `.dockerignore`), never a whole tree with excludes.
- `ARCHITECTURE.md`, and bilingual `README.md` / `README.en.md` kept rigorously in sync.
- `docs/changelog/<version>/changelog.md` per release, with the number in `VERSION`. `scripts/changelog.py` compiles the notes and `stage.py` refuses to stage a version that has none. A corpus's own `changelog/` and `VERSION` replace these.
- `.githooks/pre-push`. Activate it per clone with `git config core.hooksPath .githooks`; its first job is refusing to push a corpus document.

**Language:** UI strings are bilingual French and English. Code, comments and developer docs are in English.

**Hardening posture** (from `../justelesRCP/CLAUDE.md`): Caddy on plain HTTP with TLS terminated upstream, `read_only: true`, `cap_drop: ALL` paired with `cap_add: [NET_BIND_SERVICE]`, `no-new-privileges`, tmpfs for scratch dirs, strict `default-src 'self'` CSP with no CDNs or external fonts, per-IP rate limiting at the edge via an xcaddy build with the `caddy-ratelimit` plugin, detailed stats endpoints blocked at the edge.

**Services are separate containers** (Caddy, the search service, the page service) so the web server can stay fully read-only, each with narrow read-only bind mounts and run as the host UID.

## Commands

Every script is a PEP 723 single file, run with `uv run`. Set the corpus first, for example `export CORPUS_DIR=local/<name>`. There is no linter config: do not invent one.

The test suite is `uv run tests/run.py`: pytest over the scripts and `node --test` over `src/` and `server/`, against `corpus.example/`, installing nothing permanently and needing neither a corpus nor a built index; then `$CORPUS_DIR/tests` when the variable is set. `env -u CORPUS_DIR uv run tests/run.py` runs the software's tests alone. `.githooks/pre-push` runs it on every push, and skips its corpus gates when `CORPUS_DIR` is unset.

Beside the tests there are hard gates, which refuse rather than warn: `verify_chunks.py`, `check_served.py` and `check_search.mjs` in the build chain (and on every `git push` when `CORPUS_DIR` is set), `check_ui.mjs` and `check_chrome.mjs` in a browser, by hand, after any change to `src/`, and `smoke_deployed.sh` against the running site, which a deploy script should run over ssh at the end of every deploy: nothing that runs on the build machine can see whether the deployed stack can reach the encoder.

The build chain, in the order it must run. Each step is content-hash gated, so re-running the whole chain with nothing changed takes under a minute:

```sh
uv run scripts/check_pdfs.py   # hard gate: every file the site would serve is a PDF a browser opens.
                               # First because everything below believes the corpus: pymupdf renders
                               # HTML too, so an error page saved as a PDF would be chunked and served.
uv run scripts/manifest.py     # data/MANIFEST.tsv: per-document metadata. Fills BLANKS ONLY, never
                               # overwrites a curated value, so it is safe to re-run. --dry-run to look first.
uv run scripts/ocr.py          # data/OCR/: text layers for documents with scanned pages (--dry-run counts)
uv run scripts/chunk.py        # data/chunks/: chunks with page + bbox provenance (--force ignores the hash gate)
                               # 170 to 512 tokens, no overlap: cut at a SECTION boundary once 170 tokens
                               # are behind it, 512 a ceiling (DESIGN.md, "a 170 to 512 token window" and
                               # "the chunk is 256 tokens, not 1024"). Tables with no inner rules are read
                               # from the pymupdf-layout model's grid, ~1.2 s a page, cached per PDF sha256
                               # in data/private/layout/, so the first run on a new corpus is slow.
                               # It also appends one chunk per figure described in data/private/figures/
                               # (scripts/figures.py: census, extract, emit, apply). --no-figures to leave out.
uv run scripts/verify_chunks.py  # hard gate: no skipped PDF, no empty document, no undrawable box
uv run scripts/embed.py --out-name meta-gpu --weights model.onnx --gpu
                               # data/vectors/meta-gpu/: the shipped chunk bake. --variant defaults to
                               # meta, which prepends the curated title, issuer and year BEFORE the chunk
                               # (DESIGN.md, "a metadata label belongs BEFORE the passage").
uv run scripts/build_index.py --dims 1024
                               # dist/index/: the FULL index, read by the search service (bind-mounted
                               # read-only) and by the gates, never served as a whole. --quant defaults
                               # to binary; int8 retrieves better and costs several times the disk, RAM
                               # and rank time (DESIGN.md, "binary is what ships").
uv run scripts/stage.py        # dist/www/: the served tree. OPEN PDFs (symlinked) + MANIFEST.tsv + src/ +
                               # vendor/ + changelog.json + app-version.js + the index's PUBLIC half.
                               # RESTRICTED PDFs go to dist/restricted/, a sibling of dist/www/, and their
                               # per-document JSON is published without its text.
uv run scripts/check_served.py # hard gate: no restricted document, and no word of one, under dist/www/
node scripts/check_search.mjs  # hard gate: runs src/search.js against the real dist/index/
```

`build_index.py` also needs the PAGE bake, which only changes when the PDFs do. Build it once, and again whenever a document is added:

```sh
uv run scripts/chunk.py --page-chunks --out data/chunks-page
uv run scripts/embed.py --chunks data/chunks-page --out-name page-meta-gpu --weights model.onnx --gpu
```

It is what the 15% page term in every score comes from. `build_index.py` refuses to build without it; `--page-weight 0` is the way to say you meant it.

`build_index.py` needs a bake, which `stage.py` does not, so restaging after an edit to `src/` is milliseconds and does not touch the index. `stage.py` REFUSES to stage when the version has no release notes. `uv run scripts/changelog.py --check` runs that parse on its own, and `docs/changelog/README.md` has the format.

```sh
uv run scripts/catalog.py            # writes $CORPUS_DIR/CATALOG.md, the public list of what is served
uv run scripts/catalog.py --check    # what .githooks/pre-push runs; refuses a stale one
uv run scripts/vendor.py             # re-fetch vendor/pdfjs/ (pinned, SHA-256 verified, allow-listed)
```

Retrieval work, none of it in the deploy chain:

```sh
uv run scripts/sample_passages.py   # data/EVAL_PASSAGES.tsv: the sampled ground-truth passages
uv run scripts/eval_queries.py      # data/EVAL_QUERIES.tsv and data/EVAL_QUERIES_CROSS.tsv from
                                    # $CORPUS_DIR/eval/queries.tsv. Read BOTH sets: the crosslingual
                                    # one moves independently of the main one.
uv run scripts/evaluate.py --variant title --dims 1024,256   # data/EVAL_RESULTS.tsv
uv run scripts/compare_bakes.py --a plain --b plain-gpu      # how far two bakes disagree
uv run scripts/chunk_quality.py     # countable chunking defects; --baseline to diff against another
uv run scripts/inspect_chunks.py --doc <name> --page 1      # the chunks of one page, as a reader met them.
                                    # The pair to reach for before changing chunk.py.
uv run scripts/eval_pagefirst.py    # a page's own vector: as a gate, or as a term
uv run scripts/eval_centring.py     # centring before binarising: rejected, kept to re-measure
uv run scripts/eval_language_axis.py  # projecting out a language direction: rejected, kept
uv run scripts/bench.py             # the chain over the eval documents plus seeded distractors
EMBED_URL=http://127.0.0.1:8461 node scripts/evaluate_rescore.mjs   # BM25 weight, through the SHIPPED ranker
QUERIES=data/EVAL_QUERIES_CROSS.tsv EMBED_URL=http://127.0.0.1:8461 node scripts/evaluate_rescore.mjs
EMBED_URL=http://127.0.0.1:8461 INDEX_DIR=data/index-sweep/index node scripts/sweep_blend.mjs
                                    # section / page / previous-page weight grid (needs a section bake
                                    # and an index built with --section-weight into data/index-sweep)
uv run scripts/embed.py --variant <v> --out-name <v>-gpu --weights model.onnx --gpu
uv run scripts/build_index.py --vectors data/vectors/<v>-gpu --page-weight 0 --out data/index-<v>
                                    # the metadata LABEL arms (metabi, metaq, metanl, metaqnl), all
                                    # measured and rejected (DESIGN.md). Kept so a re-measure after a
                                    # chunk-size change is a bake and an index each. To confirm a WINNER
                                    # against the shipped blend also bake page-<v>-gpu: build_index.py
                                    # refuses to mix variants across the passage and page bakes.
                                    # The lever nobody has pulled yet is the label's LENGTH: --meta-fields title.
SVC=http://127.0.0.1:8650 node scripts/measure_service.mjs latency|burst 50|rss
```

The chunking sweep is its own command and runs for hours. `./grid_search.sh` builds one index per chunking strategy, dumps what the SHIPPED ranker would show for every eval query, and grades every retrieved passage with `Qwen3-Reranker-4B` on the GPU, because the `(file, page)` label counts a better passage from another rendition as a miss. `data/GRID_RESULTS.tsv` prints both metrics side by side and they are meant to be read together. Everything is resumable (content-hash gates, skipped outputs, verdicts appended to `data/grid/judge-scores.jsonl`): Ctrl-C at any point, re-run to continue. The artefacts live under `data/grid/` (untracked, never deployed).

```sh
./grid_search.sh --list                         # the strategy table and what is already built
./grid_search.sh --docs /tmp/ten.txt --only t170-512-sec     # the whole chain on ten documents
uv run scripts/judge_rerank.py --calibrate 32   # what the judging pass will cost
./grid_search.sh                                # the whole sweep
./grid_search.sh --stage report                 # recompile data/GRID_RESULTS.tsv from what is judged so far
uv run scripts/judge_rerank.py --doc-tokens 256 --batch-rows 32   # the length-bias control
```

The card has to be FREE for both the bake and the judge: the judge refuses to start under 9.5 GiB and says so, while `embed.py` quietly drops to a 512 MiB batch budget and bakes twenty times slower, which is the one failure here that costs hours without announcing itself. `--batch-rows` changes no pair's score, so runs at different values stay comparable.

### Running locally

Run the sibling's real encoder and this repo's services. Never reimplement the embedding endpoint here: its wire format (base64 int8, fixed scale 127) is the shared contract, and a second definition would drift silently. The request carries `dim`, so this site picks its own width regardless of the encoder's default.

```sh
cd ../justelesRCP && uv run src/embed-service.py --port 8461 --no-backlog   # the real encoder, ~50 s to load
EMBED_URL=http://127.0.0.1:8461 INDEX_DIR=$CORPUS_DIR/dist/index node server/service.mjs   # search on :8650
PAGES_DIR=$CORPUS_DIR/dist/restricted INDEX_DIR=$CORPUS_DIR/dist/index uv run server/pages.py   # pages on :8651
uv run scripts/dev_server.py                     # dist/www on :8649, /api/search and /api/page proxied
EMBED_URL=http://127.0.0.1:8461 node scripts/check_search.mjs   # the gate, plus real queries printed
EMBED_URL=http://127.0.0.1:8461 SVC=http://127.0.0.1:8650 node scripts/check_search.mjs
                                                 # ... plus: is the RUNNING service on 8650 stale?
```

`node server/load.mjs [dist/index]` loads the index under Node and prints load time, rank time and RSS, with no HTTP. `dev_server.py` proxies the search and page paths and answers 404 for every other `/api/*`, as Caddy does. `ssh -N -L 8461:127.0.0.1:8461 <the server>` forwards a deployed encoder instead of running one; `.githooks/pre-push` defaults `EMBED_URL` to `http://127.0.0.1:8461` for that reason, and the live-query part of the gate skips itself when nothing answers.

The two browser gates. `check_ui.mjs` is the only thing that can see a highlight land on the wrong words; `check_chrome.mjs` covers the per-page chrome (`src/site.js`), driven by `window.__APP_CONFIG__` and wall-clock time, which it fakes per scenario:

```sh
PW=<path to a playwright install> SITE=http://127.0.0.1:8649 REQUIRE_SEARCH=1 node scripts/check_ui.mjs
PW=<path to a playwright install> SITE=http://127.0.0.1:8649 node scripts/check_chrome.mjs
```

This repo has no `node_modules` and no `package.json`, so `PW` points at an existing playwright install (`ls ~/.npm/_npx/*/node_modules/playwright`, and its version has to match a browser in `~/.cache/ms-playwright`). Chromium is what the MCP browser tools use; the system Chrome they default to is not installed here.

### Things about `embed.py` that are not guessable and cost hours if got wrong

- Every `.npz` records `chunks_hash`, the `src_hash` of the chunk file it was baked from, and `build_index.py` refuses a bake whose `chunks_hash` disagrees with the chunks on disk, or that has none. A file without it is also STALE to `embed.py`, so an ordinary bake or `embed.py --check` picks it up. A rechunk with no rebake therefore stops the index build instead of shipping vectors that describe other text.
- `--check` needs the SAME `--weights` the bake used, or it lies. The model identity is in the hashed params and the default `--weights` is the int8 CPU graph, so `--check --out-name meta-gpu` on its own reports every document stale against a bake made with `--weights model.onnx`. That reads exactly like a corpus change and invites a full rebake of a bake that was current.
- It holds an exclusive `flock` on its output directory. A second bake into the same `--out-name` is refused, deliberately.
- `--threads 0` (the default) means one intra-op thread per *physical* core. Letting onnxruntime take every hardware thread is 3.3x **slower**.
- `--weights model.onnx --gpu` bakes on the GPU in minutes instead of hours, but those are not the weights the server embeds queries with. The int8 graph gains nothing from a GPU (its quantised operators have no CUDA kernels), and the current model publishes no fp16 graph, so fp32 is the GPU artefact; `../justelesRCP/scripts/download-model.sh --keep-fp32` leaves it on disk. The float-for-int8 substitution was measured only on the previous model (DESIGN.md, "fp16 weights bake 50x faster"); `compare_bakes.py` is the way to re-check it.

## Gotchas

- **uv needs the sandbox off.** `uv run`, bakes and the test suite fail inside the Claude Code sandbox; run them with it disabled, and log long jobs under `/tmp/logs`.
- `sudo docker`, not `docker`. `python`, not `python3`.
- The user runs several agent sessions at once. Do not use `git add -A` or `git commit -a`; add by explicit path.
- `local/` is gitignored and holds corpus repositories that are never pushed. Never copy a corpus's names, counts, documents or paths into this repository.
- `README.md` (FR) and `README.en.md` (EN) are one document in two languages: every change to one is a change to the other, in the same commit. `.githooks/pre-push` refuses a push that touches only one and a push where the two carry a different number of `##` sections, but nothing can see a paragraph rewritten in French and left stale in English.
- `docker/docker-compose.yml` sets `name: ${SITE_ID:-justelesdocs}`, and two sites on one host need two `SITE_ID`s. Compose names a project after the directory holding the compose file, which is `docker/` here and in `../justelesRCP`, and both projects call their Caddy service `web`; with the same project name `docker compose up --force-recreate` in either replaces the other's web container. A deployment that predates `SITE_ID` must set it to the name it already runs under.
- The SEARCH container (`<SITE_ID>-search`), not the Caddy one, reaches the encoder by CONTAINER NAME (`justelesrcp-embed:8461` by default) over the external Docker network `EMBED_NETWORK` (default `justeles-embed`). Caddy sits only on the compose default network and proxies `/api/search` to `search:8650`. So the network has to exist before either stack starts (`sudo docker network create justeles-embed`), and a `curl` from the host proves nothing, because nothing is published on the host. Two host-based routes cannot work: a loopback publish is unreachable from a container, and compose's `ports:` takes an IP and never a hostname. DESIGN.md, "Embedding service", records why.
- Pin `NETWORK_SUBNET` per site: a host whose Docker address pool is exhausted hands out a fallback subnet that may have no outbound route (DESIGN.md, "Docker networking: pin the subnet").
- A service left running from an EARLIER session is the trap of this repo, because none of the three reloads on an edit and all three hold their port. The new launch dies on `Address already in use` in a log nobody is reading, and the old code answers every request. It has cost five debugging sessions: a 400 on a filter field the running build predated, a 404 on a route it did not have, a weight change that landed after the search service came up so the browser gates certified the old value, a service that answered from the previous vectors after an index rebuild, and a dev server 404ing a path out of its own allow-list, which the page read as a feature switched off (the worst kind, because the failure looks like a setting). `ss -ltnp | grep -E '8649|8650|8651'` plus `ps -o lstart= -p <pid>` is the whole diagnosis. The SEARCH service can be checked rather than remembered: `SVC=http://127.0.0.1:8650 node scripts/check_search.mjs` compares the `ranker_sha` in its health payload against the `src/search.js` on disk, and its `index_sha` (a hash of the loaded `meta.json`'s bytes, so a metadata-only rebuild shows too) against `dist/index/`, and fails with the pid hint when either has moved. The gates cannot see this on their own, because they load the ranker and the index in a fresh process. Nothing checks the page service or the dev server this way yet.
- A relevant hit scores a cosine of about 0.5 to 0.7, not 0.9. `SEARCH_FLOOR` has to be set from that, per model and corpus, not from intuition.
