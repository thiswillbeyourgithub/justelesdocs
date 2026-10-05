<!-- English version. Version française : README.md
     IMPORTANT: README.md (FR) and README.en.md (EN) must stay in sync.
     When you change one, update the other accordingly. -->

# justelesdocs

*Lire en [français](README.md).*

**A static, ad-free site that serves a curated PDF corpus with multilingual semantic search.** You ask a question in plain language, and the answer is shown **on the page of the original PDF, with the passage highlighted**. There is no language model in the serving path: only embeddings and a nearest-neighbour search.

justelesdocs is the software. The documents, their metadata and their configuration live in a separate directory, the *corpus*, that the software reads through the `CORPUS_DIR` variable. One installation of the software can serve any corpus.

## Contents

- [What the site does](#what-the-site-does)
- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Running locally](#running-locally)
- [Deploying with docker compose](#deploying-with-docker-compose)
- [Configuration](#configuration)
- [Tests and gates](#tests-and-gates)
- [Credits](#credits)

## What the site does

- You ask a question, not keywords. The search runs over every passage of every document at once.
- A hit opens the PDF page it came from with the passage highlighted. Several passages on one page each get a highlight. From there, one control opens the whole PDF or downloads it.
- Filters (issuer, country, year, language, document type, topic, access) come from the corpus's metadata. Vocabularies are closed, multi-valued fields match on any value, and the year is a range slider.
- Several renditions of the same document (a summary, the full text, a supporting report) collapse into the best-ranked one, with the others offered beside it.
- Multilingual model: a question in one language can find a passage in another. The interface is bilingual French and English, and a corpus can add or override strings.
- An `access: restricted` tier: some documents can be read without being handed over. Only the passage's page and one page either side are served, cut out of the PDF on each request, and the file itself is never reachable.
- Figures can be searched through descriptions written offline by a vision model, labelled as such.
- Nothing is tracked. The question is sent once to the site's search service, which logs neither the question nor the results. Analytics are off unless configured, and even then record only a page view without the question.

## How it works

Everything heavy happens offline, on a machine with a GPU: OCR of scanned pages, cutting each PDF into passages with their exact coordinates on the page, computing vectors, building the index. At serving time nothing heavy runs.

The build chain is a series of `uv run scripts/*.py` steps, each gated by a content hash, so re-running the whole chain with nothing changed takes under a minute and a new document only recomputes what concerns it. Hard gates refuse rather than warn: no skipped PDF, no empty document, no restricted document under the web root, a ranker that finds its own passages.

At serving time three hardened containers run: Caddy (static files, strict CSP, per-IP rate limiting), a Node search service that holds the index in memory and ranks, and a Python page service that cuts one page out of a restricted document. The query encoder is external (the `embed` container of the sibling project justelesRCP) or optional in the same stack (the compose `embed` profile). The index is binary, 1024 dimensions, one bit per dimension, which fits a small server with no GPU.

For the detailed architecture, see [ARCHITECTURE.md](ARCHITECTURE.md); for the technical decisions and their measurements, [DESIGN.md](DESIGN.md).

## Quick start

1. Copy the example corpus into a directory of your own, outside this repository or in `local/` (gitignored):

   ```sh
   mkdir -p local/mycorpus && cp corpus.example/corpus.toml local/mycorpus/
   export CORPUS_DIR=local/mycorpus
   ```

   `CORPUS_DIR` is required and has no default: every `data/...` and `dist/...` path is resolved inside it.

2. Edit `$CORPUS_DIR/corpus.toml`: site name, facets, vocabularies, tiers, renditions, languages. Every table is commented.

3. Put your PDFs in `$CORPUS_DIR/data/GUIDELINES/`. Only that directory is ever read, indexed and served.

4. Run the chain:

   ```sh
   uv run scripts/check_pdfs.py
   uv run scripts/manifest.py      # data/MANIFEST.tsv, fills blanks only
   uv run scripts/ocr.py           # if some pages are scans
   uv run scripts/chunk.py
   uv run scripts/verify_chunks.py
   uv run scripts/chunk.py --page-chunks --out data/chunks-page
   uv run scripts/embed.py --chunks data/chunks-page --out-name page-meta-gpu --weights model.onnx --gpu
   uv run scripts/embed.py --out-name meta-gpu --weights model.onnx --gpu
   uv run scripts/build_index.py --dims 1024
   uv run scripts/stage.py
   uv run scripts/check_served.py
   node scripts/check_search.mjs
   ```

   The two `embed.py` steps want a GPU: minutes with one, hours without. The model weights come from the sibling project justelesRCP (`scripts/download-model.sh --keep-fp32`).

## Running locally

```sh
cd ../justelesRCP && uv run src/embed-service.py --port 8461 --no-backlog   # the encoder
EMBED_URL=http://127.0.0.1:8461 INDEX_DIR=$CORPUS_DIR/dist/index node server/service.mjs
PAGES_DIR=$CORPUS_DIR/dist/restricted INDEX_DIR=$CORPUS_DIR/dist/index uv run server/pages.py
uv run scripts/dev_server.py    # the site on http://127.0.0.1:8649
```

None of these services reloads when a file changes. After an edit, check that the old process is no longer holding its port (`ss -ltnp`).

## Deploying with docker compose

The server needs this repository's `docker/`, `server/` and the `src/` modules the search image copies, plus the built tree `dist/` (`www/`, `index/`, `restricted/`), by default next to `docker/` (`DIST_DIR=../dist`).

```sh
cp docker/env.example docker/.env      # then edit
sudo docker network create justeles-embed   # once, the network to the encoder
cd docker && sudo docker compose up -d --build
```

The site listens on `127.0.0.1:8648` and expects a TLS reverse proxy in front of it. `scripts/smoke_deployed.sh <url>`, piped to the server (`ssh ... "sh -s -- <url>" < scripts/smoke_deployed.sh`), checks that search answers and that a restricted document is only served page by page.

## Configuration

- `$CORPUS_DIR/corpus.toml`: everything about the corpus (`[site]`, `[facets]`, `[vocabularies.*]`, `[tiers]`, `[renditions]`, `[languages.*]`, `[figures]`, `[ocr]`, `[manifest]`, `[judge]`). [corpus.example/corpus.toml](corpus.example/corpus.toml) documents each key.
- `$CORPUS_DIR/strings/<lang>.json`: optional overrides of the interface strings in `src/i18n.js`.
- `$CORPUS_DIR/scenarios.json`: what the gates expect of this corpus (example queries, documents to find); without it they fall back on what the index contains.
- `$CORPUS_DIR/changelog/` and `$CORPUS_DIR/VERSION`: if present, they replace the software's release notes and version on the site.
- `docker/.env` (from [docker/env.example](docker/env.example)): `SITE_ID` (compose project name and container prefix, default `justelesdocs`; two sites on one host need two), `DIST_DIR`, `NETWORK_SUBNET`, `EMBED_NETWORK`, `PORT`, `BIND_ADDR`, analytics, search limits.

## Tests and gates

```sh
uv run tests/run.py
```

runs the software's tests against the example corpus, with no real corpus or index needed, plus `$CORPUS_DIR/tests` when the variable is set. Activate the pre-push hook once per clone with `git config core.hooksPath .githooks`: it runs the tests, and the corpus gates when `CORPUS_DIR` is set.

The gates complement the tests: `verify_chunks.py`, `check_served.py` and `check_search.mjs` in the build chain, `check_ui.mjs` and `check_chrome.mjs` in a browser after any change to `src/`, `smoke_deployed.sh` against the running site.

## Credits

Developed with [Claude Code](https://claude.com/claude-code). PDF rendering by [pdf.js](https://mozilla.github.io/pdf.js/) (Apache 2.0), vendored in `vendor/pdfjs/`.
