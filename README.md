<!-- English version. Version française : README.fr.md
     IMPORTANT: README.md (EN) and README.fr.md (FR) must stay in sync.
     When you change one, update the other accordingly. -->

# justelesdocs

*Lire en [français](README.fr.md).*

**A static, ad-free site that serves a curated PDF corpus with multilingual semantic search.** You ask a question in plain language, and the answer is shown **on the page of the original PDF, with the passage highlighted**. There is no language model in the serving path: only embeddings and a nearest-neighbour search.

justelesdocs is the software. The documents, their metadata and their configuration live in a separate directory, the *corpus*, that the software reads through the `CORPUS_DIR` variable. One installation of the software can serve any corpus.

## Contents

- [What the site does](#what-the-site-does)
- [How it works](#how-it-works)
- [Using it with an AI assistant](#using-it-with-an-ai-assistant)
- [Quick start](#quick-start)
- [Running locally](#running-locally)
- [Deploying with docker compose](#deploying-with-docker-compose)
- [Configuration](#configuration)
- [Tests and gates](#tests-and-gates)
- [Credits](#credits)
- [License](#license)

## What the site does

- You ask a question, not keywords. The search runs over every passage of every document at once.
- A hit opens the PDF page it came from with the passage highlighted. Several passages on one page each get a highlight. From there, one control opens the whole PDF or downloads it.
- Filters (issuer, country, year, language, document type, topic, access) come from the corpus's metadata. Vocabularies are closed, multi-valued fields match on any value, and the year is a range slider.
- Several renditions of the same document (a summary, the full text, a supporting report) collapse into the best-ranked one, with the others offered beside it.
- Multilingual model: a question in one language can find a passage in another. The interface is bilingual French and English, and a corpus can add or override strings.
- An `access: restricted` tier: some documents can be read without being handed over. Only the passage's pages and two pages either side are offered to the reader, cut out of the PDF on each request, and the file itself is never reachable.
- Figures can be searched through descriptions written offline by a vision model, labelled as such.
- Nothing is tracked. The question is sent once to the site's search service, which logs neither the question nor the results. Analytics are off unless configured, and even then record only a page view without the question. The pages meant for AI assistants count one page view per request on the server, with the path, the assistant's name and no question.

## How it works

Everything heavy happens offline, on a machine with a GPU: OCR of scanned pages, cutting each PDF into passages with their exact coordinates on the page, computing vectors, building the index. At serving time nothing heavy runs.

The build chain is a series of `uv run scripts/*.py` steps, each gated by a content hash, so re-running the whole chain with nothing changed takes under a minute and a new document only recomputes what concerns it. Hard gates refuse rather than warn: no skipped PDF, no empty document, no restricted document under the web root, a ranker that finds its own passages.

At serving time three hardened containers run: Caddy (static files, strict CSP, per-IP rate limiting), a Node search service that holds the index in memory and ranks, and a Python page service that cuts one page out of a restricted document. The query encoder is external (the `embed` container of the sibling project justelesRCP) or optional in the same stack (the compose `embed` profile). The index is binary, 1024 dimensions, one bit per dimension, which fits a small server with no GPU.

For the detailed architecture, see [ARCHITECTURE.md](ARCHITECTURE.md); for the technical decisions and their measurements, [DESIGN.md](DESIGN.md).

## Using it with an AI assistant

A deployed site is AI-ready. Paste its `/llms.txt` link once into Claude, ChatGPT, Mistral's Le Chat, Gemini or any assistant that can read web pages, and ask it to remember the site. The search page says so under the box, and links to it.

That page, written by `stage.py` for the corpus being served, tells the assistant what the corpus holds and how to use three plain-text routes:

- `GET /api/search.md?q=<question>`: the same search as the page, answered in Markdown, with the ten best passages, their documents' metadata and links. It takes exactly the search page's own query string (filters, year range, ranking choices), so an address copied from the address bar works once its path is changed. Rate-limited per IP more tightly than the page's own search, since it is what a crawler would hit.
- `/api/doc/`: the list of documents, then one Markdown page per document (metadata, links, the first words of each page) and one per page (its text), written at stage time and served as static files.
- the viewer links, for handing a person the passage highlighted on its page.

Two buttons do the same for a reader: "Copy for an AI" in the viewer's menu copies the passage, its citation and these links, and the one above the results copies the Markdown address of the search on screen.

The `access: restricted` tier keeps its promise here too: a restricted document gets no Markdown page and no text, and the search only quotes the one passage the page would show. `check_served.py` refuses a stage where a restricted document's text appears under `api/doc/`. `robots.txt` asks crawlers to stay out of `/api/` while letting in the agents an assistant fetches with when a person asked it something (Claude-User, ChatGPT-User and the like), and every Markdown answer carries `X-Robots-Tag: noindex`.

**Why no MCP server?** The [Model Context Protocol](https://en.wikipedia.org/wiki/Model_Context_Protocol) is the other common way to plug a site into an AI assistant, but it needs a server program running next to the site, a connector that each user has to add by hand, and ongoing upkeep as the protocol evolves. It is also more attack surface on a server that is meant to stay small. justelesdocs follows the [`/llms.txt`](https://llmstxt.org) convention instead: plain files and one GET route that any assistant able to read a web page can use with nothing to install.

Built with Claude Code.

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

## License

[GNU Affero General Public License v3.0](LICENSE). The AGPL covers use over a network: if you run a modified version as a public site, its readers must be able to get your modified source. The footer's source link is how this site offers it, so point `[site] repo_url` in your `corpus.toml` at your own repository when you change the code.

The license covers the software only. A corpus (its documents, metadata and strings) is not part of it and keeps whatever terms its own documents carry.
