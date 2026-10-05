# Design decisions

This file records the software decisions behind justelesdocs and the reason for each one. Most of the reasons are measurements. Unless stated otherwise, every number below was measured on the corpus the software was first built for (a bilingual FR/EN corpus of about 500 clinical guidelines, called "the first corpus" below), with a hand-built evaluation set of about 200 queries whose ground truth is a (file, page) pair. A different corpus can move any of these numbers, so treat them as the starting point and re-measure with the tools named in each section. A corpus's own notes (what is in it, how it was sorted, what its documents need) belong in that corpus's directory, not here.

The section titles are cited from code comments and other docs. Rename one only together with every file that cites it (`git grep -n 'DESIGN.md'`).

## Decided

### Page rendering: pdf.js, client-side

A hit is shown as the PDF page itself, rendered in the browser by pdf.js. Pre-rendered page images were rejected: they cost disk on a small server, lose the text layer (selection, search, accessibility) and need a second component for the whole-document view. One viewer component serves both the single page and the whole document. pdf.js runs with `isEvalSupported: false` and a same-origin worker, and is vendored by `scripts/vendor.py` from a pinned archive whose SHA-256 is verified and whose members are allow-listed. The CSP stays `default-src 'self'`, plus `blob:` for `worker-src` and `img-src`, `object-src 'none'`, and no `wasm-unsafe-eval`.

### OCR: a local preprocessing step, and the OCR'd file is the one we serve

Scanned pages get a text layer offline with `ocrmypdf`, never on the server. The flags are measured choices: `--skip-text` silently skipped every page that held any text at all (a scanned page with a printed header), `--force-ocr` rasterises digital text and degrades it, so the mode is `--redo-ocr`; `--output-type pdf` avoids the Ghostscript PDF/A pass, which rewrote files for no benefit. The OCR'd derivative is the file that gets chunked AND served, because a highlight drawn from OCR boxes only lands on the right words in the file those boxes came from. Originals are never modified. A page counts as missing text only when it also carries a large image or many vector drawings, so blank versos do not trigger OCR. Every page is checked before and after. The cache key is the source bytes plus the language, mode and flags, and `ocr_index.json` records the original's hash so a stale derivative is refused rather than served. When `chunk.py` finds documents that still need OCR it stops and says so.

### Figures are searched through text instead

A multimodal embedder was rejected: on the first corpus about 5% of pages are figure pages, which does not pay for a second resident model, and the single-encoder rule (below) forbids one anyway. Instead `scripts/figures.py` (census, extract, emit, apply) has a vision model describe each figure offline. The model sees the whole page beside the crop, so it can use the surrounding text, and is told to leave out credits and boilerplate. Each description row is stamped with its model; the latest row per crop wins. `chunk.py` appends one chunk per described figure, after all the text chunks of its document, labelled in the UI as a model's description rather than the document's words. The descriptions enter a document's content hash only when it has some, so a new batch rechunks and rebakes only its own documents. The encoder input opens with the figure's kind word in its own language ("Tableau : ", "Flowchart: "), which is unmeasured. The figures filter (include, exclude, only) is a cut of each document's chunk range, which works because figure chunks always come last. There are no figure queries in the evaluation set yet, so nothing here is measured for retrieval.

### Embedding service: this project owns none, and that is the decision

The query encoder runs in a sibling project's container (justelesRCP's `embed` service), and this project contains no copy of it. A copied encoder service drifts silently: `compare_bakes.py` showed that a different weighting of the same model reorders about 10% of a top 10, which no test notices. On a small server exactly one copy of the model may be loaded. The search service reaches it by container name over an external Docker network (`EMBED_NETWORK`, default `justeles-embed`), which must exist before either stack starts.

Two host-based routes were tried first and neither can work. A port published on the host's 127.0.0.1 is unreachable from a container, because the connection arrives from the bridge subnet, not from loopback. Compose's `ports:` takes an IP and never a hostname, so `host.docker.internal` is rejected at parse time. Only exact paths are proxied: a wildcard would expose the sibling's crawl endpoint.

The width travels with each request (`dim`), so this site picks its own width regardless of the encoder's default. Truncating a Matryoshka vector and renormalising is exact, so the encoder can serve any width. Width is asserted in three places: the encoder answers 400 for a width it cannot give, the client refuses a reply of the wrong length, and the search service refuses to start when `SEARCH_DIM` disagrees with the index. The model must be multilingual, so that a question in one language can find a passage in another. For a host with no sibling running, the optional compose `embed` profile builds the same encoder from a sibling checkout with that repo's own Dockerfile (still one definition of it).

### Deployment target

The site binds to loopback by default (`BIND_ADDR`) on port 8648, and TLS is terminated by a reverse proxy upstream. The web image is deploy-invariant: Caddy plus the rate-limit plugin, with everything else bind-mounted, so only a plugin change needs `--rebuild-web`.

### The server side: one container, and the PDFs are not cached

Caddy compresses text types only: PDFs are already compressed and are fetched by Range requests, which compression breaks. Index files with a content hash in their name are served immutable. `meta.json` is `no-cache`, because it holds the offsets into the hashed files and a mixed pair reads the wrong rows. PDFs are `no-cache` too, because the highlight coordinates must match the bytes being served; `no-cache` still lets the browser store the file, so a revalidation is a 304 that costs nothing.

### What gets shipped: an allow list, built by a script, not an rsync exclude list

`stage.py` builds the served tree from the chunk files' `source` field, which is the single definition of which file is served. PDFs are symlinked and transferred with `rsync -L`, and the tree is pruned before linking. An rsync exclude list was rejected because it fails open: a forgotten pattern ships a document that must not be served. The root `.dockerignore` follows the same rule: it denies everything and re-admits named files.

### Decided: a restricted document is cut page by page, on the fly

A document marked `access: restricted` may be read but not downloaded. `stage.py` puts those PDFs in `dist/restricted/`, a sibling of the web root that no Caddy root can reach, and strips the `text` field from their per-document JSON (geometry only, which is all the highlight layer reads; the cost is that BM25 cannot see them). `server/pages.py` cuts one page per request with `insert_pdf` and `tobytes(garbage=3, deflate=True)`: measured at 4.3 ms and 53 KB for a page in the middle of a 1,275-page book. It keeps an LRU of open documents (`PAGES_CACHE`, default 4), allows two concurrent cuts behind a semaphore plus a reentrant lock (mupdf is not thread-safe, and a plain `Lock` deadlocked), and answers `Cache-Control: private, no-store`. `scripts/check_served.py` refuses a staged tree that holds a restricted file or a word of one.

Rejected: pre-cutting every page (tens of thousands of files to stage and ship), page images (they destroy the text layer, accessibility and the highlight), and a database (pre-cutting in disguise). There is deliberately no token on the endpoint: a token would not stop anyone scraping one page at a time, which is what the per-IP limit (`PAGE_RATE_EVENTS`, 120 a minute) is for. The viewer allows one page either side of the page the reader opened, centred on that page rather than on the current one, and hides the download and full-PDF links; `check_ui.mjs` checks this.

## Deliberately deferred: the index backend

Search is an exact brute-force scan. Approximate nearest-neighbour indexes only pay above roughly 100k to 1M vectors at this width, and with a binary index even the upper end scans in tens of milliseconds. faiss's flat index is the same matrix product. No vector database: it adds a service, a format and a failure mode for nothing at this size. Revisit when a corpus is an order of magnitude larger.

## Decided: how the search is evaluated

Ground truth is a (file, page) pair, never a chunk id, so every chunking can be scored against the same labels. Passages are sampled uniformly over documents with a fixed seed (`sample_passages.py`), so long documents do not dominate the set. Queries come in kinds (precise, vague, crosslingual, table, dose) and are written in a tracked script, not in data, so they are reviewable. A query may name acceptable alternative answers (`gold_alt`). A crosslingual mirror set (82 queries on the first corpus) asks for a passage in the other language; it runs at about a quarter of the main set's MRR and moves independently of it, so both are read. Known biases are recorded with the set: queries written from a passage favour that passage's wording, and the label counts a better passage in another document as a miss.

The standard error of a hit rate is about 0.5/sqrt(n), so about 0.035 at 200 queries. Arms are compared with paired better/worse counts and sign tests, not only with means. `scripts/evaluate_rescore.mjs` and `sweep_blend.mjs` import the shipped `src/search.js`, so what is measured is what ships; there is no second implementation of the ranker anywhere.

The chunking grid search (`grid_search.sh`) adds an offline judge, `Qwen3-Reranker-4B` on the GPU, that grades every retrieved passage, because the (file, page) label counts a better passage from another rendition of the same guidance as a miss. The two metrics are printed side by side. A length-bias control re-judges the same pairs at a fixed passage window (`--doc-tokens 256`). There are no figure queries yet.

### Decided: an experiment runs over the eval corpus, not over the whole shelf

`bench.py` builds indexes over the documents the eval set points at plus a seeded sample of distractors, which makes a sweep cheap. A subset's page@1 is inflated against the whole corpus, so compare arms only within one sweep, and confirm a winner on the full index.

## Measured: subtracting a French/English axis does nothing

Projecting out the direction between the mean French and mean English chunk was meant to help crosslingual queries. It did not: language is not one direction in the embedding. Rejected; `eval_language_axis.py` is kept so it can be re-measured.

## Measured: vector width matters, int8 does not

Matryoshka truncation is not free: going from 1024 to 256 dims cost 7.7 points of page@1, and the crosslingual queries lost most. Precision costs much less than width.

### Measured: spend the bits on WIDTH, not on precision. 1024-dim binary beats 256-dim int8 at half the size

At 128 bytes a vector, 1024 dims at one bit each beat 256 dims at a byte each on every page-level metric. Scoring is asymmetric: the passage is one bit per dimension, the query keeps its int8 precision, and a table built once per query holds what each of the 128 byte positions contributes for each of the 256 byte values, so a passage costs 128 lookups instead of 1024 multiplications. It is twice as fast as the 256-dim int8 scan.

### Measured: centring the corpus fills every bit of the index and buys nothing, which settles PCA as well

Subtracting the corpus mean before taking signs balances the bits, but bit balance is not information: the entire quantisation loss was 0.017 MRR, and centring did not recover it. PCA rotation and ITQ cannot recover more than that loss either, so they are not worth a format change. `eval_centring.py` is kept.

### Measured: the relevance floor, and what it cannot separate

`SEARCH_FLOOR` was about 0.28 on the binary index of the first corpus; re-measure it for every model and corpus. It separates "not this domain at all" from the corpus, not "not in the corpus": a question from an adjacent domain scores above the worst correct hits. It ships off by default. A relevant hit scores a cosine of about 0.5 to 0.65, not 0.9, so a floor set by intuition hides good answers. The value in use is echoed in every answer so the page can say "nothing above the floor" honestly.

### Decided: the query algebra subtracts by projection, not by subtraction

Only `-(` and `+(` open groups. `A | B`, `(A|B)` and `A +(B)` all mean the averaged embedding. A negative group is averaged and removed by projection, `q - (q·n)n`, so what ranks is the part of the question not about the subtracted term rather than its opposite, and there is no weight to tune. A query that cancels to nothing, more than `MAX_QUERY_TERMS` (10) terms or more than `MAX_QUERY_CHARS` (512) characters is refused. Degenerate syntax parses to nothing rather than to an error.

### Measured: a reference penalty has to be mined from the corpus, and has to be a contrast

Describing a bibliography to the encoder does not work: the model encodes the look of a reference list, not the idea, and the projection on a description query is near zero. The axis is mined instead: the mean of the chunks that `looks_like_references` flags, minus the mean of all chunks. Without the contrast 69% of chunks scored high; with it, 15%. Each chunk stores a `uint8` score, and a request that asks for it loses `0.30 * score/255 * ref_scale` of a cosine: a penalty, not a filter, so a citation list that really is the best answer still wins.

### Measured as a cost: favouring recent documents

A bonus of 0.05 of a cosine with an 8-year half-life, `0.05 * 0.5 ** ((newest - year) / 8)`. The evaluation set cannot justify it (its answers are not biased to recent documents), so it was measured only as a cost, which is small. Ages count from the newest year in the index, not from today, so a rebuild next year changes nothing. A blank year is treated as the median. Exponential rather than linear, so very old documents are not pushed off the list.

### Measured: a metadata label belongs BEFORE the passage

Every chunk is embedded behind a label made of the document's curated title, issuer and year (`--variant meta`). The label is worth +0.080 MRR. Moving the same label after the passage costs about 0.057 MRR even though the current encoder pools on the last token. On the current encoder the curated label also beats the filename stem. Two labels at once lose. A query-side domain prefix ("this is a question about ...") does not help. The open lever is the label's length (`--meta-fields title`).

### Measured on the previous encoder: the curated title is a WORSE prefix than the filename

On the previous encoder (arctic-embed-l-v2.0, CLS pooling) the filename stem beat the curated title, and issuer and year added nothing. That reversed with the change to a last-token encoder, which is why the curated label ships now. The lesson: a prefix measurement belongs to its encoder and must be re-taken when the model changes.

### Measured: the bilingual metadata prefix, rejected on the chunk size

Putting the title in both languages into the label cost 0.021 MRR: at a 256-token chunk the longer label takes 21% of the pooled sequence instead of 14.5%, and recall drops at short chunks. Rejected.

### Measured: the label's delimiter, four cells, no effect

Quoting the title, a newline instead of ". ", or both: no measurable effect, because a delimiter is one token out of about 250. The arms (`metaq`, `metanl`, `metaqnl`, `metabi`) stay in `embed.py` so a re-measure after a chunk-size change is a bake and an index each.

### Measured: BM25 over the shortlist

The top 50 candidates are re-scored by `cosine + w * bm25 / scale`, with BM25 computed over those 50 chunks' own text, so nothing is shipped for it (no term statistics, no inverted index; IDF comes from the candidates). Positive query terms only. It runs before folding, because folding picks which rendition stands for a family. At 256-token chunks `w = 0.15` and turning it off costs 0.0985 MRR, about nineteen queries of page@1; at 1024-token chunks it was worth 0.008, about eight queries at the time. The UI checkbox is nevertheless off by default: a lexical boost makes failures harder to understand and breaks the "meaning first" promise. A link carries it as `bm25=1`.

### Measured: BM25 was a same-language preference in disguise, and normalising it per language is free

With one global `scale`, BM25 voted for the reader's language: the shortlist for a French question was mostly French documents, and those are the ones it shares words with. `scale` is now the best BM25 among candidates in the same language as the one being scored, floored at half the best in the whole shortlist (`LEXICAL_GROUP_FLOOR`, 0.5). That is a Pareto improvement: the main set did not lose and crosslingual MRR doubled. A floor of 0 trades 26 main-set queries for 20 crosslingual ones, a trade rather than a gain.

## Measured: page-level vectors, and the Section-level retrieval term

Every score blends `0.85 * chunk cosine + 0.15 * page cosine`, where each page has its own vector from a separate page bake (`chunk.py --page-chunks`). The page term is not a quantisation artefact: the int8 index gains the same. Gating by pages first (search pages, then chunks within them) is worse, because it caps recall. `build_index.py` refuses to build without the page bake; `--page-weight 0` is how to say you meant it.

A section vector as a third term found nothing: the median section is shorter than a chunk, so it adds no context. The previous page as a term and max-pooling over pages also found nothing. The index format keeps a slot for a section file, unused. Any win measured at one quantisation has to be reproduced at fp32 before it ships.

## Measured: the shipped index

Exact numbers move with every rechunk, so only the shape is recorded here: on the first corpus the binary index held on the order of 10^5 chunks, page@1 was around 0.5 to 0.6 on the main set, and the crosslingual set ran at about a quarter of that in MRR. The index is good enough that the first browser run's main finding was a rendering bug, not a ranking one (see the gates below).

### Measured: in the SIBLING's regime, the index format barely matters

In the sibling's regime (search within one document) every format tested ranks the same, because the candidate set is small. The width-over-precision result above matters only when the whole corpus competes.

### Decided: binary is what ships, and int8 is the better index nobody can afford

At 1024 dims, int8 retrieves better by about 0.025 MRR, but on the first corpus it cost 8x the disk, 5.4x the rank time (437 ms against 81 ms) and 9x the RAM (302 MB against 34 MB) on a small server. Binary ships. This is a cost decision, re-taken each time a corpus changes size or the server changes: `build_index.py --quant int8` builds the other one, and `src/search.js` reads either.

### Measured: crossing languages works, and costs about half the recall

A French question finds an English passage and the reverse, at about half the recall of a same-language question. The UI says so. A translation step was rejected: it would add a model to the serving path.

## Decided: chunking

### Decided: a 170 to 512 token window, no overlap

A chunk is cut at a section boundary once 170 tokens are behind it, and 512 tokens is a ceiling, so a section up to 512 tokens stays whole (`FILL_FLOOR` 0.332, no overshoot, `SENTENCE_FLOOR` 0.2). Section boundaries are preferred, then paragraphs, then sentences; a list is split last. This was chosen deliberately against the grid search, which preferred 256-token chunks with 26 of overlap: a reader prefers a whole section to a fragment that ranks marginally better.

### Measured: the chunk is 256 tokens, not 1024

The grid search over about two dozen strategies, judged by both the reranker and the (file, page) labels, agreed that 256 beats 1024: +0.046 judge@1 and +0.098 MRR. The length-bias control (both passages judged at the same 256-token window) widened the gap, so the judge was not simply preferring short passages. Bigger chunks lose at rank 1 and straddle pages, which makes the highlight less useful.

### Measured: overlap does not help

Overlap carried only across a cut the budget forced is a no-op in practice (few cuts are forced). Overlap at every cut is worse. Neither ships.

### Measured: chunk size barely matters, and chunks that end where the text ends

Within roughly 200 to 1024 tokens, the earlier size sweeps moved retrieval by little; the 200-token row lost slightly, which is why the packer prefers to fill towards the target rather than cut early. Ending a chunk where the text ends (a paragraph, a list, a heading, a page break, `--boundaries`) measured better than cutting wherever the budget runs out, and is the default. `--page-chunks` (one chunk per page) is an experiment, not a shipping mode.

### Measured: a section-sized chunk loses

A chunk that IS a section lost by three standard errors: long sections dilute, and short ones are fragments with too little context.

### Measured: carrying the section heading into a chunk buys rank 1 and costs the tail, so it stays off

Prepending the section heading (`--heading-tokens`) won some rank-1 hits and lost more further down. Off.

### The chunker reads the page, not the text stream

The line is the unit of provenance: each chunk carries the boxes of the lines it came from, so the highlight is exact. Budgets are counted with the encoder's real tokenizer. A running header needs two signals to be dropped (it recurs across pages AND sits in the margin). Rows with no letter or digit, or a box under 1 pt, are dropped. Two bugs were caught only by disbelieving output (padding in a batch, an overlap index off by one), so `chunk_quality.py` counts the defects a reader would see and `inspect_chunks.py` prints the chunks of one page as a reader meets them; use both before changing the chunker. Retrieval mostly does not move when a text defect is fixed; those fixes are kept for what the reader sees. Tests are built from pages reported by hand.

Dehyphenation is decided by the document's own lexicon (format 3): a hyphen at a line end is dropped only when the joined word occurs elsewhere in the document.

### Chunk format 4: the page's layout is read before its text

The gutter of a two-column page is detected, the writing direction is read, margins are computed from each page's own height, and soft hyphens are handled.

### Chunk formats 5 and 6: a cut has to leave a finished sentence behind

Rows are split at sentence ends and cut candidates are ranked, which took chunks ending mid-sentence from 46% to 4.8%.

### Chunk formats 7 and 8: a table is read out cell by cell

Tables are found with pymupdf's `find_tables`, behind a prefilter for ruled lines and a fill test so prose is not mistaken for a grid. One table row becomes one text row, and a caption carries the table's shape.

### Chunk format 9: a table continued on the next page keeps its column names

A continuation table inherits the previous page's column names, under four guards that stop an unrelated table from inheriting them.

### Chunk format 14: a line that stops short of the margin ended on purpose

If the next word would have fitted on the line, the line break is real and is kept as a newline (`flow_margins`), so a list of short items does not arrive as one long line.

### Chunk format 15: a table the ruled-line detector cannot see is read from the layout model's grid

Tables ruled only around their header are read from the grid the `pymupdf-layout` model predicts. Inference costs about a second a page, so grids are cached per PDF hash under `data/private/layout/`. The model reads an upright copy of a rotated page. Two guards (`prose_columns`, `contents_listing`) stop it turning multi-column prose or a table of contents into a table.

### Chunk formats 16 to 18: a sideways page reads in its own direction, and a doubted space is decided by the document

A sideways page is read in its own reading order. A soft hyphen inside a cell is handled. A space the PDF may have invented is decided by the gaps relative to that line and by the document's own vocabulary.

### Chunk format 19 (and 20): a column is named once, and a paragraph cannot pass for a table

Column names are given once, in the caption, instead of on every row. `runs_across_rows` refuses a grid whose text reads as prose running across rows.

### Chunk format 21: the characters the encoder was fed but nobody could read

Control codes, accents detached from their letter (decided by their box), zero-width characters, U+FFFD and letters a font repurposed are repaired. Ligature codes are decided by a vote over the whole document.

### Chunking runs one document per core

A process pool, one document per core, because the work is Python-bound and the GIL serialises threads.

### Chunk output is not tracked

Chunk files are derived and content-hash gated. Bump `CHUNK_FORMAT_VERSION` whenever a rule changes, or the gate keeps stale chunks.

## Embedding

### Measured: every vector depended on which passages shared its batch

Padding was not masked correctly, so a passage's vector depended on what else was in its batch. Fixed, and `EMBED_FORMAT_VERSION` bumped to invalidate every bake. Sorting passages longest first is legitimate only now that a vector is independent of its batch. `bench.py` keeps a list of passages that exercise this.

### Closed: the page ceiling is 8192 tokens, and the batch is planned per call

The tokenizer truncated at 512, silently cutting page rows from page vectors. `MAX_PASSAGE_TOKENS` is now 8192, the encoder's window, and each call plans its batch by memory because attention is quadratic in length. The size sweeps, not the encoder, are what limit a chunk.

### Measured: a bigger batch is not a faster one

Throughput plateaus at a batch of about 64. One intra-op thread per physical core: letting onnxruntime take every hardware thread is 3.3x slower.

### fp16 weights bake 50x faster, and the substitution costs nothing measurable

The int8 graph is a CPU artefact: on a GPU its quantised operators have no kernels and onnxruntime splits the graph with Memcpy nodes. So the bake runs float weights on the GPU, while the server embeds queries with int8 on the CPU. Measured on the previous model (fp16 against int8): the substitution reorders about 10% of a top 10 (`compare_bakes.py`) and changes retrieval by nothing 117 queries can resolve. The current model publishes no fp16 graph, so its GPU artefact is fp32, a wider gap that has not been re-measured; `compare_bakes.py` is how to check it.

## Decided: duplicate renditions are ranked away, not deleted

The same guidance often exists as several renditions (a summary, the full text, a long supporting report). They retrieve against each other and the longest dominates by volume. Each document may carry a `family` and a `rendition`, from vocabularies in `corpus.toml`. Redundant renditions collapse to the best-scoring one, with the others offered as alternates. Companions (an annex, a form) never collapse. Editions never collapse: newest-wins was rejected so older editions stay reachable. An unknown rendition never collapses, and a rendition name in neither set fails at import. `check_titles` forbids two documents with the same title (accent-folded).

### Long documents

Three fixes were considered for long documents crowding out short ones: a per-document cap, length normalisation and grouping. The per-document cap of three passages (`PER_DOC`) ships, because it is the only one that cannot change which document ranks first. Length normalisation was not adopted. Grouping by document is a view of the same list (the nested view holds exactly the passage rows). A browse listing of the whole corpus exists.

## Decided: the facet vocabulary is closed, and what the reader reads is not what is stored

Facets and their vocabularies come from `corpus.toml`. `manifest.py` and `build_index.py` both refuse a value outside its vocabulary, because a wrong value hides a document from the filter it belongs in. Multi-valued facets match on ANY; the reader can tick several values in a popup with a search box, and ticked values move to the top when it opens (beware `.choice[hidden]`: a CSS rule can silently override the attribute). Slugs are stored and labels live in i18n; `check_search` refuses a slug with no label. Year is a range slider, recorded in the link only once narrowed. The value separator is shipped in `meta.json` rather than hardcoded twice.

The visibility tiers form a nested ladder; a blank tier passes. Under the last result, one notch wider is offered (a widened list can come back shorter, because folding changes). The advanced panel and the "?" help sit beside the search box.

### Metadata is filled, never overwritten

`manifest.py` fills blanks only, so a curated value survives a re-run. Issuer acronyms in `ISSUER_PATTERNS` are wrapped in `(?-i:)`: matched case-insensitively they hit ordinary words, and an import-time check now runs every pattern against innocent prose. DOI and ISBN are read from the file itself: XMP first, then the first two pages with a marker required; a page with two DOIs is refused; an ISBN must pass its check digit; an ISBN links to a WorldCat search. A `-` in the column means "looked, none found" (`NO_IDENTIFIER`), and `--rescan-identifiers` looks again.

### Finding where each document was published

`source_url` is found by a language-model lookup and then verified (fetch, Crossref or an archive capture), with a log of the confidence and the evidence. The page labels it "Source (AI)" so a reader knows how it was found.

## Decided: the viewer shows one page, one passage, and the way out of both

One passage is highlighted as a per-line fill with `mix-blend-mode: multiply`, on a real text layer. The passage's text sits in a `details` element under the pager (absent for restricted documents). A click fills the screen (one state, not a zoom ladder). A single "full PDF (N pages)" control opens the whole document. The pager sits under the page, and the passage in a result is a link.

## Decided: a document can be shown without being handed over

See "a restricted document is cut page by page" above: the front end hides the download, the stage keeps the file outside the web root, and the page service hands out one page at a time.

## Invariants enforced by gates rather than by tests

`verify_chunks.py` (no skipped PDF, no empty document, no box outside its page), `check_pdfs.py` (every served file is a PDF a browser opens; pymupdf renders HTML too, so an error page saved as a PDF was once chunked and served), `check_served.py` (no restricted file or word under the web root), `check_search.mjs` (self-retrieval against the real index; with `SVC` it compares the running service's `ranker_sha` and `index_sha` to the files on disk), `check_ui.mjs` (the words under the highlight; it found a flipped y axis the first time it ran), `check_chrome.mjs` (banner and footer), and `smoke_deployed.sh` against the running site. The lesson behind the last one: a gate has to run where the failure can happen. The first deploy came up healthy and answered every query with "service unavailable", because the deployed stack could not reach the encoder, which nothing on the build machine could see.

## Measured: the ranking on the server

Ranking runs in the search container, not in the browser: the index stays off the wire and the page stays light. Ranking on the server was measured with `scripts/measure_service.mjs`.

### First visit

About 136 KB gzipped, with no index download.

### Latency, one request at a time

p50 133 ms for a cached question, 211 ms for a new one (including the encoder).

### Fifty at once

The queue is bounded: a burst of 50 gets 503s with `Retry-After` beyond the depth, instead of everyone waiting longer. The queue yields with `setImmediate`; without it the bound never held, because ranking blocked the event loop before the next request could be refused.

### Memory, and where the shipped defaults come from

The 256 MB heap cap is load-bearing: without it V8 let the garbage from parsing per-document JSON grow past 1 GB before collecting (0.9 to 1.4 GB RSS after a hundred questions, 350 to 400 MB with the cap, same latency). The text cache holds 64 documents: 16 saved nothing visible, and 0 costs about 120 ms per answer.

### Still owed

The page service and the dev server have no stale-service check yet.

## Measured: what a reranker costs on a small server

On a small server memory, not CPU, is the limit: a cross-encoder peaked at 0.7 to 0.96 GB against about 1.05 GB free. One was built, deployed, crashed the server by exhausting its RAM, and was removed. A reranker survives only as the offline judge of the grid search.

## Docker networking: pin the subnet

Each stack pins its default network (`NETWORK_SUBNET`, default `172.16.242.0/24`). A host whose Docker default address pool is exhausted falls through to a fallback range that may have no outbound route, so a container comes up with no network and nothing says so. Two sites on one host pick two different /24s.

## Still open

- The metadata label's length (`--meta-fields title`).
- fp32-against-int8 bake substitution on the current encoder (`compare_bakes.py`).
- Figure queries in the evaluation set.
- Every browser but Chromium.
