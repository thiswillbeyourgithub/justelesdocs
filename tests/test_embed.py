"""The two things embed.py decides before it encodes anything.

`batch_feed` is the sibling's, not this repository's, but the bug it now covers was
found here and its effect landed here: a version-1 bake's vectors depended on which
other passages shared their batch of 64. The test loads the sibling the way embed.py
does, so it exercises the file the bake actually runs.

Written by Claude Code (Opus 5).
"""

from __future__ import annotations

from pathlib import Path

import pytest

MODEL_DIR = (Path(__file__).resolve().parent.parent.parent
             / "justelesRCP/models/jinaai/jina-embeddings-v5-text-small-retrieval")
SIBLING = (Path(__file__).resolve().parent.parent.parent
           / "justelesRCP/src/onnx_embed.py")


@pytest.fixture(scope="session")
def onnx_embed(embed):
    """The sibling encoder module, imported without onnxruntime installed.

    The `embed` fixture has already put a stub in sys.modules for onnxruntime,
    which this module imports at import time and uses only inside its functions,
    so `batch_feed` is reachable without the real package (see tests/run.py).
    """
    if not SIBLING.is_file():
        pytest.skip(f"{SIBLING} not checked out next to this repository")
    return embed.load_sibling_module(SIBLING, "onnx_embed")


@pytest.fixture(scope="session")
def tokenizer(onnx_embed):
    if not (MODEL_DIR / "tokenizer.json").is_file():
        pytest.skip("model not downloaded (scripts/download-model.sh)")
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))


SHORT = "court"
LONG = "une phrase nettement plus longue que la premiere ci-dessus"


@pytest.fixture(scope="session")
def padding_tokenizer(tokenizer):
    """The same tokenizer with padding forced ON.

    Which shipped tokenizer.json pads is a property of the MODEL and has already
    changed once: arctic's enabled it (pad_id 1, direction right), jina's enables
    nothing. The bug below can therefore no longer be provoked with the tokenizer as
    it comes, so it is provoked deliberately instead, and the guard stops depending
    on a file the next model replaces.
    """
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    tok.enable_padding()
    return tok


def test_batch_feed_squares_rows_the_tokenizer_left_ragged(onnx_embed, tokenizer):
    """jina's tokenizer.json sets no padding, so encode_batch returns ragged rows.

    The feed still has to be rectangular, because that is what the graph takes, and
    squaring it is `batch_feed`'s job rather than the tokenizer's.
    """
    rows = tokenizer.encode_batch([SHORT, LONG])
    feed = onnx_embed.batch_feed(tokenizer, [SHORT, LONG], max_len=512)
    assert feed["input_ids"].shape == feed["attention_mask"].shape
    assert feed["input_ids"].shape[1] == max(len(row.ids) for row in rows)
    assert int(feed["attention_mask"][0].sum()) == len(rows[0].ids)


def test_padding_is_masked_out(onnx_embed, tokenizer):
    """A short passage's mask must not grow because a long one shares its batch."""
    alone = onnx_embed.batch_feed(tokenizer, [SHORT], max_len=512)
    together = onnx_embed.batch_feed(tokenizer, [SHORT, LONG], max_len=512)
    assert int(alone["attention_mask"].sum()) == int(together["attention_mask"][0].sum())
    # And the mask marks exactly the real tokens, wherever the row ends.
    real = len(tokenizer.encode(SHORT).ids)
    width = together["attention_mask"].shape[1]
    assert together["attention_mask"][0].tolist() == [1] * real + [0] * (width - real)


def test_padding_is_masked_out_when_the_tokenizer_pads(onnx_embed, padding_tokenizer):
    """The bug itself: the mask must be the tokenizer's, not `1 * len(ids)`.

    Once the tokenizer has padded, `len(ids)` IS the padded length, so rebuilding the
    mask from it declares every pad token real and the transformer attends to a run of
    them. Measured at the time on 512 corpus chunks, batch 1 against batch 64: cosine
    of a passage against itself 0.987 mean and 0.920 worst, against 0.99999+ after.
    """
    rows = padding_tokenizer.encode_batch([SHORT, LONG])
    assert len(rows[0].ids) == len(rows[1].ids)          # the premise, forced on
    feed = onnx_embed.batch_feed(padding_tokenizer, [SHORT, LONG], max_len=512)
    marked = int(feed["attention_mask"][0].sum())
    assert marked == sum(rows[0].attention_mask)
    assert marked < len(rows[0].ids)                     # i.e. the padding is inert


def test_masked_positions_hold_the_pad_token(onnx_embed, tokenizer):
    """Masked positions carry one inert id, whichever id the tokenizer names.

    arctic declared pad_id 1; jina declares no padding at all, and `batch_feed` then
    falls back to 0. With a correct mask this is arithmetic nobody reads, which is why
    the assertion is about uniformity rather than about the value.
    """
    feed = onnx_embed.batch_feed(tokenizer, [SHORT, LONG], max_len=512)
    ids, mask = feed["input_ids"][0], feed["attention_mask"][0]
    pad_id = (tokenizer.padding or {}).get("pad_id", 0)
    assert set(ids[mask == 0].tolist()) <= {pad_id}


def test_the_tokenizer_imposes_no_ceiling_of_its_own(tokenizer):
    """The premise of the second bug, and the reason it cannot recur as it was.

    arctic's tokenizer.json capped rows at 512, applied inside `encode_batch` before
    anything here saw the ids, so a caller asking for 1024 tokens silently got 512
    until the Encoder began raising the limit at construction. jina's sets no
    truncation, so the only ceiling is the one the Encoder passes. If this ever fails,
    a new model has brought its own cap back and that wiring needs checking again.
    """
    assert tokenizer.truncation is None
    assert len(tokenizer.encode("phrase " * 2000).ids) > 512


def test_max_len_cuts_every_row(onnx_embed, tokenizer):
    feed = onnx_embed.batch_feed(tokenizer, ["un texte assez long " * 50], max_len=8)
    assert feed["input_ids"].shape[1] == 8


class _FakeSession:
    """An ONNX session stand-in whose hidden state is a function of the ids alone.

    It also records the number of REAL tokens in every row it is handed, in the
    order it was handed them, which is the only way to see from outside whether
    `encode` reordered its input before slicing it into batches.
    """

    HIDDEN = 4

    def __init__(self):
        self.seen: list[int] = []

    def run(self, outputs, feed):  # noqa: ARG002 - the output name is irrelevant here
        import numpy as np
        ids, mask = feed["input_ids"], feed["attention_mask"]
        self.seen.extend(int(n) for n in mask.sum(axis=1))
        hidden = np.zeros((ids.shape[0], ids.shape[1], self.HIDDEN), dtype=np.float32)
        for k in range(self.HIDDEN):
            hidden[:, :, k] = ids.astype(np.float32) + k + 1
        return [hidden]


def _fake_encoder(onnx_embed, tokenizer):
    """An Encoder with a fake session: everything `encode` touches, no weights.

    Mean pooling on purpose. It is the pooling that reads every position rather
    than one, so a row whose padding was counted as real text comes out with a
    different vector, which is what makes the order test below meaningful.
    """
    encoder = object.__new__(onnx_embed.Encoder)
    encoder.tokenizer = tokenizer
    encoder.session = _FakeSession()
    encoder.pooling = "mean"
    encoder.passage_prefix = ""
    encoder._input_names = {"input_ids", "attention_mask"}
    encoder._token_output = "last_hidden_state"
    encoder._out_dim = None
    encoder.dim = _FakeSession.HIDDEN
    encoder.full_dim = _FakeSession.HIDDEN
    return encoder


# Decreasing on purpose: the natural order is the worst case for the sort, so a
# test that sees the sorted order cannot be seeing the input order by accident.
LADDER = ["mot " * (40 - 4 * i) for i in range(10)]


def test_encode_groups_similar_lengths_before_batching(onnx_embed, tokenizer):
    """The 12% the sort buys (108.6 -> 121.7 sections/s on 3816 real RCP sections).

    A batch is padded to its own longest row and attention is quadratic in that
    padded length, so mixing a 40-word row with a 4-word one makes the short one
    pay for the long one. Sorting first is the whole optimisation.
    """
    encoder = _fake_encoder(onnx_embed, tokenizer)
    encoder.encode(LADDER, batch_size=3)
    assert encoder.session.seen == sorted(encoder.session.seen)


def test_encode_returns_the_rows_in_the_order_it_was_given(onnx_embed, tokenizer):
    """The sort is invisible to the caller, which pairs these rows with its chunks.

    Getting this wrong is worse than the slow version: every vector would still be
    a real vector of a real passage, just filed under someone else's passage, and
    nothing downstream can tell.
    """
    import numpy as np
    encoder = _fake_encoder(onnx_embed, tokenizer)
    together = encoder.encode(LADDER, batch_size=3)
    alone = np.vstack([encoder.encode([text], batch_size=3) for text in LADDER])
    assert np.allclose(together, alone, atol=1e-6)


def test_a_group_that_fits_one_batch_is_left_alone(onnx_embed, tokenizer):
    """What keeps this repository's bake unaffected.

    `encode_group` plans its own groups against a memory budget and then asks for
    exactly one batch per call, so there is never anything to regroup. Reordering
    a single batch would change nothing anyway, since the whole batch pads to the
    same length either way.
    """
    encoder = _fake_encoder(onnx_embed, tokenizer)
    encoder.encode(LADDER, batch_size=len(LADDER))
    natural = [len(tokenizer.encode(text).ids) for text in LADDER]
    assert encoder.session.seen == natural


def test_batch_feed_counts_what_it_had_to_cut(onnx_embed, tokenizer):
    """The ceiling is a claim about the corpus, so the bakes check it rather than trust it.

    A truncated row embeds without complaining; the only symptom is a passage that
    cannot be retrieved by the half of itself that was dropped, which is invisible
    until someone searches for exactly that. `embed-rcp.py` reads this at the end of
    every run and warns.
    """
    onnx_embed.truncation_report(reset=True)
    onnx_embed.batch_feed(tokenizer, [SHORT, "phrase " * 500], max_len=16)
    cut, longest = onnx_embed.truncation_report(reset=True)
    assert cut == 1
    assert longest > 500
    onnx_embed.batch_feed(tokenizer, [SHORT, LONG], max_len=1024)
    cut, longest = onnx_embed.truncation_report(reset=True)
    assert cut == 0
    assert 0 < longest < 1024


def test_the_sibling_ceiling_is_the_one_the_rcp_bake_needs(onnx_embed):
    """1024 against a corpus whose longest section is 307 tokens (median 123, p90 182).

    It was 192 until 2026-09-21, which cut roughly the top 10% of RCP sections. The
    number is over three times the longest section rather than snug against it
    because the next corpus refresh does not get to silently re-open that hole.
    """
    assert onnx_embed.PASSAGE_MAX_TOKENS == 1024


def test_the_token_ceiling_clears_the_longest_page_in_the_corpus(embed):
    """The first of the two chunker defects, fixed on 2026-09-19.

    The page bake feeds whole pages, the longest of which tokenises to 4481, and the
    ceiling used to be 1024: everything past the first quarter of that page went
    through no forward pass at all and could not be retrieved, silently, because a
    truncated row embeds perfectly happily. 8192 is the model's own position limit,
    so at this value the corpus cannot be cut by our choice of budget, only by the
    model's. Lowering it again is a decision to stop indexing the tail of a page.
    """
    assert embed.MAX_PASSAGE_TOKENS == 8192


def test_padded_lengths_counts_tokens_and_not_padding(embed, tokenizer):
    """The mistake this function exists to avoid: `len(ids)` is the padded length.

    Sizing a batch on `len(ids)` would give every short row in a document the cost of
    the document's longest, whenever the tokenizer pads (arctic's did, jina's does
    not). Counting the mask instead is right either way.
    """
    texts = ["court", "un texte nettement plus long que le premier, avec des mots"]
    lengths = embed.padded_lengths(texts, tokenizer=tokenizer, ceiling=512)
    assert lengths[0] < lengths[1]
    assert lengths[0] == len(tokenizer.encode(texts[0]).ids)


def test_padded_lengths_stops_at_the_ceiling(embed, tokenizer):
    """Mirrors batch_feed's own ids[:max_len]: a row cannot cost more than the cut."""
    assert embed.padded_lengths(["phrase " * 2000], tokenizer=tokenizer, ceiling=8) == [8]


def test_a_row_costs_far_more_than_twice_as_much_when_it_is_twice_as_long(embed):
    """Attention is quadratic in the padded length, which is the whole reason the
    batch cannot be sized in rows. Measured on this card, a row of 4096 tokens costs
    1536 MiB against 384 at 2048, so anything close to linear here is a bug."""
    assert embed.row_cost_mib(4096) > 3.5 * embed.row_cost_mib(2048)
    # And the model stays within a factor of two of the measured 1536 and 6144 MiB.
    assert 768 < embed.row_cost_mib(4096) < 3072
    assert 3072 < embed.row_cost_mib(8192) < 12288


def test_memory_budget_scales_with_the_free_memory(embed):
    roomy, why = embed.memory_budget(24_000)
    busy, _ = embed.memory_budget(5_000)
    assert busy < roomy
    assert "free allows" in why
    # The budget is only useful if a full-length row fits in it at all, which is what
    # lets the page bake run rather than fail on its longest page.
    assert roomy > embed.row_cost_mib(embed.MAX_PASSAGE_TOKENS)


def test_memory_budget_never_goes_negative(embed):
    """Less free VRAM than the session itself costs. The arithmetic goes below zero
    and plan_batches would then be dividing groups against a negative budget."""
    budget, why = embed.memory_budget(1)
    assert budget == embed.MIN_BUDGET_MIB
    assert "tight" in why


def test_memory_budget_without_a_gpu(embed):
    budget, why = embed.memory_budget(None)
    assert budget == embed.FALLBACK_BUDGET_MIB
    assert "nvidia-smi" in why


def test_a_batch_stops_at_the_throughput_plateau(embed):
    """BATCH_PLATEAU is measured, not a memory limit: budget left over is not spent."""
    assert embed.plan_batches([250] * 200, budget_mib=1e9) == [64, 64, 64, 8]


def test_long_rows_get_small_batches_and_short_rows_large_ones(embed):
    """The whole point: one document, one budget, two very different batch sizes.

    Sorted longest first, as the bake feeds them. 8000-token rows cost about 6 GiB
    each, so a 14 GiB budget takes two at a time; the 200-token rows that follow cost
    about 17 MiB each and would fit hundreds, so they run at the plateau instead.
    """
    lengths = [8000, 8000, 8000] + [200] * 70
    sizes = embed.plan_batches(lengths, budget_mib=14_000)
    assert sizes[0] == 2
    assert max(sizes) == 64
    assert sum(sizes) == len(lengths)


def test_a_row_bigger_than_the_whole_budget_still_gets_embedded(embed):
    """There is no call smaller than one row, so the alternative is not embedding it."""
    assert embed.plan_batches([8192, 8192], budget_mib=100) == [1, 1]


def test_every_row_lands_in_exactly_one_batch(embed):
    """A planner that lost or repeated a row would produce a bake of the wrong height
    and `matrix[order] = grouped` would raise or, worse, misalign."""
    lengths = [7000, 5000, 5000, 900, 900, 900, 120, 120, 40]
    sizes = embed.plan_batches(lengths, budget_mib=12_000)
    assert sum(sizes) == len(lengths)
    assert all(size >= 1 for size in sizes)


def test_raising_the_ceiling_did_not_shrink_an_ordinary_batch(embed):
    """Why the sizing rule changed in the same commit as the ceiling.

    The old rule sized every batch for a worst-case row, so moving the ceiling from
    1024 to 8192 would have cut the batch from the plateau of 64 to one row, since a
    worst case eight times longer is sixty-four times more expensive. Sizing from the
    rows in hand keeps a corpus of 250-token chunks at the plateau, and only the
    pages that really are long pay.
    """
    budget, _ = embed.memory_budget(23_500)
    assert embed.plan_batches([250] * 200, budget_mib=budget) == [64, 64, 64, 8]


def test_an_allocator_refusal_is_retried_at_half_the_batch(embed, monkeypatch):
    """A 40-minute bake must not die because the memory model was 10% optimistic on
    one unusual document. Anything that is not an allocator refusal still raises."""
    import numpy as np

    class Card:
        passage_prefix = "passage: "

        def __init__(self, limit):
            self.limit = limit
            self.tried = []

        def encode(self, texts, *, prefix, batch_size, max_len):
            self.tried.append(batch_size)
            if batch_size > self.limit:
                raise RuntimeError("Failed to allocate memory for requested buffer")
            return np.zeros((len(texts), 4), dtype=np.float32)

    card = Card(limit=2)
    vectors, splits = embed.encode_group(card, ["a"] * 8, max_len=512)
    assert vectors.shape == (8, 4)
    assert splits == 2 and card.tried[:3] == [8, 4, 2]

    class Broken(Card):
        def encode(self, texts, *, prefix, batch_size, max_len):
            raise RuntimeError("the graph does not run")

    with pytest.raises(RuntimeError, match="does not run"):
        embed.encode_group(Broken(limit=0), ["a"] * 4, max_len=512)

def test_meta_fields_are_stripped_and_blanks_dropped(embed):
    assert embed.meta_field_names(" title , issuer ,, year ", "meta") == \
        ("title", "issuer", "year")


def test_an_empty_meta_fields_is_refused_before_the_model_loads(embed):
    """It used to reach the first document as an IndexError out of the encode loop."""
    import click

    with pytest.raises(click.ClickException) as caught:
        embed.meta_field_names(" , ", "meta")
    assert "--meta-fields" in str(caught.value)
    # The other variants never read it, so an empty list is not their problem.
    assert embed.meta_field_names("", "plain") == ()


def test_bake_params_come_from_the_profile_the_encoder_uses(embed, onnx_embed):
    """`--check` hashes without a built encoder, so its parameters must be the
    ones `Encoder.__init__` would report for the runtime model, or the check and
    the bake would disagree about what is stale."""
    profile = onnx_embed._profile(onnx_embed.RUNTIME_MODEL)
    params = embed.bake_params(onnx_embed, variant="meta", weights="model_fp16.onnx",
                               fields=("title", "issuer", "year"))
    assert params["model"] == onnx_embed.RUNTIME_MODEL
    assert params["pooling"] == profile["pooling"]
    assert params["passage_prefix"] == profile["passage"]
    assert params["weights"] == "model_fp16.onnx"
    assert params["meta_fields"] == "title,issuer,year"
    # Only the meta variant carries the field list, so older bakes still match.
    assert "meta_fields" not in embed.bake_params(onnx_embed, variant="plain", weights="",
                                                  fields=())


def test_cached_digest_trusts_only_a_readable_vector_file(embed, tmp_path):
    import numpy as np

    assert embed.cached_digest(tmp_path / "missing.npz") is None
    (tmp_path / "junk.npz").write_bytes(b"not a zip")
    assert embed.cached_digest(tmp_path / "junk.npz") is None
    np.savez_compressed(tmp_path / "ok.npz", vectors=np.zeros((1, 4)), src_hash=np.array("abc"),
                        chunks_hash=np.array("c"))
    assert embed.cached_digest(tmp_path / "ok.npz") == "abc"


def test_a_vector_file_without_chunks_hash_needs_baking(embed, tmp_path):
    """build_index.py refuses such a file, so the bake must not call it current."""
    import numpy as np
    np.savez_compressed(tmp_path / "old.npz", vectors=np.zeros((1, 4)), src_hash=np.array("abc"))
    assert embed.cached_digest(tmp_path / "old.npz") is None


def test_check_reports_a_missing_bake_and_accepts_a_current_one(embed, onnx_embed, tmp_path):
    """What deploy.sh branches on: exit 1 while a document has no current
    vectors, exit 0 once it has, with no model loaded either way."""
    import json

    import numpy as np
    from click.testing import CliRunner

    chunks = tmp_path / "chunks"
    chunks.mkdir()
    payload = {"file": "a.pdf", "chunks": [{"text": "un passage"}, {"text": "un autre"}]}
    (chunks / "a.json").write_text(json.dumps(payload), encoding="utf-8")
    args = ["--chunks", str(chunks), "--out", str(tmp_path / "vectors"), "--variant", "plain",
            "--onnx-embed", str(SIBLING), "--check"]

    missing = CliRunner().invoke(embed.main, args)
    assert missing.exit_code == 1, missing.output

    params = embed.bake_params(onnx_embed, variant="plain", weights="", fields=())
    digest = embed.source_hash(["un passage", "un autre"], params)
    (tmp_path / "vectors" / "plain").mkdir(parents=True)
    np.savez_compressed(tmp_path / "vectors" / "plain" / "a.npz",
                        vectors=np.zeros((2, 4)), src_hash=np.array(digest),
                        chunks_hash=np.array("c"))
    current = CliRunner().invoke(embed.main, args)
    assert current.exit_code == 0, current.output

    # A changed text is stale again: the gate is content, not presence.
    payload["chunks"][0]["text"] = "un passage corrigé"
    (chunks / "a.json").write_text(json.dumps(payload), encoding="utf-8")
    assert CliRunner().invoke(embed.main, args).exit_code == 1


# --------------------------------------------------------------------------------
# passage_texts: where the label sits.
#
# Untested until 2026-09-25, when the function grew from two layouts to five. The
# reason it needs covering now is that the encoder changed: Snowflake arctic pooled
# on CLS, so a label only ever made sense as a PREFIX and every prefix measurement
# in DESIGN.md was taken that way; jina-embeddings-v5 pools on the LAST token, so a
# SUFFIX now sits where the pooled state is read. Comparing the two placements is a
# real experiment, and it is only a fair one if each layout puts the label exactly
# where it claims to and leaves the chunk's own text untouched.
# --------------------------------------------------------------------------------

DOC = "INSEE_LGT-enquete&logement2017.pdf"
BODY = "Le logement doit être suivi."
# The filename stem with `_` and `-` opened into spaces, which is the whole of what
# the `title` variant contributes and, per DESIGN.md, the accidental reason it wins:
# it reads like the keyword list a reader types rather than a formal sentence.
STEM = "INSEE LGT enquete&logement2017"
TITLE = "Logements : enquête et résultats"


@pytest.fixture
def payload():
    return {"file": DOC, "chunks": [{"text": BODY}, {"text": "Un autre passage."}]}


@pytest.fixture
def one_row():
    return {DOC: {"title": TITLE, "issuer": "INSEE", "year": "2017"}}


def test_plain_hands_over_the_chunk_text_untouched(embed, payload):
    """`plain` is the control arm, so it must add nothing at all."""
    assert embed.passage_texts(payload, variant="plain") == [BODY, "Un autre passage."]


def test_plain_needs_no_manifest(embed, payload):
    """The control arm has to work on a corpus whose manifest is not curated yet."""
    assert embed.passage_texts(payload, variant="plain", manifest=None)[0] == BODY


def test_a_figure_chunk_is_embedded_after_its_kind_word_in_its_own_language(embed, one_row):
    """A described table never says it is a table, and a reader types "tableau"."""
    chunks = [{"text": "Plomb : 0,6 à 0,8.", "figure": {"kind": "table", "language": "fr"}},
              {"text": "Lead: 0.6 to 0.8.", "figure": {"kind": "table", "language": "en"}},
              {"text": "Une photo.", "figure": {"kind": "photo", "language": "fr"}},
              {"text": BODY}]
    texts = embed.passage_texts({"file": DOC, "chunks": chunks}, variant="meta",
                                manifest=one_row)
    assert texts == [f"{TITLE} (INSEE, 2017). Tableau : Plomb : 0,6 à 0,8.",
                     f"{TITLE} (INSEE, 2017). Table: Lead: 0.6 to 0.8.",
                     f"{TITLE} (INSEE, 2017). Figure : Une photo.",
                     f"{TITLE} (INSEE, 2017). {BODY}"]
    assert chunks[0]["text"] == "Plomb : 0,6 à 0,8."   # what the reader sees is untouched


def test_title_prepends_the_filename_stem_with_punctuation_opened_up(embed, payload):
    assert embed.passage_texts(payload, variant="title")[0] == f"{STEM}. {BODY}"


def test_meta_prepends_the_curated_title_with_issuer_and_year(embed, payload, one_row):
    """The shipped layout. This string is what `meta-gpu` on disk was baked from."""
    assert (embed.passage_texts(payload, variant="meta", manifest=one_row)[0]
            == f"{TITLE} (INSEE, 2017). {BODY}")


def test_titleend_and_metaend_move_the_same_labels_to_the_other_end(embed, payload,
                                                                   one_row):
    """The suffix arms carry an IDENTICAL label, only relocated.

    If the label itself differed between the prefix and suffix arms, the experiment
    would be measuring two things at once and could not attribute a difference to
    placement, which is the only thing it is meant to resolve.
    """
    assert embed.passage_texts(payload, variant="titleend")[0] == f"{BODY} {STEM}."
    assert (embed.passage_texts(payload, variant="metaend", manifest=one_row)[0]
            == f"{BODY} {TITLE} (INSEE, 2017).")


def test_both_puts_the_curated_label_first_and_the_filename_stem_last(embed, payload,
                                                                     one_row):
    """The hypothesis arm: the stem gets the end last-token pooling reads."""
    assert (embed.passage_texts(payload, variant="both", manifest=one_row)[0]
            == f"{TITLE} (INSEE, 2017). {BODY} {STEM}.")


# The English title of the same document, as data/TITLES_OTHER.tsv holds it.
TITLE_OTHER = "Housing: survey and results"


@pytest.fixture
def one_row_bilingual():
    return {DOC: {"title": TITLE, "title_other": TITLE_OTHER, "lang_other": "en",
                  "issuer": "INSEE", "year": "2017"}}


def test_metabi_puts_both_titles_at_the_head_own_language_first(embed, payload,
                                                               one_row_bilingual):
    """The bilingual label, and the order matters twice over.

    The document's own title comes first, so the string is a superset of what the
    shipped `meta` bake feeds the encoder rather than a different label: the only
    difference between the two bakes is the words that were added, which is what
    makes the comparison attributable. One bracket block, not two, because issuer
    and year are language-neutral (`INSEE`, `2017`) and duplicating them would spend
    tokens on nothing.
    """
    assert (embed.passage_texts(payload, variant="metabi",
                                manifest=one_row_bilingual)[0]
            == f"{TITLE} / {TITLE_OTHER} (INSEE, 2017). {BODY}")


def test_metabi_prints_one_title_when_there_is_nothing_to_translate(embed, payload):
    """35 titles are a filename, which is the same string in both languages.

    Repeating it would spend prefix tokens on nothing and would also tell the encoder
    those words weigh twice what they do, which is the opposite of the point.
    """
    rows = {DOC: {"title": TITLE, "title_other": TITLE, "issuer": "INSEE", "year": "2017"}}
    assert (embed.passage_texts(payload, variant="metabi", manifest=rows)[0]
            == f"{TITLE} (INSEE, 2017). {BODY}")


def test_metabi_refuses_a_document_with_no_translation(embed, payload, one_row):
    """A silent fallback here would bake half a bilingual index and measure nothing.

    A missing translation is the failure to expect while the file is being written by
    hand for every document, and it has to name the document: the alternative is a
    bake where an unknown subset of chunks carries a monolingual prefix, which scores
    somewhere between the two arms and cannot be attributed to either.
    """
    with pytest.raises(Exception, match="no translated title"):
        embed.passage_texts(payload, variant="metabi", manifest=one_row)


def test_metabi_with_no_issuer_or_year_is_just_the_two_titles(embed, payload):
    """The bracket block is optional in the bilingual layout too."""
    rows = {DOC: {"title": TITLE, "title_other": TITLE_OTHER, "issuer": "", "year": ""}}
    assert (embed.passage_texts(payload, variant="metabi", manifest=rows)[0]
            == f"{TITLE} / {TITLE_OTHER}. {BODY}")


def test_load_manifest_merges_the_other_language_titles(embed, tmp_path):
    """The translations arrive as extra columns on the manifest row.

    Merging at load time is what keeps `document_digests` and `passage_texts` taking
    one dictionary instead of two, and it is also what makes an edited translation
    invalidate exactly that document's vectors: the per-document hash covers the text
    the layout builds from this row.
    """
    manifest = tmp_path / "MANIFEST.tsv"
    manifest.write_text("file\ttitle\tissuer\tyear\n"
                        f"{DOC}\t{TITLE}\tINSEE\t2017\n", encoding="utf-8")
    titles = tmp_path / "TITLES_OTHER.tsv"
    titles.write_text(f"file\tlang\ttitle\n{DOC}\ten\t{TITLE_OTHER}\n",
                      encoding="utf-8")
    rows = embed.load_manifest(manifest, titles)
    assert rows[DOC]["title_other"] == TITLE_OTHER
    assert rows[DOC]["lang_other"] == "en"
    # And the manifest reads the same when the translations are not asked for, so the
    # shipped `meta` bake cannot change because this file was added.
    assert "title_other" not in embed.load_manifest(manifest)[DOC]


def test_load_manifest_ignores_a_translation_for_a_document_that_left_the_corpus(
        embed, tmp_path):
    """A stale row is a curation artefact, not a reason to fail a bake.

    Documents move between `data/GUIDELINES/` and the discard buckets, and the
    translation file is hand-written, so it will outlive some of its rows.
    """
    manifest = tmp_path / "MANIFEST.tsv"
    manifest.write_text(f"file\ttitle\n{DOC}\t{TITLE}\n", encoding="utf-8")
    titles = tmp_path / "TITLES_OTHER.tsv"
    titles.write_text(f"file\tlang\ttitle\n{DOC}\ten\t{TITLE_OTHER}\n"
                      "gone.pdf\tfr\tUn titre\n", encoding="utf-8")
    assert set(embed.load_manifest(manifest, titles)) == {DOC}


def offered_variants(embed) -> tuple[str, ...]:
    """Every layout --variant offers, read off the CLI rather than listed again.

    Two tests sweep "all the layouts" and both listed them by hand, so both missed
    `metabi` on the day it was added. What a user can ask for is what these have to
    be complete over, and click knows it.
    """
    param = next(p for p in embed.main.params if p.name == "variant")
    return tuple(param.type.choices)


def test_every_layout_keeps_the_chunk_text_verbatim_and_whole(embed, payload,
                                                              one_row_bilingual):
    """A label is added around the text, never spliced into or trimmed off it.

    This is what lets metadata be chosen after chunking: the chunker's token budget
    is spent entirely on document text, so no chunk shrinks because a title is long.
    """
    for variant in offered_variants(embed):
        for text, chunk in zip(embed.passage_texts(payload, variant=variant,
                                                  manifest=one_row_bilingual),
                               payload["chunks"]):
            assert chunk["text"] in text, variant
    # And the count is preserved: one string out per chunk in, for every layout.
    for variant in offered_variants(embed):
        assert len(embed.passage_texts(payload, variant=variant,
                                       manifest=one_row_bilingual)) == 2, variant


def test_a_blank_issuer_prints_no_empty_brackets(embed, payload):
    """42 corpus rows have no issuer because a journal article has no issuing body.

    `data/CURATED_ISSUERS.tsv` (read by manifest.py) records that as a decision rather than a gap,
    so the label must read "Title (2017)" and never "Title (, 2017)".
    """
    rows = {DOC: {"title": TITLE, "issuer": "", "year": "2017"}}
    assert (embed.passage_texts(payload, variant="meta", manifest=rows)[0]
            == f"{TITLE} (2017). {BODY}")


def test_a_label_with_no_issuer_and_no_year_carries_no_brackets_at_all(embed, payload):
    rows = {DOC: {"title": TITLE, "issuer": "", "year": ""}}
    assert (embed.passage_texts(payload, variant="meta", manifest=rows)[0]
            == f"{TITLE}. {BODY}")


def test_a_curated_layout_refuses_a_document_with_no_manifest_row(embed, payload):
    """Refuse rather than silently bake a weaker label.

    Falling back to the filename stem here would hide a missing manifest row inside
    a bake that looks finished, and the fallback is the OTHER arm of the experiment,
    so it would also quietly contaminate the comparison.
    """
    import click
    for variant in ("meta", "metaend", "both"):
        with pytest.raises(click.ClickException, match="has no row in the manifest"):
            embed.passage_texts(payload, variant=variant, manifest={})


def test_meta_fields_choose_what_goes_in_the_brackets(embed, payload, one_row):
    """`--meta-fields` is part of the bake's cache key, so it must really bite."""
    assert (embed.passage_texts(payload, variant="meta", manifest=one_row,
                                meta_fields=("title", "year"))[0]
            == f"{TITLE} (2017). {BODY}")


def test_a_missing_title_falls_back_to_the_RAW_filename_stem(embed, payload):
    """A row can exist with an empty title; the label still has to name something.

    Note the fallback keeps `_` and `-` as they are, so it is NOT the same string
    the `title` variant builds from the same filename ("INSEE LGT enquete&logement2017",
    punctuation opened into spaces). Pinned here because the two are one edit away
    from being confused, and a `meta` bake whose label silently became the `title`
    label would make the placement comparison measure nothing.
    """
    rows = {DOC: {"title": "", "issuer": "INSEE", "year": "2017"}}
    assert (embed.passage_texts(payload, variant="meta", manifest=rows)[0]
            == f"INSEE_LGT-enquete&logement2017 (INSEE, 2017). {BODY}")
    assert Path(DOC).stem != STEM, "the two stem spellings really do differ"


# --------------------------------------------------------------------------------
# CURATED_VARIANTS: the bug that cost a 90-minute bake.
#
# `metaend` and `both` were added to passage_texts' layouts table but not to the
# four places that asked `variant == "meta"`, so `embed.py` never loaded the
# manifest for them and both bakes died one document in with "no row in the
# manifest" for a file whose row was sitting in MANIFEST.tsv all along. The fix is
# one frozenset; these tests exist so the set cannot drift from the table again.
# --------------------------------------------------------------------------------

def test_the_curated_set_is_exactly_the_layouts_that_need_a_manifest_row(embed,
                                                                        payload):
    """Derive the answer from behaviour, not from repeating the list.

    Asserting a hardcoded set here would restate `CURATED_VARIANTS` rather than
    check it, and a layout added tomorrow would sail past. Asking each
    variant what it does with an EMPTY manifest is the real question: the ones
    that refuse are the ones needing a row, and that set must be the constant.
    """
    import click
    refuses = set()
    for variant in offered_variants(embed):
        try:
            embed.passage_texts(payload, variant=variant, manifest={})
        except click.ClickException:
            refuses.add(variant)
    assert refuses == set(embed.CURATED_VARIANTS)


def test_every_curated_variant_refuses_empty_meta_fields(embed):
    """`--meta-fields ""` has nothing to add, whichever end the label goes on."""
    import click
    for variant in embed.CURATED_VARIANTS:
        with pytest.raises(click.ClickException, match="--meta-fields is empty"):
            embed.meta_field_names("", variant)
    # ... and a variant that builds its label from the filename does not care.
    assert embed.meta_field_names("", "title") == ()


def test_meta_fields_land_in_the_cache_key_of_every_curated_variant(embed,
                                                                   onnx_embed):
    """Otherwise a re-bake with different columns would reuse the old vectors.

    This was the quieter half of the same drift: `metaend` and `both` hashed as
    though the field choice did not change their text, which it plainly does.
    """
    for variant in embed.CURATED_VARIANTS:
        params = embed.bake_params(onnx_embed, variant=variant, weights="",
                                   fields=("title", "year"))
        assert params["meta_fields"] == "title,year", variant
    # `plain` and `title` keep their pre-option hash, so old bakes still match.
    plain = embed.bake_params(onnx_embed, variant="plain", weights="", fields=("title",))
    assert "meta_fields" not in plain


# --- the label's delimiter, the 2x2 swept on 2026-09-27 -----------------------------
#
# The label was separated from the passage by ". " and nothing else for a year, which
# makes it read as the passage's own opening sentence. These three layouts are the
# other cells of the 2x2 (quote the title or not, crossed with ". " or a newline).
# What they measured is in DESIGN.md, "the label's delimiter".

def test_metaq_wraps_the_title_in_straight_quotes(embed, payload, one_row):
    """Quotes go around the TITLE only, never around issuer and year.

    A title is the part whose end is ambiguous, and several carry brackets of their
    own ("... (rapport annuel, INSEE 2004)"), which is exactly the case a reader of the
    string cannot parse. The issuer and year bracket is already delimited.
    """
    out = embed.passage_texts(payload, variant="metaq", manifest=one_row)[0]
    assert out.startswith(f'"{TITLE}" (INSEE, 2017). ')
    assert out.endswith(BODY)


def test_metanl_ends_the_label_with_a_newline_and_no_full_stop(embed, payload, one_row):
    """The newline REPLACES the ". ", rather than being added after it."""
    out = embed.passage_texts(payload, variant="metanl", manifest=one_row)[0]
    assert out == f"{TITLE} (INSEE, 2017)\n{BODY}"
    assert ". \n" not in out


def test_metaqnl_does_both(embed, payload, one_row):
    out = embed.passage_texts(payload, variant="metaqnl", manifest=one_row)[0]
    assert out == f'"{TITLE}" (INSEE, 2017)\n{BODY}'


def test_the_shipped_meta_label_is_byte_identical_after_the_sweep(embed, payload, one_row):
    """The one string this file must pin.

    `meta` is what dist/index is baked from, and its src_hash is a hash of exactly
    this text. Adding a separator column to the layouts table must not move it, or
    every document in the corpus silently needs re-baking.
    """
    out = embed.passage_texts(payload, variant="meta", manifest=one_row)
    assert out == [f"{TITLE} (INSEE, 2017). {BODY}", f"{TITLE} (INSEE, 2017). Un autre passage."]


def test_a_quoted_layout_still_needs_a_manifest_row(embed, payload):
    """The quoted layouts are curated ones, so they belong in CURATED_VARIANTS."""
    import click  # imported here like every other click assertion in this file

    for variant in ("metaq", "metanl", "metaqnl"):
        assert variant in embed.CURATED_VARIANTS, variant
        with pytest.raises(click.ClickException, match="has no row in the manifest"):
            embed.passage_texts(payload, variant=variant, manifest={})
