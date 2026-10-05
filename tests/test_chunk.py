"""The command-line side of scripts/chunk.py: its document lists and its cache key.

`chunk_params` decides which bakes a change invalidates, and its failure is silent
in both directions, so it is tested parameter by parameter. The chunking stages
themselves are tested per stage: test_chunk_text.py (text repair, glyphs,
sentences), test_chunk_tables.py, test_chunk_extract.py (rows, columns, margins,
boilerplate) and test_chunk_pack.py (packing, boundaries, headings, sections).
test_chunk_golden.py pins the output of the whole pipeline.

Written by Claude Code (Opus 5). Split by Claude Code (Opus 5.5), along the
lib/chunk_*.py stages.
"""

from __future__ import annotations


def test_a_document_list_ignores_blank_lines_and_comments(chunk, tmp_path):
    """A generated list carries a header saying where it came from."""
    listing = tmp_path / "docs.txt"
    listing.write_text("# 2 documents, seed 1\n\nalpha.pdf\n  beta.pdf  \n\n",
                       encoding="utf-8")
    assert chunk.read_name_list(listing) == {"alpha.pdf", "beta.pdf"}


# The cache key. Every test below calls `chunk_params` directly, because its
# failure mode is silent in both directions: a parameter left out reuses chunks
# that a change should have invalidated, and one wrongly included throws away a
# bake that is still good. Neither shows up as an error, only as a wrong index or
# as hours of GPU time.
def params(chunk, **overrides):
    """`chunk_params` with the shipped 256/26 passage settings as the baseline."""
    kwargs = dict(target_tokens=256, overlap_tokens=26, tokenizer_name="jina",
                  boundaries=True, sections=True, heading_tokens=0,
                  page_chunks=False, section_chunks=False, tables=True)
    kwargs.update(overrides)
    return chunk.chunk_params(**kwargs)


def test_the_policy_is_only_a_cache_key_when_boundaries_are_on(chunk):
    """`--no-section-first` is recorded only when it changes something.

    Every chunk baked before the flag existed still has to match its cache entry,
    so the paragraph-preferred policy is the one that gets written down.
    """
    assert "section_first" not in params(chunk, sections=True)
    assert params(chunk, sections=False)["section_first"] is False
    # With boundaries off there is no rung ladder to have an order, so the two
    # must share a cache entry rather than splitting it.
    assert "section_first" not in params(chunk, boundaries=False, sections=False)


def test_the_budget_is_not_a_cache_key_for_a_packer_that_has_no_budget(chunk):
    """The 2026-09-26 regression: a page bake thrown away by a passage decision.

    `pack_pages` and `pack_sections` take neither the target nor the overlap, so
    moving the passage target from 1024 tokens to 256 must leave both of those
    bakes alone. Before `chunk_params` existed it invalidated all 534 documents
    of `data/chunks-page`, for a number no page chunk consulted.
    """
    for mode in ("page_chunks", "section_chunks"):
        wide = params(chunk, target_tokens=1024, overlap_tokens=256, **{mode: True})
        narrow = params(chunk, target_tokens=256, overlap_tokens=26, **{mode: True})
        assert wide == narrow, f"{mode} hashes the budget it never reads"
        assert "target_tokens" not in wide
        assert "overlap_tokens" not in wide
        # The boundary ladder is `pack`'s too, for the same reason.
        assert "section_first" not in params(chunk, sections=False, **{mode: True})


def test_the_budget_IS_a_cache_key_for_the_packer_that_has_one(chunk):
    """The other half of the rule, which is the expensive one to get wrong."""
    assert params(chunk, target_tokens=256) != params(chunk, target_tokens=1024)
    assert params(chunk, overlap_tokens=26) != params(chunk, overlap_tokens=51)
    assert params(chunk, tokenizer_name="jina") != params(chunk, tokenizer_name="arctic")


def test_a_flag_is_recorded_only_when_it_is_on(chunk):
    """So a bake predating a flag still matches the default the flag replaced."""
    off = params(chunk, boundaries=False, sections=False, tables=False)
    assert sorted(off) == ["overlap_tokens", "target_tokens", "tokenizer"]
    # And each one shows up on its own once it is asked for.
    for flag in ("boundaries", "page_chunks", "section_chunks", "tables"):
        asked = dict(boundaries=False, sections=False, tables=False)
        asked[flag] = True
        assert flag in params(chunk, **asked)


def test_the_heading_cap_is_its_value_not_a_flag(chunk):
    """Two caps change the text differently, so they cannot share a cache entry."""
    assert "headings" not in params(chunk, heading_tokens=0)
    assert params(chunk, heading_tokens=24)["headings"] == 24
    assert params(chunk, heading_tokens=24) != params(chunk, heading_tokens=48)
