"""The manifest's facet policy: vocabularies, guideline tiers, rendition classes.

The VALUES come from the corpus's `corpus.toml` (lib/corpus_config.py); this module
gives them names, types and the consistency checks every consumer relies on.

One definition, imported by every program that needs it: `manifest.py` writes and
checks the TSV against it, `build_index.py` ships part of it in meta.json for the
client, `review_manifest.py` validates a curator's decisions against it, and the
tests check they agree.

It used to live in `manifest.py` and be read out of that file's SOURCE with `ast`
(lib/policy.py), because importing `manifest.py` runs its `import pymupdf`, a
20 MB wheel, in a script that wants two frozensets. A module of its own with no
dependency outside the standard library removes the reason for the trick, and with
it the trick's failure mode (a constant that stopped being a plain literal).

Module imported, not run.

Written by Claude Code (Opus 5, moved here by Opus 5.5).
"""

from __future__ import annotations

from typing import Any

from lib import corpus_config, vocabulary

# Multi-valued cells hold several slugs separated by this. Semicolon rather than
# comma because a comma reads as part of a value to anyone editing the TSV in a
# spreadsheet, and because no slug in the vocabularies below contains one.
#
# `build_index.py` reads this constant out of this file's SOURCE and ships it in
# meta.json, so `src/search.js` splits on whatever is written here. Restating ";"
# in JavaScript would drift in the worst direction: the client would stop splitting
# and every multi-valued cell would become one unselectable facet option.
VALUE_SEPARATOR = ";"

# The vocabularies, the tiers, the facets and the rendition classes describe ONE
# corpus, so they are read from its `corpus.toml` (lib/corpus_config.py) rather
# than written here; `corpus.example/corpus.toml` documents every key. They are
# exposed as module constants because every consumer imports them by name, and
# checked at import, so a configuration that disagrees with itself stops the first
# script that reads it rather than shipping.
_facets = corpus_config.section("facets")
_tiers = corpus_config.section("tiers")
_renditions = corpus_config.section("renditions")

# The corpus's closed vocabularies, by manifest column ("doc_type", "topic" for
# the first corpus). `check_vocabulary` rejects any slug not listed, so a typo
# in the TSV fails the build instead of quietly creating a facet option that
# matches exactly one document.
CLOSED_VOCABULARIES: dict[str, frozenset[str]] = {
    column: frozenset(table["values"])
    for column, table in corpus_config.section("vocabularies").items()}

# The rights tier. A software notion rather than a corpus one (stage.py and the
# page service implement it), so it stays here. Two values, checked like any other
# vocabulary: a typo in a rights cell ("restrcited") would otherwise read as "open"
# to every consumer, which is the one direction in which a silent fallback is not
# acceptable here.
ACCESS: frozenset[str] = frozenset({"open", "restricted"})

# The GRADED column: one ordered vocabulary where choosing a value means "this or
# narrower" (the first corpus calls it `guideline`: how much of a guideline a
# document is). NARROWEST FIRST, and the order is the whole point, which is why it
# is a tuple: the UI offers a level and keeps every tier down to it.
# `build_index.py` ships this list in meta.json in this order and `src/search.js`
# compares positions in it, so reordering the configured list changes what the
# site shows.
#
# A BLANK cell is the absence of a tier and is never hidden by the level control,
# on the same reasoning as the year range (a document nobody has got to yet must not
# quietly vanish).
TIER_COLUMN: str = _tiers["column"]
TIERS: tuple[str, ...] = tuple(_tiers["order"])

# Where a blank tier is derived from: a closed vocabulary column, and which of its
# slugs puts a document in which tier. Every slug of that vocabulary appears
# exactly once, which `_check_tiers` enforces below: a slug added to the vocabulary
# and forgotten here would autofill nothing and silently leave its documents
# uncurated. A cell holding several slugs takes the NARROWEST tier any of them
# earns (`tier_from_cell` below).
TIER_SOURCE: str = _tiers["derived_from"]
TIER_BY_VALUE: dict[str, str] = dict(_tiers["by_value"])

VOCABULARIES: dict[str, frozenset[str]] = {
    **CLOSED_VOCABULARIES, "access": ACCESS, TIER_COLUMN: frozenset(TIERS)}

# The manifest columns offered as filters, in panel order, and the ones shown as a
# numeric range rather than a dropdown. Shipped in meta.json, so the page and the
# search service build their filters from the index rather than from a list of
# their own that would drift from this one.
FACET_FIELDS: tuple[str, ...] = tuple(_facets["fields"])
RANGE_FIELDS: tuple[str, ...] = tuple(_facets.get("range", ()))

# The corpus's own manifest columns, in TSV order: its closed vocabularies and its
# tier column, placed by manifest.py after the standard ones. Checked by
# `_check_tiers` to name nothing else, so a column cannot be declared without a
# vocabulary to validate it against.
CORPUS_COLUMNS: tuple[str, ...] = tuple(corpus_config.section("manifest")["corpus_columns"])


def _check_tiers() -> None:
    """Refuse to run if the tier mapping and its source vocabulary disagree.

    Raises
    ------
    RuntimeError
        If the source column is not a closed vocabulary, if one of its slugs has
        no tier, if TIER_BY_VALUE names a slug outside it, or assigns a tier
        outside TIERS.

    Notes
    -----
    Run at import, like `_check_issuer_patterns` and `_check_curated_issuers`. The
    cost of getting this wrong is quiet: the autofill simply skips a document whose
    source cell has no tier, so a forgotten slug shows up as a handful of blank
    cells among hundreds of other blanks, and nobody notices for weeks.
    """
    if TIER_SOURCE not in CLOSED_VOCABULARIES:
        raise RuntimeError(f"[tiers] derived_from = {TIER_SOURCE!r} is not a [vocabularies] table")
    source = CLOSED_VOCABULARIES[TIER_SOURCE]
    missing = source - TIER_BY_VALUE.keys()
    if missing:
        raise RuntimeError(
            f"[tiers.by_value] has no tier for {sorted(missing)}. Every {TIER_SOURCE} "
            "slug needs one, or documents carrying it autofill to nothing."
        )
    extra = TIER_BY_VALUE.keys() - source
    if extra:
        raise RuntimeError(f"[tiers.by_value] names slugs outside {TIER_SOURCE}: {sorted(extra)}")
    bad = {tier for tier in TIER_BY_VALUE.values() if tier not in TIERS}
    if bad:
        raise RuntimeError(f"[tiers.by_value] uses tiers outside [tiers] order: {sorted(bad)}")
    declared = set(CLOSED_VOCABULARIES) | {TIER_COLUMN}
    if set(CORPUS_COLUMNS) != declared or len(CORPUS_COLUMNS) != len(declared):
        raise RuntimeError(
            f"[manifest] corpus_columns {list(CORPUS_COLUMNS)} must list each [vocabularies] "
            f"table and the [tiers] column exactly once: {sorted(declared)}")
    unknown = set(FACET_FIELDS) - set(RANGE_FIELDS) - {"issuer", "country", "language", "access"} \
        - CLOSED_VOCABULARIES.keys()
    if unknown:
        raise RuntimeError(f"[facets] fields names columns the manifest does not have: {sorted(unknown)}")


_check_tiers()


def tier_from_cell(cell: str) -> str:
    """Derive a tier from a cell of the TIER_SOURCE column.

    Parameters
    ----------
    cell
        A raw cell, possibly blank, possibly holding several slugs separated by
        `VALUE_SEPARATOR`.

    Returns
    -------
    str
        The narrowest tier any of the cell's slugs earns, or `""` for a blank cell
        or one whose slugs are all unknown. A blank is returned rather than guessed
        because the level control never hides a blank, so "I do not know" is the
        safe answer and the widest tier is not.

    Examples
    --------
    With the first corpus's tiers:

    >>> tier_from_cell("recommandation")             # doctest: +SKIP
    'strict'
    >>> tier_from_cell("recommandation;outil")       # doctest: +SKIP
    'strict'
    >>> tier_from_cell("")
    ''
    """
    tiers = [TIER_BY_VALUE[slug]
             for slug in vocabulary.split_values(cell, VALUE_SEPARATOR)
             if slug in TIER_BY_VALUE]
    if not tiers:
        return ""
    return min(tiers, key=TIERS.index)


# --------------------------------------------------------------------------
# Curated document families
# --------------------------------------------------------------------------
# One piece of guidance can appear in the corpus more than once, in renditions of
# different lengths. Left alone these retrieve against each other and fill a
# result page with the same guidance three times. The decision (see DESIGN.md) is
# to keep every rendition and collapse them at ranking time, so these classes are
# what the search reads to know which results are siblings. The distinction that
# matters is NOT "same family" but "same content":
#
#   REDUNDANT_RENDITIONS restate one piece of guidance at different lengths: two
#   hits from one family that are both redundant are the same answer twice, so the
#   result list shows only the best-ranking one.
#
#   COMPANION_RENDITIONS share a family but carry DIFFERENT content. Collapsing
#   them would silently hide the only document that answers the query.
#
#   EDITION_RENDITIONS are successive editions of one book. Collapse keeps the
#   best-SCORING sibling where the right winner for editions is the newest, so
#   build_index.py ships only the two classes above and src/search.js treats a
#   rendition it does not know as never-collapse: editions rank on their own.
REDUNDANT_RENDITIONS: frozenset[str] = frozenset(_renditions["redundant"])
COMPANION_RENDITIONS: frozenset[str] = frozenset(_renditions["companion"])
EDITION_RENDITIONS: frozenset[str] = frozenset(_renditions.get("edition", ()))


def policy() -> dict[str, Any]:
    """The part of the policy that leaves Python, as JSON-ready lists.

    Returns
    -------
    dict[str, Any]
        `{"renditions": {"redundant": [...], "companion": [...]},
          "separator": ";", "guideline_tiers": [...], "tier_field": "...",
          "facet_fields": [...], "range_fields": [...],
          "vocabularies": {<column>: [...], ...}}`, every vocabulary sorted except
        the tier column's, which is narrowest first and whose order IS the policy.

    Notes
    -----
    The client has to know the rendition policy: it decides whether two documents
    from one family are the same answer twice (collapse to the best-ranking one and
    offer the rest as alternate renditions) or two different answers that must both
    stand. It also has to know `VALUE_SEPARATOR`, because closed-vocabulary cells
    hold lists and the facet panel splits them, and which columns are filters at
    all. All of it is shipped in meta.json rather than restated in JavaScript,
    because a second copy drifts silently in the worst direction: a rendition
    wrongly treated as redundant hides the only document that answers the query.

    The vocabularies are not shipped, only checked: the client derives its facet
    options from the documents themselves. The tiers are shipped, because they are
    the only one whose ORDER carries meaning: the level control keeps every document
    at the chosen tier or narrower, a question about positions in this list. Sorting
    it would quietly invert the control, hence `list`, not `sorted`. The key keeps
    its historical name, `guideline_tiers`, so an index stays readable by a page
    from before the tier column was configurable.
    """
    return {
        "renditions": {"redundant": sorted(REDUNDANT_RENDITIONS),
                       "companion": sorted(COMPANION_RENDITIONS)},
        "separator": VALUE_SEPARATOR,
        "guideline_tiers": list(TIERS),
        "tier_field": TIER_COLUMN,
        "facet_fields": list(FACET_FIELDS),
        "range_fields": list(RANGE_FIELDS),
        "vocabularies": {**{column: sorted(values) for column, values in CLOSED_VOCABULARIES.items()},
                         TIER_COLUMN: list(TIERS)},
    }
