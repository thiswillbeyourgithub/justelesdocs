"""Split a multi-valued facet cell, and refuse the ones naming an unknown slug.

`doc_type` and `topic` are closed vocabularies, defined in `lib/manifest_policy.py`,
and two scripts gate them: `manifest.py` when it writes the TSV, `build_index.py`
when it reads any TSV it is pointed at. They share this check rather than each
carrying a copy.

What the gate is for: a typo like "recommandaton" is otherwise an error nowhere. The
manifest accepts it, the index ships it, the facet grows an option matching exactly
one document, and the UI's bilingual label lookup falls back to the raw slug, so the
mistake reaches the reader looking like a deliberate filter nobody can explain.

This module is imported, not run, and needs nothing outside the standard library.
It raises `Unknown` rather than a `click.ClickException` so that it stays usable
from anywhere; both callers translate it, which is also where each keeps its own
reason for gating.

Written by Claude Code (Opus 5).
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping


class Unknown(ValueError):
    """A facet cell named a slug outside its vocabulary.

    A subclass of ValueError so a caller that only catches ValueError still stops.
    """


def split_values(cell: str, separator: str) -> list[str]:
    """Split a multi-valued manifest cell into its slugs.

    Parameters
    ----------
    cell
        A raw TSV cell, possibly blank, possibly holding several slugs.
    separator
        `lib/manifest_policy.py`'s `VALUE_SEPARATOR`, which owns it.

    Returns
    -------
    list[str]
        The non-empty slugs, stripped. A blank cell gives an empty list.

    Notes
    -----
    Mirrored by `cellValues` in `src/search.js`, which splits on the separator
    `meta.json` ships. Tolerating stray whitespace matters because these cells are
    edited by hand: "tsa; adulte" and "tsa;adulte" must mean the same thing, or the
    facet grows a phantom " adulte" option.
    """
    return [piece.strip() for piece in cell.split(separator) if piece.strip()]


def check(rows: Iterable[Mapping[str, str]],
          vocabularies: Mapping[str, Collection[str]],
          separator: str) -> None:
    """Refuse a set of rows if any facet cell names a slug outside its vocabulary.

    Parameters
    ----------
    rows
        The manifest rows, each a column-to-value mapping. A dict keyed by
        filename is passed as `.values()`.
    vocabularies
        Column name to the slugs allowed in it, e.g. `{"doc_type": {...}}`. A
        column absent from a row, or blank in it, is "not curated yet" and passes:
        the UI can ignore a blank, and cannot ignore a typo.
    separator
        `lib/manifest_policy.py`'s `VALUE_SEPARATOR`.

    Raises
    ------
    Unknown
        Naming every offending slug, grouped by column, so one run reports the
        whole problem rather than the first row of it.

    Notes
    -----
    It refuses rather than warns, in keeping with the other gates in this repo: a
    warning in a 500-row run scrolls past.
    """
    unknown: dict[str, set[str]] = {}
    for row in rows:
        for column, vocabulary in vocabularies.items():
            for slug in split_values(row.get(column, ""), separator):
                if slug not in vocabulary:
                    unknown.setdefault(column, set()).add(slug)
    if unknown:
        detail = "; ".join(
            f"{column}: {sorted(slugs)}" for column, slugs in sorted(unknown.items())
        )
        raise Unknown(f"manifest uses slugs outside the vocabulary ({detail})")
