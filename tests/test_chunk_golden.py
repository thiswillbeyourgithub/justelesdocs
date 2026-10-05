"""Golden values that pin chunking against an accidental change.

Every shipped vector is keyed by its chunk file's `src_hash`, and a chunk's text
is what got embedded. A refactor of chunk.py that changed either, even by one
space, would quietly invalidate (or worse, silently mismatch) a bake that takes a
GPU half an hour to redo. These tests do not say the values are RIGHT; they say
they have not MOVED. When a change is meant to move them, bump
CHUNK_FORMAT_VERSION and recompute the constants below in the same commit.

Written by Claude Code (Opus 5.5), before splitting chunk.py, so that the split
could be shown to be hash-identical.
"""

import hashlib
import json

import pytest

from conftest import WordTokenizer

TOKENIZER = "jina-embeddings-v5-text-small-retrieval"
BASE = dict(target_tokens=512, overlap_tokens=0, tokenizer_name=TOKENIZER, boundaries=True,
            sections=True, heading_tokens=0, page_chunks=False, section_chunks=False,
            tables=True, table_layout="9.9.9")

# arm -> (the CLI flags it stands for, overrides of BASE, expected source_hash of GOLDEN_BYTES)
ARMS = {
    "default": ({}, "da2c91b03d9c35c8f1254bad999dd5661ff0126b984fce95e349eba8786ecb15"),
    "t256-o26": ({"target_tokens": 256, "overlap_tokens": 26},
                 "f9ccb4a97882db457cf6a3fded61ba03227dc19584174e950f7971d005a567d8"),
    "greedy": ({"boundaries": False}, "bc19417244bb387d4f67503addc9bae785897993333e562a4310665e86c2e963"),
    "paragraph-first": ({"sections": False}, "50898c66ac9d9515822830afdb7f87ef2bdbc71b2123e641b352263f8a9f679f"),
    "headings": ({"heading_tokens": 60}, "a2fd74fc4ff5443b744cd08375e7451ce40c1e38d4485fdfe188257307da7c9c"),
    "page": ({"page_chunks": True}, "1da3dd520ece903592cce9fa8bbb14d2f685ae8d171de482fbd6e7597189f39c"),
    "section": ({"section_chunks": True}, "c79cce03aab8c154d42ab2a64f6c2d31ec54e3d51ebf67a95fb97789045095cc"),
    "overlap-at-boundaries": ({"overlap_tokens": 51, "overlap_boundaries": True},
                              "47ff67a149a97afd83e7d84190bf8721881227a806f4069d2b5b15ca36d059a2"),
    "no-tables": ({"tables": False, "table_layout": None},
                  "13d0362e0a6c4dc0adfd8a9ed2827d1cec6030c5b751af61f4a9d0e0fdeada12"),
}
GOLDEN_BYTES = b"%PDF-1.7 golden bytes\n"


def test_the_format_version_is_the_one_these_values_were_taken_at(chunk):
    assert chunk.CHUNK_FORMAT_VERSION == 24, \
        "a format bump moves every source_hash: recompute ARMS in the same commit"


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_source_hash_has_not_moved(chunk, tmp_path, arm):
    overrides, expected = ARMS[arm]
    pdf = tmp_path / "golden.pdf"
    pdf.write_bytes(GOLDEN_BYTES)
    params = chunk.chunk_params(**{**BASE, **overrides})
    assert chunk.source_hash(pdf, params) == expected, params


def golden_pdf(path):
    """Three pages: headings at a larger size, body paragraphs, a page break mid-section."""
    import pymupdf

    doc = pymupdf.open()
    words = ("insulation structure pressure retrofits heat balancing monitoring "
             "thermal effects renewal prevention refurbishment combination").split()
    for page_no in range(3):
        page = doc.new_page()
        y = 72
        for section in range(2):
            page.insert_text((72, y), f"Section {page_no}.{section} heading", fontsize=16)
            y += 28
            for para in range(4):
                line = " ".join(words[(page_no + section + para + i) % len(words)] for i in range(12))
                for _ in range(2):
                    page.insert_text((72, y), line, fontsize=10)
                    y += 13
                y += 8
    doc.save(path)


def digest(chunks):
    """What reaches the encoder and the highlight layer: text, pages, boxes."""
    payload = [(c.text, [(r.page, [round(v, 2) for v in r.bbox]) for r in c.rows]) for c in chunks]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


# mode -> (chunk_document keyword arguments, expected digest of its chunks). The default
# and paragraph-first agree on this PDF because no section here outgrows the budget.
OUTPUTS = {
    "default": (dict(target=60, overlap=0, boundaries=True),
               "888b972895f3ac2f1d1e6a1d4438d0691d5ca74453cc321035490b9d55bd4c90"),
    "greedy-overlap": (dict(target=60, overlap=10),
                      "d43d64d8d7c1c08692580c7e083d954eb34a223574cf73b76245570b1ac7a11c"),
    "paragraph-first": (dict(target=60, overlap=0, boundaries=True, sections=False),
                       "888b972895f3ac2f1d1e6a1d4438d0691d5ca74453cc321035490b9d55bd4c90"),
    "headings": (dict(target=60, overlap=0, boundaries=True, headings=20),
                "9d0856c02f4d149341bcef51ebe4dc7f2b4860ea8d7074e236531cd564e24228"),
    "page": (dict(target=60, overlap=0, page_chunks=True),
            "ab96ba65b480e139fb22254a400991b9b808f1791a8cc5931024457c8842b893"),
    "section": (dict(target=60, overlap=0, section_chunks=True),
               "0981cf4522560cf734663509d34054e693d10f663a3dc03ebb962b801419963a"),
}


@pytest.mark.parametrize("mode", sorted(OUTPUTS))
def test_chunk_output_has_not_moved(chunk, tmp_path, mode):
    kwargs, expected = OUTPUTS[mode]
    pdf = tmp_path / "golden.pdf"
    golden_pdf(pdf)
    chunks, _stats = chunk.chunk_document(pdf, WordTokenizer(),
                                          chunk.ChunkSettings(tables=False, **kwargs))
    assert chunks, "the golden PDF produced no chunks"
    assert digest(chunks) == expected, (mode, digest(chunks), len(chunks))
