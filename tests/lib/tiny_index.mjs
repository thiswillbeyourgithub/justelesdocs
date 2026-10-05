/* A three-chunk, format-5 index written to a temporary directory.
 *
 * Shared by the loader and service tests: both need an index on disk and neither
 * may depend on dist/index/, which the suite does not have.
 *
 * Written by Claude Code (Fable 5.1).
 */

import { mkdtemp, mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

/* The sentinel comes from the ranker it is read by, not a second literal: were it
   ever widened, a copy here would keep writing the old value and every chunk of the
   tiny index would suddenly point at page row 65535. */
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const { NO_PAGE_ROW } = await import(`${root}/src/search.js`);

export const DIMS = 8;

/** One int8 unit vector along `axis`. */
export const axis = (d) => Int8Array.from({ length: DIMS }, (_, i) => (i === d ? 127 : 0));

/** Write a format-5 index of three chunks in one document and return its path. */
export async function writeTinyIndex() {
  const dir = await mkdtemp(join(tmpdir(), "tiny-index-"));
  const vectors = new Int8Array(3 * DIMS);
  [axis(0), axis(1), axis(2)].forEach((v, i) => vectors.set(v, i * DIMS));
  // [doc, page, page row, section row] per chunk; no page or section files ship.
  const locations = Uint16Array.from([
    0, 1, NO_PAGE_ROW, NO_PAGE_ROW, 0, 2, NO_PAGE_ROW, NO_PAGE_ROW, 0, 2, NO_PAGE_ROW, NO_PAGE_ROW]);
  const meta = {
    format_version: 5, quant: "int8", dims: DIMS, bytes_per_vector: DIMS,
    n_documents: 1, n_chunks: 3, n_pages: 0, n_sections: 0,
    vectors_file: "vectors-t.i8", chunks_file: "chunks-t.u16",
    pages_file: "", sections_file: "", page_weight: 0, section_weight: 0, prev_page_weight: 0,
    renditions: { redundant: [], companion: [] },
    // The facets the page and the service accept, as build_index.py ships them from
    // the corpus's configuration.
    facet_fields: ["issuer", "country", "year", "language", "doc_type", "topic", "access"],
    range_fields: ["year"], tier_field: "guideline",
    documents: [{ id: 0, file: "a.pdf", chunk_offset: 0, chunk_count: 3, title: "A", issuer: "",
      country: "", year: "2020", language: "fr", doc_type: "", topic: "", family: "", rendition: "" }],
  };
  await writeFile(join(dir, "meta.json"), JSON.stringify(meta));
  await writeFile(join(dir, "vectors-t.i8"), vectors);
  await writeFile(join(dir, "chunks-t.u16"), new Uint8Array(locations.buffer));
  await mkdir(join(dir, "doc"));
  await writeFile(join(dir, "doc/0.json"), JSON.stringify({
    chunks: [1, 2, 3].map((n) => ({ text: `passage ${n}`, pages: [n < 2 ? 1 : 2], boxes: [] })),
  }));
  return dir;
}
