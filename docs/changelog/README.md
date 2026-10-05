# Release notes

One directory per released version: `docs/changelog/<major.minor.patch>/changelog.md`, following the convention used in `../justelesRCP`.

Write for a casual reader of the site, not for a developer: what changed for them, one short bullet each, no internal jargon. Not every version needs an entry, but every version worth telling users about does.

Format:

```markdown
# 0.1.0 - 2026-09-14

## New features
- Search the whole corpus in natural language. [0ccd941]
  fr: Recherche en langage naturel dans tout le corpus.
```

Rules, kept identical to the sibling project so the two can share tooling later:

- The title must be `# <version> - <YYYY-MM-DD>`, and the version must match the directory name.
- Only four category headings, displayed in this order: `## New features`, `## Improvements`, `## Bug fixes`, `## Documentation`.
- Each bullet is the English line; the indented `fr:` line under it is the French text. Both are required, because the site is bilingual and neither language is the translation of record.
- A bullet may end with the commit sha(s) it comes from, in brackets: `[0ccd941]` or `[0ccd941, c4d82f1]`.

## What reads them

These are the SOFTWARE's notes. A corpus that wants its readers to see its own history (documents added, filters curated) keeps a `changelog/` directory and a `VERSION` file of the same format at its root, and `scripts/stage.py` then uses both instead of these two. They are taken together or not at all, since the version is checked against the notes.

`scripts/changelog.py` compiles every file here into `dist/changelog.json`, and `scripts/stage.py` calls it on every stage, so the shipped notes cannot be older than the shipped site. The version they are checked against is the one line in the `VERSION` file at the repo root, which `stage.py` also writes into `dist/app-version.js` for the browser.

It is a gate as much as a compiler: staging fails when the current version has no notes, when a file does not parse, or when a bullet is missing its `fr:` line. The failure that guards against is a quiet one, a deploy that shows the previous release's notes under the new version's name.

`src/changelog.js` renders them in the "Quoi de neuf ?" popup. A returning reader whose last visit predates this version gets it opened for them, showing only the releases they have not seen; a first-time visitor gets nothing, because there is no "since" to show. The footer button opens the whole history at any time. `scripts/check_chrome.mjs` covers those rules in a browser.

To compile or check the notes on their own:

```sh
uv run scripts/changelog.py --check
```

Written with Claude Code.
