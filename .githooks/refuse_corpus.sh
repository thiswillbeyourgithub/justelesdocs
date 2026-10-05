#!/usr/bin/env bash
# Refuse a push that would carry a corpus document or one of its derivatives.
#
# Called by .githooks/pre-push with the hook's own stdin: one line per ref being
# pushed, "<local ref> <local sha> <remote ref> <remote sha>". Exits 1 and names
# every offending path when any is found, 0 otherwise.
#
# It lives in its own file rather than inside pre-push so that a test can run the
# real thing against a throwaway repository (tests/test_prepush.py): the check
# that guards the project's legal boundary is the one that most needs to be
# exercised end to end, not only its pattern.
#
# Two sets of paths are inspected, because either alone has a hole:
#   - the TREE at the pushed tip, which catches a file that is present now,
#     whatever history brought it in;
#   - every path ADDED OR MODIFIED by any commit in the pushed range, which
#     catches a PDF that one commit adds and a later one deletes. The tip tree no
#     longer holds it, but the blob is in history and leaves this machine with
#     the push. A pushed blob cannot be unpushed.
# `-m` diffs a merge against each of its parents, so a file introduced by the
# merge commit itself (an "evil merge") is listed too.
#
# Written by Claude Code.
set -u

# `[/-]` after the folder names is load-bearing. Without it the pattern also
# matched top-level tables such as `data/SORTING_LOG.tsv`, which are tracked
# records of the sorting meant to be pushed, and refused every push from
# 2026-09-17 onwards. The separator keeps the folders caught (`data/UNSURE/x.pdf`,
# `data/chunks-page/` via the dash) while letting a top-level table through.
#
# `data/private/` is the working-notes folder: gitignored, and refused as well so
# that a `git add -f` on one of its files cannot turn into a push.
#
# `local/` is where a corpus lives inside a checkout of the software (gitignored,
# its own repository). The `data/` alternatives above name no corpus by content:
# they are the folder layout scripts/lib/corpus.py resolves under any CORPUS_DIR,
# kept so that a corpus checked out at the root by mistake is refused too.
pattern='\.pdf$|^data/(GUIDELINES|UNSURE|DISCARDED|OCR|chunks)[/-]|^data/private/|^dist/|^models/|^local/'

zero=0000000000000000000000000000000000000000
status=0
while read -r _local_ref local_sha _remote_ref remote_sha; do
  # A deletion push has a zero local sha and no tree to inspect.
  case "$local_sha" in *[^0]*) ;; *) continue ;; esac

  # The commits this push sends. A new branch (zero remote sha), or a remote sha
  # this clone does not have (someone else pushed, then this is a force push),
  # has no usable base, so fall back to everything no remote-tracking ref holds:
  # a superset of what the push sends, which is the safe direction to be wrong in.
  if [ "$remote_sha" != "$zero" ] && git cat-file -e "$remote_sha^{commit}" 2>/dev/null; then
    range=("$remote_sha..$local_sha")
  else
    range=("$local_sha" --not --remotes)
  fi

  offenders="$( { git ls-tree -r --name-only "$local_sha"
                  git log -m --no-renames --format= --name-only --diff-filter=AM "${range[@]}"
                } | LC_ALL=C sort -u | grep -Ei "$pattern" || true)"
  if [ -n "$offenders" ]; then
    echo "[pre-push] REFUSED: pushing $local_sha would send files that must never be pushed:" >&2
    echo "$offenders" | sed 's/^/    /' >&2
    echo "[pre-push] These are corpus documents or derived artefacts. Note vendor/ IS" >&2
    echo "[pre-push] tracked on purpose (see scripts/vendor.py) and is not checked here." >&2
    echo "[pre-push] Removing them from the working tree is not enough: a path listed here" >&2
    echo "[pre-push] is in the tip tree OR in a commit of the pushed range, so rewrite the" >&2
    echo "[pre-push] history that added it." >&2
    status=1
  fi
done
exit "$status"
