#!/usr/bin/env bash
# Refuse a push of anything but the local branch `main` to the remote branch `main`.
#
# Called by .githooks/pre-push with the hook's own stdin: one line per ref being
# pushed, "<local ref> <local sha> <remote ref> <remote sha>". Exits 1 and says
# why when the checked-out branch is not `main`, or when any pushed ref is not
# refs/heads/main -> refs/heads/main; 0 otherwise.
#
# The repository's history before the public cut (other local branches) holds the
# corpus's records and names, and was never meant to leave this machine. Pushing
# only `main` to `main` is what keeps it here: a `git push github old-branch`, a
# `--all`, a tag or a push to a second remote branch is refused rather than
# trusted to the corpus check, which inspects paths and not names or prose.
# Deleting a remote branch is refused too, for the same "only main" reason.
#
# Its own file, like refuse_corpus.sh, so tests/test_prepush.py runs the real
# thing against a throwaway repository.
#
# Written by Claude Code.
set -u

status=0
current="$(git symbolic-ref --quiet --short HEAD || echo "(detached HEAD)")"
if [ "$current" != "main" ]; then
  echo "[pre-push] REFUSED: the checked-out branch is $current, not main. Only main is pushed." >&2
  status=1
fi
while read -r local_ref _ remote_ref _; do
  [ -z "$local_ref" ] && continue
  if [ "$local_ref" != "refs/heads/main" ] || [ "$remote_ref" != "refs/heads/main" ]; then
    echo "[pre-push] REFUSED: pushing $local_ref to $remote_ref. Only main is pushed, and only to main." >&2
    status=1
  fi
done
exit "$status"
