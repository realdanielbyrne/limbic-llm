#!/usr/bin/env bash
# Bring the branch up to date with the remote and push, tolerating dirty files that the local
# commits do not touch (logs still being written by running jobs). With no local commits it
# fast-forwards; with local commits it replays them on the remote branch (a rebase done with
# reset --keep + cherry-pick, because `git pull --rebase` refuses to run while a running job is
# rewriting a tracked log). Never force-pushes; on a conflict it restores the original HEAD and
# exits 1, and a rejected push exits 1 too. Resolve those by hand.
#
#   experiments/evolve/sync.sh            (pushes the current branch)
set -u -o pipefail
cd "$(dirname "$0")/../.."
B=$(git rev-parse --abbrev-ref HEAD)
git fetch -q origin "$B" || exit 1
ORIG=$(git rev-parse HEAD)
BASE=$(git merge-base HEAD "origin/$B")
PICKS=$(git rev-list --reverse "$BASE..HEAD")
if [ -n "$PICKS" ] && [ "$(git rev-parse "origin/$B")" != "$BASE" ]; then
  git reset -q --keep "origin/$B" || { echo "reset --keep refused"; exit 1; }
  for c in $PICKS; do
    if ! git cherry-pick --allow-empty "$c" > /dev/null; then
      git cherry-pick --abort; git reset -q --keep "$ORIG"
      echo "CONFLICT replaying $c; restored $ORIG"; exit 1
    fi
  done
elif [ -z "$PICKS" ]; then
  git merge --ff-only -q "origin/$B" || { echo "fast-forward refused"; exit 1; }
fi
git push origin "$B" 2>&1 | tail -1 || { echo "push failed"; exit 1; }
git log --oneline -3
