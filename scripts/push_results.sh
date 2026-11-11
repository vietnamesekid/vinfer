#!/usr/bin/env bash
# Get results off a rented machine before shutting it down.
#
# 1. Copy everything (summary JSON, raw per-request records, server logs) to the persistent
#    volume first: this step needs nothing but `cp`, so results survive even if step 2 fails.
# 2. Commit the summary JSON and experiment reports on a branch and push it.
#
# Usage: scripts/push_results.sh [branch]   (default: results/<date>-<pod id or host>)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
VOLUME="${VINFER_VOLUME:-/workspace}"
branch="${1:-results/$(date +%Y-%m-%d)-${RUNPOD_POD_ID:-$(hostname)}}"

backup="$VOLUME/results-backup/$(date +%Y%m%dT%H%M%S)"
mkdir -p "$backup"
cp -R bench/results "$backup/" && cp -R experiments "$backup/" || {
  echo "ERROR: could not copy results to $backup" >&2; exit 1; }
echo "1/2 results copied to $backup"

if ! git config user.email >/dev/null; then
  echo "2/2 skipped: no git identity. Results are safe in $backup." >&2
  echo "    Set one (git config user.name/user.email) and re-run to push." >&2
  exit 1
fi
git checkout -B "$branch" || exit 1
# Official results and experiment reports only; raw/ and unofficial results stay on the volume.
shopt -s nullglob
official=(bench/results/*.json)
(( ${#official[@]} )) && { git add -- "${official[@]}" || exit 1; }
git add -- experiments/ || exit 1
if git diff --cached --quiet; then
  echo "2/2 nothing to push: no official result or report changed (see $backup)."
  echo "    Unofficial results (bench/results/raw/unofficial/) are only in the backup." >&2
  exit 0
fi
git commit -m "Results from ${RUNPOD_POD_ID:-$(hostname)} on $(date -u +%FT%TZ)" || exit 1
if git push -u origin "$branch"; then
  echo "2/2 pushed $branch (${#official[@]} official result files)"
else
  echo "2/2 push failed (credentials or remote?). Results are safe in $backup." >&2
  exit 1
fi
