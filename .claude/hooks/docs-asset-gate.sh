#!/usr/bin/env bash
# Block Claude from finishing when docs contain inline assets or stale references.
set -uo pipefail

root="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null)}" || exit 0
[ -d "$root/docs" ] || exit 0

if ! result=$(python3 "$root/docs/_assets/check-doc-assets.py" --check 2>&1); then
  {
    echo "docs asset gate: external shared assets are required:"
    echo "$result"
  } >&2
  exit 2
fi
