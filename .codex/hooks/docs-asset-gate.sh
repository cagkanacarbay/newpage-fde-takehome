#!/usr/bin/env bash
# Block Codex from finishing when docs contain inline assets or stale references.
set -uo pipefail

root="$(git rev-parse --show-toplevel 2>/dev/null)" || { echo '{}'; exit 0; }
[ -d "$root/docs" ] || { echo '{}'; exit 0; }

if ! result=$(python3 "$root/docs/_assets/check-doc-assets.py" --check 2>&1); then
  jq -n --arg r "docs asset gate: external shared assets are required:\n$result" \
    '{decision:"block", reason:$r}'
  exit 0
fi

echo '{}'
