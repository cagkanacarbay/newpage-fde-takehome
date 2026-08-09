#!/usr/bin/env bash
# Codex PostToolUse hook — auto-format Python files touched by apply_patch.
#
# Mirrors .claude/hooks/format.sh. Codex fires PostToolUse for apply_patch (its
# file-edit tool) and delivers the patch payload as JSON on stdin. We pull every
# *.py path out of the raw payload and run ruff format + ruff check --fix on just
# those files. Non-blocking by design: formatting is mechanical.
set -uo pipefail

root="$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0
input=$(cat)

# Paths may sit in the patch text or the changes map; grep the raw payload for
# Python files regardless of the exact apply_patch shape.
files=$(printf '%s' "$input" | grep -oE '[A-Za-z0-9_./-]+\.py' | sort -u)
[ -z "$files" ] && exit 0

cd "$root" || exit 0
while IFS= read -r f; do
  # Resolve to an absolute path: payloads carry repo-relative or absolute paths.
  case "$f" in
    /*) full="$f" ;;
    *)  full="$root/$f" ;;
  esac
  [ -f "$full" ] || continue
  uv run ruff format "$full" >/dev/null 2>&1 || true
  uv run ruff check --fix "$full" >/dev/null 2>&1 || true
done <<< "$files"
exit 0
