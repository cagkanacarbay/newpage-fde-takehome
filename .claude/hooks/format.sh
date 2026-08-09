#!/usr/bin/env bash
# PostToolUse hook (Claude Code) — auto-format the Python file just edited.
#
# Reads the tool-call JSON on stdin, pulls the edited file path, and if it is a
# .py file, runs `ruff format` + `ruff check --fix` on just that file. Scoped to
# the single edited file so it stays fast. Non-blocking by design: formatting is
# mechanical, so it never blocks the agent — it just keeps the file clean as
# edits happen. Wired in .claude/settings.json under PostToolUse (Write|Edit|MultiEdit).
set -uo pipefail

root="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null)}" || exit 0
[ -n "$root" ] || exit 0

input=$(cat)
file=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')

# Only act on Python files.
case "$file" in
  *.py) ;;
  *) exit 0 ;;
esac
[ -f "$file" ] || exit 0

cd "$root" || exit 0
uv run ruff format "$file" >/dev/null 2>&1 || true
uv run ruff check --fix "$file" >/dev/null 2>&1 || true
exit 0
