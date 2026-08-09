#!/usr/bin/env bash
# Codex Stop hook — block turn end until the project is lint-clean and fast tests pass.
#
# Mirrors .claude/hooks/gate.sh. Claude signals a block with exit 2; Codex instead
# reads JSON on stdout — {"decision":"block","reason":...} re-prompts the turn with
# the reason so the agent fixes it before finishing. Same two fast checks:
#   1. ruff check .        (lint must be clean)
#   2. pytest -m "not e2e" (fast tests pass; rc 5 = no tests yet = allow)
# The e2e suite (@pytest.mark.e2e) is excluded — it drives the real app and runs
# separately via `uv run pytest -m e2e`.
set -uo pipefail

root="$(git rev-parse --show-toplevel 2>/dev/null)" || { echo '{}'; exit 0; }
cd "$root" || { echo '{}'; exit 0; }

# Pre-Python repo: no pyproject.toml yet, so there is nothing for ruff or pytest
# to run against. Pass silently until the Python project is scaffolded.
[ -f pyproject.toml ] || { echo '{}'; exit 0; }

block() { jq -n --arg r "$1" '{decision:"block", reason:$r}'; exit 0; }

if ! lint=$(uv run ruff check . 2>&1); then
  block "gate: ruff check failed — fix before finishing:
$lint"
fi

tests=$(uv run pytest -m "not e2e" 2>&1); rc=$?
if [ "$rc" -ne 0 ] && [ "$rc" -ne 5 ]; then
  block "gate: fast tests failed (pytest -m 'not e2e'):
$(printf '%s\n' "$tests" | tail -25)"
fi

echo '{}'
exit 0
