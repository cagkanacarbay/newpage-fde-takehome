#!/usr/bin/env bash
# Stop hook (Claude Code) — block finishing until the project is lint-clean and
# its fast tests pass.
#
# exit 2 is the blocking signal for a Stop hook (per Claude Code hooks docs): the
# agent is not allowed to finish, and stderr is fed back so it knows what to fix.
# We gate on two deterministic, fast checks only:
#   1. ruff check .        (lint must be clean)
#   2. pytest -m "not e2e" (fast tests must pass)
# The e2e suite (@pytest.mark.e2e) is deliberately excluded — it drives the real app,
# is slow, and runs separately via `uv run pytest -m e2e`. Run everything by hand with
# `uv run pytest`.
#
# A brand-new repo with no tests yet returns pytest exit code 5 (nothing
# collected); that is treated as a pass so the gate never traps an empty project.
set -uo pipefail

root="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null)}" || exit 0
cd "$root" || exit 0

# Pre-Python repo: no pyproject.toml yet, so there is nothing for ruff or pytest
# to run against. Pass silently until the Python project is scaffolded.
[ -f pyproject.toml ] || exit 0

if ! lint=$(uv run ruff check . 2>&1); then
  {
    echo "gate: ruff check failed — fix before finishing:"
    echo "$lint"
  } >&2
  exit 2
fi

tests=$(uv run pytest -m "not e2e" 2>&1); rc=$?
# rc 5 = no tests collected (fresh project) — allow. Any other non-zero = fail.
if [ "$rc" -ne 0 ] && [ "$rc" -ne 5 ]; then
  {
    echo "gate: fast tests failed (pytest -m 'not e2e'):"
    printf '%s\n' "$tests" | tail -25
  } >&2
  exit 2
fi
exit 0
