#!/usr/bin/env bash
# The Codex Stop hook must allow a valid external-assets-only docs set.
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
output=$(bash "$root/.codex/hooks/docs-asset-gate.sh")
test "$output" = '{}'
