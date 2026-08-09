#!/usr/bin/env bash
# A docs page is valid only when it uses the shared external assets.
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
python3 "$root/docs/_assets/check-doc-assets.py" --check
