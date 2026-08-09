#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
research="$root/docs/research/parsing.html"
decision="$root/docs/decisions.html"
combined="$(mktemp)"
trap 'rm -f "$combined"' EXIT

cat "$research" "$decision" > "$combined"

for finding in \
  'Six candidates' \
  'Hierarchy is recoverable from geometry' \
  "Docling's own fix, tested" \
  'MinerU vs Docling, measured' \
  'What MinerU costs' \
  'Candidates re-examined' \
  "Why the others don't replace Docling" \
  'GriTS score of 0.12' \
  'HeadingHierarchyOptions(enabled=True)' \
  'DoclingReader' \
  'PDF parsing: Docling' \
  'How we use Docling' \
  'Use MinerU only as the comparison baseline' \
  'papers 021-027'; do
  if ! rg -Fq "$finding" "$combined"; then
    echo "missing parsing finding: $finding" >&2
    exit 1
  fi
done

if ! rg -q '\.numbered li b, \.numbered li span\.d' "$root/docs/_assets/docs.css"; then
  echo 'numbered-list labels and descriptions do not share a column' >&2
  exit 1
fi

for rule in 'max-width: 100%;' 'min-width: 0;' 'table.wide { min-width: 44rem; }'; do
  if ! rg -Fq "$rule" "$root/docs/_assets/docs.css"; then
    echo "missing resilient-table rule: $rule" >&2
    exit 1
  fi
done
