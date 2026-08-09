#!/usr/bin/env bash
# Codex Stop hook — mirror of .claude/hooks/ui-verify-gate.sh.
#
# Same intent: block finishing when this session changed UI structurally/visually but
# never rendered the page. Two harness differences from the Claude twin:
#   - Block signal is JSON on stdout — {"decision":"block","reason":...} — not exit 2.
#   - Scope comes from the git working tree (HEAD diff + untracked), because the Codex
#     rollout is harder to parse for "files edited this session" than Claude's transcript.
#     Slightly broader scope; the structural classifier + render check still gate it.
# Shares the classifier at .claude/hooks/ui_structural_diff.py. Fails OPEN on any error.
set -uo pipefail

root="$(git rev-parse --show-toplevel 2>/dev/null)" || { echo '{}'; exit 0; }
classifier="$root/.claude/hooks/ui_structural_diff.py"
ui_re='\.(html|htm|css|js|jsx|tsx|vue|svelte)$'
input=$(cat)

pass() { echo '{}'; exit 0; }
block() { jq -n --arg r "$1" '{decision:"block", reason:$r}'; exit 0; }

# 1. Changed UI files: working tree vs HEAD, plus untracked.
changed=$( { git -C "$root" diff --name-only HEAD; \
             git -C "$root" ls-files --others --exclude-standard; } 2>/dev/null \
           | grep -Ei "$ui_re" | sort -u)
[ -z "$changed" ] && pass

# 2. Keep only edits that changed the visual/DOM skeleton.
needs_look=""
while IFS= read -r rel; do
  [ -n "$rel" ] || continue
  # docs/ is HTML documentation, not product UI — never gate on it.
  case "$rel" in docs/*) continue ;; esac
  case "$rel" in
    *.html|*.htm)
      old=$(git -C "$root" show "HEAD:$rel" 2>/dev/null) || true
      verdict=$(printf '%s' "$old" | python3 "$classifier" "$root/$rel" 2>/dev/null) || verdict=structural
      [ "$verdict" = "structural" ] && needs_look+="$rel"$'\n'
      ;;
    *) needs_look+="$rel"$'\n' ;;
  esac
done <<< "$changed"
needs_look=$(printf '%s' "$needs_look" | sed '/^$/d')
[ -z "$needs_look" ] && pass

# 3. Did the agent render anything? Only trust an explicit rollout path for THIS
#    session from stdin — a time-based "newest rollout" guess can grab a different
#    concurrent session and match mere mentions of the tools. When Codex gives no
#    rollout path, skip render detection and let the marker backstop cap it at one
#    nudge per structural-UI session.
transcript=$(printf '%s' "$input" | jq -r '.transcript_path // .rollout_path // .session.rollout_path // empty' 2>/dev/null)
if [ -n "$transcript" ] && [ -f "$transcript" ] \
   && grep -Eq 'shot-scraper|shot_scraper|agent-browser|playwright' "$transcript"; then
  pass
fi

# Loop backstop: one consecutive nudge, then give way (marker fresh < 30 min).
gitdir=$(git -C "$root" rev-parse --git-dir 2>/dev/null); gitdir="${gitdir:-$root/.git}"
case "$gitdir" in /*) ;; *) gitdir="$root/$gitdir" ;; esac
marker="$gitdir/.ui-verify-nudged"
if [ -f "$marker" ] && [ -z "$(find "$marker" -mmin +30 2>/dev/null)" ]; then
  rm -f "$marker"; pass
fi
rm -f "$marker"; : > "$marker"

first=$(printf '%s\n' "$needs_look" | grep -Ei '\.html?$' | head -1)
[ -n "$first" ] || first=$(printf '%s\n' "$needs_look" | head -1)
target="$root/$first"   # static HTML: pass this path to shot-scraper; served route: use its localhost URL
files_csv=$(printf '%s' "$needs_look" | paste -sd, -)

block "UI verify gate: you changed $files_csv with structural/visual edits but never rendered the page this session. Read rules/verification.md, run the check it specifies for what changed ($target), review the result, and fix anything broken before finishing. If you believe this is a false positive, run the cheapest check in that rule to clear the gate."
