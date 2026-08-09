#!/usr/bin/env bash
# Stop hook (Claude Code) — block finishing a session that changed UI
# structurally/visually without ever rendering the page. The how-to-verify rules
# live in rules/verification.md; this hook only enforces that a look happened.
#
# Fires only when ALL three hold:
#   1. A UI file (.html/.css/.js/.jsx/.tsx/.vue/.svelte) was EDITED this session.
#   2. The edit changed the DOM/visual skeleton, not just text between tags
#      (decided by ui_structural_diff.py — CSS/JS edits always count).
#   3. No shot-scraper / agent-browser / playwright command ran this session.
#
# Block signal is exit 2: Claude feeds stderr back as its next instruction. A marker
# file in the git dir breaks any block->retry loop — at most one consecutive nudge.
# Fails OPEN on any internal error: a buggy gate must never trap the agent.
set -uo pipefail

root="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null)}" || exit 0
[ -n "$root" ] || exit 0
classifier="$root/.claude/hooks/ui_structural_diff.py"
ui_re='\.(html|htm|css|js|jsx|tsx|vue|svelte)$'

input=$(cat)
transcript=$(printf '%s' "$input" | jq -r '.transcript_path // empty' 2>/dev/null)
# No transcript => can't scope to this session's edits; allow (fail open).
[ -n "$transcript" ] && [ -f "$transcript" ] || exit 0

# 1. UI files this session edited via Edit/Write/MultiEdit (absolute paths).
edited=$(jq -r '
  (.message.content // []) []?
  | select(.type? == "tool_use" and (.name? == "Edit" or .name? == "Write" or .name? == "MultiEdit"))
  | .input.file_path? // empty
' "$transcript" 2>/dev/null | grep -Ei "$ui_re" | sort -u)
[ -z "$edited" ] && exit 0

# 2. Keep only edits that changed the visual/DOM skeleton.
needs_look=""
while IFS= read -r path; do
  [ -n "$path" ] || continue
  rel="${path#"$root"/}"
  # docs/ is HTML documentation, not product UI — never gate on it.
  case "$rel" in docs/*) continue ;; esac
  case "$path" in
    *.html|*.htm)
      old=$(git -C "$root" show "HEAD:$rel" 2>/dev/null) || true
      verdict=$(printf '%s' "$old" | python3 "$classifier" "$path" 2>/dev/null) || verdict=structural
      [ "$verdict" = "structural" ] && needs_look+="$rel"$'\n'
      ;;
    *)  # css/js/jsx/tsx/vue/svelte: any edit is visual or behavioral
      needs_look+="$rel"$'\n'
      ;;
  esac
done <<< "$edited"
needs_look=$(printf '%s' "$needs_look" | sed '/^$/d')
[ -z "$needs_look" ] && exit 0

# 3. Did the agent actually RUN a render this session? Match real Bash invocations
#    of the verify tools, not prose mentions of them (a session can discuss
#    shot-scraper without ever rendering). Capture jq output first, then match with
#    a bash `case` — piping jq straight into `grep -q` makes grep close the pipe
#    early, jq dies on SIGPIPE, and pipefail reports the whole pipeline as failed.
ran_cmds=$(jq -r '
  (.message.content // []) []?
  | select(.type? == "tool_use" and .name? == "Bash")
  | .input.command? // empty
' "$transcript" 2>/dev/null) || ran_cmds=""
case "$ran_cmds" in
  *shot-scraper*|*shot_scraper*|*agent-browser*|*playwright*) exit 0 ;;
esac

# Loop backstop: one consecutive nudge, then give way (marker fresh < 30 min).
gitdir=$(git -C "$root" rev-parse --git-dir 2>/dev/null); gitdir="${gitdir:-$root/.git}"
case "$gitdir" in /*) ;; *) gitdir="$root/$gitdir" ;; esac
marker="$gitdir/.ui-verify-nudged"
if [ -f "$marker" ] && [ -z "$(find "$marker" -mmin +30 2>/dev/null)" ]; then
  rm -f "$marker"
  exit 0
fi
rm -f "$marker"; : > "$marker"

first=$(printf '%s\n' "$needs_look" | grep -Ei '\.html?$' | head -1)
[ -n "$first" ] || first=$(printf '%s\n' "$needs_look" | head -1)
target="$root/$first"   # static HTML: pass this path to shot-scraper; served route: use its localhost URL
files_csv=$(printf '%s' "$needs_look" | paste -sd, -)

echo "UI verify gate: you changed $files_csv with structural/visual edits but never rendered the page this session. Read rules/verification.md, run the check it specifies for what changed ($target), review the result, and fix anything broken before finishing. If you believe this is a false positive, run the cheapest check in that rule to clear the gate." >&2
exit 2
