#!/usr/bin/env bash
# Tests the public behavior of the README ownership guard.
set -euo pipefail

guard="$(git rev-parse --show-toplevel)/.codex/hooks/protect-readme.py"

expect_block() {
  local payload="$1"
  local output
  output=$(printf '%s' "$payload" | python3 "$guard")
  grep -Fq '"permissionDecision":"deny"' <<<"$output"
}

expect_allow() {
  local payload="$1"
  local output
  output=$(printf '%s' "$payload" | python3 "$guard")
  ! grep -Fq '"permissionDecision":"deny"' <<<"$output"
}

human_edit=$'{"hook_event_name":"PreToolUse","tool_name":"apply_patch","tool_input":{"command":"*** Begin Patch\\n*** Update File: README.md\\n@@\\n-# <Project title>\\n+# Changed by an agent\\n*** End Patch"}}'
agent_edit=$'{"hook_event_name":"PreToolUse","tool_name":"apply_patch","tool_input":{"command":"*** Begin Patch\\n*** Update File: README.md\\n@@\\n There is no runnable code yet. The project is at the setup stage: the repository carries\\n+The setup command will be added with the application.\\n its development rules, agent configuration, and hooks, but no application, dependencies,\\n*** End Patch"}}'
shell_write=$'{"hook_event_name":"PreToolUse","tool_name":"Bash","tool_input":{"command":"printf changed > README.md"}}'
shell_read=$'{"hook_event_name":"PreToolUse","tool_name":"Bash","tool_input":{"command":"sed -n \'1,10p\' README.md"}}'
claude_agent_edit=$'{"hook_event_name":"PreToolUse","tool_name":"Edit","tool_input":{"file_path":"README.md","old_string":"There is no runnable code yet. The project is at the setup stage: the repository carries","new_string":"The application can now run."}}'
claude_human_edit=$'{"hook_event_name":"PreToolUse","tool_name":"Edit","tool_input":{"file_path":"README.md","old_string":"# <Project title>","new_string":"# Changed by an agent"}}'

expect_block "$human_edit"
expect_allow "$agent_edit"
expect_block "$shell_write"
expect_allow "$shell_read"
expect_allow "$claude_agent_edit"
expect_block "$claude_human_edit"
