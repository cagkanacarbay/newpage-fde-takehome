#!/usr/bin/env python3
"""Block agent edits outside explicitly agent-owned README sections."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
AGENT_MARKER = "<!-- Maintained by the agent:"
HEADING = re.compile(r"^(#{1,6})\\s+")


def deny(message: str, *, codex: bool) -> int:
    if codex:
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": message,
                    }
                },
                separators=(",", ":"),
            )
        )
        return 0
    print(message, file=sys.stderr)
    return 2


def agent_owned_ranges(lines: list[str]) -> list[tuple[int, int]]:
    """Return editable line ranges from README ownership comments.

    A marker makes the content after it agent-owned until the next heading at
    the same or higher level. The heading and marker remain human-owned.
    """
    ranges: list[tuple[int, int]] = []
    headings = [
        (index, len(match.group(1)))
        for index, line in enumerate(lines)
        if (match := HEADING.match(line))
    ]

    for marker_index, line in enumerate(lines):
        if AGENT_MARKER not in line:
            continue
        owner_heading = next(
            ((index, level) for index, level in reversed(headings) if index < marker_index),
            None,
        )
        if owner_heading is None:
            continue
        _, owner_level = owner_heading
        end = next(
            (index for index, level in headings if index > marker_index and level <= owner_level),
            len(lines),
        )
        ranges.append((marker_index + 1, end))
    return ranges


def is_agent_owned(line_index: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= line_index < end for start, end in ranges)


def is_agent_owned_boundary(line_index: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= line_index <= end for start, end in ranges)


def targets_root_readme(path_text: str) -> bool:
    candidate = Path(path_text.strip())
    if candidate.is_absolute():
        return candidate.resolve() == README
    return Path(os.path.normpath(path_text.strip())) == Path("README.md")


def patch_sections(command: str) -> list[tuple[str, str, str]]:
    """Return (operation, path, body) sections from Codex apply_patch input."""
    matches = list(re.finditer(r"^\\*\\*\\* (Add|Update|Delete) File: (.+)$", command, re.MULTILINE))
    sections: list[tuple[str, str, str]] = []
    for index, match in enumerate(matches):
        body_end = matches[index + 1].start() if index + 1 < len(matches) else command.find("*** End Patch", match.end())
        if body_end == -1:
            body_end = len(command)
        sections.append((match.group(1), match.group(2), command[match.end() : body_end]))
    return sections


def locate_hunk(lines: list[str], hunk: list[str]) -> tuple[int, list[tuple[str, int]]] | None:
    old_lines = [line[1:] for line in hunk if line.startswith((" ", "-"))]
    if not old_lines:
        return None
    starts = [
        index
        for index in range(len(lines) - len(old_lines) + 1)
        if lines[index : index + len(old_lines)] == old_lines
    ]
    if len(starts) != 1:
        return None

    cursor = starts[0]
    changes: list[tuple[str, int]] = []
    for line in hunk:
        if line.startswith(" "):
            if cursor >= len(lines) or lines[cursor] != line[1:]:
                return None
            cursor += 1
        elif line.startswith("-"):
            if cursor >= len(lines) or lines[cursor] != line[1:]:
                return None
            changes.append(("delete", cursor))
            cursor += 1
        elif line.startswith("+"):
            changes.append(("insert", cursor))
    return starts[0], changes


def patch_stays_in_agent_sections(command: str, lines: list[str]) -> bool:
    ranges = agent_owned_ranges(lines)
    if not ranges:
        return False

    sections = patch_sections(command)
    if not sections:
        return True

    for operation, path_text, body in sections:
        if not targets_root_readme(path_text):
            continue
        if operation != "Update":
            return False
        hunks = [chunk.splitlines() for chunk in re.split(r"^@@.*$", body, flags=re.MULTILINE)[1:]]
        if not hunks:
            return False
        for hunk in hunks:
            located = locate_hunk(lines, hunk)
            if located is None:
                return False
            _, changes = located
            if not changes:
                return False
            for kind, line_index in changes:
                if kind == "delete" and not is_agent_owned(line_index, ranges):
                    return False
                if kind == "insert" and not is_agent_owned_boundary(line_index, ranges):
                    return False
    return True


def bash_is_safe_to_read(command: str) -> bool:
    if "README.md" not in command:
        return True
    if any(token in command for token in (">", "tee", "rm ", "mv ", "cp ", "sed -i", "perl -i")):
        return False
    return bool(re.match(r"^\\s*(cat|sed(?!.*-i)|rg|grep|awk|git (add|diff|show|status))\\b", command))


def claude_edit_stays_in_agent_section(tool_input: object, lines: list[str]) -> bool:
    if not isinstance(tool_input, dict):
        return False
    file_path = str(tool_input.get("file_path", ""))
    if not targets_root_readme(file_path):
        return True
    edits = tool_input.get("edits")
    if not isinstance(edits, list):
        edits = [tool_input]
    source = "\\n".join(lines)
    ranges = agent_owned_ranges(lines)
    for edit in edits:
        if not isinstance(edit, dict):
            return False
        old_string = edit.get("old_string", edit.get("oldString"))
        if not isinstance(old_string, str) or not old_string:
            return False
        start = source.find(old_string)
        if start == -1 or source.find(old_string, start + 1) != -1:
            return False
        first_line = source[:start].count("\\n")
        last_line = first_line + old_string.count("\\n")
        if not all(is_agent_owned(index, ranges) for index in range(first_line, last_line + 1)):
            return False
    return True


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    if not README.is_file():
        return 0

    tool_name = event.get("tool_name", "")
    tool_input = event.get("tool_input", {})
    codex = tool_name in {"apply_patch", "Bash"}
    lines = README.read_text(encoding="utf-8").splitlines()

    if tool_name == "apply_patch":
        command = tool_input.get("command", "") if isinstance(tool_input, dict) else ""
        if not patch_stays_in_agent_sections(command, lines):
            return deny("README ownership: only the marked Installation section is agent-owned.", codex=True)
        return 0

    if tool_name == "Bash":
        command = tool_input.get("command", "") if isinstance(tool_input, dict) else ""
        if not bash_is_safe_to_read(command):
            return deny("README ownership: agents cannot write README.md through shell commands.", codex=True)
        return 0

    if not claude_edit_stays_in_agent_section(tool_input, lines):
        return deny("README ownership: only the marked Installation section is agent-owned.", codex=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
