#!/usr/bin/env python3
"""Block no-mistakes when a branch contains no code or tooling changes."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

CODE_PREFIXES = (
    "src/",
    "tests/",
    "web/",
    "scripts/",
)
NON_APPLICATION_PREFIXES = ("tests/hooks/",)
SAFE_TOP_LEVEL_COMMANDS = {"status", "runs", "doctor", "help", "--help", "version"}
SAFE_AXI_COMMANDS = {"status", "abort", "logs", "respond", "sync", "help", "--help"}


def command_segments(command: str) -> list[list[str]]:
    segments: list[list[str]] = []
    for raw_segment in command.replace("&&", ";").replace("||", ";").split(";"):
        try:
            tokens = shlex.split(raw_segment)
        except ValueError:
            continue
        if tokens:
            segments.append(tokens)
    return segments


def starts_no_mistakes(command: str) -> bool:
    for tokens in command_segments(command):
        names = [Path(token).name for token in tokens]
        if "git" in names and "push" in tokens:
            push_index = tokens.index("push")
            if "no-mistakes" in tokens[push_index + 1 :]:
                return True

        try:
            executable_index = names.index("no-mistakes")
        except ValueError:
            continue
        arguments = tokens[executable_index + 1 :]
        if not arguments:
            return True
        if arguments[0] == "axi":
            if len(arguments) == 1:
                continue
            if arguments[1] not in SAFE_AXI_COMMANDS:
                return True
            continue
        if arguments[0] not in SAFE_TOP_LEVEL_COMMANDS:
            return True
    return False


def has_code_changes(paths: list[str]) -> bool:
    return any(
        not any(path.startswith(prefix) for prefix in NON_APPLICATION_PREFIXES)
        and any(path.startswith(prefix) for prefix in CODE_PREFIXES)
        for path in paths
    )


def git_output(root: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def changed_files(root: Path) -> list[str]:
    for base in ("origin/main", "main"):
        if git_output(root, "rev-parse", "--verify", "--quiet", base) is None:
            continue
        merge_base = git_output(root, "merge-base", "HEAD", base)
        if merge_base is None:
            continue
        output = git_output(root, "diff", "--name-only", f"{merge_base}..HEAD")
        return output.splitlines() if output else []

    output = git_output(
        root,
        "diff-tree",
        "--root",
        "--no-commit-id",
        "--name-only",
        "-r",
        "HEAD",
    )
    return output.splitlines() if output else []


def project_root() -> Path | None:
    configured = os.environ.get("CLAUDE_PROJECT_DIR")
    if configured:
        return Path(configured)
    output = git_output(Path.cwd(), "rev-parse", "--show-toplevel")
    return Path(output) if output else None


def deny(message: str) -> int:
    if os.environ.get("CLAUDE_PROJECT_DIR"):
        print(message, file=sys.stderr)
        return 2
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


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    tool_input = event.get("tool_input", {})
    if not isinstance(tool_input, dict):
        return 0
    command = str(tool_input.get("command", tool_input.get("cmd", "")))
    if not starts_no_mistakes(command):
        return 0

    root = project_root()
    if root is None or has_code_changes(changed_files(root)):
        return 0
    return deny(
        "No-mistakes scope: this branch changes no application code paths. "
        "Do not use no-mistakes for documentation, instructions, or tooling work."
    )


if __name__ == "__main__":
    raise SystemExit(main())
