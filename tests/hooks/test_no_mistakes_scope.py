from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

HOOK = Path(__file__).parents[2] / ".codex" / "hooks" / "no_mistakes_scope.py"


def load_hook() -> ModuleType:
    spec = importlib.util.spec_from_file_location("no_mistakes_scope", HOOK)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def commit_file(repo: Path, relative_path: str, content: str) -> None:
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    git(repo, "add", relative_path)
    git(repo, "commit", "-m", f"change {relative_path}")


def make_repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-b", "main")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Test User")
    commit_file(tmp_path, "docs/index.html", "baseline")
    git(tmp_path, "switch", "-c", "change")
    return tmp_path


def hook_event(command: str, *, workdir: Path | None = None) -> str:
    tool_input: dict[str, str] = {"command": command}
    if workdir is not None:
        tool_input["workdir"] = str(workdir)
    return json.dumps({"tool_name": "Bash", "tool_input": tool_input})


def run_hook(
    repo: Path,
    command: str,
    *,
    env: dict[str, str] | None = None,
    process_workdir: Path | None = None,
    tool_workdir: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HOOK)],
        cwd=process_workdir or repo,
        input=hook_event(command, workdir=tool_workdir),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_docs_only_branch_blocks_no_mistakes(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    commit_file(repo, "docs/decisions.html", "decision")

    result = run_hook(repo, "git push no-mistakes change")

    assert result.returncode == 0
    assert '"permissionDecision":"deny"' in result.stdout
    assert "no application code paths" in result.stdout


def test_src_change_allows_no_mistakes(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    commit_file(repo, "src/app.py", "print('ready')")

    result = run_hook(repo, "no-mistakes axi run --intent test")

    assert result.returncode == 0
    assert result.stdout == ""


def test_tool_workdir_selects_a_feature_worktree(tmp_path: Path) -> None:
    feature_repo = tmp_path / "feature"
    feature_repo.mkdir()
    make_repo(feature_repo)
    commit_file(feature_repo, "src/app.py", "print('ready')")
    launcher_repo = tmp_path / "launcher"
    launcher_repo.mkdir()
    make_repo(launcher_repo)

    result = run_hook(
        feature_repo,
        "no-mistakes axi run --intent test",
        process_workdir=launcher_repo,
        tool_workdir=feature_repo,
    )

    assert result.returncode == 0
    assert result.stdout == ""


def test_command_project_dir_selects_a_feature_worktree(tmp_path: Path) -> None:
    feature_repo = tmp_path / "feature"
    feature_repo.mkdir()
    make_repo(feature_repo)
    commit_file(feature_repo, "web/app.ts", "export const ready = true")
    launcher_repo = tmp_path / "launcher"
    launcher_repo.mkdir()
    make_repo(launcher_repo)

    result = run_hook(
        feature_repo,
        (f"env CLAUDE_PROJECT_DIR={feature_repo} no-mistakes axi run --intent test"),
        process_workdir=launcher_repo,
    )

    assert result.returncode == 0
    assert result.stdout == ""


def test_claude_receives_blocking_exit_code(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    commit_file(repo, "AGENTS.md", "instructions")

    result = run_hook(
        repo,
        "no-mistakes axi run --intent test",
        env={**os.environ, "CLAUDE_PROJECT_DIR": str(repo)},
    )

    assert result.returncode == 2
    assert "no application code paths" in result.stderr


def test_only_application_folders_are_code_paths() -> None:
    hook = load_hook()

    assert hook.has_code_changes(["tests/test_app.py"])
    assert hook.has_code_changes(["web/app/page.tsx"])
    assert hook.has_code_changes(["docs/index.html", "scripts/fetch_corpus.py"])
    assert not hook.has_code_changes(["tests/hooks/test_scope.py", ".codex/hooks/gate.sh"])
    assert not hook.has_code_changes([".codex/hooks/gate.sh", "pyproject.toml"])
    assert not hook.has_code_changes(["docs/index.html", "AGENTS.md"])


def test_status_commands_remain_available() -> None:
    hook = load_hook()

    assert not hook.starts_no_mistakes("no-mistakes axi status")
    assert not hook.starts_no_mistakes("no-mistakes axi abort")
    assert hook.starts_no_mistakes("no-mistakes axi run --intent test")
    assert hook.starts_no_mistakes("git push no-mistakes")
