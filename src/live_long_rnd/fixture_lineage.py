"""Deterministic production-input lineage for the committed retrieval fixture."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

REINDEX_ENTRY_POINT = Path("scripts/reindex_golden_retrieval_fixture.py")
ENVIRONMENT_INPUTS = (
    Path(".python-version"),
    Path("pyproject.toml"),
    Path("uv.lock"),
)
DEFAULT_MANIFEST = Path("tests/fixtures/golden_retrieval_index/manifest.json")


class StaleGoldenFixtureError(RuntimeError):
    """Raised when committed fixture lineage differs from its production inputs."""


class LineageDiscoveryError(RuntimeError):
    """Raised when the production indexing import graph cannot be traced safely."""


def production_index_input_hashes(repository_root: Path | None = None) -> dict[str, str]:
    """Hash the reindex entry point, local import closure, and locked environment."""
    root = (repository_root or Path(__file__).resolve().parents[2]).resolve()
    discovered = _discover_local_import_closure(root, REINDEX_ENTRY_POINT)
    inputs = {*ENVIRONMENT_INPUTS, *discovered}
    missing = [path.as_posix() for path in sorted(inputs) if not (root / path).is_file()]
    if missing:
        raise LineageDiscoveryError(f"Production indexing inputs are missing: {', '.join(missing)}")
    return {
        path.as_posix(): hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in sorted(inputs)
    }


def validate_fixture_lineage(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    repository_root: Path | None = None,
) -> None:
    """Fail when a committed fixture does not match current production inputs."""
    root = (repository_root or Path(__file__).resolve().parents[2]).resolve()
    resolved_manifest = manifest_path if manifest_path.is_absolute() else root / manifest_path
    manifest: dict[str, Any] = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    recorded = manifest.get("production_input_sha256")
    current = production_index_input_hashes(root)
    if recorded == current:
        return

    recorded_hashes = recorded if isinstance(recorded, dict) else {}
    changed = sorted(
        path
        for path in current.keys() & recorded_hashes.keys()
        if current[path] != recorded_hashes[path]
    )
    added = sorted(current.keys() - recorded_hashes.keys())
    removed = sorted(recorded_hashes.keys() - current.keys())
    differences = [
        *(f"changed {path}" for path in changed),
        *(f"added {path}" for path in added),
        *(f"removed {path}" for path in removed),
    ]
    if not differences:
        differences.append("missing or invalid production_input_sha256")
    raise StaleGoldenFixtureError(
        "Golden retrieval fixture lineage is stale: "
        + ", ".join(differences)
        + ". Rebuild it with ALLOW_CORPUS_REINDEX=1 uv run python "
        "scripts/reindex_golden_retrieval_fixture.py."
    )


def _discover_local_import_closure(root: Path, entry_point: Path) -> set[Path]:
    pending = [entry_point]
    discovered: set[Path] = set()
    while pending:
        source_path = pending.pop()
        if source_path in discovered:
            continue
        absolute_path = root / source_path
        if not absolute_path.is_file():
            raise LineageDiscoveryError(f"Production indexing input is missing: {source_path}")
        discovered.add(source_path)
        source = absolute_path.read_text(encoding="utf-8")
        module_name = _module_name(source_path)
        package_name = (
            module_name if source_path.name == "__init__.py" else module_name.rpartition(".")[0]
        )
        for imported_module in _imported_modules(source, package_name=package_name):
            pending.extend(
                imported_path
                for imported_path in _local_module_paths(root, imported_module)
                if imported_path not in discovered
            )
    return discovered


def _module_name(source_path: Path) -> str:
    if source_path.parts[:1] == ("src",):
        module_parts = list(source_path.with_suffix("").parts[1:])
    else:
        module_parts = list(source_path.with_suffix("").parts)
    if module_parts[-1] == "__init__":
        module_parts.pop()
    return ".".join(module_parts)


def _imported_modules(source: str, *, package_name: str) -> set[str]:
    tree = ast.parse(source)
    imports: set[str] = set()
    import_module_names, builtin_import_names, importlib_modules, builtins_modules = (
        _dynamic_import_bindings(tree)
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base_module = _resolve_import_from(node, package_name=package_name)
            if base_module:
                imports.add(base_module)
                imports.update(
                    f"{base_module}.{alias.name}" for alias in node.names if alias.name != "*"
                )
        elif isinstance(node, ast.Call):
            dynamic_module = _literal_dynamic_import(
                node,
                import_module_names=import_module_names,
                builtin_import_names=builtin_import_names,
                importlib_modules=importlib_modules,
                builtins_modules=builtins_modules,
            )
            if dynamic_module:
                imports.add(dynamic_module)
    return imports


def _resolve_import_from(node: ast.ImportFrom, *, package_name: str) -> str:
    if node.level == 0:
        return node.module or ""
    package_parts = package_name.split(".")
    keep = len(package_parts) - node.level + 1
    if keep < 0:
        raise LineageDiscoveryError(f"Relative import escapes local package in {package_name}")
    base_parts = package_parts[:keep]
    if node.module:
        base_parts.extend(node.module.split("."))
    return ".".join(base_parts)


def _dynamic_import_bindings(tree: ast.AST) -> tuple[set[str], set[str], set[str], set[str]]:
    import_module_names = {"import_module"}
    builtin_import_names = {"__import__"}
    importlib_modules: set[str] = set()
    builtins_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlib":
                    importlib_modules.add(alias.asname or "importlib")
                elif alias.name == "builtins":
                    builtins_modules.add(alias.asname or "builtins")
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            for alias in node.names:
                if node.module == "importlib" and alias.name == "import_module":
                    import_module_names.add(alias.asname or alias.name)
                elif node.module == "builtins" and alias.name == "__import__":
                    builtin_import_names.add(alias.asname or alias.name)
    return import_module_names, builtin_import_names, importlib_modules, builtins_modules


def _literal_dynamic_import(
    node: ast.Call,
    *,
    import_module_names: set[str],
    builtin_import_names: set[str],
    importlib_modules: set[str],
    builtins_modules: set[str],
) -> str | None:
    if not _is_dynamic_import_call(
        node,
        import_module_names=import_module_names,
        builtin_import_names=builtin_import_names,
        importlib_modules=importlib_modules,
        builtins_modules=builtins_modules,
    ):
        return None
    if not node.args:
        raise LineageDiscoveryError("Production indexing uses a dynamic import without a module name")
    module = node.args[0]
    if isinstance(module, ast.Constant) and isinstance(module.value, str):
        return module.value
    raise LineageDiscoveryError("Production indexing uses a non-literal dynamic import")


def _is_dynamic_import_call(
    node: ast.Call,
    *,
    import_module_names: set[str],
    builtin_import_names: set[str],
    importlib_modules: set[str],
    builtins_modules: set[str],
) -> bool:
    if isinstance(node.func, ast.Name):
        return node.func.id in import_module_names | builtin_import_names
    if not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr == "import_module":
        if isinstance(node.func.value, ast.Name) and node.func.value.id in importlib_modules:
            return True
        raise LineageDiscoveryError("Production indexing uses an unsupported dynamic import")
    if node.func.attr == "__import__":
        if isinstance(node.func.value, ast.Name) and node.func.value.id in builtins_modules:
            return True
        raise LineageDiscoveryError("Production indexing uses an unsupported dynamic import")
    return False


def _local_module_paths(root: Path, module_name: str) -> Iterable[Path]:
    if not module_name.startswith("live_long_rnd"):
        return ()
    module_path = Path("src", *module_name.split("."))
    source_file = module_path.with_suffix(".py")
    package_init = module_path / "__init__.py"
    resolved: list[Path] = []
    if (root / source_file).is_file():
        resolved.append(source_file)
    elif (root / package_init).is_file():
        resolved.append(package_init)
    package_root = Path("src/live_long_rnd/__init__.py")
    if (root / package_root).is_file() and package_root not in resolved:
        resolved.append(package_root)
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry: validate the committed fixture before quality scoring."""
    parser = argparse.ArgumentParser(prog="golden-fixture-lineage")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    validate_fixture_lineage(args.manifest)
    print("Golden retrieval fixture lineage is current.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
