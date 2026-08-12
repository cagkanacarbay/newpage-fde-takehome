"""Behavioral checks for the committed retrieval fixture's production lineage."""

import json
from pathlib import Path

import pytest

from live_long_rnd.fixture_lineage import (
    StaleGoldenFixtureError,
    production_index_input_hashes,
    validate_fixture_lineage,
)


def _write_minimal_indexing_repo(root: Path) -> None:
    files = {
        ".python-version": "3.12\n",
        "pyproject.toml": "[project]\nname = 'fixture-test'\n",
        "uv.lock": "version = 1\n",
        "scripts/reindex_golden_retrieval_fixture.py": (
            "from live_long_rnd.ingest import ingest_pdf\n"
        ),
        "src/live_long_rnd/__init__.py": '"""Fixture test package."""\n',
        "src/live_long_rnd/ingest.py": "from live_long_rnd.parsing import parse_pdf\n",
        "src/live_long_rnd/parsing.py": "def parse_pdf():\n    return None\n",
    }
    for relative_path, content in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _write_manifest(root: Path) -> Path:
    manifest_path = root / "tests/fixtures/golden_retrieval_index/manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {"production_input_sha256": production_index_input_hashes(root)},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest_path


# Brief: the quality gate must reject a fixture after production parsing behavior changes.
def test_fixture_lineage_rejects_changed_production_parser(tmp_path: Path) -> None:
    _write_minimal_indexing_repo(tmp_path)
    manifest_path = _write_manifest(tmp_path)
    parser_path = tmp_path / "src/live_long_rnd/parsing.py"
    parser_path.write_text("def parse_pdf():\n    return []\n", encoding="utf-8")

    with pytest.raises(StaleGoldenFixtureError, match=r"src/live_long_rnd/parsing\.py"):
        validate_fixture_lineage(manifest_path, repository_root=tmp_path)


# Brief: a newly imported indexing module must join lineage without a hand-edited file list.
def test_fixture_lineage_discovers_new_local_indexing_import(tmp_path: Path) -> None:
    _write_minimal_indexing_repo(tmp_path)
    manifest_path = _write_manifest(tmp_path)
    provenance_path = tmp_path / "src/live_long_rnd/provenance.py"
    provenance_path.write_text("FIELD = 'page_numbers'\n", encoding="utf-8")
    ingest_path = tmp_path / "src/live_long_rnd/ingest.py"
    ingest_path.write_text(
        ingest_path.read_text(encoding="utf-8") + "from live_long_rnd.provenance import FIELD\n",
        encoding="utf-8",
    )

    with pytest.raises(StaleGoldenFixtureError, match=r"src/live_long_rnd/provenance\.py"):
        validate_fixture_lineage(manifest_path, repository_root=tmp_path)


# Brief: package-relative imports must not bypass the recursively discovered closure.
def test_fixture_lineage_discovers_new_package_relative_import(tmp_path: Path) -> None:
    _write_minimal_indexing_repo(tmp_path)
    manifest_path = _write_manifest(tmp_path)
    provenance_path = tmp_path / "src/live_long_rnd/provenance.py"
    provenance_path.write_text("FIELD = 'bboxes'\n", encoding="utf-8")
    package_path = tmp_path / "src/live_long_rnd/__init__.py"
    package_path.write_text(
        package_path.read_text(encoding="utf-8") + "from .provenance import FIELD\n",
        encoding="utf-8",
    )

    with pytest.raises(StaleGoldenFixtureError, match=r"src/live_long_rnd/provenance\.py"):
        validate_fixture_lineage(manifest_path, repository_root=tmp_path)


# Brief: qualified and aliased dynamic imports must join the indexed source closure.
def test_fixture_lineage_discovers_qualified_and_aliased_dynamic_imports(tmp_path: Path) -> None:
    _write_minimal_indexing_repo(tmp_path)
    (tmp_path / "src/live_long_rnd/provenance.py").write_text(
        "FIELD = 'page_numbers'\n", encoding="utf-8"
    )
    (tmp_path / "src/live_long_rnd/metadata.py").write_text("FIELD = 'bboxes'\n", encoding="utf-8")
    ingest_path = tmp_path / "src/live_long_rnd/ingest.py"
    ingest_path.write_text(
        "import importlib as loader\n"
        "from importlib import import_module as load_module\n"
        "loader.import_module('live_long_rnd.provenance')\n"
        "load_module('live_long_rnd.metadata')\n",
        encoding="utf-8",
    )

    hashes = production_index_input_hashes(tmp_path)

    assert "src/live_long_rnd/provenance.py" in hashes
    assert "src/live_long_rnd/metadata.py" in hashes


# Brief: importing a nested module must include each package initializer it executes.
def test_fixture_lineage_includes_intermediate_package_initializers(tmp_path: Path) -> None:
    _write_minimal_indexing_repo(tmp_path)
    indexing_dir = tmp_path / "src/live_long_rnd/indexing"
    indexing_dir.mkdir()
    (indexing_dir / "__init__.py").write_text("FORMAT = 'citation-v1'\n", encoding="utf-8")
    (indexing_dir / "writer.py").write_text("def write():\n    return None\n", encoding="utf-8")
    ingest_path = tmp_path / "src/live_long_rnd/ingest.py"
    ingest_path.write_text(
        "from live_long_rnd.indexing.writer import write\n",
        encoding="utf-8",
    )

    hashes = production_index_input_hashes(tmp_path)

    assert "src/live_long_rnd/indexing/__init__.py" in hashes
    assert "src/live_long_rnd/indexing/writer.py" in hashes


# Brief: a literal relative dynamic import must resolve through its package.
def test_fixture_lineage_discovers_relative_dynamic_import(tmp_path: Path) -> None:
    _write_minimal_indexing_repo(tmp_path)
    (tmp_path / "src/live_long_rnd/provenance.py").write_text(
        "FIELD = 'page_numbers'\n", encoding="utf-8"
    )
    ingest_path = tmp_path / "src/live_long_rnd/ingest.py"
    ingest_path.write_text(
        "from importlib import import_module\n"
        "import_module('.provenance', package='live_long_rnd')\n",
        encoding="utf-8",
    )

    hashes = production_index_input_hashes(tmp_path)

    assert "src/live_long_rnd/provenance.py" in hashes
