"""Behavioral checks for the explicit golden-fixture rebuild command."""

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts/reindex_golden_retrieval_fixture.py"
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "reindex_golden_retrieval_fixture", _SCRIPT_PATH
)
assert _SCRIPT_SPEC is not None
assert _SCRIPT_SPEC.loader is not None
reindex = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(reindex)


class _Store:
    def __init__(self, index_dir: Path, table_name: str) -> None:
        self.index_dir = index_dir
        self.table_name = table_name

    def finalize(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)


# Brief: the rebuild command must reject an attempt without explicit corpus approval.
def test_rebuild_requires_explicit_corpus_approval(tmp_path: Path, monkeypatch: Any) -> None:
    output = tmp_path / "golden_retrieval_index"
    output.mkdir()
    marker = output / "keep-existing-fixture"
    marker.write_text("preserved", encoding="utf-8")
    monkeypatch.setattr(reindex, "DEFAULT_OUTPUT", output)
    monkeypatch.delenv("ALLOW_CORPUS_REINDEX", raising=False)

    with pytest.raises(SystemExit):
        reindex.main(["--output", str(output)])

    assert marker.read_text(encoding="utf-8") == "preserved"


# Brief: an approved rebuild must write hashes for the complete discovered input closure.
def test_explicit_rebuild_records_current_production_lineage(
    tmp_path: Path, monkeypatch: Any
) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    for number in range(1, 28):
        (corpus_dir / f"{number:03d}-paper.pdf").write_bytes(b"%PDF fixture")
    output = tmp_path / "golden_retrieval_index"
    monkeypatch.setattr(reindex, "DEFAULT_OUTPUT", output)
    monkeypatch.setattr(reindex, "LanceDBNodeStore", _Store)
    monkeypatch.setattr(reindex, "create_reader", object)
    monkeypatch.setattr(reindex, "ingest_pdf", lambda *_args, **_kwargs: 1)
    monkeypatch.setenv("ALLOW_CORPUS_REINDEX", "1")

    expected_lineage = reindex.production_index_input_hashes()

    assert reindex.main(["--corpus-dir", str(corpus_dir), "--output", str(output)]) == 0

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["production_input_sha256"] == expected_lineage
