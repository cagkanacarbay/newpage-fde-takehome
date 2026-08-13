"""Release asset validation tests."""

from pathlib import Path

import pytest

from live_long_rnd.release_assets import ReleaseAssetError, validate_release_assets

CORPUS = Path("data/corpus/longevity")
GOLDEN_INDEX = Path("tests/fixtures/golden_retrieval_index")


def test_complete_fixture_has_release_corpus_and_citation_contract() -> None:
    validate_release_assets(CORPUS, GOLDEN_INDEX)


def test_release_assets_reject_an_incomplete_corpus(tmp_path: Path) -> None:
    with pytest.raises(ReleaseAssetError, match="Expected 27 PDFs"):
        validate_release_assets(tmp_path, GOLDEN_INDEX)
