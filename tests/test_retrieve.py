"""Hybrid retrieval tests: ranked results, source diversity, and citations."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from live_long_rnd.ingest import (
    LanceDBNodeStore,
    SentenceTransformerEmbedder,
    ingest_pdf,
)
from live_long_rnd.retrieve import (
    LanceDBHybridStore,
    RetrievalResult,
    retrieve,
    to_citation_payload,
)


class StubEmbedder:
    """Embedding boundary double with a deterministic query vector."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


class StubHybridStore:
    """Store boundary double that returns fused rows in rank order."""

    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self._rows = list(rows)

    def hybrid_search(
        self, query: str, query_vector: Sequence[float], *, limit: int
    ) -> list[Mapping[str, Any]]:
        del query, query_vector
        return self._rows[:limit]


def _row(document_id: str, score: float) -> dict[str, Any]:
    return {
        "metadata": {
            "document_id": document_id,
            "page_numbers": json.dumps([2]),
            "bboxes": json.dumps([{"page": 2, "l": 10.0, "t": 20.0, "r": 30.0, "b": 40.0}]),
            "heading_path": json.dumps(["Biomarkers", "Results"]),
            "original_text": "Epigenetic age was measured in worker bees.",
        },
        "_relevance_score": score,
    }


def test_retrieve_returns_ranked_chunks_with_complete_provenance() -> None:
    results = retrieve(
        "epigenetic age",
        k=2,
        store=StubHybridStore([_row("paper-a", 0.04), _row("paper-b", 0.03)]),
        embedder=StubEmbedder(),
    )

    assert results == [
        RetrievalResult(
            document_id="paper-a",
            page_numbers=[2],
            bboxes=[{"page": 2, "l": 10.0, "t": 20.0, "r": 30.0, "b": 40.0}],
            heading_path=["Biomarkers", "Results"],
            original_text="Epigenetic age was measured in worker bees.",
            score=0.04,
        ),
        RetrievalResult(
            document_id="paper-b",
            page_numbers=[2],
            bboxes=[{"page": 2, "l": 10.0, "t": 20.0, "r": 30.0, "b": 40.0}],
            heading_path=["Biomarkers", "Results"],
            original_text="Epigenetic age was measured in worker bees.",
            score=0.03,
        ),
    ]


def test_retrieve_skips_overflow_chunks_and_keeps_later_documents() -> None:
    rows = [
        _row("paper-a", 0.05),
        _row("paper-a", 0.04),
        _row("paper-a", 0.03),
        _row("paper-a", 0.02),
        _row("paper-b", 0.01),
    ]

    results = retrieve(
        "epigenetic age",
        k=4,
        store=StubHybridStore(rows),
        embedder=StubEmbedder(),
    )

    assert [result.document_id for result in results] == [
        "paper-a",
        "paper-a",
        "paper-a",
        "paper-b",
    ]
    assert [result.score for result in results] == [0.05, 0.04, 0.03, 0.01]


def test_retrieve_keeps_every_chunk_when_documents_are_below_the_cap() -> None:
    rows = [
        _row("paper-a", 0.03),
        _row("paper-a", 0.02),
        _row("paper-b", 0.01),
    ]

    results = retrieve(
        "epigenetic age",
        k=10,
        store=StubHybridStore(rows),
        embedder=StubEmbedder(),
    )

    assert [result.document_id for result in results] == [
        "paper-a",
        "paper-a",
        "paper-b",
    ]


def test_retrieve_can_disable_the_per_document_cap() -> None:
    rows = [
        _row("paper-a", 0.04),
        _row("paper-a", 0.03),
        _row("paper-a", 0.02),
        _row("paper-a", 0.01),
    ]

    results = retrieve(
        "epigenetic age",
        k=4,
        per_document_cap=None,
        store=StubHybridStore(rows),
        embedder=StubEmbedder(),
    )

    assert [result.document_id for result in results] == ["paper-a"] * 4


def test_citation_payload_uses_the_first_provenance_entry() -> None:
    result = RetrievalResult(
        document_id="paper-a",
        page_numbers=[2, 3],
        bboxes=[
            {"page": 3, "l": 1.0, "t": 2.0, "r": 3.0, "b": 4.0},
            {"page": 2, "l": 5.0, "t": 6.0, "r": 7.0, "b": 8.0},
        ],
        heading_path=["Biomarkers", "Results"],
        original_text="Epigenetic age was measured in worker bees.",
        score=0.04,
    )

    assert to_citation_payload(result) == {
        "document_id": "paper-a",
        "page": 3,
        "heading_path": ["Biomarkers", "Results"],
        "bbox": {"l": 1.0, "t": 2.0, "r": 3.0, "b": 4.0},
        "snippet": "Epigenetic age was measured in worker bees.",
    }


@pytest.mark.integration
@pytest.mark.e2e
def test_ingested_corpus_paper_is_retrievable_with_hybrid_search(tmp_path: Path) -> None:
    source = next(Path("data/corpus/longevity").glob("011-*.pdf"))
    index_dir = tmp_path / "index"
    embedder = SentenceTransformerEmbedder()

    ingest_pdf(
        source,
        embedder=embedder,
        store=LanceDBNodeStore(index_dir),
    )
    results = retrieve(
        "epigenetic clock in insects",
        k=10,
        per_document_cap=1,
        store=LanceDBHybridStore(index_dir),
        embedder=embedder,
    )

    assert results
    assert sum(result.document_id == source.stem for result in results) <= 1
    for result in results:
        assert result.document_id == source.stem
        assert result.page_numbers
        assert result.bboxes
        assert result.heading_path
        assert result.original_text
        assert result.score > 0
        citation = to_citation_payload(result)
        assert citation["page"] >= 1
        assert citation["bbox"].keys() == {"l", "t", "r", "b"}
