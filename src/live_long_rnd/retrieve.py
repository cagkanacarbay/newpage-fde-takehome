"""Hybrid dense and BM25 retrieval with citation-ready provenance."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypedDict, cast

import lancedb
from lancedb.rerankers import RRFReranker

from live_long_rnd.ingest import (
    DEFAULT_INDEX_DIR,
    DEFAULT_TABLE_NAME,
    Embedder,
    SentenceTransformerEmbedder,
)

FUSION_CANDIDATE_MULTIPLIER = 4


class HybridStore(Protocol):
    """Store boundary that returns RRF-fused rows in relevance order."""

    def hybrid_search(
        self, query: str, query_vector: Sequence[float], *, limit: int
    ) -> Sequence[Mapping[str, Any]]:
        """Run dense and BM25 search, then return fused rows."""


class LanceDBHybridStore:
    """LanceDB adapter using native dense, BM25, and RRF hybrid search."""

    def __init__(self, index_dir: Path, table_name: str = DEFAULT_TABLE_NAME) -> None:
        self._table = lancedb.connect(str(index_dir)).open_table(table_name)

    def hybrid_search(
        self, query: str, query_vector: Sequence[float], *, limit: int
    ) -> list[Mapping[str, Any]]:
        rows = (
            self._table.search(
                query_type="hybrid",
                vector_column_name="vector",
                fts_columns="text",
            )
            .vector(list(query_vector))
            .text(query)
            .rerank(RRFReranker())
            .limit(limit)
            .to_list()
        )
        return cast(list[Mapping[str, Any]], rows)


@dataclass
class RetrievalResult:
    """One winning chunk with its score and exact source provenance."""

    document_id: str
    page_numbers: list[int]
    bboxes: list[dict[str, int | float]]
    heading_path: list[str]
    original_text: str
    score: float


class CitationBBox(TypedDict):
    l: int | float  # noqa: E741 - the chat API requires the PDF geometry key
    t: int | float
    r: int | float
    b: int | float


class CitationPayload(TypedDict):
    """Citation contract consumed by the chat API."""

    document_id: str
    page: int
    heading_path: list[str]
    bbox: CitationBBox
    snippet: str


def _result_from_row(row: Mapping[str, Any]) -> RetrievalResult:
    metadata = row["metadata"]
    return RetrievalResult(
        document_id=str(metadata["document_id"]),
        page_numbers=json.loads(metadata["page_numbers"]),
        bboxes=json.loads(metadata["bboxes"]),
        heading_path=json.loads(metadata["heading_path"]),
        original_text=str(metadata["original_text"]),
        score=float(row["_relevance_score"]),
    )


def to_citation_payload(result: RetrievalResult) -> CitationPayload:
    """Map one retrieval result to the chat API citation contract."""
    first_bbox = result.bboxes[0]
    return {
        "document_id": result.document_id,
        "page": int(first_bbox["page"]),
        "heading_path": list(result.heading_path),
        "bbox": {
            "l": first_bbox["l"],
            "t": first_bbox["t"],
            "r": first_bbox["r"],
            "b": first_bbox["b"],
        },
        "snippet": result.original_text,
    }


def retrieve(
    query: str,
    *,
    k: int = 10,
    per_document_cap: int | None = 3,
    store: HybridStore | None = None,
    embedder: Embedder | None = None,
) -> list[RetrievalResult]:
    """Return top RRF-fused chunks, capped per source. Set the cap to None to disable it."""
    active_store = store if store is not None else LanceDBHybridStore(DEFAULT_INDEX_DIR)
    active_embedder = embedder if embedder is not None else SentenceTransformerEmbedder()
    [query_vector] = active_embedder.embed([query])
    candidate_limit = k if per_document_cap is None else k * FUSION_CANDIDATE_MULTIPLIER
    rows = active_store.hybrid_search(query, query_vector, limit=candidate_limit)

    results: list[RetrievalResult] = []
    document_counts: dict[str, int] = {}
    for row in rows:
        result = _result_from_row(row)
        count = document_counts.get(result.document_id, 0)
        if per_document_cap is not None and count >= per_document_cap:
            continue
        results.append(result)
        document_counts[result.document_id] = count + 1
        if len(results) == k:
            break
    return results
