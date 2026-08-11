"""The LanceDB hybrid retriever behind the chat API's Retriever seam."""

import asyncio
import json
from collections.abc import Mapping, Sequence
from typing import Any

from live_long_rnd.api.retrieval import LanceDbRetriever, create_retriever
from live_long_rnd.query_planning import QueryPlan, SearchIntent
from live_long_rnd.retrieve import RetrievalDependencies, RetrievalResult


class FakeStore:
    def dense_search(
        self, query_vector: Sequence[float], *, filters: object, limit: int
    ) -> Sequence[Mapping[str, Any]]:
        del query_vector, filters, limit
        return [
            {
                "metadata": {
                    "document_id": "015-hickson-2019",
                    "page_numbers": json.dumps([2]),
                    "bboxes": json.dumps([{"page": 2, "l": 1.0, "t": 2.0, "r": 3.0, "b": 4.0}]),
                    "heading_path": json.dumps(["1. Introduction"]),
                    "original_text": "Senolytics target senescent cells.",
                },
                "_relevance_score": 0.9,
            }
        ]

    def sparse_search(
        self, query: str, *, filters: object, limit: int
    ) -> Sequence[Mapping[str, Any]]:
        del query, filters, limit
        return []


class FakeEmbedder:
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]


class FakePlanner:
    def plan(self, message: str, history: object = ()) -> QueryPlan:
        del message, history
        return QueryPlan(
            action="retrieve",
            search_intents=[
                SearchIntent(
                    dense_query="Do senolytics reverse cellular senescence?",
                    sparse_query="senolytics reverse cellular senescence",
                )
            ],
        )


class FakeReranker:
    def rerank(self, query: str, candidates: Sequence[RetrievalResult]) -> list[RetrievalResult]:
        del query
        return list(candidates)


def test_lancedb_retriever_returns_chunks_with_contract_citations() -> None:
    retriever = LanceDbRetriever(
        dependencies=RetrievalDependencies(
            store=FakeStore(),
            embedder=FakeEmbedder(),
            planner=FakePlanner(),
            reranker=FakeReranker(),
        )
    )
    chunks = asyncio.run(retriever.retrieve("do senolytics reverse senescence?"))
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.text == "Senolytics target senescent cells."
    assert chunk.citation == {
        "document_id": "015-hickson-2019",
        "page": 2,
        "heading_path": ["1. Introduction"],
        "bbox": {"l": 1.0, "t": 2.0, "r": 3.0, "b": 4.0},
        "snippet": "Senolytics target senescent cells.",
    }


def test_create_retriever_defaults_to_stub() -> None:
    assert type(create_retriever({})).__name__ == "StubRetriever"


def test_create_retriever_selects_lancedb() -> None:
    assert type(create_retriever({"LIVE_LONG_RETRIEVER": "lancedb"})).__name__ == (
        "LanceDbRetriever"
    )
