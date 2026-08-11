"""Hybrid retrieval tests: ranked results, source diversity, and citations."""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import lancedb
import pytest
from openai import OpenAI

from live_long_rnd.embeddings import OpenAIEmbedder
from live_long_rnd.ingest import IngestDependencies, LanceDBNodeStore, ingest_pdf
from live_long_rnd.query_planning import MetadataFilters, QueryPlan, SearchIntent
from live_long_rnd.retrieve import (
    DEFAULT_RETRIEVAL_CONFIG,
    FlashRankCrossEncoder,
    IdentityReranker,
    LanceDBHybridStore,
    RetrievalConfig,
    RetrievalDependencies,
    RetrievalResult,
    retrieve,
    retrieve_baseline,
    to_citation_payload,
)


def test_runtime_defaults_match_the_calibrated_configuration() -> None:
    assert (
        RetrievalConfig(
            candidate_depth=20,
            source_budget_tokens=12_000,
            document_diversity_penalty=0.15,
        )
        == DEFAULT_RETRIEVAL_CONFIG
    )


class StubEmbedder:
    """Embedding boundary double with a deterministic query vector."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


class RecordingEmbedder:
    """Embedding boundary double that records the semantic inputs."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [[0.1, 0.2] for _ in texts]


class NoOpReranker:
    def rerank(self, query: str, candidates: Sequence[RetrievalResult]) -> list[RetrievalResult]:
        del query
        return list(candidates)


class StubHybridStore:
    """Store boundary double that returns fused rows in rank order."""

    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self._rows = list(rows)

    def hybrid_search(
        self, query: str, query_vector: Sequence[float], *, limit: int
    ) -> list[Mapping[str, Any]]:
        del query, query_vector
        return self._rows[:limit]


def _row(
    document_id: str,
    score: float,
    original_text: str = "Epigenetic age was measured in worker bees.",
) -> dict[str, Any]:
    return {
        "metadata": {
            "document_id": document_id,
            "page_numbers": json.dumps([2]),
            "bboxes": json.dumps([{"page": 2, "l": 10.0, "t": 20.0, "r": 30.0, "b": 40.0}]),
            "heading_path": json.dumps(["Biomarkers", "Results"]),
            "original_text": original_text,
        },
        "_relevance_score": score,
    }


def _result(document_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        document_id=document_id,
        page_numbers=[2],
        bboxes=[{"page": 2, "l": 10.0, "t": 20.0, "r": 30.0, "b": 40.0}],
        heading_path=["Biomarkers", "Results"],
        original_text="Epigenetic age was measured in worker bees.",
        score=score,
    )


def test_identity_reranker_preserves_fused_candidate_order() -> None:
    candidates = [_result("paper-a", 0.8), _result("paper-b", 0.4)]

    results = IdentityReranker().rerank("query", candidates)

    assert results == candidates


def _openai_embedder() -> OpenAIEmbedder:
    def handle_request(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        inputs = payload["input"]
        value = 1.0 / math.sqrt(3_072)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "index": index,
                        "embedding": [value] * 3_072,
                    }
                    for index, _text in enumerate(inputs)
                ],
                "model": "text-embedding-3-large",
                "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)},
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handle_request))
    client = OpenAI(api_key="test-key", base_url="https://openai.test/v1", http_client=http_client)
    return OpenAIEmbedder(client=client)


def test_planned_retrieval_uses_distinct_dense_and_sparse_queries() -> None:
    calls: list[tuple[str, object]] = []

    class Planner:
        def plan(self, message: str, history: object = ()) -> QueryPlan:
            del message, history
            return QueryPlan(
                action="retrieve",
                search_intents=[
                    SearchIntent(
                        dense_query="What dose did the IPF senolytic pilot use?",
                        sparse_query="IPF D 100 mg/day Q 1250 mg/day",
                    )
                ],
            )

    class Store:
        def dense_search(
            self, query_vector: Sequence[float], *, filters: object, limit: int
        ) -> Sequence[Mapping[str, Any]]:
            calls.append(("dense", list(query_vector)))
            del filters, limit
            return [_row("paper-dense", 0.8)]

        def sparse_search(
            self, query: str, *, filters: object, limit: int
        ) -> Sequence[Mapping[str, Any]]:
            calls.append(("sparse", query))
            del filters, limit
            return [_row("paper-sparse", 0.7)]

    embedder = RecordingEmbedder()

    results = retrieve(
        "What dose did it use?",
        dependencies=RetrievalDependencies(
            store=Store(),
            embedder=embedder,
            planner=Planner(),
            reranker=NoOpReranker(),
        ),
    )

    assert embedder.texts == ["What dose did the IPF senolytic pilot use?"]
    assert calls == [
        ("dense", [0.1, 0.2]),
        ("sparse", "IPF D 100 mg/day Q 1250 mg/day"),
    ]
    assert {result.document_id for result in results} == {"paper-dense", "paper-sparse"}


def test_multi_aspect_retrieval_deduplicates_shared_evidence() -> None:
    class Planner:
        def plan(self, message: str, history: object = ()) -> QueryPlan:
            del message, history
            return QueryPlan(
                action="retrieve",
                search_intents=[
                    SearchIntent(dense_query="benefits", sparse_query="lifespan"),
                    SearchIntent(dense_query="risks", sparse_query="toxicity"),
                ],
            )

    class Store:
        def dense_search(
            self, query_vector: Sequence[float], *, filters: object, limit: int
        ) -> Sequence[Mapping[str, Any]]:
            del query_vector, filters, limit
            return [_row("paper-shared", 0.8)]

        def sparse_search(
            self, query: str, *, filters: object, limit: int
        ) -> Sequence[Mapping[str, Any]]:
            del query, filters, limit
            return [_row("paper-shared", 0.7)]

    embedder = RecordingEmbedder()
    results = retrieve(
        "Compare benefits and risks",
        dependencies=RetrievalDependencies(
            store=Store(),
            embedder=embedder,
            planner=Planner(),
            reranker=NoOpReranker(),
        ),
    )

    assert embedder.texts == ["benefits", "risks"]
    assert [result.document_id for result in results] == ["paper-shared"]


def test_clarify_plan_stops_before_retrieval() -> None:
    class Planner:
        def plan(self, message: str, history: object = ()) -> QueryPlan:
            del message, history
            return QueryPlan(action="clarify", search_intents=[])

    class UnusedAdapter:
        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            raise AssertionError(f"unexpected embedding call: {texts}")

        def dense_search(
            self, query_vector: Sequence[float], *, filters: object, limit: int
        ) -> Sequence[Mapping[str, Any]]:
            raise AssertionError(f"unexpected dense search: {query_vector}, {filters}, {limit}")

        def sparse_search(
            self, query: str, *, filters: object, limit: int
        ) -> Sequence[Mapping[str, Any]]:
            raise AssertionError(f"unexpected sparse search: {query}, {filters}, {limit}")

        def rerank(
            self, query: str, candidates: Sequence[RetrievalResult]
        ) -> list[RetrievalResult]:
            raise AssertionError(f"unexpected reranking: {query}, {candidates}")

    unused = UnusedAdapter()
    results = retrieve(
        "Here is a pasted abstract.",
        dependencies=RetrievalDependencies(
            store=unused,
            embedder=unused,
            planner=Planner(),
            reranker=unused,
        ),
    )

    assert results == []


def test_planned_retrieval_returns_cross_encoder_order() -> None:
    class Planner:
        def plan(self, message: str, history: object = ()) -> QueryPlan:
            del message, history
            return QueryPlan(
                action="retrieve",
                search_intents=[
                    SearchIntent(dense_query="semantic question", sparse_query="exact terms")
                ],
            )

    class Store:
        def dense_search(
            self, query_vector: Sequence[float], *, filters: object, limit: int
        ) -> Sequence[Mapping[str, Any]]:
            del query_vector, filters, limit
            return [_row("paper-a", 0.8), _row("paper-b", 0.7)]

        def sparse_search(
            self, query: str, *, filters: object, limit: int
        ) -> Sequence[Mapping[str, Any]]:
            del query, filters, limit
            return []

    class CrossEncoder:
        def rerank(
            self, query: str, candidates: Sequence[RetrievalResult]
        ) -> list[RetrievalResult]:
            assert query == "semantic question"
            return [
                replace(candidates[1], score=0.9),
                replace(candidates[0], score=0.4),
            ]

    results = retrieve(
        "raw message",
        dependencies=RetrievalDependencies(
            store=Store(),
            embedder=StubEmbedder(),
            planner=Planner(),
            reranker=CrossEncoder(),
        ),
    )

    assert [result.document_id for result in results] == ["paper-b", "paper-a"]


def test_flashrank_adapter_preserves_provenance_while_replacing_scores() -> None:
    candidates = [
        _result("paper-a", 0.02),
        _result("paper-b", 0.01),
    ]

    class Ranker:
        def rerank(self, request: Any) -> list[dict[str, object]]:
            passages = request.passages
            return [
                {**passages[1], "score": 0.9},
                {**passages[0], "score": 0.4},
            ]

    reranked = FlashRankCrossEncoder(ranker=Ranker()).rerank("query", candidates)

    assert [result.document_id for result in reranked] == ["paper-b", "paper-a"]
    assert [result.score for result in reranked] == [0.9, 0.4]
    assert reranked[0].page_numbers == candidates[1].page_numbers
    assert reranked[0].bboxes == candidates[1].bboxes
    assert reranked[0].original_text == candidates[1].original_text


def test_evidence_packing_keeps_only_complete_chunks_within_the_token_budget() -> None:
    class Planner:
        def plan(self, message: str, history: object = ()) -> QueryPlan:
            del message, history
            return QueryPlan(
                action="retrieve",
                search_intents=[SearchIntent(dense_query="question", sparse_query="terms")],
            )

    class Store:
        def dense_search(
            self, query_vector: Sequence[float], *, filters: object, limit: int
        ) -> Sequence[Mapping[str, Any]]:
            del query_vector, filters, limit
            return [
                _row("paper-a", 0.8, "alpha beta"),
                _row("paper-b", 0.7, "gamma delta"),
            ]

        def sparse_search(
            self, query: str, *, filters: object, limit: int
        ) -> Sequence[Mapping[str, Any]]:
            del query, filters, limit
            return []

    results = retrieve(
        "question",
        config=RetrievalConfig(source_budget_tokens=3),
        dependencies=RetrievalDependencies(
            store=Store(),
            embedder=StubEmbedder(),
            planner=Planner(),
            reranker=NoOpReranker(),
        ),
    )

    assert [result.original_text for result in results] == ["alpha beta"]
    assert results[0].bboxes == [{"page": 2, "l": 10.0, "t": 20.0, "r": 30.0, "b": 40.0}]


def test_evidence_packing_softly_prefers_document_diversity() -> None:
    class Planner:
        def plan(self, message: str, history: object = ()) -> QueryPlan:
            del message, history
            return QueryPlan(
                action="retrieve",
                search_intents=[SearchIntent(dense_query="question", sparse_query="terms")],
            )

    class Store:
        def dense_search(
            self, query_vector: Sequence[float], *, filters: object, limit: int
        ) -> Sequence[Mapping[str, Any]]:
            del query_vector, filters, limit
            return [
                _row("paper-a", 0.9, "first fact"),
                _row("paper-a", 0.8, "second fact"),
                _row("paper-b", 0.7, "other evidence"),
            ]

        def sparse_search(
            self, query: str, *, filters: object, limit: int
        ) -> Sequence[Mapping[str, Any]]:
            del query, filters, limit
            return []

    results = retrieve(
        "question",
        config=RetrievalConfig(
            source_budget_tokens=4,
            document_diversity_penalty=0.15,
        ),
        dependencies=RetrievalDependencies(
            store=Store(),
            embedder=StubEmbedder(),
            planner=Planner(),
            reranker=NoOpReranker(),
        ),
    )

    assert [result.document_id for result in results] == ["paper-a", "paper-b"]


def test_evidence_packing_has_no_fixed_per_document_cap() -> None:
    class Planner:
        def plan(self, message: str, history: object = ()) -> QueryPlan:
            del message, history
            return QueryPlan(
                action="retrieve",
                search_intents=[SearchIntent(dense_query="question", sparse_query="terms")],
            )

    class Store:
        def dense_search(
            self, query_vector: Sequence[float], *, filters: object, limit: int
        ) -> Sequence[Mapping[str, Any]]:
            del query_vector, filters, limit
            return [_row("paper-a", 0.9, f"fact {index}") for index in range(1, 5)]

        def sparse_search(
            self, query: str, *, filters: object, limit: int
        ) -> Sequence[Mapping[str, Any]]:
            del query, filters, limit
            return []

    results = retrieve(
        "question",
        config=RetrievalConfig(source_budget_tokens=20),
        dependencies=RetrievalDependencies(
            store=Store(),
            embedder=StubEmbedder(),
            planner=Planner(),
            reranker=NoOpReranker(),
        ),
    )

    assert [result.document_id for result in results] == ["paper-a"] * 4


def test_retrieve_returns_ranked_chunks_with_complete_provenance() -> None:
    results = retrieve_baseline(
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

    results = retrieve_baseline(
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

    results = retrieve_baseline(
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

    results = retrieve_baseline(
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
    embedder = _openai_embedder()

    ingest_pdf(
        source,
        IngestDependencies(
            embedder=embedder,
            store=LanceDBNodeStore(index_dir),
        ),
    )

    class Planner:
        def plan(self, message: str, history: object = ()) -> QueryPlan:
            del message, history
            return QueryPlan(
                action="retrieve",
                search_intents=[
                    SearchIntent(
                        dense_query="Can epigenetic clocks generalize to insects?",
                        sparse_query="epigenetic clock insects",
                    )
                ],
            )

    results = retrieve(
        "epigenetic clock in insects",
        dependencies=RetrievalDependencies(
            store=LanceDBHybridStore(index_dir),
            embedder=embedder,
            planner=Planner(),
            reranker=NoOpReranker(),
        ),
    )

    assert results
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


@pytest.mark.e2e
def test_lancedb_applies_document_filter_before_candidate_limit(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    rows = [
        {
            "id": "other",
            "vector": [1.0, 0.0],
            "text": "other paper",
            "metadata": _row("001-other-paper", 0.0)["metadata"],
        },
        {
            "id": "target",
            "vector": [0.0, 1.0],
            "text": "target paper",
            "metadata": _row("015-hickson-2019", 0.0)["metadata"],
        },
    ]
    lancedb.connect(str(index_dir)).create_table("chunks", data=rows)
    store = LanceDBHybridStore(index_dir)

    results = store.dense_search(
        [1.0, 0.0],
        filters=MetadataFilters(document_id="015-hickson-2019"),
        limit=1,
    )

    assert [row["id"] for row in results] == ["target"]


@pytest.mark.e2e
def test_lancedb_matches_an_accented_author_filter(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    lancedb.connect(str(index_dir)).create_table(
        "chunks",
        data=[
            {
                "id": "target",
                "vector": [1.0, 0.0],
                "text": "target paper",
                "metadata": _row("002-garcia-barranquero-2025", 0.0)["metadata"],
            }
        ],
    )

    results = LanceDBHybridStore(index_dir).dense_search(
        [1.0, 0.0],
        filters=MetadataFilters(author="García Barranquero"),
        limit=1,
    )

    assert [row["id"] for row in results] == ["target"]
