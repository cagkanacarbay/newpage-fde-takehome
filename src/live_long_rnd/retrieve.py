"""Hybrid dense and BM25 retrieval with citation-ready provenance."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, TypedDict, cast

import lancedb
import tiktoken
from flashrank import Ranker as FlashRanker
from flashrank import RerankRequest
from lancedb.rerankers import RRFReranker

from live_long_rnd.embeddings import EMBEDDING_ENCODING_NAME, Embedder, OpenAIEmbedder
from live_long_rnd.index_config import DEFAULT_INDEX_DIR, DEFAULT_TABLE_NAME
from live_long_rnd.query_planning import (
    ConversationMessage,
    MetadataFilters,
    OpenAIQueryPlanner,
    QueryPlanner,
)

FUSION_CANDIDATE_MULTIPLIER = 4
RRF_RANK_CONSTANT = 60
FLASHRANK_MODEL = "ms-marco-MiniLM-L-12-v2"
FLASHRANK_MAX_LENGTH = 512
DEFAULT_RERANKER_CACHE_DIR = Path("data/models/flashrank")
DEFAULT_CANDIDATE_DEPTH = 40
DEFAULT_SOURCE_BUDGET_TOKENS = 12_000
DEFAULT_DOCUMENT_DIVERSITY_PENALTY = 0.15


class HybridStore(Protocol):
    """Store boundary that returns RRF-fused rows in relevance order."""

    def hybrid_search(
        self, query: str, query_vector: Sequence[float], *, limit: int
    ) -> Sequence[Mapping[str, Any]]:
        """Run dense and BM25 search, then return fused rows."""


class PlannedStore(Protocol):
    """Store seam that keeps semantic and lexical searches distinct."""

    def dense_search(
        self,
        query_vector: Sequence[float],
        *,
        filters: MetadataFilters,
        limit: int,
    ) -> Sequence[Mapping[str, Any]]: ...

    def sparse_search(
        self,
        query: str,
        *,
        filters: MetadataFilters,
        limit: int,
    ) -> Sequence[Mapping[str, Any]]: ...


class Reranker(Protocol):
    """Cross-encoder seam that orders fused candidates by relevance."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
    ) -> list[RetrievalResult]: ...


@dataclass(frozen=True)
class RetrievalConfig:
    """Measured runtime values for candidate search and evidence packing."""

    candidate_depth: int = DEFAULT_CANDIDATE_DEPTH
    source_budget_tokens: int = DEFAULT_SOURCE_BUDGET_TOKENS
    document_diversity_penalty: float = DEFAULT_DOCUMENT_DIVERSITY_PENALTY


@dataclass(frozen=True)
class RetrievalDependencies:
    """Replaceable adapters used by one planned retrieval operation."""

    store: PlannedStore | None = None
    embedder: Embedder | None = None
    planner: QueryPlanner | None = None
    reranker: Reranker | None = None


@dataclass(frozen=True)
class _ActiveRuntime:
    store: PlannedStore
    embedder: Embedder
    planner: QueryPlanner
    reranker: Reranker


DEFAULT_RETRIEVAL_CONFIG = RetrievalConfig()


class FlashRankClient(Protocol):
    """FlashRank subset used by the cross-encoder adapter."""

    def rerank(self, request: RerankRequest) -> list[dict[str, Any]]: ...


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

    def dense_search(
        self,
        query_vector: Sequence[float],
        *,
        filters: MetadataFilters,
        limit: int,
    ) -> list[Mapping[str, Any]]:
        """Search only the vector leg with the semantic query embedding."""
        search = self._table.search(
            list(query_vector),
            query_type="vector",
            vector_column_name="vector",
        )
        search = _apply_store_filters(search, filters)
        rows = search.limit(limit).to_list()
        return _filter_rows(cast(list[Mapping[str, Any]], rows), filters)

    def sparse_search(
        self,
        query: str,
        *,
        filters: MetadataFilters,
        limit: int,
    ) -> list[Mapping[str, Any]]:
        """Search only the BM25 leg with the literal lexical query."""
        search = self._table.search(
            query,
            query_type="fts",
            fts_columns="text",
        )
        search = _apply_store_filters(search, filters)
        rows = search.limit(limit).to_list()
        return _filter_rows(cast(list[Mapping[str, Any]], rows), filters)


@dataclass
class RetrievalResult:
    """One winning chunk with its score and exact source provenance."""

    document_id: str
    page_numbers: list[int]
    bboxes: list[dict[str, int | float]]
    heading_path: list[str]
    original_text: str
    score: float


class FlashRankCrossEncoder:
    """Local ONNX cross-encoder that retains each candidate's provenance."""

    def __init__(
        self,
        *,
        model_name: str = FLASHRANK_MODEL,
        cache_dir: Path = DEFAULT_RERANKER_CACHE_DIR,
        ranker: FlashRankClient | None = None,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._ranker = ranker

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
    ) -> list[RetrievalResult]:
        if not candidates:
            return []
        ranker = self._ranker
        if ranker is None:
            ranker = cast(
                FlashRankClient,
                FlashRanker(
                    model_name=self._model_name,
                    cache_dir=str(self._cache_dir),
                    max_length=FLASHRANK_MAX_LENGTH,
                ),
            )
            self._ranker = ranker

        passages = [
            {"id": index, "text": candidate.original_text}
            for index, candidate in enumerate(candidates)
        ]
        ranked_passages = ranker.rerank(RerankRequest(query=query, passages=passages))
        return [
            replace(candidates[int(passage["id"])], score=float(passage["score"]))
            for passage in ranked_passages
        ]


class IdentityReranker:
    """Keep fused order for controlled reranker calibration."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
    ) -> list[RetrievalResult]:
        del query
        return list(candidates)


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
    history: Sequence[ConversationMessage] = (),
    config: RetrievalConfig = DEFAULT_RETRIEVAL_CONFIG,
    dependencies: RetrievalDependencies | None = None,
) -> list[RetrievalResult]:
    """Plan, retrieve, rerank, and token-pack complete evidence."""
    selected = dependencies or RetrievalDependencies()
    runtime = _ActiveRuntime(
        store=selected.store or LanceDBHybridStore(DEFAULT_INDEX_DIR),
        embedder=selected.embedder or OpenAIEmbedder(),
        planner=selected.planner or OpenAIQueryPlanner(),
        reranker=selected.reranker or FlashRankCrossEncoder(),
    )
    return _retrieve_planned(query, history, config, runtime)


def retrieve_baseline(
    query: str,
    *,
    k: int = 10,
    per_document_cap: int | None = 3,
    store: HybridStore | None = None,
    embedder: Embedder | None = None,
) -> list[RetrievalResult]:
    """Run the PR #31 hybrid retriever unchanged for calibration."""
    active_store = store or LanceDBHybridStore(DEFAULT_INDEX_DIR)
    active_embedder = embedder or OpenAIEmbedder()
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


def _retrieve_planned(
    query: str,
    history: Sequence[ConversationMessage],
    config: RetrievalConfig,
    runtime: _ActiveRuntime,
) -> list[RetrievalResult]:
    plan = runtime.planner.plan(query, history)
    if plan.action == "clarify":
        return []

    dense_queries = [intent.dense_query for intent in plan.search_intents]
    query_vectors = runtime.embedder.embed(dense_queries)
    ranked_lists: list[Sequence[Mapping[str, Any]]] = []
    candidate_limit = config.candidate_depth
    for intent, query_vector in zip(plan.search_intents, query_vectors, strict=True):
        ranked_lists.append(
            runtime.store.dense_search(query_vector, filters=intent.filters, limit=candidate_limit)
        )
        ranked_lists.append(
            runtime.store.sparse_search(
                intent.sparse_query,
                filters=intent.filters,
                limit=candidate_limit,
            )
        )

    rows_by_key: dict[str, Mapping[str, Any]] = {}
    scores_by_key: dict[str, float] = {}
    for ranked_rows in ranked_lists:
        for rank, row in enumerate(ranked_rows, start=1):
            key = _candidate_key(row)
            rows_by_key.setdefault(key, row)
            scores_by_key[key] = scores_by_key.get(key, 0.0) + 1.0 / (RRF_RANK_CONSTANT + rank)

    ranked_keys = sorted(scores_by_key, key=scores_by_key.__getitem__, reverse=True)
    results: list[RetrievalResult] = []
    for key in ranked_keys[: config.candidate_depth]:
        row = dict(rows_by_key[key])
        row["_relevance_score"] = scores_by_key[key]
        results.append(_result_from_row(row))
    rerank_query = "\n".join(intent.dense_query for intent in plan.search_intents)
    results = runtime.reranker.rerank(rerank_query, results)
    return _pack_evidence(
        results,
        source_budget_tokens=config.source_budget_tokens,
        document_diversity_penalty=config.document_diversity_penalty,
    )


def _candidate_key(row: Mapping[str, Any]) -> str:
    if row.get("id"):
        return str(row["id"])
    metadata = cast(Mapping[str, Any], row["metadata"])
    return "\x1f".join(
        str(metadata.get(field, "")) for field in ("document_id", "page_numbers", "original_text")
    )


def _filter_rows(
    rows: Sequence[Mapping[str, Any]], filters: MetadataFilters
) -> list[Mapping[str, Any]]:
    """Apply high-confidence filters without interpolating values into SQL."""
    selected: list[Mapping[str, Any]] = []
    for row in rows:
        metadata = cast(Mapping[str, Any], row["metadata"])
        document_id = str(metadata["document_id"])
        if filters.document_id is not None and document_id != filters.document_id:
            continue
        if filters.author is not None:
            author_slug = filters.author.casefold().replace(" ", "-")
            if f"-{author_slug}-" not in f"-{document_id.casefold()}-":
                continue
        selected.append(row)
    return selected


def _apply_store_filters(search: Any, filters: MetadataFilters) -> Any:
    clauses: list[str] = []
    if filters.document_id is not None:
        clauses.append(f"metadata.document_id = '{_sql_literal(filters.document_id)}'")
    if filters.author is not None:
        author_slug = _slug(filters.author)
        clauses.append(f"metadata.document_id LIKE '%-{_sql_literal(author_slug)}-%'")
    if not clauses:
        return search
    return search.where(" AND ".join(clauses), prefilter=True)


def _sql_literal(value: str) -> str:
    return value.replace("'", "''")


def _slug(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_value.casefold()).strip("-")


def _pack_evidence(
    candidates: Sequence[RetrievalResult],
    *,
    source_budget_tokens: int,
    document_diversity_penalty: float,
) -> list[RetrievalResult]:
    encoding = tiktoken.get_encoding(EMBEDDING_ENCODING_NAME)
    packed: list[RetrievalResult] = []
    used_tokens = 0
    document_counts: dict[str, int] = {}
    remaining = list(candidates)
    while remaining:
        candidate = max(
            remaining,
            key=lambda item: (
                item.score
                / (1.0 + document_diversity_penalty * document_counts.get(item.document_id, 0))
            ),
        )
        remaining.remove(candidate)
        candidate_tokens = len(encoding.encode(candidate.original_text))
        if used_tokens + candidate_tokens > source_budget_tokens:
            continue
        packed.append(candidate)
        used_tokens += candidate_tokens
        document_counts[candidate.document_id] = document_counts.get(candidate.document_id, 0) + 1
    return packed
