import asyncio
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, TypedDict, cast

from live_long_rnd.embeddings import OpenAIEmbedder
from live_long_rnd.index_config import DEFAULT_INDEX_DIR
from live_long_rnd.query_planning import ConversationMessage, OpenAIQueryPlanner
from live_long_rnd.retrieve import (
    DEFAULT_RETRIEVAL_CONFIG,
    FlashRankCrossEncoder,
    LanceDBHybridStore,
    RetrievalConfig,
    RetrievalDependencies,
    to_citation_payload,
)
from live_long_rnd.retrieve import retrieve as hybrid_retrieve


class BoundingBox(TypedDict):
    l: float  # noqa: E741 - required by the frontend citation contract
    t: float
    r: float
    b: float


class Citation(TypedDict):
    document_id: str
    page: int
    heading_path: list[str]
    bbox: BoundingBox
    snippet: str


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    citation: Citation


class Retriever(Protocol):
    async def retrieve(
        self,
        message: str,
        history: Sequence[ConversationMessage] = (),
    ) -> Sequence[RetrievedChunk]: ...


class LanceDbRetriever:
    """Plan, search, rerank, and token-pack evidence from LanceDB."""

    def __init__(
        self,
        *,
        config: RetrievalConfig = DEFAULT_RETRIEVAL_CONFIG,
        dependencies: RetrievalDependencies | None = None,
    ) -> None:
        self._config = config
        self._dependencies = dependencies

    def _dependencies_for_retrieval(self) -> RetrievalDependencies:
        if self._dependencies is None:
            self._dependencies = RetrievalDependencies(
                store=LanceDBHybridStore(DEFAULT_INDEX_DIR),
                embedder=OpenAIEmbedder(),
                planner=OpenAIQueryPlanner(),
                reranker=FlashRankCrossEncoder(),
            )
        return self._dependencies

    async def retrieve(
        self,
        message: str,
        history: Sequence[ConversationMessage] = (),
    ) -> Sequence[RetrievedChunk]:
        results = await asyncio.to_thread(
            hybrid_retrieve,
            message,
            history=history,
            config=self._config,
            dependencies=self._dependencies_for_retrieval(),
        )
        return [
            RetrievedChunk(
                text=result.original_text,
                citation=cast(Citation, to_citation_payload(result)),
            )
            for result in results
        ]


def create_retriever(settings: Mapping[str, str] | None = None) -> Retriever:
    """Pick the retriever from configuration; stub unless lancedb is requested."""
    environ = os.environ if settings is None else settings
    adapter = environ.get("LIVE_LONG_RETRIEVER", "stub").strip().lower()
    if adapter == "stub":
        return StubRetriever()
    if adapter == "lancedb":
        return LanceDbRetriever()
    raise ValueError(f"Unsupported LIVE_LONG_RETRIEVER value {adapter!r}. Use 'stub' or 'lancedb'.")


class StubRetriever:
    async def retrieve(
        self,
        message: str,
        history: Sequence[ConversationMessage] = (),
    ) -> Sequence[RetrievedChunk]:
        del message, history
        return (
            RetrievedChunk(
                text=(
                    "Senolytics selectively target senescent cells. Dasatinib and "
                    "quercetin were evaluated together in the first human trial."
                ),
                citation={
                    "document_id": (
                        "015-hickson-2019-senolytics-dasatinib-quercetin-first-in-human"
                    ),
                    "page": 2,
                    "heading_path": ["Research in context", "1. Introduction"],
                    "bbox": {"l": 310.5, "t": 332.6, "r": 561.6, "b": 53.3},
                    "snippet": ("By definition, the target of senolytics is senescent cells..."),
                },
            ),
            RetrievedChunk(
                text=(
                    "A pilot study reported improved physical function after "
                    "intermittent senolytic treatment in pulmonary fibrosis."
                ),
                citation={
                    "document_id": (
                        "016-justice-2019-senolytics-idiopathic-pulmonary-fibrosis-trial"
                    ),
                    "page": 5,
                    "heading_path": ["Results", "Physical function"],
                    "bbox": {"l": 72.4, "t": 418.2, "r": 286.1, "b": 294.8},
                    "snippet": ("Physical function measures improved after senolytic therapy..."),
                },
            ),
            RetrievedChunk(
                text=(
                    "Senolytic treatment did not reverse the established DNA "
                    "methylation signature of cellular senescence."
                ),
                citation={
                    "document_id": (
                        "017-kasamoto-2026-senolytics-do-not-reverse-senescence-methylation"
                    ),
                    "page": 7,
                    "heading_path": ["Discussion"],
                    "bbox": {"l": 301.7, "t": 517.9, "r": 557.2, "b": 351.4},
                    "snippet": (
                        "Senolytics did not reverse senescence-associated DNA methylation..."
                    ),
                },
            ),
        )
