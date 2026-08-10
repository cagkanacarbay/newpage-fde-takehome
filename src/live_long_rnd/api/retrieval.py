import asyncio
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, TypedDict, cast

from live_long_rnd.ingest import Embedder
from live_long_rnd.retrieve import HybridStore, to_citation_payload
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
    async def retrieve(self, message: str) -> Sequence[RetrievedChunk]: ...


class LanceDbRetriever:
    """The real hybrid retriever: LanceDB dense + BM25 with RRF and a per-document cap."""

    def __init__(
        self,
        *,
        k: int = 10,
        per_document_cap: int | None = 3,
        store: HybridStore | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._k = k
        self._per_document_cap = per_document_cap
        self._store = store
        self._embedder = embedder

    async def retrieve(self, message: str) -> Sequence[RetrievedChunk]:
        results = await asyncio.to_thread(
            hybrid_retrieve,
            message,
            k=self._k,
            per_document_cap=self._per_document_cap,
            store=self._store,
            embedder=self._embedder,
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
    async def retrieve(self, message: str) -> Sequence[RetrievedChunk]:
        del message
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
