from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypedDict


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
