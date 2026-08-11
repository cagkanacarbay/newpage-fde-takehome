import asyncio

from live_long_rnd.api.retrieval import StubRetriever


def test_stub_retriever_returns_cited_longevity_chunks() -> None:
    chunks = asyncio.run(StubRetriever().retrieve("What do senolytics target?"))

    assert len(chunks) == 3
    assert chunks[0].citation == {
        "document_id": ("015-hickson-2019-senolytics-dasatinib-quercetin-first-in-human"),
        "page": 2,
        "heading_path": ["Research in context", "1. Introduction"],
        "bbox": {"l": 310.5, "t": 332.6, "r": 561.6, "b": 53.3},
        "snippet": ("By definition, the target of senolytics is senescent cells..."),
    }
    assert all(chunk.text and chunk.citation["document_id"] for chunk in chunks)
