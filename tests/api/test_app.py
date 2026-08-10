import asyncio
import json

import httpx
import pytest

from live_long_rnd.api.app import create_app
from live_long_rnd.api.llm import StubLLM
from live_long_rnd.api.retrieval import StubRetriever


@pytest.mark.e2e
def test_chat_streams_answer_citations_and_done() -> None:
    async def exercise() -> tuple[httpx.Response, str]:
        transport = httpx.ASGITransport(
            app=create_app(retriever=StubRetriever(), llm_client=StubLLM())
        )
        async with (
            httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client,
            client.stream(
                "POST",
                "/api/chat",
                json={"message": "What do senolytics target?"},
            ) as response,
        ):
            body = "".join([part async for part in response.aiter_text()])
            return response, body

    response, body = asyncio.run(exercise())
    frames = body.rstrip("\n").split("\n\n")
    assert all(frame.startswith("data: ") for frame in frames)
    events = [json.loads(frame.removeprefix("data: ")) for frame in frames]

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert [event["type"] for event in events] == [
        "token",
        "token",
        "token",
        "token",
        "citations",
        "done",
    ]
    citations = events[-2]["citations"]
    assert len(citations) == 3
    assert citations[0] == {
        "document_id": ("015-hickson-2019-senolytics-dasatinib-quercetin-first-in-human"),
        "page": 2,
        "heading_path": ["Research in context", "1. Introduction"],
        "bbox": {"l": 310.5, "t": 332.6, "r": 561.6, "b": 53.3},
        "snippet": "By definition, the target of senolytics is senescent cells...",
    }
    assert all(
        set(citation)
        == {
            "document_id",
            "page",
            "heading_path",
            "bbox",
            "snippet",
        }
        and set(citation["bbox"]) == {"l", "t", "r", "b"}
        for citation in citations
    )


@pytest.mark.e2e
def test_chat_emits_setup_error_when_openai_key_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LIVE_LONG_LLM", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def exercise() -> str:
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/api/chat",
                json={"message": "What do senolytics target?"},
            )
            return response.text

    assert asyncio.run(exercise()) == (
        'data: {"type":"error","message":"OpenAI is selected but '
        'OPENAI_API_KEY is not set. Set OPENAI_API_KEY and restart the server."}\n\n'
    )
