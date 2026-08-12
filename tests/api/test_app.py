import asyncio
import json
import logging
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import cast

import httpx
import pytest

from live_long_rnd.api.app import ApplicationConfig, create_app
from live_long_rnd.api.conversations import StoredMessage
from live_long_rnd.api.llm import StubLLM
from live_long_rnd.api.retrieval import RetrievedChunk, StubRetriever


class HistoryEchoLLM:
    def stream_answer(
        self,
        message: str,
        chunks: Sequence[RetrievedChunk],
        history: Sequence[StoredMessage],
    ) -> AsyncIterator[str]:
        del message, chunks

        async def tokens() -> AsyncIterator[str]:
            yield f"{history[0].text} [1]." if history else "No prior history [1]."

        return tokens()


class RecordingHistoryLLM:
    def __init__(self) -> None:
        self.histories: list[list[str]] = []

    def stream_answer(
        self,
        message: str,
        chunks: Sequence[RetrievedChunk],
        history: Sequence[StoredMessage],
    ) -> AsyncIterator[str]:
        del message, chunks
        self.histories.append([item.text for item in history])

        async def tokens() -> AsyncIterator[str]:
            yield "answer [1]."

        return tokens()


class FirstTitleRaceLLM:
    def __init__(self) -> None:
        self.second_request_started = asyncio.Event()
        self.allow_second_request = asyncio.Event()

    def stream_answer(
        self,
        message: str,
        chunks: Sequence[RetrievedChunk],
        history: Sequence[StoredMessage],
    ) -> AsyncIterator[str]:
        del chunks, history

        async def tokens() -> AsyncIterator[str]:
            if message == "second request":
                self.second_request_started.set()
                await self.allow_second_request.wait()
            else:
                await self.second_request_started.wait()
            yield "answer [1]."

        return tokens()


@pytest.mark.e2e
def test_new_chat_streams_conversation_before_answer(
    tmp_path: Path,
) -> None:
    async def exercise() -> tuple[httpx.Response, str]:
        transport = httpx.ASGITransport(
            app=create_app(
                retriever=StubRetriever(),
                llm_client=StubLLM(),
                config=ApplicationConfig(database_path=tmp_path / "conversations.db"),
            )
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
        "conversation",
        "token",
        "citations",
        "done",
    ]
    assert events[0]["title"] == "What do senolytics target?"
    assert isinstance(events[0]["id"], str)
    assert events[0]["id"]
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
def test_completed_chat_is_retrievable_with_citations(tmp_path: Path) -> None:
    async def exercise() -> tuple[list[dict[str, object]], httpx.Response]:
        transport = httpx.ASGITransport(
            app=create_app(
                retriever=StubRetriever(),
                llm_client=StubLLM(),
                config=ApplicationConfig(database_path=tmp_path / "conversations.db"),
            )
        )
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            stream = await client.post(
                "/api/chat",
                json={"message": "What do senolytics target?"},
            )
            events = [
                json.loads(frame.removeprefix("data: "))
                for frame in stream.text.rstrip("\n").split("\n\n")
            ]
            conversation = await client.get(f"/api/conversations/{events[0]['id']}")
            return events, conversation

    events, response = asyncio.run(exercise())

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == events[0]["id"]
    assert payload["title"] == "What do senolytics target?"
    assert [message["role"] for message in payload["messages"]] == [
        "user",
        "assistant",
    ]
    assert payload["messages"][0]["text"] == "What do senolytics target?"
    assert payload["messages"][0]["citations"] == []
    assert payload["messages"][1]["text"].startswith("Senolytics are drugs")
    assert payload["messages"][1]["citations"] == events[-2]["citations"]
    assert all(message["created_at"] for message in payload["messages"])


@pytest.mark.e2e
def test_first_chat_uses_and_titles_an_empty_conversation(tmp_path: Path) -> None:
    async def exercise() -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
        transport = httpx.ASGITransport(
            app=create_app(
                retriever=StubRetriever(),
                llm_client=StubLLM(),
                config=ApplicationConfig(database_path=tmp_path / "conversations.db"),
            )
        )
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            created = (await client.post("/api/conversations")).json()
            stream = await client.post(
                "/api/chat",
                json={
                    "conversation_id": created["id"],
                    "message": "How do epigenetic clocks compare with mortality markers?",
                },
            )
            events = [
                json.loads(frame.removeprefix("data: "))
                for frame in stream.text.rstrip("\n").split("\n\n")
            ]
            loaded = (await client.get(f"/api/conversations/{created['id']}")).json()
            return created, events, loaded

    created, events, loaded = asyncio.run(exercise())

    assert created["title"] == "New conversation"
    assert events[0] == {
        "type": "conversation",
        "id": created["id"],
        "title": "How do epigenetic clocks compare with mortality markers?",
    }
    assert loaded["id"] == created["id"]
    assert loaded["title"] == events[0]["title"]
    loaded_messages = cast(list[dict[str, object]], loaded["messages"])
    assert [message["role"] for message in loaded_messages] == [
        "user",
        "assistant",
    ]


@pytest.mark.e2e
def test_first_completed_chat_claims_an_empty_conversation_title(tmp_path: Path) -> None:
    async def exercise() -> dict[str, object]:
        llm = FirstTitleRaceLLM()
        transport = httpx.ASGITransport(
            app=create_app(
                retriever=StubRetriever(),
                llm_client=llm,
                config=ApplicationConfig(database_path=tmp_path / "conversations.db"),
            )
        )
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            created = (await client.post("/api/conversations")).json()
            second = asyncio.create_task(
                client.post(
                    "/api/chat",
                    json={
                        "conversation_id": created["id"],
                        "message": "second request",
                    },
                )
            )
            await llm.second_request_started.wait()
            first = await client.post(
                "/api/chat",
                json={
                    "conversation_id": created["id"],
                    "message": "first request",
                },
            )
            llm.allow_second_request.set()
            completed_second = await second
            loaded = await client.get(f"/api/conversations/{created['id']}")
            assert first.status_code == 200
            assert first.headers["content-type"].startswith("text/event-stream")
            assert completed_second.status_code == 200
            assert completed_second.headers["content-type"].startswith("text/event-stream")
            return cast(dict[str, object], loaded.json())

    conversation = asyncio.run(exercise())

    assert conversation["title"] == "first request"


@pytest.mark.e2e
def test_conversations_are_sorted_by_last_activity(tmp_path: Path) -> None:
    async def exercise() -> tuple[list[dict[str, object]], str, str]:
        transport = httpx.ASGITransport(
            app=create_app(
                retriever=StubRetriever(),
                llm_client=StubLLM(),
                config=ApplicationConfig(database_path=tmp_path / "conversations.db"),
            )
        )
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            active = (await client.post("/api/conversations")).json()
            inactive = (await client.post("/api/conversations")).json()
            await client.post(
                "/api/chat",
                json={
                    "conversation_id": active["id"],
                    "message": "What did the first senolytics trial measure?",
                },
            )
            listed = (await client.get("/api/conversations")).json()
            return listed, active["id"], inactive["id"]

    listed, active_id, inactive_id = asyncio.run(exercise())

    assert [item["id"] for item in listed] == [active_id, inactive_id]
    assert listed[0]["title"] == "What did the first senolytics trial measure?"
    assert all(item["updated_at"] for item in listed)


@pytest.mark.e2e
def test_deleted_conversation_is_no_longer_retrievable(tmp_path: Path) -> None:
    async def exercise() -> tuple[httpx.Response, httpx.Response, list[object]]:
        transport = httpx.ASGITransport(
            app=create_app(config=ApplicationConfig(database_path=tmp_path / "conversations.db"))
        )
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            created = (await client.post("/api/conversations")).json()
            deleted = await client.delete(f"/api/conversations/{created['id']}")
            loaded = await client.get(f"/api/conversations/{created['id']}")
            listed = (await client.get("/api/conversations")).json()
            return deleted, loaded, listed

    deleted, loaded, listed = asyncio.run(exercise())

    assert deleted.status_code == 204
    assert deleted.content == b""
    assert loaded.status_code == 404
    assert listed == []


@pytest.mark.e2e
def test_follow_up_answer_receives_the_completed_history(tmp_path: Path) -> None:
    async def exercise() -> list[dict[str, object]]:
        transport = httpx.ASGITransport(
            app=create_app(
                retriever=StubRetriever(),
                llm_client=HistoryEchoLLM(),
                config=ApplicationConfig(database_path=tmp_path / "conversations.db"),
            )
        )
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            first = await client.post(
                "/api/chat",
                json={"message": "What did the Hickson trial report?"},
            )
            first_events = [
                json.loads(frame.removeprefix("data: "))
                for frame in first.text.rstrip("\n").split("\n\n")
            ]
            second = await client.post(
                "/api/chat",
                json={
                    "conversation_id": first_events[0]["id"],
                    "message": "What about its limitations?",
                },
            )
            return [
                json.loads(frame.removeprefix("data: "))
                for frame in second.text.rstrip("\n").split("\n\n")
            ]

    events = asyncio.run(exercise())

    answer = "".join(str(event["text"]) for event in events if event["type"] == "token")
    assert answer == "What did the Hickson trial report? [1]."


@pytest.mark.e2e
def test_history_drops_whole_old_turns_with_visible_notice(tmp_path: Path) -> None:
    llm = RecordingHistoryLLM()

    async def exercise() -> tuple[list[dict[str, object]], dict[str, object]]:
        transport = httpx.ASGITransport(
            app=create_app(
                retriever=StubRetriever(),
                llm_client=llm,
                config=ApplicationConfig(
                    database_path=tmp_path / "conversations.db",
                    history_token_budget=7,
                ),
            )
        )
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            first = await client.post("/api/chat", json={"message": "one"})
            first_event = json.loads(first.text.split("\n\n", maxsplit=1)[0].removeprefix("data: "))
            for message in ("two", "three"):
                response = await client.post(
                    "/api/chat",
                    json={
                        "conversation_id": first_event["id"],
                        "message": message,
                    },
                )
            events = [
                json.loads(frame.removeprefix("data: "))
                for frame in response.text.rstrip("\n").split("\n\n")
            ]
            conversation = (await client.get(f"/api/conversations/{first_event['id']}")).json()
            return events, conversation

    events, conversation = asyncio.run(exercise())

    assert llm.histories == [[], ["one", "answer [1]."], ["two", "answer [1]."]]
    assert {
        "type": "history_notice",
        "text": "Earlier messages were dropped to fit the context window",
    } in events
    conversation_messages = cast(
        list[dict[str, object]],
        conversation["messages"],
    )
    assert [message["role"] for message in conversation_messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "system",
        "user",
        "assistant",
    ]
    assert conversation_messages[4]["text"] == (
        "Earlier messages were dropped to fit the context window"
    )


@pytest.mark.e2e
def test_conversation_load_returns_only_the_last_thirty_turns(tmp_path: Path) -> None:
    llm = RecordingHistoryLLM()

    async def exercise() -> dict[str, object]:
        transport = httpx.ASGITransport(
            app=create_app(
                retriever=StubRetriever(),
                llm_client=llm,
                config=ApplicationConfig(database_path=tmp_path / "conversations.db"),
            )
        )
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            conversation_id: str | None = None
            for number in range(1, 32):
                response = await client.post(
                    "/api/chat",
                    json={
                        "conversation_id": conversation_id,
                        "message": f"question {number}",
                    },
                )
                if conversation_id is None:
                    first_event = json.loads(
                        response.text.split("\n\n", maxsplit=1)[0].removeprefix("data: ")
                    )
                    conversation_id = str(first_event["id"])
            response = await client.get(f"/api/conversations/{conversation_id}")
            return cast(dict[str, object], response.json())

    conversation = asyncio.run(exercise())

    messages = cast(list[dict[str, object]], conversation["messages"])
    assert len(messages) == 60
    assert messages[0]["text"] == "question 2"
    assert messages[-2]["text"] == "question 31"
    assert llm.histories[-1][0] == "question 1"


@pytest.mark.e2e
def test_configured_sqlite_history_survives_a_new_application(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "persistent" / "conversations.db"
    monkeypatch.setenv("LIVE_LONG_CONVERSATIONS_DB", str(database_path))

    async def exercise() -> tuple[dict[str, object], list[dict[str, object]]]:
        first_transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(
            transport=first_transport,
            base_url="http://testserver",
        ) as client:
            created = (await client.post("/api/conversations")).json()

        second_transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(
            transport=second_transport,
            base_url="http://testserver",
        ) as client:
            listed = (await client.get("/api/conversations")).json()
        return created, listed

    created, listed = asyncio.run(exercise())

    assert database_path.is_file()
    assert listed == [
        {
            "id": created["id"],
            "title": "New conversation",
            "updated_at": listed[0]["updated_at"],
        }
    ]


@pytest.mark.e2e
def test_chat_emits_setup_error_when_openai_key_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("LIVE_LONG_LLM", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def exercise() -> tuple[list[dict[str, object]], dict[str, object]]:
        transport = httpx.ASGITransport(
            app=create_app(config=ApplicationConfig(database_path=tmp_path / "conversations.db"))
        )
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            stream = await client.post(
                "/api/chat",
                json={"message": "What do senolytics target?"},
            )
            events = [
                json.loads(frame.removeprefix("data: "))
                for frame in stream.text.rstrip("\n").split("\n\n")
            ]
            conversation = (await client.get(f"/api/conversations/{events[0]['id']}")).json()
            return events, conversation

    with caplog.at_level(logging.ERROR):
        events, conversation = asyncio.run(exercise())

    assert events[-1] == {
        "type": "error",
        "message": (
            "OpenAI is selected but OPENAI_API_KEY is not set. "
            "Set OPENAI_API_KEY and restart the server."
        ),
    }
    assert conversation["messages"] == []
    assert conversation["title"] == "New conversation"
    error_record = next(record for record in caplog.records if record.levelno == logging.ERROR)
    assert error_record.getMessage() == f"Chat turn failed for conversation {events[0]['id']}."
    assert error_record.exc_info is not None
