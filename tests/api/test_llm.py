import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from live_long_rnd.api.llm import (
    LLMConfigurationError,
    OpenAILLM,
    StubLLM,
    create_llm_client,
)
from live_long_rnd.api.retrieval import StubRetriever


async def _collect(stream: AsyncIterator[str]) -> list[str]:
    return [token async for token in stream]


def test_stub_llm_streams_a_deterministic_evidence_based_answer() -> None:
    async def exercise() -> list[str]:
        chunks = await StubRetriever().retrieve("What do senolytics target?")
        return await _collect(
            StubLLM().stream_answer(
                "What do senolytics target?",
                chunks,
            )
        )

    tokens = asyncio.run(exercise())

    assert len(tokens) > 1
    assert "".join(tokens) == (
        "Senolytics are drugs designed to selectively target senescent cells. "
        "Early human studies suggest possible physical-function benefits, while "
        "newer evidence shows they may not reverse established DNA methylation "
        "signatures of senescence."
    )


def test_openai_llm_requires_an_api_key_when_streaming() -> None:
    async def exercise() -> None:
        chunks = await StubRetriever().retrieve("What do senolytics target?")
        await _collect(
            OpenAILLM(api_key=None, model="gpt-5.6-luna").stream_answer(
                "What do senolytics target?",
                chunks,
            )
        )

    with pytest.raises(LLMConfigurationError) as error:
        asyncio.run(exercise())

    assert str(error.value) == (
        "OpenAI is selected but OPENAI_API_KEY is not set. "
        "Set OPENAI_API_KEY and restart the server."
    )


def test_openai_llm_lazily_streams_with_high_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeResponses:
        async def create(self, **kwargs: object) -> AsyncIterator[SimpleNamespace]:
            captured.update(kwargs)

            async def events() -> AsyncIterator[SimpleNamespace]:
                yield SimpleNamespace(
                    type="response.output_text.delta",
                    delta="Evidence-based answer.",
                )
                yield SimpleNamespace(type="response.completed")

            return events()

    class FakeAsyncOpenAI:
        def __init__(self, *, api_key: str) -> None:
            captured["api_key"] = api_key
            self.responses = FakeResponses()

    monkeypatch.setattr("live_long_rnd.api.llm.AsyncOpenAI", FakeAsyncOpenAI)
    llm = OpenAILLM(api_key="test-key", model="gpt-test")
    assert captured == {}

    async def exercise() -> list[str]:
        chunks = await StubRetriever().retrieve("What do senolytics target?")
        return await _collect(llm.stream_answer("What do senolytics target?", chunks))

    assert asyncio.run(exercise()) == ["Evidence-based answer."]
    assert captured["api_key"] == "test-key"
    assert captured["model"] == "gpt-test"
    assert captured["reasoning"] == {"effort": "high"}
    assert captured["stream"] is True


def test_environment_selects_the_llm_adapter_and_model() -> None:
    assert isinstance(create_llm_client({}), StubLLM)

    client = create_llm_client({"LIVE_LONG_LLM": "openai", "OPENAI_API_KEY": "test-key"})

    assert isinstance(client, OpenAILLM)
    assert client.model == "gpt-5.6-luna"
