"""OpenAI embedding adapter tests."""

import json
import re
from typing import Any

import httpx
import pytest
from openai import OpenAI

from live_long_rnd.embeddings import (
    EmbeddingConfigurationError,
    EmbeddingInputTooLongError,
    OpenAIEmbedder,
)


def test_openai_embedder_reports_when_the_api_key_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    embedder = OpenAIEmbedder()

    with pytest.raises(
        EmbeddingConfigurationError,
        match=re.escape(
            "OpenAI embeddings are selected but OPENAI_API_KEY is not set. "
            "Set OPENAI_API_KEY and retry."
        ),
    ):
        embedder.embed(["epigenetic age"])


def test_openai_embedder_batches_large_inputs_and_preserves_vector_order() -> None:
    requests: list[dict[str, Any]] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        inputs = payload["input"]
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "index": index,
                        "embedding": [float(text)],
                    }
                    for index, text in reversed(list(enumerate(inputs)))
                ],
                "model": "text-embedding-3-large",
                "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)},
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handle_request))
    client = OpenAI(api_key="test-key", base_url="https://openai.test/v1", http_client=http_client)
    embedder = OpenAIEmbedder(client=client)

    vectors = embedder.embed([str(index) for index in range(2_049)])

    assert [len(request["input"]) for request in requests] == [2_048, 1]
    assert [vector[0] for vector in vectors] == list(range(2_049))


def test_openai_embedder_rejects_an_input_above_the_model_limit() -> None:
    requests: list[httpx.Request] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    http_client = httpx.Client(transport=httpx.MockTransport(handle_request))
    client = OpenAI(api_key="test-key", base_url="https://openai.test/v1", http_client=http_client)
    embedder = OpenAIEmbedder(client=client)

    with pytest.raises(
        EmbeddingInputTooLongError,
        match=re.escape("Embedding input 1 has 8193 tokens; the maximum is 8192."),
    ):
        embedder.embed([" aging" * 8_193])

    assert requests == []


def test_openai_embedder_rejects_a_later_over_limit_input_before_any_request() -> None:
    requests: list[httpx.Request] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    http_client = httpx.Client(transport=httpx.MockTransport(handle_request))
    client = OpenAI(api_key="test-key", base_url="https://openai.test/v1", http_client=http_client)
    embedder = OpenAIEmbedder(client=client)

    with pytest.raises(
        EmbeddingInputTooLongError,
        match=re.escape("Embedding input 2049 has 8193 tokens; the maximum is 8192."),
    ):
        embedder.embed(["age"] * 2_048 + [" aging" * 8_193])

    assert requests == []
