"""OpenAI text embeddings behind one batch-oriented interface."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from typing import Protocol

import tiktoken
from openai import OpenAI

EMBEDDING_MODEL_NAME = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3_072
EMBEDDING_MAX_TOKENS = 8_192
EMBEDDING_ENCODING_NAME = "cl100k_base"
EMBEDDING_REQUEST_MAX_INPUTS = 2_048
EMBEDDING_REQUEST_MAX_TOKENS = 300_000


class Embedder(Protocol):
    """Embedding seam shared by ingestion and query retrieval."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts and preserve input order."""


class EmbeddingConfigurationError(RuntimeError):
    """Raised when OpenAI embedding configuration is incomplete."""


class EmbeddingInputTooLongError(ValueError):
    """Raised before OpenAI receives an input above its token limit."""


class OpenAIEmbedder:
    """OpenAI embedding adapter shared by ingestion and query retrieval."""

    def __init__(
        self,
        *,
        model_name: str = EMBEDDING_MODEL_NAME,
        dimensions: int = EMBEDDING_DIMENSIONS,
        api_key: str | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self._model_name = model_name
        self._dimensions = dimensions
        self._api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self._client = client
        self._encoding = tiktoken.get_encoding(EMBEDDING_ENCODING_NAME)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self._client is None:
            if not self._api_key:
                raise EmbeddingConfigurationError(
                    "OpenAI embeddings are selected but OPENAI_API_KEY is not set. "
                    "Set OPENAI_API_KEY and retry."
                )
            self._client = OpenAI(api_key=self._api_key)
        vectors: list[list[float]] = []
        for batch in self._batches(texts):
            response = self._client.embeddings.create(
                input=batch,
                model=self._model_name,
                dimensions=self._dimensions,
                encoding_format="float",
            )
            vectors.extend(
                item.embedding for item in sorted(response.data, key=lambda item: item.index)
            )
        return vectors

    def _batches(self, texts: Sequence[str]) -> Iterator[list[str]]:
        batch: list[str] = []
        batch_tokens = 0
        for input_number, text in enumerate(texts, start=1):
            text_tokens = len(self._encoding.encode(text))
            if text_tokens > EMBEDDING_MAX_TOKENS:
                raise EmbeddingInputTooLongError(
                    f"Embedding input {input_number} has {text_tokens} tokens; "
                    f"the maximum is {EMBEDDING_MAX_TOKENS}."
                )
            if batch and (
                len(batch) == EMBEDDING_REQUEST_MAX_INPUTS
                or batch_tokens + text_tokens > EMBEDDING_REQUEST_MAX_TOKENS
            ):
                yield batch
                batch = []
                batch_tokens = 0
            batch.append(text)
            batch_tokens += text_tokens
        if batch:
            yield batch
