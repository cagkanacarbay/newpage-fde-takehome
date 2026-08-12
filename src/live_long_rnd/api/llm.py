import os
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Protocol

from openai import AsyncOpenAI
from openai.types.responses.response_input_param import ResponseInputParam

from live_long_rnd.api.conversations import StoredMessage
from live_long_rnd.api.retrieval import RetrievedChunk


class LLMClient(Protocol):
    def stream_answer(
        self,
        message: str,
        chunks: Sequence[RetrievedChunk],
        history: Sequence[StoredMessage],
    ) -> AsyncIterator[str]: ...


class LLMConfigurationError(RuntimeError):
    """Raised when the selected LLM adapter lacks required configuration."""


class StubLLM:
    def stream_answer(
        self,
        message: str,
        chunks: Sequence[RetrievedChunk],
        history: Sequence[StoredMessage],
    ) -> AsyncIterator[str]:
        del message, chunks, history
        return _stub_tokens()


class OpenAILLM:
    def __init__(self, *, api_key: str | None, model: str) -> None:
        self.api_key = api_key
        self.model = model

    async def stream_answer(
        self,
        message: str,
        chunks: Sequence[RetrievedChunk],
        history: Sequence[StoredMessage],
    ) -> AsyncIterator[str]:
        if not self.api_key:
            raise LLMConfigurationError(
                "OpenAI is selected but OPENAI_API_KEY is not set. "
                "Set OPENAI_API_KEY and restart the server."
            )

        sources = "\n\n".join(
            f"Source {index}: {chunk.text}" for index, chunk in enumerate(chunks, start=1)
        )
        input_messages: ResponseInputParam = [
            {"role": item.role, "content": item.text} for item in history
        ]
        input_messages.append(
            {
                "role": "user",
                "content": f"Question: {message}\n\nSources:\n{sources}",
            }
        )
        client = AsyncOpenAI(api_key=self.api_key)
        stream = await client.responses.create(
            model=self.model,
            instructions=(
                "Answer the researcher's question using only the supplied sources. "
                "State uncertainty plainly and do not invent findings."
            ),
            input=input_messages,
            reasoning={"effort": "high"},
            stream=True,
        )
        async for event in stream:
            if event.type == "response.output_text.delta":
                yield event.delta


def create_llm_client(environ: Mapping[str, str] | None = None) -> LLMClient:
    settings = os.environ if environ is None else environ
    adapter = settings.get("LIVE_LONG_LLM", "stub").strip().lower()
    if adapter == "stub":
        return StubLLM()
    if adapter == "openai":
        model = settings.get("LIVE_LONG_MODEL", "gpt-5.6-luna")
        return OpenAILLM(
            api_key=settings.get("OPENAI_API_KEY"),
            model=model,
        )
    raise LLMConfigurationError(
        f"Unsupported LIVE_LONG_LLM value {adapter!r}. Use 'stub' or 'openai'."
    )


async def _stub_tokens() -> AsyncIterator[str]:
    tokens = (
        "Senolytics are drugs designed to selectively target senescent cells. ",
        "Early human studies suggest possible physical-function benefits, while ",
        "newer evidence shows they may not reverse established DNA methylation ",
        "signatures of senescence.",
    )
    for token in tokens:
        yield token
