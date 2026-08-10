from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from live_long_rnd.api.llm import LLMClient, create_llm_client
from live_long_rnd.api.retrieval import Retriever, StubRetriever
from live_long_rnd.api.sse import encode_sse


class ChatRequest(BaseModel):
    message: str


def create_app(
    *,
    retriever: Retriever | None = None,
    llm_client: LLMClient | None = None,
) -> FastAPI:
    selected_retriever = StubRetriever() if retriever is None else retriever
    selected_llm = create_llm_client() if llm_client is None else llm_client
    application = FastAPI(title="Live Long R&D assistant")

    @application.post("/api/chat")
    async def chat(request: ChatRequest) -> StreamingResponse:
        stream = _stream_chat(request.message, selected_retriever, selected_llm)
        return StreamingResponse(
            stream,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    return application


async def _stream_chat(
    message: str,
    retriever: Retriever,
    llm_client: LLMClient,
) -> AsyncIterator[str]:
    try:
        chunks = await retriever.retrieve(message)
        async for token in llm_client.stream_answer(message, chunks):
            yield encode_sse({"type": "token", "text": token})
        yield encode_sse(
            {
                "type": "citations",
                "citations": [chunk.citation for chunk in chunks],
            }
        )
        yield encode_sse({"type": "done"})
    except Exception as error:
        message = str(error) or "The chat request failed."
        yield encode_sse({"type": "error", "message": message})


app = create_app()
