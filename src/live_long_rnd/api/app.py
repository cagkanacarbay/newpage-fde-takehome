import os
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from live_long_rnd.api.documents import DEFAULT_CORPUS_DIR, DocumentStore
from live_long_rnd.api.llm import LLMClient, create_llm_client
from live_long_rnd.api.retrieval import Retriever, create_retriever
from live_long_rnd.api.sse import encode_sse


class ChatRequest(BaseModel):
    message: str


def create_app(
    *,
    retriever: Retriever | None = None,
    llm_client: LLMClient | None = None,
    web_dir: Path | None = None,
    corpus_dir: Path | None = None,
) -> FastAPI:
    selected_retriever = create_retriever() if retriever is None else retriever
    selected_llm = create_llm_client() if llm_client is None else llm_client
    documents = DocumentStore(corpus_dir or DEFAULT_CORPUS_DIR)
    application = FastAPI(title="Live Long R&D assistant")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["POST"],
        allow_headers=["*"],
    )

    @application.post("/api/chat")
    async def chat(request: ChatRequest) -> StreamingResponse:
        stream = _stream_chat(request.message, selected_retriever, selected_llm)
        return StreamingResponse(
            stream,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @application.get("/api/documents/{document_id}")
    async def document_metadata(document_id: str) -> dict[str, str]:
        title = documents.title_for(document_id)
        if title is None:
            raise HTTPException(status_code=404, detail="Unknown document.")
        return {"document_id": document_id, "title": title}

    @application.get("/api/documents/{document_id}/pdf")
    async def document_pdf(document_id: str) -> FileResponse:
        pdf_path = documents.pdf_path_for(document_id)
        if pdf_path is None:
            raise HTTPException(status_code=404, detail="Unknown document.")
        return FileResponse(pdf_path, media_type="application/pdf")

    if web_dir is not None:
        application.mount("/", StaticFiles(directory=web_dir, html=True), name="web")

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


configured_web_dir = os.environ.get("LIVE_LONG_WEB_DIR")
configured_corpus_dir = os.environ.get("LIVE_LONG_CORPUS_DIR")
app = create_app(
    web_dir=Path(configured_web_dir) if configured_web_dir else None,
    corpus_dir=Path(configured_corpus_dir) if configured_corpus_dir else None,
)
