import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from live_long_rnd.api.conversations import (
    DEFAULT_DATABASE_PATH,
    DEFAULT_HISTORY_TOKEN_BUDGET,
    HISTORY_NOTICE,
    ConversationStore,
    ConversationSummary,
)
from live_long_rnd.api.documents import DEFAULT_CORPUS_DIR, DocumentStore
from live_long_rnd.api.llm import LLMClient, create_llm_client
from live_long_rnd.api.retrieval import Retriever, create_retriever
from live_long_rnd.api.sse import encode_sse

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str


@dataclass(frozen=True)
class ApplicationConfig:
    web_dir: Path | None = None
    corpus_dir: Path | None = None
    database_path: Path | None = None
    history_token_budget: int = DEFAULT_HISTORY_TOKEN_BUDGET


@dataclass(frozen=True)
class _ChatRuntime:
    retriever: Retriever
    llm_client: LLMClient
    conversations: ConversationStore


def create_app(
    *,
    retriever: Retriever | None = None,
    llm_client: LLMClient | None = None,
    config: ApplicationConfig | None = None,
) -> FastAPI:
    settings = config or ApplicationConfig()
    selected_retriever = create_retriever() if retriever is None else retriever
    selected_llm = create_llm_client() if llm_client is None else llm_client
    selected_database_path = settings.database_path or Path(
        os.environ.get("LIVE_LONG_CONVERSATIONS_DB", DEFAULT_DATABASE_PATH)
    )
    documents = DocumentStore(settings.corpus_dir or DEFAULT_CORPUS_DIR)
    conversations = ConversationStore(
        selected_database_path,
        history_token_budget=settings.history_token_budget,
    )
    chat_runtime = _ChatRuntime(selected_retriever, selected_llm, conversations)
    application = FastAPI(title="Live Long R&D assistant")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    @application.post("/api/chat")
    async def chat(request: ChatRequest) -> StreamingResponse:
        conversation = conversations.begin_turn(request.conversation_id, request.message)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Unknown conversation.")
        stream = _stream_chat(
            request.message,
            conversation,
            chat_runtime,
        )
        return StreamingResponse(
            stream,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @application.post("/api/conversations")
    async def create_conversation() -> dict[str, str]:
        conversation = conversations.create_empty()
        return {"id": conversation.id, "title": conversation.title}

    @application.get("/api/conversations")
    async def list_conversations() -> list[dict[str, str]]:
        return [
            {
                "id": conversation.id,
                "title": conversation.title,
                "updated_at": conversation.updated_at,
            }
            for conversation in conversations.list()
        ]

    @application.get("/api/conversations/{conversation_id}")
    async def get_conversation(conversation_id: str) -> dict[str, object]:
        conversation = conversations.get(conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Unknown conversation.")
        return {
            "id": conversation.id,
            "title": conversation.title,
            "messages": [
                {
                    "role": message.role,
                    "text": message.text,
                    "citations": message.citations,
                    "created_at": message.created_at,
                }
                for message in conversation.messages
            ],
        }

    @application.delete(
        "/api/conversations/{conversation_id}",
        status_code=204,
    )
    async def delete_conversation(conversation_id: str) -> Response:
        if not conversations.delete(conversation_id):
            raise HTTPException(status_code=404, detail="Unknown conversation.")
        return Response(status_code=204)

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

    if settings.web_dir is not None:
        application.mount(
            "/",
            StaticFiles(directory=settings.web_dir, html=True),
            name="web",
        )

    return application


async def _stream_chat(
    message: str,
    conversation: ConversationSummary,
    runtime: _ChatRuntime,
) -> AsyncIterator[str]:
    try:
        yield encode_sse(
            {
                "type": "conversation",
                "id": conversation.id,
                "title": conversation.title,
            }
        )
        chunks = await runtime.retriever.retrieve(message)
        history = runtime.conversations.history_window(conversation.id).messages
        answer_parts: list[str] = []
        async for token in runtime.llm_client.stream_answer(message, chunks, history):
            answer_parts.append(token)
            yield encode_sse({"type": "token", "text": token})
        citations = [chunk.citation for chunk in chunks]
        runtime.conversations.save_turn(
            conversation.id,
            message,
            "".join(answer_parts),
            citations,
            conversation.title,
        )
        if runtime.conversations.history_window(conversation.id).excluded_turns > 0:
            yield encode_sse(
                {
                    "type": "history_notice",
                    "text": HISTORY_NOTICE,
                }
            )
        yield encode_sse(
            {
                "type": "citations",
                "citations": citations,
            }
        )
        yield encode_sse({"type": "done"})
    except Exception as error:
        logger.exception(
            "Chat turn failed.",
            extra={"conversation_id": conversation.id},
        )
        message = str(error) or "The chat request failed."
        yield encode_sse({"type": "error", "message": message})


configured_web_dir = os.environ.get("LIVE_LONG_WEB_DIR")
configured_corpus_dir = os.environ.get("LIVE_LONG_CORPUS_DIR")
app = create_app(
    config=ApplicationConfig(
        web_dir=Path(configured_web_dir) if configured_web_dir else None,
        corpus_dir=Path(configured_corpus_dir) if configured_corpus_dir else None,
    )
)
