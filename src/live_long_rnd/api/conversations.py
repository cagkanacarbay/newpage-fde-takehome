import json
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

import tiktoken

DEFAULT_DATABASE_PATH = Path("data/conversations.db")
TITLE_MAX_LENGTH = 60
VISIBLE_TURNS = 30
DEFAULT_HISTORY_TOKEN_BUDGET = 100_000
HISTORY_NOTICE = "Earlier messages were dropped to fit the context window"
MessageRole = Literal["user", "assistant", "system"]


@dataclass(frozen=True)
class ConversationSummary:
    id: str
    title: str
    updated_at: str


@dataclass(frozen=True)
class StoredMessage:
    role: MessageRole
    text: str
    citations: list[object]
    created_at: str


@dataclass(frozen=True)
class Conversation:
    id: str
    title: str
    messages: list[StoredMessage]


@dataclass(frozen=True)
class HistoryWindow:
    messages: list[StoredMessage]
    excluded_turns: int


class ConversationStore:
    def __init__(
        self,
        database_path: Path,
        *,
        history_token_budget: int = DEFAULT_HISTORY_TOKEN_BUDGET,
    ) -> None:
        self._database_path = database_path
        self._history_token_budget = history_token_budget
        self._encoding = tiktoken.get_encoding("o200k_base")
        database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    text TEXT NOT NULL,
                    citations_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                        ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS messages_by_conversation
                ON messages (conversation_id, id)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS conversations_by_activity
                ON conversations (updated_at DESC, id DESC)
                """
            )

    def create_from_message(self, message: str) -> ConversationSummary:
        return self._create(_title_from_message(message))

    def create_empty(self) -> ConversationSummary:
        return self._create("New conversation")

    def begin_turn(
        self,
        conversation_id: str | None,
        message: str,
    ) -> ConversationSummary | None:
        if conversation_id is None:
            conversation = self.create_empty()
            return ConversationSummary(
                id=conversation.id,
                title=_title_from_message(message),
                updated_at=conversation.updated_at,
            )
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT c.title, c.updated_at, EXISTS (
                    SELECT 1 FROM messages m WHERE m.conversation_id = c.id
                )
                FROM conversations c
                WHERE c.id = ?
                """,
                (conversation_id,),
            ).fetchone()
        if row is None:
            return None
        title = str(row[0]) if bool(row[2]) else _title_from_message(message)
        return ConversationSummary(
            id=conversation_id,
            title=title,
            updated_at=str(row[1]),
        )

    def _create(self, title: str) -> ConversationSummary:
        summary = ConversationSummary(
            id=str(uuid4()),
            title=title,
            updated_at=_now(),
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations (id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (summary.id, summary.title, summary.updated_at, summary.updated_at),
            )
        return summary

    def save_turn(
        self,
        conversation_id: str,
        user_text: str,
        assistant_text: str,
        citations: Sequence[Mapping[str, object]],
        conversation_title: str,
    ) -> None:
        created_at = _now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE conversations
                SET title = CASE
                    WHEN NOT EXISTS (
                        SELECT 1 FROM messages WHERE conversation_id = ?
                    ) THEN ?
                    ELSE title
                END,
                updated_at = ?
                WHERE id = ?
                """,
                (conversation_id, conversation_title, created_at, conversation_id),
            )
            connection.executemany(
                """
                INSERT INTO messages (
                    conversation_id, role, text, citations_json, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (conversation_id, "user", user_text, "[]", created_at),
                    (
                        conversation_id,
                        "assistant",
                        assistant_text,
                        json.dumps(citations, ensure_ascii=False),
                        created_at,
                    ),
                ),
            )

    def get(self, conversation_id: str) -> Conversation | None:
        with self._connect() as connection:
            summary = connection.execute(
                "SELECT id, title FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if summary is None:
                return None
            rows = connection.execute(
                """
                SELECT role, text, citations_json, created_at
                FROM (
                    SELECT id, role, text, citations_json, created_at
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id
                """,
                (conversation_id, VISIBLE_TURNS * 2),
            ).fetchall()
        messages = [_stored_message(row) for row in rows]
        window = self.history_window(conversation_id)
        if window.excluded_turns > 0:
            marker_index = max(0, len(messages) - len(window.messages))
            marker_time = (
                messages[marker_index].created_at if marker_index < len(messages) else _now()
            )
            messages.insert(
                marker_index,
                StoredMessage(
                    role="system",
                    text=HISTORY_NOTICE,
                    citations=[],
                    created_at=marker_time,
                ),
            )
        return Conversation(
            id=str(summary[0]),
            title=str(summary[1]),
            messages=messages,
        )

    def history_window(self, conversation_id: str) -> HistoryWindow:
        with self._connect() as connection:
            total_messages = int(
                connection.execute(
                    "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()[0]
            )
            rows = iter(
                connection.execute(
                    """
                    SELECT role, text, citations_json, created_at
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY id DESC
                    """,
                    (conversation_id,),
                )
            )
            newest_first_turns: list[list[StoredMessage]] = []
            used_tokens = 0
            while True:
                descending_turn = list(islice(rows, 2))
                if len(descending_turn) < 2:
                    break
                turn = [_stored_message(row) for row in reversed(descending_turn)]
                turn_tokens = sum(len(self._encoding.encode(item.text)) for item in turn)
                if used_tokens + turn_tokens > self._history_token_budget:
                    break
                newest_first_turns.append(turn)
                used_tokens += turn_tokens
        messages = [message for turn in reversed(newest_first_turns) for message in turn]
        return HistoryWindow(
            messages=messages,
            excluded_turns=(total_messages - len(messages)) // 2,
        )

    def list(self) -> list[ConversationSummary]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, title, updated_at
                FROM conversations
                ORDER BY updated_at DESC, id DESC
                """
            ).fetchall()
        return [
            ConversationSummary(
                id=str(row[0]),
                title=str(row[1]),
                updated_at=str(row[2]),
            )
            for row in rows
        ]

    def delete(self, conversation_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,),
            )
        return cursor.rowcount > 0

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=5.0)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection


def _title_from_message(message: str) -> str:
    collapsed = re.sub(r"\s+", " ", message).strip()
    if len(collapsed) <= TITLE_MAX_LENGTH:
        return collapsed
    word_boundary = collapsed.rfind(" ", 0, TITLE_MAX_LENGTH + 1)
    end = word_boundary if word_boundary > 0 else TITLE_MAX_LENGTH
    return f"{collapsed[:end]}…"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _stored_message(row: Sequence[object]) -> StoredMessage:
    return StoredMessage(
        role=cast(MessageRole, str(row[0])),
        text=str(row[1]),
        citations=json.loads(str(row[2])),
        created_at=str(row[3]),
    )
