"""Strict query plans for semantic and lexical retrieval."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Any, Literal, Protocol, TypedDict, cast

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, model_validator

from live_long_rnd.providers import DEFAULT_GEMINI_MODEL, gemini_base_url

QUERY_PLANNER_MODEL = DEFAULT_GEMINI_MODEL


class ConversationMessage(TypedDict):
    """One saved conversation message consumed by query planning."""

    role: Literal["user", "assistant"]
    content: str


class MetadataFilters(BaseModel):
    """Optional filters that are safe only when the message names them."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str | None = None
    author: str | None = None


class SearchIntent(BaseModel):
    """One independent evidence need with separate dense and sparse queries."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dense_query: str = Field(min_length=1)
    sparse_query: str = Field(min_length=1)
    variant: Literal["dual-query", "raw-only"] = "dual-query"
    filters: MetadataFilters = Field(default_factory=MetadataFilters)

    @model_validator(mode="after")
    def validate_query_forms(self) -> SearchIntent:
        """Reserve identical query forms for the controlled raw-only variant."""
        dense_form = " ".join(self.dense_query.casefold().split())
        sparse_form = " ".join(self.sparse_query.casefold().split())
        if not dense_form or not sparse_form:
            raise ValueError("search intent query forms must contain non-whitespace content")
        if self.variant == "dual-query" and dense_form == sparse_form:
            raise ValueError("dual-query search intents require distinct query forms")
        return self


class QueryPlan(BaseModel):
    """Strict retrieval decision returned by the query planner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["retrieve", "clarify"]
    search_intents: list[SearchIntent] = Field(min_length=0, max_length=3)

    @model_validator(mode="after")
    def validate_action(self) -> QueryPlan:
        """Keep action and intent count consistent."""
        if self.action == "clarify" and self.search_intents:
            raise ValueError("clarify plan must not contain search intents")
        if self.action == "retrieve" and not self.search_intents:
            raise ValueError("retrieve plan requires at least one search intent")
        return self


class _SingleIntentQueryPlan(QueryPlan):
    """Structured-output schema used by the controlled dual-query variant."""

    search_intents: list[SearchIntent] = Field(min_length=0, max_length=1)


class GeminiClient(Protocol):
    """OpenAI-compatible Gemini client subset used by query planning."""

    beta: Any


class QueryPlanner(Protocol):
    """Planner seam shared by live retrieval and deterministic evaluation."""

    def plan(
        self,
        message: str,
        history: Sequence[ConversationMessage] = (),
    ) -> QueryPlan: ...


class QueryPlanningError(RuntimeError):
    """Raised when query planning cannot produce a valid plan."""


@dataclass(frozen=True)
class PlannerUsage:
    """Cumulative token usage reported by the query-planning provider."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class RawQueryPlanner:
    """Use one unchanged message for controlled raw-query evaluation."""

    def plan(
        self,
        message: str,
        history: Sequence[ConversationMessage] = (),
    ) -> QueryPlan:
        del history
        return QueryPlan(
            action="retrieve",
            search_intents=[
                SearchIntent(
                    dense_query=message,
                    sparse_query=message,
                    variant="raw-only",
                )
            ],
        )


class GeminiQueryPlanner:
    """Create a strict QueryPlan with one Gemini Flash call."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = QUERY_PLANNER_MODEL,
        base_url: str | None = None,
        client: GeminiClient | None = None,
        max_intents: int = 3,
    ) -> None:
        if not 1 <= max_intents <= 3:
            raise ValueError("max_intents must be between 1 and 3")
        self._api_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
        self._model = model
        self._base_url = base_url or gemini_base_url()
        self._client = client
        self._max_intents = max_intents
        self._usage = PlannerUsage()
        self._usage_lock = Lock()

    @property
    def usage(self) -> PlannerUsage:
        """Return cumulative provider-reported planning usage."""
        return self._usage

    def plan(
        self,
        message: str,
        history: Sequence[ConversationMessage] = (),
    ) -> QueryPlan:
        """Resolve one message and saved history into zero to three evidence needs."""
        client = self._client
        if client is None:
            if not self._api_key:
                raise QueryPlanningError(
                    "Gemini query planning is selected but GEMINI_API_KEY is not set. "
                    "Set GEMINI_API_KEY and retry."
                )
            client = cast(
                GeminiClient,
                OpenAI(api_key=self._api_key, base_url=self._base_url),
            )
            self._client = client

        response = client.beta.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": _planner_instructions(self._max_intents)},
                *history,
                {"role": "user", "content": message},
            ],
            response_format=_SingleIntentQueryPlan if self._max_intents == 1 else QueryPlan,
            reasoning_effort="minimal",
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise QueryPlanningError("The query planner returned no structured output.")
        if any(intent.variant == "raw-only" for intent in parsed.search_intents):
            raise QueryPlanningError("The query planner returned a raw-only search intent.")
        if len(parsed.search_intents) > self._max_intents:
            raise QueryPlanningError(
                f"The query planner returned {len(parsed.search_intents)} "
                f"search intents; maximum is {self._max_intents}."
            )
        usage = getattr(response, "usage", None)
        with self._usage_lock:
            self._usage = PlannerUsage(
                calls=self._usage.calls + 1,
                input_tokens=self._usage.input_tokens + int(getattr(usage, "prompt_tokens", 0)),
                output_tokens=self._usage.output_tokens
                + int(getattr(usage, "completion_tokens", 0)),
            )
        return cast(QueryPlan, parsed)


def _planner_instructions(max_intents: int) -> str:
    return f"""You plan retrieval for a longevity-research corpus.
Return action clarify when the current message has no actionable request.
Otherwise return action retrieve and one SearchIntent by default.
Return at most {max_intents} SearchIntent objects.
Use more than one intent only for a genuine comparison or independent requested aspects.
Each dense_query must state the complete resolved evidence need in natural language.
Each sparse_query must preserve exact paper IDs, authors, drug names, doses, years,
numbers, and quoted phrases from the current message or saved conversation history.
Resolve references from history, but never invent facts, entities, filters, or synonyms.
Set metadata filters only when the user names the document ID or author with high confidence.
"""
