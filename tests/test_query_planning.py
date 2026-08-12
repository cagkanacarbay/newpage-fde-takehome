"""Query planning tests through the public planner interface."""

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from live_long_rnd.query_planning import (
    GeminiQueryPlanner,
    MetadataFilters,
    PlannerUsage,
    QueryPlan,
    QueryPlanningError,
    RawQueryPlanner,
    SearchIntent,
)


def test_direct_question_with_history_produces_one_search_intent() -> None:
    captured: dict[str, Any] = {}
    expected = QueryPlan(
        action="retrieve",
        search_intents=[
            SearchIntent(
                dense_query=(
                    "What doses and adverse events did Hickson report for dasatinib and quercetin?"
                ),
                sparse_query="Hickson dasatinib quercetin doses adverse events",
                filters=MetadataFilters(author="Hickson"),
            )
        ],
    )

    class FakeCompletions:
        def parse(self, **kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(parsed=expected))]
            )

    client = SimpleNamespace(
        beta=SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    )
    planner = GeminiQueryPlanner(client=client, model="planner-model")

    plan = planner.plan(
        "What doses and adverse events did it report?",
        history=[
            {
                "role": "user",
                "content": "Tell me about Hickson's dasatinib and quercetin trial.",
            }
        ],
    )

    assert plan == expected
    assert captured["model"] == "planner-model"
    assert captured["response_format"] is QueryPlan
    assert captured["reasoning_effort"] == "minimal"
    assert captured["messages"][-1] == {
        "role": "user",
        "content": "What doses and adverse events did it report?",
    }


def test_clarify_plan_rejects_search_intents() -> None:
    with pytest.raises(ValidationError, match="clarify plan must not contain search intents"):
        QueryPlan(
            action="clarify",
            search_intents=[
                SearchIntent(
                    dense_query="Search pasted material",
                    sparse_query="pasted material",
                )
            ],
        )


def test_retrieve_plan_requires_at_least_one_search_intent() -> None:
    with pytest.raises(ValidationError, match="retrieve plan requires at least one search intent"):
        QueryPlan(action="retrieve", search_intents=[])


def test_dual_query_plan_rejects_identical_query_forms() -> None:
    with pytest.raises(
        ValidationError, match="dual-query search intents require distinct query forms"
    ):
        SearchIntent(dense_query="What dose was used?", sparse_query="What dose was used?")


def test_dual_query_plan_rejects_case_and_whitespace_only_differences() -> None:
    with pytest.raises(
        ValidationError, match="dual-query search intents require distinct query forms"
    ):
        SearchIntent(
            dense_query="  Rapamycin dietary   restriction ",
            sparse_query="rapamycin dietary restriction",
        )


def test_search_intent_rejects_whitespace_only_query_forms() -> None:
    with pytest.raises(ValidationError, match="query forms must contain non-whitespace content"):
        SearchIntent(dense_query="metformin", sparse_query=" \t ")


def test_single_intent_planner_rejects_model_decomposition() -> None:
    decomposed = QueryPlan(
        action="retrieve",
        search_intents=[
            SearchIntent(dense_query="benefits", sparse_query="lifespan"),
            SearchIntent(dense_query="risks", sparse_query="toxicity"),
        ],
    )

    class FakeCompletions:
        def parse(self, **kwargs: Any) -> SimpleNamespace:
            del kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(parsed=decomposed))]
            )

    planner = GeminiQueryPlanner(
        client=SimpleNamespace(
            beta=SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        ),
        max_intents=1,
    )

    with pytest.raises(QueryPlanningError, match="returned 2 search intents; maximum is 1"):
        planner.plan("Compare benefits and risks")


def test_raw_planner_preserves_the_message_for_both_query_forms() -> None:
    planner = RawQueryPlanner()

    plan = planner.plan("What dose was 100 mg/day?", history=[])

    assert plan == QueryPlan(
        action="retrieve",
        search_intents=[
            SearchIntent(
                dense_query="What dose was 100 mg/day?",
                sparse_query="What dose was 100 mg/day?",
                variant="raw-only",
            )
        ],
    )


def test_gemini_planner_rejects_raw_only_search_intents() -> None:
    raw_only = QueryPlan(
        action="retrieve",
        search_intents=[
            SearchIntent(
                dense_query="What dose was used?",
                sparse_query="What dose was used?",
                variant="raw-only",
            )
        ],
    )

    class FakeCompletions:
        def parse(self, **kwargs: Any) -> SimpleNamespace:
            del kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(parsed=raw_only))]
            )

    planner = GeminiQueryPlanner(
        client=SimpleNamespace(
            beta=SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        )
    )

    with pytest.raises(QueryPlanningError, match="returned a raw-only search intent"):
        planner.plan("What dose was used?")


def test_gemini_planner_accumulates_reported_token_usage() -> None:
    expected = QueryPlan(
        action="retrieve",
        search_intents=[SearchIntent(dense_query="resolved", sparse_query="literal")],
    )

    class FakeCompletions:
        def parse(self, **kwargs: Any) -> SimpleNamespace:
            del kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(parsed=expected))],
                usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30),
            )

    planner = GeminiQueryPlanner(
        client=SimpleNamespace(
            beta=SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        )
    )

    planner.plan("first")
    planner.plan("second")

    assert planner.usage == PlannerUsage(calls=2, input_tokens=240, output_tokens=60)


def test_gemini_planner_requires_a_gemini_key() -> None:
    planner = GeminiQueryPlanner(api_key=None)

    with pytest.raises(QueryPlanningError, match="GEMINI_API_KEY is not set"):
        planner.plan("What did the trial find?")
