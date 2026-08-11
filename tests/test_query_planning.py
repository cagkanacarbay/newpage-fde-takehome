"""Query planning tests through the public planner interface."""

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from live_long_rnd.query_planning import (
    MetadataFilters,
    OpenAIQueryPlanner,
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

    class FakeResponses:
        def parse(self, **kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(output_parsed=expected)

    client = SimpleNamespace(responses=FakeResponses())
    planner = OpenAIQueryPlanner(client=client, model="planner-model")

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
    assert captured["text_format"] is QueryPlan
    assert captured["store"] is False
    assert captured["input"][-1] == {
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


def test_single_intent_planner_rejects_model_decomposition() -> None:
    decomposed = QueryPlan(
        action="retrieve",
        search_intents=[
            SearchIntent(dense_query="benefits", sparse_query="lifespan"),
            SearchIntent(dense_query="risks", sparse_query="toxicity"),
        ],
    )

    class FakeResponses:
        def parse(self, **kwargs: Any) -> SimpleNamespace:
            del kwargs
            return SimpleNamespace(output_parsed=decomposed)

    planner = OpenAIQueryPlanner(
        client=SimpleNamespace(responses=FakeResponses()),
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
            )
        ],
    )


def test_openai_planner_accumulates_reported_token_usage() -> None:
    expected = QueryPlan(
        action="retrieve",
        search_intents=[SearchIntent(dense_query="resolved", sparse_query="literal")],
    )

    class FakeResponses:
        def parse(self, **kwargs: Any) -> SimpleNamespace:
            del kwargs
            return SimpleNamespace(
                output_parsed=expected,
                usage=SimpleNamespace(input_tokens=120, output_tokens=30),
            )

    planner = OpenAIQueryPlanner(client=SimpleNamespace(responses=FakeResponses()))

    planner.plan("first")
    planner.plan("second")

    assert planner.usage == PlannerUsage(calls=2, input_tokens=240, output_tokens=60)
