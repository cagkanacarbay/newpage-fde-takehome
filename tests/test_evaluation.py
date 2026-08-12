"""Deterministic scoring for the 24-item retrieval quality suite."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from live_long_rnd.evaluation import (
    CITATION_TARGETS,
    EVALUATION_ITEMS,
    AnswerQualityScores,
    CitationScores,
    GateScores,
    score_answer_quality,
    score_citation_support,
    score_quality_gates,
    selected_runtime_quality_passes,
)
from live_long_rnd.golden_embeddings import GoldenFixtureEmbedder
from live_long_rnd.query_planning import OpenAIQueryPlanner, QueryPlan, SearchIntent
from live_long_rnd.retrieve import (
    FlashRankCrossEncoder,
    LanceDBHybridStore,
    RetrievalConfig,
    RetrievalDependencies,
    RetrievalResult,
    retrieve,
)

_FIXTURE_INDEX = Path("tests/fixtures/golden_retrieval_index")


@dataclass
class _PlannerResponse:
    output_parsed: QueryPlan


class _PlannerResponses:
    def parse(self, **kwargs: Any) -> _PlannerResponse:
        message = kwargs["input"][-1]["content"]
        plans = {
            "Does senolytic treatment reverse cellular senescence in humans?": [
                SearchIntent(
                    dense_query="Senolytic treatment effects on cellular senescence in humans",
                    sparse_query="senolytic cellular senescence humans",
                ),
                SearchIntent(
                    dense_query="Senolytic treatment cellular senescence in humans",
                    sparse_query="senolytic treatment cellular senescence humans",
                ),
            ],
            "Is metformin a proven anti-aging drug compared with rapamycin?": [
                SearchIntent(
                    dense_query="Evidence comparing metformin and rapamycin as anti-aging drugs",
                    sparse_query="metformin rapamycin anti-aging",
                ),
                SearchIntent(
                    dense_query="Metformin compared with rapamycin for anti-aging",
                    sparse_query="metformin rapamycin anti-aging",
                ),
            ],
            "What D+Q doses were used for idiopathic pulmonary fibrosis?": [
                SearchIntent(
                    dense_query="Doses for idiopathic pulmonary fibrosis",
                    sparse_query="doses idiopathic pulmonary fibrosis",
                )
            ],
            "Does rapamycin, rather than metformin, mirror dietary restriction?": [
                SearchIntent(
                    dense_query=(
                        "How rapamycin, compared with metformin, reproduces dietary "
                        "restriction effects"
                    ),
                    sparse_query="rapamycin dietary restriction",
                )
            ],
        }
        return _PlannerResponse(
            QueryPlan(
                action="retrieve",
                search_intents=plans.get(
                    message,
                    [
                        SearchIntent(
                            dense_query=f"Evidence needed to answer: {message}",
                            sparse_query=message,
                        )
                    ],
                ),
            )
        )


class _PlannerClient:
    responses = _PlannerResponses()


def test_quality_gates_pass_at_the_decided_threshold_edge() -> None:
    rankings = {
        item.item_id: [*item.gold_document_ids, *item.distractor_document_ids]
        for item in EVALUATION_ITEMS
    }
    rankings["C12"] = ["999"]

    assert score_quality_gates(rankings) == GateScores(
        contradiction_pairs=6,
        entity_traps=6,
        gold_documents=11,
        c1_passed=True,
        passed=True,
    )


def test_quality_gates_fail_when_mandatory_c1_misses() -> None:
    rankings = {
        item.item_id: [*item.gold_document_ids, *item.distractor_document_ids]
        for item in EVALUATION_ITEMS
    }
    rankings["C1"] = ["999"]

    scores = score_quality_gates(rankings)

    assert scores.gold_documents == 11
    assert scores.c1_passed is False
    assert scores.passed is False


def test_quality_gates_take_the_top_ten_unique_document_ids() -> None:
    rankings = {
        item.item_id: [*item.gold_document_ids, *item.distractor_document_ids]
        for item in EVALUATION_ITEMS
    }
    rankings["A2"] = ["017"] * 10 + ["015"]

    scores = score_quality_gates(rankings)

    assert scores.contradiction_pairs == 6


def test_citation_support_requires_every_target_page_for_each_item() -> None:
    results = {
        item_id: [
            _retrieval_result(document_id=document_id, pages=[page])
            for document_id, page in targets
        ]
        for item_id, targets in CITATION_TARGETS.items()
    }
    results["C1"] = [
        _retrieval_result(
            document_id="009-paper",
            pages=[1, 10],
            citation_page=1,
        )
    ]

    scores = score_citation_support(results)

    assert scores == CitationScores(supported_items=23, total_items=24)


def test_selected_runtime_quality_requires_citations_and_answer_evidence() -> None:
    ranking_scores = GateScores(6, 6, 12, True, True)
    answer_scores = AnswerQualityScores(49, 60, 18, 24, 49 / 60, {})

    assert selected_runtime_quality_passes(
        ranking_scores,
        CitationScores(13, 24),
        answer_scores,
    )
    assert not selected_runtime_quality_passes(
        ranking_scores,
        CitationScores(12, 24),
        answer_scores,
    )
    assert not selected_runtime_quality_passes(
        ranking_scores,
        CitationScores(13, 24),
        AnswerQualityScores(48, 60, 18, 24, 48 / 60, {}),
    )
    assert not selected_runtime_quality_passes(
        ranking_scores,
        CitationScores(13, 24),
        AnswerQualityScores(49, 60, 17, 24, 49 / 60, {}),
    )
    assert not selected_runtime_quality_passes(
        GateScores(6, 6, 12, True, False),
        CitationScores(13, 24),
        answer_scores,
    )


@pytest.mark.e2e
def test_retrieval_quality_gates_block_regressions(tmp_path: Path) -> None:
    assert _FIXTURE_INDEX.is_dir()
    _assert_fixture_matches_corpus()
    store = LanceDBHybridStore(_FIXTURE_INDEX)
    rankings: dict[str, list[str]] = {}
    results_by_item: dict[str, list[RetrievalResult]] = {}

    for item in EVALUATION_ITEMS:
        results = retrieve(
            item.question,
            config=RetrievalConfig(),
            dependencies=RetrievalDependencies(
                store=store,
                embedder=GoldenFixtureEmbedder(),
                planner=OpenAIQueryPlanner(client=_PlannerClient()),
                reranker=FlashRankCrossEncoder(cache_dir=tmp_path / "flashrank"),
            ),
        )
        results_by_item[item.item_id] = results
        rankings[item.item_id] = [result.document_id for result in results]

    scores = score_quality_gates(rankings)
    citation_scores = score_citation_support(results_by_item)
    answer_scores = score_answer_quality(results_by_item)

    ranking_report = json.dumps(rankings, indent=2)
    assert scores.contradiction_pairs == 6, ranking_report
    assert scores.entity_traps == 6, ranking_report
    assert scores.gold_documents >= 11, rankings
    assert scores.c1_passed, rankings
    assert scores.passed, rankings
    assert selected_runtime_quality_passes(scores, citation_scores, answer_scores), (
        citation_scores,
        answer_scores,
    )


def _assert_fixture_matches_corpus() -> None:
    manifest = json.loads((_FIXTURE_INDEX / "manifest.json").read_text(encoding="utf-8"))
    corpus_hashes = {
        source.name: hashlib.sha256(source.read_bytes()).hexdigest()
        for source in sorted(Path("data/corpus/longevity").glob("*.pdf"))
    }
    assert manifest["source_sha256"] == corpus_hashes


def _retrieval_result(
    *,
    document_id: str,
    pages: list[int],
    citation_page: int | None = None,
) -> RetrievalResult:
    emitted_page = citation_page if citation_page is not None else pages[0]
    return RetrievalResult(
        document_id=document_id,
        page_numbers=pages,
        bboxes=[{"page": emitted_page, "l": 1.0, "t": 2.0, "r": 3.0, "b": 4.0}],
        heading_path=["Evaluation evidence"],
        original_text="Evidence",
        score=1.0,
    )
