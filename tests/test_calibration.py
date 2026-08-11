"""Retrieval calibration reporting tests."""

from collections.abc import Callable, Iterator, Sequence

from live_long_rnd.calibration import (
    ApiCost,
    CachingReranker,
    MeasuredRun,
    VariantMeasurement,
    measure_variant,
    select_best_run,
)
from live_long_rnd.evaluation import (
    CITATION_TARGETS,
    EVALUATION_ITEMS,
    AnswerQualityScores,
    CitationScores,
    GateScores,
    score_answer_quality,
)
from live_long_rnd.retrieve import RetrievalResult


def test_measure_variant_records_gates_pages_tokens_and_latency() -> None:
    items_by_question = {item.question: item for item in EVALUATION_ITEMS}
    ticks = iter(value / 100 for value in range(48))

    def search(question: str) -> list[RetrievalResult]:
        item = items_by_question[question]
        return [_result(document_id, page) for document_id, page in CITATION_TARGETS[item.item_id]]

    measurement = measure_variant(
        name="controlled",
        settings={"candidate_depth": 40},
        search=search,
        clock=_clock(ticks),
    )

    assert measurement.gates.contradiction_pairs == 6
    assert measurement.gates.entity_traps == 6
    assert measurement.gates.gold_documents == 12
    assert measurement.citations.supported_items == 24
    assert measurement.median_latency_ms == 10.0
    assert measurement.p95_latency_ms == 10.0
    assert measurement.mean_packed_tokens > 0
    assert measurement.rankings["C1"] == ["009-paper", "009-paper"]
    assert measurement.page_hits["C1"] == ["009-paper:1", "009-paper:10"]


def test_answer_quality_scores_required_facts_in_packed_evidence() -> None:
    results = {
        "B2": [
            _result(
                "016",
                3,
                text="Intermittent D (100 mg/day) plus Q (1250 mg/day) were given.",
            )
        ],
        "C2": [
            _result(
                "023",
                1,
                text="Resilience disappears at a lifespan limit of 120 to 150 years.",
            )
        ],
    }

    scores = score_answer_quality(results)

    assert scores.item_coverage["B2"] == 0.6667
    assert scores.item_coverage["C2"] == 1.0
    assert scores.item_coverage["A1"] == 0.0
    assert scores.fully_answerable_items == 1
    assert scores.total_facts == 60
    assert scores.total_items == 24


def test_measure_variant_records_answer_quality() -> None:
    ticks = iter(value / 100 for value in range(48))
    evidence_by_question = {
        EVALUATION_ITEMS[7].question: (
            "D 100 mg/day and Q 1250 mg/day were given over three consecutive days "
            "in three consecutive weeks."
        ),
        EVALUATION_ITEMS[13].question: (
            "A complete loss of resilience sets the limit at 120 to 150 years."
        ),
    }

    measurement = measure_variant(
        name="controlled",
        settings={},
        search=lambda question: [
            _result("999", 1, text=evidence_by_question.get(question, "irrelevant"))
        ],
        clock=_clock(ticks),
    )

    assert measurement.answer_quality.fully_answerable_items == 2
    assert measurement.answer_quality.item_coverage["B2"] == 1.0
    assert measurement.answer_quality.item_coverage["A1"] == 0.0


def test_runtime_selection_uses_measured_answer_quality() -> None:
    faster_cheaper = _measured_run(
        "faster-cheaper",
        fully_answerable_items=10,
        covered_facts=40,
        p95_latency_ms=10.0,
        estimated_usd=0.001,
    )
    higher_answer_quality = _measured_run(
        "higher-answer-quality",
        fully_answerable_items=11,
        covered_facts=42,
        p95_latency_ms=100.0,
        estimated_usd=0.01,
    )

    selected = select_best_run((faster_cheaper, higher_answer_quality))

    assert selected.measurement.name == "higher-answer-quality"


def test_runtime_selection_uses_measured_api_cost_as_a_tie_break() -> None:
    expensive = _measured_run(
        "expensive",
        fully_answerable_items=11,
        covered_facts=42,
        p95_latency_ms=100.0,
        estimated_usd=0.01,
    )
    cheaper = _measured_run(
        "cheaper",
        fully_answerable_items=11,
        covered_facts=42,
        p95_latency_ms=100.0,
        estimated_usd=0.001,
    )

    selected = select_best_run((expensive, cheaper))

    assert selected.measurement.name == "cheaper"


def test_caching_reranker_reuses_identical_candidate_rankings() -> None:
    reranker = RecordingReranker()
    caching_reranker = CachingReranker(reranker)
    candidates = [_result("009", 1), _result("007", 2)]

    first = caching_reranker.rerank("Which clock predicts mortality?", candidates)
    second = caching_reranker.rerank("Which clock predicts mortality?", candidates)

    assert first == second
    assert reranker.calls == 1


class RecordingReranker:
    def __init__(self) -> None:
        self.calls = 0

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
    ) -> list[RetrievalResult]:
        del query
        self.calls += 1
        return list(reversed(candidates))


def _clock(ticks: Iterator[float]) -> Callable[[], float]:
    def read() -> float:
        return next(ticks)

    return read


def _result(
    document_id: str,
    page: int,
    *,
    text: str = "Complete evidence passage.",
) -> RetrievalResult:
    return RetrievalResult(
        document_id=f"{document_id}-paper",
        page_numbers=[page],
        bboxes=[{"page": page, "l": 1.0, "t": 2.0, "r": 3.0, "b": 4.0}],
        heading_path=["Evidence"],
        original_text=text,
        score=1.0,
    )


def _measured_run(
    name: str,
    *,
    fully_answerable_items: int,
    covered_facts: int,
    p95_latency_ms: float,
    estimated_usd: float,
) -> MeasuredRun:
    return MeasuredRun(
        measurement=VariantMeasurement(
            name=name,
            settings={},
            gates=GateScores(6, 6, 12, True, True),
            citations=CitationScores(24, 24),
            answer_quality=AnswerQualityScores(
                covered_facts=covered_facts,
                total_facts=60,
                fully_answerable_items=fully_answerable_items,
                total_items=24,
                mean_fact_coverage=covered_facts / 60,
                item_coverage={},
            ),
            median_latency_ms=p95_latency_ms,
            p95_latency_ms=p95_latency_ms,
            mean_packed_tokens=1_000.0,
            rankings={},
            page_hits={},
        ),
        api_cost=ApiCost(
            planner_calls=0,
            planner_input_tokens=0,
            planner_output_tokens=0,
            embedding_input_tokens=0,
            estimated_usd=estimated_usd,
        ),
    )
