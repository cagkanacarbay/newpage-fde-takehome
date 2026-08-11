"""Retrieval calibration reporting tests."""

from collections.abc import Callable, Iterator, Sequence

from live_long_rnd.calibration import CachingReranker, measure_variant
from live_long_rnd.evaluation import CITATION_TARGETS, EVALUATION_ITEMS
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


def _result(document_id: str, page: int) -> RetrievalResult:
    return RetrievalResult(
        document_id=f"{document_id}-paper",
        page_numbers=[page],
        bboxes=[{"page": page, "l": 1.0, "t": 2.0, "r": 3.0, "b": 4.0}],
        heading_path=["Evidence"],
        original_text="Complete evidence passage.",
        score=1.0,
    )
