"""Measured evaluation for retrieval variants and runtime settings."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import tiktoken
from openai import APIConnectionError, APITimeoutError, RateLimitError

from live_long_rnd.embeddings import EMBEDDING_ENCODING_NAME, Embedder, OpenAIEmbedder
from live_long_rnd.evaluation import (
    EVALUATION_ITEMS,
    AnswerQualityScores,
    CitationScores,
    GateScores,
    score_answer_quality,
    score_citation_support,
    score_quality_gates,
)
from live_long_rnd.query_planning import (
    ConversationMessage,
    OpenAIQueryPlanner,
    PlannerUsage,
    QueryPlan,
    QueryPlanner,
    RawQueryPlanner,
)
from live_long_rnd.retrieve import (
    FLASHRANK_MODEL,
    FlashRankCrossEncoder,
    IdentityReranker,
    LanceDBHybridStore,
    Reranker,
    RetrievalConfig,
    RetrievalDependencies,
    RetrievalResult,
    retrieve,
    retrieve_baseline,
)

SettingValue = str | int | float
Search = Callable[[str], list[RetrievalResult]]
Clock = Callable[[], float]
EMBEDDING_INPUT_USD_PER_MILLION = 0.13
PLANNER_INPUT_USD_PER_MILLION = 0.20
PLANNER_OUTPUT_USD_PER_MILLION = 1.20


@dataclass(frozen=True)
class VariantMeasurement:
    """One full 24-item run with quality and operating measurements."""

    name: str
    settings: Mapping[str, SettingValue]
    gates: GateScores
    citations: CitationScores
    answer_quality: AnswerQualityScores
    median_latency_ms: float
    p95_latency_ms: float
    mean_packed_tokens: float
    rankings: Mapping[str, list[str]]
    page_hits: Mapping[str, list[str]]


@dataclass(frozen=True)
class ApiCost:
    """Provider usage and estimated standard API cost for one suite."""

    planner_calls: int
    planner_input_tokens: int
    planner_output_tokens: int
    embedding_input_tokens: int
    estimated_usd: float


@dataclass(frozen=True)
class MeasuredRun:
    """One measurement and its API usage."""

    measurement: VariantMeasurement
    api_cost: ApiCost


@dataclass(frozen=True)
class CalibrationReport:
    """Variant comparison, coordinate search, and selected runtime values."""

    variants: tuple[MeasuredRun, ...]
    tuning: tuple[MeasuredRun, ...]
    selected: Mapping[str, SettingValue]


def measure_variant(
    *,
    name: str,
    settings: Mapping[str, SettingValue],
    search: Search,
    clock: Clock = time.perf_counter,
) -> VariantMeasurement:
    """Run all fixed questions and retain evidence needed to audit the result."""
    encoding = tiktoken.get_encoding(EMBEDDING_ENCODING_NAME)
    results_by_item: dict[str, list[RetrievalResult]] = {}
    latencies_ms: list[float] = []
    packed_tokens: list[int] = []
    for item in EVALUATION_ITEMS:
        started_at = clock()
        results = search(item.question)
        latencies_ms.append((clock() - started_at) * 1_000)
        results_by_item[item.item_id] = results
        packed_tokens.append(sum(len(encoding.encode(result.original_text)) for result in results))

    rankings = {
        item_id: [result.document_id for result in results]
        for item_id, results in results_by_item.items()
    }
    page_hits = {
        item_id: [
            f"{result.document_id}:{','.join(str(page) for page in result.page_numbers)}"
            for result in results
        ]
        for item_id, results in results_by_item.items()
    }
    ordered_latencies = sorted(latencies_ms)
    p95_index = max(0, math.ceil(len(ordered_latencies) * 0.95) - 1)
    return VariantMeasurement(
        name=name,
        settings=settings,
        gates=score_quality_gates(rankings),
        citations=score_citation_support(results_by_item),
        answer_quality=score_answer_quality(results_by_item),
        median_latency_ms=round(statistics.median(latencies_ms), 3),
        p95_latency_ms=round(ordered_latencies[p95_index], 3),
        mean_packed_tokens=round(statistics.mean(packed_tokens), 1),
        rankings=rankings,
        page_hits=page_hits,
    )


class CachingPlanner:
    """Reuse plans after one end-to-end evaluation pass."""

    def __init__(self, planner: QueryPlanner) -> None:
        self._planner = planner
        self._plans: dict[tuple[str, tuple[tuple[str, str], ...]], QueryPlan] = {}

    @property
    def usage(self) -> PlannerUsage:
        usage = getattr(self._planner, "usage", PlannerUsage())
        return cast(PlannerUsage, usage)

    def plan(
        self,
        message: str,
        history: Sequence[ConversationMessage] = (),
    ) -> QueryPlan:
        history_key = tuple((item["role"], item["content"]) for item in history)
        key = (message, history_key)
        if key not in self._plans:
            self._plans[key] = _retry(lambda: self._planner.plan(message, history))
        return self._plans[key]

    def prewarm(self, messages: Sequence[str]) -> list[QueryPlan]:
        """Resolve a suite before timing retrieval-only operations."""
        with ThreadPoolExecutor(max_workers=4) as executor:
            return list(executor.map(self.plan, messages))


class CachingEmbedder:
    """Reuse vectors after one end-to-end evaluation pass."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._vectors: dict[str, list[float]] = {}
        self._encoding = tiktoken.get_encoding(EMBEDDING_ENCODING_NAME)

    @property
    def input_tokens(self) -> int:
        return sum(len(self._encoding.encode(text)) for text in self._vectors)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        missing = list(dict.fromkeys(text for text in texts if text not in self._vectors))
        if missing:
            vectors = _retry(lambda: self._embedder.embed(missing))
            self._vectors.update(zip(missing, vectors, strict=True))
        return [self._vectors[text] for text in texts]

    def prewarm(self, texts: Sequence[str]) -> None:
        """Embed all missing query forms in one provider batch."""
        missing = list(dict.fromkeys(text for text in texts if text not in self._vectors))
        if missing:
            vectors = _retry(lambda: self._embedder.embed(missing))
            self._vectors.update(zip(missing, vectors, strict=True))


class CachingReranker:
    """Reuse expensive cross-encoder rankings across packing controls."""

    def __init__(self, reranker: Reranker) -> None:
        self._reranker = reranker
        self._rankings: dict[
            tuple[str, tuple[tuple[str, tuple[int, ...], str, float], ...]],
            list[RetrievalResult],
        ] = {}

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
    ) -> list[RetrievalResult]:
        candidate_key = tuple(
            (
                candidate.document_id,
                tuple(candidate.page_numbers),
                candidate.original_text,
                candidate.score,
            )
            for candidate in candidates
        )
        key = (query, candidate_key)
        if key not in self._rankings:
            self._rankings[key] = self._reranker.rerank(query, candidates)
        return list(self._rankings[key])


def _retry[T](operation: Callable[[], T]) -> T:
    """Retry transient provider failures during a long calibration run."""
    for attempt in range(4):
        try:
            return operation()
        except (APIConnectionError, APITimeoutError, RateLimitError):
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class _PlannedRuntime:
    store: LanceDBHybridStore
    planner: CachingPlanner
    embedder: CachingEmbedder
    reranker: Reranker


def run_live_calibration(index_dir: Path) -> CalibrationReport:
    """Run all requested variants, then tune the winning planned query form."""
    store = LanceDBHybridStore(index_dir)
    baseline_embedder = CachingEmbedder(OpenAIEmbedder())
    raw_planner = CachingPlanner(RawQueryPlanner())
    raw_embedder = CachingEmbedder(OpenAIEmbedder())
    dual_planner = CachingPlanner(OpenAIQueryPlanner(max_intents=1))
    dual_embedder = CachingEmbedder(OpenAIEmbedder())
    aspect_planner = CachingPlanner(OpenAIQueryPlanner(max_intents=3))
    aspect_embedder = CachingEmbedder(OpenAIEmbedder())
    mini_reranker = CachingReranker(FlashRankCrossEncoder())
    base_config = RetrievalConfig(candidate_depth=10)
    questions = [item.question for item in EVALUATION_ITEMS]
    baseline_embedder.prewarm(questions)
    raw_plans = raw_planner.prewarm(questions)
    raw_embedder.prewarm(
        [intent.dense_query for plan in raw_plans for intent in plan.search_intents]
    )
    dual_plans = dual_planner.prewarm(questions)
    dual_embedder.prewarm(
        [intent.dense_query for plan in dual_plans for intent in plan.search_intents]
    )
    aspect_plans = aspect_planner.prewarm(questions)
    aspect_embedder.prewarm(
        [intent.dense_query for plan in aspect_plans for intent in plan.search_intents]
    )
    _prewarm_reranker(mini_reranker)

    variants = (
        _measure_baseline(store, baseline_embedder),
        _measure_planned(
            "raw-only",
            _PlannedRuntime(store, raw_planner, raw_embedder, mini_reranker),
            base_config,
            latency_scope="prewarmed retrieval",
        ),
        _measure_planned(
            "dual-query",
            _PlannedRuntime(store, dual_planner, dual_embedder, mini_reranker),
            base_config,
            latency_scope="prewarmed retrieval",
        ),
        _measure_planned(
            "aspect-decomposed",
            _PlannedRuntime(store, aspect_planner, aspect_embedder, mini_reranker),
            base_config,
            latency_scope="prewarmed retrieval",
        ),
    )
    planner, embedder, query_variant = _winning_query_runtime(
        variants,
        raw=(raw_planner, raw_embedder),
        dual=(dual_planner, dual_embedder),
        aspect=(aspect_planner, aspect_embedder),
    )

    candidate_runs = tuple(
        _measure_planned(
            f"candidate-depth-{depth}",
            _PlannedRuntime(store, planner, embedder, IdentityReranker()),
            RetrievalConfig(candidate_depth=depth),
            latency_scope="cached planning and embedding",
        )
        for depth in (10, 20, 40)
    )
    candidate_depth = int(select_best_run(candidate_runs).measurement.settings["candidate_depth"])

    rerankers: tuple[tuple[str, Reranker], ...] = (
        ("RRF only", IdentityReranker()),
        (FLASHRANK_MODEL, mini_reranker),
    )
    reranker_runs = tuple(
        _measure_planned(
            f"reranker-{name}",
            _PlannedRuntime(store, planner, embedder, reranker),
            RetrievalConfig(candidate_depth=candidate_depth),
            latency_scope="cached planning and embedding",
            reranker_name=name,
        )
        for name, reranker in rerankers
    )
    winning_reranker_run = select_best_run(reranker_runs)
    reranker_name = str(winning_reranker_run.measurement.settings["reranker"])
    reranker = dict(rerankers)[reranker_name]

    diversity_runs = tuple(
        _measure_planned(
            f"diversity-{penalty}",
            _PlannedRuntime(store, planner, embedder, reranker),
            RetrievalConfig(
                candidate_depth=candidate_depth,
                document_diversity_penalty=penalty,
            ),
            latency_scope="cached planning and embedding",
            reranker_name=reranker_name,
        )
        for penalty in (0.0, 0.15)
    )
    diversity_penalty = float(
        select_best_run(diversity_runs).measurement.settings["document_diversity_penalty"]
    )

    budget_runs = tuple(
        _measure_planned(
            f"source-budget-{budget}",
            _PlannedRuntime(store, planner, embedder, reranker),
            RetrievalConfig(
                candidate_depth=candidate_depth,
                source_budget_tokens=budget,
                document_diversity_penalty=diversity_penalty,
            ),
            latency_scope="cached planning and embedding",
            reranker_name=reranker_name,
        )
        for budget in (6_000, 12_000)
    )
    source_budget = int(select_best_run(budget_runs).measurement.settings["source_budget_tokens"])

    return CalibrationReport(
        variants=variants,
        tuning=(*candidate_runs, *reranker_runs, *diversity_runs, *budget_runs),
        selected={
            "query_variant": query_variant,
            "candidate_depth": candidate_depth,
            "reranker": reranker_name,
            "document_diversity_penalty": diversity_penalty,
            "source_budget_tokens": source_budget,
        },
    )


def _prewarm_reranker(reranker: Reranker) -> None:
    reranker.rerank(
        "warmup",
        [
            RetrievalResult(
                document_id="warmup",
                page_numbers=[1],
                bboxes=[{"page": 1, "l": 0.0, "t": 0.0, "r": 1.0, "b": 1.0}],
                heading_path=["warmup"],
                original_text="warmup",
                score=0.0,
            )
        ],
    )


def _measure_baseline(
    store: LanceDBHybridStore,
    embedder: CachingEmbedder,
) -> MeasuredRun:
    measurement = measure_variant(
        name="PR-31 baseline",
        settings={"k": 10, "per_document_cap": 3, "latency_scope": "prewarmed retrieval"},
        search=lambda question: retrieve_baseline(
            question,
            k=10,
            per_document_cap=3,
            store=store,
            embedder=embedder,
        ),
    )
    return MeasuredRun(measurement=measurement, api_cost=_api_cost(None, embedder))


def _measure_planned(
    name: str,
    runtime: _PlannedRuntime,
    config: RetrievalConfig,
    *,
    latency_scope: str,
    reranker_name: str = FLASHRANK_MODEL,
) -> MeasuredRun:
    measurement = measure_variant(
        name=name,
        settings={
            "candidate_depth": config.candidate_depth,
            "reranker": reranker_name,
            "document_diversity_penalty": config.document_diversity_penalty,
            "source_budget_tokens": config.source_budget_tokens,
            "latency_scope": latency_scope,
        },
        search=lambda question: retrieve(
            question,
            config=config,
            dependencies=RetrievalDependencies(
                store=runtime.store,
                planner=runtime.planner,
                embedder=runtime.embedder,
                reranker=runtime.reranker,
            ),
        ),
    )
    return MeasuredRun(
        measurement=measurement,
        api_cost=_api_cost(runtime.planner, runtime.embedder),
    )


def _api_cost(planner: CachingPlanner | None, embedder: CachingEmbedder) -> ApiCost:
    usage = planner.usage if planner is not None else PlannerUsage()
    estimated_usd = (
        usage.input_tokens * PLANNER_INPUT_USD_PER_MILLION
        + usage.output_tokens * PLANNER_OUTPUT_USD_PER_MILLION
        + embedder.input_tokens * EMBEDDING_INPUT_USD_PER_MILLION
    ) / 1_000_000
    return ApiCost(
        planner_calls=usage.calls,
        planner_input_tokens=usage.input_tokens,
        planner_output_tokens=usage.output_tokens,
        embedding_input_tokens=embedder.input_tokens,
        estimated_usd=round(estimated_usd, 6),
    )


def _winning_query_runtime(
    variants: tuple[MeasuredRun, ...],
    *,
    raw: tuple[CachingPlanner, CachingEmbedder],
    dual: tuple[CachingPlanner, CachingEmbedder],
    aspect: tuple[CachingPlanner, CachingEmbedder],
) -> tuple[CachingPlanner, CachingEmbedder, str]:
    planned_runs = variants[1:]
    winning_name = select_best_run(planned_runs).measurement.name
    runtimes = {"raw-only": raw, "dual-query": dual, "aspect-decomposed": aspect}
    planner, embedder = runtimes[winning_name]
    return planner, embedder, winning_name


def select_best_run(runs: tuple[MeasuredRun, ...]) -> MeasuredRun:
    """Select a runtime from measured quality, performance, and cost evidence."""
    return max(runs, key=_quality_key)


def _quality_key(run: MeasuredRun) -> tuple[int | float, ...]:
    measurement = run.measurement
    return (
        int(measurement.gates.passed),
        measurement.gates.contradiction_pairs,
        measurement.gates.entity_traps,
        measurement.gates.gold_documents,
        int(measurement.gates.c1_passed),
        measurement.citations.supported_items,
        measurement.answer_quality.fully_answerable_items,
        measurement.answer_quality.covered_facts,
        -measurement.p95_latency_ms,
        -run.api_cost.estimated_usd,
        -measurement.mean_packed_tokens,
    )


def main(argv: list[str] | None = None) -> int:
    """Run live calibration and write an auditable JSON report."""
    parser = argparse.ArgumentParser(prog="live_long_rnd.calibration")
    parser.add_argument("--index-dir", type=Path, default=Path("data/index"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run_live_calibration(args.index_dir)
    output = json.dumps(asdict(report), indent=2)
    if args.output is None:
        print(output)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
        print(f"Wrote calibration report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
