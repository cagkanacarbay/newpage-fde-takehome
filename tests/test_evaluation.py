"""Deterministic scoring for the 24-item retrieval quality suite."""

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from lancedb.index import FTS

from live_long_rnd.evaluation import (
    CITATION_TARGETS,
    EVALUATION_ITEMS,
    CitationScores,
    GateScores,
    score_citation_support,
    score_quality_gates,
)
from live_long_rnd.query_planning import QueryPlan, SearchIntent
from live_long_rnd.retrieve import (
    LanceDBHybridStore,
    RetrievalConfig,
    RetrievalDependencies,
    RetrievalResult,
    retrieve,
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_VECTOR_DIMENSIONS = 128

_CORPUS_EVIDENCE = {
    "001": "Hallmarks of Aging therapies include senolytics, NAD boosters, metformin, and rapamycin.",
    "002": "SENS reverses aging damage, while Hallmarks of Aging optimizes function and healthspan.",
    "003": "Cellular senescence triggers include DNA damage, oxidative stress, telomere shortening, SASP, and cell-cycle arrest.",
    "004": "The single shortest telomere is the primary cause of proliferation limit and senescence.",
    "005": "Telomere dysfunction is not the primary senescence driver; proteostasis decline is a distinct pathway.",
    "006": "The NAD homeostasis supplement framework combines NMN NR, PQQ, and EGT.",
    "007": "DNAm PhenoAge is a person's predicted epigenetic age.",
    "008": "DunedinPoAm is a rate measure and speedometer for how fast a subject is aging.",
    "009": "In BASE-II, DunedinPACE best predicted mortality and ranked above GrimAge after adjustment.",
    "010": "One mammalian epigenetic clock predicts age across more than 100 mammal species.",
    "011": "The insect results do not support a claim comparable to mammalian epigenetic clocks.",
    "012": "Evidence that metformin is a proven anti-aging drug or increases lifespan remains controversial.",
    "013": "A meta-analysis of 911 effects from 167 papers and eight vertebrate species found rapamycin, not metformin, mirrors dietary restriction.",
    "014": "Combined trametinib and rapamycin increased mouse median and maximum lifespan more than either treatment.",
    "015": "In humans, senolytic treatment with D plus Q used dasatinib 100 mg and quercetin 1000 mg for diabetic kidney disease and reduced cellular senescent-cell burden.",
    "016": "The first-in-human senolytic pilot used D 100 mg/day plus Q 1250 mg/day doses for idiopathic pulmonary fibrosis and supported feasibility.",
    "017": "Senolytic treatment did not reverse cellular senescence methylation signatures in humans.",
    "018": "Calorie restriction changed mouse-liver DNA methylation toward a younger epigenetic age.",
    "019": "Chemical partial reprogramming extended C. elegans lifespan and healthspan.",
    "020": "Chemical partial reprogramming caused mouse lipid accumulation, weight loss, and toxicity.",
    "021": "Network theory with mean-field and homogeneity assumptions derived the Gompertz mortality law.",
    "022": "The Gompertz law emerges from inter-dependencies between organism sub-components in a stochastic system.",
    "023": "Loss of physiological resilience implies a human lifespan limit of 120 to 150 years.",
    "024": "A hierarchical process model links behavioral aging, VMC timing, and lifespan in C. elegans.",
    "025": "The epigenetic pacemaker uses elastic-net penalized regression to reduce prediction error.",
    "026": "The Strehler-Mildvan correlation links mortality parameters for population health monitoring.",
    "027": "The Gompertz-Makeham compensation effect was fit across 251 countries and regions.",
}


class FixedEmbedder:
    """Stable lexical hashing double for dense retrieval in CI."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [_vector(text) for text in texts]


class RawPlanner:
    """Use the fixed evaluation question for both retrieval legs."""

    def plan(self, message: str, history: object = ()) -> QueryPlan:
        del history
        return QueryPlan(
            action="retrieve",
            search_intents=[SearchIntent(dense_query=message, sparse_query=message)],
        )


class FixedCrossEncoder:
    """Deterministic lexical cross-encoder double for CI."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
    ) -> list[RetrievalResult]:
        query_terms = set(_terms(query))
        scored = [
            replace(
                candidate,
                score=float(len(query_terms.intersection(_terms(candidate.original_text)))),
            )
            for candidate in candidates
        ]
        return sorted(scored, key=lambda candidate: candidate.score, reverse=True)


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
    results["C1"] = [_retrieval_result(document_id="009-paper", pages=[1])]

    scores = score_citation_support(results)

    assert scores == CitationScores(supported_items=23, total_items=24)


@pytest.mark.e2e
def test_retrieval_quality_gates_block_regressions(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    rows = [_evidence_row(document_id, text) for document_id, text in _CORPUS_EVIDENCE.items()]
    table = lancedb.connect(str(index_dir)).create_table("chunks", data=rows)
    table.create_index("text", config=FTS(), replace=True)
    store = LanceDBHybridStore(index_dir)
    rankings: dict[str, list[str]] = {}

    for item in EVALUATION_ITEMS:
        results = retrieve(
            item.question,
            config=RetrievalConfig(
                candidate_depth=40,
                source_budget_tokens=100_000,
                document_diversity_penalty=0.15,
            ),
            dependencies=RetrievalDependencies(
                store=store,
                embedder=FixedEmbedder(),
                planner=RawPlanner(),
                reranker=FixedCrossEncoder(),
            ),
        )
        rankings[item.item_id] = [result.document_id for result in results[:10]]

    scores = score_quality_gates(rankings)

    ranking_report = json.dumps(rankings, indent=2)
    assert scores.contradiction_pairs == 6, ranking_report
    assert scores.entity_traps == 6, ranking_report
    assert scores.gold_documents >= 11, rankings
    assert scores.c1_passed, rankings
    assert scores.passed, rankings


def _evidence_row(document_id: str, text: str) -> dict[str, object]:
    return {
        "id": f"{document_id}-evidence",
        "vector": _vector(text),
        "text": text,
        "metadata": {
            "document_id": document_id,
            "page_numbers": json.dumps([1]),
            "bboxes": json.dumps([{"page": 1, "l": 1.0, "t": 2.0, "r": 3.0, "b": 4.0}]),
            "heading_path": json.dumps(["Evaluation evidence"]),
            "original_text": text,
        },
    }


def _retrieval_result(*, document_id: str, pages: list[int]) -> RetrievalResult:
    return RetrievalResult(
        document_id=document_id,
        page_numbers=pages,
        bboxes=[{"page": pages[0], "l": 1.0, "t": 2.0, "r": 3.0, "b": 4.0}],
        heading_path=["Evaluation evidence"],
        original_text="Evidence",
        score=1.0,
    )


def _vector(text: str) -> list[float]:
    vector = [0.0] * _VECTOR_DIMENSIONS
    for term in _terms(text):
        digest = hashlib.sha256(term.encode()).digest()
        vector[int.from_bytes(digest[:2]) % _VECTOR_DIMENSIONS] += 1.0
    magnitude = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / magnitude for value in vector]


def _terms(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.casefold())
