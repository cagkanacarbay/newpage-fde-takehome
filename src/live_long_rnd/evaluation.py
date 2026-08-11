"""The fixed 24-item retrieval suite and document-level quality gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

EVALUATION_DEPTH = 10


@dataclass(frozen=True)
class EvaluationItem:
    """One deterministic document-level retrieval check."""

    item_id: str
    gate: Literal["contradiction", "entity", "gold"]
    question: str
    gold_document_ids: tuple[str, ...]
    distractor_document_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class GateScores:
    """Aggregate quality-gate result for one retrieval variant."""

    contradiction_pairs: int
    entity_traps: int
    gold_documents: int
    c1_passed: bool
    passed: bool


@dataclass(frozen=True)
class CitationScores:
    """Page-level support for the exact passages in the evaluation set."""

    supported_items: int
    total_items: int


class EvidenceResult(Protocol):
    """Retrieval result fields needed for citation-support scoring."""

    document_id: str
    bboxes: list[Mapping[str, int | float]]


EVALUATION_ITEMS: tuple[EvaluationItem, ...] = (
    EvaluationItem(
        "A1",
        "contradiction",
        "Does telomere dysfunction drive senescence, or is it one pathway among several?",
        ("004", "005"),
    ),
    EvaluationItem(
        "A2",
        "contradiction",
        "Does senolytic treatment reverse cellular senescence in humans?",
        ("017", "015"),
    ),
    EvaluationItem(
        "A3",
        "contradiction",
        "Does chemical reprogramming rejuvenate tissues safely?",
        ("019", "020"),
    ),
    EvaluationItem(
        "A4",
        "contradiction",
        "Is metformin a proven anti-aging drug compared with rapamycin?",
        ("012", "013"),
    ),
    EvaluationItem(
        "A5",
        "contradiction",
        "How do the Hallmarks of Aging and SENS frameworks compare?",
        ("001", "002"),
    ),
    EvaluationItem(
        "A6",
        "contradiction",
        "Do mammalian epigenetic clocks generalize to insects?",
        ("010", "011"),
    ),
    EvaluationItem(
        "B1",
        "entity",
        "What happened when trametinib was combined with rapamycin?",
        ("014",),
        ("013", "012"),
    ),
    EvaluationItem(
        "B2",
        "entity",
        "What D+Q doses were used for idiopathic pulmonary fibrosis?",
        ("016",),
        ("015",),
    ),
    EvaluationItem(
        "B3",
        "entity",
        "What D+Q regimen was used for diabetic kidney disease?",
        ("015",),
        ("016",),
    ),
    EvaluationItem(
        "B4",
        "entity",
        "How is DunedinPoAm distinct from PhenoAge?",
        ("007", "008"),
        ("009",),
    ),
    EvaluationItem(
        "B5",
        "entity",
        "Does rapamycin, rather than metformin, mirror dietary restriction?",
        ("013",),
        ("012", "014"),
    ),
    EvaluationItem(
        "B6",
        "entity",
        "How does calorie restriction affect mouse-liver methylation?",
        ("018",),
        ("013", "010"),
    ),
    EvaluationItem(
        "C1",
        "gold",
        "Which biomarker best predicted mortality in BASE-II: DunedinPACE, ranked against GrimAge?",
        ("009",),
    ),
    EvaluationItem(
        "C2",
        "gold",
        "What human lifespan limit follows from loss of physiological resilience?",
        ("023",),
    ),
    EvaluationItem(
        "C3",
        "gold",
        "How can the Gompertz law arise from inter-dependencies between organism sub-components?",
        ("022",),
    ),
    EvaluationItem(
        "C4",
        "gold",
        "What is the Strehler-Mildvan correlation, and how can it monitor population health?",
        ("026",),
    ),
    EvaluationItem(
        "C5",
        "gold",
        "What hierarchical process model links behavioral aging and lifespan in C. elegans?",
        ("024",),
    ),
    EvaluationItem(
        "C6",
        "gold",
        "What is the Gompertz-Makeham compensation effect across 251 countries and regions?",
        ("027",),
    ),
    EvaluationItem(
        "C7",
        "gold",
        "How does elastic-net penalized regression support the epigenetic pacemaker?",
        ("025",),
    ),
    EvaluationItem(
        "C8",
        "gold",
        "Which meta-analysis covered 911 effects, 167 papers, and eight vertebrate species?",
        ("013",),
    ),
    EvaluationItem(
        "C9",
        "gold",
        "What supplement framework combines NMN/NR, PQQ, and EGT for NAD+ homeostasis?",
        ("006",),
    ),
    EvaluationItem(
        "C10",
        "gold",
        "Which study derived the Gompertz mortality law from network theory?",
        ("021",),
    ),
    EvaluationItem(
        "C11",
        "gold",
        "What are the triggers and hallmarks of cellular senescence, including SASP?",
        ("003",),
    ),
    EvaluationItem(
        "C12",
        "gold",
        "What did the first-in-human senolytic pilot in IPF find about feasibility?",
        ("016",),
    ),
)

CITATION_TARGETS: Mapping[str, tuple[tuple[str, int], ...]] = {
    "A1": (("004", 2), ("005", 1)),
    "A2": (("017", 1), ("015", 1)),
    "A3": (("019", 1), ("020", 1)),
    "A4": (("012", 1), ("013", 1)),
    "A5": (("001", 1), ("002", 3)),
    "A6": (("010", 1), ("011", 1)),
    "B1": (("014", 2),),
    "B2": (("016", 3),),
    "B3": (("015", 1),),
    "B4": (("008", 4), ("007", 2)),
    "B5": (("013", 1),),
    "B6": (("018", 1),),
    "C1": (("009", 1), ("009", 10)),
    "C2": (("023", 1),),
    "C3": (("022", 2),),
    "C4": (("026", 1),),
    "C5": (("024", 18),),
    "C6": (("027", 1),),
    "C7": (("025", 2),),
    "C8": (("013", 1),),
    "C9": (("006", 1),),
    "C10": (("021", 1),),
    "C11": (("003", 2),),
    "C12": (("016", 1),),
}


def score_quality_gates(rankings: Mapping[str, Sequence[str]]) -> GateScores:
    """Score top-10 document rankings against the three decided gates."""
    passes: dict[str, bool] = {}
    for item in EVALUATION_ITEMS:
        documents = list(
            dict.fromkeys(_paper_id(value) for value in rankings.get(item.item_id, ()))
        )[:10]
        if item.gate == "entity":
            passes[item.item_id] = _entity_item_passes(item, documents)
        else:
            passes[item.item_id] = all(
                gold_document_id in documents for gold_document_id in item.gold_document_ids
            )

    contradiction_pairs = sum(
        passes[item.item_id] for item in EVALUATION_ITEMS if item.gate == "contradiction"
    )
    entity_traps = sum(passes[item.item_id] for item in EVALUATION_ITEMS if item.gate == "entity")
    gold_documents = sum(passes[item.item_id] for item in EVALUATION_ITEMS if item.gate == "gold")
    c1_passed = passes["C1"]
    return GateScores(
        contradiction_pairs=contradiction_pairs,
        entity_traps=entity_traps,
        gold_documents=gold_documents,
        c1_passed=c1_passed,
        passed=(
            contradiction_pairs == 6 and entity_traps == 6 and gold_documents >= 11 and c1_passed
        ),
    )


def score_citation_support(
    results: Mapping[str, Sequence[EvidenceResult]],
) -> CitationScores:
    """Count items whose packed evidence contains every exact target page."""
    supported_items = 0
    for item_id, targets in CITATION_TARGETS.items():
        item_results = results.get(item_id, ())
        if all(
            any(
                _paper_id(result.document_id) == document_id
                and _emitted_citation_page(result) == page
                for result in item_results
            )
            for document_id, page in targets
        ):
            supported_items += 1
    return CitationScores(
        supported_items=supported_items,
        total_items=len(CITATION_TARGETS),
    )


def _emitted_citation_page(result: EvidenceResult) -> int:
    """Return the page exposed by the API citation payload."""
    return int(result.bboxes[0]["page"])


def _entity_item_passes(item: EvaluationItem, documents: Sequence[str]) -> bool:
    ranks = {document_id: rank for rank, document_id in enumerate(documents)}
    if any(gold not in ranks for gold in item.gold_document_ids):
        return False
    distractor_ranks = [
        ranks[distractor] for distractor in item.distractor_document_ids if distractor in ranks
    ]
    return not distractor_ranks or all(
        ranks[gold] < min(distractor_ranks) for gold in item.gold_document_ids
    )


def _paper_id(document_id: str) -> str:
    """Accept a full corpus document ID or its stable three-digit prefix."""
    return document_id.split("-", maxsplit=1)[0]
