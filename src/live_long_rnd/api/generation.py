import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from live_long_rnd.api.retrieval import RetrievedChunk

_CITATION_MARKER = re.compile(r"\[(\d+)]")
_GROUPED_CITATION_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)+)]")
_CITED_CLAIM_END = re.compile(r"(?:[.!?]\s*(?:\[\d+]\s*)+|(?:\[\d+]\s*)+[.!?]?)(?=\s|$)")

NO_SUPPORTED_ANSWER = "The retrieved evidence did not provide a supported answer."
NO_RELEVANT_EVIDENCE = "No sufficiently relevant evidence was found in the corpus."
PERSONAL_MEDICAL_REFUSAL = (
    "I can summarize study evidence, but I cannot provide personal diagnosis, "
    "treatment, or dosing advice."
)

_PERSONAL_MEDICAL_PATTERNS = (
    re.compile(
        r"\b(?:should|can|could|do)\s+i\s+(?:take|use|start|stop|change)\b",
        re.I,
    ),
    re.compile(r"\b(?:diagnos\w*|treat\w*)\s+me\b", re.I),
    re.compile(
        r"\b(?:is|would|will|can|could)\b.*\b(?:safe|harmful|dangerous)\b.*"
        r"\b(?:for me|for myself|personally)\b",
        re.I,
    ),
    re.compile(r"\b(?:do|could|can)\s+i\s+have\b", re.I),
    re.compile(r"\b(?:hurt|harm)\s+me\b", re.I),
    re.compile(r"\b(?:what(?:'s| is)|is)\s+wrong\s+with\s+me\b", re.I),
)

_PERSONAL_HEALTH_CONTEXT = re.compile(
    r"\b(?:i\s+(?:have|am|was|take|use)\b|for\s+me\b|with\s+me\b|"
    r"my\s+(?!(?:papers?|uploaded|research|corpus|documents?|files?)\b))",
    re.I,
)
_PERSONAL_MEDICAL_REQUEST = re.compile(
    r"\b(?:safe|safety|contraindicat\w*|interact\w*|risk\w*|side effects?|"
    r"okay|advisable|dose|dosing|treat\w*|diagnos\w*|medications?|"
    r"supplements?|recommend\w*|prescrib\w*|help\w*|wrong)\b",
    re.I,
)


@dataclass(frozen=True)
class CitedEvidence:
    marker: int
    chunk: RetrievedChunk


@dataclass(frozen=True)
class CitedClaim:
    text: str
    evidence: tuple[CitedEvidence, ...]


@dataclass(frozen=True)
class EvidenceQuote:
    marker: int
    exact_text: str


@dataclass(frozen=True)
class ClaimVerification:
    supported: bool
    evidence: tuple[EvidenceQuote, ...]


class ClaimVerifier(Protocol):
    async def verify_claims(
        self,
        question: str,
        claims: Sequence[CitedClaim],
    ) -> Sequence[ClaimVerification]: ...


class StubClaimVerifier:
    async def verify_claims(
        self,
        question: str,
        claims: Sequence[CitedClaim],
    ) -> Sequence[ClaimVerification]:
        del question
        return tuple(
            ClaimVerification(
                supported=True,
                evidence=tuple(
                    EvidenceQuote(marker=item.marker, exact_text=item.chunk.text)
                    for item in claim.evidence
                ),
            )
            for claim in claims
        )


@dataclass(frozen=True)
class VerifiedAnswer:
    text: str
    citations: tuple[RetrievedChunk, ...]


def personal_medical_refusal(message: str) -> str | None:
    direct_request = any(pattern.search(message) for pattern in _PERSONAL_MEDICAL_PATTERNS)
    if direct_request:
        return PERSONAL_MEDICAL_REFUSAL
    contextual_request = bool(
        _PERSONAL_HEALTH_CONTEXT.search(message) and _PERSONAL_MEDICAL_REQUEST.search(message)
    )
    if contextual_request:
        return PERSONAL_MEDICAL_REFUSAL
    return None


async def verify_draft(
    question: str,
    draft: str,
    chunks: Sequence[RetrievedChunk],
    verifier: ClaimVerifier,
) -> VerifiedAnswer:
    draft = _normalize_citation_markers(draft)
    candidates: list[tuple[str, tuple[int, ...], CitedClaim]] = []
    for claim in _split_claims(draft):
        markers = tuple(dict.fromkeys(int(match) for match in _CITATION_MARKER.findall(claim)))
        if not markers or any(marker < 1 or marker > len(chunks) for marker in markers):
            continue
        cited = tuple(CitedEvidence(marker=marker, chunk=chunks[marker - 1]) for marker in markers)
        if any(not _has_provenance(item.chunk) for item in cited):
            continue
        candidates.append((claim, markers, CitedClaim(text=claim, evidence=cited)))

    if not candidates:
        return VerifiedAnswer(text=NO_SUPPORTED_ANSWER, citations=())

    verifications = await verifier.verify_claims(
        question,
        [candidate for _claim, _markers, candidate in candidates],
    )
    if len(verifications) != len(candidates):
        return VerifiedAnswer(text=NO_SUPPORTED_ANSWER, citations=())

    accepted: list[tuple[str, tuple[int, ...]]] = []
    for (claim, _markers, candidate), verification in zip(
        candidates,
        verifications,
        strict=True,
    ):
        cited = candidate.evidence
        supported_markers = _supported_markers(verification, cited)
        if not supported_markers:
            continue
        accepted.append((_keep_markers(claim, supported_markers), supported_markers))

    if not accepted:
        return VerifiedAnswer(text=NO_SUPPORTED_ANSWER, citations=())

    used_markers = tuple(
        dict.fromkeys(marker for _claim, markers in accepted for marker in markers)
    )
    compact_markers = {original: compact for compact, original in enumerate(used_markers, start=1)}
    compacted_claims = tuple(
        _CITATION_MARKER.sub(
            lambda match: f"[{compact_markers[int(match.group(1))]}]",
            claim,
        )
        for claim, _markers in accepted
    )
    text = _join_claims(compacted_claims)
    return VerifiedAnswer(
        text=text,
        citations=tuple(chunks[marker - 1] for marker in used_markers),
    )


def _split_claims(draft: str) -> tuple[str, ...]:
    claims: list[str] = []
    for line in draft.splitlines():
        start = 0
        for match in _CITED_CLAIM_END.finditer(line.strip()):
            claim = line.strip()[start : match.end()].strip()
            if claim:
                claims.append(claim)
            start = match.end()
        remainder = line.strip()[start:].strip()
        if remainder:
            claims.append(remainder)
    return tuple(claims)


def _normalize_citation_markers(draft: str) -> str:
    return _GROUPED_CITATION_MARKER.sub(
        lambda match: "".join(f"[{marker.strip()}]" for marker in match.group(1).split(",")),
        draft,
    )


def _has_provenance(chunk: RetrievedChunk) -> bool:
    citation = chunk.citation
    bbox = citation.get("bbox")
    return bool(
        chunk.text.strip()
        and citation.get("document_id")
        and citation.get("page")
        and citation.get("snippet")
        and isinstance(bbox, dict)
        and set(bbox) == {"l", "t", "r", "b"}
    )


def _supported_markers(
    verification: ClaimVerification,
    cited: Sequence[CitedEvidence],
) -> tuple[int, ...]:
    if not verification.supported or not verification.evidence:
        return ()
    cited_by_marker = {item.marker: item for item in cited}
    evidence_by_marker = {item.marker: item.exact_text for item in verification.evidence}
    if not set(evidence_by_marker).issubset(cited_by_marker):
        return ()
    return tuple(
        item.marker
        for item in cited
        if item.marker in evidence_by_marker
        and bool(evidence_by_marker[item.marker].strip())
        and evidence_by_marker[item.marker] in item.chunk.text
    )


def _keep_markers(claim: str, markers: Sequence[int]) -> str:
    retained = set(markers)
    text = _CITATION_MARKER.sub(
        lambda match: match.group(0) if int(match.group(1)) in retained else "",
        claim,
    )
    return re.sub(r"\s+([.!?,;:])", r"\1", text).strip()


def _join_claims(claims: Sequence[str]) -> str:
    text = ""
    for claim in claims:
        if not text:
            text = claim
            continue
        separator = "\n" if re.match(r"(?:[-*+]\s+|\d+\.\s+)", claim) else " "
        text += separator + claim
    return text
