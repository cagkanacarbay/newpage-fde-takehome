import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
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
    corrected_text: str | None = None


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


@dataclass(frozen=True)
class _DraftCandidate:
    line_index: int
    text: str
    markers: tuple[int, ...]
    claim: CitedClaim


@dataclass(frozen=True)
class _AcceptedClaim:
    line_index: int
    text: str
    markers: tuple[int, ...]
    evidence: tuple[EvidenceQuote, ...]


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
    headings, candidates = _draft_candidates(draft, chunks)

    if not candidates:
        return VerifiedAnswer(text=NO_SUPPORTED_ANSWER, citations=())

    verifications = await verifier.verify_claims(
        question,
        [candidate.claim for candidate in candidates],
    )
    if len(verifications) != len(candidates):
        return VerifiedAnswer(text=NO_SUPPORTED_ANSWER, citations=())

    accepted = _accepted_claims(candidates, verifications)

    if not accepted:
        return VerifiedAnswer(text=NO_SUPPORTED_ANSWER, citations=())

    used_markers = tuple(
        dict.fromkeys(marker for accepted_claim in accepted for marker in accepted_claim.markers)
    )
    compact_markers = {original: compact for compact, original in enumerate(used_markers, start=1)}
    accepted_lines: dict[int, list[str]] = {}
    evidence_by_marker: dict[int, str] = {}
    for accepted_claim in accepted:
        compacted = _CITATION_MARKER.sub(
            lambda match: f"[{compact_markers[int(match.group(1))]}]",
            accepted_claim.text,
        )
        accepted_lines.setdefault(accepted_claim.line_index, []).append(compacted)
        for quote in accepted_claim.evidence:
            evidence_by_marker.setdefault(quote.marker, quote.exact_text)
    text = _rebuild_answer(draft, headings, accepted_lines)
    return VerifiedAnswer(
        text=text,
        citations=tuple(
            _citation_for_evidence(chunks[marker - 1], evidence_by_marker.get(marker))
            for marker in used_markers
        ),
    )


def _draft_candidates(
    draft: str,
    chunks: Sequence[RetrievedChunk],
) -> tuple[dict[int, str], list[_DraftCandidate]]:
    headings: dict[int, str] = {}
    candidates: list[_DraftCandidate] = []
    for line_index, line in enumerate(draft.splitlines()):
        stripped = line.strip()
        if re.match(r"^#{1,6}\s+\S", stripped):
            headings[line_index] = stripped
            continue
        for claim in _split_claims(line):
            markers = tuple(dict.fromkeys(int(match) for match in _CITATION_MARKER.findall(claim)))
            if not markers or any(marker < 1 or marker > len(chunks) for marker in markers):
                continue
            cited = tuple(
                CitedEvidence(marker=marker, chunk=chunks[marker - 1]) for marker in markers
            )
            if any(not _has_provenance(item.chunk) for item in cited):
                continue
            candidates.append(
                _DraftCandidate(
                    line_index=line_index,
                    text=claim,
                    markers=markers,
                    claim=CitedClaim(text=claim, evidence=cited),
                )
            )
    return headings, candidates


def _accepted_claims(
    candidates: Sequence[_DraftCandidate],
    verifications: Sequence[ClaimVerification],
) -> list[_AcceptedClaim]:
    accepted: list[_AcceptedClaim] = []
    for candidate, verification in zip(candidates, verifications, strict=True):
        supported_markers = _verified_markers(verification, candidate.claim.evidence)
        if not supported_markers:
            continue
        if verification.supported:
            accepted_text = _keep_markers(candidate.text, supported_markers)
        elif verification.corrected_text:
            accepted_text = _with_markers(verification.corrected_text, supported_markers)
        else:
            continue
        accepted.append(
            _AcceptedClaim(
                line_index=candidate.line_index,
                text=accepted_text,
                markers=supported_markers,
                evidence=verification.evidence,
            )
        )
    return accepted


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


def _verified_markers(
    verification: ClaimVerification,
    cited: Sequence[CitedEvidence],
) -> tuple[int, ...]:
    if not verification.evidence:
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


def _with_markers(corrected_text: str, markers: Sequence[int]) -> str:
    text = _CITATION_MARKER.sub("", corrected_text).strip()
    punctuation = text[-1] if text.endswith((".", "!", "?", ":")) else ""
    if punctuation:
        text = text[:-1].rstrip()
    citations = "".join(f"[{marker}]" for marker in markers)
    return f"{text} {citations}{punctuation or '.'}"


def _citation_for_evidence(chunk: RetrievedChunk, exact_text: str | None) -> RetrievedChunk:
    if not exact_text:
        return chunk
    normalized_quote = " ".join(exact_text.split())
    matching_spans = [
        span for span in chunk.citation_spans if normalized_quote in " ".join(span["text"].split())
    ]
    if not matching_spans:
        return replace(
            chunk,
            citation={**chunk.citation, "snippet": exact_text},
        )
    span = min(matching_spans, key=lambda item: len(item["text"]))
    return replace(
        chunk,
        citation={
            **chunk.citation,
            "page": span["page"],
            "bbox": span["bbox"],
            "snippet": exact_text,
        },
    )


def _rebuild_answer(
    draft: str,
    headings: dict[int, str],
    accepted_lines: dict[int, list[str]],
) -> str:
    kept_headings: set[int] = set()
    heading_stack: list[tuple[int, int]] = []
    for line_index, _line in enumerate(draft.splitlines()):
        heading = headings.get(line_index)
        if heading is not None:
            level = len(heading) - len(heading.lstrip("#"))
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, line_index))
        if line_index in accepted_lines:
            kept_headings.update(index for _level, index in heading_stack)

    rebuilt: list[str] = []
    for line_index, line in enumerate(draft.splitlines()):
        if line_index in kept_headings:
            rebuilt.append(headings[line_index])
        elif line_index in accepted_lines:
            rebuilt.append(_join_claims(accepted_lines[line_index]))
        elif not line.strip() and rebuilt and rebuilt[-1] != "":
            rebuilt.append("")

    while rebuilt and rebuilt[-1] == "":
        rebuilt.pop()
    return _renumber_ordered_lists("\n".join(rebuilt))


def _renumber_ordered_lists(text: str) -> str:
    counters: dict[str, int] = {}
    result: list[str] = []
    pattern = re.compile(r"^(\s*)\d+([.)])(\s+.*)$")
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            indent, delimiter, remainder = match.groups()
            counters[indent] = counters.get(indent, 0) + 1
            result.append(f"{indent}{counters[indent]}{delimiter}{remainder}")
            continue
        if line.strip():
            counters.clear()
        result.append(line)
    return "\n".join(result)


def _join_claims(claims: Sequence[str]) -> str:
    text = ""
    for claim in claims:
        if not text:
            text = claim
            continue
        separator = "\n" if re.match(r"(?:[-*+]\s+|\d+\.\s+)", claim) else " "
        text += separator + claim
    return text
