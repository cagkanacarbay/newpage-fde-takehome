import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import httpx
import pytest

from live_long_rnd.api.app import ApplicationConfig, create_app
from live_long_rnd.api.conversations import StoredMessage
from live_long_rnd.api.generation import (
    CitedEvidence,
    ClaimVerification,
    EvidenceQuote,
    StubClaimVerifier,
    personal_medical_refusal,
    verify_draft,
)
from live_long_rnd.api.retrieval import RetrievedChunk, StubRetriever


class CitedDraftLLM:
    def __init__(self, draft: str = "Senolytics selectively target senescent cells [1].") -> None:
        self.draft = draft

    def stream_answer(
        self,
        message: str,
        chunks: Sequence[RetrievedChunk],
        history: Sequence[StoredMessage],
    ) -> AsyncIterator[str]:
        del message, chunks, history

        async def tokens() -> AsyncIterator[str]:
            yield self.draft

        return tokens()


class ExactEvidenceVerifier:
    def __init__(self, *, supported: bool = True, fail: bool = False) -> None:
        self.supported = supported
        self.fail = fail
        self.claims: list[str] = []

    async def verify_claim(
        self,
        question: str,
        claim: str,
        evidence: Sequence[CitedEvidence],
    ) -> ClaimVerification:
        del question
        self.claims.append(claim)
        if self.fail:
            raise RuntimeError("Verifier service failed.")
        return ClaimVerification(
            supported=self.supported,
            evidence=tuple(
                EvidenceQuote(marker=item.marker, exact_text=item.chunk.text) for item in evidence
            ),
        )


class EmptyRetriever:
    async def retrieve(
        self,
        message: str,
        history: Sequence[object] = (),
    ) -> Sequence[RetrievedChunk]:
        del message, history
        return ()


class RecordingLLM(CitedDraftLLM):
    def __init__(self, draft: str) -> None:
        super().__init__(draft)
        self.calls = 0

    def stream_answer(
        self,
        message: str,
        chunks: Sequence[RetrievedChunk],
        history: Sequence[StoredMessage],
    ) -> AsyncIterator[str]:
        self.calls += 1
        return super().stream_answer(message, chunks, history)


def test_personal_health_context_allows_corpus_research_questions() -> None:
    assert personal_medical_refusal("I take metformin. What trials studied it?") is None


def _events(response: httpx.Response) -> list[dict[str, object]]:
    return [
        json.loads(frame.removeprefix("data: "))
        for frame in response.text.rstrip("\n").split("\n\n")
    ]


@pytest.mark.e2e
def test_supported_answer_returns_one_verified_claim_and_exact_citation(
    tmp_path: Path,
) -> None:
    async def exercise() -> httpx.Response:
        transport = httpx.ASGITransport(
            app=create_app(
                retriever=StubRetriever(),
                llm_client=CitedDraftLLM(),
                verifier=ExactEvidenceVerifier(),
                config=ApplicationConfig(database_path=tmp_path / "conversations.db"),
            )
        )
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post(
                "/api/chat",
                json={"message": "What do senolytics target?"},
            )

    events = _events(asyncio.run(exercise()))

    assert [event["type"] for event in events] == [
        "conversation",
        "token",
        "citations",
        "done",
    ]
    assert events[1] == {
        "type": "token",
        "text": "Senolytics selectively target senescent cells [1].",
    }
    assert events[2] == {
        "type": "citations",
        "citations": [
            {
                "document_id": ("015-hickson-2019-senolytics-dasatinib-quercetin-first-in-human"),
                "page": 2,
                "heading_path": ["Research in context", "1. Introduction"],
                "bbox": {"l": 310.5, "t": 332.6, "r": 561.6, "b": 53.3},
                "snippet": "By definition, the target of senolytics is senescent cells...",
            }
        ],
    }


async def _chat(
    tmp_path: Path,
    *,
    draft: str,
    verifier: ExactEvidenceVerifier,
) -> list[dict[str, object]]:
    transport = httpx.ASGITransport(
        app=create_app(
            retriever=StubRetriever(),
            llm_client=CitedDraftLLM(draft),
            verifier=verifier,
            config=ApplicationConfig(database_path=tmp_path / "conversations.db"),
        )
    )
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return _events(
            await client.post(
                "/api/chat",
                json={"message": "What do senolytics target?"},
            )
        )


@pytest.mark.e2e
def test_unknown_citation_marker_fails_closed(tmp_path: Path) -> None:
    verifier = ExactEvidenceVerifier()

    events = asyncio.run(
        _chat(
            tmp_path,
            draft="A claim with an unknown source [4].",
            verifier=verifier,
        )
    )

    assert verifier.claims == []
    assert events[1] == {
        "type": "token",
        "text": "The retrieved evidence did not provide a supported answer.",
    }
    assert events[2] == {"type": "citations", "citations": []}
    assert "unknown source" not in json.dumps(events)


@pytest.mark.e2e
def test_uncited_factual_claim_is_not_returned(tmp_path: Path) -> None:
    events = asyncio.run(
        _chat(
            tmp_path,
            draft=(
                "Senolytics selectively target senescent cells [1]. "
                "They always extend human lifespan."
            ),
            verifier=ExactEvidenceVerifier(),
        )
    )

    assert events[1]["text"] == "Senolytics selectively target senescent cells [1]."
    assert "always extend" not in json.dumps(events)


@pytest.mark.e2e
def test_verifier_rejection_removes_the_claim(tmp_path: Path) -> None:
    events = asyncio.run(
        _chat(
            tmp_path,
            draft="Senolytics extend human lifespan [1].",
            verifier=ExactEvidenceVerifier(supported=False),
        )
    )

    assert events[1] == {
        "type": "token",
        "text": "The retrieved evidence did not provide a supported answer.",
    }
    assert "extend human lifespan" not in json.dumps(events)


@pytest.mark.e2e
def test_verifier_failure_returns_error_without_draft_text(tmp_path: Path) -> None:
    events = asyncio.run(
        _chat(
            tmp_path,
            draft="This unverified draft must stay private [1].",
            verifier=ExactEvidenceVerifier(fail=True),
        )
    )

    assert [event["type"] for event in events] == ["conversation", "error"]
    assert events[-1] == {"type": "error", "message": "The chat request failed."}
    assert "unverified draft" not in json.dumps(events)


@pytest.mark.e2e
def test_openai_verifier_missing_key_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LIVE_LONG_VERIFIER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def exercise() -> list[dict[str, object]]:
        transport = httpx.ASGITransport(
            app=create_app(
                retriever=StubRetriever(),
                llm_client=CitedDraftLLM("Private draft [1]."),
                config=ApplicationConfig(database_path=tmp_path / "conversations.db"),
            )
        )
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return _events(
                await client.post(
                    "/api/chat",
                    json={"message": "What do senolytics target?"},
                )
            )

    events = asyncio.run(exercise())

    assert [event["type"] for event in events] == ["conversation", "error"]
    assert events[-1] == {
        "type": "error",
        "message": (
            "OpenAI verification is selected but OPENAI_API_KEY is not set. "
            "Set OPENAI_API_KEY and restart the server."
        ),
    }
    assert "Private draft" not in json.dumps(events)


@pytest.mark.e2e
def test_conflict_answer_preserves_and_cites_both_sides(tmp_path: Path) -> None:
    events = asyncio.run(
        _chat(
            tmp_path,
            draft=(
                "A pilot study reported improved physical function [2]. "
                "A newer study found no reversal of the methylation signature [3]."
            ),
            verifier=ExactEvidenceVerifier(),
        )
    )

    assert events[1]["text"] == (
        "A pilot study reported improved physical function [1]. "
        "A newer study found no reversal of the methylation signature [2]."
    )
    citations = events[2]["citations"]
    assert isinstance(citations, list)
    assert [citation["document_id"] for citation in citations] == [
        "016-justice-2019-senolytics-idiopathic-pulmonary-fibrosis-trial",
        "017-kasamoto-2026-senolytics-do-not-reverse-senescence-methylation",
    ]


@pytest.mark.e2e
def test_no_relevant_evidence_skips_generation(tmp_path: Path) -> None:
    llm = RecordingLLM("This must never be generated [1].")

    async def exercise() -> list[dict[str, object]]:
        transport = httpx.ASGITransport(
            app=create_app(
                retriever=EmptyRetriever(),
                llm_client=llm,
                verifier=ExactEvidenceVerifier(),
                config=ApplicationConfig(database_path=tmp_path / "conversations.db"),
            )
        )
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return _events(await client.post("/api/chat", json={"message": "No match?"}))

    events = asyncio.run(exercise())

    assert llm.calls == 0
    assert events[1] == {
        "type": "token",
        "text": "No sufficiently relevant evidence was found in the corpus.",
    }
    assert events[2] == {"type": "citations", "citations": []}


@pytest.mark.e2e
def test_personal_dosing_question_is_declined_without_generation(tmp_path: Path) -> None:
    llm = RecordingLLM("You should take 100 mg [1].")

    async def exercise() -> list[dict[str, object]]:
        transport = httpx.ASGITransport(
            app=create_app(
                retriever=StubRetriever(),
                llm_client=llm,
                verifier=ExactEvidenceVerifier(),
                config=ApplicationConfig(database_path=tmp_path / "conversations.db"),
            )
        )
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return _events(
                await client.post(
                    "/api/chat",
                    json={"message": "Is 100 mg safe for me?"},
                )
            )

    events = asyncio.run(exercise())

    assert llm.calls == 0
    assert events[1] == {
        "type": "token",
        "text": (
            "I can summarize study evidence, but I cannot provide personal diagnosis, "
            "treatment, or dosing advice."
        ),
    }
    assert events[2] == {"type": "citations", "citations": []}


@pytest.mark.e2e
def test_personal_condition_and_safety_question_is_declined_without_generation(
    tmp_path: Path,
) -> None:
    llm = RecordingLLM("Creatine is safe for you [1].")

    async def exercise() -> list[dict[str, object]]:
        transport = httpx.ASGITransport(
            app=create_app(
                retriever=StubRetriever(),
                llm_client=llm,
                verifier=ExactEvidenceVerifier(),
                config=ApplicationConfig(database_path=tmp_path / "conversations.db"),
            )
        )
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return _events(
                await client.post(
                    "/api/chat",
                    json={"message": "I have kidney disease. Is creatine safe?"},
                )
            )

    events = asyncio.run(exercise())

    assert llm.calls == 0
    assert events[1]["text"] == (
        "I can summarize study evidence, but I cannot provide personal diagnosis, "
        "treatment, or dosing advice."
    )
    assert events[2] == {"type": "citations", "citations": []}


def test_missing_provenance_rejects_claim_before_model_verification() -> None:
    chunk = RetrievedChunk(
        text="A result.",
        citation={
            "document_id": "",
            "page": 1,
            "heading_path": [],
            "bbox": {"l": 1, "t": 2, "r": 3, "b": 0},
            "snippet": "A result.",
        },
    )

    answer = asyncio.run(
        verify_draft("What was the result?", "A result [1].", [chunk], StubClaimVerifier())
    )

    assert answer.text == "The retrieved evidence did not provide a supported answer."
    assert answer.citations == ()


def test_evidence_text_absent_from_cited_chunk_rejects_claim() -> None:
    class InventedQuoteVerifier:
        async def verify_claim(
            self,
            question: str,
            claim: str,
            evidence: Sequence[CitedEvidence],
        ) -> ClaimVerification:
            del question, claim, evidence
            return ClaimVerification(
                supported=True,
                evidence=(EvidenceQuote(marker=1, exact_text="Invented quote."),),
            )

    async def exercise() -> str:
        chunk = (await StubRetriever().retrieve("question"))[0]
        answer = await verify_draft(
            "What do senolytics target?",
            "Senolytics target senescent cells [1].",
            [chunk],
            InventedQuoteVerifier(),
        )
        return answer.text

    assert asyncio.run(exercise()) == ("The retrieved evidence did not provide a supported answer.")


def test_verifier_receives_the_user_question_with_each_claim() -> None:
    received: list[tuple[str, str]] = []

    class RecordingVerifier:
        async def verify_claim(
            self,
            question: str,
            claim: str,
            evidence: Sequence[CitedEvidence],
        ) -> ClaimVerification:
            received.append((question, claim))
            return ClaimVerification(
                supported=True,
                evidence=tuple(
                    EvidenceQuote(marker=item.marker, exact_text=item.chunk.text)
                    for item in evidence
                ),
            )

    async def exercise() -> None:
        chunk = (await StubRetriever().retrieve("question"))[0]
        await verify_draft(
            "What do senolytics target?",
            "Senolytics target senescent cells [1].",
            [chunk],
            RecordingVerifier(),
        )

    asyncio.run(exercise())

    assert received == [
        (
            "What do senolytics target?",
            "Senolytics target senescent cells [1].",
        )
    ]
