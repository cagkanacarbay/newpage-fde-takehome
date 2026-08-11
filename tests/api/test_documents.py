import asyncio
from pathlib import Path

import httpx
import pytest

from live_long_rnd.api.app import ApplicationConfig, create_app
from live_long_rnd.api.llm import StubLLM
from live_long_rnd.api.retrieval import StubRetriever

PDF_BYTES = b"%PDF-1.4\n%fake fixture\n%%EOF\n"


def make_corpus(tmp_path: Path) -> Path:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "MANIFEST.md").write_text(
        "# Corpus\n\n"
        "| # | Filename | Title |\n"
        "|---|---|---|\n"
        "| 001 | `001-alpha-2024-senolytics-review.pdf` | Senolytics in humans |\n",
        encoding="utf-8",
    )
    (corpus_dir / "001-alpha-2024-senolytics-review.pdf").write_bytes(PDF_BYTES)
    return corpus_dir


def get(app_corpus_dir: Path, path: str) -> httpx.Response:
    async def exercise() -> httpx.Response:
        transport = httpx.ASGITransport(
            app=create_app(
                retriever=StubRetriever(),
                llm_client=StubLLM(),
                config=ApplicationConfig(corpus_dir=app_corpus_dir),
            )
        )
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.get(path)

    return asyncio.run(exercise())


@pytest.mark.e2e
def test_document_metadata_returns_title_from_manifest(tmp_path: Path) -> None:
    corpus_dir = make_corpus(tmp_path)

    response = get(corpus_dir, "/api/documents/001-alpha-2024-senolytics-review")

    assert response.status_code == 200
    assert response.json() == {
        "document_id": "001-alpha-2024-senolytics-review",
        "title": "Senolytics in humans",
    }


@pytest.mark.e2e
def test_document_metadata_rejects_unknown_id(tmp_path: Path) -> None:
    corpus_dir = make_corpus(tmp_path)

    response = get(corpus_dir, "/api/documents/999-no-such-paper")

    assert response.status_code == 404


@pytest.mark.e2e
def test_document_metadata_falls_back_to_prettified_slug(tmp_path: Path) -> None:
    corpus_dir = make_corpus(tmp_path)
    (corpus_dir / "002-beta-2025-rapamycin-meta-analysis.pdf").write_bytes(PDF_BYTES)

    response = get(corpus_dir, "/api/documents/002-beta-2025-rapamycin-meta-analysis")

    assert response.status_code == 200
    assert response.json()["title"] == "Beta 2025 rapamycin meta analysis"


@pytest.mark.e2e
def test_document_pdf_serves_the_corpus_file(tmp_path: Path) -> None:
    corpus_dir = make_corpus(tmp_path)

    response = get(corpus_dir, "/api/documents/001-alpha-2024-senolytics-review/pdf")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content == PDF_BYTES


@pytest.mark.e2e
def test_document_pdf_rejects_unknown_and_traversal_ids(tmp_path: Path) -> None:
    corpus_dir = make_corpus(tmp_path)
    secret = tmp_path / "secret.pdf"
    secret.write_bytes(PDF_BYTES)

    for document_id in ["999-no-such-paper", "..%2Fsecret", "..%2F..%2Fetc%2Fpasswd"]:
        response = get(corpus_dir, f"/api/documents/{document_id}/pdf")
        assert response.status_code == 404, document_id
