"""Ingestion pipeline tests: provenance contract, contextual text, embeddings, storage."""

import json
import os
import subprocess
import sys
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any, ClassVar

import lancedb
import pytest
from llama_index.core import Document
from llama_index.core.schema import (
    BaseNode,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)

import live_long_rnd.ingest as ingest_module
from live_long_rnd.ingest import IngestDependencies, ingest_pdf
from live_long_rnd.parsing import CitationProvenanceError


class OpenAIEmbeddingsHandler(BaseHTTPRequestHandler):
    """Local OpenAI-compatible endpoint for the ingestion process test."""

    requests: ClassVar[list[dict[str, Any]]] = []

    def do_POST(self) -> None:
        body_size = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(body_size))
        self.requests.append(payload)
        inputs = payload["input"]
        if isinstance(inputs, str):
            inputs = [inputs]
        body = json.dumps(
            {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "index": index,
                        "embedding": [float(index + 1)] * 3072,
                    }
                    for index, _text in enumerate(inputs)
                ],
                "model": "text-embedding-3-large",
                "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *args: object) -> None:
        del args


class StubReader:
    """Reader boundary double: one document with parseable Docling-like content."""

    def __init__(self) -> None:
        texts = [{"text": f"unused {index}"} for index in range(5)]
        texts.append({"text": "Senescent cells accumulate with age."})
        texts.append({"text": "Rapamycin extends lifespan in mice."})
        payload = {"body": {"children": [{"$ref": "#/texts/5"}]}, "texts": texts}
        self.document = Document(text=json.dumps(payload))

    def load_data(self, _file_path: Path) -> list[Document]:
        return [self.document]


class StubNodeParser:
    """Node-parser boundary double: returns pre-built nodes unchanged."""

    def __init__(self, nodes: Sequence[BaseNode]) -> None:
        self._nodes = list(nodes)

    def get_nodes_from_documents(
        self, _documents: Sequence[Document], show_progress: bool = False
    ) -> list[BaseNode]:
        del show_progress  # the stub always returns its pre-built nodes
        return self._nodes


class StubEmbedder:
    """Embedding boundary double: deterministic vectors derived from text length."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(len(text))] * 4 for text in texts]


class StubStore:
    """Vector-store boundary double: captures added nodes in memory."""

    def __init__(self) -> None:
        self.nodes: list[BaseNode] = []
        self.finalize_count = 0

    def add(self, nodes: list[BaseNode]) -> list[str]:
        self.nodes.extend(nodes)
        return [node.node_id for node in nodes]

    def finalize(self) -> None:
        self.finalize_count += 1


def _linked_node(document: Document, metadata: dict[str, Any]) -> TextNode:
    node = TextNode(text="Senescent cells accumulate with age.", metadata=metadata)
    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=document.doc_id)
    return node


def _full_metadata() -> dict[str, Any]:
    return {
        "headings": ["Biomarkers of aging", "1 Introduction"],
        "doc_items": [
            {
                "self_ref": "#/texts/5",
                "label": "text",
                "prov": [
                    {
                        "page_no": 2,
                        "bbox": {"l": 10.0, "t": 20.0, "r": 30.0, "b": 40.0},
                        "charspan": [0, 36],
                    }
                ],
            }
        ],
    }


def test_ingestion_fails_loudly_when_a_chunk_has_no_heading_path(tmp_path: Path) -> None:
    source = tmp_path / "paper.pdf"
    source.touch()
    reader = StubReader()
    metadata = _full_metadata()
    del metadata["headings"]
    node = _linked_node(reader.document, metadata)

    with pytest.raises(CitationProvenanceError, match="heading path"):
        ingest_pdf(
            source,
            IngestDependencies(
                reader=reader,
                node_parser=StubNodeParser([node]),
                embedder=StubEmbedder(),
                store=StubStore(),
            ),
        )


@pytest.mark.parametrize(
    "artifact_text",
    ["1234567890();,:", "Contents lists available at ScienceDirect"],
)
def test_ingestion_excludes_headingless_front_matter_when_the_paper_has_headings(
    tmp_path: Path,
    artifact_text: str,
) -> None:
    source = tmp_path / "paper.pdf"
    source.touch()
    reader = StubReader()
    artifact_metadata = _full_metadata()
    del artifact_metadata["headings"]
    artifact = TextNode(text=artifact_text, metadata=artifact_metadata)
    artifact.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
        node_id=reader.document.doc_id
    )
    valid_node = _linked_node(reader.document, _full_metadata())
    store = StubStore()

    count = ingest_pdf(
        source,
        IngestDependencies(
            reader=reader,
            node_parser=StubNodeParser([artifact, valid_node]),
            embedder=StubEmbedder(),
            store=store,
        ),
    )

    assert count == 1
    assert [node.metadata["original_text"] for node in store.nodes] == [
        "Senescent cells accumulate with age."
    ]


def test_ingested_chunk_keeps_original_text_and_carries_full_citation(tmp_path: Path) -> None:
    source = tmp_path / "paper.pdf"
    source.touch()
    reader = StubReader()
    metadata = _full_metadata()
    metadata["captions"] = ["Table 1: Hallmarks of aging"]
    node = _linked_node(reader.document, metadata)
    store = StubStore()

    ingest_pdf(
        source,
        IngestDependencies(
            reader=reader,
            node_parser=StubNodeParser([node]),
            embedder=StubEmbedder(),
            store=store,
        ),
    )

    [stored] = store.nodes
    indexed_text = stored.get_content()
    assert indexed_text == (
        "Paper: Biomarkers of aging\n"
        "Section: Biomarkers of aging > 1 Introduction\n"
        "Element: text\n"
        "Caption: Table 1: Hallmarks of aging\n"
        "Page 2\n\n"
        "Senescent cells accumulate with age."
    )

    stored_metadata = stored.metadata
    assert stored_metadata["original_text"] == "Senescent cells accumulate with age."
    assert stored_metadata["document_id"] == source.stem
    assert stored_metadata["source_path"] == str(source.resolve())
    assert json.loads(stored_metadata["page_numbers"]) == [2]
    assert json.loads(stored_metadata["bboxes"]) == [
        {"page": 2, "l": 10.0, "t": 20.0, "r": 30.0, "b": 40.0}
    ]
    assert json.loads(stored_metadata["heading_path"]) == [
        "Biomarkers of aging",
        "1 Introduction",
    ]
    assert json.loads(stored_metadata["element_types"]) == ["text"]
    assert json.loads(stored_metadata["captions"]) == ["Table 1: Hallmarks of aging"]
    assert json.loads(stored_metadata["doc_item_refs"]) == ["#/texts/5"]
    assert json.loads(stored_metadata["citation_spans"]) == [
        {
            "text": "Senescent cells accumulate with age.",
            "page": 2,
            "bbox": {"l": 10.0, "t": 20.0, "r": 30.0, "b": 40.0},
        }
    ]


def test_ingested_chunk_is_keyed_by_the_stable_document_id(tmp_path: Path) -> None:
    source = tmp_path / "paper.pdf"
    source.touch()
    reader = StubReader()
    store = StubStore()

    ingest_pdf(
        source,
        IngestDependencies(
            reader=reader,
            node_parser=StubNodeParser([_linked_node(reader.document, _full_metadata())]),
            embedder=StubEmbedder(),
            store=store,
        ),
    )

    [stored] = store.nodes
    assert stored.ref_doc_id == source.stem


def test_ingested_chunks_carry_dense_embeddings_and_return_count(tmp_path: Path) -> None:
    source = tmp_path / "paper.pdf"
    source.touch()
    reader = StubReader()
    nodes = [
        _linked_node(reader.document, _full_metadata()),
        _linked_node(reader.document, _full_metadata()),
    ]
    store = StubStore()

    count = ingest_pdf(
        source,
        IngestDependencies(
            reader=reader,
            node_parser=StubNodeParser(nodes),
            embedder=StubEmbedder(),
            store=store,
        ),
    )

    assert count == 2
    assert len(store.nodes) == 2
    for stored in store.nodes:
        expected = float(len(stored.get_content()))
        assert stored.embedding == [expected] * 4
    assert store.finalize_count == 1


def test_ingested_chunks_keep_text_geometry_when_the_parser_shares_source_relationship(
    tmp_path: Path,
) -> None:
    source = tmp_path / "paper.pdf"
    source.touch()
    reader = StubReader()
    shared_source = RelatedNodeInfo(node_id=reader.document.doc_id)
    first = _linked_node(reader.document, _full_metadata())
    first.relationships[NodeRelationship.SOURCE] = shared_source
    second_metadata = _full_metadata()
    second_metadata["doc_items"][0]["self_ref"] = "#/texts/6"
    second_metadata["doc_items"][0]["prov"][0]["bbox"] = {
        "l": 50.0,
        "t": 60.0,
        "r": 70.0,
        "b": 80.0,
    }
    second = TextNode(text="Rapamycin extends lifespan in mice.", metadata=second_metadata)
    second.relationships[NodeRelationship.SOURCE] = shared_source
    store = StubStore()

    ingest_pdf(
        source,
        IngestDependencies(
            reader=reader,
            node_parser=StubNodeParser([first, second]),
            embedder=StubEmbedder(),
            store=store,
        ),
    )

    assert [json.loads(node.metadata["citation_spans"]) for node in store.nodes] == [
        [
            {
                "text": "Senescent cells accumulate with age.",
                "page": 2,
                "bbox": {"l": 10.0, "t": 20.0, "r": 30.0, "b": 40.0},
            }
        ],
        [
            {
                "text": "Rapamycin extends lifespan in mice.",
                "page": 2,
                "bbox": {"l": 50.0, "t": 60.0, "r": 70.0, "b": 80.0},
            }
        ],
    ]


@pytest.mark.e2e
def test_ingested_table_chunk_keeps_its_serialized_text_geometry() -> None:
    source = Path("data/corpus/longevity/010-koch-2026-pan-epigenetic-age-prediction-mammals.pdf")
    store = StubStore()

    ingest_pdf(
        source,
        IngestDependencies(
            embedder=StubEmbedder(),
            store=store,
        ),
    )

    table_nodes = [
        node for node in store.nodes if "table" in json.loads(node.metadata["element_types"])
    ]
    assert table_nodes
    assert all(json.loads(node.metadata["citation_spans"]) for node in table_nodes)
    assert any(
        "Canadian Epigenetics, Environment and Health Research Consortium" in span["text"]
        for node in table_nodes
        for span in json.loads(node.metadata["citation_spans"])
    )


def test_directory_ingestion_finalizes_the_fts_index_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "corpus"
    source_dir.mkdir()
    first_pdf = source_dir / "first.pdf"
    second_pdf = source_dir / "second.pdf"
    first_pdf.touch()
    second_pdf.touch()
    store = StubStore()
    finalized_per_pdf: list[bool] = []
    index_dir = tmp_path / "index"

    def ingest_stub(source: Path, dependencies: IngestDependencies) -> int:
        assert source in {first_pdf, second_pdf}
        finalized_per_pdf.append(dependencies.finalize)
        return 1

    def create_store(staged_index: Path) -> StubStore:
        staged_index.mkdir()
        (staged_index / "complete").touch()
        return store

    def create_stub_reader() -> StubReader:
        return StubReader()

    monkeypatch.setattr(ingest_module, "LanceDBNodeStore", create_store)
    monkeypatch.setattr(ingest_module, "create_reader", create_stub_reader)
    monkeypatch.setattr(ingest_module, "ingest_pdf", ingest_stub)

    assert ingest_module.main([str(source_dir), "--index-dir", str(index_dir)]) == 0
    assert finalized_per_pdf == [False, False]
    assert store.finalize_count == 1
    assert (index_dir / "complete").is_file()


def test_failed_directory_ingestion_preserves_the_last_complete_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "corpus"
    source_dir.mkdir()
    first_pdf = source_dir / "first.pdf"
    second_pdf = source_dir / "second.pdf"
    first_pdf.touch()
    second_pdf.touch()
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "last-complete").write_text("old", encoding="utf-8")

    def create_store(staged_index: Path) -> StubStore:
        staged_index.mkdir()
        (staged_index / "partial").write_text("new", encoding="utf-8")
        return StubStore()

    def failing_ingest(source: Path, dependencies: IngestDependencies) -> int:
        del dependencies
        if source == second_pdf:
            raise RuntimeError("embedding request failed")
        return 1

    monkeypatch.setattr(ingest_module, "LanceDBNodeStore", create_store)
    monkeypatch.setattr(ingest_module, "create_reader", StubReader)
    monkeypatch.setattr(ingest_module, "ingest_pdf", failing_ingest)

    with pytest.raises(RuntimeError, match="embedding request failed"):
        ingest_module.main([str(source_dir), "--index-dir", str(index_dir)])

    assert (index_dir / "last-complete").read_text(encoding="utf-8") == "old"
    assert not (index_dir / "partial").exists()


def _run_ingest_cli(
    source: Path,
    index_dir: Path,
    working_dir: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "live_long_rnd.ingest",
            str(source),
            "--index-dir",
            str(index_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=working_dir,
        env={
            key: value
            for key, value in os.environ.items()
            if key not in {"OPENAI_API_KEY", "OPENAI_BASE_URL"}
        },
    )


@pytest.mark.integration
@pytest.mark.e2e
def test_cli_ingests_one_corpus_pdf_with_openai_into_lancedb(tmp_path: Path) -> None:
    source = Path(
        "data/corpus/longevity/011-maleszka-2025-no-epigenetic-clock-in-insect.pdf"
    ).resolve()
    index_dir = tmp_path / "index"
    OpenAIEmbeddingsHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), OpenAIEmbeddingsHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    (tmp_path / ".env").write_text(
        f"OPENAI_API_KEY=test-key\nOPENAI_BASE_URL=http://127.0.0.1:{server.server_port}/v1\n",
        encoding="utf-8",
    )
    try:
        first_result = _run_ingest_cli(source, index_dir, tmp_path)
        result = _run_ingest_cli(source, index_dir, tmp_path)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert first_result.returncode == 0, first_result.stderr
    assert result.returncode == 0, result.stderr
    assert "create_fts_index is deprecated" not in result.stderr
    assert f"{source.name}: 1 chunks" in result.stdout
    assert "Indexed 1 chunks from 1 document(s)" in result.stdout
    table = lancedb.connect(str(index_dir)).open_table("chunks")
    rows = table.to_arrow().to_pylist()

    assert len(rows) == 1
    assert OpenAIEmbeddingsHandler.requests
    assert all(
        request["model"] == "text-embedding-3-large" for request in OpenAIEmbeddingsHandler.requests
    )
    assert all(request["dimensions"] == 3072 for request in OpenAIEmbeddingsHandler.requests)
    assert all(len(row["vector"]) == 3072 for row in rows)
    for row in rows:
        metadata = row["metadata"]
        assert metadata["document_id"] == source.stem
        assert metadata["doc_id"] == source.stem
        assert json.loads(metadata["page_numbers"])
        assert json.loads(metadata["heading_path"])
        assert metadata["original_text"]
        bboxes = json.loads(metadata["bboxes"])
        assert bboxes
        assert all({"l", "t", "r", "b"} <= bbox.keys() for bbox in bboxes)
        assert all(bbox["page"] >= 1 for bbox in bboxes)
        citation_spans = json.loads(metadata["citation_spans"])
        assert citation_spans
        assert all(span["text"].strip() for span in citation_spans)
        assert all(span["page"] >= 1 for span in citation_spans)
        assert all({"l", "t", "r", "b"} == span["bbox"].keys() for span in citation_spans)

    index_descriptions = [str(index).lower() for index in table.list_indices()]
    assert any("fts" in description for description in index_descriptions)

    expected_id = rows[0]["id"]
    dense_matches = table.search(rows[0]["vector"], query_type="vector").limit(1).to_list()
    assert dense_matches[0]["id"] == expected_id

    fts_matches = table.search("epigenetic", query_type="fts").limit(1).to_list()
    assert fts_matches[0]["id"] == expected_id
