"""Parse research PDFs into LlamaIndex documents with source identity."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import HeadingHierarchyOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from llama_index.core import Document
from llama_index.core.schema import BaseNode, NodeRelationship, RelatedNodeInfo
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.readers.docling import DoclingReader


class EmptyParseResultError(RuntimeError):
    """Raised when a parser reports success but emits no document content."""


class CitationProvenanceError(RuntimeError):
    """Raised when a searchable node cannot point to an exact source location."""


class DocumentReader(Protocol):
    """Public reader boundary used by the parser and its failure-path tests."""

    def load_data(self, file_path: Path) -> list[Document]:
        """Load LlamaIndex documents from one source file."""


class NodeParser(Protocol):
    """Public node-parser boundary used by the provenance bridge."""

    def get_nodes_from_documents(
        self, documents: Sequence[Document], show_progress: bool = False
    ) -> list[BaseNode]:
        """Create searchable nodes from parsed documents."""


def create_reader() -> DoclingReader:
    """Create one configured Docling reader for reuse across PDF files."""
    pipeline_options = PdfPipelineOptions(
        do_ocr=False, heading_hierarchy_options=HeadingHierarchyOptions(enabled=True)
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    return DoclingReader(
        export_type=DoclingReader.ExportType.JSON,
        doc_converter=converter,
    )


def _has_content(document: Document) -> bool:
    payload = json.loads(document.get_content())
    collections = ("texts", "tables", "pictures", "key_value_items", "form_items")
    return bool(payload.get("body", {}).get("children")) or any(
        payload.get(collection) for collection in collections
    )


def parse_pdf(source: Path, *, reader: DocumentReader | None = None) -> list[Document]:
    """Parse one PDF and fail when the parser emits no content."""
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(source)

    documents = (reader or create_reader()).load_data(source)
    if not documents or any(not _has_content(document) for document in documents):
        raise EmptyParseResultError(f"{source} produced no content")

    source_metadata = {
        "source_document_id": source.stem,
        "source_path": str(source.resolve()),
    }
    for document in documents:
        document.metadata.update(source_metadata)
    return documents


def _provenance_entries(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in metadata.get("doc_items", []):
        if isinstance(item, dict):
            entries.extend(prov for prov in item.get("prov", []) if isinstance(prov, dict))
    return entries


def _require_provenance(node: BaseNode) -> None:
    entries = _provenance_entries(node.metadata)
    if not entries or any(
        entry.get("page_no", 0) < 1 or not {"l", "t", "r", "b"}.issubset(entry.get("bbox", {}))
        for entry in entries
    ):
        raise CitationProvenanceError(f"Node {node.node_id} has no complete source geometry")


def chunk_documents(
    documents: Sequence[Document], *, node_parser: NodeParser | None = None
) -> list[BaseNode]:
    """Chunk parsed documents and preserve source identity and exact geometry."""
    parser = node_parser or DoclingNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    documents_by_id = {document.doc_id: document for document in documents}

    for node in nodes:
        source = node.relationships.get(NodeRelationship.SOURCE)
        if not isinstance(source, RelatedNodeInfo):
            raise CitationProvenanceError(f"Node {node.node_id} has no source document")
        source_document = documents_by_id.get(source.node_id)
        if source_document is None:
            raise CitationProvenanceError(f"Node {node.node_id} has no source document")
        node.metadata.update(source_document.metadata)
        if not node.metadata.get("source_document_id"):
            raise CitationProvenanceError(f"Node {node.node_id} has no source document ID")
        _require_provenance(node)
    return nodes
