"""Ingest corpus PDFs into a hybrid LanceDB index with full citation provenance."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol
from uuid import uuid4

import tiktoken
from docling_core.transforms.chunker.hierarchical_chunker import ChunkingDocSerializer
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from docling_core.types.doc.document import DoclingDocument
from dotenv import load_dotenv
from lancedb.index import FTS
from llama_index.core.schema import BaseNode, NodeRelationship, RelatedNodeInfo
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.vector_stores.lancedb import LanceDBVectorStore

from live_long_rnd.embeddings import EMBEDDING_ENCODING_NAME, Embedder, OpenAIEmbedder
from live_long_rnd.index_config import DEFAULT_INDEX_DIR, DEFAULT_TABLE_NAME
from live_long_rnd.parsing import (
    CitationProvenanceError,
    DocumentReader,
    NodeParser,
    chunk_documents,
    create_reader,
    parse_pdf,
)

EMBEDDING_CHUNK_MAX_TOKENS = 7_500


class LanceDBNodeStore:
    """LanceDB table with a dense vector leg and a full-text (BM25) leg."""

    def __init__(self, index_dir: Path, table_name: str = DEFAULT_TABLE_NAME) -> None:
        self._store = LanceDBVectorStore(uri=str(index_dir), table_name=table_name, mode="create")

    def add(self, nodes: list[BaseNode]) -> list[str]:
        return self._store.add(nodes)

    def finalize(self) -> None:
        """Build the full-text leg after all documents are stored."""
        table = self._store.table
        assert table is not None  # add() creates the table when it does not exist
        text_key = self._store.text_key
        assert text_key is not None  # LanceDB requires the text field for its FTS index
        table.create_index(  # type: ignore[call-overload]  # LanceDB's FTS stub omits replace.
            text_key, config=FTS(with_position=True), replace=True
        )


def _create_node_parser() -> DoclingNodeParser:
    tokenizer = OpenAITokenizer(
        tokenizer=tiktoken.get_encoding(EMBEDDING_ENCODING_NAME),
        max_tokens=EMBEDDING_CHUNK_MAX_TOKENS,
    )
    return DoclingNodeParser(chunker=HybridChunker(tokenizer=tokenizer))


class NodeStore(Protocol):
    """Vector-store boundary: persists nodes with their embeddings."""

    def add(self, nodes: list[BaseNode]) -> list[str]:
        """Add nodes to the store and return their IDs."""

    def finalize(self) -> None:
        """Build indexes after all nodes have been added."""


@dataclass
class IngestDependencies:
    """Optional seams for one ingestion operation."""

    reader: DocumentReader | None = None
    node_parser: NodeParser | None = None
    embedder: Embedder | None = None
    store: NodeStore | None = None
    finalize: bool = True


def _provenance_geometry(metadata: dict[str, Any]) -> tuple[list[int], list[dict[str, Any]]]:
    pages: set[int] = set()
    bboxes: list[dict[str, Any]] = []
    for item in metadata.get("doc_items", []):
        if not isinstance(item, dict):
            continue
        for prov in item.get("prov", []):
            if not isinstance(prov, dict):
                continue
            page_no = prov.get("page_no", 0)
            bbox = prov.get("bbox", {})
            if page_no >= 1 and {"l", "t", "r", "b"}.issubset(bbox):
                pages.add(page_no)
                bboxes.append({"page": page_no, **{k: bbox[k] for k in ("l", "t", "r", "b")}})
    return sorted(pages), bboxes


def _document_item_texts(document: BaseNode) -> dict[str, str]:
    payload = json.loads(document.get_content())
    item_texts: dict[str, str] = {}
    for collection in ("texts", "tables", "pictures", "key_value_items", "form_items"):
        for index, item in enumerate(payload.get(collection, [])):
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                continue
            reference = str(item.get("self_ref") or f"#/{collection}/{index}")
            item_texts[reference] = item["text"]
    if payload.get("schema_name") == "DoclingDocument":
        docling_document = DoclingDocument.model_validate(payload)
        serializer = ChunkingDocSerializer(doc=docling_document)
        for table in docling_document.tables:
            serialized = serializer.table_serializer.serialize(
                item=table,
                doc_serializer=serializer,
                doc=docling_document,
            )
            if serialized.text:
                item_texts[table.self_ref] = serialized.text
    return item_texts


def _citation_spans(
    metadata: dict[str, Any],
    item_texts: dict[str, str],
) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for item in metadata.get("doc_items", []):
        if not isinstance(item, dict):
            continue
        item_text = item_texts.get(str(item.get("self_ref", "")), "")
        for provenance in item.get("prov", []):
            if not isinstance(provenance, dict):
                continue
            page = provenance.get("page_no", 0)
            bbox = provenance.get("bbox", {})
            if page < 1 or not {"l", "t", "r", "b"}.issubset(bbox):
                continue
            text = item_text
            charspan = provenance.get("charspan")
            if (
                isinstance(charspan, list)
                and len(charspan) == 2
                and all(isinstance(offset, int) for offset in charspan)
            ):
                start, end = charspan
                if 0 <= start < end <= len(item_text):
                    text = item_text[start:end]
            if text.strip():
                spans.append(
                    {
                        "text": text,
                        "page": page,
                        "bbox": {key: bbox[key] for key in ("l", "t", "r", "b")},
                    }
                )
    return spans


def _contextual_text(
    heading_path: list[str],
    element_types: list[str],
    captions: list[str],
    pages: list[int],
    original_text: str,
) -> str:
    lines = [
        f"Paper: {heading_path[0]}",
        f"Section: {' > '.join(heading_path)}",
        f"Element: {', '.join(element_types)}",
    ]
    if captions:
        lines.append(f"Caption: {' '.join(captions)}")
    page_label = "Pages" if len(pages) > 1 else "Page"
    lines.append(f"{page_label} {', '.join(str(page) for page in pages)}")
    return "\n".join(lines) + "\n\n" + original_text


def _prepare_node(node: BaseNode, item_texts: dict[str, str]) -> None:
    """Enforce the citation contract and attach deterministic contextual text."""
    metadata = node.metadata
    document_id = metadata.get("source_document_id")
    if not document_id:
        raise CitationProvenanceError(f"Node {node.node_id} has no source document ID")

    pages, bboxes = _provenance_geometry(metadata)
    if not pages or not bboxes:
        raise CitationProvenanceError(f"Node {node.node_id} has no complete source geometry")

    heading_path = [heading for heading in metadata.get("headings") or [] if heading]
    if not heading_path:
        raise CitationProvenanceError(f"Node {node.node_id} has no heading path")

    doc_items = [item for item in metadata.get("doc_items", []) if isinstance(item, dict)]
    element_types = sorted({str(item.get("label")) for item in doc_items if item.get("label")})
    item_refs = [str(item["self_ref"]) for item in doc_items if item.get("self_ref")]
    captions = [str(caption) for caption in metadata.get("captions") or [] if caption]
    citation_spans = _citation_spans(metadata, item_texts)

    original_text = node.get_content()
    node.set_content(_contextual_text(heading_path, element_types, captions, pages, original_text))

    source_relationship = node.relationships.get(NodeRelationship.SOURCE)
    if isinstance(source_relationship, RelatedNodeInfo):
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
            node_id=str(document_id),
            node_type=source_relationship.node_type,
            metadata=source_relationship.metadata,
            hash=source_relationship.hash,
        )

    node.metadata = {
        "document_id": str(document_id),
        "source_path": str(metadata.get("source_path", "")),
        "page_numbers": json.dumps(pages),
        "bboxes": json.dumps(bboxes),
        "heading_path": json.dumps(heading_path),
        "element_types": json.dumps(element_types),
        "captions": json.dumps(captions),
        "doc_item_refs": json.dumps(item_refs),
        "citation_spans": json.dumps(citation_spans),
        "original_text": original_text,
    }


def _has_heading_path(node: BaseNode) -> bool:
    return any(heading for heading in node.metadata.get("headings") or [])


def _keep_navigable_nodes(nodes: list[BaseNode]) -> list[BaseNode]:
    if not any(_has_heading_path(node) for node in nodes):
        return nodes
    return [node for node in nodes if _has_heading_path(node)]


def ingest_pdf(source: Path, dependencies: IngestDependencies | None = None) -> int:
    """Parse, chunk, contextualize, embed, and index one PDF. Return the chunk count."""
    active_dependencies = dependencies or IngestDependencies()
    documents = parse_pdf(source, reader=active_dependencies.reader)
    nodes = chunk_documents(
        documents, node_parser=active_dependencies.node_parser or _create_node_parser()
    )
    nodes = _keep_navigable_nodes(nodes)
    item_texts_by_document = {
        document.doc_id: _document_item_texts(document) for document in documents
    }
    for node in nodes:
        source_relationship = node.relationships.get(NodeRelationship.SOURCE)
        item_texts = (
            item_texts_by_document.get(source_relationship.node_id, {})
            if isinstance(source_relationship, RelatedNodeInfo)
            else {}
        )
        _prepare_node(node, item_texts)

    texts = [node.get_content() for node in nodes]
    vectors = (active_dependencies.embedder or OpenAIEmbedder()).embed(texts)
    for node, vector in zip(nodes, vectors, strict=True):
        node.embedding = vector

    active_store = active_dependencies.store or LanceDBNodeStore(DEFAULT_INDEX_DIR)
    active_store.add(nodes)
    if active_dependencies.finalize:
        active_store.finalize()
    return len(nodes)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry: ingest one PDF or every PDF in a directory into the index."""
    load_dotenv(Path.cwd() / ".env")
    argument_parser = argparse.ArgumentParser(
        prog="live_long_rnd.ingest",
        description="Ingest corpus PDFs into the LanceDB hybrid index.",
    )
    argument_parser.add_argument("source", type=Path, help="One PDF or a directory of PDFs.")
    argument_parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="LanceDB persistence directory.",
    )
    args = argument_parser.parse_args(argv)

    if args.source.is_dir():
        sources = sorted(args.source.glob("*.pdf"))
    elif args.source.is_file():
        sources = [args.source]
    else:
        argument_parser.error(f"{args.source} is not a file or directory")
    if not sources:
        argument_parser.error(f"{args.source} contains no PDFs")

    args.index_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{args.index_dir.name}-build-",
        dir=args.index_dir.parent,
    ) as temporary_dir:
        staged_index = Path(temporary_dir) / "index"
        embedder = OpenAIEmbedder()
        store = LanceDBNodeStore(staged_index)
        reader = create_reader()
        total = 0
        for pdf in sources:
            count = ingest_pdf(
                pdf,
                IngestDependencies(
                    reader=reader,
                    embedder=embedder,
                    store=store,
                    finalize=False,
                ),
            )
            total += count
            print(f"{pdf.name}: {count} chunks")
        store.finalize()
        del store
        _replace_index(staged_index, args.index_dir)
    print(f"Indexed {total} chunks from {len(sources)} document(s) into {args.index_dir}")
    return 0


def _replace_index(staged_index: Path, index_dir: Path) -> None:
    """Replace one complete index while preserving the prior index on failure."""
    backup = index_dir.with_name(f".{index_dir.name}-backup-{uuid4().hex}")
    moved_existing = False
    try:
        if index_dir.exists():
            index_dir.replace(backup)
            moved_existing = True
        staged_index.replace(index_dir)
    except Exception:
        if moved_existing and backup.exists() and not index_dir.exists():
            backup.replace(index_dir)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
