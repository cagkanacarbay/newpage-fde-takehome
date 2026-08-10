"""Ingest corpus PDFs into a hybrid LanceDB index with full citation provenance."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import tiktoken
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
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
        table.create_index(self._store.text_key, config=FTS(), replace=True)


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


def _prepare_node(node: BaseNode) -> None:
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

    original_text = node.get_content()
    node.set_content(_contextual_text(heading_path, element_types, captions, pages, original_text))

    source_relationship = node.relationships.get(NodeRelationship.SOURCE)
    if isinstance(source_relationship, RelatedNodeInfo):
        source_relationship.node_id = str(document_id)

    node.metadata = {
        "document_id": str(document_id),
        "source_path": str(metadata.get("source_path", "")),
        "page_numbers": json.dumps(pages),
        "bboxes": json.dumps(bboxes),
        "heading_path": json.dumps(heading_path),
        "element_types": json.dumps(element_types),
        "captions": json.dumps(captions),
        "doc_item_refs": json.dumps(item_refs),
        "original_text": original_text,
    }


def _has_heading_path(node: BaseNode) -> bool:
    return any(heading for heading in node.metadata.get("headings") or [])


def _keep_navigable_nodes(nodes: list[BaseNode]) -> list[BaseNode]:
    if not any(_has_heading_path(node) for node in nodes):
        return nodes
    return [node for node in nodes if _has_heading_path(node)]


def ingest_pdf(
    source: Path,
    *,
    reader: DocumentReader | None = None,
    node_parser: NodeParser | None = None,
    embedder: Embedder | None = None,
    store: NodeStore | None = None,
    finalize: bool = True,
) -> int:
    """Parse, chunk, contextualize, embed, and index one PDF. Return the chunk count."""
    documents = parse_pdf(source, reader=reader)
    nodes = chunk_documents(documents, node_parser=node_parser or _create_node_parser())
    nodes = _keep_navigable_nodes(nodes)
    for node in nodes:
        _prepare_node(node)

    texts = [node.get_content() for node in nodes]
    vectors = (embedder or OpenAIEmbedder()).embed(texts)
    for node, vector in zip(nodes, vectors, strict=True):
        node.embedding = vector

    active_store = store or LanceDBNodeStore(DEFAULT_INDEX_DIR)
    active_store.add(nodes)
    if finalize:
        active_store.finalize()
    return len(nodes)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry: ingest one PDF or every PDF in a directory into the index."""
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

    embedder = OpenAIEmbedder()
    store = LanceDBNodeStore(args.index_dir)
    reader = create_reader()
    total = 0
    for pdf in sources:
        count = ingest_pdf(pdf, reader=reader, embedder=embedder, store=store, finalize=False)
        total += count
        print(f"{pdf.name}: {count} chunks")
    store.finalize()
    print(f"Indexed {total} chunks from {len(sources)} document(s) into {args.index_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
