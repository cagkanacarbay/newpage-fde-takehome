"""Validate the corpus and indexed retrieval assets before an image release."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import lancedb

from live_long_rnd.embeddings import EMBEDDING_DIMENSIONS
from live_long_rnd.index_config import DEFAULT_INDEX_DIR, DEFAULT_TABLE_NAME

EXPECTED_DOCUMENTS = 27
EXPECTED_ROWS = 696
REQUIRED_BBOX_KEYS = {"page", "l", "t", "r", "b"}


class ReleaseAssetError(RuntimeError):
    """Report an incomplete or inconsistent release asset set."""


def validate_release_assets(
    corpus_dir: Path,
    index_dir: Path,
    *,
    expected_documents: int = EXPECTED_DOCUMENTS,
    expected_rows: int = EXPECTED_ROWS,
) -> None:
    """Require the complete corpus, dense vectors, BM25, and citation metadata."""
    pdfs = sorted(corpus_dir.glob("*.pdf"))
    if len(pdfs) != expected_documents:
        raise ReleaseAssetError(
            f"Expected {expected_documents} PDFs in {corpus_dir}, found {len(pdfs)}."
        )

    try:
        table = lancedb.connect(str(index_dir)).open_table(DEFAULT_TABLE_NAME)
    except Exception as error:
        raise ReleaseAssetError(
            f"Could not open LanceDB table {DEFAULT_TABLE_NAME!r} in {index_dir}."
        ) from error

    row_count = table.count_rows()
    if row_count != expected_rows:
        raise ReleaseAssetError(f"Expected {expected_rows} indexed rows, found {row_count}.")

    vector_type = table.schema.field("vector").type
    if getattr(vector_type, "list_size", None) != EMBEDDING_DIMENSIONS:
        raise ReleaseAssetError(
            f"Expected {EMBEDDING_DIMENSIONS}-dimension dense vectors, found {vector_type}."
        )

    fts_indexes = [
        index
        for index in table.list_indices()
        if index.index_type == "FTS" and index.columns == ["text"]
    ]
    if not fts_indexes:
        raise ReleaseAssetError("The LanceDB text column has no BM25 index.")
    if any(
        index.num_indexed_rows != expected_rows or index.num_unindexed_rows != 0
        for index in fts_indexes
    ):
        raise ReleaseAssetError("The LanceDB BM25 index does not cover every row.")

    corpus_document_ids = {pdf.stem for pdf in pdfs}
    indexed_document_ids: set[str] = set()
    metadata_rows: list[dict[str, Any]] = table.to_arrow().column("metadata").to_pylist()
    for row_number, metadata in enumerate(metadata_rows, start=1):
        document_id = str(metadata.get("document_id", ""))
        indexed_document_ids.add(document_id)
        pages = _json_list(metadata.get("page_numbers"), row_number, "page_numbers")
        bboxes = _json_list(metadata.get("bboxes"), row_number, "bboxes")
        headings = _json_list(metadata.get("heading_path"), row_number, "heading_path")
        if not pages or not all(isinstance(page, int) and page >= 1 for page in pages):
            raise ReleaseAssetError(f"Index row {row_number} has invalid citation pages.")
        if not bboxes or not all(
            isinstance(bbox, dict)
            and set(bbox) == REQUIRED_BBOX_KEYS
            and isinstance(bbox.get("page"), int)
            and bbox["page"] >= 1
            for bbox in bboxes
        ):
            raise ReleaseAssetError(f"Index row {row_number} has invalid highlight metadata.")
        if not headings or not all(isinstance(heading, str) and heading for heading in headings):
            raise ReleaseAssetError(f"Index row {row_number} has no citation heading path.")
        if not str(metadata.get("original_text", "")).strip():
            raise ReleaseAssetError(f"Index row {row_number} has no original citation text.")

    if indexed_document_ids != corpus_document_ids:
        missing = sorted(corpus_document_ids - indexed_document_ids)
        unexpected = sorted(indexed_document_ids - corpus_document_ids)
        raise ReleaseAssetError(
            f"Corpus and index document IDs differ. Missing: {missing}; unexpected: {unexpected}."
        )


def _json_list(value: object, row_number: int, field: str) -> list[Any]:
    if not isinstance(value, str):
        raise ReleaseAssetError(f"Index row {row_number} has invalid {field} metadata.")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ReleaseAssetError(f"Index row {row_number} has invalid {field} JSON.") from error
    if not isinstance(parsed, list):
        raise ReleaseAssetError(f"Index row {row_number} has non-list {field} metadata.")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    """Validate release assets from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus/longevity"))
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    args = parser.parse_args(argv)
    validate_release_assets(args.corpus_dir, args.index_dir)
    print(f"Validated {EXPECTED_DOCUMENTS} PDFs and {EXPECTED_ROWS} indexed rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
