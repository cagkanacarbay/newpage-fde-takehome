"""Build the committed retrieval fixture from the canonical PDF corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import lancedb
from lancedb.index import FTS

VECTOR_DIMENSIONS = 128
TABLE_NAME = "chunks"
FIXTURE_VERSION = 1
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def main(argv: list[str] | None = None) -> int:
    """Create a page-level LanceDB fixture after explicit corpus approval."""
    parser = argparse.ArgumentParser(prog="reindex-golden-retrieval-fixture")
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus/longevity"))
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/golden_retrieval_index"))
    args = parser.parse_args(argv)
    if os.environ.get("ALLOW_CORPUS_REINDEX") != "1":
        parser.error("Set ALLOW_CORPUS_REINDEX=1 to rebuild the golden corpus fixture.")
    sources = sorted(args.corpus_dir.glob("*.pdf"))
    if len(sources) != 27:
        parser.error(f"Expected 27 canonical PDFs in {args.corpus_dir}, found {len(sources)}.")
    if args.output.exists():
        shutil.rmtree(args.output)
    rows = [row for source in sources for row in _rows_from_pdf(source)]
    table = lancedb.connect(str(args.output)).create_table(TABLE_NAME, data=rows)
    table.create_index("text", config=FTS(with_position=True), replace=True)
    manifest = {
        "fixture_version": FIXTURE_VERSION,
        "table_name": TABLE_NAME,
        "vector_dimensions": VECTOR_DIMENSIONS,
        "source_sha256": {source.name: _sha256(source) for source in sources},
        "row_count": len(rows),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def _rows_from_pdf(source: Path) -> list[dict[str, Any]]:
    text = subprocess.run(
        ["pdftotext", "-layout", str(source), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    document_id = source.stem
    pages = [page for page, value in enumerate(text.split("\f"), start=1) if value.strip()]
    original_text = text[:20_000]
    return [
        {
            "id": f"{document_id}-evidence",
            "vector": _vector(original_text),
            "text": original_text,
            "metadata": {
                "document_id": document_id,
                "page_numbers": json.dumps(pages),
                "bboxes": json.dumps([{"page": pages[0], "l": 0.0, "t": 0.0, "r": 1.0, "b": 1.0}]),
                "heading_path": json.dumps([document_id]),
                "original_text": original_text,
            },
        }
    ]


def _vector(text: str) -> list[float]:
    vector = [0.0] * VECTOR_DIMENSIONS
    for term in _TOKEN_PATTERN.findall(text.casefold()):
        digest = hashlib.sha256(term.encode()).digest()
        vector[int.from_bytes(digest[:2]) % VECTOR_DIMENSIONS] += 1.0
    magnitude = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / magnitude for value in vector]


def _sha256(source: Path) -> str:
    return hashlib.sha256(source.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
