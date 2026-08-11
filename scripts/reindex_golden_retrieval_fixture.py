"""Build the committed retrieval fixture from the canonical PDF corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from live_long_rnd.golden_embeddings import GOLDEN_EMBEDDING_DIMENSIONS, GoldenFixtureEmbedder
from live_long_rnd.ingest import IngestDependencies, LanceDBNodeStore, ingest_pdf
from live_long_rnd.parsing import create_reader

TABLE_NAME = "chunks"
FIXTURE_VERSION = 1


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
    store = LanceDBNodeStore(args.output, table_name=TABLE_NAME)
    reader = create_reader()
    embedder = GoldenFixtureEmbedder()
    row_count = sum(
        ingest_pdf(
            source,
            IngestDependencies(
                reader=reader,
                embedder=embedder,
                store=store,
                finalize=False,
            ),
        )
        for source in sources
    )
    store.finalize()
    manifest = {
        "fixture_version": FIXTURE_VERSION,
        "table_name": TABLE_NAME,
        "vector_dimensions": GOLDEN_EMBEDDING_DIMENSIONS,
        "source_sha256": {source.name: _sha256(source) for source in sources},
        "row_count": row_count,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0
def _sha256(source: Path) -> str:
    return hashlib.sha256(source.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
