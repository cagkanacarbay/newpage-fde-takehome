"""Corpus document lookup: titles from MANIFEST.md and PDF file resolution.

Document ids are the corpus filename stems (`{document_id}.pdf`). The store
parses the manifest once at startup and only ever serves ids that correspond
to real PDF files directly inside the corpus directory.
"""

import re
from pathlib import Path

DEFAULT_CORPUS_DIR = Path("data/corpus/longevity")

_BACKTICKED_PDF = re.compile(r"`(?P<filename>[^`]+\.pdf)`")


def _prettify_slug(document_id: str) -> str:
    words = document_id.split("-")
    if words and words[0].isdigit():
        words = words[1:]
    return " ".join(words).capitalize()


class DocumentStore:
    """Titles and PDF paths for one corpus directory."""

    def __init__(self, corpus_dir: Path) -> None:
        self._corpus_dir = corpus_dir
        self._pdf_paths = {
            path.stem: path for path in sorted(corpus_dir.glob("*.pdf")) if path.is_file()
        }
        self._titles = self._parse_manifest(corpus_dir / "MANIFEST.md")

    @staticmethod
    def _parse_manifest(manifest_path: Path) -> dict[str, str]:
        titles: dict[str, str] = {}
        if not manifest_path.is_file():
            return titles
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            for index, cell in enumerate(cells):
                match = _BACKTICKED_PDF.search(cell)
                if match and index + 1 < len(cells):
                    titles[Path(match["filename"]).stem] = cells[index + 1]
                    break
        return titles

    def title_for(self, document_id: str) -> str | None:
        """The manifest title, or a prettified slug; None for unknown ids."""
        if document_id not in self._pdf_paths:
            return None
        return self._titles.get(document_id) or _prettify_slug(document_id)

    def pdf_path_for(self, document_id: str) -> Path | None:
        """The PDF path for a known id; anything else (incl. traversal) is None."""
        path = self._pdf_paths.get(document_id)
        if path is None:
            return None
        resolved = path.resolve()
        if resolved.parent != self._corpus_dir.resolve() or resolved.suffix != ".pdf":
            return None
        return resolved
