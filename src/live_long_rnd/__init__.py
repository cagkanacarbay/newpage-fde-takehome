"""Live Long R&D document assistant."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from live_long_rnd.parsing import (
        CitationProvenanceError,
        EmptyParseResultError,
        chunk_documents,
        parse_pdf,
    )

__all__ = [
    "CitationProvenanceError",
    "EmptyParseResultError",
    "chunk_documents",
    "parse_pdf",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return getattr(import_module("live_long_rnd.parsing"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
