"""Deterministic embeddings for the committed retrieval fixture."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

GOLDEN_EMBEDDING_DIMENSIONS = 3_072
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class GoldenFixtureEmbedder:
    """Stable replacement for the embedding provider during fixture generation."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [_vector(text) for text in texts]


def _vector(text: str) -> list[float]:
    vector = [0.0] * GOLDEN_EMBEDDING_DIMENSIONS
    for term in _TOKEN_PATTERN.findall(text.casefold()):
        digest = hashlib.sha256(term.encode()).digest()
        vector[int.from_bytes(digest[:2]) % GOLDEN_EMBEDDING_DIMENSIONS] += 1.0
    magnitude = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / magnitude for value in vector]
