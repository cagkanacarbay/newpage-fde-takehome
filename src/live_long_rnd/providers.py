"""Shared provider configuration for the Live Long R&D model adapters."""

import os
from collections.abc import Mapping

DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def gemini_base_url(environ: Mapping[str, str] | None = None) -> str:
    """Return the configured OpenAI-compatible Gemini endpoint."""
    settings = os.environ if environ is None else environ
    return settings.get("GEMINI_BASE_URL", "").strip() or GEMINI_OPENAI_BASE_URL
