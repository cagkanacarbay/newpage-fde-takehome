import json
from collections.abc import Mapping


def encode_sse(event: Mapping[str, object]) -> str:
    """Encode one JSON value as an SSE data event."""
    payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    return f"data: {payload}\n\n"
