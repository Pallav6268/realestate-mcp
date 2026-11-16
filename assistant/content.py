"""Turn MCP tool results (a string or a list of content blocks) back into data."""

from __future__ import annotations

import json
from typing import Any


def text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(text_of(c) for c in content)
    if isinstance(content, dict):
        return str(content.get("text", ""))
    return str(getattr(content, "text", content))


def parse(content: Any) -> Any:
    text = text_of(content)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text
