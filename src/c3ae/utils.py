"""Shared utilities."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

import orjson


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def json_dumps(obj: Any) -> str:
    return orjson.dumps(obj).decode()


def json_loads(data: str | bytes) -> Any:
    return orjson.loads(data)


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        # Try to break at a paragraph or sentence boundary
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", " "]:
                idx = text.rfind(sep, start + max_chars // 2, end)
                if idx != -1:
                    end = idx + len(sep)
                    break
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]


def iso_str(dt: datetime) -> str:
    return dt.isoformat()


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)
