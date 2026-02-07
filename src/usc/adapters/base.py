"""
N5: Framework Adapter Protocol.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class USCAdapter(Protocol):
    """Protocol for framework-specific adapters."""

    name: str

    def ingest(self, data: Any) -> bytes:
        """Convert framework-specific data to USC-encodable text, then encode."""
        ...

    def retrieve(self, blob: bytes) -> Any:
        """Decode USC blob and convert back to framework-specific format."""
        ...
