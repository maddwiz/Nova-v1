"""
N5: LangChain Memory Adapter.

Converts LangChain ChatMessageHistory to USC packets and back.
Only importable if langchain-core is installed.
"""
from __future__ import annotations

from typing import Any, List, Tuple

import zstandard as zstd


class LangChainMemoryAdapter:
    """Adapter for LangChain ChatMessageHistory."""

    name: str = "langchain"

    def ingest(self, messages: List[Tuple[str, str]]) -> bytes:
        """
        Convert a list of (role, content) message tuples to USC packet.

        Args:
            messages: List of (role, content) tuples, e.g. [("human", "hello"), ("ai", "hi")]
        """
        lines = []
        for role, content in messages:
            lines.append(f"[{role}] {content}")
        text = "\n".join(lines)
        cctx = zstd.ZstdCompressor(level=10)
        return cctx.compress(text.encode("utf-8"))

    def retrieve(self, blob: bytes) -> List[Tuple[str, str]]:
        """Decode USC blob back to list of (role, content) tuples."""
        dctx = zstd.ZstdDecompressor()
        text = dctx.decompress(blob).decode("utf-8")
        messages = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("[") and "] " in line:
                bracket_end = line.index("] ")
                role = line[1:bracket_end]
                content = line[bracket_end + 2:]
                messages.append((role, content))
            else:
                messages.append(("unknown", line))
        return messages

    def ingest_langchain_messages(self, messages: Any) -> bytes:
        """
        Ingest LangChain BaseMessage objects directly.
        Import at function level to avoid hard dependency.
        """
        from langchain_core.messages import BaseMessage
        tuples = []
        for msg in messages:
            if hasattr(msg, 'type'):
                tuples.append((msg.type, msg.content))
            else:
                tuples.append(("unknown", str(msg)))
        return self.ingest(tuples)

    def retrieve_langchain_messages(self, blob: bytes) -> Any:
        """
        Decode to LangChain BaseMessage objects.
        Import at function level to avoid hard dependency.
        """
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

        tuples = self.retrieve(blob)
        messages = []
        for role, content in tuples:
            if role == "human":
                messages.append(HumanMessage(content=content))
            elif role == "ai":
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))
        return messages
