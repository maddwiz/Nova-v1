"""RLM Reader — Recursive out-of-core reader producing Evidence Packs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from c3ae.memory_spine.spine import MemorySpine
from c3ae.types import EvidencePack
from c3ae.utils import chunk_text


@dataclass
class ReadBudget:
    """Controls how much the reader processes."""
    max_chunks: int = 100
    max_depth: int = 3
    chunks_processed: int = 0
    depth: int = 0

    @property
    def exhausted(self) -> bool:
        return self.chunks_processed >= self.max_chunks or self.depth >= self.max_depth

    def consume(self, n: int = 1) -> None:
        self.chunks_processed += n


@dataclass
class ReadResult:
    """Output of the RLM reader."""
    chunks_processed: int
    evidence_packs: list[EvidencePack] = field(default_factory=list)
    summaries: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class RLMReader:
    """Recursive paging reader for large documents.

    Processes documents in chunks, extracting evidence packs at each level.
    Budget enforcement prevents unbounded processing.
    """

    def __init__(self, spine: MemorySpine) -> None:
        self.spine = spine

    async def read_text(self, text: str, topic: str = "",
                        budget: ReadBudget | None = None) -> ReadResult:
        """Read a text document, producing evidence packs."""
        budget = budget or ReadBudget()
        chunks = chunk_text(text)
        return await self._process_chunks(chunks, topic, budget)

    async def read_file(self, file_path: Path, topic: str = "",
                        budget: ReadBudget | None = None) -> ReadResult:
        """Read a file, producing evidence packs."""
        text = file_path.read_text(errors="replace")
        return await self.read_text(text, topic, budget)

    async def _process_chunks(self, chunks: list[str], topic: str,
                               budget: ReadBudget) -> ReadResult:
        """Process chunks with budget enforcement."""
        result = ReadResult(chunks_processed=0)

        for chunk in chunks:
            if budget.exhausted:
                break

            # Ingest chunk into memory
            chunk_ids = await self.spine.ingest_text(chunk, source_id=f"rlm:{topic}")
            budget.consume(1)
            result.chunks_processed += 1

            # Extract claims from chunk (simplified — in production, LLM would do this)
            claims = self._extract_claims(chunk, topic)
            for claim, reasoning in claims:
                pack = self.spine.add_evidence(
                    claim=claim,
                    sources=[f"chunk:{chunk_ids[0]}" if chunk_ids else "inline"],
                    confidence=0.5,
                    reasoning=reasoning,
                )
                result.evidence_packs.append(pack)

        result.metadata = {
            "topic": topic,
            "total_chunks": len(chunks),
            "budget_remaining": budget.max_chunks - budget.chunks_processed,
        }
        return result

    @staticmethod
    def _extract_claims(chunk: str, topic: str) -> list[tuple[str, str]]:
        """Extract claims from a chunk (heuristic — LLM integration point).

        Returns list of (claim, reasoning) tuples.
        """
        claims = []
        sentences = [s.strip() for s in chunk.split(".") if len(s.strip()) > 20]
        # Take up to 3 key sentences as claims
        for s in sentences[:3]:
            claim = s.strip().rstrip(".")
            reasoning = f"Extracted from text about '{topic}': '{claim[:100]}...'" if topic else f"Direct extract: '{claim[:100]}...'"
            claims.append((claim, reasoning))
        return claims
