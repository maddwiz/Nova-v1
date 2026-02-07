"""Guardian — write validation rules for governance gate."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from c3ae.config import GovernanceConfig
from c3ae.exceptions import GovernanceError
from c3ae.storage.sqlite_store import SQLiteStore
from c3ae.types import ReasoningEntry, SearchResult

if TYPE_CHECKING:
    from c3ae.embeddings.cache import EmbeddingCache
    from c3ae.embeddings.venice import VeniceEmbedder
    from c3ae.storage.faiss_store import FAISSStore


class Guardian:
    """Validates writes before they touch the database."""

    def __init__(
        self,
        store: SQLiteStore,
        config: GovernanceConfig | None = None,
        faiss_store: "FAISSStore | None" = None,
        embedder: "VeniceEmbedder | None" = None,
        embed_cache: "EmbeddingCache | None" = None,
    ) -> None:
        self.store = store
        self.config = config or GovernanceConfig()
        self.faiss = faiss_store
        self.embedder = embedder
        self.embed_cache = embed_cache

    def validate_reasoning_entry(self, entry: ReasoningEntry) -> list[str]:
        """Validate a reasoning entry (sync — keyword contradiction only)."""
        issues = []

        if not entry.title.strip():
            issues.append("Title must not be empty")

        if not entry.content.strip():
            issues.append("Content must not be empty")

        if len(entry.content.encode()) > self.config.max_entry_bytes:
            issues.append(
                f"Content exceeds max size ({len(entry.content.encode())} > {self.config.max_entry_bytes})"
            )

        if self.config.require_evidence and not entry.evidence_ids:
            issues.append("Evidence is required — provide at least one evidence_id")

        if self.config.contradiction_check:
            contradictions = self._check_contradictions_keyword(entry)
            issues.extend(contradictions)

        return issues

    async def validate_async(self, entry: ReasoningEntry) -> list[str]:
        """Full validation including semantic contradiction detection."""
        issues = self.validate_reasoning_entry(entry)

        if self.config.contradiction_check and self.faiss and self.embedder:
            semantic_issues = await self._check_contradictions_semantic(entry)
            issues.extend(semantic_issues)

        return issues

    def gate(self, entry: ReasoningEntry) -> None:
        """Block writes that fail validation. Raises GovernanceError."""
        issues = self.validate_reasoning_entry(entry)
        if issues:
            raise GovernanceError(
                f"Write blocked by governance: {'; '.join(issues)}"
            )

    async def gate_async(self, entry: ReasoningEntry) -> None:
        """Async gate with semantic contradiction detection."""
        issues = await self.validate_async(entry)
        if issues:
            raise GovernanceError(
                f"Write blocked by governance: {'; '.join(issues)}"
            )

    def validate_and_report(self, entry: ReasoningEntry) -> tuple[bool, list[str]]:
        """Validate and return (pass, issues)."""
        issues = self.validate_reasoning_entry(entry)
        return len(issues) == 0, issues

    async def validate_and_report_async(
        self, entry: ReasoningEntry,
    ) -> tuple[bool, list[str], list[str]]:
        """Async validate with semantic check.

        Returns (passed, blocking_issues, advisory_warnings).
        Blocking issues prevent the write. Warnings are logged but don't block.
        """
        blocking = self._validate_hard(entry)
        warnings: list[str] = []

        if self.config.contradiction_check:
            warnings.extend(self._check_contradictions_keyword(entry))
            if self.faiss and self.embedder:
                warnings.extend(await self._check_contradictions_semantic(entry))

        return len(blocking) == 0, blocking, warnings

    def _validate_hard(self, entry: ReasoningEntry) -> list[str]:
        """Hard validation rules that block writes."""
        issues = []
        if not entry.title.strip():
            issues.append("Title must not be empty")
        if not entry.content.strip():
            issues.append("Content must not be empty")
        if len(entry.content.encode()) > self.config.max_entry_bytes:
            issues.append(
                f"Content exceeds max size ({len(entry.content.encode())} > {self.config.max_entry_bytes})"
            )
        if self.config.require_evidence and not entry.evidence_ids:
            issues.append("Evidence is required — provide at least one evidence_id")
        return issues

    def _check_contradictions_keyword(self, entry: ReasoningEntry) -> list[str]:
        """Keyword-based contradiction detection using FTS5."""
        issues = []
        try:
            existing = self.store.search_reasoning_fts(entry.title, limit=5)
        except Exception:
            return issues

        for result in existing:
            existing_entry = self.store.get_reasoning_entry(result.id)
            if existing_entry and existing_entry.status.value == "active":
                if result.score > 5.0:
                    issues.append(
                        f"Potential contradiction with entry {result.id}: "
                        f"'{existing_entry.title}' (similarity {result.score:.2f}). "
                        f"Consider superseding instead."
                    )
        return issues

    async def _check_contradictions_semantic(self, entry: ReasoningEntry) -> list[str]:
        """Semantic contradiction detection using embeddings + cosine similarity.

        Embeds the new entry's content, searches FAISS for similar existing entries,
        and flags high-similarity matches as potential contradictions requiring review.
        """
        issues = []
        if not self.faiss or self.faiss.size == 0:
            return issues

        try:
            # Embed the new entry's title + content
            text = f"{entry.title}. {entry.content}"

            # Check cache first
            vec = None
            if self.embed_cache:
                vec = self.embed_cache.get(text)
            if vec is None:
                vec = await self.embedder.embed_single(text)
                if self.embed_cache:
                    self.embed_cache.put(text, vec)

            # Search FAISS for similar reasoning entries
            # We search chunks, but we're interested in reasoning bank overlaps
            hits = self.faiss.search(vec, top_k=5)

            for ext_id, cosine_score in hits:
                if cosine_score < 0.75:
                    continue  # Not similar enough to flag

                # Look up the chunk to see if it's from a reasoning entry
                chunk = self.store.get_chunk(ext_id)
                if not chunk:
                    continue

                # Search reasoning bank for entries with overlapping content
                try:
                    # Use first few words as FTS query
                    words = chunk.content.split()[:5]
                    fts_query = " ".join(words)
                    rb_hits = self.store.search_reasoning_fts(fts_query, limit=3)
                    for rb_hit in rb_hits:
                        existing = self.store.get_reasoning_entry(rb_hit.id)
                        if existing and existing.status.value == "active":
                            issues.append(
                                f"Semantic overlap (cosine={cosine_score:.3f}) with existing entry "
                                f"'{existing.title}' ({existing.id[:8]}). "
                                f"Review for contradiction or consider superseding."
                            )
                except Exception:
                    pass

        except Exception:
            pass  # Embedding failures shouldn't block writes

        return issues
