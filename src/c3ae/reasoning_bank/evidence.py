"""Evidence Pack management."""

from __future__ import annotations

from c3ae.storage.sqlite_store import SQLiteStore
from c3ae.types import EvidencePack


class EvidenceManager:
    """CRUD for evidence packs that anchor reasoning entries."""

    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    def create(self, claim: str, sources: list[str],
               confidence: float = 0.0, reasoning: str = "") -> EvidencePack:
        pack = EvidencePack(
            claim=claim,
            sources=sources,
            confidence=confidence,
            reasoning=reasoning,
        )
        self.store.insert_evidence_pack(pack)
        return pack

    def get(self, pack_id: str) -> EvidencePack | None:
        return self.store.get_evidence_pack(pack_id)

    def validate(self, pack: EvidencePack) -> list[str]:
        """Return list of validation issues (empty = valid)."""
        issues = []
        if not pack.claim.strip():
            issues.append("Evidence pack must have a claim")
        if not pack.sources:
            issues.append("Evidence pack must have at least one source")
        if not 0.0 <= pack.confidence <= 1.0:
            issues.append("Confidence must be between 0.0 and 1.0")
        return issues
