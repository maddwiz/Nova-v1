"""ReasoningBank — append-only, evidence-anchored knowledge store."""

from __future__ import annotations

from c3ae.storage.sqlite_store import SQLiteStore
from c3ae.types import ReasoningEntry, SearchResult, EntryStatus


class ReasoningBank:
    """Append-only knowledge bank with FTS search and supersession chains."""

    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    def add(self, title: str, content: str,
            tags: list[str] | None = None,
            evidence_ids: list[str] | None = None,
            session_id: str | None = None,
            metadata: dict | None = None) -> ReasoningEntry:
        """Add a new reasoning entry."""
        entry = ReasoningEntry(
            title=title,
            content=content,
            tags=tags or [],
            evidence_ids=evidence_ids or [],
            session_id=session_id,
            metadata=metadata or {},
        )
        self.store.insert_reasoning_entry(entry)
        return entry

    def supersede(self, old_id: str, new_title: str, new_content: str,
                  tags: list[str] | None = None,
                  evidence_ids: list[str] | None = None,
                  session_id: str | None = None) -> ReasoningEntry:
        """Create a new entry that supersedes an existing one."""
        old = self.store.get_reasoning_entry(old_id)
        if old is None:
            raise ValueError(f"Entry {old_id} not found")
        new_entry = ReasoningEntry(
            title=new_title,
            content=new_content,
            tags=tags or old.tags,
            evidence_ids=evidence_ids or [],
            session_id=session_id,
        )
        self.store.supersede_reasoning_entry(old_id, new_entry)
        return new_entry

    def retract(self, entry_id: str) -> None:
        """Mark an entry as retracted (soft delete)."""
        entry = self.store.get_reasoning_entry(entry_id)
        if entry is None:
            raise ValueError(f"Entry {entry_id} not found")
        # Use supersede mechanics to mark as retracted
        self.store._conn.execute(
            "UPDATE reasoning_bank SET status=? WHERE id=?",
            (EntryStatus.RETRACTED.value, entry_id),
        )
        self.store._conn.commit()

    def get(self, entry_id: str) -> ReasoningEntry | None:
        return self.store.get_reasoning_entry(entry_id)

    def list_active(self, limit: int = 100) -> list[ReasoningEntry]:
        return self.store.list_reasoning_entries(status="active", limit=limit)

    def search(self, query: str, limit: int = 20) -> list[SearchResult]:
        return self.store.search_reasoning_fts(query, limit=limit)

    def get_chain(self, entry_id: str) -> list[ReasoningEntry]:
        """Follow the supersession chain from an entry."""
        chain = []
        current = self.store.get_reasoning_entry(entry_id)
        while current:
            chain.append(current)
            if current.superseded_by:
                current = self.store.get_reasoning_entry(current.superseded_by)
            else:
                break
        return chain
