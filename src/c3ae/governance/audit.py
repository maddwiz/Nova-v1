"""Audit log writer/reader."""

from __future__ import annotations

from c3ae.storage.sqlite_store import SQLiteStore
from c3ae.types import AuditEvent


class AuditLog:
    """Append-only audit trail for all governance-relevant operations."""

    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    def log(self, action: str, target_type: str, target_id: str,
            detail: str = "", outcome: str = "ok") -> AuditEvent:
        event = AuditEvent(
            action=action,
            target_type=target_type,
            target_id=target_id,
            detail=detail,
            outcome=outcome,
        )
        try:
            self.store.insert_audit_event(event)
        except Exception:
            pass  # Audit is best-effort — never crash a request
        return event

    def log_write(self, target_type: str, target_id: str, detail: str = "") -> AuditEvent:
        return self.log("write", target_type, target_id, detail)

    def log_blocked(self, target_type: str, target_id: str, reason: str) -> AuditEvent:
        return self.log("write_blocked", target_type, target_id, reason, outcome="blocked")

    def log_search(self, query: str, result_count: int) -> AuditEvent:
        return self.log("search", "query", query, f"results={result_count}")

    def recent(self, limit: int = 100, target_type: str | None = None) -> list[AuditEvent]:
        return self.store.list_audit_events(limit=limit, target_type=target_type)
