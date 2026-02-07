"""Skill Capsule registry — reusable procedure modules."""

from __future__ import annotations

from c3ae.storage.sqlite_store import SQLiteStore
from c3ae.types import SearchResult, SkillCapsule


class SkillRegistry:
    """CRUD and search for skill capsules."""

    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    def register(self, name: str, description: str, procedure: str,
                 tags: list[str] | None = None,
                 metadata: dict | None = None) -> SkillCapsule:
        capsule = SkillCapsule(
            name=name,
            description=description,
            procedure=procedure,
            tags=tags or [],
            metadata=metadata or {},
        )
        self.store.insert_skill_capsule(capsule)
        return capsule

    def get(self, capsule_id: str) -> SkillCapsule | None:
        return self.store.get_skill_capsule(capsule_id)

    def list_all(self, limit: int = 100) -> list[SkillCapsule]:
        return self.store.list_skill_capsules(limit=limit)

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        return self.store.search_skills_fts(query, limit=limit)

    def update_procedure(self, capsule_id: str, new_procedure: str) -> SkillCapsule:
        """Update a capsule's procedure, incrementing version."""
        capsule = self.store.get_skill_capsule(capsule_id)
        if capsule is None:
            raise ValueError(f"Capsule {capsule_id} not found")
        self.store._conn.execute(
            "UPDATE skill_capsules SET procedure=?, version=? WHERE id=?",
            (new_procedure, capsule.version + 1, capsule_id),
        )
        self.store._conn.commit()
        return self.store.get_skill_capsule(capsule_id)
