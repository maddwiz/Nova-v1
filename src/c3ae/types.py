"""Core data types for C3/Ae."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return uuid4().hex


class MemoryTier(str, Enum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


class EntryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"


class Chunk(BaseModel):
    id: str = Field(default_factory=_uuid)
    source_id: str = ""
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_now)


class EvidencePack(BaseModel):
    id: str = Field(default_factory=_uuid)
    claim: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_now)


class ReasoningEntry(BaseModel):
    id: str = Field(default_factory=_uuid)
    title: str
    content: str
    tags: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    status: EntryStatus = EntryStatus.ACTIVE
    superseded_by: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_now)


class CarryOverSummary(BaseModel):
    id: str = Field(default_factory=_uuid)
    session_id: str
    sequence: int = 0
    summary: str
    key_facts: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_now)


class SkillCapsule(BaseModel):
    id: str = Field(default_factory=_uuid)
    name: str
    description: str
    procedure: str
    tags: list[str] = Field(default_factory=list)
    version: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_now)


class AuditEvent(BaseModel):
    id: str = Field(default_factory=_uuid)
    action: str
    target_type: str
    target_id: str
    detail: str = ""
    outcome: str = "ok"
    created_at: datetime = Field(default_factory=_now)


class SearchResult(BaseModel):
    id: str
    content: str
    score: float
    source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class VaultFile(BaseModel):
    id: str = Field(default_factory=_uuid)
    path: str
    content_hash: str
    size_bytes: int = 0
    mime_type: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_now)
