"""Memory Spine — orchestrator for Hot/Warm/Cold tiers + retrieval + governance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from c3ae.config import Config
from c3ae.cos.cos import COSManager
from c3ae.embeddings.cache import EmbeddingCache
from c3ae.embeddings.venice import VeniceEmbedder
from c3ae.exceptions import GovernanceError
from c3ae.governance.audit import AuditLog
from c3ae.governance.guardian import Guardian
from c3ae.reasoning_bank.bank import ReasoningBank
from c3ae.reasoning_bank.evidence import EvidenceManager
from c3ae.retrieval.hybrid import HybridSearch
from c3ae.retrieval.keyword import KeywordSearch
from c3ae.retrieval.vector import VectorSearch
from c3ae.skill_capsules.registry import SkillRegistry
from c3ae.storage.faiss_store import FAISSStore
from c3ae.storage.sqlite_store import SQLiteStore
from c3ae.storage.vault import Vault
from c3ae.usc_bridge.compressed_vault import CompressedVault
from c3ae.types import (
    Chunk,
    EvidencePack,
    MemoryTier,
    ReasoningEntry,
    SearchResult,
    SkillCapsule,
)
from c3ae.utils import chunk_text


class MemorySpine:
    """Central orchestrator wiring all memory subsystems."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.config.ensure_dirs()

        # Storage backends
        self.sqlite = SQLiteStore(self.config.db_path)
        self.faiss = FAISSStore(
            dims=self.config.venice.embedding_dims,
            faiss_dir=self.config.faiss_dir,
            ivf_threshold=self.config.retrieval.faiss_ivf_threshold,
        )
        self.vault = CompressedVault(self.config.vault_dir)

        # Subsystems
        self.embedder = VeniceEmbedder(self.config.venice)
        self.embed_cache = EmbeddingCache(self.sqlite, self.config.venice.embedding_model)
        self.keyword_search = KeywordSearch(self.sqlite)
        self.vector_search = VectorSearch(self.faiss, self.sqlite)
        self.hybrid_search = HybridSearch(
            self.keyword_search, self.vector_search, self.config.retrieval
        )
        self.cos = COSManager(self.sqlite)
        self.bank = ReasoningBank(self.sqlite)
        self.evidence = EvidenceManager(self.sqlite)
        self.skills = SkillRegistry(self.sqlite)
        self.guardian = Guardian(
            self.sqlite, self.config.governance,
            faiss_store=self.faiss,
            embedder=self.embedder,
            embed_cache=self.embed_cache,
        )
        self.audit = AuditLog(self.sqlite)

        # Hot tier: in-memory cache for recent items
        self._hot_cache: dict[str, Any] = {}

        # USC cognitive deduplication store (lazy init)
        self._cogstore = None

    # --- Ingest ---

    async def ingest_text(self, text: str, source_id: str = "",
                          metadata: dict[str, Any] | None = None) -> list[str]:
        """Chunk text, embed, and index. Returns chunk IDs."""
        chunks_text = chunk_text(text)
        chunk_ids = []
        for ct in chunks_text:
            chunk = Chunk(content=ct, source_id=source_id, metadata=metadata or {})
            self.sqlite.insert_chunk(chunk)
            chunk_ids.append(chunk.id)

        # Embed and index
        await self._embed_and_index(chunk_ids, chunks_text)
        self.audit.log_write("chunks", source_id or "inline", f"ingested {len(chunk_ids)} chunks")
        return chunk_ids

    async def ingest_file(self, file_path: Path, metadata: dict[str, Any] | None = None) -> list[str]:
        """Ingest a file from disk."""
        data = file_path.read_bytes()
        text = data.decode("utf-8", errors="replace")
        # Store in vault
        content_hash = self.vault.store_document(data, file_path.name, metadata)
        self.sqlite.insert_file(
            content_hash, str(file_path), content_hash, len(data),
            "", metadata,
        )
        return await self.ingest_text(text, source_id=content_hash, metadata=metadata)

    # --- Search ---

    async def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Hybrid search across all memory tiers."""
        top_k = top_k or self.config.retrieval.default_top_k

        # Try to get query embedding (may fail if no API key)
        query_vec = None
        try:
            query_vec = await self._embed_text(query)
        except Exception:
            pass  # Fall back to keyword-only

        results = self.hybrid_search.search(query, query_vector=query_vec, top_k=top_k)
        self.audit.log_search(query, len(results))
        return results

    def search_keyword(self, query: str, top_k: int = 20) -> list[SearchResult]:
        """Keyword-only search (synchronous, no embedding needed)."""
        results = self.keyword_search.search_all(query, limit=top_k)
        self.audit.log_search(query, len(results))
        return results

    # --- Reasoning Bank (governed) ---

    async def add_knowledge(self, title: str, content: str,
                            tags: list[str] | None = None,
                            evidence_ids: list[str] | None = None,
                            session_id: str | None = None,
                            bypass_governance: bool = False) -> ReasoningEntry:
        """Add a reasoning entry through the governance gate."""
        entry = ReasoningEntry(
            title=title, content=content,
            tags=tags or [], evidence_ids=evidence_ids or [],
            session_id=session_id,
        )
        if not bypass_governance:
            ok, issues, warnings = await self.guardian.validate_and_report_async(entry)
            if not ok:
                self.audit.log_blocked("reasoning_entry", entry.id, "; ".join(issues))
                raise GovernanceError(f"Write blocked: {'; '.join(issues)}")
            if warnings:
                self.audit.log("warning", "reasoning_entry", entry.id,
                               "; ".join(warnings))

        self.sqlite.insert_reasoning_entry(entry)

        # Also index as a chunk so it's discoverable via vector search
        # and semantic contradiction detection can find it in FAISS
        combined = f"{title}. {content}"
        chunk = Chunk(content=combined, source_id=entry.id, metadata={"type": "reasoning_entry"})
        self.sqlite.insert_chunk(chunk)
        await self._embed_and_index([chunk.id], [combined])

        self.audit.log_write("reasoning_entry", entry.id, title)
        return entry

    def supersede_knowledge(self, old_id: str, new_title: str, new_content: str,
                            tags: list[str] | None = None,
                            evidence_ids: list[str] | None = None) -> ReasoningEntry:
        entry = self.bank.supersede(old_id, new_title, new_content, tags, evidence_ids)
        self.audit.log_write("reasoning_entry", entry.id, f"supersedes {old_id}")
        return entry

    # --- Evidence ---

    def add_evidence(self, claim: str, sources: list[str],
                     confidence: float = 0.0, reasoning: str = "") -> EvidencePack:
        pack = self.evidence.create(claim, sources, confidence, reasoning)
        self.audit.log_write("evidence_pack", pack.id, claim)
        return pack

    # --- Skills ---

    def register_skill(self, name: str, description: str, procedure: str,
                       tags: list[str] | None = None) -> SkillCapsule:
        capsule = self.skills.register(name, description, procedure, tags)
        self.audit.log_write("skill_capsule", capsule.id, name)
        return capsule

    # --- Sessions ---

    def start_session(self, session_id: str, metadata: dict[str, Any] | None = None) -> str:
        self.sqlite.create_session(session_id, metadata)
        self.audit.log("session_start", "session", session_id)
        return session_id

    def end_session(self, session_id: str) -> None:
        self.sqlite.end_session(session_id)
        self.audit.log("session_end", "session", session_id)

    # --- USC Cognitive Dedup ---

    @property
    def cogstore(self):
        """Lazy-init C3-backed cognitive dedup store."""
        if self._cogstore is None:
            from c3ae.usc_bridge.c3_cogstore import C3CogStore
            self._cogstore = C3CogStore(self.config.db_path)
        return self._cogstore

    @property
    def _predictor(self):
        """Lazy-init predictive compressor."""
        if not hasattr(self, '_lazy_predictor'):
            from usc.cogdedup.predictor import PredictiveCompressor
            self._lazy_predictor = PredictiveCompressor(self.cogstore)
        return self._lazy_predictor

    @property
    def _integrity_verifier(self):
        """Lazy-init integrity verifier for delta/hash checks."""
        if not hasattr(self, '_lazy_integrity_verifier'):
            from usc.cogdedup.integrity import IntegrityVerifier, SecurityPolicy
            self._lazy_integrity_verifier = IntegrityVerifier(SecurityPolicy())
        return self._lazy_integrity_verifier

    @property
    def _anomaly_detector(self):
        """Lazy-init anomaly detector for compression ratio monitoring."""
        if not hasattr(self, '_lazy_anomaly_detector'):
            from usc.cogdedup.anomaly import AnomalyDetector
            self._lazy_anomaly_detector = AnomalyDetector()
        return self._lazy_anomaly_detector

    @property
    def _context_compactor(self):
        """Lazy-init context compactor for LLM prompt compression."""
        if not hasattr(self, '_lazy_context_compactor'):
            from usc.cogdedup.context_compactor import ContextCompactor
            self._lazy_context_compactor = ContextCompactor(self.cogstore)
        return self._lazy_context_compactor

    @property
    def _temporal_tracker(self):
        """Lazy-init temporal motif tracker for event sequence detection."""
        if not hasattr(self, '_lazy_temporal_tracker'):
            from usc.cogdedup.temporal import TemporalMotifTracker
            self._lazy_temporal_tracker = TemporalMotifTracker()
        return self._lazy_temporal_tracker

    @property
    def _recursive_compressor(self):
        """Lazy-init recursive compressor for C3's own state."""
        if not hasattr(self, '_lazy_recursive_compressor'):
            from usc.cogdedup.recursive import RecursiveCompressor
            self._lazy_recursive_compressor = RecursiveCompressor(self.cogstore)
        return self._lazy_recursive_compressor

    def compress_with_dedup(self, data: bytes, data_id: str = "") -> tuple[bytes, dict]:
        """Compress data using cognitive deduplication with integrity + anomaly detection.

        Returns (compressed_blob, stats).
        Stats includes integrity_hash for verification and anomaly_alert if detected.
        The more data processed, the better compression becomes.

        Args:
            data: Raw bytes to compress
            data_id: Optional ID for compression-aware retrieval mapping
        """
        from usc.cogdedup.codec import cogdedup_encode
        blob, stats = cogdedup_encode(data, self.cogstore,
                                      data_id=data_id, predictor=self._predictor)

        # Integrity: store hash of original data for later verification
        stats["integrity_hash"] = self._integrity_verifier.compute_hash(data).hex()

        # Anomaly detection: observe compression ratio
        ratio = len(data) / max(1, len(blob))
        alert = self._anomaly_detector.observe(ratio, label=data_id)
        if alert:
            stats["anomaly_alert"] = {"type": alert.severity, "z_score": alert.z_score}
            self.audit.log("anomaly_detected", "cogdedup", data_id,
                           f"{alert.severity}: z={alert.z_score:.2f}")

        return blob, stats

    def decompress_with_dedup(self, blob: bytes, expected_hash: str = "") -> bytes:
        """Decompress a cogdedup blob with optional integrity verification.

        Args:
            blob: UCOG compressed blob
            expected_hash: Hex hash from compress stats for verification
        """
        from usc.cogdedup.codec import cogdedup_decode
        data = cogdedup_decode(blob, self.cogstore, predictor=self._predictor)

        if expected_hash:
            actual = self._integrity_verifier.compute_hash(data).hex()
            if actual != expected_hash:
                raise ValueError(
                    f"Integrity check failed: expected {expected_hash[:16]}..., "
                    f"got {actual[:16]}..."
                )

        return data

    def stream_compressor(self, data_id: str = ""):
        """Create a streaming cogdedup encoder for real-time session compression.

        Returns a CogdedupStream instance. Feed data as it arrives,
        call finish() to get the UCOG blob.

        Usage:
            stream = spine.stream_compressor(data_id="session-123")
            stream.feed(b"[TOOL_CALL] web_search ...\\n")
            stream.feed(b"[TOOL_RESULT] ...\\n")
            blob, stats = stream.finish()
        """
        from usc.cogdedup.streaming import CogdedupStream
        return CogdedupStream(self.cogstore, data_id=data_id,
                              predictor=self._predictor)

    # --- Context Compaction (LLM Prompt Compression) ---

    def compress_prompt(self, prompt: str) -> dict:
        """Compress an LLM prompt by replacing known chunks with REF placeholders.

        Returns dict with compressed text, token savings, refs used.
        """
        result = self._context_compactor.compress_prompt(prompt)
        return {
            "compressed": result.text,
            "token_savings": result.savings_pct,
            "refs_used": result.refs_inserted,
            "original_tokens": result.original_tokens,
        }

    def expand_response(self, text: str) -> str:
        """Expand REF placeholders in a response back to full content."""
        return self._context_compactor.expand_response(text)

    # --- Temporal Event Tracking ---

    def track_event(self, event_type: str) -> dict | None:
        """Track a temporal event and return motif if a recurring pattern is detected."""
        motif = self._temporal_tracker.observe(event_type)
        if motif:
            return {"pattern": motif.pattern, "count": motif.occurrences,
                    "length": len(motif.pattern)}
        return None

    def track_events_batch(self, events: list[str]) -> list[dict]:
        """Track a batch of events, return all detected motifs."""
        self._temporal_tracker.observe_batch(events)
        return [{"pattern": m.pattern, "count": m.occurrences}
                for m in self._temporal_tracker.detected_motifs()]

    # --- Recursive Self-Compression ---

    def compress_memories(self, memories: list[dict]) -> tuple[bytes, dict]:
        """Compress a batch of C3 memories using cognitive dedup.

        Returns (blob, stats_dict).
        """
        result = self._recursive_compressor.compress_memories(memories)
        return result.blob, {
            "ratio": result.ratio,
            "original_size": result.original_size,
            "compressed_size": result.compressed_size,
        }

    def compress_reasoning_bank(self) -> tuple[bytes, dict]:
        """Compress the entire reasoning bank for archival."""
        entries = self.bank.list_active()
        items = [e.content for e in entries]
        result = self._recursive_compressor.compress_reasoning_bank(items)
        return result.blob, {
            "ratio": result.ratio,
            "entries": len(entries),
            "original_size": result.original_size,
            "compressed_size": result.compressed_size,
        }

    def compress_audit_log(self, limit: int = 10000) -> tuple[bytes, dict]:
        """Compress recent audit log for archival."""
        events = self.audit.recent(limit=limit)
        items = [{"action": e.action, "target_type": e.target_type,
                  "target_id": e.target_id, "detail": e.detail,
                  "created_at": str(e.created_at)} for e in events]
        result = self._recursive_compressor.compress_audit_log(items)
        return result.blob, {
            "ratio": result.ratio,
            "events": len(items),
            "original_size": result.original_size,
            "compressed_size": result.compressed_size,
        }

    # --- Session Orchestrator ---

    def compress_session(self, session_data: bytes, session_id: str = "") -> dict:
        """Compress an agent session with all cogdedup upgrades active.

        Combines: cognitive dedup, integrity hashing, anomaly detection,
        and temporal motif tracking in a single call.

        Returns dict with blob, stats, session_id, compressed_size, original_size.
        """
        # 1. Compress with cognitive dedup (includes integrity + anomaly)
        blob, stats = self.compress_with_dedup(session_data, data_id=session_id)

        # 2. Track temporal patterns from session event types
        lines = session_data.decode("utf-8", errors="replace").split("\n")
        event_types = []
        for line in lines:
            if line.startswith("[TOOL_CALL]"):
                event_types.append("tool_call")
            elif line.startswith("[TOOL_RESULT]"):
                event_types.append("tool_result")
            elif line.startswith("[SEARCH]"):
                event_types.append("search")
            elif line.startswith("[ERROR]"):
                event_types.append("error")
        if event_types:
            self._temporal_tracker.observe_batch(event_types)
            motifs = self._temporal_tracker.detected_motifs()
            stats["temporal_motifs"] = len(motifs)

        # 3. Audit
        ratio = len(session_data) / max(1, len(blob))
        self.audit.log("session_compressed", "session", session_id,
                       f"ratio={ratio:.1f}x, size={len(blob)}")

        return {
            "blob": blob,
            "stats": stats,
            "session_id": session_id,
            "compressed_size": len(blob),
            "original_size": len(session_data),
        }

    # --- Compression-Aware Retrieval ---

    def structural_similarity(self, data_id_a: str, data_id_b: str) -> float:
        """Compute structural similarity between two memories using shared chunks.

        Returns Jaccard similarity (0.0 to 1.0) over shared chunk IDs.
        This is a free signal from the compression layer — no embeddings needed.
        """
        return self.cogstore.structural_similarity(data_id_a, data_id_b)

    def find_structurally_similar(self, data_id: str, threshold: float = 0.3) -> list[tuple]:
        """Find memories structurally similar to the given one.

        Uses chunk overlap from cognitive deduplication.
        Returns list of (memory_id, jaccard_score) sorted by similarity.
        """
        return self.cogstore.find_structurally_similar(data_id, threshold)

    def deduplicate_results(self, results: list[SearchResult],
                            threshold: float = 0.8) -> list[SearchResult]:
        """Remove structurally duplicate search results.

        Uses chunk-level Jaccard similarity to identify results that are
        minor variants of each other. Keeps the highest-scored result
        from each cluster.
        """
        if not results or self._cogstore is None:
            return results

        keep = []
        seen_groups: list[set[int]] = []

        for r in results:
            r_chunks = self.cogstore.get_chunk_ids_for_data(r.id)
            if not r_chunks:
                keep.append(r)
                continue

            is_dup = False
            for group_chunks in seen_groups:
                if not group_chunks:
                    continue
                intersection = r_chunks & group_chunks
                union = r_chunks | group_chunks
                jaccard = len(intersection) / len(union) if union else 0.0
                if jaccard >= threshold:
                    is_dup = True
                    break

            if not is_dup:
                keep.append(r)
                seen_groups.append(r_chunks)

        return keep

    # --- Status ---

    def status(self) -> dict[str, Any]:
        status = {
            "chunks": self.sqlite.count_chunks(),
            "vectors": self.faiss.size,
            "reasoning_entries": len(self.bank.list_active()),
            "skills": len(self.skills.list_all()),
            "vault_documents": len(self.vault.list_documents()),
        }
        # Add compression stats if vault is CompressedVault
        if hasattr(self.vault, 'compression_stats'):
            status["compression"] = self.vault.compression_stats()
        if self._cogstore is not None:
            status["cogdedup"] = self._cogstore.stats()
        if hasattr(self, '_lazy_anomaly_detector'):
            report = self._anomaly_detector.drift_report()
            status["anomaly"] = {
                "total_observations": self._anomaly_detector._observation_count,
                "alerts": report.alerts_count,
                "mean_ratio": round(report.current_mean, 2),
            }
        if hasattr(self, '_lazy_temporal_tracker'):
            motifs = self._temporal_tracker.detected_motifs()
            status["temporal"] = {"motifs_detected": len(motifs)}
        return status

    # --- Internals ---

    async def _embed_text(self, text: str) -> np.ndarray:
        cached = self.embed_cache.get(text)
        if cached is not None:
            return cached
        vec = await self.embedder.embed_single(text)
        self.embed_cache.put(text, vec)
        return vec

    async def _embed_and_index(self, chunk_ids: list[str], texts: list[str]) -> None:
        """Embed texts with caching and add to FAISS index."""
        results, miss_indices = self.embed_cache.get_batch(texts)

        if miss_indices:
            miss_texts = [texts[i] for i in miss_indices]
            try:
                new_vecs = await self.embedder.embed(miss_texts)
                self.embed_cache.put_batch(miss_texts, new_vecs)
                for j, mi in enumerate(miss_indices):
                    results[mi] = new_vecs[j]
            except Exception:
                # If embedding fails, skip vector indexing
                return

        # Index all successfully embedded chunks
        for cid, vec in zip(chunk_ids, results):
            if vec is not None:
                self.faiss.add(vec, cid)

        # Save FAISS index
        if self.config.faiss_dir:
            self.faiss.save()

    async def close(self) -> None:
        await self.embedder.close()
        self.sqlite.close()
        if self.config.faiss_dir:
            self.faiss.save()
