"""Cross-Agent Knowledge Transfer via Cogstore Federation.

Upgrade #9: Multiple agents can share structural patterns while keeping
content private. Each agent has its own local tier, and a shared tier
holds common patterns (tool call formats, error templates, protocol headers).

Chunk IDs are namespaced:
  - local:N  — private to this agent
  - shared:N — visible across all federated agents

When a local chunk's ref_count exceeds a threshold across multiple agents,
it gets promoted to the shared tier.

Usage:
    shared_store = MemoryCogStore()
    federation = CogstoreFederation(shared_store, promote_threshold=5)

    # Each agent gets a federated store
    agent_a = federation.create_agent_store("agent-a")
    agent_b = federation.create_agent_store("agent-b")

    # Encoding checks shared first, then local
    blob, stats = cogdedup_encode(data, agent_a)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

from usc.cogdedup.store import CogStore, MemoryCogStore, ChunkEntry
from usc.cogdedup.hasher import sha256_hash, simhash64


class FederatedStore(CogStore):
    """A federated store that checks shared tier first, then local.

    Stores new chunks locally. When a chunk's cross-agent ref_count
    exceeds the promotion threshold, it gets promoted to shared.
    """

    def __init__(
        self,
        agent_id: str,
        local: MemoryCogStore,
        shared: MemoryCogStore,
        promote_threshold: int = 5,
    ) -> None:
        self._agent_id = agent_id
        self._local = local
        self._shared = shared
        self._promote_threshold = promote_threshold
        # Track which chunks have been promoted
        self._promoted: Set[str] = set()  # sha256 hashes
        # Local ID -> shared ID mapping
        self._id_remap: Dict[int, int] = {}

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def lookup_exact(self, sha256: str) -> Optional[ChunkEntry]:
        # Check shared first (more likely to hit for common patterns)
        entry = self._shared.lookup_exact(sha256)
        if entry is not None:
            return entry
        return self._local.lookup_exact(sha256)

    def lookup_similar(self, simhash: int) -> Optional[ChunkEntry]:
        # Check shared first
        entry = self._shared.lookup_similar(simhash)
        if entry is not None:
            return entry
        return self._local.lookup_similar(simhash)

    def store(self, data: bytes) -> ChunkEntry:
        sha = sha256_hash(data)

        # Check if already in shared
        shared_entry = self._shared.lookup_exact(sha)
        if shared_entry is not None:
            return shared_entry

        # Store locally
        entry = self._local.store(data)

        # Check for promotion: if ref_count exceeds threshold
        if entry.ref_count >= self._promote_threshold and sha not in self._promoted:
            self._promote(entry)

        return entry

    def get(self, chunk_id: int) -> Optional[ChunkEntry]:
        # Check shared first (promoted chunks live there)
        entry = self._shared.get(chunk_id)
        if entry is not None:
            return entry
        # Check remapped IDs
        if chunk_id in self._id_remap:
            entry = self._shared.get(self._id_remap[chunk_id])
            if entry is not None:
                return entry
        return self._local.get(chunk_id)

    def _promote(self, entry: ChunkEntry) -> None:
        """Promote a local chunk to the shared tier."""
        if entry.data is None:
            return
        shared_entry = self._shared.store(entry.data)
        self._promoted.add(entry.sha256)
        self._id_remap[entry.chunk_id] = shared_entry.chunk_id

    def record_cooccurrence(self, chunk_ids: List[int]) -> None:
        self._local.record_cooccurrence(chunk_ids)

    def get_predicted_chunks(self, chunk_id: int, top_k: int = 5) -> List[ChunkEntry]:
        # Merge predictions from both tiers
        local_pred = self._local.get_predicted_chunks(chunk_id, top_k)
        shared_pred = self._shared.get_predicted_chunks(chunk_id, top_k)
        # Deduplicate by sha256
        seen: Set[str] = set()
        result: List[ChunkEntry] = []
        for e in shared_pred + local_pred:
            if e.sha256 not in seen:
                seen.add(e.sha256)
                result.append(e)
        return result[:top_k]

    def register_data_chunks(self, data_id: str, chunk_ids: Set[int]) -> None:
        self._local.register_data_chunks(data_id, chunk_ids)

    def get_chunk_ids_for_data(self, data_id: str) -> Set[int]:
        return self._local.get_chunk_ids_for_data(data_id)

    def stats(self) -> dict:
        local_stats = self._local.stats()
        shared_stats = self._shared.stats()
        return {
            "agent_id": self._agent_id,
            "local": local_stats,
            "shared": shared_stats,
            "promoted_chunks": len(self._promoted),
            "id_remaps": len(self._id_remap),
        }


class CogstoreFederation:
    """Manages a federation of agent-specific cogstores with a shared tier.

    Args:
        shared_store: The shared MemoryCogStore for cross-agent patterns
        promote_threshold: Ref count at which local chunks get promoted
    """

    def __init__(
        self,
        shared_store: Optional[MemoryCogStore] = None,
        promote_threshold: int = 5,
    ) -> None:
        self._shared = shared_store or MemoryCogStore()
        self._promote_threshold = promote_threshold
        self._agents: Dict[str, FederatedStore] = {}

    def create_agent_store(self, agent_id: str) -> FederatedStore:
        """Create or retrieve a federated store for an agent."""
        if agent_id in self._agents:
            return self._agents[agent_id]

        local = MemoryCogStore()
        store = FederatedStore(
            agent_id=agent_id,
            local=local,
            shared=self._shared,
            promote_threshold=self._promote_threshold,
        )
        self._agents[agent_id] = store
        return store

    def get_agent_store(self, agent_id: str) -> Optional[FederatedStore]:
        return self._agents.get(agent_id)

    @property
    def shared_store(self) -> MemoryCogStore:
        return self._shared

    @property
    def agent_ids(self) -> List[str]:
        return list(self._agents.keys())

    def federation_stats(self) -> dict:
        return {
            "shared": self._shared.stats(),
            "agents": {aid: s.stats() for aid, s in self._agents.items()},
            "total_agents": len(self._agents),
        }
