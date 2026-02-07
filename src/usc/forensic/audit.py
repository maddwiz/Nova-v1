"""
N2: Forensic Mode — audit trail, integrity verification, and structural diff.

Depends on G2 (Merkle chains) and N4 (zero-copy for efficient scanning).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Set

from usc.merkle.chain import (
    MerkleChain,
    build_chain,
    verify_chain,
    verify_packet,
    hash_packet,
)


@dataclass
class AuditEntry:
    """A single entry in the audit trail."""
    timestamp: float
    operation: str  # "encode", "append", "modify", "verify"
    packet_index: int
    packet_hash: bytes
    encoder_version: str = "usc-0.1.0"


@dataclass
class AuditResult:
    """Result of an integrity verification."""
    valid: bool
    total_packets: int
    verified_packets: int
    corrupted_indices: List[int] = field(default_factory=list)
    message: str = ""


@dataclass
class ForensicDiff:
    """Structural diff between two packet lists."""
    added_indices: List[int] = field(default_factory=list)
    removed_indices: List[int] = field(default_factory=list)
    changed_indices: List[int] = field(default_factory=list)
    unchanged_count: int = 0


class ForensicAudit:
    """Wraps encoded packets + Merkle chain for forensic analysis."""

    def __init__(
        self,
        packets: List[bytes],
        chain: Optional[MerkleChain] = None,
    ) -> None:
        self.packets = packets
        self.chain = chain or build_chain(packets)
        self._audit_trail: List[AuditEntry] = []

        # Record initial encode
        for i, pkt in enumerate(packets):
            self._audit_trail.append(AuditEntry(
                timestamp=time.time(),
                operation="encode",
                packet_index=i,
                packet_hash=hash_packet(pkt),
            ))

    def verify_integrity(self) -> AuditResult:
        """Verify all packets against the Merkle chain."""
        if verify_chain(self.chain, self.packets):
            return AuditResult(
                valid=True,
                total_packets=len(self.packets),
                verified_packets=len(self.packets),
                message="All packets verified OK",
            )

        # Find specific corrupted packets
        corrupted = []
        verified = 0
        for i in range(len(self.packets)):
            if i < len(self.chain):
                if verify_packet(self.chain, i, self.packets[i]):
                    verified += 1
                else:
                    corrupted.append(i)
            else:
                corrupted.append(i)

        return AuditResult(
            valid=False,
            total_packets=len(self.packets),
            verified_packets=verified,
            corrupted_indices=corrupted,
            message=f"Corruption detected in {len(corrupted)} packet(s)",
        )

    def chain_of_custody(self) -> List[AuditEntry]:
        """Return the audit trail in chronological order."""
        return sorted(self._audit_trail, key=lambda e: e.timestamp)

    def record_operation(self, operation: str, packet_index: int) -> None:
        """Record a forensic audit event."""
        pkt_hash = hash_packet(self.packets[packet_index]) if packet_index < len(self.packets) else b"\x00" * 32
        self._audit_trail.append(AuditEntry(
            timestamp=time.time(),
            operation=operation,
            packet_index=packet_index,
            packet_hash=pkt_hash,
        ))

    @staticmethod
    def diff_states(packets_a: List[bytes], packets_b: List[bytes]) -> ForensicDiff:
        """
        Compute a structural diff between two packet lists.
        Uses packet hashes for comparison.
        """
        hashes_a = {i: hash_packet(p) for i, p in enumerate(packets_a)}
        hashes_b = {i: hash_packet(p) for i, p in enumerate(packets_b)}

        hash_set_a = set(hashes_a.values())
        hash_set_b = set(hashes_b.values())

        diff = ForensicDiff()

        # Find changed, unchanged in overlapping range
        min_len = min(len(packets_a), len(packets_b))
        for i in range(min_len):
            if hashes_a[i] == hashes_b[i]:
                diff.unchanged_count += 1
            else:
                diff.changed_indices.append(i)

        # Added (in B but not in range of A)
        for i in range(min_len, len(packets_b)):
            diff.added_indices.append(i)

        # Removed (in A but not in range of B)
        for i in range(min_len, len(packets_a)):
            diff.removed_indices.append(i)

        return diff
