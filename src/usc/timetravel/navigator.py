"""
N7: Time-Travel Debug — walk forward/backward through packet history.

Depends on G2 (Merkle for ordering) and N6 (delta storage for snapshots).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from usc.merkle.chain import build_chain, MerkleChain, hash_packet


@dataclass
class DecodedState:
    """State snapshot at a given point in the timeline."""
    index: int
    packet: bytes
    packet_hash: bytes
    cumulative_packets: List[bytes]


class TimelineNavigator:
    """Navigate forward/backward through a packet history."""

    def __init__(self, packets: List[bytes]) -> None:
        self._packets = packets
        self._chain = build_chain(packets)
        self._position = 0

    @property
    def length(self) -> int:
        return len(self._packets)

    @property
    def position(self) -> int:
        return self._position

    def snapshot_at(self, index: int) -> DecodedState:
        """Get the state at a specific point in history."""
        if index < 0 or index >= len(self._packets):
            raise IndexError(f"Timeline index {index} out of range [0, {len(self._packets)})")
        return DecodedState(
            index=index,
            packet=self._packets[index],
            packet_hash=hash_packet(self._packets[index]),
            cumulative_packets=self._packets[:index + 1],
        )

    def current(self) -> DecodedState:
        """Get current position state."""
        return self.snapshot_at(self._position)

    def forward(self) -> Optional[DecodedState]:
        """Move forward one step. Returns None if at end."""
        if self._position >= len(self._packets) - 1:
            return None
        self._position += 1
        return self.current()

    def backward(self) -> Optional[DecodedState]:
        """Move backward one step. Returns None if at start."""
        if self._position <= 0:
            return None
        self._position -= 1
        return self.current()

    def seek(self, index: int) -> DecodedState:
        """Jump to a specific position."""
        if index < 0 or index >= len(self._packets):
            raise IndexError(f"Timeline index {index} out of range")
        self._position = index
        return self.current()

    def first(self) -> DecodedState:
        """Jump to start."""
        return self.seek(0)

    def last(self) -> DecodedState:
        """Jump to end."""
        return self.seek(len(self._packets) - 1)

    def diff_between(self, idx_a: int, idx_b: int) -> List[int]:
        """
        Return indices of packets that differ between position a and b.
        Useful for finding what changed between two points in time.
        """
        start = min(idx_a, idx_b)
        end = max(idx_a, idx_b)
        return list(range(start + 1, end + 1))

    def chain(self) -> MerkleChain:
        """Access the underlying Merkle chain."""
        return self._chain
