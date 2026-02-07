from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List, Tuple

MAGIC = b"UMKL"  # 4 bytes — USC Merkle chain


@dataclass
class MerkleEntry:
    packet_hash: bytes  # 32 bytes SHA-256
    prev_hash: bytes    # 32 bytes (zeros for first entry)


@dataclass
class MerkleChain:
    entries: List[MerkleEntry] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    def root_hash(self) -> bytes:
        """Return hash of last entry, or 32 zero bytes if empty."""
        if not self.entries:
            return b"\x00" * 32
        return self.entries[-1].packet_hash


def hash_packet(data: bytes) -> bytes:
    """SHA-256 hash of raw packet bytes."""
    return hashlib.sha256(data).digest()


def build_chain(packets: List[bytes]) -> MerkleChain:
    """Build a Merkle chain from an ordered list of packets."""
    chain = MerkleChain()
    prev = b"\x00" * 32
    for pkt in packets:
        h = hash_packet(pkt)
        # Chain link: hash includes prev_hash for ordering integrity
        link = hashlib.sha256(prev + h).digest()
        chain.entries.append(MerkleEntry(packet_hash=link, prev_hash=prev))
        prev = link
    return chain


def verify_chain(chain: MerkleChain, packets: List[bytes]) -> bool:
    """Verify that a chain matches the given packets exactly."""
    if len(chain.entries) != len(packets):
        return False
    prev = b"\x00" * 32
    for entry, pkt in zip(chain.entries, packets):
        h = hash_packet(pkt)
        expected_link = hashlib.sha256(prev + h).digest()
        if entry.packet_hash != expected_link:
            return False
        if entry.prev_hash != prev:
            return False
        prev = expected_link
    return True


def verify_packet(chain: MerkleChain, index: int, packet: bytes) -> bool:
    """Verify a single packet at a given index against the chain."""
    if index < 0 or index >= len(chain.entries):
        return False
    entry = chain.entries[index]
    prev = chain.entries[index - 1].packet_hash if index > 0 else b"\x00" * 32
    h = hash_packet(packet)
    expected_link = hashlib.sha256(prev + h).digest()
    return entry.packet_hash == expected_link and entry.prev_hash == prev


def _u32(x: int) -> bytes:
    return int(x).to_bytes(4, "little", signed=False)


def _read_u32(buf: bytes, off: int) -> Tuple[int, int]:
    return int.from_bytes(buf[off:off + 4], "little", signed=False), off + 4


def serialize_chain(chain: MerkleChain) -> bytes:
    """
    Wire format:
        UMKL (4B) + count (u32) + [packet_hash (32B) + prev_hash (32B)] * count
    """
    out = bytearray(MAGIC)
    out += _u32(len(chain.entries))
    for entry in chain.entries:
        out += entry.packet_hash
        out += entry.prev_hash
    return bytes(out)


def deserialize_chain(blob: bytes) -> MerkleChain:
    """Deserialize a UMKL blob back to a MerkleChain."""
    if len(blob) < 8:
        raise ValueError("merkle: blob too small")
    if blob[:4] != MAGIC:
        raise ValueError("merkle: bad magic")
    count, off = _read_u32(blob, 4)
    expected = 8 + count * 64
    if len(blob) < expected:
        raise ValueError("merkle: truncated blob")
    entries = []
    for _ in range(count):
        pkt_hash = blob[off:off + 32]
        off += 32
        prev_hash = blob[off:off + 32]
        off += 32
        entries.append(MerkleEntry(packet_hash=pkt_hash, prev_hash=prev_hash))
    return MerkleChain(entries=entries)
