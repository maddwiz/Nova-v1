"""
G4: FLEET Mode — Multi-Agent Stream Multiplexer.

Interleaves packets from multiple agent streams into a single blob,
with per-agent Merkle integrity chains.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict, List, Tuple

import zstandard as zstd

from usc.mem.varint import encode_uvarint, decode_uvarint
from usc.merkle.chain import build_chain, verify_chain, serialize_chain, deserialize_chain, MerkleChain

MAGIC = b"UFLT"  # 4 bytes


@dataclass
class FleetMeta:
    n_agents: int
    total_packets: int
    compressed_bytes: int
    raw_bytes: int


def fleet_encode(
    streams: Dict[str, List[bytes]],
    level: int = 10,
) -> Tuple[bytes, FleetMeta]:
    """
    Encode multiple agent streams into a single FLEET blob.

    Wire format:
        UFLT (4B)
        n_agents (u16)
        For each agent:
            agent_name_len (u8) + agent_name (UTF-8)
            n_packets (u32)
            merkle_chain_len (u32) + merkle_chain_bytes
        compressed_len (u32)
        zstd(interleaved_packets)

    Interleaved packets format (inside zstd):
        For each packet:
            agent_index (u16) + packet_len (u32) + packet_bytes
    """
    agent_names = sorted(streams.keys())
    agent_index = {name: i for i, name in enumerate(agent_names)}

    # Build Merkle chains per agent
    chains: Dict[str, MerkleChain] = {}
    for name in agent_names:
        chains[name] = build_chain(streams[name])

    # Interleave packets (round-robin)
    interleaved = bytearray()
    total_packets = 0
    max_len = max((len(streams[n]) for n in agent_names), default=0)
    for pkt_idx in range(max_len):
        for name in agent_names:
            if pkt_idx < len(streams[name]):
                pkt = streams[name][pkt_idx]
                interleaved += struct.pack("<H", agent_index[name])
                interleaved += struct.pack("<I", len(pkt))
                interleaved += pkt
                total_packets += 1

    raw_bytes = len(interleaved)
    cctx = zstd.ZstdCompressor(level=level)
    compressed = cctx.compress(bytes(interleaved))

    # Build header
    out = bytearray(MAGIC)
    out += struct.pack("<H", len(agent_names))

    for name in agent_names:
        name_bytes = name.encode("utf-8")
        out += struct.pack("B", len(name_bytes))
        out += name_bytes
        out += struct.pack("<I", len(streams[name]))
        chain_blob = serialize_chain(chains[name])
        out += struct.pack("<I", len(chain_blob))
        out += chain_blob

    out += struct.pack("<I", len(compressed))
    out += compressed

    meta = FleetMeta(
        n_agents=len(agent_names),
        total_packets=total_packets,
        compressed_bytes=len(compressed),
        raw_bytes=raw_bytes,
    )
    return bytes(out), meta


def fleet_decode(blob: bytes) -> Dict[str, List[bytes]]:
    """Decode a FLEET blob back to per-agent packet streams."""
    if len(blob) < 6:
        raise ValueError("fleet: blob too small")
    if blob[:4] != MAGIC:
        raise ValueError("fleet: bad magic")

    off = 4
    n_agents = struct.unpack_from("<H", blob, off)[0]
    off += 2

    agent_names: List[str] = []
    agent_packet_counts: List[int] = []
    agent_chains: List[MerkleChain] = []

    for _ in range(n_agents):
        name_len = blob[off]
        off += 1
        name = blob[off:off + name_len].decode("utf-8")
        off += name_len
        agent_names.append(name)

        n_packets = struct.unpack_from("<I", blob, off)[0]
        off += 4
        agent_packet_counts.append(n_packets)

        chain_len = struct.unpack_from("<I", blob, off)[0]
        off += 4
        chain_blob = blob[off:off + chain_len]
        off += chain_len
        agent_chains.append(deserialize_chain(chain_blob))

    comp_len = struct.unpack_from("<I", blob, off)[0]
    off += 4
    compressed = blob[off:off + comp_len]

    dctx = zstd.ZstdDecompressor()
    interleaved = dctx.decompress(compressed)

    # De-interleave
    streams: Dict[str, List[bytes]] = {name: [] for name in agent_names}
    pos = 0
    while pos < len(interleaved):
        agent_idx = struct.unpack_from("<H", interleaved, pos)[0]
        pos += 2
        pkt_len = struct.unpack_from("<I", interleaved, pos)[0]
        pos += 4
        pkt = interleaved[pos:pos + pkt_len]
        pos += pkt_len
        streams[agent_names[agent_idx]].append(bytes(pkt))

    return streams


def fleet_verify(blob: bytes) -> Dict[str, bool]:
    """Verify per-agent Merkle integrity. Returns name -> verified mapping."""
    if blob[:4] != MAGIC:
        raise ValueError("fleet: bad magic")

    off = 4
    n_agents = struct.unpack_from("<H", blob, off)[0]
    off += 2

    agent_names: List[str] = []
    agent_chains: List[MerkleChain] = []

    for _ in range(n_agents):
        name_len = blob[off]
        off += 1
        name = blob[off:off + name_len].decode("utf-8")
        off += name_len
        agent_names.append(name)

        _n_packets = struct.unpack_from("<I", blob, off)[0]
        off += 4

        chain_len = struct.unpack_from("<I", blob, off)[0]
        off += 4
        chain_blob = blob[off:off + chain_len]
        off += chain_len
        agent_chains.append(deserialize_chain(chain_blob))

    streams = fleet_decode(blob)

    results: Dict[str, bool] = {}
    for name, chain in zip(agent_names, agent_chains):
        results[name] = verify_chain(chain, streams[name])
    return results
