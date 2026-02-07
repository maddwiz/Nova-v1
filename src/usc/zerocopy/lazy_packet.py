"""
N4: Zero-Copy Decode — LazyBlob and memoryview-based packet access.

Provides lazy decompression: only decompress packets on demand.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import zstandard as zstd


# Known magic bytes and their header sizes
_MAGIC_REGISTRY: Dict[bytes, str] = {
    b"USC_ODC1": "odc1",
    b"USC_ODC2": "odc2",
    b"UFLT": "fleet",
    b"UGST": "gist",
    b"UTOK": "token",
    b"UPII": "pii",
    b"UMKL": "merkle",
}


def _identify_format(data: bytes) -> str:
    """Identify the wire format from magic bytes."""
    if len(data) >= 8:
        magic8 = bytes(data[:8])
        if magic8 in _MAGIC_REGISTRY:
            return _MAGIC_REGISTRY[magic8]
    if len(data) >= 4:
        magic4 = bytes(data[:4])
        if magic4 in _MAGIC_REGISTRY:
            return _MAGIC_REGISTRY[magic4]
    return "unknown"


@dataclass
class LazyBlob:
    """
    Zero-copy blob wrapper. Parses header and packet offsets without
    decompressing the full payload. Individual packets are decompressed
    on demand.
    """
    _buf: memoryview = field(repr=False)
    _format: str = ""
    _header_parsed: bool = False
    _metadata: Dict[str, Any] = field(default_factory=dict)
    _packet_cache: Dict[int, bytes] = field(default_factory=dict)

    # For ODC2 indexed format
    _block_offsets: List[Tuple[int, int]] = field(default_factory=list)
    _n_packets: int = 0

    # For simple framed packet formats
    _framed_data: Optional[bytes] = field(default=None, repr=False)
    _packet_offsets: List[Tuple[int, int]] = field(default_factory=list)

    def __init__(self, buf: bytes) -> None:
        self._buf = memoryview(buf)
        self._format = _identify_format(buf)
        self._header_parsed = False
        self._metadata = {}
        self._packet_cache = {}
        self._block_offsets = []
        self._n_packets = 0
        self._framed_data = None
        self._packet_offsets = []

    def format(self) -> str:
        """Return the identified wire format name."""
        return self._format

    def header(self) -> Dict[str, Any]:
        """Parse and return header metadata without decompressing payload."""
        if not self._header_parsed:
            self._parse_header()
        return dict(self._metadata)

    def num_packets(self) -> int:
        """Number of packets in this blob."""
        if not self._header_parsed:
            self._parse_header()
        return self._n_packets

    def packet(self, index: int) -> bytes:
        """Decompress and return a single packet by index."""
        if index in self._packet_cache:
            return self._packet_cache[index]

        if not self._header_parsed:
            self._parse_header()

        if index < 0 or index >= self._n_packets:
            raise IndexError(f"Packet index {index} out of range [0, {self._n_packets})")

        # Lazy decompress
        pkt = self._decompress_packet(index)
        self._packet_cache[index] = pkt
        return pkt

    def packets(self, start: int = 0, end: Optional[int] = None) -> List[bytes]:
        """Decompress and return a range of packets."""
        if end is None:
            end = self.num_packets()
        return [self.packet(i) for i in range(start, min(end, self.num_packets()))]

    def all_packets(self) -> List[bytes]:
        """Decompress all packets (full decode)."""
        return self.packets(0, self.num_packets())

    def _parse_header(self) -> None:
        """Parse header based on format."""
        self._header_parsed = True
        buf = self._buf

        if self._format == "odc1":
            self._parse_odc1_header(buf)
        elif self._format == "odc2":
            self._parse_odc2_header(buf)
        else:
            # Generic: try to parse as a framed packet stream
            self._parse_generic(buf)

    def _parse_odc1_header(self, buf: memoryview) -> None:
        """Parse ODC1 header."""
        off = 8  # skip magic
        level = int.from_bytes(buf[off:off + 4], "little")
        off += 4
        dict_len = int.from_bytes(buf[off:off + 4], "little")
        off += 4
        off += dict_len  # skip dict bytes
        framed_len = int.from_bytes(buf[off:off + 4], "little")
        off += 4
        comp_len = int.from_bytes(buf[off:off + 4], "little")
        off += 4

        self._metadata = {
            "magic": "USC_ODC1",
            "level": level,
            "dict_len": dict_len,
            "framed_len": framed_len,
            "compressed_len": comp_len,
        }

        # We need to decompress to count packets, but do it lazily
        self._metadata["_comp_start"] = off
        self._metadata["_dict_start"] = 16  # 8 + 4 + 4
        self._metadata["_dict_len"] = dict_len

        # Decompress framed data to find packet offsets
        self._decompress_framed_odc1(buf)

    def _decompress_framed_odc1(self, buf: memoryview) -> None:
        """Decompress ODC1 framed data to find packet offsets."""
        off = 8 + 4  # past magic + level
        dict_len = int.from_bytes(buf[off:off + 4], "little")
        off += 4
        dict_bytes = bytes(buf[off:off + dict_len])
        off += dict_len
        _framed_len = int.from_bytes(buf[off:off + 4], "little")
        off += 4
        comp_len = int.from_bytes(buf[off:off + 4], "little")
        off += 4
        comp = bytes(buf[off:off + comp_len])

        if len(dict_bytes) > 0:
            ddict = zstd.ZstdCompressionDict(dict_bytes)
            dctx = zstd.ZstdDecompressor(dict_data=ddict)
        else:
            dctx = zstd.ZstdDecompressor()
        self._framed_data = dctx.decompress(comp)
        self._parse_framed_packets()

    def _parse_odc2_header(self, buf: memoryview) -> None:
        """Parse ODC2 header — has indexed blocks."""
        # ODC2 has more complex structure, decompress fully
        self._metadata = {"magic": "USC_ODC2"}
        # Fall back to full decompress for now
        from usc.api.codec_odc2_indexed import odc2_decode_all_packets
        pkts = odc2_decode_all_packets(bytes(buf))
        for i, p in enumerate(pkts):
            self._packet_cache[i] = p
        self._n_packets = len(pkts)

    def _parse_generic(self, buf: memoryview) -> None:
        """Try to parse as framed packet stream."""
        self._metadata = {"magic": bytes(buf[:4]).decode("ascii", errors="replace")}
        # Store raw buffer; single packet
        self._packet_cache[0] = bytes(buf)
        self._n_packets = 1

    def _parse_framed_packets(self) -> None:
        """Parse framed packet data (u32 count + [u32 len + data] * count)."""
        if self._framed_data is None:
            return
        data = self._framed_data
        if len(data) < 4:
            self._n_packets = 0
            return
        count = int.from_bytes(data[0:4], "little")
        off = 4
        offsets = []
        for _ in range(count):
            if off + 4 > len(data):
                break
            pkt_len = int.from_bytes(data[off:off + 4], "little")
            off += 4
            offsets.append((off, pkt_len))
            off += pkt_len
        self._packet_offsets = offsets
        self._n_packets = len(offsets)

    def _decompress_packet(self, index: int) -> bytes:
        """Decompress a single packet."""
        if self._framed_data is not None and index < len(self._packet_offsets):
            start, length = self._packet_offsets[index]
            return self._framed_data[start:start + length]
        if index in self._packet_cache:
            return self._packet_cache[index]
        raise ValueError(f"Cannot decompress packet {index}")
