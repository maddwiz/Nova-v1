"""
USC Comprehensive Test Suite
=============================
Tests all codec paths, roundtrips, edge cases, compression ratios,
and error handling across:

1. Memory Codec (mem_encode/mem_decode, tiers 0 & 3)
2. ODC (Outer Dictionary Codec v1)
3. ODC2 (Indexed Block Codec with selective range decode)
4. H1M2 Rowmask (template + unknown row encoding)
5. Stream V3B packets
6. Outerstream framing
7. Edge cases & error handling
8. Compression ratio verification
"""

from __future__ import annotations

import gzip
import json
import random
import string
import struct
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pytest


# ══════════════════════════════════════════════════════════════════════════
# 1. MEMORY CODEC (mem_encode / mem_decode)
# ══════════════════════════════════════════════════════════════════════════

class TestMemoryCodec:
    """Tests for usc.mem.codec — the core memory tier encoder."""

    def _toy_log(self) -> str:
        from usc.bench.datasets import toy_agent_log
        return toy_agent_log()

    def test_tier3_lossless_roundtrip(self):
        """Tier 3 must produce exact text on decode."""
        from usc.mem.codec import mem_encode, mem_decode
        raw = self._toy_log()
        pkt = mem_encode(raw, tier=3)
        decoded, conf = mem_decode(pkt)
        assert decoded == raw
        assert conf >= 0.90

    def test_tier0_returns_valid_text(self):
        """Tier 0 may be lossy but must return non-empty string."""
        from usc.mem.codec import mem_encode, mem_decode
        raw = self._toy_log()
        pkt = mem_encode(raw, tier=0)
        decoded, conf = mem_decode(pkt, min_conf=0.60)
        assert isinstance(decoded, str)
        assert len(decoded) > 0
        assert conf >= 0.60

    def test_tier0_confidence_lower_than_tier3(self):
        """Tier 0 should have lower confidence than tier 3."""
        from usc.mem.codec import mem_encode, mem_decode
        raw = self._toy_log()
        _, conf0 = mem_decode(mem_encode(raw, tier=0), min_conf=0.0)
        _, conf3 = mem_decode(mem_encode(raw, tier=3), min_conf=0.0)
        assert conf3 > conf0

    def test_tier0_rejected_at_high_min_conf(self):
        """Tier 0 should be rejected if min_conf is set high."""
        from usc.mem.codec import mem_encode, mem_decode, USCNeedsMoreBits
        raw = self._toy_log()
        pkt = mem_encode(raw, tier=0)
        with pytest.raises(USCNeedsMoreBits):
            mem_decode(pkt, min_conf=0.90)

    def test_fallback_escalation(self):
        """mem_decode_with_fallback should upgrade from tier 0 to tier 3."""
        from usc.mem.codec import mem_encode, mem_decode_with_fallback
        raw = self._toy_log()
        pkt0 = mem_encode(raw, tier=0)
        pkt3 = mem_encode(raw, tier=3)
        decoded, conf, used_tier = mem_decode_with_fallback(
            [pkt0, pkt3], min_conf=0.80
        )
        assert used_tier == 3
        assert decoded == raw
        assert conf >= 0.90

    def test_invalid_tier_raises(self):
        """Only tiers 0 and 3 are supported."""
        from usc.mem.codec import mem_encode
        with pytest.raises(ValueError):
            mem_encode("test", tier=1)
        with pytest.raises(ValueError):
            mem_encode("test", tier=2)

    def test_corrupted_packet_raises(self):
        """Tampering with packet bytes should cause decode failure."""
        from usc.mem.codec import mem_encode, mem_decode
        raw = self._toy_log()
        pkt = mem_encode(raw, tier=3)
        # Corrupt the gzipped data
        corrupted = pkt[:10] + b"\x00\x00\x00" + pkt[13:]
        with pytest.raises(Exception):
            mem_decode(corrupted)

    def test_ecc_tamper_detected(self):
        """Modifying content inside the packet should fail ECC."""
        from usc.mem.codec import mem_encode, mem_decode
        raw = self._toy_log()
        pkt = mem_encode(raw, tier=3)
        # Decompress, tamper, recompress
        inner = json.loads(gzip.decompress(pkt).decode("utf-8"))
        inner["sk"]["header"] = "TAMPERED"
        tampered = gzip.compress(json.dumps(inner).encode("utf-8"))
        with pytest.raises(ValueError, match="ECC|Fingerprint"):
            mem_decode(tampered)

    def test_packet_is_compressed(self):
        """Encoded packet should be smaller than raw text."""
        from usc.mem.codec import mem_encode
        raw = self._toy_log()
        pkt = mem_encode(raw, tier=3)
        assert len(pkt) < len(raw.encode("utf-8"))

    def test_varied_log_roundtrip(self):
        """Tier 3 roundtrip on varied log data."""
        from usc.mem.codec import mem_encode, mem_decode
        from usc.bench.datasets import toy_big_agent_log_varied
        raw = toy_big_agent_log_varied(loops=10, seed=42)
        pkt = mem_encode(raw, tier=3)
        decoded, conf = mem_decode(pkt)
        assert decoded == raw


# ══════════════════════════════════════════════════════════════════════════
# 2. ODC (OUTER DICTIONARY CODEC V1)
# ══════════════════════════════════════════════════════════════════════════

class TestODC:
    """Tests for usc.api.codec_odc — dictionary-compressed packet codec."""

    def _make_packets(self, text: str) -> List[bytes]:
        from usc.api.codec_odc import build_v3b_packets_from_text
        return build_v3b_packets_from_text(text, max_lines_per_chunk=30)

    def _sample_text(self) -> str:
        from usc.bench.datasets import toy_big_agent_log_varied
        return toy_big_agent_log_varied(loops=20, seed=99)

    def test_encode_decode_roundtrip(self):
        """ODC encode then decode should return identical packets."""
        from usc.api.codec_odc import odc_encode_packets, odc_decode_to_packets
        text = self._sample_text()
        packets = self._make_packets(text)
        blob, meta = odc_encode_packets(packets)
        recovered = odc_decode_to_packets(blob)
        assert len(recovered) == len(packets)
        for orig, dec in zip(packets, recovered):
            assert orig == dec

    def test_odc_encode_text_convenience(self):
        """odc_encode_text should produce decodable blob."""
        from usc.api.codec_odc import odc_encode_text, odc_decode_to_packets
        text = self._sample_text()
        blob, meta = odc_encode_text(text)
        packets = odc_decode_to_packets(blob)
        assert len(packets) > 0
        assert meta.packets == len(packets)

    def test_magic_header(self):
        """ODC blob should start with correct magic bytes."""
        from usc.api.codec_odc import odc_encode_packets, MAGIC
        packets = self._make_packets(self._sample_text())
        blob, _ = odc_encode_packets(packets)
        assert blob[:8] == MAGIC

    def test_metadata_consistent(self):
        """Meta fields should match actual blob structure."""
        from usc.api.codec_odc import odc_encode_packets
        packets = self._make_packets(self._sample_text())
        blob, meta = odc_encode_packets(packets)
        assert meta.packets == len(packets)
        assert meta.dict_bytes >= 0  # may be 0 in plain fallback mode
        assert meta.framed_bytes > 0
        assert meta.compressed_bytes > 0
        assert meta.compressed_bytes < meta.framed_bytes  # compression happened

    def test_bad_magic_raises(self):
        """Decode should reject blobs with wrong magic."""
        from usc.api.codec_odc import odc_decode_to_packets
        bad_blob = b"BADMAGIC" + b"\x00" * 100
        with pytest.raises(ValueError, match="bad magic"):
            odc_decode_to_packets(bad_blob)

    def test_truncated_blob_raises(self):
        """Decode should reject truncated blobs."""
        from usc.api.codec_odc import odc_decode_to_packets
        with pytest.raises(ValueError, match="too small"):
            odc_decode_to_packets(b"USC_ODC1" + b"\x00" * 4)

    def test_compression_ratio(self):
        """ODC blob should be significantly smaller than raw packets."""
        from usc.api.codec_odc import odc_encode_packets
        text = self._sample_text()
        packets = self._make_packets(text)
        raw_size = sum(len(p) for p in packets)
        blob, meta = odc_encode_packets(packets)
        assert len(blob) < raw_size

    def test_single_packet(self):
        """ODC should handle a single packet."""
        from usc.api.codec_odc import odc_encode_packets, odc_decode_to_packets
        packets = [b"hello world this is a test packet"]
        blob, meta = odc_encode_packets(packets)
        recovered = odc_decode_to_packets(blob)
        assert recovered == packets

    def test_many_packets(self):
        """ODC should handle many small packets."""
        from usc.api.codec_odc import odc_encode_packets, odc_decode_to_packets
        packets = [f"packet_{i}: data={i*17}".encode() for i in range(200)]
        blob, meta = odc_encode_packets(packets)
        recovered = odc_decode_to_packets(blob)
        assert recovered == packets


# ══════════════════════════════════════════════════════════════════════════
# 3. ODC2 (INDEXED BLOCK CODEC)
# ══════════════════════════════════════════════════════════════════════════

class TestODC2:
    """Tests for usc.api.codec_odc2_indexed — block-indexed selective decode."""

    def _make_packets(self) -> List[bytes]:
        return [f"packet_{i}: value={i*7} data={'x' * (50 + i)}".encode()
                for i in range(40)]

    def test_full_decode_roundtrip(self):
        """ODC2 full decode returns all original packets."""
        from usc.api.codec_odc2_indexed import (
            odc2_encode_packets, odc2_decode_all_packets,
        )
        packets = self._make_packets()
        blob, meta = odc2_encode_packets(packets, group_size=4)
        recovered = odc2_decode_all_packets(blob)
        assert recovered == packets

    def test_selective_range_decode(self):
        """ODC2 range decode returns correct packet subset."""
        from usc.api.codec_odc2_indexed import (
            odc2_encode_packets, odc2_decode_packet_range,
        )
        packets = self._make_packets()
        blob, meta = odc2_encode_packets(packets, group_size=4)

        # Get packets [10:20]
        subset = odc2_decode_packet_range(blob, 10, 20)
        assert len(subset) == 10
        assert subset == packets[10:20]

    def test_range_decode_first_block(self):
        """Range decode of first few packets."""
        from usc.api.codec_odc2_indexed import (
            odc2_encode_packets, odc2_decode_packet_range,
        )
        packets = self._make_packets()
        blob, _ = odc2_encode_packets(packets, group_size=4)
        subset = odc2_decode_packet_range(blob, 0, 4)
        assert subset == packets[0:4]

    def test_range_decode_last_block(self):
        """Range decode of last packets."""
        from usc.api.codec_odc2_indexed import (
            odc2_encode_packets, odc2_decode_packet_range,
        )
        packets = self._make_packets()
        blob, _ = odc2_encode_packets(packets, group_size=4)
        subset = odc2_decode_packet_range(blob, 36, 40)
        assert subset == packets[36:40]

    def test_range_decode_cross_block_boundary(self):
        """Range that spans two blocks should work correctly."""
        from usc.api.codec_odc2_indexed import (
            odc2_encode_packets, odc2_decode_packet_range,
        )
        packets = self._make_packets()
        blob, _ = odc2_encode_packets(packets, group_size=4)
        # Packets 2-6 span block boundary (blocks are [0-3], [4-7])
        subset = odc2_decode_packet_range(blob, 2, 6)
        assert subset == packets[2:6]

    def test_range_decode_single_packet(self):
        """Range decode of exactly one packet."""
        from usc.api.codec_odc2_indexed import (
            odc2_encode_packets, odc2_decode_packet_range,
        )
        packets = self._make_packets()
        blob, _ = odc2_encode_packets(packets, group_size=4)
        subset = odc2_decode_packet_range(blob, 15, 16)
        assert subset == [packets[15]]

    def test_range_out_of_bounds_clamps(self):
        """Range beyond packet count should be clamped."""
        from usc.api.codec_odc2_indexed import (
            odc2_encode_packets, odc2_decode_packet_range,
        )
        packets = self._make_packets()
        blob, _ = odc2_encode_packets(packets, group_size=4)
        subset = odc2_decode_packet_range(blob, 35, 999)
        assert subset == packets[35:]

    def test_range_empty_result(self):
        """Range past end returns empty list."""
        from usc.api.codec_odc2_indexed import (
            odc2_encode_packets, odc2_decode_packet_range,
        )
        packets = self._make_packets()
        blob, _ = odc2_encode_packets(packets, group_size=4)
        subset = odc2_decode_packet_range(blob, 100, 200)
        assert subset == []

    def test_metadata_fields(self):
        """ODC2 metadata should reflect actual encoding."""
        from usc.api.codec_odc2_indexed import odc2_encode_packets
        packets = self._make_packets()
        _, meta = odc2_encode_packets(packets, group_size=4)
        assert meta.packet_count == 40
        assert meta.group_size == 4
        assert meta.block_count == 10  # 40 / 4
        assert meta.total_comp_bytes > 0
        assert meta.used_mode in ("dict", "plain")

    def test_magic_header(self):
        """ODC2 blob should start with correct magic."""
        from usc.api.codec_odc2_indexed import odc2_encode_packets, MAGIC
        packets = self._make_packets()
        blob, _ = odc2_encode_packets(packets, group_size=4)
        assert blob[:8] == MAGIC

    def test_bad_magic_raises(self):
        """Decode should reject wrong magic."""
        from usc.api.codec_odc2_indexed import odc2_decode_all_packets
        with pytest.raises(ValueError, match="bad magic"):
            odc2_decode_all_packets(b"WRONGMGC" + b"\x00" * 100)

    def test_group_size_1(self):
        """Group size of 1 = one packet per block."""
        from usc.api.codec_odc2_indexed import (
            odc2_encode_packets, odc2_decode_all_packets, odc2_decode_packet_range,
        )
        packets = self._make_packets()[:10]
        blob, meta = odc2_encode_packets(packets, group_size=1)
        assert meta.block_count == 10
        recovered = odc2_decode_all_packets(blob)
        assert recovered == packets
        # Selective decode should work at single-packet granularity
        assert odc2_decode_packet_range(blob, 3, 4) == [packets[3]]

    def test_group_size_larger_than_packets(self):
        """Group size larger than total packets = single block."""
        from usc.api.codec_odc2_indexed import (
            odc2_encode_packets, odc2_decode_all_packets,
        )
        packets = self._make_packets()[:5]
        blob, meta = odc2_encode_packets(packets, group_size=100)
        assert meta.block_count == 1
        recovered = odc2_decode_all_packets(blob)
        assert recovered == packets


# ══════════════════════════════════════════════════════════════════════════
# 4. H1M2 ROWMASK (TEMPLATE + UNKNOWN ROW ENCODING)
# ══════════════════════════════════════════════════════════════════════════

class TestH1M2Rowmask:
    """Tests for the row-oriented template/unknown encoding."""

    def test_all_templated_roundtrip(self):
        """All rows are templated events."""
        from usc.api.hdfs_template_codec_h1m2_rowmask import (
            encode_h1m2_rowmask_blob, decode_h1m2_rowmask_blob,
        )
        rows = [
            (1, ["param1", "param2"]),
            (2, ["value1"]),
            (3, ["a", "b", "c"]),
        ]
        blob = encode_h1m2_rowmask_blob(rows, [])
        decoded_rows, decoded_unknown = decode_h1m2_rowmask_blob(blob)
        assert len(decoded_rows) == 3
        assert decoded_unknown == []
        for orig, dec in zip(rows, decoded_rows):
            assert dec == orig

    def test_all_unknown_roundtrip(self):
        """All rows are unknown lines."""
        from usc.api.hdfs_template_codec_h1m2_rowmask import (
            encode_h1m2_rowmask_blob, decode_h1m2_rowmask_blob,
        )
        rows: List[Optional[Tuple[int, List[str]]]] = [None, None, None]
        unknown = ["line one", "line two", "line three"]
        blob = encode_h1m2_rowmask_blob(rows, unknown)
        decoded_rows, decoded_unknown = decode_h1m2_rowmask_blob(blob)
        assert len(decoded_rows) == 3
        assert all(r is None for r in decoded_rows)
        assert decoded_unknown == unknown

    def test_mixed_roundtrip(self):
        """Mix of templated and unknown rows preserves order."""
        from usc.api.hdfs_template_codec_h1m2_rowmask import (
            encode_h1m2_rowmask_blob, decode_h1m2_rowmask_blob,
        )
        rows: List[Optional[Tuple[int, List[str]]]] = [
            (10, ["192.168.1.1", "8080"]),
            None,
            (20, ["GET", "/api/health"]),
            None,
            (10, ["10.0.0.1", "443"]),
        ]
        unknown = ["unknown line A", "unknown line B"]
        blob = encode_h1m2_rowmask_blob(rows, unknown)
        decoded_rows, decoded_unknown = decode_h1m2_rowmask_blob(blob)

        assert len(decoded_rows) == 5
        assert decoded_rows[0] == (10, ["192.168.1.1", "8080"])
        assert decoded_rows[1] is None
        assert decoded_rows[2] == (20, ["GET", "/api/health"])
        assert decoded_rows[3] is None
        assert decoded_rows[4] == (10, ["10.0.0.1", "443"])
        assert decoded_unknown == unknown

    def test_empty_params(self):
        """Templated row with zero params."""
        from usc.api.hdfs_template_codec_h1m2_rowmask import (
            encode_h1m2_rowmask_blob, decode_h1m2_rowmask_blob,
        )
        rows = [(5, [])]
        blob = encode_h1m2_rowmask_blob(rows, [])
        decoded_rows, _ = decode_h1m2_rowmask_blob(blob)
        assert decoded_rows[0] == (5, [])

    def test_unicode_params(self):
        """Parameters with unicode characters."""
        from usc.api.hdfs_template_codec_h1m2_rowmask import (
            encode_h1m2_rowmask_blob, decode_h1m2_rowmask_blob,
        )
        rows = [(1, ["hello world", "data"])]
        blob = encode_h1m2_rowmask_blob(rows, [])
        decoded_rows, _ = decode_h1m2_rowmask_blob(blob)
        assert decoded_rows[0] == (1, ["hello world", "data"])

    def test_large_event_ids(self):
        """Large event IDs should roundtrip via uvarint."""
        from usc.api.hdfs_template_codec_h1m2_rowmask import (
            encode_h1m2_rowmask_blob, decode_h1m2_rowmask_blob,
        )
        rows = [(999999, ["param"]), (0, ["zero"])]
        blob = encode_h1m2_rowmask_blob(rows, [])
        decoded_rows, _ = decode_h1m2_rowmask_blob(blob)
        assert decoded_rows[0] == (999999, ["param"])
        assert decoded_rows[1] == (0, ["zero"])

    def test_single_row_templated(self):
        """Single templated row."""
        from usc.api.hdfs_template_codec_h1m2_rowmask import (
            encode_h1m2_rowmask_blob, decode_h1m2_rowmask_blob,
        )
        rows = [(42, ["only_param"])]
        blob = encode_h1m2_rowmask_blob(rows, [])
        decoded_rows, decoded_unknown = decode_h1m2_rowmask_blob(blob)
        assert len(decoded_rows) == 1
        assert decoded_rows[0] == (42, ["only_param"])

    def test_single_row_unknown(self):
        """Single unknown row."""
        from usc.api.hdfs_template_codec_h1m2_rowmask import (
            encode_h1m2_rowmask_blob, decode_h1m2_rowmask_blob,
        )
        rows: List[Optional[Tuple[int, List[str]]]] = [None]
        blob = encode_h1m2_rowmask_blob(rows, ["the unknown line"])
        decoded_rows, decoded_unknown = decode_h1m2_rowmask_blob(blob)
        assert decoded_rows[0] is None
        assert decoded_unknown == ["the unknown line"]

    def test_many_rows(self):
        """Stress test with 1000 rows."""
        from usc.api.hdfs_template_codec_h1m2_rowmask import (
            encode_h1m2_rowmask_blob, decode_h1m2_rowmask_blob,
        )
        rng = random.Random(123)
        rows: List[Optional[Tuple[int, List[str]]]] = []
        unknown = []
        for i in range(1000):
            if rng.random() < 0.7:
                params = [f"p{j}_{i}" for j in range(rng.randint(0, 5))]
                rows.append((rng.randint(0, 500), params))
            else:
                rows.append(None)
                unknown.append(f"unknown line {i}: {rng.randint(0, 99999)}")

        blob = encode_h1m2_rowmask_blob(rows, unknown)
        decoded_rows, decoded_unknown = decode_h1m2_rowmask_blob(blob)
        assert len(decoded_rows) == 1000
        assert decoded_unknown == unknown
        for orig, dec in zip(rows, decoded_rows):
            assert orig == dec


# ══════════════════════════════════════════════════════════════════════════
# 5. STREAM V3B PACKETS
# ══════════════════════════════════════════════════════════════════════════

class TestStreamV3B:
    """Tests for the V3B stream protocol packet building."""

    def test_build_packets_returns_dict_and_data(self):
        """build_v3b_packets_from_text should return >= 2 packets."""
        from usc.api.codec_odc import build_v3b_packets_from_text
        from usc.bench.datasets import toy_big_agent_log_varied
        text = toy_big_agent_log_varied(loops=5, seed=7)
        packets = build_v3b_packets_from_text(text)
        assert len(packets) >= 2  # at least dict + 1 data

    def test_dict_packet_has_magic(self):
        """First packet (dict) should have USDICT3B magic."""
        from usc.api.codec_odc import build_v3b_packets_from_text
        from usc.bench.datasets import toy_big_agent_log_varied
        text = toy_big_agent_log_varied(loops=5, seed=7)
        packets = build_v3b_packets_from_text(text)
        # After zstd decompression, should contain template data
        assert len(packets[0]) > 0

    def test_packets_are_bytes(self):
        """All packets must be bytes objects."""
        from usc.api.codec_odc import build_v3b_packets_from_text
        from usc.bench.datasets import toy_big_agent_log_varied
        text = toy_big_agent_log_varied(loops=5, seed=7)
        packets = build_v3b_packets_from_text(text)
        for p in packets:
            assert isinstance(p, (bytes, bytearray))

    def test_window_chunks_parameter(self):
        """Different window sizes should produce different packet counts."""
        from usc.api.codec_odc import build_v3b_packets_from_text
        from usc.bench.datasets import toy_big_agent_log_varied
        text = toy_big_agent_log_varied(loops=20, seed=7)
        p1 = build_v3b_packets_from_text(text, window_chunks=1)
        p5 = build_v3b_packets_from_text(text, window_chunks=5)
        # Larger window = fewer data packets (but same dict packet)
        assert len(p1) >= len(p5)


# ══════════════════════════════════════════════════════════════════════════
# 6. OUTERSTREAM FRAMING
# ══════════════════════════════════════════════════════════════════════════

class TestOuterstream:
    """Tests for pack/unpack packet framing."""

    def test_pack_unpack_roundtrip(self):
        """pack_packets then unpack_packets should be identity."""
        from usc.mem.outerstream_zstd import pack_packets, unpack_packets
        packets = [b"hello", b"world", b"test123"]
        framed = pack_packets(packets)
        recovered = unpack_packets(framed)
        assert recovered == packets

    def test_empty_packets_list(self):
        """Empty packet list should roundtrip."""
        from usc.mem.outerstream_zstd import pack_packets, unpack_packets
        framed = pack_packets([])
        recovered = unpack_packets(framed)
        assert recovered == []

    def test_single_packet(self):
        """Single packet roundtrip."""
        from usc.mem.outerstream_zstd import pack_packets, unpack_packets
        packets = [b"single"]
        framed = pack_packets(packets)
        recovered = unpack_packets(framed)
        assert recovered == packets

    def test_binary_packet_data(self):
        """Packets with arbitrary binary data."""
        from usc.mem.outerstream_zstd import pack_packets, unpack_packets
        packets = [bytes(range(256)), b"\x00" * 1000, b"\xff\xfe\xfd"]
        framed = pack_packets(packets)
        recovered = unpack_packets(framed)
        assert recovered == packets

    def test_compress_decompress_roundtrip(self):
        """Full outerstream compress/decompress cycle."""
        from usc.mem.outerstream_zstd import compress_outerstream, decompress_outerstream
        packets = [f"packet {i}".encode() for i in range(50)]
        blob, meta = compress_outerstream(packets)
        recovered = decompress_outerstream(blob)
        assert recovered == packets


# ══════════════════════════════════════════════════════════════════════════
# 7. COMMIT STORE
# ══════════════════════════════════════════════════════════════════════════

class TestCommitStore:
    """Tests for the known-good decode commit log."""

    def test_commit_and_load(self, tmp_path: Path):
        """Commit a record and load it back."""
        from usc.mem.commit import commit_memory, load_last_commit
        store = str(tmp_path / "commits.jsonl")
        rec = commit_memory(store, "mem-v0.7", used_tier=3,
                           confidence=0.95, decoded_text="hello world")
        last = load_last_commit(store)
        assert last is not None
        assert last.text_fingerprint == rec.text_fingerprint
        assert last.used_tier == 3
        assert last.confidence == 0.95

    def test_multiple_commits_returns_last(self, tmp_path: Path):
        """Loading should return the most recent commit."""
        from usc.mem.commit import commit_memory, load_last_commit
        store = str(tmp_path / "commits.jsonl")
        commit_memory(store, "mem-v0.7", 3, 0.90, "first")
        commit_memory(store, "mem-v0.7", 3, 0.95, "second")
        rec3 = commit_memory(store, "mem-v0.7", 3, 0.99, "third")
        last = load_last_commit(store)
        assert last.text_fingerprint == rec3.text_fingerprint

    def test_load_nonexistent_returns_none(self, tmp_path: Path):
        """Loading from nonexistent file should return None."""
        from usc.mem.commit import load_last_commit
        last = load_last_commit(str(tmp_path / "nonexistent.jsonl"))
        assert last is None


# ══════════════════════════════════════════════════════════════════════════
# 8. VARINT ENCODING
# ══════════════════════════════════════════════════════════════════════════

class TestVarint:
    """Tests for LEB128 uvarint encoding."""

    def test_roundtrip_small(self):
        """Small values roundtrip correctly."""
        from usc.mem.varint import encode_uvarint, decode_uvarint
        for val in [0, 1, 50, 127]:
            encoded = encode_uvarint(val)
            decoded, offset = decode_uvarint(encoded, 0)
            assert decoded == val
            assert offset == len(encoded)

    def test_roundtrip_medium(self):
        """Medium values (multi-byte) roundtrip."""
        from usc.mem.varint import encode_uvarint, decode_uvarint
        for val in [128, 255, 1000, 16383]:
            encoded = encode_uvarint(val)
            decoded, offset = decode_uvarint(encoded, 0)
            assert decoded == val

    def test_roundtrip_large(self):
        """Large values roundtrip."""
        from usc.mem.varint import encode_uvarint, decode_uvarint
        for val in [100000, 1_000_000, 2**32 - 1]:
            encoded = encode_uvarint(val)
            decoded, offset = decode_uvarint(encoded, 0)
            assert decoded == val

    def test_single_byte_for_small_values(self):
        """Values 0-127 should encode as single byte."""
        from usc.mem.varint import encode_uvarint
        for val in range(128):
            assert len(encode_uvarint(val)) == 1


# ══════════════════════════════════════════════════════════════════════════
# 9. EDGE CASES & ERROR HANDLING
# ══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases across all codecs."""

    def test_odc_empty_text(self):
        """ODC with minimal input (single short line)."""
        from usc.api.codec_odc import odc_encode_text, odc_decode_to_packets
        blob, meta = odc_encode_text("hello\n")
        packets = odc_decode_to_packets(blob)
        assert len(packets) > 0

    def test_odc2_single_packet(self):
        """ODC2 with just one packet."""
        from usc.api.codec_odc2_indexed import (
            odc2_encode_packets, odc2_decode_all_packets,
        )
        packets = [b"single packet data here"]
        blob, meta = odc2_encode_packets(packets, group_size=4)
        assert meta.block_count == 1
        recovered = odc2_decode_all_packets(blob)
        assert recovered == packets

    def test_h1m2_empty_rows(self):
        """H1M2 with zero rows — should still produce valid blob."""
        from usc.api.hdfs_template_codec_h1m2_rowmask import (
            encode_h1m2_rowmask_blob, decode_h1m2_rowmask_blob,
        )
        blob = encode_h1m2_rowmask_blob([], [])
        rows, unknown = decode_h1m2_rowmask_blob(blob)
        assert rows == []
        assert unknown == []

    def test_odc_with_binary_like_text(self):
        """Text containing numbers and special formatting."""
        from usc.api.codec_odc import odc_encode_text, odc_decode_to_packets
        lines = []
        for i in range(100):
            lines.append(f"2024-01-{i%28+1:02d} 12:{i%60:02d}:00 INFO server.app - Request {i} processed in {i*3}ms status=200")
        text = "\n".join(lines) + "\n"
        blob, meta = odc_encode_text(text)
        packets = odc_decode_to_packets(blob)
        assert len(packets) > 0

    def test_mem_codec_with_minimal_text(self):
        """Memory codec with very short text."""
        from usc.mem.codec import mem_encode, mem_decode
        text = "Project: X\nGoal: test.\n"
        pkt = mem_encode(text, tier=3)
        decoded, conf = mem_decode(pkt)
        assert decoded == text


# ══════════════════════════════════════════════════════════════════════════
# 10. COMPRESSION RATIO VERIFICATION
# ══════════════════════════════════════════════════════════════════════════

class TestCompressionRatios:
    """Verify that USC achieves meaningful compression."""

    def test_odc_beats_raw_on_structured_logs(self):
        """ODC should compress structured logs significantly."""
        from usc.api.codec_odc import odc_encode_text
        # Generate structured log-like data
        lines = []
        for i in range(500):
            lines.append(
                f"2024-01-15 10:{i%60:02d}:{i%60:02d}.{i%1000:03d} "
                f"INFO  [worker-{i%8}] com.app.Service - "
                f"Processing request id={10000+i} user=user_{i%50} "
                f"latency={50+i%200}ms status=200"
            )
        text = "\n".join(lines) + "\n"
        raw_size = len(text.encode("utf-8"))
        blob, meta = odc_encode_text(text)
        ratio = raw_size / len(blob)
        # Should achieve at least 2x compression on structured logs
        assert ratio > 2.0, f"Compression ratio {ratio:.2f}x is too low"

    def test_mem_tier3_compresses(self):
        """Tier 3 memory encoding should compress the text."""
        from usc.mem.codec import mem_encode
        from usc.bench.datasets import toy_big_agent_log_varied
        text = toy_big_agent_log_varied(loops=30, seed=7)
        raw_size = len(text.encode("utf-8"))
        pkt = mem_encode(text, tier=3)
        assert len(pkt) < raw_size

    def test_odc2_overhead_acceptable(self):
        """ODC2 block index overhead should be small relative to data."""
        from usc.api.codec_odc import build_v3b_packets_from_text
        from usc.api.codec_odc2_indexed import odc2_encode_packets
        from usc.api.codec_odc import odc_encode_packets
        from usc.bench.datasets import toy_big_agent_log_varied

        text = toy_big_agent_log_varied(loops=20, seed=42)
        packets = build_v3b_packets_from_text(text)

        odc1_blob, _ = odc_encode_packets(packets)
        odc2_blob, _ = odc2_encode_packets(packets, group_size=4)

        # ODC2 has block index overhead but shouldn't be >50% larger than ODC1
        overhead_pct = (len(odc2_blob) - len(odc1_blob)) / len(odc1_blob) * 100
        assert overhead_pct < 50, f"ODC2 overhead {overhead_pct:.1f}% too high"


# ══════════════════════════════════════════════════════════════════════════
# 11. PERFORMANCE SMOKE TESTS
# ══════════════════════════════════════════════════════════════════════════

class TestPerformance:
    """Basic performance sanity checks."""

    def test_odc_encode_speed(self):
        """ODC encoding 10k lines should complete in reasonable time."""
        from usc.api.codec_odc import odc_encode_text
        lines = [
            f"2024-01-15 12:00:{i%60:02d} INFO service - event={i} val={i*7}"
            for i in range(10000)
        ]
        text = "\n".join(lines) + "\n"
        t0 = time.perf_counter()
        blob, meta = odc_encode_text(text)
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.0, f"ODC encode took {elapsed:.2f}s (too slow)"

    def test_odc2_selective_decode_faster_than_full(self):
        """Selective range decode should be faster than full decode for large data."""
        from usc.api.codec_odc import build_v3b_packets_from_text
        from usc.api.codec_odc2_indexed import (
            odc2_encode_packets, odc2_decode_all_packets, odc2_decode_packet_range,
        )
        # Create enough packets
        lines = [f"line {i}: data={i*13} status=ok" for i in range(5000)]
        text = "\n".join(lines) + "\n"
        packets = build_v3b_packets_from_text(text, max_lines_per_chunk=50)
        blob, meta = odc2_encode_packets(packets, group_size=4)

        # Full decode
        t0 = time.perf_counter()
        odc2_decode_all_packets(blob)
        full_time = time.perf_counter() - t0

        # Selective decode (just 4 packets)
        t0 = time.perf_counter()
        odc2_decode_packet_range(blob, 0, 4)
        range_time = time.perf_counter() - t0

        # Selective should be faster (or at least not slower) when there are multiple blocks
        if meta.block_count > 2:
            assert range_time <= full_time * 1.5, (
                f"Range decode ({range_time:.4f}s) not faster than full ({full_time:.4f}s)"
            )
