"""Tests for N4: Zero-Copy Decode."""
import pytest
from usc.zerocopy import LazyBlob
from usc.api.codec_odc import (
    build_v3b_packets_from_text,
    odc_encode_packets,
    odc_decode_to_packets,
)
from usc.bench.datasets import toy_big_agent_log_varied


def _make_odc_blob():
    text = toy_big_agent_log_varied(loops=10)
    packets = build_v3b_packets_from_text(text)
    blob, meta = odc_encode_packets(packets)
    return blob, packets


class TestLazyBlob:
    def test_format_detection(self):
        blob, _ = _make_odc_blob()
        lazy = LazyBlob(blob)
        assert lazy.format() == "odc1"

    def test_header_without_decompress(self):
        blob, _ = _make_odc_blob()
        lazy = LazyBlob(blob)
        header = lazy.header()
        assert header["magic"] == "USC_ODC1"
        assert "level" in header
        assert "compressed_len" in header

    def test_num_packets(self):
        blob, packets = _make_odc_blob()
        lazy = LazyBlob(blob)
        assert lazy.num_packets() == len(packets)

    def test_single_packet_access(self):
        blob, packets = _make_odc_blob()
        lazy = LazyBlob(blob)
        # Access first and last packet
        assert lazy.packet(0) == packets[0]
        assert lazy.packet(len(packets) - 1) == packets[-1]

    def test_packet_range(self):
        blob, packets = _make_odc_blob()
        lazy = LazyBlob(blob)
        subset = lazy.packets(1, 3)
        assert subset == packets[1:3]

    def test_all_packets_match_full_decode(self):
        blob, packets = _make_odc_blob()
        lazy = LazyBlob(blob)
        all_pkts = lazy.all_packets()
        assert all_pkts == packets

    def test_packet_caching(self):
        blob, packets = _make_odc_blob()
        lazy = LazyBlob(blob)
        # Access same packet twice
        p1 = lazy.packet(0)
        p2 = lazy.packet(0)
        assert p1 is p2  # should be same object (cached)

    def test_out_of_bounds_raises(self):
        blob, packets = _make_odc_blob()
        lazy = LazyBlob(blob)
        with pytest.raises(IndexError):
            lazy.packet(999)
        with pytest.raises(IndexError):
            lazy.packet(-1)

    def test_unknown_format(self):
        blob = b"XXXX" + b"\x00" * 100
        lazy = LazyBlob(blob)
        assert lazy.format() == "unknown"
        # Should still work as single-packet fallback
        assert lazy.num_packets() == 1
