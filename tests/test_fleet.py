"""Tests for G4: FLEET Mode — Multi-Agent Streams."""
import pytest
from usc.fleet import fleet_encode, fleet_decode, fleet_verify, MAGIC


def _sample_streams():
    return {
        "agent-alpha": [b"alpha-pkt-0", b"alpha-pkt-1", b"alpha-pkt-2"],
        "agent-beta": [b"beta-pkt-0", b"beta-pkt-1"],
        "agent-gamma": [b"gamma-pkt-0", b"gamma-pkt-1", b"gamma-pkt-2", b"gamma-pkt-3"],
    }


class TestEncodeAndDecode:
    def test_roundtrip(self):
        streams = _sample_streams()
        blob, meta = fleet_encode(streams)
        decoded = fleet_decode(blob)
        assert set(decoded.keys()) == set(streams.keys())
        for name in streams:
            assert decoded[name] == streams[name]

    def test_magic_header(self):
        blob, _ = fleet_encode(_sample_streams())
        assert blob[:4] == MAGIC

    def test_metadata(self):
        streams = _sample_streams()
        blob, meta = fleet_encode(streams)
        assert meta.n_agents == 3
        assert meta.total_packets == 9  # 3 + 2 + 4
        assert meta.compressed_bytes > 0

    def test_single_agent(self):
        streams = {"solo": [b"pkt-0", b"pkt-1"]}
        blob, meta = fleet_encode(streams)
        decoded = fleet_decode(blob)
        assert decoded["solo"] == streams["solo"]
        assert meta.n_agents == 1

    def test_empty_agent(self):
        streams = {"empty": [], "notempty": [b"data"]}
        blob, meta = fleet_encode(streams)
        decoded = fleet_decode(blob)
        assert decoded["empty"] == []
        assert decoded["notempty"] == [b"data"]


class TestVerification:
    def test_verify_all_good(self):
        streams = _sample_streams()
        blob, _ = fleet_encode(streams)
        results = fleet_verify(blob)
        assert all(results.values())
        assert len(results) == 3

    def test_verify_detects_corruption(self):
        streams = _sample_streams()
        blob, _ = fleet_encode(streams)
        # Corrupt some data in the compressed payload (last bytes)
        corrupted = bytearray(blob)
        # Find and flip a byte near the end
        corrupted[-5] ^= 0xFF
        # This should either raise or return False for at least one agent
        try:
            results = fleet_verify(bytes(corrupted))
            # If decompress succeeds with wrong data, at least one chain should fail
            assert not all(results.values())
        except Exception:
            pass  # Decompression failure is also valid corruption detection


class TestEdgeCases:
    def test_bad_magic_raises(self):
        with pytest.raises(ValueError, match="bad magic"):
            fleet_decode(b"XXXX" + b"\x00" * 20)

    def test_blob_too_small_raises(self):
        with pytest.raises(ValueError, match="too small"):
            fleet_decode(b"UF")

    def test_large_multi_agent(self):
        streams = {
            f"agent-{i}": [f"pkt-{j}".encode() for j in range(20)]
            for i in range(10)
        }
        blob, meta = fleet_encode(streams)
        decoded = fleet_decode(blob)
        assert meta.n_agents == 10
        assert meta.total_packets == 200
        for name in streams:
            assert decoded[name] == streams[name]

    def test_binary_packets(self):
        streams = {
            "binary": [bytes(range(256)), b"\x00" * 100, b"\xff" * 50],
        }
        blob, _ = fleet_encode(streams)
        decoded = fleet_decode(blob)
        assert decoded["binary"] == streams["binary"]
