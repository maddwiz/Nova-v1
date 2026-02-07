"""Tests for G2: Merkle Integrity Chains."""
from usc.merkle import (
    build_chain,
    verify_chain,
    verify_packet,
    hash_packet,
    serialize_chain,
    deserialize_chain,
    MAGIC,
)


def _sample_packets():
    return [b"packet-0-hello", b"packet-1-world", b"packet-2-data"]


class TestBuildAndVerify:
    def test_build_returns_correct_length(self):
        pkts = _sample_packets()
        chain = build_chain(pkts)
        assert len(chain) == 3

    def test_verify_passes_on_good_data(self):
        pkts = _sample_packets()
        chain = build_chain(pkts)
        assert verify_chain(chain, pkts) is True

    def test_verify_fails_on_tampered_packet(self):
        pkts = _sample_packets()
        chain = build_chain(pkts)
        tampered = [pkts[0], b"TAMPERED", pkts[2]]
        assert verify_chain(chain, tampered) is False

    def test_verify_fails_on_extra_packet(self):
        pkts = _sample_packets()
        chain = build_chain(pkts)
        assert verify_chain(chain, pkts + [b"extra"]) is False

    def test_verify_fails_on_missing_packet(self):
        pkts = _sample_packets()
        chain = build_chain(pkts)
        assert verify_chain(chain, pkts[:2]) is False

    def test_verify_fails_on_reordered_packets(self):
        pkts = _sample_packets()
        chain = build_chain(pkts)
        reordered = [pkts[2], pkts[0], pkts[1]]
        assert verify_chain(chain, reordered) is False


class TestVerifySinglePacket:
    def test_verify_packet_at_index_0(self):
        pkts = _sample_packets()
        chain = build_chain(pkts)
        assert verify_packet(chain, 0, pkts[0]) is True

    def test_verify_packet_at_last_index(self):
        pkts = _sample_packets()
        chain = build_chain(pkts)
        assert verify_packet(chain, 2, pkts[2]) is True

    def test_verify_packet_fails_wrong_data(self):
        pkts = _sample_packets()
        chain = build_chain(pkts)
        assert verify_packet(chain, 1, b"wrong") is False

    def test_verify_packet_out_of_bounds(self):
        pkts = _sample_packets()
        chain = build_chain(pkts)
        assert verify_packet(chain, 5, pkts[0]) is False
        assert verify_packet(chain, -1, pkts[0]) is False


class TestSerialization:
    def test_serialize_deserialize_roundtrip(self):
        pkts = _sample_packets()
        chain = build_chain(pkts)
        blob = serialize_chain(chain)
        restored = deserialize_chain(blob)
        assert len(restored) == len(chain)
        for a, b in zip(chain.entries, restored.entries):
            assert a.packet_hash == b.packet_hash
            assert a.prev_hash == b.prev_hash

    def test_magic_header(self):
        chain = build_chain(_sample_packets())
        blob = serialize_chain(chain)
        assert blob[:4] == MAGIC

    def test_deserialized_chain_still_verifies(self):
        pkts = _sample_packets()
        chain = build_chain(pkts)
        blob = serialize_chain(chain)
        restored = deserialize_chain(blob)
        assert verify_chain(restored, pkts) is True

    def test_bad_magic_raises(self):
        import pytest
        with pytest.raises(ValueError, match="bad magic"):
            deserialize_chain(b"XXXX" + b"\x00" * 4)

    def test_truncated_blob_raises(self):
        import pytest
        chain = build_chain(_sample_packets())
        blob = serialize_chain(chain)
        with pytest.raises(ValueError):
            deserialize_chain(blob[:20])


class TestEdgeCases:
    def test_empty_packets(self):
        chain = build_chain([])
        assert len(chain) == 0
        assert chain.root_hash() == b"\x00" * 32
        assert verify_chain(chain, []) is True

    def test_single_packet(self):
        pkts = [b"only-one"]
        chain = build_chain(pkts)
        assert len(chain) == 1
        assert verify_chain(chain, pkts) is True
        assert verify_packet(chain, 0, pkts[0]) is True

    def test_large_chain(self):
        pkts = [f"packet-{i}".encode() for i in range(1000)]
        chain = build_chain(pkts)
        assert len(chain) == 1000
        assert verify_chain(chain, pkts) is True

    def test_hash_packet_deterministic(self):
        a = hash_packet(b"test")
        b = hash_packet(b"test")
        assert a == b
        assert len(a) == 32

    def test_hash_packet_distinct(self):
        a = hash_packet(b"test-a")
        b = hash_packet(b"test-b")
        assert a != b

    def test_root_hash_changes_with_different_data(self):
        chain1 = build_chain([b"a", b"b"])
        chain2 = build_chain([b"a", b"c"])
        assert chain1.root_hash() != chain2.root_hash()
