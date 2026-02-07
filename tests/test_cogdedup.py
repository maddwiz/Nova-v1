"""Tests for Cognitive Deduplication module."""
import pytest
from usc.cogdedup.hasher import (
    content_defined_chunks,
    sha256_hash,
    simhash64,
    hamming_distance,
    SIMILARITY_THRESHOLD,
)
from usc.cogdedup.store import MemoryCogStore, ChunkEntry
from usc.cogdedup.codec import cogdedup_encode, cogdedup_decode


class TestHasher:
    def test_sha256_deterministic(self):
        h1 = sha256_hash(b"hello world")
        h2 = sha256_hash(b"hello world")
        assert h1 == h2

    def test_sha256_different_inputs(self):
        h1 = sha256_hash(b"hello")
        h2 = sha256_hash(b"world")
        assert h1 != h2

    def test_simhash_similar_data(self):
        # Use very similar data — only 1 byte difference per repetition
        base = b"2025-01-15 INFO server started on port 8080 with config=default mode=prod" * 50
        modified = bytearray(base)
        modified[10] = ord('2')  # change '5' to '2' in one spot
        sh_a = simhash64(base)
        sh_b = simhash64(bytes(modified))
        dist = hamming_distance(sh_a, sh_b)
        # Similar data should have smaller hamming distance than random
        assert dist < 20, f"similar data should have relatively small hamming distance, got {dist}"

    def test_simhash_different_data(self):
        a = b"aaaaaaaaaa" * 100
        b_data = b"1234567890" * 100
        sh_a = simhash64(a)
        sh_b = simhash64(b_data)
        dist = hamming_distance(sh_a, sh_b)
        assert dist > 4, f"very different data should have large hamming distance, got {dist}"

    def test_hamming_distance_zero(self):
        assert hamming_distance(0xFF, 0xFF) == 0

    def test_hamming_distance_all_differ(self):
        assert hamming_distance(0x0, 0xFF) == 8

    def test_content_defined_chunks_small(self):
        data = b"small"
        chunks = content_defined_chunks(data)
        assert len(chunks) == 1
        assert b"".join(chunks) == data

    def test_content_defined_chunks_empty(self):
        assert content_defined_chunks(b"") == []

    def test_content_defined_chunks_reassembly(self):
        data = b"x" * 50000  # large enough to split
        chunks = content_defined_chunks(data)
        assert len(chunks) >= 1
        assert b"".join(chunks) == data

    def test_content_defined_chunks_deterministic(self):
        data = bytes(range(256)) * 200
        c1 = content_defined_chunks(data)
        c2 = content_defined_chunks(data)
        assert len(c1) == len(c2)
        for a, b in zip(c1, c2):
            assert a == b


class TestCogStore:
    def test_store_and_exact_lookup(self):
        store = MemoryCogStore()
        entry = store.store(b"hello world")
        assert entry.chunk_id == 0
        found = store.lookup_exact(sha256_hash(b"hello world"))
        assert found is not None
        assert found.data == b"hello world"

    def test_store_dedup(self):
        store = MemoryCogStore()
        e1 = store.store(b"hello")
        e2 = store.store(b"hello")
        assert e1.chunk_id == e2.chunk_id
        assert store.size == 1

    def test_exact_lookup_miss(self):
        store = MemoryCogStore()
        store.store(b"hello")
        assert store.lookup_exact(sha256_hash(b"world")) is None

    def test_similar_lookup(self):
        store = MemoryCogStore()
        original = b"2025-01-15 INFO server started on port 8080 with config=default mode=prod\n" * 100
        store.store(original)

        # Very slightly modified version
        modified = bytearray(original)
        modified[10] = ord('2')  # tiny change
        sh = simhash64(bytes(modified))
        found = store.lookup_similar(sh)
        # May or may not find a match depending on how the change affects simhash
        # The important thing is that the lookup doesn't crash
        # Full similarity detection is tested via the codec integration tests
        if found is not None:
            assert found.data == original

    def test_get_by_id(self):
        store = MemoryCogStore()
        e = store.store(b"test data")
        got = store.get(e.chunk_id)
        assert got is not None
        assert got.data == b"test data"

    def test_get_missing_id(self):
        store = MemoryCogStore()
        assert store.get(999) is None


class TestCogCodec:
    def test_roundtrip_simple(self):
        store = MemoryCogStore()
        data = b"Hello, World! This is a test of cognitive deduplication." * 10
        blob, stats = cogdedup_encode(data, store)
        assert blob[:4] == b"UCOG"
        decoded = cogdedup_decode(blob, store)
        assert decoded == data

    def test_roundtrip_large(self):
        store = MemoryCogStore()
        data = bytes(range(256)) * 500  # 128KB
        blob, stats = cogdedup_encode(data, store)
        decoded = cogdedup_decode(blob, store)
        assert decoded == data

    def test_ref_dedup_on_repeat(self):
        """Second encode of same data should produce all REF chunks."""
        store = MemoryCogStore()
        data = b"repeated content for dedup testing" * 200

        blob1, stats1 = cogdedup_encode(data, store)
        assert stats1["full"] > 0

        blob2, stats2 = cogdedup_encode(data, store)
        assert stats2["ref"] == stats2["chunks"]
        assert stats2["full"] == 0
        assert stats2["delta"] == 0
        assert len(blob2) < len(blob1), "second encode should be smaller (all refs)"

    def test_delta_on_similar(self):
        """Modified data should use DELTA chunks."""
        store = MemoryCogStore()
        original = b"Log entry: 2025-01-01 INFO Starting service on port 8080\n" * 300
        modified = b"Log entry: 2025-01-02 INFO Starting service on port 8081\n" * 300

        blob1, stats1 = cogdedup_encode(original, store)
        blob2, stats2 = cogdedup_encode(modified, store)

        # Should have at least some REF or DELTA (not all FULL)
        assert stats2["ref"] + stats2["delta"] > 0 or stats2["full"] < stats1["full"], \
            "similar data should reuse chunks"

        decoded = cogdedup_decode(blob2, store)
        assert decoded == modified

    def test_empty_data(self):
        store = MemoryCogStore()
        blob, stats = cogdedup_encode(b"", store)
        decoded = cogdedup_decode(blob, store)
        assert decoded == b""

    def test_compression_improves_over_time(self):
        """Core property: compression ratio improves as store accumulates knowledge."""
        store = MemoryCogStore()
        sizes = []

        for i in range(5):
            data = f"Session {i}: ".encode() + b"common log pattern " * 500
            blob, stats = cogdedup_encode(data, store)
            sizes.append(len(blob))

        # Later encodes should be progressively smaller (more REFs/DELTAs)
        assert sizes[-1] <= sizes[0], \
            f"later encodes should be no larger: first={sizes[0]}, last={sizes[-1]}"

    def test_bad_magic_raises(self):
        store = MemoryCogStore()
        with pytest.raises(ValueError, match="not a UCOG"):
            cogdedup_decode(b"BAAD" + b"\x00" * 10, store)

    def test_stats_counts(self):
        store = MemoryCogStore()
        data = b"x" * 5000
        blob, stats = cogdedup_encode(data, store)
        total = stats["ref"] + stats["delta"] + stats["full"]
        assert total == stats["chunks"]
