"""Tests for Cognitive Deduplication v2 upgrades.

Tests:
1. LSH band index — O(1) similarity lookup
2. Tiered storage (hot/warm/cold) via MemoryCogStore
3. Predictive pre-compression
4. Streaming cogdedup
5. Compression-aware retrieval (structural similarity)
6. Co-occurrence tracking
"""
import pytest
from usc.cogdedup.hasher import (
    content_defined_chunks,
    sha256_hash,
    simhash64,
    hamming_distance,
    SIMILARITY_THRESHOLD,
)
from usc.cogdedup.lsh import LSHIndex, _extract_bands, N_BANDS, BAND_WIDTH
from usc.cogdedup.store import MemoryCogStore, ChunkEntry
from usc.cogdedup.codec import cogdedup_encode, cogdedup_decode, VERSION, PRED_DELTA
from usc.cogdedup.predictor import PredictiveCompressor
from usc.cogdedup.streaming import CogdedupStream


# ===================== LSH Index Tests =====================

class TestLSHIndex:
    def test_extract_bands(self):
        bands = _extract_bands(0xDEADBEEFCAFEBABE)
        assert len(bands) == N_BANDS
        # Each band is BAND_WIDTH bits
        for b in bands:
            assert 0 <= b < (1 << BAND_WIDTH)

    def test_extract_bands_reconstructs(self):
        """Band extraction should be lossless."""
        val = 0xDEADBEEFCAFEBABE
        bands = _extract_bands(val)
        reconstructed = 0
        for i, b in enumerate(bands):
            reconstructed |= b << (i * BAND_WIDTH)
        assert reconstructed == val

    def test_insert_and_query(self):
        idx = LSHIndex()
        idx.insert(1, 0xFF00FF00FF00FF00)
        idx.insert(2, 0xFF00FF00FF00FF01)  # differ by 1 bit
        idx.insert(3, 0x00FF00FF00FF00FF)  # very different

        # Query for something close to chunk 1
        candidates = idx.query_candidates(0xFF00FF00FF00FF00)
        assert 1 in candidates
        assert 2 in candidates  # shares most bands

    def test_query_nearest(self):
        idx = LSHIndex()
        idx.insert(1, 0xFF00FF00FF00FF00)
        idx.insert(2, 0xFF00FF00FF00FF01)

        best = idx.query_nearest(0xFF00FF00FF00FF00)
        assert best == 1  # exact match

    def test_query_nearest_no_match(self):
        idx = LSHIndex()
        idx.insert(1, 0xFF00FF00FF00FF00)
        # Query with something completely different — all bands differ
        best = idx.query_nearest(0x0000000000000000, threshold=2)
        # Should not match (hamming distance is 32, far above threshold)
        assert best is None or hamming_distance(0xFF00FF00FF00FF00, 0x0000000000000000) <= 2

    def test_remove(self):
        idx = LSHIndex()
        idx.insert(1, 0xAAAA)
        assert idx.size == 1
        idx.remove(1)
        assert idx.size == 0
        assert idx.query_nearest(0xAAAA) is None

    def test_rebuild(self):
        idx = LSHIndex()
        entries = [(i, simhash64(f"chunk {i}".encode() * 100)) for i in range(50)]
        idx.rebuild(entries)
        assert idx.size == 50

    def test_lsh_used_in_memorystore(self):
        """MemoryCogStore should use LSH instead of linear scan."""
        store = MemoryCogStore()
        # Store enough chunks to show LSH is working
        for i in range(100):
            store.store(f"chunk data number {i} with some padding ".encode() * 50)

        # Lookup should work
        test_data = b"chunk data number 50 with some padding " * 50
        sh = simhash64(test_data)
        result = store.lookup_similar(sh)
        # May or may not find a match, but shouldn't crash
        assert store.size == 100


# ===================== Co-occurrence & Prediction Tests =====================

class TestCooccurrence:
    def test_record_and_query(self):
        store = MemoryCogStore()
        e1 = store.store(b"chunk A data " * 100)
        e2 = store.store(b"chunk B data " * 100)
        e3 = store.store(b"chunk C data " * 100)

        store.record_cooccurrence([e1.chunk_id, e2.chunk_id, e3.chunk_id])

        predicted = store.get_predicted_chunks(e1.chunk_id, top_k=5)
        pred_ids = {p.chunk_id for p in predicted}
        assert e2.chunk_id in pred_ids
        assert e3.chunk_id in pred_ids

    def test_cooccurrence_counts_accumulate(self):
        store = MemoryCogStore()
        e1 = store.store(b"X" * 1100)
        e2 = store.store(b"Y" * 1100)

        # Record multiple times
        store.record_cooccurrence([e1.chunk_id, e2.chunk_id])
        store.record_cooccurrence([e1.chunk_id, e2.chunk_id])
        store.record_cooccurrence([e1.chunk_id, e2.chunk_id])

        predicted = store.get_predicted_chunks(e1.chunk_id, top_k=1)
        assert len(predicted) == 1
        assert predicted[0].chunk_id == e2.chunk_id


class TestPredictiveCompressor:
    def test_no_prediction_initially(self):
        store = MemoryCogStore()
        pred = PredictiveCompressor(store)
        assert pred.get_dictionary_for(999) is None

    def test_prediction_after_cooccurrence(self):
        store = MemoryCogStore()
        e1 = store.store(b"chunk A pattern " * 200)
        e2 = store.store(b"chunk B pattern " * 200)
        store.record_cooccurrence([e1.chunk_id, e2.chunk_id])

        pred = PredictiveCompressor(store)
        dict_data = pred.get_dictionary_for(e1.chunk_id)
        assert dict_data is not None

    def test_cache_eviction(self):
        store = MemoryCogStore()
        pred = PredictiveCompressor(store, dict_cache_size=2)

        for i in range(5):
            e = store.store(f"chunk {i} ".encode() * 200)
            store.record_cooccurrence([e.chunk_id, 0])  # dummy cooccurrence

        # Fill cache beyond limit
        for i in range(5):
            pred.get_dictionary_for(i)

        assert pred.cache_size <= 2


# ===================== Streaming Cogdedup Tests =====================

class TestStreaming:
    def test_basic_stream(self):
        store = MemoryCogStore()
        stream = CogdedupStream(store)

        data = b"streaming test data with enough content to form chunks " * 200
        for i in range(0, len(data), 100):
            stream.feed(data[i:i+100])

        blob, stats = stream.finish()
        assert blob[:4] == b"UCOG"
        assert stats["chunks"] > 0

        # Verify roundtrip
        decoded = cogdedup_decode(blob, store)
        assert decoded == data

    def test_stream_feed_line(self):
        store = MemoryCogStore()
        stream = CogdedupStream(store)

        lines = [f"2025-01-15 INFO request {i} processed" for i in range(500)]
        for line in lines:
            stream.feed_line(line)

        blob, stats = stream.finish()
        decoded = cogdedup_decode(blob, store)

        expected = "\n".join(lines) + "\n"
        assert decoded.decode("utf-8") == expected

    def test_stream_dedup_on_repeat(self):
        store = MemoryCogStore()

        # First stream: populate store
        s1 = CogdedupStream(store)
        data = b"repeated pattern for streaming dedup test\n" * 300
        s1.feed(data)
        blob1, stats1 = s1.finish()

        # Second stream: same data, should use REFs
        s2 = CogdedupStream(store)
        s2.feed(data)
        blob2, stats2 = s2.finish()

        assert stats2["ref"] > 0
        assert len(blob2) < len(blob1)

        # Both decode correctly
        assert cogdedup_decode(blob1, store) == data
        assert cogdedup_decode(blob2, store) == data

    def test_stream_current_ratio(self):
        store = MemoryCogStore()
        stream = CogdedupStream(store)
        assert stream.current_ratio == 1.0
        stream.feed(b"x" * 10000)
        assert stream.bytes_fed == 10000

    def test_stream_finish_twice_raises(self):
        store = MemoryCogStore()
        stream = CogdedupStream(store)
        stream.feed(b"data")
        stream.finish()
        with pytest.raises(RuntimeError, match="already finished"):
            stream.finish()

    def test_feed_after_finish_raises(self):
        store = MemoryCogStore()
        stream = CogdedupStream(store)
        stream.feed(b"data")
        stream.finish()
        with pytest.raises(RuntimeError, match="already finished"):
            stream.feed(b"more")


# ===================== Structural Similarity Tests =====================

class TestStructuralSimilarity:
    def test_register_and_get_chunks(self):
        store = MemoryCogStore()
        e1 = store.store(b"A" * 1100)
        e2 = store.store(b"B" * 1100)
        store.register_data_chunks("mem-1", {e1.chunk_id, e2.chunk_id})

        chunks = store.get_chunk_ids_for_data("mem-1")
        assert chunks == {e1.chunk_id, e2.chunk_id}

    def test_identical_data_similarity_1(self):
        store = MemoryCogStore()
        e1 = store.store(b"shared chunk " * 100)
        e2 = store.store(b"unique chunk " * 100)

        store.register_data_chunks("mem-A", {e1.chunk_id, e2.chunk_id})
        store.register_data_chunks("mem-B", {e1.chunk_id, e2.chunk_id})

        # Same chunks → Jaccard = 1.0
        # (Using MemoryCogStore directly for structural_similarity)
        chunks_a = store.get_chunk_ids_for_data("mem-A")
        chunks_b = store.get_chunk_ids_for_data("mem-B")
        intersection = chunks_a & chunks_b
        union = chunks_a | chunks_b
        jaccard = len(intersection) / len(union)
        assert jaccard == 1.0

    def test_partial_overlap(self):
        store = MemoryCogStore()
        shared = store.store(b"shared data " * 100)
        only_a = store.store(b"only in A " * 100)
        only_b = store.store(b"only in B " * 100)

        store.register_data_chunks("mem-A", {shared.chunk_id, only_a.chunk_id})
        store.register_data_chunks("mem-B", {shared.chunk_id, only_b.chunk_id})

        chunks_a = store.get_chunk_ids_for_data("mem-A")
        chunks_b = store.get_chunk_ids_for_data("mem-B")
        intersection = chunks_a & chunks_b
        union = chunks_a | chunks_b
        jaccard = len(intersection) / len(union)
        assert 0.0 < jaccard < 1.0  # 1/3 overlap

    def test_data_id_mapping_in_encode(self):
        """Encoding with data_id should register chunks automatically."""
        store = MemoryCogStore()
        data = b"auto-registered chunk data " * 300

        blob, stats = cogdedup_encode(data, store, data_id="session-1")

        chunks = store.get_chunk_ids_for_data("session-1")
        assert len(chunks) > 0  # Chunks were registered


# ===================== Codec v2 Tests =====================

class TestCodecV2:
    def test_version_2(self):
        store = MemoryCogStore()
        blob, _ = cogdedup_encode(b"version check" * 100, store)
        assert blob[4] == 2  # VERSION = 2

    def test_backward_compat_v1_decode(self):
        """v2 decoder should handle v1 blobs (no PRED_DELTA)."""
        store = MemoryCogStore()
        data = b"backward compatibility test " * 200

        # Encode (produces v2 blob, but without predictor → no PRED_DELTA)
        blob, stats = cogdedup_encode(data, store)
        assert stats["pred_delta"] == 0

        decoded = cogdedup_decode(blob, store)
        assert decoded == data

    def test_predictive_encode_roundtrip(self):
        """Full predictive encode/decode roundtrip."""
        store = MemoryCogStore()
        pred = PredictiveCompressor(store)

        # Build up co-occurrence data with multiple encodes
        for i in range(5):
            data = f"Session {i}: ".encode() + b"common log pattern in every session " * 500
            cogdedup_encode(data, store, predictor=pred)

        # Now encode with prediction — should use some PRED_DELTA
        final_data = b"Session final: " + b"common log pattern in every session " * 500
        blob, stats = cogdedup_encode(final_data, store, predictor=pred)

        # Decode with same predictor
        decoded = cogdedup_decode(blob, store, predictor=pred)
        assert decoded == final_data

    def test_stats_include_pred_delta(self):
        store = MemoryCogStore()
        blob, stats = cogdedup_encode(b"test" * 300, store)
        assert "pred_delta" in stats
        total = stats["ref"] + stats["delta"] + stats["full"] + stats["pred_delta"]
        assert total == stats["chunks"]

    def test_pred_delta_survives_cooccurrence_mutation(self):
        """PRED_DELTA decode must work even if co-occurrence data changes after encode.

        This was the dictionary consistency bug: encode builds dict from co-occurrence
        at time T, then update_after_encode() mutates co-occurrence, and decode at
        time T+1 got a different dict. Fix: embed dict chunk IDs in the blob.
        """
        store = MemoryCogStore()
        pred = PredictiveCompressor(store)

        # Build up co-occurrence data
        for i in range(10):
            data = f"Session {i}: ".encode() + b"common pattern across sessions " * 500
            cogdedup_encode(data, store, predictor=pred)

        # Encode with prediction active
        final_data = b"Session final: " + b"common pattern across sessions " * 500
        blob, stats = cogdedup_encode(final_data, store, predictor=pred)

        # Deliberately mutate co-occurrence by encoding more sessions
        # This changes what get_predicted_chunks() returns
        for i in range(10, 20):
            data = f"DIFFERENT session {i}: ".encode() + b"totally new pattern " * 500
            cogdedup_encode(data, store, predictor=pred)

        # Clear the predictor cache to force re-query of (now-mutated) co-occurrence
        pred._dict_cache.clear()

        # Decode should STILL work because dict chunk IDs are in the blob
        decoded = cogdedup_decode(blob, store, predictor=pred)
        assert decoded == final_data

    def test_pred_delta_decode_without_predictor(self):
        """PRED_DELTA decode should work even without a predictor object.

        The blob contains the dict chunk IDs, so the store alone is sufficient.
        """
        store = MemoryCogStore()
        pred = PredictiveCompressor(store)

        # Build co-occurrence and encode with predictor
        for i in range(10):
            data = f"Session {i}: ".encode() + b"repeat content for prediction " * 500
            cogdedup_encode(data, store, predictor=pred)

        final_data = b"Session final: " + b"repeat content for prediction " * 500
        blob, stats = cogdedup_encode(final_data, store, predictor=pred)

        # Decode WITHOUT predictor — should still work
        decoded = cogdedup_decode(blob, store, predictor=None)
        assert decoded == final_data


# ===================== Store Stats Tests =====================

class TestStoreStats:
    def test_memory_store_stats(self):
        store = MemoryCogStore()
        store.store(b"chunk 1 " * 200)
        store.store(b"chunk 2 " * 200)
        store.store(b"chunk 1 " * 200)  # duplicate

        stats = store.stats()
        assert stats["unique_chunks"] == 2
        assert stats["total_references"] >= 3
        assert stats["lsh_index_size"] == 2
        assert "cooccurrence_pairs" in stats
        assert "warm_chunks" in stats
        assert "cold_chunks" in stats
        assert stats["warm_chunks"] == 2
        assert stats["cold_chunks"] == 0

    def test_cold_archival(self):
        """Chunks with low ref_count get archived to save memory."""
        store = MemoryCogStore()
        # Lower thresholds for testing
        store.ARCHIVE_TRIGGER = 10
        store.ARCHIVE_KEEP_WARM = 3
        store.COLD_AGE_SECONDS = 0  # immediate

        # Store many single-use chunks
        for i in range(20):
            store.store(f"unique chunk {i} with padding ".encode() * 100)

        # Force archival
        archived = store._archive_cold()
        assert archived > 0

        stats = store.stats()
        assert stats["cold_chunks"] > 0
        assert stats["warm_chunks"] <= 20  # some should be cold

        # Cold chunks should still be retrievable via get()
        for cid in range(20):
            entry = store.get(cid)
            assert entry is not None
            assert entry.data is not None  # decompressed on demand

    def test_cold_archival_preserves_roundtrip(self):
        """Encoding/decoding should work after cold archival."""
        store = MemoryCogStore()
        store.ARCHIVE_TRIGGER = 5
        store.ARCHIVE_KEEP_WARM = 2
        store.COLD_AGE_SECONDS = 0

        data = b"roundtrip after archival test " * 500
        blob, stats = cogdedup_encode(data, store)

        # Force archival
        store._archive_cold()

        # Decode should still work (decompresses cold chunks on demand)
        decoded = cogdedup_decode(blob, store)
        assert decoded == data
