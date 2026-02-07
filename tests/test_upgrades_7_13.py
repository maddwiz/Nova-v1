"""Tests for Opus 4.6 Upgrades #7-#13.

#7:  Self-Compressing Context Windows (context_compactor)
#8:  Compression as Anomaly Detection (anomaly)
#9:  Cross-Agent Knowledge Transfer (federation)
#10: Adversarial Robustness (integrity)
#11: Recursive Self-Compression (recursive)
#12: Temporal Compression (temporal)
#13: Cost-Per-Retrieval Metric (tested in bench script)
"""
import pytest
from usc.cogdedup.store import MemoryCogStore
from usc.cogdedup.codec import cogdedup_encode, cogdedup_decode
from usc.cogdedup.predictor import PredictiveCompressor


# ===================== #7: Context Compactor =====================

class TestContextCompactor:
    def test_no_compression_on_novel_data(self):
        """Fresh store should not compress anything."""
        from usc.cogdedup.context_compactor import ContextCompactor
        store = MemoryCogStore()
        compactor = ContextCompactor(store, min_ref_count=2)

        prompt = "This is a brand new prompt that has never been seen before." * 50
        result = compactor.compress_prompt(prompt)

        assert result.refs_inserted == 0
        assert result.text == prompt

    def test_compression_after_repeated_storage(self):
        """Chunks seen multiple times should be replaced with refs."""
        from usc.cogdedup.context_compactor import ContextCompactor
        store = MemoryCogStore()

        # Store the same data multiple times to build ref_count
        data = b"This is a commonly repeated chunk of text that appears in many prompts. " * 100
        for _ in range(5):
            cogdedup_encode(data, store)

        compactor = ContextCompactor(store, min_ref_count=3, min_chunk_len=50)
        prompt = data.decode("utf-8")
        result = compactor.compress_prompt(prompt)

        # Should have some refs
        assert result.refs_inserted > 0
        assert len(result.text) < len(prompt)
        assert result.savings_pct > 0

    def test_expand_restores_original(self):
        """Expanding compressed text should restore original content."""
        from usc.cogdedup.context_compactor import ContextCompactor
        store = MemoryCogStore()

        data = b"Repeated pattern for context compaction testing. " * 100
        for _ in range(5):
            cogdedup_encode(data, store)

        compactor = ContextCompactor(store, min_ref_count=3, min_chunk_len=20)
        prompt = data.decode("utf-8")
        result = compactor.compress_prompt(prompt)

        if result.refs_inserted > 0:
            expanded = compactor.expand_prompt(result.text)
            assert expanded == prompt

    def test_stats_tracking(self):
        from usc.cogdedup.context_compactor import ContextCompactor
        store = MemoryCogStore()
        compactor = ContextCompactor(store)

        compactor.compress_prompt("test " * 500)
        compactor.compress_prompt("another test " * 500)

        stats = compactor.stats()
        assert stats["total_compressions"] == 2

    def test_empty_prompt(self):
        from usc.cogdedup.context_compactor import ContextCompactor
        store = MemoryCogStore()
        compactor = ContextCompactor(store)

        result = compactor.compress_prompt("")
        assert result.text == ""
        assert result.refs_inserted == 0


# ===================== #8: Anomaly Detection =====================

class TestAnomalyDetector:
    def test_no_alerts_on_stable_data(self):
        """Stable compression ratios should not trigger alerts."""
        from usc.cogdedup.anomaly import AnomalyDetector
        detector = AnomalyDetector(window_size=20)

        for i in range(30):
            alert = detector.observe(10.0 + (i % 3) * 0.1, label=f"session-{i}")

        # Small variations shouldn't trigger
        assert len(detector.alerts) == 0 or all(
            a.severity == "low" for a in detector.alerts
        )

    def test_alert_on_sudden_drop(self):
        """A sharp drop in ratio should trigger an alert."""
        from usc.cogdedup.anomaly import AnomalyDetector
        detector = AnomalyDetector(window_size=20, z_threshold_low=-2.0)

        # Build stable baseline
        for i in range(25):
            detector.observe(20.0, label=f"stable-{i}")

        # Sudden drop
        alert = detector.observe(1.0, label="anomaly")
        assert alert is not None
        assert alert.z_score < 0
        assert alert.severity in ("medium", "high")

    def test_alert_on_spike(self):
        """An abnormally high ratio should trigger an alert."""
        from usc.cogdedup.anomaly import AnomalyDetector
        detector = AnomalyDetector(window_size=20, z_threshold_high=2.0)

        for i in range(25):
            detector.observe(5.0, label=f"stable-{i}")

        alert = detector.observe(500.0, label="spike")
        assert alert is not None
        assert alert.z_score > 0

    def test_drift_report(self):
        """Drift report should detect improving trend."""
        from usc.cogdedup.anomaly import AnomalyDetector
        detector = AnomalyDetector(window_size=50)

        # Improving ratio over time
        for i in range(50):
            detector.observe(5.0 + i * 0.5, label=f"session-{i}")

        report = detector.drift_report()
        assert report.trend > 0  # Positive trend = improving
        assert report.window_size == 50

    def test_reset(self):
        from usc.cogdedup.anomaly import AnomalyDetector
        detector = AnomalyDetector()
        for i in range(10):
            detector.observe(10.0)
        assert detector.observation_count == 10
        detector.reset()
        assert detector.observation_count == 0


# ===================== #9: Federation =====================

class TestFederation:
    def test_create_agents(self):
        """Federation should create separate agent stores."""
        from usc.cogdedup.federation import CogstoreFederation
        federation = CogstoreFederation(promote_threshold=3)

        store_a = federation.create_agent_store("agent-a")
        store_b = federation.create_agent_store("agent-b")

        assert store_a.agent_id == "agent-a"
        assert store_b.agent_id == "agent-b"
        assert len(federation.agent_ids) == 2

    def test_local_isolation(self):
        """Agents should not see each other's local data."""
        from usc.cogdedup.federation import CogstoreFederation
        federation = CogstoreFederation(promote_threshold=100)

        store_a = federation.create_agent_store("agent-a")
        store_b = federation.create_agent_store("agent-b")

        data_a = b"agent A private data with enough length for chunking " * 50
        entry_a = store_a.store(data_a)

        # B shouldn't find A's data (not promoted yet)
        from usc.cogdedup.hasher import sha256_hash
        sha = sha256_hash(data_a)
        assert store_b.lookup_exact(sha) is None

    def test_promotion_to_shared(self):
        """High-frequency chunks get promoted to shared tier."""
        from usc.cogdedup.federation import CogstoreFederation
        federation = CogstoreFederation(promote_threshold=3)

        store_a = federation.create_agent_store("agent-a")
        store_b = federation.create_agent_store("agent-b")

        # Store same data 3+ times in agent A (triggers promotion)
        data = b"common pattern shared across agents " * 50
        for _ in range(4):
            store_a.store(data)

        # Now agent B should find it in the shared tier
        from usc.cogdedup.hasher import sha256_hash
        sha = sha256_hash(data)
        result = store_b.lookup_exact(sha)
        assert result is not None

    def test_federation_encode_decode(self):
        """Full encode/decode should work through federated stores."""
        from usc.cogdedup.federation import CogstoreFederation
        federation = CogstoreFederation(promote_threshold=2)

        store_a = federation.create_agent_store("agent-a")

        data = b"federated encode test data " * 500
        blob, stats = cogdedup_encode(data, store_a)
        decoded = cogdedup_decode(blob, store_a)
        assert decoded == data

    def test_federation_stats(self):
        from usc.cogdedup.federation import CogstoreFederation
        federation = CogstoreFederation()
        federation.create_agent_store("a")
        federation.create_agent_store("b")

        stats = federation.federation_stats()
        assert stats["total_agents"] == 2
        assert "shared" in stats
        assert "a" in stats["agents"]


# ===================== #10: Adversarial Robustness =====================

class TestIntegrity:
    def test_hash_roundtrip(self):
        """Hash computation should be deterministic."""
        from usc.cogdedup.integrity import fast_hash, fast_hash_bytes, verify_hash
        data = b"test data for hashing"
        h1 = fast_hash(data)
        h2 = fast_hash(data)
        assert h1 == h2

        hb = fast_hash_bytes(data)
        assert verify_hash(data, hb)

    def test_corrupt_data_fails_verify(self):
        from usc.cogdedup.integrity import fast_hash_bytes, verify_hash
        data = b"original data"
        hb = fast_hash_bytes(data)
        assert not verify_hash(b"corrupted data", hb)

    def test_integrity_verifier(self):
        from usc.cogdedup.integrity import IntegrityVerifier
        verifier = IntegrityVerifier()

        data = b"test chunk data"
        h = verifier.compute_hash(data)
        assert verifier.verify(data, h)
        assert not verifier.verify(b"wrong data", h)

        stats = verifier.stats()
        assert stats["verified"] == 1
        assert stats["failed"] == 1

    def test_delta_expansion_check(self):
        from usc.cogdedup.integrity import IntegrityVerifier, SecurityPolicy
        policy = SecurityPolicy(max_delta_expansion=10.0)
        verifier = IntegrityVerifier(policy)

        assert verifier.check_delta_expansion(100, 500)   # 5x OK
        assert not verifier.check_delta_expansion(100, 1500)  # 15x too much

    def test_ref_count_check(self):
        from usc.cogdedup.integrity import IntegrityVerifier, SecurityPolicy
        policy = SecurityPolicy(max_ref_count_for_similarity=50)
        verifier = IntegrityVerifier(policy)

        assert verifier.check_ref_count(10)
        assert verifier.check_ref_count(50)
        assert not verifier.check_ref_count(51)

    def test_security_policy_defaults(self):
        from usc.cogdedup.integrity import SecurityPolicy
        policy = SecurityPolicy()
        assert policy.max_ref_count_for_similarity == 1000
        assert policy.verify_deltas is True
        assert policy.max_delta_expansion == 100.0


# ===================== #11: Recursive Self-Compression =====================

class TestRecursiveCompressor:
    def test_compress_memories_roundtrip(self):
        """Compress and decompress memory dicts."""
        from usc.cogdedup.recursive import RecursiveCompressor
        store = MemoryCogStore()
        compressor = RecursiveCompressor(store)

        memories = [
            {"id": f"mem-{i}", "content": f"Memory about topic {i % 5}" * 100,
             "timestamp": 1000 + i}
            for i in range(20)
        ]

        result = compressor.compress_memories(memories, batch_id="test")
        assert result.ratio >= 1.0
        assert result.items_count == 20

        decompressed = compressor.decompress_memories(result.blob)
        assert len(decompressed) == 20
        for orig, dec in zip(memories, decompressed):
            assert orig == dec

    def test_compress_reasoning_bank_roundtrip(self):
        from usc.cogdedup.recursive import RecursiveCompressor
        store = MemoryCogStore()
        compressor = RecursiveCompressor(store)

        entries = [
            f"Reasoning step {i}: analyzing the problem from angle {i % 3}" * 50
            for i in range(10)
        ]

        result = compressor.compress_reasoning_bank(entries, batch_id="r1")
        decompressed = compressor.decompress_reasoning_bank(result.blob)
        assert decompressed == entries

    def test_compress_audit_log_roundtrip(self):
        from usc.cogdedup.recursive import RecursiveCompressor
        store = MemoryCogStore()
        compressor = RecursiveCompressor(store)

        events = [
            {"event": "tool_call", "tool": "web_search", "ts": 1000 + i}
            for i in range(15)
        ]

        result = compressor.compress_audit_log(events, batch_id="a1")
        decompressed = compressor.decompress_audit_log(result.blob)
        assert decompressed == events

    def test_recursive_improvement(self):
        """Second compression of similar data should be better."""
        from usc.cogdedup.recursive import RecursiveCompressor
        store = MemoryCogStore()
        compressor = RecursiveCompressor(store)

        # First batch
        memories1 = [
            {"id": f"m-{i}", "content": "Common pattern across batches " * 50}
            for i in range(10)
        ]
        r1 = compressor.compress_memories(memories1, batch_id="b1")

        # Second batch (similar content)
        memories2 = [
            {"id": f"n-{i}", "content": "Common pattern across batches " * 50}
            for i in range(10)
        ]
        r2 = compressor.compress_memories(memories2, batch_id="b2")

        # Second should be better compressed (more REFs)
        assert r2.ratio >= r1.ratio or r2.compressed_size <= r1.compressed_size

    def test_stats(self):
        from usc.cogdedup.recursive import RecursiveCompressor
        store = MemoryCogStore()
        compressor = RecursiveCompressor(store)

        compressor.compress_memories([{"x": "y" * 500}])
        stats = compressor.stats()
        assert stats["batches_compressed"] == 1
        assert stats["total_original_bytes"] > 0


# ===================== #12: Temporal Compression =====================

class TestTemporalMotifTracker:
    def test_detect_simple_motif(self):
        """Repeated event sequence should be detected as motif."""
        from usc.cogdedup.temporal import TemporalMotifTracker
        tracker = TemporalMotifTracker(min_pattern_len=3, min_occurrences=2)

        # Feed a repeating pattern
        pattern = ["search", "read", "think"]
        for _ in range(5):
            for event in pattern:
                tracker.observe(event)

        motifs = tracker.detected_motifs()
        assert len(motifs) > 0
        # The exact pattern should be among detected motifs
        patterns = [m.pattern for m in motifs]
        assert tuple(pattern) in patterns

    def test_observe_returns_motif_on_completion(self):
        from usc.cogdedup.temporal import TemporalMotifTracker
        tracker = TemporalMotifTracker(min_pattern_len=3, min_occurrences=2)

        # First occurrence
        for e in ["a", "b", "c"]:
            tracker.observe(e)

        # Second occurrence — should return motif on completion
        tracker.observe("a")
        tracker.observe("b")
        result = tracker.observe("c")
        assert result is not None
        assert result.pattern == ("a", "b", "c")

    def test_observe_batch(self):
        from usc.cogdedup.temporal import TemporalMotifTracker
        tracker = TemporalMotifTracker(min_pattern_len=3, min_occurrences=2)

        events = ["x", "y", "z"] * 5
        tracker.observe_batch(events)
        assert tracker.motif_count > 0

    def test_stats(self):
        from usc.cogdedup.temporal import TemporalMotifTracker
        tracker = TemporalMotifTracker()
        tracker.observe_batch(["a", "b", "c"] * 3)
        stats = tracker.stats()
        assert stats["events_observed"] == 9


class TestTemporalEncoder:
    def test_encode_decode_roundtrip(self):
        from usc.cogdedup.temporal import TemporalMotifTracker, TemporalEncoder
        tracker = TemporalMotifTracker(min_pattern_len=3, min_occurrences=2)

        events = ["search", "read", "think", "search", "read", "think", "respond"]
        tracker.observe_batch(events)

        encoder = TemporalEncoder(tracker)
        result = encoder.encode(events)
        decoded = encoder.decode(result)
        assert decoded == events

    def test_compression_savings(self):
        from usc.cogdedup.temporal import TemporalMotifTracker, TemporalEncoder
        tracker = TemporalMotifTracker(min_pattern_len=3, min_occurrences=2)

        events = ["a", "b", "c", "d"] * 20
        tracker.observe_batch(events)

        encoder = TemporalEncoder(tracker)
        result = encoder.encode(events)

        assert result.compressed_tokens < result.original_events
        assert result.savings_pct > 0
        assert result.motifs_used > 0

    def test_no_compression_on_unique(self):
        from usc.cogdedup.temporal import TemporalMotifTracker, TemporalEncoder
        tracker = TemporalMotifTracker(min_pattern_len=3, min_occurrences=2)

        events = [f"unique-event-{i}" for i in range(20)]
        tracker.observe_batch(events)

        encoder = TemporalEncoder(tracker)
        result = encoder.encode(events)

        assert result.compressed_tokens == result.original_events
        assert result.savings_pct == 0.0


# ===================== Integration Tests =====================

class TestIntegration:
    def test_all_imports(self):
        """All new modules should import cleanly."""
        from usc.cogdedup import (
            ContextCompactor, CompactionResult,
            AnomalyDetector, AnomalyAlert, DriftReport,
            CogstoreFederation, FederatedStore,
            IntegrityVerifier, SecurityPolicy,
            RecursiveCompressor, CompressionResult,
            TemporalMotifTracker, TemporalEncoder,
        )

    def test_full_pipeline(self):
        """End-to-end: encode with federation, check anomaly, compress recursively."""
        from usc.cogdedup.federation import CogstoreFederation
        from usc.cogdedup.anomaly import AnomalyDetector
        from usc.cogdedup.recursive import RecursiveCompressor

        # Setup
        federation = CogstoreFederation(promote_threshold=2)
        store = federation.create_agent_store("nova")
        detector = AnomalyDetector(window_size=10)
        compressor = RecursiveCompressor(store)

        # Simulate sessions
        for i in range(15):
            data = f"Session {i}: ".encode() + b"common agent output pattern " * 300
            blob, stats = cogdedup_encode(data, store)
            ratio = len(data) / max(1, len(blob))
            detector.observe(ratio, label=f"session-{i}")

            # Decode check
            decoded = cogdedup_decode(blob, store)
            assert decoded == data

        # Check drift
        report = detector.drift_report()
        assert report.window_size > 0

        # Recursive compression of memories
        memories = [{"session": i, "data": f"memory {i}" * 100} for i in range(5)]
        result = compressor.compress_memories(memories)
        assert result.ratio >= 1.0
        dec = compressor.decompress_memories(result.blob)
        assert dec == memories
