"""Tests for USC-C3 bridge integration."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure imports work (monorepo src/ layout)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from c3ae.usc_bridge.compressed_vault import CompressedVault, _compress_smart, _decompress
from c3ae.usc_bridge.c3_cogstore import C3CogStore


class TestCompressedVault:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.vault = CompressedVault(self.tmpdir)

    def test_store_and_retrieve_small(self):
        """Small files should not be compressed."""
        data = b"hello world"
        h = self.vault.store_document(data, "test.txt")
        retrieved, meta = self.vault.get_document(h)
        assert retrieved == data
        assert meta["compression_method"] == "none"

    def test_store_and_retrieve_large(self):
        """Large files should be compressed."""
        data = b"log line: 2025-01-15 INFO starting server on port 8080\n" * 100
        h = self.vault.store_document(data, "server.log")
        retrieved, meta = self.vault.get_document(h)
        assert retrieved == data
        assert meta["compression_method"] != "none"
        assert meta["compression_ratio"] > 1.0

    def test_store_raw_log_compressed(self):
        """Session logs should be compressed."""
        log = "2025-01-15 INFO request processed in 42ms\n" * 200
        path = self.vault.store_raw_log(log, "session-123", "agent.log")
        assert path.exists()
        # Should have .usc extension
        assert path.suffix == ".usc"

    def test_compression_stats(self):
        data = b"structured log data for testing " * 100
        self.vault.store_document(data, "test.log")
        stats = self.vault.compression_stats()
        assert stats["document_count"] == 1
        assert stats["overall_ratio"] >= 1.0

    def test_list_documents(self):
        data = b"test content " * 100
        self.vault.store_document(data, "file1.txt")
        docs = self.vault.list_documents()
        assert len(docs) == 1
        assert "compression_method" in docs[0]


class TestSmartCompress:
    def test_small_data_not_compressed(self):
        compressed, method = _compress_smart(b"tiny")
        assert method == "none"
        assert compressed == b"tiny"

    def test_large_data_compressed(self):
        data = b"x" * 10000
        compressed, method = _compress_smart(data)
        assert method != "none"
        assert len(compressed) < len(data)

    def test_roundtrip(self):
        data = b"log: 2025-01-15 INFO server started\n" * 100
        compressed, method = _compress_smart(data)
        decompressed = _decompress(compressed, method)
        assert decompressed == data


class TestC3CogStore:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.store = C3CogStore(self.db_path)

    def teardown_method(self):
        self.store.close()

    def test_store_and_lookup(self):
        entry = self.store.store(b"test chunk data")
        assert entry.chunk_id > 0
        found = self.store.lookup_exact(entry.sha256)
        assert found is not None
        assert found.data == b"test chunk data"

    def test_dedup(self):
        e1 = self.store.store(b"same data")
        e2 = self.store.store(b"same data")
        assert e1.chunk_id == e2.chunk_id
        assert self.store.size == 1

    def test_get_by_id(self):
        entry = self.store.store(b"get me")
        got = self.store.get(entry.chunk_id)
        assert got is not None
        assert got.data == b"get me"

    def test_stats(self):
        self.store.store(b"chunk 1")
        self.store.store(b"chunk 2")
        self.store.store(b"chunk 1")  # duplicate
        stats = self.store.stats()
        assert stats["unique_chunks"] == 2
        assert stats["total_references"] >= 3  # 2 unique + 1 ref bump

    def test_persistence(self):
        """Data persists after closing and reopening."""
        self.store.store(b"persistent data")
        self.store.close()

        store2 = C3CogStore(self.db_path)
        from usc.cogdedup.hasher import sha256_hash
        found = store2.lookup_exact(sha256_hash(b"persistent data"))
        assert found is not None
        assert found.data == b"persistent data"
        store2.close()

    def test_cogdedup_roundtrip(self):
        """Full cognitive dedup encode/decode through C3 store."""
        from usc.cogdedup.codec import cogdedup_encode, cogdedup_decode

        data = b"session log entry: 2025-01-15 INFO processing request\n" * 100

        blob1, stats1 = cogdedup_encode(data, self.store)
        decoded = cogdedup_decode(blob1, self.store)
        assert decoded == data
        assert stats1["full"] > 0

        # Second encode should use references
        blob2, stats2 = cogdedup_encode(data, self.store)
        decoded2 = cogdedup_decode(blob2, self.store)
        assert decoded2 == data
        assert stats2["ref"] > 0
        assert len(blob2) < len(blob1)

    def test_cross_session_dedup(self):
        """Simulates two agent sessions sharing a store."""
        from usc.cogdedup.codec import cogdedup_encode, cogdedup_decode

        # Session 1: encode common agent patterns
        session1 = (
            b"[TOOL_CALL] web_search query='python logging'\n"
            b"[TOOL_RESULT] Found 5 results\n"
            b"[THINKING] Analyzing search results...\n"
        ) * 50

        blob1, stats1 = cogdedup_encode(session1, self.store)
        assert cogdedup_decode(blob1, self.store) == session1

        # Session 2: similar patterns should compress better
        session2 = (
            b"[TOOL_CALL] web_search query='python debugging'\n"
            b"[TOOL_RESULT] Found 3 results\n"
            b"[THINKING] Analyzing search results...\n"
        ) * 50

        blob2, stats2 = cogdedup_encode(session2, self.store)
        assert cogdedup_decode(blob2, self.store) == session2

        # Session 2 should benefit from session 1's stored chunks
        total_reuse = stats2["ref"] + stats2["delta"]
        assert total_reuse > 0 or len(blob2) <= len(blob1), \
            "cross-session encoding should reuse chunks"
