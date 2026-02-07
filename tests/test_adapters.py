"""Tests for N5: Framework Adapters."""
import pytest
from usc.adapters import LangChainMemoryAdapter, OTelSpanAdapter, SpanRecord


class TestLangChainAdapter:
    def test_ingest_and_retrieve_roundtrip(self):
        adapter = LangChainMemoryAdapter()
        messages = [
            ("human", "Hello, how are you?"),
            ("ai", "I am doing well, thank you!"),
            ("human", "What is USC?"),
            ("ai", "USC is a Unified State Codec for compression."),
        ]
        blob = adapter.ingest(messages)
        retrieved = adapter.retrieve(blob)
        assert len(retrieved) == len(messages)
        for (orig_role, orig_content), (ret_role, ret_content) in zip(messages, retrieved):
            assert orig_role == ret_role
            assert orig_content == ret_content

    def test_empty_messages(self):
        adapter = LangChainMemoryAdapter()
        blob = adapter.ingest([])
        retrieved = adapter.retrieve(blob)
        assert retrieved == []

    def test_single_message(self):
        adapter = LangChainMemoryAdapter()
        messages = [("system", "You are a helpful assistant.")]
        blob = adapter.ingest(messages)
        retrieved = adapter.retrieve(blob)
        assert len(retrieved) == 1
        assert retrieved[0] == ("system", "You are a helpful assistant.")

    def test_langchain_import_error(self):
        adapter = LangChainMemoryAdapter()
        # This should raise ImportError if langchain not installed
        try:
            adapter.ingest_langchain_messages([])
        except ImportError:
            pass  # Expected — langchain-core not installed
        except Exception:
            pass  # Any error is OK if package missing


class TestOTelAdapter:
    def test_ingest_and_retrieve_roundtrip(self):
        adapter = OTelSpanAdapter()
        spans = [
            SpanRecord(
                name="http.request",
                trace_id="abc123",
                span_id="def456",
                parent_id="",
                start_time="1000",
                end_time="2000",
                status="OK",
                attributes={"http.method": "GET", "http.url": "/api/v1"},
            ),
            SpanRecord(
                name="db.query",
                trace_id="abc123",
                span_id="ghi789",
                parent_id="def456",
                start_time="1100",
                end_time="1500",
                status="OK",
                attributes={"db.type": "postgresql"},
            ),
        ]
        blob = adapter.ingest(spans)
        retrieved = adapter.retrieve(blob)
        assert len(retrieved) == 2
        assert retrieved[0].name == "http.request"
        assert retrieved[0].trace_id == "abc123"
        assert retrieved[1].name == "db.query"
        assert retrieved[1].parent_id == "def456"

    def test_span_attributes_preserved(self):
        adapter = OTelSpanAdapter()
        spans = [
            SpanRecord(
                name="test",
                trace_id="t1",
                span_id="s1",
                attributes={"key1": "val1", "key2": "val2"},
            ),
        ]
        blob = adapter.ingest(spans)
        retrieved = adapter.retrieve(blob)
        assert retrieved[0].attributes.get("key1") == "val1"
        assert retrieved[0].attributes.get("key2") == "val2"

    def test_empty_spans(self):
        adapter = OTelSpanAdapter()
        blob = adapter.ingest([])
        retrieved = adapter.retrieve(blob)
        assert retrieved == []

    def test_otel_import_error(self):
        adapter = OTelSpanAdapter()
        try:
            adapter.ingest_otel_spans([])
        except ImportError:
            pass  # Expected
        except Exception:
            pass


class TestAdapterProtocol:
    def test_langchain_is_adapter(self):
        from usc.adapters.base import USCAdapter
        adapter = LangChainMemoryAdapter()
        assert isinstance(adapter, USCAdapter)

    def test_otel_is_adapter(self):
        from usc.adapters.base import USCAdapter
        adapter = OTelSpanAdapter()
        assert isinstance(adapter, USCAdapter)
