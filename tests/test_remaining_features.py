"""Tests for N6-N12: Remaining Must-Haves."""
import pytest


# ===== N6: Diff/Delta Storage =====

class TestDiffStorage:
    def test_delta_encode_decode_roundtrip(self):
        from usc.diff import delta_encode, delta_decode
        old = b"Hello world, this is the original content. " * 10
        new = b"Hello world, this is the modified content! " * 10
        delta = delta_encode(old, new)
        restored = delta_decode(old, delta)
        assert restored == new

    def test_delta_smaller_than_full(self):
        from usc.diff import delta_encode
        old = b"AAAA" * 1000
        new = b"AAAA" * 999 + b"BBBB"
        delta = delta_encode(old, new)
        # Delta should be smaller than encoding new alone
        assert len(delta) < len(new)

    def test_delta_magic(self):
        from usc.diff import delta_encode, MAGIC
        old = b"old data"
        new = b"new data"
        delta = delta_encode(old, new)
        assert delta[:4] == MAGIC

    def test_wrong_base_raises(self):
        from usc.diff import delta_encode, delta_decode
        old = b"correct base"
        wrong = b"wrong base!!"
        new = b"new content"
        delta = delta_encode(old, new)
        with pytest.raises(ValueError, match="mismatch"):
            delta_decode(wrong, delta)

    def test_empty_old(self):
        from usc.diff import delta_encode, delta_decode
        old = b""
        new = b"brand new content"
        delta = delta_encode(old, new)
        restored = delta_decode(old, delta)
        assert restored == new

    def test_identical_blobs(self):
        from usc.diff import delta_encode, delta_decode
        data = b"same content" * 100
        delta = delta_encode(data, data)
        restored = delta_decode(data, delta)
        assert restored == data


# ===== N7: Time-Travel Debug =====

class TestTimeTravel:
    def _packets(self):
        return [f"state-{i}".encode() for i in range(10)]

    def test_navigator_length(self):
        from usc.timetravel import TimelineNavigator
        nav = TimelineNavigator(self._packets())
        assert nav.length == 10

    def test_snapshot_at(self):
        from usc.timetravel import TimelineNavigator
        nav = TimelineNavigator(self._packets())
        snap = nav.snapshot_at(5)
        assert snap.index == 5
        assert snap.packet == b"state-5"
        assert len(snap.cumulative_packets) == 6

    def test_forward_backward(self):
        from usc.timetravel import TimelineNavigator
        nav = TimelineNavigator(self._packets())
        assert nav.position == 0
        nav.forward()
        assert nav.position == 1
        nav.forward()
        assert nav.position == 2
        nav.backward()
        assert nav.position == 1

    def test_forward_at_end_returns_none(self):
        from usc.timetravel import TimelineNavigator
        nav = TimelineNavigator([b"only"])
        assert nav.forward() is None
        assert nav.position == 0

    def test_backward_at_start_returns_none(self):
        from usc.timetravel import TimelineNavigator
        nav = TimelineNavigator([b"only"])
        assert nav.backward() is None

    def test_seek(self):
        from usc.timetravel import TimelineNavigator
        nav = TimelineNavigator(self._packets())
        snap = nav.seek(7)
        assert snap.index == 7
        assert nav.position == 7

    def test_first_last(self):
        from usc.timetravel import TimelineNavigator
        nav = TimelineNavigator(self._packets())
        nav.seek(5)
        nav.first()
        assert nav.position == 0
        nav.last()
        assert nav.position == 9

    def test_diff_between(self):
        from usc.timetravel import TimelineNavigator
        nav = TimelineNavigator(self._packets())
        diff = nav.diff_between(2, 5)
        assert diff == [3, 4, 5]

    def test_out_of_bounds_raises(self):
        from usc.timetravel import TimelineNavigator
        nav = TimelineNavigator(self._packets())
        with pytest.raises(IndexError):
            nav.snapshot_at(99)


# ===== N8: Auto Summaries =====

class TestAutoSummaries:
    def test_summary_shorter_than_input(self):
        from usc.summarize import auto_summarize
        texts = [
            "Line A about errors\nLine B about errors\nLine C about metrics\n"
            "Line D about users\nLine E about errors\nLine F about disk\n"
            "Line G about memory\nLine H about CPU\nLine I about network\n"
            "Line J about database\n"
        ]
        summary = auto_summarize(texts, ratio=0.3)
        assert len(summary.splitlines()) <= 4

    def test_empty_input(self):
        from usc.summarize import auto_summarize
        assert auto_summarize([]) == ""
        assert auto_summarize([""]) == ""

    def test_small_input_returns_all(self):
        from usc.summarize import auto_summarize
        texts = ["Just one line"]
        assert auto_summarize(texts) == "Just one line"

    def test_keyword_summary(self):
        from usc.summarize import summarize_with_keywords
        texts = [
            "Error in module A\nInfo about startup\nError in module B\n"
            "Debug trace for user\nWarning about memory\n"
        ]
        summary = summarize_with_keywords(texts, keywords=["error"], max_lines=2)
        lines = summary.splitlines()
        error_lines = [l for l in lines if "Error" in l]
        assert len(error_lines) >= 1


# ===== N9: Dynamic Tiering =====

class TestDynamicTiering:
    def test_small_text_gets_tier3(self):
        from usc.tiering import DynamicTierPolicy
        policy = DynamicTierPolicy()
        rec = policy.select_tier("short")
        assert rec.tier == 3

    def test_repetitive_text_gets_tier0(self):
        from usc.tiering import DynamicTierPolicy
        policy = DynamicTierPolicy()
        text = ("Reminder: same thing\n" * 100)
        rec = policy.select_tier(text)
        assert rec.tier == 0
        assert rec.repetitiveness > 0.8

    def test_varied_text_gets_tier3(self):
        from usc.tiering import DynamicTierPolicy
        policy = DynamicTierPolicy()
        lines = [f"Unique line number {i} with different content" for i in range(50)]
        text = "\n".join(lines)
        rec = policy.select_tier(text)
        assert rec.tier == 3

    def test_empty_text(self):
        from usc.tiering import DynamicTierPolicy
        policy = DynamicTierPolicy()
        rec = policy.select_tier("")
        assert rec.tier == 0

    def test_recommendation_has_fields(self):
        from usc.tiering import DynamicTierPolicy
        policy = DynamicTierPolicy()
        rec = policy.select_tier("Test data\nTest data\n" * 50)
        assert hasattr(rec, 'tier')
        assert hasattr(rec, 'reason')
        assert hasattr(rec, 'repetitiveness')
        assert hasattr(rec, 'size_bytes')


# ===== N10: Cold Re-indexing =====

class TestColdReindex:
    def test_build_index_and_query(self):
        from usc.reindex import reindex_cold_blob, query_cold_index
        texts = ["User alice logged in from 10.0.0.1", "Error: timeout"]
        index = reindex_cold_blob(texts)
        assert query_cold_index(index, {"alice"})
        assert query_cold_index(index, {"timeout"})
        assert not query_cold_index(index, {"zzzznothere12345"})

    def test_wrap_unwrap_roundtrip(self):
        from usc.reindex import reindex_cold_blob, wrap_with_index, unwrap_index
        texts = ["sample text for indexing"]
        index = reindex_cold_blob(texts)
        original = b"original-blob-data"
        wrapped = wrap_with_index(original, index)
        unwrapped_index, unwrapped_blob = unwrap_index(wrapped)
        assert unwrapped_blob == original
        assert unwrapped_index.n_bits == index.n_bits

    def test_index_has_keywords(self):
        from usc.reindex import reindex_cold_blob
        texts = ["hello world test data" * 10]
        index = reindex_cold_blob(texts)
        assert index.keywords > 0


# ===== N11: Budgeted Decode =====

class TestBudgetedDecode:
    def test_budget_limits_packets(self):
        from usc.budget import budgeted_decode
        from usc.api.codec_odc import build_v3b_packets_from_text, odc_encode_packets
        from usc.bench.datasets import toy_big_agent_log_varied
        text = toy_big_agent_log_varied(loops=10)
        packets = build_v3b_packets_from_text(text)
        blob, _ = odc_encode_packets(packets)
        result = budgeted_decode(blob, max_packets=3)
        assert result.decoded_packets <= 3
        assert result.total_packets == len(packets)
        assert len(result.packets) == result.decoded_packets

    def test_budget_limits_bytes(self):
        from usc.budget import budgeted_decode
        from usc.api.codec_odc import build_v3b_packets_from_text, odc_encode_packets
        from usc.bench.datasets import toy_big_agent_log_varied
        text = toy_big_agent_log_varied(loops=10)
        packets = build_v3b_packets_from_text(text)
        blob, _ = odc_encode_packets(packets)
        result = budgeted_decode(blob, max_bytes=100, max_packets=1000)
        assert result.total_bytes_decoded <= 100 + max(len(p) for p in result.packets)

    def test_full_budget_decodes_all(self):
        from usc.budget import budgeted_decode
        from usc.api.codec_odc import build_v3b_packets_from_text, odc_encode_packets
        from usc.bench.datasets import toy_big_agent_log_varied
        text = toy_big_agent_log_varied(loops=5)
        packets = build_v3b_packets_from_text(text)
        blob, _ = odc_encode_packets(packets)
        result = budgeted_decode(blob, max_bytes=10_000_000, max_packets=10_000)
        assert result.decoded_packets == len(packets)
        assert not result.budget_exhausted


# ===== N12: KV-Cache Codec =====

class TestKVCache:
    def test_encode_decode_roundtrip(self):
        from usc.kvcache import kvcache_encode, kvcache_decode
        kv = {"key1": b"value1", "key2": b"value2", "key3": b"\x00\x01\x02"}
        blob = kvcache_encode(kv)
        decoded = kvcache_decode(blob)
        assert decoded == kv

    def test_magic_header(self):
        from usc.kvcache import kvcache_encode, MAGIC
        blob = kvcache_encode({"k": b"v"})
        assert blob[:4] == MAGIC

    def test_empty_kv(self):
        from usc.kvcache import kvcache_encode, kvcache_decode
        blob = kvcache_encode({})
        decoded = kvcache_decode(blob)
        assert decoded == {}

    def test_large_values(self):
        from usc.kvcache import kvcache_encode, kvcache_decode
        kv = {f"key_{i}": bytes(range(256)) * 4 for i in range(20)}
        blob = kvcache_encode(kv)
        decoded = kvcache_decode(blob)
        assert decoded == kv

    def test_delta_encode_decode(self):
        from usc.kvcache import kvcache_delta_encode, kvcache_delta_decode
        old = {"a": b"1", "b": b"2", "c": b"3"}
        new = {"a": b"1", "b": b"updated", "d": b"4"}  # b changed, c deleted, d added
        delta = kvcache_delta_encode(old, new)
        restored = kvcache_delta_decode(old, delta)
        assert restored == new

    def test_delta_no_changes(self):
        from usc.kvcache import kvcache_delta_encode, kvcache_delta_decode
        kv = {"x": b"y"}
        delta = kvcache_delta_encode(kv, kv)
        restored = kvcache_delta_decode(kv, delta)
        assert restored == kv

    def test_bad_magic_raises(self):
        from usc.kvcache import kvcache_decode
        with pytest.raises(ValueError, match="bad magic"):
            kvcache_decode(b"XXXX" + b"\x00" * 20)
