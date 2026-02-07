"""Tests for G5: Semantic Bloom Filters."""
import math
from usc.bloom import (
    bloom_make,
    bloom_add,
    bloom_check,
    bloom_check_all,
    fnv1a_hash32,
    build_semantic_bloom,
    query_keyword,
    query_keywords_all,
    query_semantic,
    embed_to_buckets,
    estimate_false_positive_rate,
    SemanticBloom,
)


class TestKeywordBloom:
    def test_add_and_check(self):
        bits = bloom_make(1024)
        bloom_add(bits, 1024, 3, "hello")
        assert bloom_check(bits, 1024, 3, "hello")

    def test_missing_token(self):
        bits = bloom_make(1024)
        bloom_add(bits, 1024, 3, "hello")
        # Very unlikely false positive with 1024 bits and 1 token
        assert not bloom_check(bits, 1024, 3, "zzzzzzzzz_unique_missing_token")

    def test_case_insensitive(self):
        bits = bloom_make(1024)
        bloom_add(bits, 1024, 3, "Hello")
        assert bloom_check(bits, 1024, 3, "hello")
        assert bloom_check(bits, 1024, 3, "HELLO")

    def test_check_all(self):
        bits = bloom_make(4096)
        bloom_add(bits, 4096, 3, "alpha")
        bloom_add(bits, 4096, 3, "beta")
        bloom_add(bits, 4096, 3, "gamma")
        assert bloom_check_all(bits, 4096, 3, ["alpha", "beta", "gamma"])

    def test_fnv1a_deterministic(self):
        a = fnv1a_hash32("test", 42)
        b = fnv1a_hash32("test", 42)
        assert a == b

    def test_fnv1a_different_inputs(self):
        a = fnv1a_hash32("test-a", 0)
        b = fnv1a_hash32("test-b", 0)
        assert a != b


class TestSemanticBloom:
    def test_build_keyword_only(self):
        texts = [
            "User alice logged in from 10.0.0.1",
            "Error: connection timeout for server-1",
            "Metric: cpu_usage=95.2 host=web-01",
        ]
        bloom = build_semantic_bloom(texts, n_bits=4096, n_hashes=4)
        assert not bloom.has_semantic
        assert query_keyword(bloom, "alice")
        assert query_keyword(bloom, "timeout")
        assert query_keyword(bloom, "cpu_usage")

    def test_missing_keyword(self):
        texts = ["hello world test"]
        bloom = build_semantic_bloom(texts, n_bits=4096, n_hashes=4)
        assert not query_keyword(bloom, "zzzznothere12345")

    def test_query_keywords_all(self):
        texts = ["alpha beta gamma delta"]
        bloom = build_semantic_bloom(texts, n_bits=4096, n_hashes=4)
        assert query_keywords_all(bloom, ["alpha", "beta"])
        assert not query_keywords_all(bloom, ["alpha", "zzzznothere12345"])

    def test_build_with_embed_fn(self):
        texts = ["error log message", "warning about disk space"]

        def mock_embed(text):
            # Deterministic mock: hash-based vector
            h = hash(text)
            return [(h >> i) % 100 / 100 - 0.5 for i in range(16)]

        bloom = build_semantic_bloom(
            texts, n_bits=4096, n_hashes=4, embed_fn=mock_embed
        )
        assert bloom.has_semantic

    def test_semantic_query_finds_keyword_match(self):
        texts = ["User alice connected to database"]
        bloom = build_semantic_bloom(texts, n_bits=4096, n_hashes=4)
        # Keyword match should work even without embed_fn
        assert query_semantic(bloom, "alice")

    def test_semantic_query_with_embeddings(self):
        def mock_embed(text):
            # Simple mock: "error" and "failure" map to similar vectors
            if "error" in text.lower() or "failure" in text.lower():
                return [0.5, 0.3, -0.2, 0.8, 0.1, -0.5, 0.4, -0.1]
            return [-0.5, -0.3, 0.2, -0.8, -0.1, 0.5, -0.4, 0.1]

        texts = ["system error detected in module A"]
        bloom = build_semantic_bloom(
            texts, n_bits=4096, n_hashes=4,
            embed_fn=mock_embed, n_buckets=16,
        )
        # Semantic query with similar embedding should match
        assert query_semantic(
            bloom, "application failure occurred",
            embed_fn=mock_embed, n_buckets=16,
        )

    def test_semantic_query_no_embed_fn_falls_back_to_keyword(self):
        texts = ["specific token here"]
        bloom = build_semantic_bloom(texts, n_bits=4096, n_hashes=4)
        # Without embed_fn, only keyword matching
        assert query_semantic(bloom, "specific")
        assert not query_semantic(bloom, "zzzznothere12345")


class TestEmbedToBuckets:
    def test_returns_list_of_ints(self):
        vector = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]
        buckets = embed_to_buckets(vector, n_buckets=16)
        assert isinstance(buckets, list)
        assert all(isinstance(b, int) for b in buckets)

    def test_bucket_ids_in_range(self):
        vector = [0.1, -0.2, 0.3, -0.4] * 10
        buckets = embed_to_buckets(vector, n_buckets=32)
        assert all(0 <= b < 32 for b in buckets)

    def test_similar_vectors_share_buckets(self):
        v1 = [0.5, 0.3, -0.2, 0.8, 0.1, -0.5]
        v2 = [0.5, 0.3, -0.2, 0.8, 0.1, -0.5]  # identical
        b1 = set(embed_to_buckets(v1, 16))
        b2 = set(embed_to_buckets(v2, 16))
        assert b1 == b2

    def test_empty_vector(self):
        assert embed_to_buckets([], 16) == []


class TestFalsePositiveRate:
    def test_low_fill_rate(self):
        texts = ["alpha", "beta", "gamma"]
        bloom = build_semantic_bloom(texts, n_bits=8192, n_hashes=4)

        known = ["alpha", "beta", "gamma"]
        test = [f"definitely_not_a_real_word_{i}" for i in range(100)]
        fpr = estimate_false_positive_rate(bloom, test, known)
        # With 8192 bits, 3 tokens, 4 hashes, FPR should be very low
        assert fpr < 0.05

    def test_high_fill_rate(self):
        texts = [f"token_{i}" for i in range(200)]
        bloom = build_semantic_bloom(texts, n_bits=512, n_hashes=4)

        known = [f"token_{i}" for i in range(200)]
        test = [f"other_{i}" for i in range(100)]
        fpr = estimate_false_positive_rate(bloom, test, known)
        # Small bloom with many tokens → higher FPR
        assert fpr > 0.0  # some false positives expected

    def test_empty_bloom(self):
        bloom = build_semantic_bloom([], n_bits=1024, n_hashes=4)
        test = ["anything"]
        fpr = estimate_false_positive_rate(bloom, test, [])
        assert fpr == 0.0
