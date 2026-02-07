"""
G5: Semantic Bloom Filters.

Extends keyword bloom with embedding-bucket hashing via LSH.
Falls back to keyword-only if numpy is not available.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from usc.bloom.keyword import bloom_make, bloom_add, bloom_check, bloom_check_all, fnv1a_hash32

_WORD_RE = re.compile(r"[A-Za-z0-9_./:-]{2,}")

# Type alias for embedding function
EmbedFn = Callable[[str], List[float]]


def _tokenize(text: str) -> List[str]:
    """Extract keyword tokens from text."""
    return _WORD_RE.findall(text.lower())


def embed_to_buckets(vector: List[float], n_buckets: int = 64) -> List[int]:
    """
    LSH-style bucketing: project vector dimensions into bucket IDs.

    Uses simple sign-based hashing — each dimension's sign contributes
    to a bucket index. This gives O(1) hashing with decent locality.
    """
    if not vector:
        return []
    buckets: List[int] = []
    # Group dimensions into chunks, hash each chunk's sign pattern
    chunk_size = max(1, len(vector) // n_buckets)
    for i in range(0, len(vector), chunk_size):
        chunk = vector[i:i + chunk_size]
        # Sign hash: each positive dim = 1, negative = 0, then hash the pattern
        pattern = 0
        for j, v in enumerate(chunk):
            if v >= 0:
                pattern |= (1 << (j % 32))
        bucket_id = pattern % n_buckets
        buckets.append(bucket_id)
    return list(set(buckets))  # deduplicate


@dataclass
class SemanticBloom:
    """Bloom filter that supports both keyword and semantic queries."""
    n_bits: int
    n_hashes: int
    bits: bytearray = field(default_factory=lambda: bytearray())
    has_semantic: bool = False

    def __post_init__(self):
        if not self.bits:
            self.bits = bloom_make(self.n_bits)


def build_semantic_bloom(
    texts: List[str],
    n_bits: int = 8192,
    n_hashes: int = 4,
    embed_fn: Optional[EmbedFn] = None,
    n_buckets: int = 64,
) -> SemanticBloom:
    """
    Build a bloom filter from texts with optional semantic embedding.

    If embed_fn is provided, embedding bucket IDs are also added to the bloom,
    enabling approximate semantic queries.
    """
    bloom = SemanticBloom(n_bits=n_bits, n_hashes=n_hashes)

    for text in texts:
        # Always add keyword tokens
        tokens = _tokenize(text)
        for tok in tokens:
            bloom_add(bloom.bits, n_bits, n_hashes, tok)

        # Optionally add semantic buckets
        if embed_fn is not None:
            try:
                vector = embed_fn(text)
                buckets = embed_to_buckets(vector, n_buckets)
                for b in buckets:
                    bloom_add(bloom.bits, n_bits, n_hashes, f"__sem_bucket_{b}")
                bloom.has_semantic = True
            except Exception:
                pass  # graceful fallback to keyword-only

    return bloom


def query_keyword(bloom: SemanticBloom, keyword: str) -> bool:
    """Check if a keyword might be in the bloom."""
    return bloom_check(bloom.bits, bloom.n_bits, bloom.n_hashes, keyword.lower())


def query_keywords_all(bloom: SemanticBloom, keywords: List[str]) -> bool:
    """Check if ALL keywords might be in the bloom."""
    return bloom_check_all(bloom.bits, bloom.n_bits, bloom.n_hashes, keywords)


def query_semantic(
    bloom: SemanticBloom,
    query_text: str,
    embed_fn: Optional[EmbedFn] = None,
    n_buckets: int = 64,
) -> bool:
    """
    Semantic query: check if the bloom might contain content similar to query_text.

    Falls back to keyword query if embed_fn is None or bloom has no semantic data.
    """
    # Always check keyword tokens first
    tokens = _tokenize(query_text)
    keyword_hit = any(
        bloom_check(bloom.bits, bloom.n_bits, bloom.n_hashes, tok)
        for tok in tokens
    ) if tokens else False

    if keyword_hit:
        return True

    # Try semantic bucket matching
    if embed_fn is not None and bloom.has_semantic:
        try:
            vector = embed_fn(query_text)
            buckets = embed_to_buckets(vector, n_buckets)
            for b in buckets:
                if bloom_check(bloom.bits, bloom.n_bits, bloom.n_hashes, f"__sem_bucket_{b}"):
                    return True
        except Exception:
            pass

    return False


def estimate_false_positive_rate(
    bloom: SemanticBloom,
    test_tokens: List[str],
    known_tokens: List[str],
) -> float:
    """
    Estimate false positive rate by testing tokens known NOT to be in the bloom.
    """
    known_set = set(k.lower() for k in known_tokens)
    false_tokens = [t for t in test_tokens if t.lower() not in known_set]
    if not false_tokens:
        return 0.0
    false_positives = sum(
        1 for t in false_tokens
        if bloom_check(bloom.bits, bloom.n_bits, bloom.n_hashes, t)
    )
    return false_positives / len(false_tokens)
