"""Hybrid retrieval: weighted merge of keyword + vector search."""

from __future__ import annotations

import numpy as np

from c3ae.config import RetrievalConfig
from c3ae.retrieval.keyword import KeywordSearch
from c3ae.retrieval.vector import VectorSearch
from c3ae.types import SearchResult


class HybridSearch:
    """Combines FTS5 keyword and FAISS vector search with weighted scoring."""

    def __init__(
        self,
        keyword: KeywordSearch,
        vector: VectorSearch,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.keyword = keyword
        self.vector = vector
        self.config = config or RetrievalConfig()

    def search(
        self,
        query: str,
        query_vector: np.ndarray | None = None,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Run hybrid search.

        If query_vector is None, falls back to keyword-only search.
        """
        top_k = top_k or self.config.default_top_k
        # Fetch more candidates for merging
        fetch_k = top_k * 3

        # Keyword results
        kw_results = self.keyword.search_all(query, limit=fetch_k)

        # Vector results (if we have an embedding)
        vec_results: list[SearchResult] = []
        if query_vector is not None:
            vec_results = self.vector.search(query_vector, top_k=fetch_k)

        if not vec_results:
            return kw_results[:top_k]
        if not kw_results:
            return vec_results[:top_k]

        return self._merge(kw_results, vec_results, top_k)

    def _merge(
        self,
        kw_results: list[SearchResult],
        vec_results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Reciprocal rank fusion with configurable weights."""
        k = 60  # RRF constant

        scores: dict[str, float] = {}
        best_result: dict[str, SearchResult] = {}

        # Keyword contributions
        for rank, r in enumerate(kw_results):
            rrf = self.config.keyword_weight / (k + rank + 1)
            scores[r.id] = scores.get(r.id, 0.0) + rrf
            if r.id not in best_result:
                best_result[r.id] = r

        # Vector contributions
        for rank, r in enumerate(vec_results):
            rrf = self.config.vector_weight / (k + rank + 1)
            scores[r.id] = scores.get(r.id, 0.0) + rrf
            if r.id not in best_result:
                best_result[r.id] = r

        # Sort by combined score
        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        results = []
        for rid in sorted_ids[:top_k]:
            r = best_result[rid]
            results.append(SearchResult(
                id=r.id,
                content=r.content,
                score=scores[rid],
                source="hybrid",
                metadata=r.metadata,
            ))
        return results
