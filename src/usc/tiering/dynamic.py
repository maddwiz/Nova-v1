"""
N9: Dynamic Tiering — auto-select optimal compression tier.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List


@dataclass
class TierRecommendation:
    """Result of tier analysis."""
    tier: int            # 0 or 3
    reason: str
    repetitiveness: float  # 0.0 (unique) to 1.0 (highly repetitive)
    size_bytes: int


class DynamicTierPolicy:
    """Automatically select optimal compression tier based on data characteristics."""

    def __init__(
        self,
        lossless_threshold: float = 0.8,
        min_size_for_lossy: int = 100,
    ) -> None:
        self.lossless_threshold = lossless_threshold
        self.min_size_for_lossy = min_size_for_lossy

    def select_tier(self, text: str) -> TierRecommendation:
        """
        Analyze text and recommend a tier.

        Tier 0 (lossy): good for highly repetitive, less critical data
        Tier 3 (lossless): for data that must be preserved exactly
        """
        if not text.strip():
            return TierRecommendation(tier=0, reason="empty text", repetitiveness=0.0, size_bytes=0)

        size = len(text.encode("utf-8"))
        rep = self._measure_repetitiveness(text)

        # Small texts → always lossless (overhead of lossy not worth it)
        if size < self.min_size_for_lossy:
            return TierRecommendation(
                tier=3, reason="small text — lossless is safer",
                repetitiveness=rep, size_bytes=size,
            )

        # Highly repetitive → lossy is fine (can reconstruct meaning)
        if rep > self.lossless_threshold:
            return TierRecommendation(
                tier=0, reason="highly repetitive — lossy sufficient",
                repetitiveness=rep, size_bytes=size,
            )

        # Default: lossless for safety
        return TierRecommendation(
            tier=3, reason="moderate complexity — lossless recommended",
            repetitiveness=rep, size_bytes=size,
        )

    def _measure_repetitiveness(self, text: str) -> float:
        """
        Measure how repetitive the text is.
        Returns 0.0 (all unique lines) to 1.0 (all identical lines).
        """
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) <= 1:
            return 0.0
        counts = Counter(lines)
        most_common_count = counts.most_common(1)[0][1]
        return most_common_count / len(lines)
