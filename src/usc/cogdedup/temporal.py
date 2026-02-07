"""Temporal Compression — compress event sequences, not just content.

Upgrade #12: Detect and compress temporal motifs — recurring patterns
in the *order* of events, not just their content. For example:

    [search → read_results → think → tool_call → observe]

This 5-event sequence might repeat 100 times in an agent trace. Instead
of storing 100 copies of similar event sequences, we store the motif once
and reference it.

This is orthogonal to content dedup: two tool_calls with different
arguments have different *content* but the same *temporal motif*.

Usage:
    tracker = TemporalMotifTracker()

    # Feed events as they happen
    tracker.observe("search")
    tracker.observe("read_results")
    tracker.observe("think")
    tracker.observe("tool_call")

    # Check for known motifs
    motifs = tracker.detected_motifs()

    # Compress a full sequence
    encoder = TemporalEncoder(tracker)
    compressed = encoder.encode(events)
    original = encoder.decode(compressed)
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple


@dataclass
class TemporalMotif:
    """A detected recurring pattern in event sequences."""
    motif_id: int
    pattern: Tuple[str, ...]   # The event type sequence
    occurrences: int           # How many times seen
    first_seen: int            # Index of first occurrence
    avg_gap: float             # Average events between occurrences

    @property
    def length(self) -> int:
        return len(self.pattern)

    def __hash__(self) -> int:
        return hash(self.pattern)


@dataclass
class TemporalCompressionResult:
    """Result of temporal compression."""
    tokens: List  # List of (motif_ref | literal_event)
    original_events: int
    compressed_tokens: int
    savings_pct: float
    motifs_used: int


class TemporalMotifTracker:
    """Detect recurring temporal motifs in event sequences.

    Uses sliding window n-gram analysis to find recurring patterns.

    Args:
        min_pattern_len: Minimum events in a motif
        max_pattern_len: Maximum events in a motif
        min_occurrences: Minimum times a pattern must repeat to be a motif
    """

    def __init__(
        self,
        min_pattern_len: int = 3,
        max_pattern_len: int = 10,
        min_occurrences: int = 2,
    ) -> None:
        self._min_len = min_pattern_len
        self._max_len = max_pattern_len
        self._min_occ = min_occurrences

        # Full event history (event type strings)
        self._history: List[str] = []

        # n-gram counts: pattern tuple -> count
        self._ngram_counts: Counter = Counter()

        # Detected motifs
        self._motifs: Dict[Tuple[str, ...], TemporalMotif] = {}
        self._next_motif_id: int = 0

    def observe(self, event_type: str) -> Optional[TemporalMotif]:
        """Record an event and check if it completes a known motif.

        Returns the motif if the last N events match a known pattern.
        """
        self._history.append(event_type)
        idx = len(self._history)

        # Update n-gram counts for all window sizes
        matched_motif = None
        for n in range(self._min_len, min(self._max_len + 1, idx + 1)):
            pattern = tuple(self._history[-n:])
            self._ngram_counts[pattern] += 1

            count = self._ngram_counts[pattern]
            if count >= self._min_occ:
                if pattern not in self._motifs:
                    self._motifs[pattern] = TemporalMotif(
                        motif_id=self._next_motif_id,
                        pattern=pattern,
                        occurrences=count,
                        first_seen=idx - n,
                        avg_gap=0.0,
                    )
                    self._next_motif_id += 1
                else:
                    self._motifs[pattern].occurrences = count

                # Return the longest matching motif
                if matched_motif is None or n > len(matched_motif.pattern):
                    matched_motif = self._motifs[pattern]

        return matched_motif

    def observe_batch(self, events: Sequence[str]) -> None:
        """Observe a batch of events at once."""
        for event in events:
            self.observe(event)

    def detected_motifs(self, min_length: int = 0) -> List[TemporalMotif]:
        """Get all detected motifs, sorted by (occurrences * length) descending."""
        motifs = [
            m for m in self._motifs.values()
            if m.length >= max(min_length, self._min_len)
        ]
        motifs.sort(key=lambda m: m.occurrences * m.length, reverse=True)
        return motifs

    def get_motif_by_pattern(self, pattern: Tuple[str, ...]) -> Optional[TemporalMotif]:
        return self._motifs.get(pattern)

    @property
    def history_length(self) -> int:
        return len(self._history)

    @property
    def motif_count(self) -> int:
        return len(self._motifs)

    def stats(self) -> dict:
        motifs = self.detected_motifs()
        return {
            "events_observed": len(self._history),
            "unique_ngrams": len(self._ngram_counts),
            "motifs_detected": len(motifs),
            "top_motif_occurrences": motifs[0].occurrences if motifs else 0,
            "top_motif_length": motifs[0].length if motifs else 0,
        }


class TemporalEncoder:
    """Compress event sequences using detected temporal motifs.

    Replaces recognized subsequences with motif references.
    """

    def __init__(self, tracker: TemporalMotifTracker) -> None:
        self._tracker = tracker

    def encode(self, events: Sequence[str]) -> TemporalCompressionResult:
        """Compress an event sequence by replacing motifs with references.

        Uses greedy longest-match: at each position, find the longest
        motif that matches and emit a reference instead of individual events.
        """
        motifs = self._tracker.detected_motifs()
        if not motifs:
            tokens = [("literal", e) for e in events]
            return TemporalCompressionResult(
                tokens=tokens,
                original_events=len(events),
                compressed_tokens=len(tokens),
                savings_pct=0.0,
                motifs_used=0,
            )

        # Build lookup: sort motifs by length descending for greedy matching
        motifs_by_len = sorted(motifs, key=lambda m: m.length, reverse=True)

        tokens = []
        i = 0
        motifs_used: Set[int] = set()

        while i < len(events):
            matched = False
            for motif in motifs_by_len:
                end = i + motif.length
                if end <= len(events):
                    window = tuple(events[i:end])
                    if window == motif.pattern:
                        tokens.append(("motif", motif.motif_id))
                        motifs_used.add(motif.motif_id)
                        i = end
                        matched = True
                        break
            if not matched:
                tokens.append(("literal", events[i]))
                i += 1

        savings = ((len(events) - len(tokens)) / max(1, len(events))) * 100.0

        return TemporalCompressionResult(
            tokens=tokens,
            original_events=len(events),
            compressed_tokens=len(tokens),
            savings_pct=round(savings, 1),
            motifs_used=len(motifs_used),
        )

    def decode(self, result: TemporalCompressionResult) -> List[str]:
        """Decode a compressed sequence back to events."""
        events = []
        motif_lookup = {m.motif_id: m for m in self._tracker.detected_motifs()}

        for token_type, value in result.tokens:
            if token_type == "motif":
                motif = motif_lookup.get(value)
                if motif:
                    events.extend(motif.pattern)
                else:
                    raise ValueError(f"Unknown motif_id={value}")
            elif token_type == "literal":
                events.append(value)

        return events
