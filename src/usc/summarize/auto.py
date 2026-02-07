"""
N8: Auto Summaries — statistical extractive summarization.

No LLM dependency — uses template frequency and keyword density
for scoring and selecting representative lines.
"""
from __future__ import annotations

from collections import Counter
from typing import List, Optional


def auto_summarize(
    texts: List[str],
    ratio: float = 0.1,
    min_lines: int = 1,
    max_lines: int = 50,
) -> str:
    """
    Generate an extractive summary from a list of text chunks.

    Args:
        texts: List of text chunks (e.g., decoded packet contents)
        ratio: Target summary size as fraction of input lines
        min_lines: Minimum number of lines in summary
        max_lines: Maximum number of lines in summary
    """
    # Collect all lines
    all_lines = []
    for text in texts:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                all_lines.append(stripped)

    if not all_lines:
        return ""

    target = max(min_lines, min(max_lines, int(len(all_lines) * ratio)))
    if len(all_lines) <= target:
        return "\n".join(all_lines)

    # Score lines by word frequency (TF-IDF lite)
    word_freq: Counter = Counter()
    line_words: List[List[str]] = []
    for line in all_lines:
        words = line.lower().split()
        line_words.append(words)
        word_freq.update(set(words))

    # Score = sum of word frequencies / line length (normalized)
    scored = []
    for i, (line, words) in enumerate(zip(all_lines, line_words)):
        if not words:
            scored.append((i, 0.0))
            continue
        # Prefer lines with common words (representative) but not too short
        score = sum(word_freq[w] for w in set(words)) / max(len(words), 1)
        scored.append((i, score))

    # Sort by score, pick top, then restore original order
    scored.sort(key=lambda x: x[1], reverse=True)
    top_indices = sorted(i for i, _ in scored[:target])

    return "\n".join(all_lines[i] for i in top_indices)


def summarize_with_keywords(
    texts: List[str],
    keywords: List[str],
    max_lines: int = 20,
) -> str:
    """
    Summarize with keyword bias — prefer lines containing given keywords.
    """
    all_lines = []
    for text in texts:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                all_lines.append(stripped)

    if not all_lines:
        return ""

    kw_set = {k.lower() for k in keywords}

    scored = []
    for i, line in enumerate(all_lines):
        words = set(line.lower().split())
        # Base score from keyword matches
        kw_hits = len(words & kw_set)
        score = kw_hits * 10 + 1  # baseline 1 for all lines
        scored.append((i, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_indices = sorted(i for i, _ in scored[:max_lines])

    return "\n".join(all_lines[i] for i in top_indices)
