"""
G7: Utility Gisting — Agent Summaries.

Provides extractive summary generation and a gist+blob wire format
that allows gist extraction without decompressing the inner blob.
"""
from __future__ import annotations

from collections import Counter
from typing import List, Tuple

from usc.mem.varint import encode_uvarint, decode_uvarint

MAGIC = b"UGST"  # 4 bytes


def extract_gist(text: str, max_lines: int = 5) -> str:
    """
    Extractive summary: pick the most representative lines
    based on template frequency (lines that are most "typical").

    Algorithm:
    1. Tokenize each line into words
    2. Score each line by sum of word frequencies across all lines
    3. Pick top-scoring lines, preserving original order
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    if len(lines) <= max_lines:
        return "\n".join(lines)

    # Count word frequencies across all lines
    word_freq: Counter = Counter()
    line_words: List[List[str]] = []
    for ln in lines:
        words = ln.lower().split()
        line_words.append(words)
        word_freq.update(set(words))  # count unique words per line

    # Score each line
    scored = []
    for i, (ln, words) in enumerate(zip(lines, line_words)):
        if not words:
            scored.append((i, 0.0))
            continue
        score = sum(word_freq[w] for w in set(words)) / len(words)
        scored.append((i, score))

    # Sort by score descending, take top max_lines
    scored.sort(key=lambda x: x[1], reverse=True)
    top_indices = sorted(i for i, _ in scored[:max_lines])

    return "\n".join(lines[i] for i in top_indices)


def encode_with_gist(
    inner_blob: bytes,
    gist_text: str,
) -> bytes:
    """
    Wrap an inner blob with a gist header.

    Wire format:
        UGST (4B) + gist_len (uvarint) + gist_bytes + inner_len (uvarint) + inner_blob
    """
    gist_bytes = gist_text.encode("utf-8")
    out = bytearray(MAGIC)
    out += encode_uvarint(len(gist_bytes))
    out += gist_bytes
    out += encode_uvarint(len(inner_blob))
    out += inner_blob
    return bytes(out)


def extract_gist_from_blob(blob: bytes) -> str:
    """
    Extract just the gist text without decompressing the inner blob.
    Zero-cost preview.
    """
    if len(blob) < 4:
        raise ValueError("gist: blob too small")
    if blob[:4] != MAGIC:
        raise ValueError("gist: bad magic")

    off = 4
    gist_len, off = decode_uvarint(blob, off)
    gist_bytes = blob[off:off + gist_len]
    return gist_bytes.decode("utf-8")


def decode_gist_blob(blob: bytes) -> Tuple[str, bytes]:
    """
    Decode a UGST blob into (gist_text, inner_blob).
    """
    if len(blob) < 4:
        raise ValueError("gist: blob too small")
    if blob[:4] != MAGIC:
        raise ValueError("gist: bad magic")

    off = 4
    gist_len, off = decode_uvarint(blob, off)
    gist_text = blob[off:off + gist_len].decode("utf-8")
    off += gist_len

    inner_len, off = decode_uvarint(blob, off)
    inner_blob = blob[off:off + inner_len]
    return gist_text, inner_blob
