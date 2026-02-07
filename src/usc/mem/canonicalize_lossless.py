import re
from typing import List, Tuple


# ISO-ish timestamps: 2026-01-18 05:50:12 or 2026-01-18T05:50:12Z
RE_ISO_TS = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?\b"
)

# UUIDs
RE_UUID = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)

# Long hex blobs / hashes (>= 16 chars)
RE_LONG_HEX = re.compile(r"\b[0-9a-fA-F]{16,}\b")

# Long integers (>= 7 digits) often IDs
RE_LONG_INT = re.compile(r"\b\d{7,}\b")


PLACEHOLDER = "<@>"


def canonicalize_lossless(line: str) -> Tuple[str, List[str]]:
    """
    Lossless canonicalization:
    - Replace high-entropy patterns with PLACEHOLDER markers
    - Return: (canonicalized_text, removed_tokens_in_order)

    This is fully reversible by reinflate_placeholders().
    """
    tokens: List[str] = []
    s = line

    def _sub(rex: re.Pattern, text: str) -> str:
        def _repl(m: re.Match) -> str:
            tokens.append(m.group(0))
            return PLACEHOLDER
        return rex.sub(_repl, text)

    s = _sub(RE_ISO_TS, s)
    s = _sub(RE_UUID, s)
    s = _sub(RE_LONG_HEX, s)
    s = _sub(RE_LONG_INT, s)

    return s, tokens


def reinflate_placeholders(canon_text: str, tokens: List[str]) -> str:
    """
    Replaces PLACEHOLDER occurrences in-order with tokens.
    """
    out = canon_text
    for t in tokens:
        out = out.replace(PLACEHOLDER, t, 1)
    return out
