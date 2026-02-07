"""
N3: PII Detection — regex-based detection of common PII patterns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass
class PIIMatch:
    """A detected PII occurrence."""
    kind: str       # email, phone, ssn, ip, credit_card
    value: str      # the matched text
    start: int      # start position in source text
    end: int        # end position in source text


# Compiled regex patterns for PII types
_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(r"\b(?:\+1[-.]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "ip": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
}


def detect_pii(text: str) -> List[PIIMatch]:
    """Detect PII patterns in text. Returns all matches sorted by position."""
    matches: List[PIIMatch] = []
    for kind, pattern in _PATTERNS.items():
        for m in pattern.finditer(text):
            matches.append(PIIMatch(
                kind=kind,
                value=m.group(),
                start=m.start(),
                end=m.end(),
            ))
    matches.sort(key=lambda m: m.start)
    return matches


def redact_pii(text: str, replacement: str = "[REDACTED]") -> str:
    """Replace all detected PII with a redaction marker."""
    matches = detect_pii(text)
    if not matches:
        return text
    # Process in reverse order to maintain positions
    result = list(text)
    for m in reversed(matches):
        result[m.start:m.end] = list(replacement)
    return "".join(result)
